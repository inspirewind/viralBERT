import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from pyfaidx import Fasta

# Add project root to system path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from viralbert.tasks.classification.model import ViralBERTForSequenceClassification
from viralbert.data.hf_tokenizer import ViralBERTTokenizer
from viralbert.config.hf_classification_config import ViralBERTClassificationConfig
from viralbert.utils.logging import get_dist_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = get_dist_logger("Inference-Classification")

# Register custom classes with Hugging Face Auto classes
try:
    AutoConfig.register("viralbert_for_sequence_classification", ViralBERTClassificationConfig)
    AutoTokenizer.register(ViralBERTClassificationConfig, ViralBERTTokenizer)
    AutoModelForSequenceClassification.register(ViralBERTClassificationConfig, ViralBERTForSequenceClassification)
    logger.info("Successfully registered custom ViralBERT classes with HuggingFace AutoClasses")
except ValueError:
    logger.warning("Custom ViralBERT classes might be already registered.")
except Exception as e:
    logger.error(f"Failed to register custom classes: {e}", exc_info=True)


class InferenceDataset(Dataset):
    """
    A high-performance dataset for FASTA file inference.

    This dataset pre-computes an index of all sequence chunks and uses `pyfaidx`
    for fast, on-demand data loading. This approach is memory-efficient and ideal
    for large FASTA files.
    """
    def __init__(self, fasta_path: str, tokenizer, max_length: int, completed_ids: set = None):
        """
        Args:
            fasta_path: Path to the input FASTA file.
            tokenizer: Hugging Face tokenizer instance.
            max_length: Maximum sequence length for each chunk.
            completed_ids: A set of sequence IDs that have already been processed.
        """
        self.fasta = Fasta(fasta_path, sequence_always_upper=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = max_length - 2  # For [CLS] and [SEP] tokens
        self.completed_ids = completed_ids if completed_ids is not None else set()
        
        logger.info("Creating chunk index from FASTA file...")
        self.chunk_index = self._create_chunk_index()
        logger.info(f"Created {len(self.chunk_index)} chunks to process.")

    def _create_chunk_index(self):
        """Scans the FASTA file and creates an index of all chunks."""
        index = []
        total_sequences = 0
        for seq_name in self.fasta.keys():
            total_sequences += 1
            if seq_name in self.completed_ids:
                continue
            
            sequence_len = len(self.fasta[seq_name])
            if sequence_len == 0:
                logger.warning(f"Sequence {seq_name} is empty, skipping.")
                continue

            # Short sequences (<= chunk_size)
            if sequence_len <= self.chunk_size:
                index.append({'seq_id': seq_name, 'start': 0, 'end': sequence_len})
                continue
            
            # Long sequences -> chunked
            for start in range(0, sequence_len, self.chunk_size):
                end = start + self.chunk_size
                # Drop the tiny tail chunk (< half chunk)
                if (end > sequence_len) and (sequence_len - start < self.chunk_size // 2):
                    break
                index.append({'seq_id': seq_name, 'start': start, 'end': min(end, sequence_len)})
        
        logger.info(f"Total sequences in FASTA: {total_sequences}")
        logger.info(f"Already completed: {len(self.completed_ids)}")
        logger.info(f"Remaining sequences to process: {len(set(d['seq_id'] for d in index))}")
        return index

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx):
        chunk_info = self.chunk_index[idx]
        seq_id = chunk_info['seq_id']
        start = chunk_info['start']
        end = chunk_info['end']

        sequence = str(self.fasta[seq_id][start:end])

        inputs = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'seq_id': seq_id,
            'sequence_length': len(self.fasta[seq_id])
        }


def collate_fn(batch):
    """Custom collate function to handle batching of tensor and non-tensor data."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    seq_ids = [item['seq_id'] for item in batch]
    sequence_lengths = [item['sequence_length'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'seq_ids': seq_ids,
        'sequence_lengths': sequence_lengths
    }


def run_inference(args):
    """Main function to run the inference pipeline."""
    use_cuda = torch.cuda.is_available() and 'cpu' not in args.device

    if use_cuda:
        devices_str = [d.strip() for d in args.device.split(',')]
        try:
            device_ids = [int(d.split(':')[1]) for d in devices_str]
        except (ValueError, IndexError):
            logger.error(f"Invalid device format: '{args.device}'. Use 'cpu' or comma-separated 'cuda:x' (e.g., 'cuda:0,cuda:1').")
            sys.exit(1)

        main_device = f'cuda:{device_ids[0]}'
        logger.info(f"Main device: {main_device}")
        if len(device_ids) > 1:
            logger.info(f"Using DataParallel across devices: {[f'cuda:{i}' for i in device_ids]}")
    else:
        main_device = 'cpu'
        device_ids = None
        logger.info("Using device: CPU")

    # 1) Load Model and Tokenizer
    logger.info(f"Loading model from directory: {args.model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    
    model.to(main_device)
    if use_cuda and ',' in args.device:
        try:
            devices_str = [d.strip() for d in args.device.split(',')]
            device_ids = [int(d.split(':')[1]) for d in devices_str]
        except (ValueError, IndexError):
            device_ids = None
        if device_ids and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    # 2) Resume (optional)
    completed_ids = set()
    existing_results_df = None
    if args.resume and Path(args.output_path).exists():
        logger.info(f"Resuming from existing output file: {args.output_path}")
        try:
            existing_results_df = pd.read_csv(args.output_path)
            if 'sequence_id' in existing_results_df.columns:
                completed_ids = set(existing_results_df['sequence_id'].astype(str))
                logger.info(f"Found {len(completed_ids)} completed sequences.")
        except Exception as e:
            logger.warning(f"Could not read existing output file for resume: {e}")

    # 3) Dataset + DataLoader
    dataset = InferenceDataset(
        fasta_path=args.input_fasta,
        tokenizer=tokenizer,
        max_length=args.inference_max_len,
        completed_ids=completed_ids
    )

    if len(dataset) == 0:
        logger.info("No new sequences to process. Exiting.")
        if existing_results_df is not None:
            print_summary(existing_results_df)
        return

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 4) Inference
    results_buffer = defaultdict(list)
    seq_lengths = {}
    
    logger.info("Starting inference...")
    for batch in tqdm(data_loader, desc="Inference"):
        input_ids = batch['input_ids'].to(main_device)
        attention_mask = batch['attention_mask'].to(main_device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        for i, seq_id in enumerate(batch['seq_ids']):
            results_buffer[seq_id].append(probabilities[i])
            if seq_id not in seq_lengths:
                seq_lengths[seq_id] = batch['sequence_lengths'][i]

    # 5) Aggregate per-sequence
    logger.info("Aggregating results...")
    final_results = []
    for seq_id, chunk_probs in results_buffer.items():
        chunk_probs = np.array(chunk_probs)
        num_chunks = len(chunk_probs)

        # Voting
        chunk_preds = np.argmax(chunk_probs, axis=1)
        vote_prediction = np.bincount(chunk_preds).argmax()
        vote_ratio = np.sum(chunk_preds == vote_prediction) / len(chunk_preds)
        
        # Probability averaging
        avg_probs = np.mean(chunk_probs, axis=0)
        prob_prediction = np.argmax(avg_probs)
        prob_confidence = float(avg_probs.max())

        result_row = {
            'sequence_id': seq_id,
            'sequence_length': seq_lengths[seq_id],
            'predicted_label_vote': id2label[vote_prediction],
            'predicted_label_prob': id2label[prob_prediction],
            'vote_confidence': vote_ratio,
            'prob_confidence': prob_confidence,
            'num_chunks': num_chunks
        }
        for label_idx, label in id2label.items():
            result_row[f'prob_{label}'] = float(avg_probs[label_idx])
        
        final_results.append(result_row)
        
    new_results_df = pd.DataFrame(final_results)

    # 6) Save output
    if existing_results_df is not None:
        final_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
    else:
        final_df = new_results_df

    final_df.to_csv(args.output_path, index=False)
    logger.info(f"Inference complete. Results saved to {args.output_path}")

    # 7) Summary
    print_summary(final_df)


def print_summary(df):
    """Prints a summary of the prediction results."""
    logger.info("\n--- Final Processing Statistics ---")
    logger.info(f"Total sequences processed: {len(df)}")
    logger.info(f"Average sequence length: {df['sequence_length'].mean():.2f}")
    logger.info(f"Average chunks per sequence: {df['num_chunks'].mean():.2f}")
    
    logger.info("\n--- Prediction Summary (Voting Strategy) ---")
    logger.info(f"\n{df['predicted_label_vote'].value_counts().to_string()}")
    
    logger.info("\n--- Prediction Summary (Probability Strategy) ---")
    logger.info(f"\n{df['predicted_label_prob'].value_counts().to_string()}")
    
    agreement = (df['predicted_label_vote'] == df['predicted_label_prob']).mean() * 100
    logger.info(f"\n--- Strategy Agreement: {agreement:.2f}% ---")


def main():
    parser = argparse.ArgumentParser(
        description="Run sequence classification inference using a fine-tuned ViralBERT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path', required=True, 
        help='Path to the model checkpoint file (e.g., pytorch_model.bin) or its parent directory.'
    )
    parser.add_argument('--input_fasta', required=True, help='Input FASTA file.')
    parser.add_argument('--output_path', required=True, help="Path to save the output CSV file.")
    parser.add_argument(
        '--batch_size', type=int, default=128, 
        help="Number of sequence chunks to process in a single batch on the GPU."
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help="Number of CPU workers for parallel data loading."
    )
    parser.add_argument(
        '--device', type=str, default="cuda:0", 
        help="Device(s) to use for inference (e.g., 'cuda:0', 'cpu', 'cuda:0,cuda:1')."
    )
    parser.add_argument(
        "--inference_max_len", type=int, default=512, 
        help="Max length for a single model input chunk."
    )
    parser.add_argument(
        '-c', '--continue', dest='resume', action='store_true', 
        help="Resume from a previous run by reading the output file."
    )
    args = parser.parse_args()

    # Ensure model_dir is the directory
    model_path = Path(args.model_path)
    args.model_dir = str(model_path.parent if model_path.is_file() and model_path.name != "config.json" else model_path)
    
    # Ensure output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    run_inference(args)

if __name__ == "__main__":
    main()

