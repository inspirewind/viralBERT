import argparse
import logging
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
from pyfaidx import Fasta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Inference-Pretrain")

# Add project root to system path to ensure viralbert package can be found
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from viralbert.config.hf_pretrain_config import ViralBERTConfig
from viralbert.models.model import ViralBERTModel
from viralbert.data.hf_tokenizer import ViralBERTTokenizer

# Register custom classes with Hugging Face Auto classes
try:
    AutoConfig.register("viralbert", ViralBERTConfig)
    AutoTokenizer.register(ViralBERTConfig, ViralBERTTokenizer)
    AutoModel.register(ViralBERTConfig, ViralBERTModel)
    logger.info("Successfully registered custom ViralBERT classes with HuggingFace AutoClasses")
except ValueError:
    logger.warning("Custom ViralBERT classes might be already registered.")
except Exception as e:
    logger.warning(f"Failed to register custom classes: {e}")

class InferenceDataset(Dataset):
    """
    A high-performance dataset for FASTA file inference.
    
    Duplicates logic from inference_classification.py to maintain independence.
    This dataset pre-computes an index of all sequence chunks and uses `pyfaidx`
    for fast, on-demand data loading.
    """
    def __init__(self, fasta_path: str, tokenizer, max_length: int):
        """
        Args:
            fasta_path: Path to the input FASTA file.
            tokenizer: Hugging Face tokenizer instance.
            max_length: Maximum sequence length for each chunk.
        """
        self.fasta = Fasta(fasta_path, sequence_always_upper=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = max_length - 2  # For [CLS] and [SEP] tokens
        
        logger.info("Creating chunk index from FASTA file...")
        self.chunk_index = self._create_chunk_index()
        logger.info(f"Created {len(self.chunk_index)} chunks to process.")

    def _create_chunk_index(self):
        """Scans the FASTA file and creates an index of all chunks."""
        index = []
        total_sequences = 0
        for seq_name in self.fasta.keys():
            total_sequences += 1
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
                # Drop the tiny tail chunk (< half chunk) to avoid noise
                if (end > sequence_len) and (sequence_len - start < self.chunk_size // 2):
                    break
                index.append({'seq_id': seq_name, 'start': start, 'end': min(end, sequence_len)})
        
        logger.info(f"Total sequences in FASTA: {total_sequences}")
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

def get_embeddings(model, input_ids, attention_mask, pooling='mean'):
    """
    Compute embeddings for a batch of sequences.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    # Handle dict output vs object output for custom models
    if hasattr(outputs, 'last_hidden_state'):
        last_hidden_state = outputs.last_hidden_state
    elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
        last_hidden_state = outputs['last_hidden_state']
    else:
        # Fallback debugging
        avail = outputs.keys() if isinstance(outputs, dict) else dir(outputs)
        raise AttributeError(f"Could not find 'last_hidden_state' in model outputs. Available keys/attrs: {avail}")
    
    if pooling == 'cls':
        # Use the [CLS] token embedding (first token)
        return last_hidden_state[:, 0, :]
    elif pooling == 'mean':
        # Mean pooling with attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")

def run_inference(args):
    """Main function to run the inference pipeline."""
    # device setup
    use_cuda = torch.cuda.is_available() and 'cpu' not in args.device
    if use_cuda:
        device = torch.device(args.device.split(',')[0]) # Use first GPU if multiple specified for simplicity in base script
        logger.info(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        logger.info("Using device: CPU")

    # 1) Load Model and Tokenizer
    logger.info(f"Loading model from: {args.model_path}")
    # We use AutoModel to get the base transformer (no classification head)
    model = AutoModel.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    model.to(device)
    if use_cuda and ',' in args.device:
         device_ids = [int(x.split(':')[-1]) for x in args.device.split(',')]
         if len(device_ids) > 1:
            logger.info(f"Using DataParallel on devices: {device_ids}")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            
    model.eval()

    # 2) Dataset + DataLoader
    dataset = InferenceDataset(
        fasta_path=args.input_fasta,
        tokenizer=tokenizer,
        max_length=args.inference_max_len
    )

    if len(dataset) == 0:
        logger.warning("No sequences found to process.")
        return

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 3) Inference
    # Buffer to store chunk embeddings: seq_id -> list of [hidden_dim] vectors
    chunk_embeddings_buffer = defaultdict(list)
    seq_metadata = {} # seq_id -> {'length': int}
    
    logger.info(f"Starting inference with pooling strategy: {args.pooling}")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get batch embeddings [batch_size, hidden_dim]
            batch_embeddings = get_embeddings(model, input_ids, attention_mask, pooling=args.pooling)
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            for i, seq_id in enumerate(batch['seq_ids']):
                chunk_embeddings_buffer[seq_id].append(batch_embeddings[i])
                if seq_id not in seq_metadata:
                    seq_metadata[seq_id] = batch['sequence_lengths'][i]

    # 4) Aggregate or Flat
    if args.combine:
        logger.info("Aggregating chunk embeddings per sequence...")
        final_ids = []
        final_embeddings = []
        final_lengths = []
        num_chunks_list = []

        # Sort keys to ensure deterministic order
        sorted_seq_ids = sorted(chunk_embeddings_buffer.keys())
        
        for seq_id in tqdm(sorted_seq_ids, desc="Aggregating"):
            chunks = np.array(chunk_embeddings_buffer[seq_id]) # [num_chunks, hidden_dim]
            
            # Aggregate chunks: Mean of chunks is standard for representing the whole sequence
            seq_embedding = np.mean(chunks, axis=0) # [hidden_dim]
            
            final_ids.append(seq_id)
            final_embeddings.append(seq_embedding)
            final_lengths.append(seq_metadata[seq_id])
            num_chunks_list.append(len(chunks))

        final_embeddings = np.array(final_embeddings) # [num_sequences, hidden_dim]
        
        # Create DataFrame for sequence-level
        metadata_df = pd.DataFrame({
            'sequence_id': final_ids,
            'sequence_length': final_lengths,
            'num_chunks': num_chunks_list
        })

    else:
        logger.info("Skipping aggregation. Outputting chunk-level embeddings...")
        final_ids = []
        final_embeddings = []
        final_lengths = [] # original sequence length
        chunk_indices = []
        
        sorted_seq_ids = sorted(chunk_embeddings_buffer.keys())
        
        for seq_id in tqdm(sorted_seq_ids, desc="Processing Chunks"):
            chunks = np.array(chunk_embeddings_buffer[seq_id]) # [num_chunks, hidden_dim]
            
            for i in range(len(chunks)):
                final_ids.append(seq_id)
                final_embeddings.append(chunks[i])
                final_lengths.append(seq_metadata[seq_id])
                chunk_indices.append(i)
                
        final_embeddings = np.array(final_embeddings) # [total_chunks, hidden_dim]
        
        # Create DataFrame for chunk-level
        metadata_df = pd.DataFrame({
            'sequence_id': final_ids,
            'chunk_index': chunk_indices,
            'sequence_length': final_lengths
        })

    # 5) Save Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add label column if provided
    if args.label:
        metadata_df['label'] = args.label
        
    csv_path = output_dir / "metadata.csv"
    metadata_df.to_csv(csv_path, index=False)
    
    # Save Embeddings NPY
    npy_path = output_dir / "embeddings.npy"
    np.save(npy_path, final_embeddings)
    
    logger.info(f"\n--- Inference Complete ---")
    logger.info(f"Processed {len(final_ids)} sequences.")
    logger.info(f"Embedding shape: {final_embeddings.shape}")
    logger.info(f"Metadata saved to: {csv_path}")
    logger.info(f"Embeddings saved to: {npy_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a pretrained ViralBERT model for Zero-shot analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path', required=True, 
        help='Path to the pretrained model directory (HF format).'
    )
    parser.add_argument('--input_fasta', required=True, help='Input FASTA file.')
    parser.add_argument('--output_dir', required=True, help="Directory to save output files (.npy and .csv).")
    parser.add_argument(
        '--pooling', type=str, default='mean', choices=['mean', 'cls'],
        help="Pooling strategy to derive chunk embeddings."
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, 
        help="Batch size for inference."
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help="Number of CPU workers for data loading."
    )
    parser.add_argument(
        '--device', type=str, default="cuda:0", 
        help="Device to use (e.g., 'cuda:0', 'cpu')."
    )
    parser.add_argument(
        "--inference_max_len", type=int, default=512, 
        help="Max length for a single model input chunk."
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional label to assign to all sequences in the output metadata (e.g., 'virus', 'bacteria')."
    )
    parser.add_argument(
        "--combine", action='store_true', default=True,
        help="If True (default), aggregate chunks per sequence (mean). If False, output embeddings for every chunk."
    )
    parser.add_argument(
        "--no-combine", dest='combine', action='store_false',
        help="Do not aggregate chunks. Output embeddings for every chunk."
    )

    args = parser.parse_args()
    
    run_inference(args)

if __name__ == "__main__":
    main()

