import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

import hdbscan
import igraph as ig
import leidenalg
from transformers import AutoConfig, AutoTokenizer

# Add project root to system path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from viralbert.config.hf_binning_config import ViralBERTBinningConfig
from viralbert.data.hf_tokenizer import ViralBERTTokenizer
from viralbert.tasks.binning.model import ViralBERTForContrastiveBinning

logger = logging.getLogger("inference_binning")


def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def register_hf_classes() -> None:
    # Register custom classes with Hugging Face Auto classes
    try:
        AutoConfig.register("viralbert_for_contrastive_binning", ViralBERTBinningConfig)
        AutoTokenizer.register(ViralBERTBinningConfig, ViralBERTTokenizer)
        logger.info("Successfully registered custom ViralBERT classes with HuggingFace AutoClasses")
    except Exception as e:
        logger.warning(f"Failed to register custom classes: {e}. This might be expected if already registered.")


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def chunk_sequence(sequence: str, chunk_len: int, stride: int) -> List[str]:
    if not sequence:
        return []
    if len(sequence) <= chunk_len:
        return [sequence]

    chunks: List[str] = []
    for start in range(0, len(sequence) - chunk_len + 1, stride):
        chunks.append(sequence[start : start + chunk_len])

    # Ensure the end of the sequence is included.
    if (len(sequence) - chunk_len) % stride != 0:
        chunks.append(sequence[-chunk_len:])

    return chunks


@dataclass
class InferenceConfig:
    model_path: str
    input_fasta: str
    output_dir: str
    output_prefix: str
    device: str
    embed_batch_size: int
    max_length: int
    stride: int
    normalize_embeddings: bool
    save_embeddings: bool
    embeddings_path: Optional[str]

    # Leiden
    leiden_k: int
    leiden_seed: int

    # HDBSCAN
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    hdbscan_cluster_selection_epsilon: float

    # KMeans
    kmeans_n_clusters: str
    kmeans_seed: int


class BinningInferencer:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        logger.info(f"Loading model and tokenizer from: {cfg.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        self.model_config = ViralBERTBinningConfig.from_pretrained(cfg.model_path)

        self.model = ViralBERTForContrastiveBinning.from_pretrained(cfg.model_path, config=self.model_config)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(
            "Model loaded. "
            f"hidden_size={self.model_config.hidden_size}, moco_dim={self.model_config.moco_dim}, "
            f"max_length={cfg.max_length}, stride={cfg.stride}, normalize={cfg.normalize_embeddings}"
        )

    @torch.no_grad()
    def embed_one(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Returns per-contig embeddings:
        - cls: backbone CLS (dim=hidden_size)
        - projected: projector_q(backbone CLS) (dim=moco_dim)
        - mean_pool: mean pooling over content tokens (excluding padding + special tokens)
        """
        # Use max_length for token length; keep space for special tokens.
        # This mirrors evaluate's "max_length + stride" behavior but avoids truncating bases by default.
        chunk_len = max(1, self.cfg.max_length - 2)
        chunks = chunk_sequence(sequence, chunk_len=chunk_len, stride=self.cfg.stride)
        if not chunks:
            return {
                "cls": np.zeros(self.model_config.hidden_size, dtype=np.float32),
                "projected": np.zeros(self.model_config.moco_dim, dtype=np.float32),
                "mean_pool": np.zeros(self.model_config.hidden_size, dtype=np.float32),
            }

        enc = self.tokenizer(
            chunks,
                add_special_tokens=True,
            max_length=self.cfg.max_length,
                padding="max_length",
                truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        cls_chunks: List[np.ndarray] = []
        proj_chunks: List[np.ndarray] = []
        mean_pool_chunks: List[np.ndarray] = []

        for i in range(0, input_ids.size(0), self.cfg.embed_batch_size):
            batch_input_ids = input_ids[i : i + self.cfg.embed_batch_size].to(self.device)
            batch_attention_mask = attention_mask[i : i + self.cfg.embed_batch_size].to(self.device)

            outputs = self.model.bert(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            last_hidden_state = outputs["last_hidden_state"]  # [B, L, H]
            cls = last_hidden_state[:, 0]  # [B, H]
            proj = self.model.projector_q(cls)
            
            # Mean pooling: exclude [CLS] (position 0) and [SEP] (last non-padding position)
            # and padding tokens using attention_mask
            # Create mask that excludes special tokens (first and last non-padding positions)
            # For simplicity, we use attention_mask and exclude position 0 ([CLS])
            content_mask = batch_attention_mask.clone()
            content_mask[:, 0] = 0  # Exclude [CLS]
            # Find and exclude [SEP] (last 1 in each row of attention_mask)
            for b_idx in range(content_mask.size(0)):
                seq_len = batch_attention_mask[b_idx].sum().item()
                if seq_len > 1:
                    content_mask[b_idx, int(seq_len) - 1] = 0  # Exclude [SEP]
            
            # Mean pool over content tokens
            content_mask_f = content_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)  # [B, L, 1]
            summed = (last_hidden_state * content_mask_f).sum(dim=1)  # [B, H]
            counts = content_mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
            mean_pool = summed / counts  # [B, H]

            if self.cfg.normalize_embeddings:
                cls = F.normalize(cls, dim=1)
                proj = F.normalize(proj, dim=1)
                mean_pool = F.normalize(mean_pool, dim=1)

            cls_chunks.append(cls.cpu().numpy())
            proj_chunks.append(proj.cpu().numpy())
            mean_pool_chunks.append(mean_pool.cpu().numpy())

        cls_emb = np.mean(np.vstack(cls_chunks), axis=0)
        proj_emb = np.mean(np.vstack(proj_chunks), axis=0)
        mean_pool_emb = np.mean(np.vstack(mean_pool_chunks), axis=0)

        if self.cfg.normalize_embeddings:
            cls_emb = l2_normalize_np(cls_emb.reshape(1, -1)).reshape(-1)
            proj_emb = l2_normalize_np(proj_emb.reshape(1, -1)).reshape(-1)
            mean_pool_emb = l2_normalize_np(mean_pool_emb.reshape(1, -1)).reshape(-1)

        return {"cls": cls_emb, "projected": proj_emb, "mean_pool": mean_pool_emb}

    def embed_fasta(self) -> Tuple[List[str], List[int], Dict[str, np.ndarray]]:
        contig_ids: List[str] = []
        contig_lengths: List[int] = []
        cls_embeddings: List[np.ndarray] = []
        proj_embeddings: List[np.ndarray] = []
        mean_pool_embeddings: List[np.ndarray] = []

        records = list(SeqIO.parse(self.cfg.input_fasta, "fasta"))
        logger.info(f"Loaded {len(records)} contigs from {self.cfg.input_fasta}")

        for rec in tqdm(records, desc="Embedding contigs"):
            contig_ids.append(rec.id)
            contig_lengths.append(len(rec.seq))
            embs = self.embed_one(str(rec.seq))
            cls_embeddings.append(embs["cls"])
            proj_embeddings.append(embs["projected"])
            mean_pool_embeddings.append(embs["mean_pool"])

        return contig_ids, contig_lengths, {
            "cls": np.vstack(cls_embeddings),
            "projected": np.vstack(proj_embeddings),
            "mean_pool": np.vstack(mean_pool_embeddings),
        }

    def cluster_leiden(self, embeddings: np.ndarray) -> np.ndarray:
        k = min(self.cfg.leiden_k, max(1, embeddings.shape[0] - 1))
        if k < 1:
            return np.zeros((embeddings.shape[0],), dtype=int)

        logger.info(f"Running Leiden clustering (k={k})...")
        knn_graph = kneighbors_graph(embeddings, k, mode="connectivity", include_self=False)
        sources, targets = knn_graph.nonzero()
        g = ig.Graph(n=embeddings.shape[0], edges=list(zip(sources, targets)), directed=False)
        g.simplify(multiple=True, loops=False)
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=self.cfg.leiden_seed)
        return np.asarray(partition.membership, dtype=int)

    def cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        logger.info(
            "Running HDBSCAN clustering "
            f"(min_cluster_size={self.cfg.hdbscan_min_cluster_size}, min_samples={self.cfg.hdbscan_min_samples}, "
            f"epsilon={self.cfg.hdbscan_cluster_selection_epsilon})..."
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.hdbscan_min_cluster_size,
            min_samples=self.cfg.hdbscan_min_samples,
            cluster_selection_epsilon=self.cfg.hdbscan_cluster_selection_epsilon,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        return clusterer.fit_predict(embeddings)

    def _resolve_kmeans_k(self, n_samples: int, hdbscan_labels: Optional[np.ndarray]) -> int:
        if self.cfg.kmeans_n_clusters.lower() != "auto":
            k = int(self.cfg.kmeans_n_clusters)
            return max(2, min(k, n_samples))

        # Auto: prefer HDBSCAN cluster count (excluding noise), fallback to sqrt heuristic.
        if hdbscan_labels is not None:
            k = len(set(hdbscan_labels.tolist()) - {-1})
            if k >= 2:
                return min(k, n_samples)

        return max(2, min(int(np.sqrt(n_samples)), n_samples))

    def cluster_kmeans(self, embeddings: np.ndarray, hdbscan_labels: Optional[np.ndarray]) -> np.ndarray:
        k = self._resolve_kmeans_k(n_samples=embeddings.shape[0], hdbscan_labels=hdbscan_labels)
        logger.info(f"Running KMeans clustering (n_clusters={k})...")
        clusterer = KMeans(n_clusters=k, random_state=self.cfg.kmeans_seed, n_init="auto")
        return clusterer.fit_predict(embeddings)


def save_bin_map(contig_ids: List[str], labels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"contig_id": contig_ids, "bin_id": labels})
    df.to_csv(output_path, index=False, sep="\t")


def save_embeddings_npz(
    contig_ids: List[str],
    contig_lengths: List[int],
    embeddings_by_type: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save embeddings for reuse in downstream plotting/evaluation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "contig_ids": np.asarray(contig_ids),
        "contig_lengths": np.asarray(contig_lengths, dtype=np.int32),
    }
    for emb_type, emb in embeddings_by_type.items():
        save_dict[emb_type] = emb.astype(np.float32, copy=False)
    np.savez_compressed(output_path, **save_dict)


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(
        description="Unsupervised binning for metagenomic contigs using ViralBERT embeddings. "
        "Outputs bin maps for {cls,projected} x {leiden,hdbscan,kmeans}."
    )
    parser.add_argument("--model_path", required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--input_fasta", required=True, help="Input FASTA file with contigs to be binned.")
    parser.add_argument("--output_dir", required=True, help="Directory to write bin map files.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix for output files. Defaults to input FASTA stem.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use, e.g. 'cuda:0' or 'cpu'.")

    # Embedding
    parser.add_argument("--embed_batch_size", type=int, default=64, help="Batch size for chunk embedding.")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length (including special tokens).")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window chunking.")
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="L2-normalize embeddings (recommended; matches evaluation geometry).",
    )
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save embeddings to a .npz file for reuse (contig_ids, contig_lengths, cls, projected).",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Where to save embeddings .npz. Defaults to <output_dir>/<output_prefix>_embeddings.npz",
    )

    # Leiden
    parser.add_argument("--leiden_k", type=int, default=15, help="k for kNN graph in Leiden.")
    parser.add_argument("--leiden_seed", type=int, default=42, help="Random seed for Leiden.")

    # HDBSCAN
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=5, help="HDBSCAN: minimum cluster size.")
    parser.add_argument("--hdbscan_min_samples", type=int, default=1, help="HDBSCAN: min samples.")
    parser.add_argument(
        "--hdbscan_cluster_selection_epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN: cluster selection epsilon.",
    )

    # KMeans
    parser.add_argument(
        "--kmeans_n_clusters",
        type=str,
        default="auto",
        help="KMeans cluster count. Use an integer or 'auto' (use HDBSCAN cluster count or sqrt heuristic).",
    )
    parser.add_argument("--kmeans_seed", type=int, default=42, help="Random seed for KMeans.")

    args = parser.parse_args()
    output_prefix = args.output_prefix or Path(args.input_fasta).stem

    return InferenceConfig(
        model_path=args.model_path,
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        output_prefix=output_prefix,
        device=args.device,
        embed_batch_size=args.embed_batch_size,
        max_length=args.max_length,
        stride=args.stride,
        normalize_embeddings=bool(args.normalize_embeddings),
        save_embeddings=bool(args.save_embeddings),
        embeddings_path=args.embeddings_path,
        leiden_k=args.leiden_k,
        leiden_seed=args.leiden_seed,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        kmeans_n_clusters=args.kmeans_n_clusters,
        kmeans_seed=args.kmeans_seed,
    )


def main() -> None:
    setup_logger()
    register_hf_classes()
    cfg = parse_args()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inferencer = BinningInferencer(cfg)
    contig_ids, contig_lengths, embeddings_by_type = inferencer.embed_fasta()

    if cfg.save_embeddings:
        embeddings_path = Path(cfg.embeddings_path) if cfg.embeddings_path else (out_dir / f"{cfg.output_prefix}_embeddings.npz")
        save_embeddings_npz(contig_ids, contig_lengths, embeddings_by_type, embeddings_path)
        logger.info(f"Saved embeddings: {embeddings_path}")

    # Run clustering for each embedding type and save results.
    for emb_type, emb in embeddings_by_type.items():
        logger.info(f"--- Clustering for embedding_type={emb_type}, shape={emb.shape} ---")

        labels_leiden = inferencer.cluster_leiden(emb)
        labels_hdbscan = inferencer.cluster_hdbscan(emb)
        labels_kmeans = inferencer.cluster_kmeans(emb, hdbscan_labels=labels_hdbscan)

        out_leiden = out_dir / f"{cfg.output_prefix}_{emb_type}_leiden_bin_map.tsv"
        out_hdbscan = out_dir / f"{cfg.output_prefix}_{emb_type}_hdbscan_bin_map.tsv"
        out_kmeans = out_dir / f"{cfg.output_prefix}_{emb_type}_kmeans_bin_map.tsv"

        save_bin_map(contig_ids, labels_leiden, out_leiden)
        save_bin_map(contig_ids, labels_hdbscan, out_hdbscan)
        save_bin_map(contig_ids, labels_kmeans, out_kmeans)

        logger.info(f"Saved: {out_leiden}")
        logger.info(f"Saved: {out_hdbscan}")
        logger.info(f"Saved: {out_kmeans}")


if __name__ == "__main__":
    main()
