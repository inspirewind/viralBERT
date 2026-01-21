import logging
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# New imports for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
# New imports for Leiden
import igraph as ig
import leidenalg
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as patches
# Import transforms directly from matplotlib
from matplotlib import transforms

logger = logging.getLogger(__name__)


class _EvalDataset(Dataset):
    """A simple dataset to hold sequences for embedding generation."""
    def __init__(self, sequences: List[str], tokenizer, max_length: int):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class ContrastiveEvaluator:
    """
    Handles the evaluation of contrastively fine-tuned models on a clustering task.
    """
    def __init__(self, model, tokenizer, device, batch_size=64,
                 clustering_algos: List[str] = None, metrics: List[str] = None,
                 eval_seed: Optional[int] = None,
                 embedding_types: Optional[List[str]] = None,
                 normalize_embeddings: bool = False,
                 max_length: Optional[int] = None,
                 visualization_embedding_type: Optional[str] = None):
        """
        Args:
            model: The model to evaluate. Can be either the backbone (e.g., model.bert)
                   or the full MoCo-style model that contains `bert` and `projector_q`.
            tokenizer: The nucleotide tokenizer.
            device: The device to run inference on ('cuda' or 'cpu').
            batch_size: Batch size for generating embeddings.
            clustering_algos: List of clustering algorithm names to use.
            metrics: List of metric names to compute.
            eval_seed: A random seed to ensure reproducible evaluation data generation.
            embedding_types: Embedding types to evaluate. Supported: ["cls", "projected"].
            normalize_embeddings: If True, L2-normalize embeddings before clustering (cosine-friendly).
            max_length: Override for maximum sequence length used in chunking.
            visualization_embedding_type: Which embedding type to use for UMAP/plots. Defaults to first.
        """
        self.model = model
        # Backbone is what produces `last_hidden_state`.
        self.backbone = model.bert if hasattr(model, "bert") else model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.eval_seed = eval_seed
        self.embedding_types = embedding_types if embedding_types is not None else ["cls"]
        self.normalize_embeddings = normalize_embeddings
        self.max_length_override = max_length
        self.visualization_embedding_type = visualization_embedding_type or (self.embedding_types[0] if self.embedding_types else "cls")
        
        # Available algorithms and metrics mapping
        self.algo_map = {"kmeans": KMeans, "hdbscan": hdbscan.HDBSCAN}
        self.metric_map = {
            "ari": adjusted_rand_score,
            "ami": adjusted_mutual_info_score,
            "silhouette": silhouette_score,
        }
        
        self.clustering_algos_to_use = clustering_algos if clustering_algos is not None else ["kmeans"]
        self.metrics_to_use = metrics if metrics is not None else ["ari", "silhouette"]
        self.custom_metrics = ["purity", "completeness", "f1"]

    def _resolve_max_length(self) -> int:
        if self.max_length_override is not None:
            return int(self.max_length_override)
        # Backward-compatible fallbacks for older configs.
        if hasattr(self.backbone.config, "max_length"):
            return int(getattr(self.backbone.config, "max_length"))
        return int(getattr(self.backbone.config, "seq_length"))

    @staticmethod
    def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return x / norms

    def _generate_eval_data(
        self,
        fasta_file: str,
        num_species: int,
        total_contigs: int,
        abundance_dist_params: Dict[str, Any],
        contig_length_dist_params: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Dynamically generates a more realistic evaluation dataset for metagenomic binning.
        It uses a fixed number of species and distributes a total number of contigs
        among them according to a specified abundance distribution. Contig lengths
        are also sampled from a specified distribution.
        """
        if self.eval_seed is not None:
            logger.info(f"Using fixed seed for evaluation data generation: {self.eval_seed}")
            random.seed(self.eval_seed)
            np.random.seed(self.eval_seed)

        logger.info(f"Generating evaluation data from {fasta_file} with {num_species} species and {total_contigs} total contigs.")

        all_records = list(SeqIO.parse(fasta_file, "fasta"))
        if len(all_records) < num_species:
            raise ValueError(f"FASTA file has {len(all_records)} records, but {num_species} were requested.")

        selected_species_records = random.sample(all_records, num_species)
        
        # 1. Generate abundance distribution for the selected species (contig counts)
        dist_name_abundance = abundance_dist_params.get("name", "lognormal")
        if dist_name_abundance == "lognormal":
            sigma_abundance = abundance_dist_params.get("sigma", 1.5)
            abundances = np.random.lognormal(mean=0, sigma=sigma_abundance, size=num_species)
        else: # Default to uniform if name is unknown
            logger.warning(f"Unknown abundance distribution '{dist_name_abundance}'. Defaulting to uniform.")
            abundances = np.ones(num_species)

        # Normalize abundances and calculate initial contig counts
        abundances_norm = abundances / np.sum(abundances)
        contigs_per_species_counts = np.round(abundances_norm * total_contigs).astype(int)
        
        # Ensure every species has at least one contig and adjust sum to match total_contigs exactly
        contigs_per_species_counts[contigs_per_species_counts == 0] = 1
        
        diff = total_contigs - np.sum(contigs_per_species_counts)
        
        while diff != 0:
            if diff > 0: # Need to add contigs
                idx_to_adjust = np.random.choice(num_species, p=abundances_norm)
                contigs_per_species_counts[idx_to_adjust] += 1
                diff -= 1
            else: # Need to remove contigs
                eligible_indices = np.where(contigs_per_species_counts > 1)[0]
                if len(eligible_indices) == 0:
                    logger.warning(f"Cannot reduce contig count to exactly {total_contigs} because all species have only 1 contig. Final count is {np.sum(contigs_per_species_counts)}.")
                    break
                
                eligible_abundances = abundances_norm[eligible_indices]
                p_eligible = eligible_abundances / np.sum(eligible_abundances) if np.sum(eligible_abundances) > 0 else None
                idx_to_remove_in_eligible = np.random.choice(len(eligible_indices), p=p_eligible)
                actual_idx_to_remove = eligible_indices[idx_to_remove_in_eligible]
                
                contigs_per_species_counts[actual_idx_to_remove] -= 1
                diff += 1
        
        logger.info(f"Final contig distribution across species (total={np.sum(contigs_per_species_counts)}): {contigs_per_species_counts}")

        # 2. Generate all contig lengths from a distribution
        total_contigs_generated = int(np.sum(contigs_per_species_counts))
        dist_name_len = contig_length_dist_params.get("name", "lognormal")
        min_len = contig_length_dist_params.get("min", 250)
        max_len = contig_length_dist_params.get("max", 8000)
        
        if dist_name_len == "lognormal":
            sigma_len = contig_length_dist_params.get("sigma", 1.0)
            raw_lengths = np.random.lognormal(mean=0, sigma=sigma_len, size=total_contigs_generated)
            if raw_lengths.max() == raw_lengths.min():
                 scaled_lengths = np.zeros_like(raw_lengths)
            else:
                scaled_lengths = (raw_lengths - raw_lengths.min()) / (raw_lengths.max() - raw_lengths.min())
            contig_lengths = min_len + scaled_lengths * (max_len - min_len)
        else: # Default to uniform
            contig_lengths = np.random.randint(min_len, max_len + 1, size=total_contigs_generated)

        contig_lengths = np.clip(contig_lengths, min_len, max_len).astype(int)
        np.random.shuffle(contig_lengths)
        
        # 3. Generate contigs based on the distributions
        data = []
        species_map = {record.id: i for i, record in enumerate(selected_species_records)}
        current_contig_idx = 0

        for species_record, num_contigs in tqdm(zip(selected_species_records, contigs_per_species_counts), total=num_species, desc="Generating contigs"):
            seq = str(species_record.seq)
            seq_len = len(seq)
            species_id = species_map[species_record.id]

            for _ in range(int(num_contigs)):
                if current_contig_idx >= len(contig_lengths): break
                contig_len = contig_lengths[current_contig_idx]
                current_contig_idx += 1
                
                if seq_len <= contig_len:
                    contig = seq
                else:
                    start = random.randint(0, seq_len - contig_len)
                    contig = seq[start : start + contig_len]

                data.append({"sequence": contig, "species_id": species_id, "source_id": species_record.id})
        
        return pd.DataFrame(data)

    @torch.no_grad()
    def _generate_embeddings(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generates embeddings for all sequences in the DataFrame.
        Uses a sliding window approach for sequences longer than the model's max_length.
        """
        self.model.eval()
        self.backbone.eval()
        
        max_len = self._resolve_max_length()
        stride = max_len // 2  # 50% overlap
        
        all_chunks = []
        chunk_to_seq_map = []  # Maps each chunk back to its original sequence index
        
        logger.info(f"Preparing sequence chunks for embedding (max_length={max_len}, stride={stride})...")
        sequences = df["sequence"].tolist()
        
        for i, seq in enumerate(sequences):
            if len(seq) <= max_len:
                all_chunks.append(seq)
                chunk_to_seq_map.append(i)
            else:
                # Sliding window for long sequences
                for start in range(0, len(seq) - max_len + 1, stride):
                    chunk = seq[start : start + max_len]
                    all_chunks.append(chunk)
                    chunk_to_seq_map.append(i)
                # Ensure the end of the sequence is also included if missed by the stride
                if (len(seq) - max_len) % stride != 0:
                     last_chunk = seq[-max_len:]
                     all_chunks.append(last_chunk)
                     chunk_to_seq_map.append(i)


        # Create dataset and dataloader for the flattened chunks
        dataset = _EvalDataset(all_chunks, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Generate embeddings for all chunks in batches
        chunk_embeddings_by_type: Dict[str, List[np.ndarray]] = {t: [] for t in self.embedding_types}
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        for batch in tqdm(dataloader, desc="Generating Chunk Embeddings"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.backbone(**batch)
            cls = outputs["last_hidden_state"][:, 0]

            for emb_type in self.embedding_types:
                if emb_type == "cls":
                    emb_t = cls
                elif emb_type == "projected":
                    if not hasattr(self.model, "projector_q"):
                        raise ValueError("Embedding type 'projected' requires the full model with `projector_q`.")
                    emb_t = self.model.projector_q(cls)
                else:
                    raise ValueError(f"Unsupported embedding type: {emb_type}. Supported: ['cls', 'projected']")

                if self.normalize_embeddings:
                    emb_t = F.normalize(emb_t, dim=1)

                chunk_embeddings_by_type[emb_type].append(emb_t.cpu().numpy())
        
        # Handle empty case
        if not any(chunk_embeddings_by_type[t] for t in chunk_embeddings_by_type):
            logger.warning("No embeddings were generated. Returning empty array.")
            return {t: np.array([]) for t in self.embedding_types}
        
        # Aggregate chunk embeddings back to sequence embeddings via averaging
        num_original_sequences = len(sequences)
        final_embeddings_by_type: Dict[str, np.ndarray] = {}

        logger.info("Aggregating chunk embeddings to sequence embeddings...")
        # Precompute indices per sequence to avoid O(N^2) list scans.
        seq_to_chunk_indices: List[List[int]] = [[] for _ in range(num_original_sequences)]
        for j, map_idx in enumerate(chunk_to_seq_map):
            seq_to_chunk_indices[map_idx].append(j)

        for emb_type in self.embedding_types:
            chunk_embeddings = np.vstack(chunk_embeddings_by_type[emb_type])
            final_embeddings = np.zeros((num_original_sequences, chunk_embeddings.shape[1]), dtype=chunk_embeddings.dtype)

            for seq_idx in tqdm(range(num_original_sequences), desc=f"Aggregating Embeddings ({emb_type})"):
                indices = seq_to_chunk_indices[seq_idx]
                if indices:
                    final_embeddings[seq_idx] = np.mean(chunk_embeddings[indices], axis=0)
                else:
                    logger.warning(f"Sequence {seq_idx} (original index) produced no chunks. Its embedding will be zero.")

            if self.normalize_embeddings and final_embeddings.size > 0:
                final_embeddings = self._l2_normalize_np(final_embeddings)

            final_embeddings_by_type[emb_type] = final_embeddings

        return final_embeddings_by_type

    def _plot_confidence_ellipse(self, ax, points, n_std=2.0, **kwargs):
        """
        Create a plot of the covariance confidence ellipse of `x` and `y`
        """
        if points.shape[0] < 2:
            return

        cov = np.cov(points, rowvar=False)
        # Check for singular covariance matrix
        if np.isclose(np.linalg.det(cov), 0):
            logger.warning("Cannot plot ellipse for singular covariance matrix.")
            return
            
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                              facecolor='none', **kwargs)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(points[:, 0])

        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(points[:, 1])

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)


    def _visualize_clusters(
        self, 
        embeddings_2d: np.ndarray, 
        true_labels: np.ndarray, 
        all_predicted_labels: Dict[str, np.ndarray],
        visualization_algo: Optional[str],
        output_dir: Path,
        global_step: int,
        id_to_accession_map: Dict[int, str]
    ):
        """
        Generates visualizations: one colored by true labels, and one for a
        specific clustering algorithm with hulls drawn around predicted clusters.
        """
        num_true_labels = len(np.unique(true_labels))
        # Use a qualitative colormap for better distinction between categories
        if num_true_labels <= 20:
            cmap_true = plt.get_cmap('tab20', num_true_labels)
        else:
            logger.warning(f"Number of true labels ({num_true_labels}) > 20. Using a cyclical colormap (hsv) which may result in non-unique colors.")
            cmap_true = plt.get_cmap('hsv', num_true_labels)

        # Plot 1: Colored by true species (the ground truth map)
        fig_true, ax_true = plt.subplots(figsize=(16, 12))
        scatter_true = ax_true.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap=cmap_true, alpha=0.7, s=20)
        
        ax_true.set_title(f'UMAP Projection (Colored by True Species)\nStep: {global_step}', fontsize=16)
        ax_true.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax_true.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax_true.grid(True, linestyle='--', alpha=0.5)

        # Add legend for true labels
        if num_true_labels <= 25: # Only show detailed legend for a reasonable number of classes
             legend1 = ax_true.legend(*scatter_true.legend_elements(), title="Species", loc="upper right", bbox_to_anchor=(1.15, 1))
             ax_true.add_artist(legend1)
        
        save_path_true = output_dir / f"clustering_visualization_step_{global_step}_true_labels.png"
        fig_true.savefig(save_path_true, dpi=150, bbox_inches='tight')
        plt.close(fig_true)

        # Plot 2: A single plot with ellipses for the specified algorithm
        if visualization_algo and visualization_algo in all_predicted_labels:
            predicted_labels = all_predicted_labels[visualization_algo]
            
            fig_pred, ax_pred = plt.subplots(figsize=(16, 12))
            
            # Base scatter plot is still colored by TRUE labels
            scatter_pred = ax_pred.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap=cmap_true, alpha=0.4, s=20)
            
            # Overlay the confidence ellipses for the PREDICTED clusters
            unique_pred_labels = np.unique(predicted_labels)
            num_pred_labels = len(unique_pred_labels)
            # Use a different qualitative colormap for ellipses
            if num_pred_labels <= 10:
                hull_cmap = plt.get_cmap('tab10', num_pred_labels)
            elif num_pred_labels <= 20:
                hull_cmap = plt.get_cmap('tab20', num_pred_labels)
            else:
                logger.warning(f"Number of predicted clusters ({num_pred_labels}) > 20. Using a cyclical colormap (hsv) for ellipses which may result in non-unique colors.")
                hull_cmap = plt.get_cmap('hsv', num_pred_labels)
            
            for i, label in enumerate(unique_pred_labels):
                if label == -1: 
                    continue
                points = embeddings_2d[predicted_labels == label]
                if points.shape[0] < 3: 
                    continue
                self._plot_confidence_ellipse(ax_pred, points, edgecolor=hull_cmap(i), lw=2)

            ax_pred.set_title(f'UMAP Projection with {visualization_algo.capitalize()} Cluster Ellipses\nStep: {global_step}', fontsize=16)
            ax_pred.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax_pred.set_ylabel('UMAP Dimension 2', fontsize=12)
            ax_pred.grid(True, linestyle='--', alpha=0.5)

            # Add combined legend
            if num_true_labels <= 25 and num_pred_labels <= 25:
                # Legend for scatter points (true species)
                legend_species = ax_pred.legend(*scatter_pred.legend_elements(), title="True Species", loc="upper left", bbox_to_anchor=(1.02, 1))
                ax_pred.add_artist(legend_species)
                
                # Legend for ellipses (predicted clusters)
                from matplotlib.lines import Line2D
                legend_elements_hulls = [Line2D([0], [0], color=hull_cmap(i), lw=2, label=f'Cluster {label}') for i, label in enumerate(unique_pred_labels) if label != -1]
                ax_pred.legend(handles=legend_elements_hulls, title=f"{visualization_algo.capitalize()} Clusters", loc="lower left", bbox_to_anchor=(1.02, 0))

            save_path_pred = output_dir / f"clustering_visualization_step_{global_step}_{visualization_algo}_ellipses.png"
            fig_pred.savefig(save_path_pred, dpi=150, bbox_inches='tight')
            plt.close(fig_pred)

            # --- NEW: Plot Contingency Matrix ---
            logger.info(f"Generating contingency matrix heatmap for {visualization_algo}...")
            self._plot_contingency_matrix(
                true_labels=true_labels,
                predicted_labels=predicted_labels,
                output_dir=output_dir,
                global_step=global_step,
                algo_name=visualization_algo,
                id_to_accession_map=id_to_accession_map
            )

        elif visualization_algo:
            logger.warning(f"Visualization algorithm '{visualization_algo}' was specified, but its results were not found. Skipping hull plot.")


    def _plot_contingency_matrix(
        self,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        output_dir: Path,
        global_step: int,
        algo_name: str,
        id_to_accession_map: Dict[int, str]
    ):
        """
        Generates and saves a heatmap of the contingency matrix.
        """
        # Filter out noise points (e.g., from HDBSCAN)
        valid_indices = predicted_labels != -1
        if not np.any(valid_indices):
            logger.warning(f"No valid clusters found for {algo_name}. Skipping contingency matrix plot.")
            return

        true_labels_f = true_labels[valid_indices]
        predicted_labels_f = predicted_labels[valid_indices]

        # Use pandas crosstab for an intuitive and labeled contingency matrix
        df_contingency = pd.crosstab(true_labels_f, predicted_labels_f, rownames=['True Species ID'], colnames=['Predicted Cluster ID'])
        
        # --- NEW: Relabel y-axis with accession numbers ---
        if id_to_accession_map:
            try:
                y_tick_labels = [id_to_accession_map[int(label)] for label in df_contingency.index]
                df_contingency.index = y_tick_labels
                df_contingency.index.name = "True Species Accession"
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not map species IDs to accessions for plot: {e}. Using numerical IDs.")
        
        # Reorder columns to be as diagonal as possible using the Hungarian algorithm
        try:
            # We want to maximize the sum of diagonal elements. The assignment algorithm finds a
            # minimum cost, so we use the negative of the contingency values as the cost.
            cost_matrix = -df_contingency.values
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # col_ind gives the optimal column index for each row. We use this to reorder.
            # Get the original column names (cluster IDs) in the new optimal order.
            new_col_order = df_contingency.columns[col_ind]
            df_contingency = df_contingency[new_col_order]
            logger.info(f"Reordered contingency matrix for '{algo_name}' to improve diagonal alignment.")
        except ValueError:
            # This can happen if the contingency matrix is empty after filtering.
            logger.warning(f"Could not reorder contingency matrix for '{algo_name}', likely due to empty data. Skipping reordering.")
        
        # Dynamically adjust figure size for readability
        # Base size for each cell, plus some margin for labels
        cell_size_inch = 0.5
        figsize_x = max(12, len(df_contingency.columns) * cell_size_inch + 3)
        figsize_y = max(10, len(df_contingency.index) * cell_size_inch + 2)

        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
        sns.heatmap(
            df_contingency, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            ax=ax,
            linewidths=.5,
            cbar_kws={"shrink": 0.7} # Make color bar a bit smaller
        )
        
        # --- KEY CHANGE ---
        # Enforce square cells
        ax.set_aspect('equal')

        ax.set_title(
            f'Contingency Matrix Heatmap ({algo_name.capitalize()})\nStep: {global_step}', 
            fontsize=16
        )
        ax.tick_params(axis='y', labelrotation=0)
        ax.tick_params(axis='x', labelrotation=45)

        save_path = output_dir / f"contingency_matrix_step_{global_step}_{algo_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


    def _calculate_purity_completeness_f1(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculates Purity, Completeness, and F1 score for clustering,
        which are common metrics in metagenomic binning.
        """
        # Exclude noise points from HDBSCAN (labeled as -1)
        valid_indices = predicted_labels != -1
        if not np.any(valid_indices):
            return {"purity": 0.0, "completeness": 0.0, "f1": 0.0}

        true_labels_f = true_labels[valid_indices]
        predicted_labels_f = predicted_labels[valid_indices]

        # Use contingency matrix for efficient calculation
        classes, class_idx = np.unique(true_labels_f, return_inverse=True)
        clusters, cluster_idx = np.unique(predicted_labels_f, return_inverse=True)
        contingency = contingency_matrix(class_idx, cluster_idx)

        # --- Purity (Precision) ---
        # For each cluster, find the number of points from the dominant true class.
        dominant_class_counts_per_cluster = np.max(contingency, axis=0)
        purity = np.sum(dominant_class_counts_per_cluster) / len(true_labels_f)

        # --- Completeness (Recall) ---
        # Find dominant true class for each cluster
        dominant_class_indices = np.argmax(contingency, axis=0)
        dominant_true_labels = classes[dominant_class_indices]
        
        # Get counts of all true labels in the original (unfiltered) dataset
        original_true_counts = dict(zip(*np.unique(true_labels, return_counts=True)))
        
        total_counts_of_dominant_classes = np.array([original_true_counts.get(label, 1) for label in dominant_true_labels])
        total_counts_of_dominant_classes[total_counts_of_dominant_classes == 0] = 1 # avoid div by zero

        recalls_per_cluster = dominant_class_counts_per_cluster / total_counts_of_dominant_classes
        
        cluster_sizes = np.sum(contingency, axis=0)
        completeness = np.sum(cluster_sizes * recalls_per_cluster) / len(true_labels_f)

        # --- F1 Score ---
        f1 = 2 * (purity * completeness) / (purity + completeness) if (purity + completeness) > 0 else 0.0
        
        return {"purity": purity, "completeness": completeness, "f1": f1}

    def _run_clustering_and_evaluate(self, embeddings: np.ndarray, true_labels: np.ndarray) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Runs clustering algorithms, computes metrics, and returns both metrics and all predicted labels.
        """
        n_clusters = len(np.unique(true_labels))
        results = {}
        all_predicted_labels = {}

        logger.info(f"Running evaluation with algorithms: {self.clustering_algos_to_use}")
        
        for algo_name in self.clustering_algos_to_use:
            # Step 1: Run Clustering
            logger.info(f"Applying {algo_name} clustering...")
            predicted_labels = None
            try:
                if algo_name == "kmeans":
                    clusterer = self.algo_map[algo_name](n_clusters=n_clusters, random_state=42, n_init='auto')
                    predicted_labels = clusterer.fit_predict(embeddings)
                elif algo_name == "hdbscan":
                    # Using core_dist_n_jobs=-1 for parallelization
                    clusterer = self.algo_map[algo_name](min_cluster_size=5, min_samples=1, core_dist_n_jobs=-1)
                    predicted_labels = clusterer.fit_predict(embeddings)
                elif algo_name == "leiden":
                    # Leiden algorithm requires building a k-NN graph first.
                    k = 15  # A common choice, matching UMAP's default n_neighbors
                    logger.info(f"Constructing k-NN graph (k={k}) for Leiden...")
                    knn_graph = kneighbors_graph(embeddings, k, mode='connectivity', include_self=False)
                    
                    sources, targets = knn_graph.nonzero()
                    g = ig.Graph(n=embeddings.shape[0], edges=list(zip(sources, targets)), directed=False)
                    g.simplify(multiple=True, loops=False) # Remove duplicates and self-loops

                    logger.info("Running Leiden algorithm on the k-NN graph...")
                    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
                    predicted_labels = np.array(partition.membership)
                else:
                    logger.warning(f"Clustering algorithm '{algo_name}' not supported. Skipping.")
                    continue

            except Exception as e:
                logger.error(f"Clustering with {algo_name} failed: {e}", exc_info=True)
                continue
            
            if predicted_labels is None:
                continue

            all_predicted_labels[algo_name] = predicted_labels

            # Step 2: Calculate Metrics for this algorithm
            logger.info(f"Calculating metrics for {algo_name} results...")
            
            # Handle noise points (-1) for label-based metrics
            valid_indices = predicted_labels != -1
            true_labels_filtered = true_labels[valid_indices]
            predicted_labels_filtered = predicted_labels[valid_indices]
            
            # Calculate custom binning metrics
            if any(m in self.custom_metrics for m in self.metrics_to_use):
                custom_scores = self._calculate_purity_completeness_f1(true_labels, predicted_labels)
                for m_name in self.custom_metrics:
                    if m_name in self.metrics_to_use:
                        results[f"{algo_name}_{m_name}"] = custom_scores[m_name]

            # Calculate other metrics
            for metric_name in self.metrics_to_use:
                if metric_name in self.custom_metrics:
                    continue # Already handled

                key = f"{algo_name}_{metric_name}"
                try:
                    if metric_name == "silhouette":
                        # Silhouette needs at least 2 clusters to be computed
                        if len(np.unique(predicted_labels)) > 1:
                            results[key] = silhouette_score(embeddings, predicted_labels)
                        else:
                            results[key] = -1.0 # Not applicable
                    elif metric_name in self.metric_map: # ARI, AMI
                        # These metrics work on filtered labels (without noise)
                        if len(true_labels_filtered) > 0 and len(np.unique(predicted_labels_filtered)) > 0:
                            results[key] = self.metric_map[metric_name](true_labels_filtered, predicted_labels_filtered)
                        else:
                             results[key] = 0.0 # No non-noise points
                except Exception as e:
                    logger.warning(f"Could not compute metric '{key}': {e}")
                    results[key] = -1.0 # Error during computation

        return results, all_predicted_labels

    def run(
        self,
        fasta_file: str,
        num_species: int,
        total_contigs: int,
        abundance_dist_params: Dict[str, Any],
        contig_length_dist_params: Dict[str, Any],
        output_dir: Path,
        global_step: int,
        visualization_algo: Optional[str] = None
    ) -> Dict[str, float]:
        """
        The main public method to run the full evaluation pipeline and generate visualizations.
        """
        try:
            eval_df = self._generate_eval_data(
                fasta_file,
                num_species,
                total_contigs,
                abundance_dist_params,
                contig_length_dist_params
            )
            if eval_df.empty:
                logger.warning("Evaluation DataFrame is empty. Skipping evaluation.")
                return {}

            embeddings_by_type = self._generate_embeddings(eval_df)
            true_labels = eval_df["species_id"].to_numpy()
            
            # --- NEW: Create map from species ID to accession ---
            id_to_accession_map = dict(eval_df[['species_id', 'source_id']].drop_duplicates().itertuples(index=False))

            all_results: Dict[str, float] = {}
            all_predicted_labels_for_vis: Dict[str, np.ndarray] = {}
            embeddings_for_vis: Optional[np.ndarray] = None

            for emb_type, embeddings in embeddings_by_type.items():
                if embeddings.size == 0:
                    continue

                # Lightweight diagnostics for representation drift
                norms = np.linalg.norm(embeddings, axis=1)
                all_results[f"{emb_type}_embedding_norm_mean"] = float(np.mean(norms))
                all_results[f"{emb_type}_embedding_norm_std"] = float(np.std(norms))

                results, all_predicted_labels = self._run_clustering_and_evaluate(embeddings, true_labels)
                for k, v in results.items():
                    all_results[f"{emb_type}_{k}"] = v

                if emb_type == self.visualization_embedding_type:
                    all_predicted_labels_for_vis = all_predicted_labels
                    embeddings_for_vis = embeddings

            if embeddings_for_vis is not None and all_predicted_labels_for_vis:
                logger.info("Reducing embedding dimensionality for visualization using UMAP...")
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                embeddings_2d = reducer.fit_transform(embeddings_for_vis)

                self._visualize_clusters(
                    embeddings_2d=embeddings_2d,
                    true_labels=true_labels,
                    all_predicted_labels=all_predicted_labels_for_vis,
                    visualization_algo=visualization_algo,
                    output_dir=output_dir,
                    global_step=global_step,
                    id_to_accession_map=id_to_accession_map
                )

            return all_results
        except Exception as e:
            logger.error(f"An error occurred during evaluation: {e}", exc_info=True)
            return {}

