# ViralBERT

[![wakatime](https://wakatime.com/badge/user/6b890465-2476-4778-97e1-8906ea763f76/project/a3596834-1fb9-4781-8d68-91bf5d807a6f.svg)](https://wakatime.com/badge/user/6b890465-2476-4778-97e1-8906ea763f76/project/a3596834-1fb9-4781-8d68-91bf5d807a6f)

**ViralBERT** is a nucleotide language model pre-trained on large-scale metagenomic datasets, designed to tackle a wide range of tasks in viral metagenomics analysis. By leveraging deep contextual representations, ViralBERT enables powerful downstream applications including embedding extraction, binning, and viral identification.

## Installation

### Linux (CUDA)
```bash
conda create -n viralbert python=3.10
conda activate viralbert
pip install -r requirements.txt
```

### Mac (MPS)
```bash
conda create -n viralbert python=3.10
conda activate viralbert
pip install -r requirements_mps.txt
```

## Model Zoo

To run inference, you will need to download the pre-trained weights.

| Model / Dataset | Description | Link |
| :--- | :--- | :--- |
| **Pre-train Stage 1** | Initial pre-training stage with assembled genomes. | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/inspirewind/viralBERT-pretrain-stage1) |
| **Pre-train Stage 2** | Refined pre-training stage with MAGs. | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/inspirewind/viralBERT-pretrain-stage2) |
| **Binning Model** | Fine-tuned with contrastive learning for metagenomic binning. | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/inspirewind/viralBERT-binning) |
| **Classification Model** | Fine-tuned for viral identification or taxonomy classification. | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/inspirewind/viralBERT-classification) |
| **Pre-train Dataset** | Large-scale metagenomic dataset used for pre-training. | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/inspirewind/viralBERT-pretrain-dataset) |

## Inference & Usage

ViralBERT provides three main inference scripts tailored for different analysis needs.

### 1. General Embedding Extraction
> **Best for:** Dimensionality reduction (t-SNE/UMAP), sequence clustering, or feature engineering for downstream sequnece analysis.

Use the base pre-trained model to extract deep semantic vectors from DNA sequences via `scripts/inference_pretrain.py`.

**Usage:**
```bash
python scripts/inference_pretrain.py \
    --model_path models/viralbert_base \
    --input_fasta data/my_sequences.fasta \
    --output_dir results/embeddings \
    --device cuda:0
```

**Key Arguments:**
*   `--pooling`: Strategy to derive chunk embeddings (`mean` or `cls`). Default: `mean`.
*   `--combine`: Aggregates chunk embeddings into a single sequence vector. Use `--no-combine` to output embeddings for every individual chunk.
*   `--batch_size`: Batch size for inference. Default: 32.

**Outputs:**
*   `embeddings.npy`: The embedding matrix (shape: `[N_seqs, Hidden_dim]`).
*   `metadata.csv`: Corresponding sequence IDs, lengths, and chunk counts.

### 2. Metagenomic Binning
> **Best for:** Recovering viral genomes (vMAGs) from mixed metagenomic data using unsupervised clustering.

The `scripts/inference_binning.py` pipeline uses a contrastive-learning fine-tuned model to generate embeddings optimized for separating distinct genomes, followed by clustering algorithms (Leiden, HDBSCAN, or KMeans).

**Usage:**
```bash
python scripts/inference_binning.py \
    --model_path models/viralbert_binning \
    --input_fasta data/metagenome.fasta \
    --output_dir results/binning \
    --device cuda:0
```

**Key Arguments:**
*   **Clustering Parameters:**
    *   `--leiden_k`: Number of neighbors for Leiden graph construction (default: 15).
    *   `--hdbscan_min_cluster_size`: Minimum cluster size for HDBSCAN (default: 5).
    *   `--kmeans_n_clusters`: Number of clusters for KMeans (can be an integer or `auto`).
*   **Embeddings:**
    *   `--save_embeddings`: If set, saves the intermediate embeddings to a `.npz` file for reuse.

**Outputs:**
The script generates `TSV` files mapping `contig_id` to `bin_id` for different combinations of embedding types (CLS, Projected, MeanPool) and clustering algorithms:
*   e.g., `prefix_mean_pool_leiden_bin_map.tsv`

### 3. Sequence Classification
> **Best for:** Identifying viral contigs vs. host, or classifying viruses into families/genera.

`scripts/inference_classification.py` classifies sequences using a supervised fine-tuned model. It handles long sequences by splitting them into chunks and aggregating predictions via **Voting** or **Probability Averaging**.

**Usage:**
```bash
python scripts/inference_classification.py \
    --model_path models/viralbert_classification \
    --input_fasta data/unknown_contigs.fasta \
    --output_path results/classification.csv \
    --device cuda:0
```

**Key Arguments:**
*   `--inference_max_len`: Max token length per chunk (default: 512).
*   `--batch_size`: Inference batch size per GPU.
*   `-c` / `--continue`: Resume from an existing output file if the process was interrupted.

**Outputs:**
A CSV file containing:
*   `predicted_label_vote`: Prediction based on majority vote of chunks.
*   `predicted_label_prob`: Prediction based on averaged probabilities.
*   `prob_<class>`: Probability scores for each class.

## Citation


## License

This project is licensed under the MIT License.
