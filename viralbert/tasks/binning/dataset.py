import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional
import random
import pyfaidx

logger = logging.getLogger(__name__)

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning.
    It operates directly on a FASTA file, dynamically generating positive pairs.
    A positive pair consists of two random segments sliced from the same long sequence (contig/genome).
    """
    def __init__(
        self,
        fasta_path: Union[str, Path],
        tokenizer,
        max_length: int,
        num_pairs_per_epoch: Optional[int],
        min_seq_len: Optional[int] = None,
        pair_sampling_strategy: str = "two_random_crops",
        max_pos_pair_offset: Optional[int] = None,
    ):
        """
        Args:
            fasta_path: Path to the FASTA data file.
            tokenizer: A tokenizer instance.
            max_length: The length of sequence fragments to generate.
            num_pairs_per_epoch: The virtual size of the dataset for one epoch. If None or 0,
                                 it will be calculated based on the total length of sequences.
            min_seq_len: Minimum length of a sequence in the FASTA to be considered for sampling.
                         Defaults to `max_length`.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # A sequence must be at least as long as the fragment we want to cut.
        self.min_seq_len = min_seq_len if min_seq_len is not None else self.max_length

        # Positive pair sampling strategy.
        # - two_random_crops: sample two independent crops from the same contig (current behavior).
        # - nearby_crops: sample the 2nd crop near the 1st crop to enforce partial overlap.
        # - same_crop: identical crops (useful for ablations).
        self.pair_sampling_strategy = pair_sampling_strategy
        self.max_pos_pair_offset = max_pos_pair_offset if max_pos_pair_offset is not None else (self.max_length // 2)

        logger.info(f"Loading FASTA index from {fasta_path}...")
        self._fasta = pyfaidx.Fasta(str(fasta_path))
        
        # Filter for sequences that are long enough for sampling
        self.contig_keys = [
            key for key in self._fasta.keys() if len(self._fasta[key]) >= self.min_seq_len
        ]
        
        if not self.contig_keys:
            raise ValueError(f"No sequences found in {fasta_path} with length >= {self.min_seq_len}. "
                             "Please check your FASTA file or `min_seq_len` parameter.")

        logger.info(f"Found {len(self.contig_keys)} sequences suitable for sampling (length >= {self.min_seq_len}).")
        
        if num_pairs_per_epoch is None or num_pairs_per_epoch <= 0:
            logger.info("`num_pairs_per_epoch` is not set. Calculating an automatic value based on total sequence length.")
            total_eligible_bp = sum(len(self._fasta[key]) for key in self.contig_keys)
            # Define an epoch as sampling enough pairs to roughly generate total_bp of sequence data.
            # Since each pair consists of two fragments, we divide by 2 * max_length.
            self.num_pairs_per_epoch = total_eligible_bp // (2 * self.max_length)
            logger.info(f"Automatically determined `num_pairs_per_epoch`: {self.num_pairs_per_epoch}")
        else:
            self.num_pairs_per_epoch = num_pairs_per_epoch
            logger.info(f"Using fixed `num_pairs_per_epoch`: {self.num_pairs_per_epoch}")
    
    def __len__(self):
        """Returns the virtual size of the dataset for one epoch."""
        return self.num_pairs_per_epoch
    
    def __getitem__(self, idx):
        """
        Generates one positive pair dynamically.
        `idx` is ignored, as pairs are generated randomly on the fly.
        """
        # 1. Select a random long contig
        contig_key = random.choice(self.contig_keys)
        long_sequence = self._fasta[contig_key]
        seq_len = len(long_sequence)
        
        # 2. Cut two random fragments from this contig to form a positive pair
        max_start = seq_len - self.max_length
        start1 = random.randint(0, max_start)
        seq1 = str(long_sequence[start1 : start1 + self.max_length])
        
        if self.pair_sampling_strategy == "two_random_crops":
            start2 = random.randint(0, max_start)
        elif self.pair_sampling_strategy == "nearby_crops":
            offset = random.randint(-self.max_pos_pair_offset, self.max_pos_pair_offset)
            start2 = start1 + offset
            start2 = max(0, min(start2, max_start))
        elif self.pair_sampling_strategy == "same_crop":
            start2 = start1
        else:
            raise ValueError(
                f"Unknown pair_sampling_strategy: {self.pair_sampling_strategy}. "
                "Supported: ['two_random_crops', 'nearby_crops', 'same_crop']"
            )

        seq2 = str(long_sequence[start2 : start2 + self.max_length])
        
        # 3. Tokenize the pair using the standard __call__ method
        encoding1 = self.tokenizer(
            seq1,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        
        encoding2 = self.tokenizer(
            seq2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt"
        )
        
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(0),
            'attention_mask_1': encoding1['attention_mask'].squeeze(0),
            'input_ids_2': encoding2['input_ids'].squeeze(0),
            'attention_mask_2': encoding2['attention_mask'].squeeze(0),
        }

