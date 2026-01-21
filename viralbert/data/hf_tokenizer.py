# File: viralbert/data/hf_tokenizer.py

import os
import json
from typing import List, Optional, Dict, Union
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import logging

logger = logging.getLogger(__name__)

class ViralBERTTokenizer(PreTrainedTokenizer):
    """
    ViralBERT Tokenizer that inherits from PreTrainedTokenizer for full HuggingFace compatibility.
    
    This tokenizer handles nucleotide sequences (A, C, G, T, N) with proper special token support
    and is fully compatible with DataCollatorForLanguageModeling and other HF utilities.
    """
    
    vocab_files_names = {"vocab_file": "tokenizer.json"}
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        # Define the standard nucleotide vocabulary
        self.standard_bases = ['A', 'C', 'G', 'T']
        
        # Build a default vocabulary in case file loading fails
        special_tokens = [pad_token, cls_token, sep_token, mask_token, unk_token]
        nucleotides = self.standard_bases + ['N']
        
        # Create vocab mapping
        vocab = special_tokens + nucleotides
        self._vocab = {token: idx for idx, token in enumerate(vocab)}
        self._vocab_reverse = {idx: token for token, idx in self._vocab.items()}
        
        # If vocab_file is provided, load from it. This is the standard path.
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab_from_file(vocab_file)
        
        # All potential special tokens, including structural ones, should be in the vocab file.
        # We define them here to pass to the base class constructor.
        additional_special_tokens = [
            token for token in ["[M_S]", "[M_B]", "[M_I]", "[M_E]"] 
            if token in self._vocab
        ]

        # Handle additional_special_tokens to avoid parameter conflicts
        # If additional_special_tokens is already in kwargs, merge with our tokens
        if 'additional_special_tokens' in kwargs:
            existing_tokens = kwargs.pop('additional_special_tokens') or []
            # Combine and deduplicate
            all_additional_tokens = list(set(existing_tokens + additional_special_tokens))
        else:
            all_additional_tokens = additional_special_tokens

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token, 
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=all_additional_tokens,
            **kwargs
        )

        # After super().__init__, re-sync our internal vocab from the base class's full vocab.
        # This ensures that token IDs assigned by the base class are respected.
        self._vocab = self.get_vocab()
        self._vocab_reverse = {idx: token for token, idx in self._vocab.items()}
        
        logger.info(f"Initialized ViralBERTTokenizer with vocabulary size: {len(self._vocab)}")
    
    def _load_vocab_from_file(self, vocab_file: str):
        """Load vocabulary from existing tokenizer file."""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'token_to_id' in data:
                self._vocab = data['token_to_id']
                self._vocab_reverse = {idx: token for token, idx in self._vocab.items()}
                logger.info(f"Loaded vocabulary from {vocab_file}")
            else:
                logger.warning(f"No 'token_to_id' found in {vocab_file}, using default vocab")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load vocab from {vocab_file}: {e}, using default vocab")
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return self._vocab.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a nucleotide sequence into individual characters.
        
        Args:
            text: Input nucleotide sequence
            
        Returns:
            List of tokens (individual nucleotides)
        """
        # Preprocess: convert to uppercase and replace invalid chars with N
        processed = ''.join(
            c if c.upper() in self.standard_bases else 'N' 
            for c in text.upper() 
            if c.isalpha()  # Only keep alphabetic characters
        )
        
        # Return list of individual characters
        return list(processed)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its corresponding ID."""
        return self._vocab.get(token, self._vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its corresponding token."""
        return self._vocab_reverse.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens back to a string.
        For nucleotide sequences, we just concatenate without spaces.
        """
        # Filter out special tokens and concatenate
        nucleotide_tokens = [
            token for token in tokens 
            if token not in self.all_special_tokens
        ]
        return ''.join(nucleotide_tokens)
    
    def build_inputs_with_special_tokens(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence by adding special tokens.
        
        A ViralBERT sequence has the following format:
        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
    
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, 
                token_ids_1=token_ids_1, 
                already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    
    def create_token_type_ids_from_sequences(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for sequence pair classification.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """
        Save the tokenizer vocabulary to a file.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        
        vocab_data = {
            'type': 'viralbert',
            'token_to_id': self._vocab,
            'standard_bases': self.standard_bases,
            'vocab_size': len(self._vocab),
            'special_tokens': {
                'unk_token': self.unk_token,
                'sep_token': self.sep_token,
                'pad_token': self.pad_token, 
                'cls_token': self.cls_token,
                'mask_token': self.mask_token,
            }
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Vocabulary saved to {vocab_file}")
        return (vocab_file,)
    


# Convenience function to create the tokenizer  
def create_viralbert_tokenizer(
    vocab_file: Optional[str] = None
) -> ViralBERTTokenizer:
    """
    Create a ViralBERTTokenizer instance.
    
    Args:
        vocab_file: Optional path to existing vocabulary file
        
    Returns:
        ViralBERTTokenizer instance
    """
    return ViralBERTTokenizer(vocab_file=vocab_file)
