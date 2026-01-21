# File: viralbert/utils/init_tokenizer.py

import logging
import torch
import torch.distributed as dist
from viralbert.data.hf_tokenizer import ViralBERTTokenizer

logger = logging.getLogger(__name__)


def init_tokenizer(
    config, 
    tokenizer_dir: str, 
    is_main_process: bool = True
):
    """
    Initialize a Hugging Face compatible tokenizer and update the config object.
    
    Loads the tokenizer, then synchronizes vocab size and special token IDs
    back into the provided config object.
    
    Args:
        config: ViralBERTConfig instance to be updated in-place.
        tokenizer_dir: Directory containing tokenizer files.
        is_main_process: Whether this is the main process (for logging).
        
    Returns:
        ViralBERTTokenizer: Initialized tokenizer instance.
    """
    if is_main_process:
        logger.info(f"Loading Hugging Face tokenizer from directory: {tokenizer_dir}...")

    # from_pretrained handles loading tokenizer.json, tokenizer_config.json etc.
    # The tokenizer's __init__ no longer needs the masking_strategy.
    tokenizer = ViralBERTTokenizer.from_pretrained(
        tokenizer_dir
    )

    # --- Synchronize tokenizer properties back to the config object ---
    
    # 1. Update vocab_size
    config.vocab_size = tokenizer.vocab_size
    
    # 2. Update special token IDs
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    if dist.is_initialized():
        # In distributed training, only the main process loads the file,
        # so we need to broadcast the definitive vocab size to all other processes.
        device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")
        vocab_size_tensor = torch.tensor(config.vocab_size, device=device)
        dist.broadcast(vocab_size_tensor, 0)
        config.vocab_size = int(vocab_size_tensor.item())
        dist.barrier()  # Ensure all processes have the same vocab size before proceeding

    if is_main_process:
        logger.info(f"Tokenizer vocabulary size set to: {config.vocab_size}")
        logger.info(f"Config updated with tokenizer IDs: pad={config.pad_token_id}, mask={config.mask_token_id}, cls={config.cls_token_id}, sep={config.sep_token_id}")

    return tokenizer
