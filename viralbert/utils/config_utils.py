# File: viralbert/utils/config_utils.py

import json
import logging
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)


def check_config_consistency(old_config_path: str, new_config, mode: str, is_main_process: bool = True):
    """
    Check consistency between old checkpoint config and current config.
    
    Args:
        old_config_path: Path to the old config.json in checkpoint directory
        new_config: Current ViralBERTConfig instance
        mode: 'recovery' (strict) or 'epoch_extend' (lenient) or 'adaptation_extend' (most lenient)
        is_main_process: Whether this is the main process (for logging)
    
    Raises:
        ValueError: If critical parameters don't match according to mode requirements
    """
    if not is_main_process:
        return
        
    try:
        with open(old_config_path, 'r') as f:
            old_config_dict = json.load(f)
    except FileNotFoundError:
        logger.warning("No config.json found in checkpoint directory. Skipping consistency check.")
        return
    except Exception as e:
        logger.warning(f"Failed to load old config: {e}. Skipping consistency check.")
        return
    
    # Define critical parameters that must match
    model_structure_params = [
        'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
        'vocab_size', 'seq_length', 'position_embedding_type'
    ]
    
    training_params = [
        'batch_size', 'gradient_accumulation_steps', 
        'optimizer_type', 'learning_rate', 'weight_decay',
        'lr_scheduler_type', 'warmup_steps', 'min_lr_ratio', 'high_lr_steps_ratio'
    ]
    
    data_params = [
        'fasta_file', 'data_dir', 'mlm_probability', 'stride'
    ]
    
    errors = []
    warnings = []
    
    # Check model structure (always strict except for adaptation_extend)
    if mode != 'adaptation_extend':
        for param in model_structure_params:
            old_val = old_config_dict.get(param)
            new_val = getattr(new_config, param, None)
            if old_val != new_val:
                errors.append(f"Model structure parameter '{param}': old={old_val}, new={new_val}")
    else:
        # In adaptation_extend mode, only check architecture compatibility
        critical_arch_params = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'vocab_size']
        for param in critical_arch_params:
            old_val = old_config_dict.get(param)
            new_val = getattr(new_config, param, None)
            if old_val != new_val:
                errors.append(f"Critical architecture parameter '{param}': old={old_val}, new={new_val}")
        
        # For seq_length and position_embedding_type, just warn in adaptation_extend
        for param in ['seq_length', 'position_embedding_type']:
            old_val = old_config_dict.get(param)
            new_val = getattr(new_config, param, None)
            if old_val != new_val:
                warnings.append(f"Architecture parameter '{param}' changed: old={old_val}, new={new_val}")
    
    # Check training parameters based on mode
    for param in training_params:
        old_val = old_config_dict.get(param)
        new_val = getattr(new_config, param, None)
        if old_val != new_val:
            if mode == 'recovery':
                errors.append(f"Training parameter '{param}': old={old_val}, new={new_val}")
            elif mode == 'epoch_extend':
                if param in ['learning_rate', 'lr_scheduler_type', 'warmup_steps', 'min_lr_ratio', 'high_lr_steps_ratio']:
                    warnings.append(f"Training parameter '{param}' changed: old={old_val}, new={new_val}")
                else:
                    errors.append(f"Core training parameter '{param}': old={old_val}, new={new_val}")
            elif mode == 'adaptation_extend':
                # In adaptation mode, all training params can change - just log as info
                warnings.append(f"Training parameter '{param}' changed: old={old_val}, new={new_val}")
    
    # Check data parameters based on mode
    for param in data_params:
        old_val = old_config_dict.get(param)
        new_val = getattr(new_config, param, None)
        if old_val != new_val:
            if mode == 'recovery':
                errors.append(f"Data parameter '{param}': old={old_val}, new={new_val}")
            elif mode == 'epoch_extend':
                if param in ['fasta_file', 'data_dir']:
                    errors.append(f"Data parameter '{param}': old={old_val}, new={new_val}")
                else:
                    warnings.append(f"Data parameter '{param}' changed: old={old_val}, new={new_val}")
            elif mode == 'adaptation_extend':
                # In adaptation mode, data params can change - just warn for major changes
                if param in ['fasta_file', 'data_dir']:
                    warnings.append(f"Data parameter '{param}' changed: old={old_val}, new={new_val}")
                else:
                    warnings.append(f"Data parameter '{param}' changed: old={old_val}, new={new_val}")
    
    # Special check for num_train_epochs
    old_epochs = old_config_dict.get('num_train_epochs')
    new_epochs = getattr(new_config, 'num_train_epochs', None)
    if mode == 'recovery':
        if old_epochs != new_epochs:
            errors.append(f"num_train_epochs must be identical in recovery mode: old={old_epochs}, new={new_epochs}")
    elif mode == 'epoch_extend':
        if new_epochs is None or new_epochs < old_epochs:
            errors.append(f"num_train_epochs must be >= old value in epoch_extend mode: old={old_epochs}, new={new_epochs}")
        elif new_epochs > old_epochs:
            logger.info(f"Extending training from {old_epochs} to {new_epochs} epochs.")
    elif mode == 'adaptation_extend':
        # In adaptation mode, epochs can be anything
        if old_epochs != new_epochs:
            logger.info(f"Training epochs changed from {old_epochs} to {new_epochs} (adaptation mode).")
    
    # Report results
    if errors:
        error_msg = f"Config consistency check failed for mode '{mode}':\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_msg)
    
    if warnings:
        logger.warning("Config differences detected (may be intentional):")
        for warn in warnings:
            logger.warning(f"  - {warn}")
    
    logger.info(f"Config consistency check passed for mode '{mode}'.")


def validate_checkpoint_path(checkpoint_path: Union[str, Path]) -> Path:
    """
    Validate and normalize checkpoint path.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Path: Validated checkpoint path
        
    Raises:
        ValueError: If checkpoint path is invalid
    """
    if not checkpoint_path:
        raise ValueError("Checkpoint path cannot be empty")
    
    path = Path(checkpoint_path)
    
    if not path.exists():
        raise ValueError(f"Checkpoint path does not exist: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {path}")
    
    # Check for required files
    required_files = ['config.json']  # trainer_state.pt is optional
    missing_files = []
    
    for required_file in required_files:
        if not (path / required_file).exists():
            missing_files.append(required_file)
    
    if missing_files:
        logger.warning(f"Some checkpoint files are missing: {missing_files}")
    
    return path


def get_config_from_checkpoint(checkpoint_path: Union[str, Path]) -> dict:
    """
    Load configuration from checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        ValueError: If config cannot be loaded
    """
    path = validate_checkpoint_path(checkpoint_path)
    config_path = path / "config.json"
    
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return config_dict
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")
