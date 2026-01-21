# File: viralbert/utils/optimizer_utils.py

import logging
import torch.distributed as dist
from torch.optim import AdamW
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

logger = logging.getLogger(__name__)


def setup_optimizer(model, config, is_main_process: bool = True):
    """
    Setup optimizer based on configuration.
    
    Supports both MuonWithAuxAdam (mixed optimizer) and standard AdamW.
    For MuonWithAuxAdam, parameters are split based on dimensions and layer types.
    
    Args:
        model: The model to optimize
        config: ViralBERTConfig instance with optimizer settings
        is_main_process: Whether this is the main process (for logging)
        
    Returns:
        torch.optim.Optimizer: Configured optimizer instance
    """
    if config.optimizer_type == "muon_adamw":
        if is_main_process:
            logger.info("Using MuonWithAuxAdam optimizer.")

        hidden_weights, other_params, hidden_weight_names, other_param_names = [], [], [], []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Identify hidden weights for Muon (ndim >= 2) and exclude embeddings and classifier heads
            if param.ndim >= 2 and 'embeddings' not in name and 'mlm_head' not in name:
                hidden_weights.append(param)
                hidden_weight_names.append(name)
                if is_main_process:
                    logger.debug(f"Assigning to Muon: {name}")
            else:
                other_params.append(param)
                other_param_names.append(name)
                if is_main_process:
                    logger.debug(f"Assigning to AdamW: {name}")

        if not hidden_weights:
            logger.warning("No parameters were assigned to the Muon part of the optimizer. Check parameter grouping logic.")
            logger.info("Falling back to AdamW for all parameters.")
            # Standard AdamW setup if no params for Muon
            optimizer = _create_adamw_optimizer(model, config)
        else:
            # Muon and AdamW groups with their specific keys
            muon_group = dict(
                params=hidden_weights, 
                use_muon=True, 
                lr=config.muon_lr, 
                momentum=config.muon_momentum, 
                weight_decay=config.muon_weight_decay
            )
            adamw_group = dict(
                params=other_params, 
                use_muon=False, 
                lr=config.adamw_lr_for_muon_others, 
                betas=tuple(config.adamw_betas_for_muon_others), 
                eps=config.adamw_eps_for_muon_others, 
                weight_decay=config.adamw_weight_decay_for_muon_others
            )
            param_groups = [muon_group, adamw_group]
            
            # Select optimizer based on distributed training status
            if dist.is_initialized():
                optimizer = MuonWithAuxAdam(param_groups)
            else:
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

            if is_main_process:
                logger.info(f"Using {'MuonWithAuxAdam' if dist.is_initialized() else 'SingleDeviceMuonWithAuxAdam'}.")
                logger.info(f"Muon for hidden weights: {len(hidden_weights)} params, LR: {config.muon_lr}, WD: {config.muon_weight_decay}")
                for name in hidden_weight_names:
                    logger.info(f"  Muon Param: {name}")
                logger.info(f"AdamW for other params: {len(other_params)} params, LR: {config.adamw_lr_for_muon_others}, WD: {config.adamw_weight_decay_for_muon_others}")
                for name in other_param_names:
                    logger.info(f"  AdamW Param: {name}")
                    
    else:  # Default AdamW for all parameters
        optimizer = _create_adamw_optimizer(model, config)
        if is_main_process:
            logger.info(f"Using AdamW optimizer for all parameters with LR: {config.adamw_lr} and WD: {config.adamw_weight_decay}")

    return optimizer


def _create_adamw_optimizer(model, config):
    """
    Create standard AdamW optimizer with weight decay groups.
    
    Args:
        model: The model to optimize
        config: ViralBERTConfig instance with optimizer settings
        
    Returns:
        torch.optim.AdamW: Configured AdamW optimizer
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad], 
            "weight_decay": config.adamw_weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad], 
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=config.adamw_lr)