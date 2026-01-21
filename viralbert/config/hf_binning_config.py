# File: viralbert/config/hf_binning_config.py

from typing import Optional
from .hf_pretrain_config import ViralBERTConfig

class ViralBERTBinningConfig(ViralBERTConfig):
    """
    Configuration class for ViralBERT for Contrastive Binning.
    Inherits from ViralBERTConfig and adds parameters specific to the MoCo-based finetuning task.
    """
    model_type = "viralbert_for_contrastive_binning"
    base_model_prefix = "bert"

    def __init__(
        self,
        # MoCo-specific parameters
        moco_dim: int = 128,
        moco_k: int = 65536,
        moco_m: float = 0.999,
        moco_t: float = 0.07,
        
        # Training strategy
        unfreeze_last_n_layers: int = 0,

        **kwargs
    ):
        """
        Initializes the configuration.

        Args:
            moco_dim (int, optional): The feature dimension for the MoCo projection head. Defaults to 128.
            moco_k (int, optional): The size of the negative queue in MoCo. Defaults to 65536.
            moco_m (float, optional): The momentum for updating the key encoder in MoCo. Defaults to 0.999.
            moco_t (float, optional): The temperature for the softmax in the InfoNCE loss. Defaults to 0.07.
            unfreeze_last_n_layers (int, optional): Number of last BERT layers to unfreeze in the query encoder.
                                                  0 freezes all, -1 unfreezes all. Defaults to 0.
            **kwargs: Additional keyword arguments passed to the parent ViralBERTConfig.
        """
        super().__init__(**kwargs)
        self.moco_dim = moco_dim
        self.moco_k = moco_k
        self.moco_m = moco_m
        self.moco_t = moco_t
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
