from viralbert.models.model import ViralBERTModel
from viralbert.models.losses import FocalLoss, get_auto_focal_alpha
from transformers import AutoModelForMaskedLM
from ..config.hf_pretrain_config import ViralBERTConfig
# from .modeling_viralbert import ViralBERTModelForPreTraining

# Register the custom model with the AutoModelForMaskedLM class.
# This allows it to be loaded automatically using AutoModelForMaskedLM.from_pretrained(...)
# It maps the "viralbert" model type in the config to our custom model class.
# AutoModelForMaskedLM.register(ViralBERTConfig, ViralBERTModelForPreTraining)

__all__ = [
    'ViralBERTModel',
    'FocalLoss',
    'get_auto_focal_alpha',
] 