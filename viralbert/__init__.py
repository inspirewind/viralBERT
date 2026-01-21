# File: viralbert/__init__.py

from .config.hf_pretrain_config import ViralBERTConfig
from .data.hf_tokenizer import ViralBERTTokenizer

# Import pipelines to ensure registration happens on package import
from . import pipelines # noqa: F401

# from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

# AutoConfig.register("viralbert", ViralBERTConfig)
# # AutoModelForMaskedLM.register(ViralBERTConfig, ViralBERTModelForPreTraining)
# AutoTokenizer.register(ViralBERTConfig, slow_tokenizer_class=ViralBERTTokenizer)
