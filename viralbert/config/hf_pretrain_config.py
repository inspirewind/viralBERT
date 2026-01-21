# File: viralbert/config/hf_pretrain_config.py

from transformers import PretrainedConfig
from typing import List, Optional, Literal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ViralBERTConfig(PretrainedConfig):
    """
    ViralBERT Pre-training Configuration, compatible with Hugging Face Transformers.
    This class only contains parameters relevant for the pre-training task (e.g., MLM).
    """
    model_type = "viralbert"

    def __init__(
        self,
        # Tokenizer
        # tokenizer_type is deprecated as we only use ViralBERTTokenizer
        
        # Model Architecture
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 12,
        attention_head_size: Optional[int] = 64,
        intermediate_size: int = 2048,
        feed_forward_activation: Literal["swiglu", "gelu"] = "swiglu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-12,
        norm_layer_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        use_qk_norm: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        position_embedding_type: Literal["rope", "absolute"] = "rope",
        use_xpos: bool = False,
        rope_interpolation_factor: float = 1.0,

        # Sliding Window Attention
        sliding_window_size: Optional[int] = None,
        global_attn_every_n_layers: int = 0,

        # General Training Parameters
        batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        adjust_learning_rate_for_accumulation: bool = True,
        scale_loss_for_accumulation: bool = True,
        num_train_epochs: int = 2,

        # Training & Optimization
        optimizer_type: Literal["adamw", "muon_adamw"] = "muon_adamw",
        # --- AdamW Specific ---
        adamw_lr: float = 1e-5,
        adamw_weight_decay: float = 0.01,
        # --- MuonAdamW Specific ---
        muon_lr: float = 0.0015,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.05,
        adamw_lr_for_muon_others: float = 0.000267,
        adamw_betas_for_muon_others: List[float] = None,
        adamw_eps_for_muon_others: float = 1e-10,
        adamw_weight_decay_for_muon_others: float = 0.05,
        # --- Gradient Clipping ---
        global_max_grad_norm: float = 1.0,
        use_per_group_clipping: bool = False,
        muon_max_grad_norm: float = 1.0,
        adamw_max_grad_norm: float = 0.5,

        # LR Scheduler
        lr_scheduler_type: str = "cosine",
        warmup_steps: int = 4000,
        warmup_steps_ratio: Optional[float] = None,
        min_lr_ratio: float = 0.1,
        high_lr_steps_ratio: float = 0.8,
        high_lr_multiplier: float = 1.0,
        
        # Data Processing
        seq_length: int = 512,
        stride: int = 256,
        mlm_probability: float = 0.15,
        masking_strategy: Literal["simple", "structural"] = "structural",
        p_codon: float = 0.5,
        filter_n: bool = False,
        reverse_complement_prob: float = 0.5,
        data_dir: str = "data/raw",
        fasta_file: str = "bac_500_virus_all.fasta",
        
        # Logging, Saving, and Evaluation
        logging_steps: int = 1000,
        save_steps: int = 100000,
        max_eval_samples: int = 2048,

        # Reproducibility and Performance
        seed: int = 42,
        num_workers: int = 4,
        fp16: bool = True,
        use_compile: bool = True,
        compile_mode: str = "default",
        compile_fullgraph: bool = False,
        compile_backend: str = "inductor",
        
        # Checkpoint Resuming
        run_name: str = "pretrain-run",
        resume_from_checkpoint: Optional[str] = None,
        resume_mode: Optional[Literal["recovery", "epoch_extend", "adaptation_extend"]] = None,

        # WandB Integration
        wandb_enabled: bool = True,
        wandb_project: str = "viralbert",
        wandb_name: Optional[str] = None,
        wandb_group: str = "pretrain_group",
        wandb_tags: List[str] = None,
        wandb_notes: Optional[str] = None,
        wandb_watch_model: bool = False,
        wandb_watch_freq: Optional[int] = None,
        
        # Sweep Early Stopping
        max_steps_for_sweep: Optional[int] = None,
        sweep_early_stopping_patience_steps: Optional[int] = 1000,
        sweep_early_stopping_threshold: Optional[float] = 50.0,
        
        **kwargs
    ):
        super().__init__(**kwargs)

        # Assign all parameters to self
        # self.tokenizer_type = tokenizer_type # Deprecated
        # self.special_tokens = special_tokens if special_tokens is not None else ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.intermediate_size = intermediate_size
        self.feed_forward_activation = feed_forward_activation
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.norm_layer_type = norm_layer_type
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.use_qk_norm = use_qk_norm
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.position_embedding_type = position_embedding_type
        self.use_xpos = use_xpos
        self.rope_interpolation_factor = rope_interpolation_factor

        # Add vocab_size as an attribute but don't require it in __init__
        # It will be set dynamically later.
        if not hasattr(self, "vocab_size"):
            self.vocab_size = 14 # Set a reasonable default instead of None

        self.sliding_window_size = sliding_window_size
        self.global_attn_every_n_layers = global_attn_every_n_layers

        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.adjust_learning_rate_for_accumulation = adjust_learning_rate_for_accumulation
        self.scale_loss_for_accumulation = scale_loss_for_accumulation
        self.num_train_epochs = num_train_epochs

        self.optimizer_type = optimizer_type
        self.adamw_lr = adamw_lr
        self.adamw_weight_decay = adamw_weight_decay
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.muon_weight_decay = muon_weight_decay
        self.adamw_lr_for_muon_others = adamw_lr_for_muon_others
        self.adamw_betas_for_muon_others = adamw_betas_for_muon_others if adamw_betas_for_muon_others is not None else [0.90, 0.95]
        self.adamw_eps_for_muon_others = adamw_eps_for_muon_others
        self.adamw_weight_decay_for_muon_others = adamw_weight_decay_for_muon_others
        self.global_max_grad_norm = global_max_grad_norm
        self.use_per_group_clipping = use_per_group_clipping
        self.muon_max_grad_norm = muon_max_grad_norm
        self.adamw_max_grad_norm = adamw_max_grad_norm

        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.warmup_steps_ratio = warmup_steps_ratio
        self.min_lr_ratio = min_lr_ratio
        self.high_lr_steps_ratio = high_lr_steps_ratio
        self.high_lr_multiplier = high_lr_multiplier
        
        self.seq_length = seq_length
        self.stride = stride
        self.mlm_probability = mlm_probability
        self.masking_strategy = masking_strategy
        self.p_codon = p_codon
        self.filter_n = filter_n
        self.reverse_complement_prob = reverse_complement_prob
        self.data_dir = data_dir
        self.fasta_file = fasta_file
        
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_eval_samples = max_eval_samples

        self.seed = seed
        self.num_workers = num_workers
        self.fp16 = fp16
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.compile_backend = compile_backend
        
        self.run_name = run_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.resume_mode = resume_mode

        self.wandb_enabled = wandb_enabled
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_group = wandb_group
        self.wandb_tags = wandb_tags if wandb_tags is not None else []
        self.wandb_notes = wandb_notes
        self.wandb_watch_model = wandb_watch_model
        self.wandb_watch_freq = wandb_watch_freq
        
        self.max_steps_for_sweep = max_steps_for_sweep
        self.sweep_early_stopping_patience_steps = sweep_early_stopping_patience_steps
        self.sweep_early_stopping_threshold = sweep_early_stopping_threshold
        
        # Post-initialization logic
        self.post_init()

    def post_init(self):
        """Post-initialization checks and modifications."""
        # Validate Attention Head Size
        if self.attention_head_size is not None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"Configuration Error: `hidden_size` ({self.hidden_size}) must be divisible by "
                    f"`num_attention_heads` ({self.num_attention_heads})."
                )
            
            calculated_head_size = self.hidden_size // self.num_attention_heads
            if self.attention_head_size != calculated_head_size:
                raise ValueError(
                    f"Configuration Conflict: The provided `attention_head_size` ({self.attention_head_size}) "
                    f"does not match the value calculated from `hidden_size` / `num_attention_heads` "
                    f"({self.hidden_size} / {self.num_attention_heads} = {calculated_head_size}).\n"
                    "Please ensure the relationship `hidden_size = num_attention_heads * attention_head_size` holds true."
                )

        # Checkpoint-related post-processing
        if self.resume_from_checkpoint and self.resume_mode not in ["recovery", "epoch_extend", "adaptation_extend"]:
            raise ValueError(
                f"`resume_from_checkpoint` is set to '{self.resume_from_checkpoint}', "
                "but `resume_mode` is not specified or is invalid. "
                "Please set `resume_mode` to one of: 'recovery', 'epoch_extend', or 'adaptation_extend'."
            )
        
        # If wandb_name is not set, use run_name
        if self.wandb_enabled and self.wandb_name is None:
            self.wandb_name = self.run_name
        
        # If both are not set, use a timestamped format
        if self.wandb_enabled and self.wandb_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_size = "small" if self.hidden_size <= 256 else "base" if self.hidden_size <= 768 else "large"
            self.wandb_name = f"viralbert_{model_size}_{timestamp}"
            
            if self.wandb_group is None:
                self.wandb_group = f"{model_size}_model"
            
            if not self.wandb_tags:
                self.wandb_tags = [
                    model_size,
                    # self.tokenizer_type, # Deprecated
                    f"layers_{self.num_hidden_layers}",
                    f"heads_{self.num_attention_heads}"
                ]

        # Adjust learning rate for gradient accumulation
        if self.adjust_learning_rate_for_accumulation and self.gradient_accumulation_steps > 1:
            effective_lr = self.adamw_lr * self.gradient_accumulation_steps
            logger.info(
                f"AdamW learning rate adjusted for gradient accumulation. "
                f"Original LR: {self.adamw_lr}, Steps: {self.gradient_accumulation_steps}, New LR: {effective_lr}"
            )
            self.adamw_lr = effective_lr

    def get_tokenizer_config(self) -> dict:
        """Gets tokenizer-specific config."""
        return {
            # 'special_tokens': self.special_tokens # Deprecated
        }

    def get_scheduler_args(self) -> dict:
        """Gets scheduler-specific arguments."""
        # Note: This is a helper that might not be needed when using Trainer,
        # but is kept for consistency with the original config logic.
        if self.lr_scheduler_type == "warmup_cosine":
            return {
                'lr_scheduler_type': self.lr_scheduler_type,
                'warmup_steps': self.warmup_steps,
                'warmup_steps_ratio': self.warmup_steps_ratio,
                'min_lr_ratio': self.min_lr_ratio,
                'high_lr_steps_ratio': self.high_lr_steps_ratio,
                'high_lr_multiplier': self.high_lr_multiplier
            }
        else:
            raise ValueError(f"Unknown or unsupported scheduler type: {self.lr_scheduler_type}")