# File: viralbert/tasks/binning/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from copy import deepcopy
from typing import Optional

from transformers import PreTrainedModel

from viralbert.models.model import ViralBERTModel
from viralbert.config.hf_binning_config import ViralBERTBinningConfig

logger = logging.getLogger(__name__)


@torch.no_grad()
def _concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors.
    This is a critical function for MoCo to ensure the queue is consistent
    across all devices in a distributed training environment.
    Warning: torch.distributed.all_gather has no gradient.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
        
    tensors_gather = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ViralBERTForContrastiveBinning(PreTrainedModel):
    """
    A MoCo-style contrastive learning model built on top of ViralBERT, compatible with Hugging Face.
    """
    config_class = ViralBERTBinningConfig
    base_model_prefix = "bert"

    def __init__(self, config: ViralBERTBinningConfig):
        """
        Args:
            config: An instance of ViralBERTBinningConfig.
        """
        super().__init__(config)
        self.config = config

        # MoCo parameters
        self.K = config.moco_k
        self.m = config.moco_m
        self.T = config.moco_t

        # Create the encoders
        # Query encoder's backbone. `from_pretrained` will load weights into this.
        self.bert = ViralBERTModel(config, add_mlm_head=False)
        
        # Query encoder's projector
        self.projector_q = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.moco_dim)
        )

        # Key encoder is a deepcopy of the query encoder
        self.bert_k = deepcopy(self.bert)
        self.projector_k = deepcopy(self.projector_q)
        
        # Apply weight initialization to projector heads
        self.projector_q.apply(self._init_weights)
        self.projector_k.apply(self._init_weights)
        
        # CRITICAL: Freeze ALL parameters of the key encoder (MoCo requirement)
        # Key encoder is updated ONLY via momentum, never via backprop
        for param in self.bert_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False
        
        logger.info("Key encoder (bert_k and projector_k) parameters frozen.")

        self._configure_trainable_layers(config.unfreeze_last_n_layers)

        # Create the queue
        self.register_buffer("queue", torch.randn(config.moco_dim, config.moco_k))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        logger.info(f"MoCo Model Initialized: dim={config.moco_dim}, K={config.moco_k}, m={config.moco_m}, T={config.moco_t}")
        
        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        """
        Standard post_init for weight initialization.
        NOTE: This is called during __init__, BEFORE pretrained weights are loaded.
        """
        super().post_init()
    
    def sync_key_encoder_weights(self):
        """
        PUBLIC METHOD: Synchronize key encoder weights with query encoder.
        MUST be called AFTER loading pretrained weights via from_pretrained().
        """
        logger.info("Synchronizing key encoder weights with query encoder...")
        
        # Debug: Check if bert weights are non-zero before sync
        sample_param_q = next(self.bert.parameters())
        sample_param_k_before = next(self.bert_k.parameters()).clone()
        logger.debug(f"Query encoder sample weight norm: {sample_param_q.norm().item():.4f}")
        logger.debug(f"Key encoder sample weight norm (before sync): {sample_param_k_before.norm().item():.4f}")
        
        with torch.no_grad():
            # Synchronize BERT backbone
            for param_q, param_k in zip(self.bert.parameters(), self.bert_k.parameters()):
                param_k.data.copy_(param_q.data)
            
            # Synchronize projector
            for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
                param_k.data.copy_(param_q.data)
        
        # Debug: Verify sync worked
        sample_param_k_after = next(self.bert_k.parameters())
        logger.debug(f"Key encoder sample weight norm (after sync): {sample_param_k_after.norm().item():.4f}")
        logger.info(f"Key encoder weight synchronization completed. Weights match: {torch.allclose(sample_param_q, sample_param_k_after)}")

    def _init_weights(self, module):
        """Initializes the weights of the projector head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _configure_trainable_layers(self, unfreeze_last_n_layers: int):
        """
        Freezes or unfreezes BERT layers in the query encoder.
        """
        # Freeze all parameters of the BERT part of the query encoder by default
        for param in self.bert.parameters():
            param.requires_grad = False

        total_layers = self.bert.config.num_hidden_layers

        if unfreeze_last_n_layers == -1: # Unfreeze all
            for param in self.bert.parameters():
                param.requires_grad = True
            logger.info("Unfroze all BERT layers in the query encoder.")
            return

        if unfreeze_last_n_layers == 0: # Freeze all
            logger.info("Froze all BERT layers. Only the query projector will be trained.")
            return
        
        if unfreeze_last_n_layers > 0:
            unfreeze_last_n_layers = min(total_layers, unfreeze_last_n_layers)
            layers_to_unfreeze = self.bert.encoder.layers[-unfreeze_last_n_layers:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

            # Also unfreeze the pooler if it exists
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
            
            logger.info(f"Unfroze the last {unfreeze_last_n_layers} BERT layers (and Pooler) in the query encoder.")

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.bert.parameters(), self.bert_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # Gather keys from all GPUs for a consistent queue in distributed training
        keys = _concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Simplest strategy: if the batch overflows, truncate it to fit the remaining space.
        # This prevents wrap-around complexity and is robust to variable batch sizes.
        remaining_space = self.K - ptr
        
        if batch_size > remaining_space:
            # Truncate the incoming keys to fit the queue's tail.
            # The rest of the batch is discarded for this update step.
            keys_to_enqueue = keys[:remaining_space]
            logger.warning(
                f"Batch (size={batch_size}) exceeds remaining queue space ({remaining_space}) at pointer {ptr}. "
                f"Truncating batch to {remaining_space} to prevent queue wrap-around. "
                "This is expected for the last batch of an epoch."
            )
        else:
            keys_to_enqueue = keys
            
        effective_batch_size = keys_to_enqueue.shape[0]

        if effective_batch_size == 0:
            return # Nothing to enqueue.

        # Replace the keys at ptr
        self.queue[:, ptr : ptr + effective_batch_size] = keys_to_enqueue.T
        # Move pointer
        ptr = (ptr + effective_batch_size) % self.K
        self.queue_ptr[0] = ptr

    def get_embedding(self, input_ids, attention_mask, **kwargs):
        """
        Helper to get the [CLS] token embedding from the QUERY model's backbone.
        This is the representation for downstream tasks.
        """
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embedding = bert_outputs['last_hidden_state'][:, 0]
        return embedding

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, **kwargs):
        """
        Forward pass for MoCo.
        Args:
            input_ids_1, attention_mask_1: The query sequence batch (im_q)
            input_ids_2, attention_mask_2: The key sequence batch (im_k)
        """
        # Debug: Log forward pass (only occasionally to avoid spam)
        if hasattr(self, '_forward_call_count'):
            self._forward_call_count += 1
        else:
            self._forward_call_count = 1
            
        if self._forward_call_count <= 3 or self._forward_call_count % 100 == 0:
            logger.debug(f"Forward pass #{self._forward_call_count}: batch_size={input_ids_1.shape[0]}")
        # 1. Compute query features
        bert_output_q = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        cls_q = bert_output_q['last_hidden_state'][:, 0]
        q = self.projector_q(cls_q) # queries: NxC
        q = F.normalize(q, dim=1)
        
        # 2. Compute key features with no gradient
        with torch.no_grad():
            self._momentum_update_key_encoder()
            bert_output_k = self.bert_k(input_ids=input_ids_2, attention_mask=attention_mask_2)
            cls_k = bert_output_k['last_hidden_state'][:, 0]
            k = self.projector_k(cls_k) # keys: NxC
            k = F.normalize(k, dim=1)

        # 3. Compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # 4. Define labels and compute loss
        # The positive key is at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        loss = F.cross_entropy(logits, labels)
        
        # Debug: Log loss value occasionally
        if self._forward_call_count <= 5 or self._forward_call_count % 100 == 0:
            logger.debug(f"Forward pass #{self._forward_call_count}: loss={loss.item():.4f}, "
                        f"l_pos_mean={l_pos.mean().item():.4f}, l_neg_mean={l_neg.mean().item():.4f}")

        # 5. Dequeue and enqueue keys
        self._dequeue_and_enqueue(k)

        return {"loss": loss}