# File: viralbert/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging
from rotary_embedding_torch import RotaryEmbedding
from transformers import PreTrainedModel
from ..config.hf_pretrain_config import ViralBERTConfig

logger = logging.getLogger('ViralBERT')


# remove custom RMSNorm implementation, use official implementation，may need higher version of PyTorch
# remove sageattention for this version

class ViralBERTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(config.seq_length, config.hidden_size)
        
    def forward(self, input_ids=None, position_ids=None):
        embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == "absolute":
            if position_ids is None:
                seq_length = input_ids.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0) # Shape: (1, seq_length)
            
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        return embeddings

class ViralBERTAttention(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {config.hidden_size} can not be divided by num_attention_heads {config.num_attention_heads}"
            )
            
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # query, key, value are all bias-free
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Output part
        # dense is bias-free
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if getattr(config, 'norm_layer_type', 'rmsnorm') == 'layernorm':
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 如果使用RoPE，初始化旋转编码
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.attention_head_size,
                use_xpos=config.use_xpos,
                interpolate_factor=config.rope_interpolation_factor
            )
        
        # Sliding window attention parameters
        sliding_window_size = getattr(config, 'sliding_window_size', None)
        global_attn_every_n_layers = getattr(config, 'global_attn_every_n_layers', 0)

        # Determine attention type at initialization
        use_sliding_window_config = sliding_window_size is not None and sliding_window_size > 0
        is_global_attn_layer = global_attn_every_n_layers > 0 and layer_idx % global_attn_every_n_layers == 0
        self.is_sliding_attention = use_sliding_window_config and not is_global_attn_layer
        
        self.sliding_window_size = sliding_window_size # Store for dynamic mask creation
        
        if self.is_sliding_attention:
            logger.info(f"Layer {layer_idx}: Using sliding window attention with size: {sliding_window_size}")
        else:
            logger.info(f"Layer {layer_idx}: Using global attention.")

        # QK-Norm for stability
        self.use_qk_norm = getattr(config, 'use_qk_norm', False)
        if self.use_qk_norm:
            if getattr(config, 'norm_layer_type', 'rmsnorm') == 'layernorm':
                self.q_norm = nn.LayerNorm(self.attention_head_size, eps=config.layer_norm_eps)
            else:
                self.q_norm = nn.RMSNorm(self.attention_head_size, eps=config.layer_norm_eps)
            if getattr(config, 'norm_layer_type', 'rmsnorm') == 'layernorm':
                self.k_norm = nn.LayerNorm(self.attention_head_size, eps=config.layer_norm_eps)
            else:
                self.k_norm = nn.RMSNorm(self.attention_head_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm: Apply LayerNorm BEFORE attention computation
        normed_hidden_states = self.LayerNorm(hidden_states) # hidden_states: [batch_size, seq_length, hidden_size]
        batch_size, seq_length, _ = normed_hidden_states.size()
        
        # calculate Q, K, V and reshape the dimensions (using normalized input)
        query_layer = self.query(normed_hidden_states).view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        key_layer = self.key(normed_hidden_states).view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        value_layer = self.value(normed_hidden_states).view(
            batch_size, seq_length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        
        # 如果使用RoPE，应用旋转编码
        if self.position_embedding_type == "rope":
            if self.rotary_emb.use_xpos:
                query_layer, key_layer = self.rotary_emb.rotate_queries_and_keys(query_layer, key_layer)
            else:
                query_layer = self.rotary_emb.rotate_queries_or_keys(query_layer)
                key_layer = self.rotary_emb.rotate_queries_or_keys(key_layer)
        
        # Apply QK-Norm if enabled
        if self.use_qk_norm:
            query_layer = self.q_norm(query_layer)
            key_layer = self.k_norm(key_layer)
        
        # Convert attention_mask from (batch, seq_len) of 1s and 0s
        # to a boolean mask for SDPA, where True means 'attend'.
        # The shape is unsqueezed to be broadcastable to (batch, num_heads, seq_len, seq_len).
        if attention_mask is not None:
            # (N, S) -> (N, 1, 1, S)
            attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)

        if self.is_sliding_attention:
            # Dynamically create the sliding window mask based on the input's sequence length
            window_half = self.sliding_window_size // 2
            indices = torch.arange(seq_length, device=hidden_states.device)
            sliding_mask = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0)) <= window_half
            sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, seq_len, seq_len)

            if attention_mask is not None:
                # Combine with original padding mask
                attention_mask = attention_mask & sliding_mask
            else:
                attention_mask = sliding_mask

        # 使用PyTorch原生实现
        attn_output = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        # 重塑输出
        context_layer = attn_output.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.all_head_size)
        
        # Self-output part: projection + dropout + residual (Pre-Norm style)
        attention_output = self.dense(context_layer)
        attention_output = self.dropout(attention_output)
        # Pre-Norm: Add residual connection WITHOUT LayerNorm (LayerNorm was applied at the beginning)
        attention_output = attention_output + hidden_states
        
        attention_probs = None
        max_logit = None

        # 仅在训练模式或显式要求输出 attention 时计算昂贵的矩阵
        # 这避免了在长序列推理时因计算 QK^T 导致的 OOM
        if self.training:
            # 在 `no_grad` 上下文中计算，以避免影响性能
            with torch.no_grad():
                # 计算 attention scores (QK^T)
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                
                # 计算 MaxLogit
                max_logit = attention_scores.max().detach()
                
                # 计算 attention probabilities 用于返回
                attention_probs = F.softmax(attention_scores / math.sqrt(self.attention_head_size), dim=-1)
        
        # 如果未计算，返回默认值以保持返回值签名一致
        if max_logit is None:
            max_logit = torch.tensor(0.0, device=hidden_states.device)
            
        return (attention_output, attention_probs, max_logit)

class ViralBERTFeedForward(nn.Module):
    """
    A self-contained feed-forward network block that includes the SwiGLU activation,
    projection, dropout, and the final residual connection with layer normalization.
    """
    def __init__(self, config):
        super().__init__()
        self.activation = getattr(config, 'feed_forward_activation', 'swiglu')
        
        # Split the gate and up projections into two separate linear layers.
        # This can reduce peak memory usage when dealing with long sequences,
        # at the cost of a potential small performance hit compared to a fused layer.
        if self.activation == "swiglu":
            self.w_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # Down-projection layer
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # Final LayerNorm and Dropout
        if getattr(config, 'norm_layer_type', 'rmsnorm') == 'layernorm':
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: Apply LayerNorm BEFORE feed-forward computation
        residual = hidden_states
        normed_hidden_states = self.LayerNorm(hidden_states)

        if self.activation == "swiglu":
            # Projections for gate and up tensors
            gate = self.w_gate(normed_hidden_states)
            up = self.w_up(normed_hidden_states)
            # SwiGLU activation
            intermediate_states = F.silu(gate) * up
        else: # gelu
            up = self.w_up(normed_hidden_states)
            intermediate_states = F.gelu(up)

        # Down-projection and dropout
        hidden_states = self.down_proj(intermediate_states)
        hidden_states = self.dropout(hidden_states)
        
        # Pre-Norm: Add residual connection WITHOUT LayerNorm (LayerNorm was applied at the beginning)
        hidden_states = hidden_states + residual
        
        return hidden_states

class ViralBERTLayer(nn.Module):
    """BERT编码器层"""
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.attention = ViralBERTAttention(config, layer_idx=layer_idx)
        self.feed_forward = ViralBERTFeedForward(config)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention block
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        
        # Feed-forward block
        layer_output = self.feed_forward(attention_output)
        
        # The returned tuple now contains the final output of the layer,
        # and potentially the attention weights from the attention block.
        return (layer_output,) + attention_outputs[1:]

class ViralBERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ViralBERTLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        all_hidden_states = [] if self.config.output_hidden_states else None
        all_attentions = [] if self.config.output_attentions else None
        all_max_logits = []
        
        for layer in self.layers:
            if self.config.output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            
            if self.config.output_attentions:
                all_attentions.append(layer_outputs[1])
            all_max_logits.append(layer_outputs[2])
                
        if self.config.output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        return (
            hidden_states, 
            tuple(all_hidden_states) if all_hidden_states is not None else None, 
            tuple(all_attentions) if all_attentions is not None else None, 
            tuple(all_max_logits)
        )

class ViralBERTModel(PreTrainedModel):
    """ViralBERT full model"""
    config_class = ViralBERTConfig

    def __init__(self, config, add_mlm_head: bool = True):
        super().__init__(config)
        
        self.embeddings = ViralBERTEmbeddings(config)
        self.encoder = ViralBERTEncoder(config)

        self.mlm_head = None
        if add_mlm_head:
            self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
            # Tie weights if configured，default is False because our small vocab size
            if config.tie_word_embeddings:
                self.mlm_head.weight = self.embeddings.word_embeddings.weight
        
        # Call post_init() to properly initialize weights of layers defined above
        # try this to fix NaN in grad monitor
        self.post_init()

        self.print_trainable_parameters()
    
    def _init_weights(self, module):
        """Initializes the weights of the given module."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize the weights using truncated normal distribution
            std = self.config.initializer_range
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)
    
    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"ViralBERT backbone total parameters: {total_params:.2f}M")
        logger.info(f"ViralBERT backbone trainable parameters: {trainable_params:.2f}M")
        logger.info(f"ViralBERT backbone trainable parameters ratio: {100 * trainable_params / total_params:.2f}%")
                
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None, **kwargs):        
        batch_size, seq_length = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

        # get embeddings
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        
        # through encoder, pass the original attention_mask (1s for real tokens, 0s for padding)
        encoder_outputs = self.encoder(embedding_output, attention_mask=attention_mask)
        sequence_output = encoder_outputs[0]
        
        # get mlm scores
        mlm_scores = None
        if self.mlm_head is not None:
            mlm_scores = self.mlm_head(sequence_output)
        
        # calculate loss
        loss = None
        mlm_labels = kwargs.get('mlm_labels', labels)
        if mlm_labels is not None:
            if mlm_scores is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    mlm_scores.view(-1, self.config.vocab_size),
                    mlm_labels.view(-1)
                )
        
        # build return result
        output = {
            "loss": loss,
            "logits": mlm_scores,
            # add below keys to support downstream tasks
            "last_hidden_state": sequence_output,
            "max_logits": encoder_outputs[3], # encoder_outputs[3] is all_max_logits
        }

        # Conditionally add hidden states and attentions to the output
        if self.config.output_hidden_states:
            output["hidden_states"] = encoder_outputs[1]
        
        if self.config.output_attentions:
            output["attentions"] = encoder_outputs[2]
            
        return output