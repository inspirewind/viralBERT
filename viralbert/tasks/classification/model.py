# File: viralbert/tasks/classification/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from viralbert.models.model import ViralBERTModel
from viralbert.config.hf_classification_config import ViralBERTClassificationConfig

logger = logging.getLogger(__name__)


class ViralBERTForSequenceClassification(PreTrainedModel):
    """ViralBERT model for sequence classification, compatible with Hugging Face."""
    config_class = ViralBERTClassificationConfig
    base_model_prefix = "bert"

    def __init__(self, config: ViralBERTClassificationConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        # Instantiate the ViralBERT backbone without the MLM head
        self.bert = ViralBERTModel(config, add_mlm_head=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initializes the weights of the given module."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self):
        """Returns the input embeddings layer for PEFT compatibility."""
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Sets the input embeddings layer for PEFT compatibility."""
        self.bert.embeddings.word_embeddings = value

    def apply_seq_augment(self, input_ids, attention_mask):
        """Applies sequence augmentation by randomly masking input tokens."""
        if not self.training or not self.config.use_seq_augment or torch.rand(1).item() > self.config.seq_mask_prob:
            return input_ids
        
        input_ids = input_ids.clone()
        
        batch_size, seq_len = input_ids.size()
        valid_mask = attention_mask == 1
        
        for i in range(batch_size):
            valid_positions = torch.where(valid_mask[i])[0]
            if len(valid_positions) > 1:
                valid_positions = valid_positions[1:]  # Skip CLS token
                
                num_mask = int(len(valid_positions) * self.config.seq_mask_ratio)
                if num_mask > 0:
                    mask_indices = torch.randperm(len(valid_positions))[:num_mask]
                    mask_positions = valid_positions[mask_indices]
                    
                    input_ids[i, mask_positions] = self.config.n_token_id
        
        return input_ids

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        """Forward pass for sequence classification."""
        if self.training and self.config.use_seq_augment:
            input_ids = self.apply_seq_augment(input_ids, attention_mask)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        pooled_output = outputs['last_hidden_state'][:, 0]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.loss_type == 'ce':
                # Prepare class weights if provided
                class_weights = None
                if self.config.class_weights is not None:
                    class_weights = torch.tensor(
                        self.config.class_weights, 
                        device=logits.device, 
                        dtype=logits.dtype
                    )
                
                # Use standard CrossEntropyLoss with built-in support for weights and label smoothing
                loss_fct = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=self.config.label_smoothing_factor
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            elif self.config.loss_type == 'weighted_bce':
                pos_weight = self.config.pos_weight
                if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
                    pos_weight = torch.tensor(pos_weight, device=logits.device)

                if self.num_labels == 1:
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fct(logits.view(-1), labels.float().view(-1))
                else:
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    one_hot_labels = F.one_hot(labels, num_classes=self.num_labels).float()
                    loss = loss_fct(logits, one_hot_labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.get('hidden_states'),
            attentions=outputs.get('attentions'),
        )

    def freeze_bert_layers(self, num_layers: Optional[int] = None):
        """Freezes parameters of the BERT backbone."""
        if num_layers is None:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("Froze all BERT layers")
        else:
            for layer in self.bert.encoder.layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"Froze bottom {num_layers} BERT layers")

    def unfreeze_bert_layers(self):
        """Unfreezes all parameters of the BERT backbone."""
        for param in self.bert.parameters():
            param.requires_grad = True
        logger.info("Unfroze all BERT layers")
        
    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        for param in self.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(f"Trainable params: {trainable_params/1e6}M || All params: {all_params/1e6}M || Trainable%: {100 * trainable_params / all_params:.4f}")
