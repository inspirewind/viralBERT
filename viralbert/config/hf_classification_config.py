# File: viralbert/config/hf_classification_config.py

from typing import List, Optional, Literal, Union
from .hf_pretrain_config import ViralBERTConfig

class ViralBERTClassificationConfig(ViralBERTConfig):
    """
    Configuration class for ViralBERT for Sequence Classification.
    Inherits from ViralBERTConfig and adds parameters specific to the classification task.
    """
    model_type = "viralbert_for_sequence_classification"
    base_model_prefix = "bert"

    def __init__(
        self,
        # Classification-specific parameters
        num_labels: int = 2,
        classifier_dropout_prob: float = 0.1,
        
        # Loss function configurations
        loss_type: Literal["ce", "weighted_bce"] = "ce",
        class_weights: Optional[List[float]] = None, # For ce
        pos_weight: Optional[Union[float, List[float]]] = None, # For weighted_bce
        
        # Label Smoothing (now controlled directly by label_smoothing_factor)
        label_smoothing_factor: float = 0.0,
        
        # Sequence Augmentation
        use_seq_augment: bool = False,
        seq_mask_ratio: float = 0.15,
        seq_mask_prob: float = 0.5,
        n_token_id: int = 9, # Default for 'N'
        
        # Training strategy
        freeze_bert_layers: Optional[int] = None,

        **kwargs
    ):
        """
        Initializes the configuration.
        
        Args:
            num_labels (int, optional): Number of classes for the classifier. Defaults to 2.
            classifier_dropout_prob (float, optional): Dropout probability for the classifier head. Defaults to 0.1.
            loss_type (str, optional): The type of loss function to use ('ce' or 'weighted_bce'). Defaults to "ce".
            class_weights (List[float], optional): A list of weights for each class for the Cross Entropy loss. Defaults to None.
            pos_weight (Union[float, List[float]], optional): Positive weight for BCE loss. Defaults to None.
            label_smoothing_factor (float, optional): The label smoothing factor. If > 0, label smoothing is applied to the Cross Entropy loss. Defaults to 0.0.
            use_seq_augment (bool, optional): Whether to use sequence augmentation during training. Defaults to False.
            seq_mask_ratio (float, optional): Masking ratio for sequence augmentation. Defaults to 0.15.
            seq_mask_prob (float, optional): Probability of applying sequence augmentation. Defaults to 0.5.
            n_token_id (int, optional): The token ID for the 'N' token, used in sequence augmentation.
            freeze_bert_layers (int, optional): Number of bottom BERT layers to freeze during fine-tuning. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent ViralBERTConfig.
        """
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.classifier_dropout_prob = classifier_dropout_prob
        
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.pos_weight = pos_weight
        
        self.label_smoothing_factor = label_smoothing_factor
        
        self.use_seq_augment = use_seq_augment
        self.seq_mask_ratio = seq_mask_ratio
        self.seq_mask_prob = seq_mask_prob
        self.n_token_id = n_token_id
        
        self.freeze_bert_layers = freeze_bert_layers
        
        # Add id2label and label2id mappings for HF compatibility
        # These will be populated later by the data module or training script
        if not hasattr(self, 'id2label'):
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        if not hasattr(self, 'label2id'):
            self.label2id = {v: k for k, v in self.id2label.items()}
