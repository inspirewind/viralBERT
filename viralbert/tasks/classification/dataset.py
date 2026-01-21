import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import random

logger = logging.getLogger(__name__)

class SequenceClassificationDataset(Dataset):
    """序列分类数据集"""
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int,
        label2id: Dict[str, int],
        truncation_strategy: str = "end",  # 'end' or 'middle'
        use_augmentation: bool = False,
        mask_prob: float = 0.5,
        mask_ratio: float = 0.15,
    ):
        """
        Args:
            data_path: 数据文件路径（支持CSV或JSON）
            tokenizer: 分词器实例
            max_length: 最大序列长度
            label2id: 标签到ID的映射字典
            truncation_strategy: 截断策略，'end'从末尾截断，'middle'从中间截断
            use_augmentation: 是否在训练时使用数据增强
            mask_prob: 每个样本应用增强的概率
            mask_ratio: 增强时mask的token比例
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.truncation_strategy = truncation_strategy
        self.use_augmentation = use_augmentation
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        
        # 加载数据
        self._data = self._load_data(data_path)
        
        # 初始化类别分布缓存
        self._label_distribution_cache = None
        
        logger.info(f"Loaded {len(self._data)} data")
        # logger.info(f"tokenizer: {self.tokenizer}")
        logger.info(f"max_length: {self.max_length}")
        logger.info(f"label2id: {self.label2id}")
        logger.info(f"truncation_strategy: {self.truncation_strategy}")
    
    def _load_data(self, data_path: Union[str, Path]) -> List[dict]:
        """加载数据文件"""
        data_path = Path(data_path)
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            data = []
            for _, row in df.iterrows():
                data.append({
                    'sequence': row['sequence'],
                    'label': row['label']
                })
        elif data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        return data
    
    def _analyze_label_distribution(self):
        """分析并记录标签分布，使用缓存避免重复计算"""
        # 如果已经计算过，则直接返回缓存结果
        if self._label_distribution_cache is not None:
            return self._label_distribution_cache
            
        label_counts = {}
        for item in self._data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 将字典转换为按照类别ID排序的列表
        ordered_counts = []
        for label, idx in sorted(self.label2id.items(), key=lambda x: x[1]):
            ordered_counts.append(label_counts.get(label, 0))
        
        logger.info(f"Label distribution: {dict(zip(self.label2id.keys(), ordered_counts))}")
        
        # 保存到缓存
        self._label_distribution_cache = ordered_counts
        
        return ordered_counts
    
    def _truncate_sequence(self, sequence: str) -> str:
        """根据策略截断序列"""
        if self.truncation_strategy == "end":
            return sequence[:self.max_length]
        elif self.truncation_strategy == "middle":
            if len(sequence) <= self.max_length:
                return sequence
            
            # 保留序列的头部和尾部，从中间截断
            half_length = self.max_length // 2
            return sequence[:half_length] + sequence[-half_length:]
        else:
            raise ValueError(f"Unknown truncation strategy: {self.truncation_strategy}")

    def _apply_augmentation(self, input_ids: List[int]) -> List[int]:
        """Applies sequence augmentation by masking random tokens."""
        input_ids_tensor = torch.tensor(input_ids)

        # We don't want to mask special tokens
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        )
        non_special_tokens_indices = torch.where(special_tokens_mask == 0)[0]

        if len(non_special_tokens_indices) == 0:
            return input_ids # Nothing to mask

        # Determine number of tokens to mask
        num_to_mask = int(len(non_special_tokens_indices) * self.mask_ratio)
        if num_to_mask == 0:
            return input_ids

        # Randomly select indices to mask from the non-special tokens
        mask_indices = np.random.choice(non_special_tokens_indices.numpy(), num_to_mask, replace=False)

        # Apply mask
        for i in mask_indices:
            input_ids_tensor[i] = self.tokenizer.mask_token_id

        return input_ids_tensor.tolist()
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        item = self._data[idx]
        sequence = self._truncate_sequence(item['sequence'])
        label = self.label2id[item['label']]
        
        # 使用 __call__ 方法 (tokenizer(...)) 而不是 encode
        # 这会返回一个包含 'input_ids' 和 'attention_mask' 的字典
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None  # Important: return lists for now
        )

        if self.use_augmentation and random.random() < self.mask_prob:
            encoding['input_ids'] = self._apply_augmentation(encoding['input_ids'])

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label)
        }

class SequenceClassificationDataModule:
    """序列分类数据模块"""
    def __init__(
        self,
        train_path: Union[str, Path],
        val_path: Union[str, Path],
        test_path: Optional[Union[str, Path]] = None,
        tokenizer = None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        truncation_strategy: str = "end",
        use_augmentation: bool = False,
        mask_prob: float = 0.5,
        mask_ratio: float = 0.15,
    ):
        """
        Args:
            train_path: 训练集路径
            val_path: 验证集路径
            test_path: 测试集路径（可选）
            tokenizer: 分词器实例
            max_length: 最大序列长度
            batch_size: 批次大小
            num_workers: 数据加载的工作进程数
            truncation_strategy: 序列截断策略
            use_augmentation: 是否在训练时使用数据增强
            mask_prob: 每个样本应用增强的概率
            mask_ratio: 增强时mask的token比例
        """
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path) if test_path else None
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.truncation_strategy = truncation_strategy
        self.use_augmentation = use_augmentation
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

        # 构建标签映射
        self.label2id = self._build_label_mapping()
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)
        
        logger.info(f"Found {self.num_classes} classes: {list(self.label2id.keys())}")
    
    def _build_label_mapping(self) -> Dict[str, int]:
        """构建标签到ID的映射"""
        # 收集所有数据文件中的唯一标签
        unique_labels = set()
        paths = [self.train_path, self.val_path]
        if self.test_path:
            paths.append(self.test_path)
            
        for path in paths:
            if path.suffix == '.csv':
                df = pd.read_csv(path)
                unique_labels.update(df['label'].unique())
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                    unique_labels.update(item['label'] for item in data)
        
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    def setup(self):
        """准备数据集"""
        self.train_dataset = SequenceClassificationDataset(
            self.train_path,
            self.tokenizer,
            self.max_length,
            self.label2id,
            self.truncation_strategy,
            use_augmentation=self.use_augmentation,
            mask_prob=self.mask_prob,
            mask_ratio=self.mask_ratio,
        )
        
        self.val_dataset = SequenceClassificationDataset(
            self.val_path,
            self.tokenizer,
            self.max_length,
            self.label2id,
            self.truncation_strategy
        )
        
        if self.test_path:
            self.test_dataset = SequenceClassificationDataset(
                self.test_path,
                self.tokenizer,
                self.max_length,
                self.label2id,
                self.truncation_strategy
            )
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """计算类别权重用于处理类别不平衡"""
        if not hasattr(self, 'train_dataset'):
            return None
            
        label_counts = np.zeros(self.num_classes)
        for item in self.train_dataset._data:
            label_idx = self.label2id[item['label']]
            label_counts[label_idx] += 1
            
        # 使用逆频率作为权重
        weights = 1.0 / label_counts
        weights = weights / weights.sum() * self.num_classes
        
        return torch.FloatTensor(weights)
    
    def get_class_counts(self) -> List[int]:
        """返回每个类别的样本数量列表，带缓存机制避免重复计算"""
        # 如果已经计算过，直接返回缓存结果
        if hasattr(self, '_class_counts_cache') and self._class_counts_cache is not None:
            return self._class_counts_cache
            
        # 确保数据已加载
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            self.setup()
        
        # 使用数据集的方法计算或手动计算
        if hasattr(self.train_dataset, '_analyze_label_distribution'):
            class_counts = self.train_dataset._analyze_label_distribution()
        else:
            # 手动计算类别计数
            class_counts = [0] * len(self.label2id)
            for item in self.train_dataset._data:
                label_idx = self.label2id[item['label']]
                class_counts[label_idx] += 1
            
            logger.info(f"Class counts: {class_counts}")
        
        # 保存到缓存
        self._class_counts_cache = class_counts
        
        return class_counts 