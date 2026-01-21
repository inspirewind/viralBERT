# File: viralbert/data/data_module.py

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from pathlib import Path
import random
import time
import numpy as np
from typing import Optional
from tqdm import tqdm

from .dataset import ViralMLMDataset
from .hf_tokenizer import ViralBERTTokenizer
from pyfaidx import Fasta
from ..utils.logging import get_dist_logger, is_main_process

logger = get_dist_logger(__name__)

class ViralMLMDataModule:
    """数据集管理类"""
    def __init__(
        self,
        data_dir: str,
        fasta_file: str,
        tokenizer: ViralBERTTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        train_val_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        stride: int = 256,
        filter_n: bool = False,  # 添加filter_n参数
        p_codon: float = 0.5,  # 密码子掩码的概率
        masking_strategy: str = "simple",
        reverse_complement_prob: float = 0.5,
    ):
        """
        初始化数据模块
        
        Args:
            data_dir: 数据目录
            fasta_file: FASTA文件名
            tokenizer: tokenizer实例
            max_length: 最大序列长度
            mlm_probability: mask的概率
            train_val_split: 验证集比例
            batch_size: 批次大小
            num_workers: 数据加载的进程数
            seed: 随机种子
            stride: 滑动窗口的步长
            filter_n: 是否过滤掉包含N的序列片段
            p_codon: 密码子掩码的概率
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.stride = stride
        self.filter_n = filter_n
        self.p_codon = p_codon
        self.masking_strategy = masking_strategy
        self.reverse_complement_prob = reverse_complement_prob
        
        # 分布式训练
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 初始化数据集
        self.train_dataset: Optional[ViralMLMDataset] = None
        self.val_dataset: Optional[ViralMLMDataset] = None
        
        # 初始化FASTA文件句柄
        self.fasta_path = self.data_dir / fasta_file
        self.fasta = None  # 延迟初始化
        
        # 添加序列名称映射
        self.seq_name_to_id = {}
        self.id_to_seq_name = {}
        
    def _prepare_intervals(self) -> np.ndarray:
        """准备所有可能的区间信息，使用序列ID代替序列名称，返回numpy数组"""
        if self.fasta is None:
            self.fasta = Fasta(str(self.fasta_path))
            
        intervals_list = []
        # 首先建立序列名称映射
        for idx, seq_name in tqdm(enumerate(self.fasta.keys()), desc="Building sequence name mapping"):
            self.seq_name_to_id[seq_name] = idx
            self.id_to_seq_name[idx] = seq_name
            
            seq_len = len(self.fasta[seq_name])
            # 使用序列ID而不是序列名称
            for start in range(0, seq_len, self.stride):
                intervals_list.append((idx, start))
        
        # 转换为numpy数组
        intervals = np.array(intervals_list, dtype=np.int64)
                
        if is_main_process():
            logger.info(f"Total sequences: {len(self.seq_name_to_id)}")
            logger.info(f"Prepared intervals memory: {intervals.nbytes / (1024 * 1024):.2f} MB")
        return intervals
    
    def _create_dataset(self, intervals: np.ndarray) -> ViralMLMDataset:
        """创建数据集的辅助方法"""
        return ViralMLMDataset(
            intervals=intervals,
            fasta_file=self.fasta,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mlm_probability=self.mlm_probability,
            id_to_seq_name=self.id_to_seq_name,
            filter_n=self.filter_n,  # 传递filter_n参数
            p_codon=self.p_codon,  # 传递密码子掩码概率参数
            masking_strategy=self.masking_strategy,
            reverse_complement_prob=self.reverse_complement_prob,
        )
    
    def setup(self):
        """准备训练和验证数据集"""
        # 生成所有区间
        intervals = self._prepare_intervals()
        
        # 分割训练和验证集
        val_size = int(len(intervals) * self.train_val_split)
        indices = np.arange(len(intervals))
        if is_main_process():
            logger.info("shuffling intervals...")
            start_time = time.time()
        np.random.shuffle(indices)
        if is_main_process():
            end_time = time.time()
            logger.info(f"shuffling intervals time: {end_time - start_time} seconds")
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # 创建训练和验证数据集
        self.train_dataset = self._create_dataset(intervals[train_indices])
        self.val_dataset = self._create_dataset(intervals[val_indices])
        
        # 创建分布式采样器
        self._setup_samplers()
        
        # 只在主进程输出日志
        if is_main_process():
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
    
    def _setup_samplers(self):
        """设置分布式采样器"""
        if not dist.is_initialized():
            self.train_sampler = None
            self.val_sampler = None
            return
            
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed
        )
        
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.world_size if dist.is_initialized() else self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size // self.world_size if dist.is_initialized() else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )