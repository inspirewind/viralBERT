# File: viralbert/utils/scheduler.py

import math
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional

class CustomWarmupCosineScheduler(LambdaLR):
    """自定义学习率调度器，支持预热和余弦退火"""
    def __init__(
        self,
        optimizer,
        num_training_steps: int,
        num_warmup_steps: int,
        high_lr_steps: int = 0,
        high_lr_multiplier: float = 1.0,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.high_lr_steps = high_lr_steps
        self.high_lr_multiplier = high_lr_multiplier
        self.min_lr_ratio = min_lr_ratio
        
        def lr_lambda(current_step: int):
            # 预热阶段
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            
            # 高学习率阶段
            if self.high_lr_steps > 0 and current_step < (self.num_warmup_steps + self.high_lr_steps):
                return self.high_lr_multiplier
            
            # 余弦退火阶段
            decay_start_step = self.num_warmup_steps + self.high_lr_steps
            total_decay_steps = self.num_training_steps - decay_start_step
            
            # 确保衰减阶段至少有1步长
            if total_decay_steps <= 0:
                return self.min_lr_ratio
            
            progress = float(current_step - decay_start_step) / float(max(1, total_decay_steps))
            return max(
                self.min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
            
        super().__init__(optimizer, lr_lambda, last_epoch)

def create_scheduler(
    optimizer,
    num_training_steps: int,
    lr_scheduler_type: str = "warmup_cosine",
    warmup_steps: int = 2000,
    warmup_steps_ratio: Optional[float] = None,
    high_lr_steps: int = 0,
    high_lr_steps_ratio: Optional[float] = None,
    high_lr_multiplier: float = 1.0,
    min_lr_ratio: float = 0.1,
    **kwargs
) -> LambdaLR:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        num_training_steps: 总训练步数
        lr_scheduler_type: 调度器类型 ["warmup_cosine"]
        warmup_steps: 预热步数 (如果设置了warmup_steps_ratio，则此参数被忽略)
        warmup_steps_ratio: 预热步数占总步数的比例
        high_lr_steps: 高学习率阶段步数 (如果设置了high_lr_steps_ratio，则此参数被忽略)
        high_lr_steps_ratio: 高学习率阶段占总步数的比例
        high_lr_multiplier: 高学习率阶段倍数
        min_lr_ratio: 最小学习率比例
    """
    # If a ratio is set for warmup steps, calculate it dynamically
    # This takes precedence over the fixed warmup_steps value
    if warmup_steps_ratio is not None and warmup_steps_ratio > 0:
        warmup_steps = int(num_training_steps * warmup_steps_ratio)

    # 如果设置了ratio，则动态计算steps
    if high_lr_steps_ratio is not None:
        high_lr_steps = int(num_training_steps * high_lr_steps_ratio)
        
    if lr_scheduler_type in ["warmup_cosine", "cosine"]:
        return CustomWarmupCosineScheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=warmup_steps,
            high_lr_steps=high_lr_steps,
            high_lr_multiplier=high_lr_multiplier,
            min_lr_ratio=min_lr_ratio
        )
    else:
        raise ValueError(f"Unknown or unsupported scheduler type: {lr_scheduler_type}")