# File: viralbert/evaluation/evaluator.py

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Literal
from tqdm.auto import tqdm
import random
from torch.utils.data import Subset, DataLoader
from torch.cuda.amp import autocast
from ..utils.logging import get_dist_logger, is_main_process

logger = get_dist_logger(__name__)

class ModelEvaluator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def evaluate(self, 
            dataloader, 
            device,
            task: Literal["mlm"] = "mlm",
            eval_subset_size: Optional[int] = None,
            max_eval_samples: Optional[int] = None,
            disable_tqdm: bool = False) -> Dict[str, Any]:
        """评估模型性能
        
        Args:
            dataloader: 数据加载器
            device: 计算设备
            task: 评估任务类型，"mlm"
            eval_subset_size: 按比例随机采样的数量（已弃用，保留为了向后兼容）
            max_eval_samples: 最大评估样本数量（优先级高于eval_subset_size）
            disable_tqdm: 是否禁用进度条
        """
        self.model.eval()
        
        # 如果是DDP模型，临时获取原始模型
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
            
        # 确定评估样本数量
        total_samples = len(dataloader.dataset)
        
        # 使用传入的max_eval_samples参数
        if max_eval_samples is not None:
            eval_size = min(max_eval_samples, total_samples)
        # 其次使用传入的eval_subset_size参数（为向后兼容）
        elif eval_subset_size is not None:
            eval_size = min(eval_subset_size, total_samples)
        # 最后使用通用的max_eval_samples配置
        elif hasattr(self.config, 'max_eval_samples') and self.config.max_eval_samples is not None:
            eval_size = min(self.config.max_eval_samples, total_samples)
        else:
            eval_size = total_samples
        
        # 创建子集数据加载器
        if eval_size < total_samples:
            if is_main_process():
                logger.info(f"eval samples: {eval_size}/{total_samples} (task: {task})")
            indices = random.sample(range(total_samples), eval_size)
            subset = Subset(dataloader.dataset, indices)
            eval_loader = DataLoader(
                subset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn if hasattr(dataloader, 'collate_fn') else None
            )
        else:
            if is_main_process():
                logger.info(f"use all eval samples: {total_samples} (task: {task})")
            eval_loader = dataloader
        
        # 只在主进程显示进度条
        disable_tqdm = disable_tqdm or not is_main_process()
        
        # 根据任务类型选择评估方法
        if task == "mlm":
            return self._evaluate_mlm(eval_loader, model, device, disable_tqdm, total_samples)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def _evaluate_mlm(self, eval_loader, model, device, disable_tqdm, total_samples):
        """MLM任务评估"""
        total_loss = 0
        total_correct = 0
        total_predictions = 0
        total_perplexity = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating MLM", total=len(eval_loader), disable=disable_tqdm):
                # 移动数据到设备
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 使用autocast进行混合精度评估
                with autocast(enabled=self.config.fp16):
                    # 使用原始模型进行前向传播
                    outputs = model(**batch)
                    loss = outputs['loss'] # noqa: F841
                    logits = outputs['logits']
                    labels = batch['labels']
                    
                    # 只考虑被mask的位置
                    valid_mask = labels != -100
                    masked_logits = logits[valid_mask]
                    masked_labels = labels[valid_mask]
                    
                    # 计算loss和perplexity
                    masked_loss = F.cross_entropy(masked_logits, masked_labels, reduction='mean')
                    total_loss += masked_loss.item()
                    perplexity = torch.exp(masked_loss)
                    total_perplexity += perplexity.item()
                    
                    # 计算accuracy
                    predictions = torch.argmax(masked_logits, dim=-1)
                    correct = (predictions == masked_labels).sum().item()
                    total_correct += correct
                    total_predictions += len(masked_labels)
                
                num_batches += 1
        
        # 计算平均指标
        avg_loss = total_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        metrics = {
            'loss': avg_loss,
            'perplexity': avg_perplexity,
            'accuracy': accuracy,
            # 'num_samples': len(eval_loader.dataset),
            # 'num_predictions': total_predictions
        }
        
        # 只在主进程输出评估结果
        if is_main_process():
            logger.info("MLM Evaluation results:")
            logger.info(f"  Total available samples: {total_samples}")
            logger.info(f"  Evaluated on {len(eval_loader.dataset)} samples")
            logger.info(f"  Total masked positions: {total_predictions}")
            logger.info(f"  Loss: {avg_loss:.4f}")
            logger.info(f"  Perplexity: {avg_perplexity:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def _log_prediction_examples(self, input_ids, predictions, labels, n_examples=5):
        """记录一些预测示例"""
        # 只在主进程输出预测示例
        if not is_main_process():
            return
            
        input_text = self.tokenizer.decode(input_ids)
        logger.info("\nPrediction examples:")
        logger.info(f"Input text: {input_text[:100]}...")
        
        for i in range(min(n_examples, len(predictions))):
            pred_token = self.tokenizer.decode([predictions[i]])
            true_token = self.tokenizer.decode([labels[i]])
            logger.info(f"  Example {i+1}:")
            logger.info(f"    True: {true_token}")
            logger.info(f"    Pred: {pred_token}")
    
    def log_metrics(self, metrics: Dict[str, Any], global_step: int, wandb_logger=None):
        # 准备格式化的指标
        formatted_metrics = {f"eval/{k}": v for k, v in metrics.items()}
        
        # 记录到Wandb
        if wandb_logger is not None:
            wandb_logger.log_metrics(
                formatted_metrics,
                step=global_step
            )

    
    @staticmethod
    def get_prediction_text(batch, predictions, labels, tokenizer, n_examples=5):
        """
        获取预测结果的文本形式，用于可视化
        
        Args:
            batch: 输入批次数据
            predictions: 模型预测
            labels: 真实标签
            tokenizer: tokenizer实例
            n_examples: 返回的示例数量
            
        Returns:
            预测结果的文本描述
        """
        text_outputs = []
        input_ids = batch['input_ids']
        valid_mask = labels != -100
        
        for i in range(min(n_examples, len(input_ids))):
            input_text = tokenizer.decode(input_ids[i])
            true_tokens = [tokenizer.decode([label]) if label != -100 else '_' 
                         for label in labels[i]]
            pred_tokens = [tokenizer.decode([pred]) if mask else '_'
                         for pred, mask in zip(predictions[i], valid_mask[i])]
            
            text_outputs.append(
                f"Example {i+1}:\n"
                f"Input:  {input_text}\n"
                f"True:   {' '.join(true_tokens)}\n"
                f"Pred:   {' '.join(pred_tokens)}\n"
            )
        
        return "\n".join(text_outputs)