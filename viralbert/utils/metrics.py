from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, matthews_corrcoef

def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    """计算分类评估指标
    
    Args:
        preds: 模型预测的类别
        labels: 真实标签
        
    Returns:
        包含各项指标的字典：
        - accuracy: 准确率
        - precision: 精确率（宏平均）
        - recall: 召回率（宏平均）
        - f1: F1分数（宏平均）
        - mcc: Matthews相关系数，在类别不平衡问题中表现更好
    """
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='macro'
    )
    conf_mat = confusion_matrix(labels, preds)
    
    # 计算MCC (Matthews Correlation Coefficient)
    # MCC在类别不平衡的数据集上提供更可靠的评估
    mcc = matthews_corrcoef(labels, preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'confusion_matrix': conf_mat.tolist()
    }
    
    return metrics 