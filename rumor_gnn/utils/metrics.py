import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import Dict, Tuple

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Calculate various classification metrics."""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(target_np, pred_np),
        'precision': precision_score(target_np, pred_np, average='weighted'),
        'recall': recall_score(target_np, pred_np, average='weighted'),
        'f1': f1_score(target_np, pred_np, average='weighted')
    }
    
    # Calculate ROC-AUC if binary classification
    if len(np.unique(target_np)) == 2:
        metrics['roc_auc'] = roc_auc_score(target_np, pred_np)
    
    return metrics

def get_confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())

def calculate_class_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Calculate metrics for each class separately."""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    classes = np.unique(target_np)
    
    class_metrics = {}
    for cls in classes:
        class_metrics[f'class_{cls}'] = {
            'precision': precision_score(target_np, pred_np, labels=[cls], average='micro'),
            'recall': recall_score(target_np, pred_np, labels=[cls], average='micro'),
            'f1': f1_score(target_np, pred_np, labels=[cls], average='micro')
        }
    
    return class_metrics

def calculate_attention_metrics(attention_weights: torch.Tensor) -> Dict[str, float]:
    """Calculate metrics for attention weights."""
    attention_np = attention_weights.cpu().numpy()
    
    return {
        'attention_mean': float(np.mean(attention_np)),
        'attention_std': float(np.std(attention_np)),
        'attention_max': float(np.max(attention_np)),
        'attention_min': float(np.min(attention_np))
    } 