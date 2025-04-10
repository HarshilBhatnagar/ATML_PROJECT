import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict
import torch
from torch_geometric.data import Data

def plot_temporal_window_impact(window_sizes: List[int], accuracies: List[float], 
                              save_path: str = 'temporal_window.png'):
    """
    Plot the impact of different temporal window sizes on model performance.
    
    Args:
        window_sizes: List of window sizes tested
        accuracies: List of corresponding accuracies
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, accuracies, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Temporal Window Size (hours)')
    plt.ylabel('Accuracy')
    plt.title('Impact of Temporal Window Size on Model Performance')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_attention_weights(data: Data, attention_weights: torch.Tensor, 
                         save_path: str = 'attention_weights.png'):
    """
    Visualize attention weights in the GAT layer.
    
    Args:
        data: PyTorch Geometric Data object
        attention_weights: Tensor of attention weights
        save_path: Path to save the plot
    """
    # Convert attention weights to numpy array
    weights = attention_weights.detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(weights, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Attention Weights in GAT Layer')
    plt.xlabel('Target Nodes')
    plt.ylabel('Source Nodes')
    plt.savefig(save_path)
    plt.close()

def plot_training_metrics(train_losses: List[float], val_losses: List[float],
                         train_accs: List[float], val_accs: List[float],
                         save_path: str = 'training_metrics.png'):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str = 'confusion_matrix.png'):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close() 