import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, Tuple
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_temporal_distribution(df: pd.DataFrame, save_path: str = "visualizations/temporal_distribution.png"):
    """Plot the temporal distribution of rumors vs non-rumors."""
    plt.figure(figsize=(12, 6))
    
    # Create synthetic timestamps if they don't exist
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(
            start='2020-01-01',
            periods=len(df),
            freq='h'
        )
    
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by hour and rumor status
    hourly_counts = df.groupby([df['timestamp'].dt.hour, 'is_rumor']).size().unstack()
    
    # Plot
    hourly_counts.plot(kind='bar', stacked=True)
    plt.title('Temporal Distribution of Rumors vs Non-Rumors')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Posts')
    plt.legend(['Non-Rumor', 'Rumor'])
    plt.xticks(rotation=0)
    
    # Save plot
    Path("visualizations").mkdir(exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved temporal distribution plot to {save_path}")

def plot_propagation_patterns(df: pd.DataFrame, save_path: str = "visualizations/propagation_patterns.png"):
    """Plot the propagation patterns of rumors."""
    plt.figure(figsize=(12, 6))
    
    # Create synthetic timestamps if they don't exist
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(
            start='2020-01-01',
            periods=len(df),
            freq='h'
        )
    
    # Calculate time differences between consecutive posts
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600  # in hours
    
    # Group by rumor status and calculate statistics
    propagation_stats = df.groupby('is_rumor')['time_diff'].agg(['mean', 'std', 'count'])
    
    # Plot
    plt.bar(['Non-Rumor', 'Rumor'], propagation_stats['mean'], 
            yerr=propagation_stats['std'], capsize=5)
    plt.title('Average Time Between Posts')
    plt.xlabel('Post Type')
    plt.ylabel('Hours')
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved propagation patterns plot to {save_path}")

def plot_user_engagement(df: pd.DataFrame, save_path: str = "visualizations/user_engagement.png"):
    """Plot user engagement patterns for rumors vs non-rumors."""
    plt.figure(figsize=(12, 6))
    
    # Group by user and calculate statistics
    user_stats = df.groupby(['user.handle', 'is_rumor']).size().unstack().fillna(0)
    
    # Calculate average posts per user
    avg_posts = user_stats.mean()
    
    # Plot
    plt.bar(['Non-Rumor', 'Rumor'], avg_posts)
    plt.title('Average Posts per User')
    plt.xlabel('Post Type')
    plt.ylabel('Number of Posts')
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved user engagement plot to {save_path}")

def plot_topic_distribution(df: pd.DataFrame, save_path: str = "visualizations/topic_distribution.png"):
    """Plot the distribution of topics for rumors vs non-rumors."""
    plt.figure(figsize=(12, 6))
    
    # Group by topic and rumor status
    topic_counts = df.groupby(['topic', 'is_rumor']).size().unstack()
    
    # Plot
    topic_counts.plot(kind='bar', stacked=True)
    plt.title('Topic Distribution of Rumors vs Non-Rumors')
    plt.xlabel('Topic')
    plt.ylabel('Number of Posts')
    plt.legend(['Non-Rumor', 'Rumor'])
    plt.xticks(rotation=45)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved topic distribution plot to {save_path}")

def plot_attention_weights(model: torch.nn.Module, data: torch.Tensor, 
                         save_path: str = "visualizations/attention_weights.png"):
    """Plot the distribution of attention weights."""
    plt.figure(figsize=(10, 6))
    
    # Get attention weights
    attention_weights = model.get_attention_weights()
    
    # Convert attention weights to numpy array
    if isinstance(attention_weights, tuple):
        # PyTorch Geometric returns attention weights as a tuple
        attention_weights = attention_weights[1].detach().cpu().numpy()
    else:
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Plot distribution
    sns.histplot(attention_weights.flatten(), bins=50)
    plt.title('Distribution of Attention Weights')
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    
    # Save plot
    Path("visualizations").mkdir(exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved attention weights plot to {save_path}")

def plot_feature_importance(model: torch.nn.Module, feature_names: list,
                          save_path: str = "visualizations/feature_importance.png"):
    """Plot the importance of different features."""
    plt.figure(figsize=(12, 6))
    
    try:
        # Get feature importance from first layer weights
        if hasattr(model, 'conv1') and hasattr(model.conv1, 'lin_src'):
            weights = model.conv1.lin_src.weight.detach().cpu().numpy()
            importance = np.abs(weights).mean(axis=0)
            
            # Plot
            plt.barh(feature_names, importance)
            plt.title('Feature Importance')
            plt.xlabel('Average Absolute Weight')
            
            # Save plot
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved feature importance plot to {save_path}")
        else:
            logger.warning("Model does not have expected GAT layer structure. Skipping feature importance plot.")
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {str(e)}")
        plt.close()

def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor,
                        save_path: str = "visualizations/confusion_matrix.png"):
    """Plot the confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")

def plot_roc_curve(y_true: torch.Tensor, y_score: torch.Tensor,
                  save_path: str = "visualizations/roc_curve.png"):
    """Plot the ROC curve."""
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), y_score.cpu().numpy())
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")

def generate_all_visualizations(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    df: pd.DataFrame,
    feature_names: list
):
    """Generate all visualizations for the research paper."""
    logger.info("Generating visualizations for research paper...")
    
    # Create visualizations directory
    Path("visualizations").mkdir(exist_ok=True)
    
    # Generate temporal analysis plots
    plot_temporal_distribution(df)
    plot_propagation_patterns(df)
    
    # Generate user and topic analysis plots
    plot_user_engagement(df)
    plot_topic_distribution(df)
    
    # Generate model analysis plots
    # Run forward pass to get attention weights
    with torch.no_grad():
        model.eval()
        # Run forward pass on a batch of training data
        model(train_data.x, train_data.edge_index, train_data.edge_attr)
        plot_attention_weights(model, train_data)
        plot_feature_importance(model, feature_names)
    
    # Generate performance plots
    with torch.no_grad():
        train_pred = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        test_pred = model(test_data.x, test_data.edge_index, test_data.edge_attr)
        
        train_pred_labels = train_pred.argmax(dim=1)
        test_pred_labels = test_pred.argmax(dim=1)
        
        train_scores = torch.softmax(train_pred, dim=1)[:, 1]
        test_scores = torch.softmax(test_pred, dim=1)[:, 1]
    
    plot_confusion_matrix(train_data.y[train_data.train_mask], 
                         train_pred_labels[train_data.train_mask])
    plot_roc_curve(train_data.y[train_data.train_mask], 
                  train_scores[train_data.train_mask])
    
    logger.info("All visualizations generated successfully!")

if __name__ == "__main__":
    # Example usage
    from models.base_gnn import GAT
    from data.preprocess import prepare_data
    
    # Load data
    train_data, test_data = prepare_data("data/rumor_dataset.csv")
    df = pd.read_csv("data/rumor_dataset.csv")
    
    # Initialize model
    model = GAT(
        in_channels=train_data.x.size(1),
        hidden_channels=128,
        out_channels=2,
        num_heads=8,
        num_layers=3,
        dropout=0.3
    )
    
    # Generate feature names (example)
    feature_names = [f"feature_{i}" for i in range(train_data.x.size(1))]
    
    # Generate all visualizations
    generate_all_visualizations(model, train_data, test_data, df, feature_names) 