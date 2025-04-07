import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import umap
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tensorboard_data(log_dir: str):
    """Load data from TensorBoard logs."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get scalar data
    data = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        data[tag] = [event.value for event in events]
    
    return data

def plot_training_metrics(data: dict, save_path: Path):
    """Plot training metrics."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(data['Loss/train'], label='Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Plot F1 score
    axes[0, 1].plot(data['Train/f1'], label='Train F1')
    axes[0, 1].plot(data['Val/f1'], label='Val F1')
    axes[0, 1].set_title('F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision
    axes[1, 0].plot(data['Train/precision'], label='Train Precision')
    axes[1, 0].plot(data['Val/precision'], label='Val Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot recall
    axes[1, 1].plot(data['Train/recall'], label='Train Recall')
    axes[1, 1].plot(data['Val/recall'], label='Val Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_graph(data, pred, true_labels, save_path: str):
    """Visualize graph structure with predictions."""
    # Convert to networkx
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    
    # Add edges
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    # Set node colors based on predictions
    colors = ['red' if p != t else 'green' for p, t in zip(pred, true_labels)]
    
    # Plot
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, node_size=20, alpha=0.6)
    plt.title('Graph Visualization\nGreen: Correct Predictions, Red: Incorrect')
    plt.savefig(save_path)
    plt.close()

def main():
    # Load TensorBoard data
    log_dir = Path("logs/20250407_225905")
    logger.info(f"Loading TensorBoard data from {log_dir}")
    data = load_tensorboard_data(str(log_dir))
    
    # Plot metrics
    plot_training_metrics(data, log_dir / "training_metrics.png")
    
    # Save metrics to CSV
    df = pd.DataFrame(data)
    df.to_csv(log_dir / "training_metrics.csv", index=False)
    
    logger.info(f"Visualizations saved to {log_dir}")

if __name__ == '__main__':
    main() 