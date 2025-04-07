import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, Tuple
import umap
import networkx as nx

from models.base_gnn import GAT, GCN
from data.preprocess import prepare_data
from utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str, model_type: str = "gat", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with correct dimensions
    model = GAT(
        in_channels=1000,  # This should match your input features
        hidden_channels=128,
        out_channels=2,
        num_layers=2,
        heads=4,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, test_data, device):
    """Evaluate model on test data."""
    model.eval()
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index)
        pred = out.argmax(dim=1)
        metrics = calculate_metrics(pred, test_data.y)
        
        # Get probabilities for ROC curve
        probs = torch.softmax(out, dim=1)
        
        return metrics, pred.cpu().numpy(), test_data.y.cpu().numpy(), probs.cpu().numpy()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
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

def visualize_embeddings(model: GAT, data, pred, true_labels, save_path: str):
    """Visualize node embeddings using UMAP."""
    # Get node embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()
    
    # Reduce dimensionality with UMAP
    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=true_labels.cpu().numpy(), cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Node Embeddings Visualization')
    plt.savefig(save_path)
    plt.close()

def main():
    # Load data
    logger.info("Loading data...")
    train_data, test_data = prepare_data(
        data_path="data/rumor_dataset.csv",
        test_size=0.2,
        random_state=42,
        max_features=1000,
        temporal_window=24
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = test_data.to(device)
    
    # Load best model
    log_dir = Path("logs/20250407_225905")
    model_path = log_dir / "best_model.pt"
    logger.info(f"Loading model from {model_path}")
    model = load_model(str(model_path), device=device)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics, y_pred, y_true, probs = evaluate_model(model, test_data, device)
    
    # Print metrics
    logger.info("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    logger.info("\nDetailed Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=log_dir / "confusion_matrix.png")
    
    # Save predictions
    np.save(log_dir / "test_predictions.npy", y_pred)
    np.save(log_dir / "test_probabilities.npy", probs)
    
    # Visualize graph
    visualize_graph(
        test_data,
        y_pred,
        y_true,
        log_dir / "graph_visualization.png"
    )
    
    # Visualize embeddings
    visualize_embeddings(
        model,
        test_data,
        y_pred,
        y_true,
        log_dir / "embeddings_visualization.png"
    )
    
    logger.info(f"Results saved to {log_dir}")

if __name__ == '__main__':
    main() 