import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.base_gnn import GAT, GCN
from data.preprocess import prepare_data
from utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
            self,
            data_path: str = "data/rumor_dataset.csv",
            model_type: str = "gat",
            hidden_dim: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            lr: float = 0.01,
            weight_decay: float = 5e-4,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
        self.device = device
        self.model_type = model_type
        logger.info(f"Using device: {device}")
        
        # Ensure data_path is a string
        if not isinstance(data_path, str):
            raise ValueError(f"data_path must be a string, got {type(data_path)}")
            
        # Load and prepare data
        logger.info(f"Loading data from: {data_path}")
        self.train_data, self.test_data = prepare_data(
            data_path=data_path,
            test_size=0.2,
            random_state=42,
            max_features=1000,
            temporal_window=24
        )
        
        # Move data to device
        self.train_data = self.train_data.to(device)
        self.test_data = self.test_data.to(device)
        
        # Initialize model with correct dimensions
        in_channels = self.train_data.x.size(1)  # This will be 1000 (max_features)
        out_channels = 2  # Binary classification
        
        if model_type == "gat":
            self.model = GAT(
                in_channels=in_channels,
                hidden_channels=hidden_dim,
                out_channels=out_channels,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                edge_dim=1
            ).to(device)
        elif model_type == "gcn":
            self.model = GCN(
                in_channels=in_channels,
                hidden_channels=hidden_dim,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        logger.info(f"Initialized {model_type.upper()} model with:")
        logger.info(f"- Input channels: {in_channels}")
        logger.info(f"- Hidden dimension: {hidden_dim}")
        logger.info(f"- Number of heads: {num_heads}")
        logger.info(f"- Number of layers: {num_layers}")
        logger.info(f"- Dropout rate: {dropout}")
        logger.info(f"- Edge dimension: 1")
        
        # Setup experiment tracking
        self.setup_experiment_tracking()
        
        # Setup optimizer and scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=10,
            min_lr=1e-5
        )
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking with TensorBoard."""
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"logs/{timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        logger.info(f"Experiment tracking initialized at {self.log_dir}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        
        # Forward pass
        out = self.model(self.train_data.x, self.train_data.edge_index)
        
        # Ensure labels are in the correct range [0, num_classes-1]
        labels = self.train_data.y
        if labels.min() < 0 or labels.max() >= 2:  # For binary classification
            logger.warning("Labels out of range, adjusting...")
            labels = torch.clamp(labels, 0, 1)
        
        # Use cross entropy loss instead of NLL loss
        loss = F.cross_entropy(out, labels)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data) -> Dict[str, float]:
        """Evaluate model on given data."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            metrics = calculate_metrics(pred, data.y)
        return metrics
    
    def train(self, epochs: int = 100, early_stopping_patience: int = 10, min_delta: float = 0.001):
        """Train the model."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass with edge attributes
            out = self.model(
                self.train_data.x,
                self.train_data.edge_index,
                self.train_data.edge_attr
            )
            
            # Calculate loss
            loss = F.cross_entropy(out[self.train_data.train_mask], self.train_data.y[self.train_data.train_mask])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(
                    self.test_data.x,
                    self.test_data.edge_index,
                    self.test_data.edge_attr
                )
                val_loss = F.cross_entropy(val_out[self.test_data.test_mask], self.test_data.y[self.test_data.test_mask])
            
            # Calculate metrics
            train_pred = out.argmax(dim=1)
            val_pred = val_out.argmax(dim=1)
            
            train_metrics = calculate_metrics(
                train_pred[self.train_data.train_mask],
                self.train_data.y[self.train_data.train_mask]
            )
            val_metrics = calculate_metrics(
                val_pred[self.test_data.test_mask],
                self.test_data.y[self.test_data.test_mask]
            )
            
            # Log metrics
            self.log_metrics(epoch, loss.item(), train_metrics, val_metrics)
            
            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}:")
                logger.info(f"  Train Loss: {loss.item():.4f}")
                logger.info(f"  Val Loss: {val_loss.item():.4f}")
                logger.info(f"  Train Metrics: {train_metrics}")
                logger.info(f"  Val Metrics: {val_metrics}")
        
        return train_metrics, val_metrics
    
    def log_metrics(self, epoch: int, train_loss: float, 
                   train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to TensorBoard."""
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', value, epoch)
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
        
        # Log to console
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                   f"Train F1: {train_metrics['f1']:.4f}, "
                   f"Val F1: {val_metrics['f1']:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        checkpoint_path = self.log_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    # Initialize and train
    trainer = Trainer(
        data_path="data/rumor_dataset.csv",
        model_type="gat",
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    )
    
    # Train with custom parameters
    train_metrics, val_metrics = trainer.train(
        epochs=100,
        early_stopping_patience=10,
        min_delta=0.001
    )

    # Log training metrics
    logger.info(f"Train Metrics: {train_metrics}")
    logger.info(f"Val Metrics: {val_metrics}")

if __name__ == '__main__':
    main() 