import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from typing import List, Optional, Union, Dict, Any

from .base_gnn import BaseGNN

class GCN(BaseGNN):
    """Graph Convolutional Network implementation."""
    
    def _create_conv_layer(self, in_channels: int, out_channels: int, **kwargs) -> torch.nn.Module:
        return GCNConv(in_channels, out_channels, **kwargs)

class GAT(BaseGNN):
    """Graph Attention Network implementation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.5,
        heads: int = 4,
        **kwargs
    ):
        self.heads = heads
        super().__init__(input_dim, hidden_dims, num_classes, dropout, **kwargs)
    
    def _create_conv_layer(self, in_channels: int, out_channels: int, **kwargs) -> torch.nn.Module:
        return GATConv(
            in_channels,
            out_channels // self.heads,
            heads=self.heads,
            dropout=self.dropout,
            **kwargs
        )

class GraphSAGE(BaseGNN):
    """GraphSAGE implementation."""
    
    def _create_conv_layer(self, in_channels: int, out_channels: int, **kwargs) -> torch.nn.Module:
        return SAGEConv(in_channels, out_channels, **kwargs)

class GIN(BaseGNN):
    """Graph Isomorphism Network implementation."""
    
    def _create_conv_layer(self, in_channels: int, out_channels: int, **kwargs) -> torch.nn.Module:
        # Create MLP for GIN
        mlp = Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU()
        )
        return GINConv(mlp, **kwargs)

def get_gnn_model(
    model_name: str,
    input_dim: int,
    hidden_dims: List[int],
    num_classes: int,
    dropout: float = 0.5,
    **kwargs
) -> BaseGNN:
    """Factory function to create GNN models."""
    model_classes = {
        'gcn': GCN,
        'gat': GAT,
        'sage': GraphSAGE,
        'gin': GIN
    }
    
    if model_name.lower() not in model_classes:
        raise ValueError(f"Unknown model type: {model_name}. Available models: {list(model_classes.keys())}")
    
    return model_classes[model_name.lower()](
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs
    ) 