import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RumorGNN(nn.Module):
    """
    Graph Neural Network for Rumor Detection.
    Implements a GNN architecture with both GCN and GAT layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 2,
        dropout: float = 0.5,
        use_gat: bool = True,
        num_heads: int = 4
    ):
        """
        Initialize the RumorGNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
            use_gat: Whether to use GAT layers (True) or GCN layers (False)
            num_heads: Number of attention heads for GAT
        """
        super(RumorGNN, self).__init__()
        
        self.dropout = dropout
        self.use_gat = use_gat
        
        # Create layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        if use_gat:
            self.convs.append(GATConv(input_dim, hidden_dims[0], heads=num_heads))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0] * num_heads if use_gat else hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            if use_gat:
                self.convs.append(GATConv(
                    hidden_dims[i] * num_heads,
                    hidden_dims[i + 1],
                    heads=num_heads
                ))
            else:
                self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i + 1] * num_heads if use_gat else hidden_dims[i + 1]))
        
        # Output layer
        if use_gat:
            self.convs.append(GATConv(hidden_dims[-1] * num_heads, num_classes, heads=1))
        else:
            self.convs.append(GCNConv(hidden_dims[-1], num_classes))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Node features
            edge_index: Graph connectivity in COO format
            edge_weight: Optional edge weights
            
        Returns:
            Node-wise class predictions
        """
        # Apply layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return F.log_softmax(x, dim=1)
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        Get attention weights from GAT layers if using GAT.
        
        Returns:
            List of attention weight tensors for each GAT layer
        """
        if not self.use_gat:
            logger.warning("Model is not using GAT layers, no attention weights available")
            return []
        
        attention_weights = []
        for conv in self.convs:
            if isinstance(conv, GATConv):
                attention_weights.append(conv.att)
        
        return attention_weights

    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Extract node embeddings from the last layer before classification."""
        for i, conv in enumerate(self.convs[:-1]):
            if self.use_gat:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last conv layer without activation
        if self.use_gat:
            x = self.convs[-1](x, edge_index)
        else:
            x = self.convs[-1](x, edge_index, edge_weight)
        
        return x 