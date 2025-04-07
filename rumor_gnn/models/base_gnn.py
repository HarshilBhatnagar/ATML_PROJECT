import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from typing import List, Optional, Union, Dict, Any

class BaseGNN(torch.nn.Module):
    """Base class for all GNN architectures."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Initialize layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(self._create_conv_layer(input_dim, hidden_dims[0], **kwargs))
        self.batch_norms.append(BatchNorm1d(hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(self._create_conv_layer(hidden_dims[i], hidden_dims[i + 1], **kwargs))
            self.batch_norms.append(BatchNorm1d(hidden_dims[i + 1]))
        
        # Output layer
        self.classifier = Linear(hidden_dims[-1], num_classes)
    
    def _create_conv_layer(self, in_channels: int, out_channels: int, **kwargs) -> torch.nn.Module:
        """Create a graph convolution layer. To be implemented by child classes."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the network."""
        # Input layer
        x = self.convs[0](x, edge_index, edge_weight)
        x = self.batch_norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv, batch_norm in zip(self.convs[1:], self.batch_norms[1:]):
            x = conv(x, edge_index, edge_weight)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                      edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract node embeddings from the last layer before classification."""
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last conv layer without activation
        x = self.convs[-1](x, edge_index, edge_weight)
        x = self.batch_norms[-1](x)
        return x 

class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # Final layer: hidden_channels -> out_channels
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index, edge_weight)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        self.convs.append(
            GATConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
                concat=True
            )
        )
        
        # Hidden layers: hidden_channels*heads -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # Final layer: hidden_channels*heads -> out_channels
        self.convs.append(
            GATConv(
                in_channels=hidden_channels * heads,
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                concat=False
            )
        )
        
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_weight)
        return x 