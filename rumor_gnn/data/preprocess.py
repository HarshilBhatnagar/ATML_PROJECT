import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Tuple, List, Dict
import logging
from transformers import BertTokenizer, BertModel
from datetime import datetime
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the PHEME dataset from CSV file.
    
    Args:
        file_path: Path to the dataset CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    logger.info(f"Loading dataset from {file_path}")
    data = pd.read_csv(file_path)
    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and text preprocessing.
    
    Args:
        data: Raw dataset DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning dataset")
    
    # Drop rows with missing labels
    data = data.dropna(subset=['is_rumor'])
    
    # Filter to only include valid rumor labels (0 or 1)
    data = data[data['is_rumor'].isin([0, 1])]
    
    # Clean text
    data['text'] = data['text'].str.lower().str.replace('[^\\w\\s]', '', regex=True)
    
    # Convert label to integer
    data['is_rumor'] = data['is_rumor'].astype(int)
    
    return data

def create_text_features(texts: List[str], max_features: int = 1000) -> torch.Tensor:
    """
    Convert text to numerical features using TF-IDF.
    
    Args:
        texts: List of text strings
        max_features: Maximum number of features to create
        
    Returns:
        Tensor of text features
    """
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english'
    )
    features = vectorizer.fit_transform(texts)
    
    # Convert to tensor
    return torch.FloatTensor(features.toarray())

def create_graph_data(data: pd.DataFrame, max_features: int = 1000) -> Tuple[Data, Dict]:
    """
    Convert the dataset into a graph structure using PyTorch Geometric.
    
    Args:
        data: Cleaned dataset DataFrame
        max_features: Maximum number of text features
        
    Returns:
        Tuple containing:
        - PyG Data object with graph structure
        - Dictionary of node features
    """
    logger.info("Creating graph structure")
    
    # Create text features first
    x = create_text_features(data['text'].tolist(), max_features)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for idx in range(len(data)):
        G.add_node(idx)
    
    # TODO: Add edges based on reply/retweet relationships
    # For now, we'll create a simple temporal graph where each post
    # is connected to the next k posts (k=5)
    k = 5
    for i in range(len(data) - k):
        for j in range(1, k+1):
            if i + j < len(data):
                G.add_edge(i, i + j)
    
    # Convert to PyG Data object
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    y = torch.tensor(data['is_rumor'].values)
    
    # Create edge weights (all 1.0 for now)
    edge_weight = torch.ones(edge_index.size(1))
    
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y
    )
    
    graph_info = {
        'num_nodes': len(G.nodes),
        'num_edges': len(G.edges),
        'num_features': x.size(1)
    }
    
    logger.info(f"Created graph with {graph_info['num_nodes']} nodes, "
                f"{graph_info['num_edges']} edges, and "
                f"{graph_info['num_features']} features")
    
    return graph_data, graph_info

def prepare_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 1000,
    temporal_window: int = 24
) -> Tuple[Data, Data]:
    """Prepare graph data from the dataset.
    
    Args:
        data_path: Path to the dataset CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        max_features: Maximum number of text features to extract
        temporal_window: Time window in hours for edge creation
    
    Returns:
        Tuple of (train_data, test_data) PyTorch Geometric Data objects
    """
    logger.info("Loading and preprocessing data...")
    
    try:
        # Validate data path
        if not isinstance(data_path, str):
            raise ValueError(f"data_path must be a string, got {type(data_path)}")
            
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        # Load data
        logger.info(f"Reading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_columns = ['text', 'is_rumor']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create text features using TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_features)
        text_features = vectorizer.fit_transform(df['text']).toarray()
        
        # Convert to torch tensors
        x = torch.FloatTensor(text_features)
        y = torch.LongTensor(df['is_rumor'].values)
        
        # Create synthetic timestamps based on row order
        df['synthetic_timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
        
        # Create graph structure using synthetic timestamps
        edge_index = create_graph_edges(df, temporal_window)
        
        # Split data
        train_idx, test_idx = train_test_split(
            range(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Create train mask
        train_mask = torch.zeros(len(df), dtype=torch.bool)
        train_mask[train_idx] = True
        
        # Create test mask
        test_mask = torch.zeros(len(df), dtype=torch.bool)
        test_mask[test_idx] = True
        
        # Create PyTorch Geometric Data objects
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            test_mask=test_mask
        )
        
        # Split into train and test
        train_data = data.clone()
        test_data = data.clone()
        
        logger.info(f"Data prepared: {len(train_idx)} training samples, {len(test_idx)} test samples")
        logger.info(f"Feature dimension: {x.shape[1]}")
        logger.info(f"Number of edges: {edge_index.shape[1]}")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error in prepare_data: {str(e)}")
        raise

def create_graph_edges(df: pd.DataFrame, temporal_window: int) -> torch.Tensor:
    """Create graph edges based on temporal proximity using a sliding window approach.
    
    Args:
        df: DataFrame containing the data
        temporal_window: Time window in hours for edge creation
    
    Returns:
        Edge index tensor of shape [2, num_edges]
    """
    logger.info("Creating graph edges...")
    
    # Create edges using a sliding window approach
    edges = []
    window_size = temporal_window  # Number of nodes to look ahead
    
    # Use tqdm for progress tracking
    for i in tqdm(range(len(df)), desc="Creating edges"):
        # Look ahead only window_size nodes
        for j in range(i + 1, min(i + window_size + 1, len(df))):
            edges.append((i, j))
            edges.append((j, i))  # Undirected graph
    
    # Convert to edge index tensor
    edge_index = torch.LongTensor(edges).t()
    
    logger.info(f"Created {edge_index.shape[1]} edges")
    return edge_index 