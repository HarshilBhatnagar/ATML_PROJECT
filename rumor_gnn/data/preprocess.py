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

def create_text_features(texts: List[str], max_features: int = 5000) -> torch.Tensor:
    """
    Create TF-IDF features from text data.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features to extract
        
    Returns:
        features: Tensor of TF-IDF features
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(texts)
    return torch.FloatTensor(features.toarray())

def create_temporal_features(timestamps: list) -> torch.Tensor:
    """
    Create temporal features from timestamps.
    
    Args:
        timestamps: List of timestamps
        
    Returns:
        features: Tensor of temporal features
    """
    # Convert timestamps to hours since first post
    timestamps = pd.to_datetime(timestamps)
    base_time = timestamps.min()
    hours_since_base = [(t - base_time).total_seconds() / 3600 for t in timestamps]
    
    # Create cyclical features for time of day
    hours_of_day = [t.hour for t in timestamps]
    sin_hour = np.sin(2 * np.pi * np.array(hours_of_day) / 24)
    cos_hour = np.cos(2 * np.pi * np.array(hours_of_day) / 24)
    
    # Combine features
    features = np.column_stack([hours_since_base, sin_hour, cos_hour])
    return torch.FloatTensor(features)

def create_edge_features(edge_index: torch.Tensor, timestamps: list) -> torch.Tensor:
    """
    Create edge features based on temporal information.
    
    Args:
        edge_index: Tensor of edge indices
        timestamps: List of timestamps
        
    Returns:
        edge_attr: Tensor of edge features
    """
    # Calculate time differences between connected nodes
    time_diffs = []
    for i, j in edge_index.t().tolist():
        t1 = pd.to_datetime(timestamps[i])
        t2 = pd.to_datetime(timestamps[j])
        diff = abs((t1 - t2).total_seconds() / 3600)  # hours
        time_diffs.append(diff)
    
    # Normalize time differences
    time_diffs = np.array(time_diffs)
    time_diffs = (time_diffs - time_diffs.min()) / (time_diffs.max() - time_diffs.min())
    
    return torch.FloatTensor(time_diffs).view(-1, 1)

def create_graph_edges(timestamps: list, temporal_window: float = 6.0) -> torch.Tensor:
    """
    Create graph edges based on temporal proximity using a sliding window approach.
    
    Args:
        timestamps: List of timestamps
        temporal_window: Time window in hours
        
    Returns:
        edge_index: Tensor of edge indices
    """
    logger.info("Creating graph edges using sliding window approach...")
    edges = []
    timestamps = pd.to_datetime(timestamps)
    
    # Sort timestamps and get indices
    sorted_indices = np.argsort(timestamps)
    sorted_timestamps = timestamps[sorted_indices]
    
    # Use a sliding window approach
    for i in range(len(sorted_timestamps)):
        # Find the end of the window
        window_end = i + 1
        while (window_end < len(sorted_timestamps) and 
               (sorted_timestamps[window_end] - sorted_timestamps[i]).total_seconds() / 3600 <= temporal_window):
            # Add edges in both directions
            edges.append([sorted_indices[i], sorted_indices[window_end]])
            edges.append([sorted_indices[window_end], sorted_indices[i]])
            window_end += 1
    
    logger.info(f"Created {len(edges)} edges")
    return torch.LongTensor(edges).t()

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
    max_features: int = 5000,
    temporal_window: float = 6.0
) -> Tuple[Data, Data]:
    """
    Prepare graph data for training and testing.
    
    Args:
        data_path: Path to the dataset
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        max_features: Maximum number of TF-IDF features
        temporal_window: Time window for edge creation
        
    Returns:
        train_data: Training data
        test_data: Test data
    """
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with {len(df)} samples")
    logger.info(f"Dataset columns: {df.columns.tolist()}")
    
    # Validate and normalize labels
    if 'is_rumor' not in df.columns:
        raise ValueError("Dataset must contain 'is_rumor' column")
    
    # Handle NaN values in is_rumor column
    nan_count = df['is_rumor'].isna().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in 'is_rumor' column. Dropping these rows.")
        df = df.dropna(subset=['is_rumor'])
    
    # Convert labels to binary (0 or 1)
    try:
        df['is_rumor'] = df['is_rumor'].astype(int)
    except ValueError as e:
        logger.warning(f"Error converting labels to integers: {e}")
        logger.warning("Attempting to map string labels to integers...")
        # Map string labels to integers
        label_map = {label: i for i, label in enumerate(df['is_rumor'].unique())}
        df['is_rumor'] = df['is_rumor'].map(label_map)
    
    unique_labels = df['is_rumor'].unique()
    logger.info(f"Unique labels before normalization: {unique_labels}")
    
    # Ensure labels are binary (0 or 1)
    if not all(label in [0, 1] for label in unique_labels):
        logger.warning("Labels are not binary, normalizing to 0 and 1")
        # Map labels to 0 and 1
        label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
        df['is_rumor'] = df['is_rumor'].map(label_map)
        logger.info(f"Label mapping: {label_map}")
    
    # Create synthetic timestamps if they don't exist
    if 'timestamp' not in df.columns:
        logger.info("Creating synthetic timestamps based on data order")
        # Create timestamps spaced 1 hour apart starting from a fixed date
        df['timestamp'] = pd.date_range(
            start='2020-01-01',
            periods=len(df),
            freq='h'  # Changed from 'H' to 'h' to fix deprecation warning
        )
    
    # Create features
    text_features = create_text_features(df['text'].tolist(), max_features)
    temporal_features = create_temporal_features(df['timestamp'].tolist())
    
    # Combine features
    x = torch.cat([text_features, temporal_features], dim=1)
    
    # Create graph edges
    edge_index = create_graph_edges(df['timestamp'].tolist(), temporal_window)
    edge_attr = create_edge_features(edge_index, df['timestamp'].tolist())
    
    # Create labels - ensure they are in the correct format
    y = torch.LongTensor(df['is_rumor'].values)
    logger.info(f"Label distribution: {torch.bincount(y)}")
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Create masks
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # Create PyTorch Geometric data objects
    train_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )
    
    test_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask
    )
    
    logger.info(f"Created graph with {x.size(0)} nodes and {edge_index.size(1)} edges")
    logger.info(f"Training samples: {train_mask.sum().item()}, Test samples: {test_mask.sum().item()}")
    logger.info(f"Label distribution in training set: {torch.bincount(y[train_mask])}")
    logger.info(f"Label distribution in test set: {torch.bincount(y[test_mask])}")
    
    return train_data, test_data 