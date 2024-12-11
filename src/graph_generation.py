# src/graph_generation.py
import torch
from torch_geometric.data import Data

def create_graph_node_features(data):
    """Convert omics data into node features."""
    return torch.tensor(data.values, dtype=torch.float)

def create_graph_edge_index(data):
    """Create edges based on correlations or predefined relationships."""
    n_samples = data.shape[0]
    edge_index = torch.combinations(torch.arange(n_samples), r=2).t()
    return edge_index

def build_graph(data):
    """Build a PyTorch Geometric graph."""
    node_features = create_graph_node_features(data)
    edge_index = create_graph_edge_index(data)
    return Data(x=node_features, edge_index=edge_index)
