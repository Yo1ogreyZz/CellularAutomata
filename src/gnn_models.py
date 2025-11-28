"""
GNN Models for ECA Rule Embedding

Implements Graph Autoencoders to learn structural embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from typing import List, Dict, Tuple
import numpy as np

class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder for learning rule embeddings.
    
    Architecture:
        Encoder: GCN layers -> latent embedding
        Decoder: Reconstruct node features from embedding
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [32, 16],
                 latent_dim: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            layers.append(GCNConv(dims[i], dims[i+1]))
        self.encoder_layers = nn.ModuleList(layers)
        
        # Decoder (reconstruct node features)
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
        self.decoder_layers = nn.ModuleList(decoder_layers)
    
    def encode(self, data: Data) -> torch.Tensor:
        """Encode graph to latent representation"""
        x, edge_index = data.x, data.edge_index
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:
                x = F.relu(x)
        
        # Global pooling to get graph-level embedding
        embedding = global_mean_pool(x, data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long))
        
        return embedding
    
    def decode(self, z: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Decode latent to node features"""
        # Expand graph embedding to all nodes
        z_expanded = z.repeat(num_nodes, 1)
        
        x = z_expanded
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(self.decoder_layers) - 1:
                x = F.relu(x)
        
        return x
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode"""
        z = self.encode(data)
        x_recon = self.decode(z, data.x.size(0))
        return z, x_recon


class GNNTrainer:
    """Training pipeline for GNN autoencoder"""
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            z, x_recon = self.model(batch)
            loss = F.mse_loss(x_recon, batch.x)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def extract_embeddings(self, dataloader: DataLoader) -> Dict[int, np.ndarray]:
        """Extract embeddings for all rules"""
        self.model.eval()
        embeddings = {}
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                z = self.model.encode(batch)
                
                # Store by rule number
                for i, rule_num in enumerate(batch.rule_number):
                    embeddings[rule_num.item()] = z[i].cpu().numpy()
        
        return embeddings
    
    def train(self, 
              dataloader: DataLoader, 
              epochs: int = 100,
              verbose: bool = True) -> List[float]:
        """Full training loop"""
        losses = []
        
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)
            losses.append(loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        return losses


def build_gnn_dataset(rule_numbers: List[int], 
                     graph_type: str = 'truth_table') -> List[Data]:
    """
    Build PyG dataset for list of rules.
    
    Args:
        rule_numbers: List of rule numbers
        graph_type: 'truth_table', 'dependency', or 'evolution'
        
    Returns:
        List of PyG Data objects
    """
    from .rule2graph import convert_rule
    from .utils import to_pyg_data
    
    dataset = []
    for rule_num in rule_numbers:
        graphs = convert_rule(rule_num, methods=[graph_type], visualize=False, verbose=False)
        graph_data = graphs[graph_type]
        pyg_data = to_pyg_data(graph_data)
        dataset.append(pyg_data)
    
    return dataset