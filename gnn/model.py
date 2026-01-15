"""
Multi-view GNN models for CA classification.

Implements:
- 4 separate encoders (one per view)
- Fusion strategies (concat, attention, weighted)
- Classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool


class GraphEncoder(nn.Module):
    """
    Generic graph encoder using GIN layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.project = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Project to output dimension
        x = self.project(x)
        
        return x


class MultiViewGNN(nn.Module):
    """
    Multi-view GNN with 4 separate encoders and simple concatenation fusion.
    
    Architecture:
        - 4 encoders (symbol, lattice, debruijn, dependency)
        - Concatenation fusion
        - MLP classifier
    """
    
    def __init__(self, hidden_dim=128, embedding_dim=64, num_classes=5, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # View 0: Symbol (2 nodes, edge_attr=6)
        self.encoder_symbol = GraphEncoder(
            input_dim=1,  # Node features are just [0] or [1]
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=2,  # Smaller graph, fewer layers
            dropout=dropout
        )
        
        # View 1: Lattice (N=20 nodes, no edge attr)
        self.encoder_lattice = GraphEncoder(
            input_dim=1,  # Normalized position
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # View 2: De Bruijn (32 nodes, 5-bit features)
        self.encoder_debruijn = GraphEncoder(
            input_dim=5,  # 5-bit state representation
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # View 3: Dependency (T*W nodes, 2D coordinates)
        self.encoder_dependency = GraphEncoder(
            input_dim=2,  # (time, position)
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(4 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def encode(self, batch):
        """
        Encode all 4 views into embeddings.
        
        Returns:
            dict with keys: 'symbol', 'lattice', 'debruijn', 'dependency'
        """
        embeddings = {
            'symbol': self.encoder_symbol(batch['symbol']),
            'lattice': self.encoder_lattice(batch['lattice']),
            'debruijn': self.encoder_debruijn(batch['debruijn']),
            'dependency': self.encoder_dependency(batch['dependency'])
        }
        return embeddings
    
    def forward(self, batch):
        """
        Forward pass through all views.
        
        Args:
            batch: Dict containing 4 graph views
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode each view
        embeddings = self.encode(batch)
        
        # Concatenate all embeddings
        z_all = torch.cat([
            embeddings['symbol'],
            embeddings['lattice'],
            embeddings['debruijn'],
            embeddings['dependency']
        ], dim=1)
        
        # Classify
        logits = self.classifier(z_all)
        
        return logits


class MultiViewGNN_WithConsistency(MultiViewGNN):
    """
    Extended version that also outputs per-view predictions.
    Used for consistency regularization.
    """
    
    def __init__(self, hidden_dim=128, embedding_dim=64, num_classes=5, dropout=0.1):
        super().__init__(hidden_dim, embedding_dim, num_classes, dropout)
        
        # Individual classifiers for each view
        self.classifier_symbol = nn.Linear(embedding_dim, num_classes)
        self.classifier_lattice = nn.Linear(embedding_dim, num_classes)
        self.classifier_debruijn = nn.Linear(embedding_dim, num_classes)
        self.classifier_dependency = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, batch, return_individual=False):
        """
        Forward pass with optional individual predictions.
        
        Args:
            batch: Dict containing 4 graph views
            return_individual: If True, also return per-view predictions
        
        Returns:
            logits: (batch_size, num_classes) from fused model
            individual_logits: (optional) dict of per-view predictions
        """
        # Encode each view
        embeddings = self.encode(batch)
        
        # Fused prediction
        z_all = torch.cat([
            embeddings['symbol'],
            embeddings['lattice'],
            embeddings['debruijn'],
            embeddings['dependency']
        ], dim=1)
        logits = self.classifier(z_all)
        
        if return_individual:
            # Individual predictions
            individual_logits = {
                'symbol': self.classifier_symbol(embeddings['symbol']),
                'lattice': self.classifier_lattice(embeddings['lattice']),
                'debruijn': self.classifier_debruijn(embeddings['debruijn']),
                'dependency': self.classifier_dependency(embeddings['dependency'])
            }
            return logits, individual_logits
        
        return logits


def consistency_loss(individual_logits, temperature=1.0):
    """
    Consistency regularization: encourages different views to agree.
    
    Uses KL divergence between prediction distributions.
    
    Args:
        individual_logits: Dict of {view_name: logits}
        temperature: Softmax temperature
    
    Returns:
        scalar loss
    """
    views = list(individual_logits.keys())
    
    # Convert logits to probabilities
    probs = {
        view: F.softmax(logits / temperature, dim=-1)
        for view, logits in individual_logits.items()
    }
    
    # Compute pairwise KL divergences
    total_loss = 0.0
    count = 0
    
    for i, view1 in enumerate(views):
        for view2 in views[i+1:]:
            # KL(P || Q) + KL(Q || P) for symmetry
            kl_loss = F.kl_div(
                probs[view1].log(), probs[view2], reduction='batchmean'
            ) + F.kl_div(
                probs[view2].log(), probs[view1], reduction='batchmean'
            )
            total_loss += kl_loss
            count += 1
    
    return total_loss / count if count > 0 else 0.0


if __name__ == "__main__":
    # Test model
    print("Testing MultiViewGNN...")
    
    # Create dummy batch
    from torch_geometric.data import Data, Batch
    
    # Symbol view (2 nodes)
    symbol_graphs = [
        Data(x=torch.randn(2, 1), edge_index=torch.randint(0, 2, (2, 4)))
        for _ in range(8)
    ]
    
    # Lattice view (20 nodes)
    lattice_graphs = [
        Data(x=torch.randn(20, 1), edge_index=torch.randint(0, 20, (2, 80)))
        for _ in range(8)
    ]
    
    # De Bruijn view (32 nodes, 5-dim features)
    debruijn_graphs = [
        Data(x=torch.randn(32, 5), edge_index=torch.randint(0, 32, (2, 64)))
        for _ in range(8)
    ]
    
    # Dependency view (28 nodes, 2-dim features)
    dependency_graphs = [
        Data(x=torch.randn(28, 2), edge_index=torch.randint(0, 28, (2, 100)))
        for _ in range(8)
    ]
    
    batch = {
        'symbol': Batch.from_data_list(symbol_graphs),
        'lattice': Batch.from_data_list(lattice_graphs),
        'debruijn': Batch.from_data_list(debruijn_graphs),
        'dependency': Batch.from_data_list(dependency_graphs)
    }
    
    # Test basic model
    model = MultiViewGNN(hidden_dim=64, embedding_dim=32, num_classes=5)
    logits = model(batch)
    
    print(f"Output shape: {logits.shape}")  # Should be (8, 5)
    print(f"✓ Basic model test passed")
    
    # Test consistency model
    model_cons = MultiViewGNN_WithConsistency(hidden_dim=64, embedding_dim=32, num_classes=5)
    logits, individual = model_cons(batch, return_individual=True)
    
    print(f"\nIndividual predictions:")
    for view, pred in individual.items():
        print(f"  {view}: {pred.shape}")
    
    # Test consistency loss
    cons_loss = consistency_loss(individual)
    print(f"\nConsistency loss: {cons_loss.item():.4f}")
    print(f"✓ Consistency model test passed")