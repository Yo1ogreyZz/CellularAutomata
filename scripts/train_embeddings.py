"""
Training Script for Three Graph Representations
==============================================

Trains GNN autoencoders on:
1. Truth-Table graphs (static)
2. Dependency graphs (static)  
3. Evolution graphs (dynamic with IC control)

Outputs: embeddings for all 88 rules in each representation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import os

sys.path.append('..')
from src.rule2graph import ECARule, TruthTableGraph, DependencyGraph, EvolutionGraph
from src.utils import WOLFRAM_CLASSES, to_pyg_data


class GraphAutoencoder(nn.Module):
    """Graph Autoencoder for learning rule embeddings"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: GCN layers
        layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            layers.append(GCNConv(dims[i], dims[i+1]))
        self.encoder_layers = nn.ModuleList(layers)
        
        # Decoder: MLP to reconstruct node features
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
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        embedding = global_mean_pool(x, batch)
        
        return embedding
    
    def decode(self, z: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Decode latent to node features"""
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


def build_truth_table_dataset(rule_numbers: List[int]) -> List[Data]:
    """Build dataset for truth-table graphs"""
    dataset = []
    
    for rule_num in rule_numbers:
        rule = ECARule(rule_num)
        graph_builder = TruthTableGraph(rule)
        graph_data = graph_builder.build()
        
        pyg_data = to_pyg_data(graph_data)
        pyg_data.rule_number = torch.tensor([rule_num])
        dataset.append(pyg_data)
    
    return dataset


def build_dependency_dataset(rule_numbers: List[int]) -> List[Data]:
    """Build dataset for dependency graphs"""
    dataset = []
    
    for rule_num in rule_numbers:
        rule = ECARule(rule_num)
        graph_builder = DependencyGraph(rule)
        graph_data = graph_builder.build()
        
        pyg_data = to_pyg_data(graph_data)
        pyg_data.rule_number = torch.tensor([rule_num])
        dataset.append(pyg_data)
    
    return dataset


def build_evolution_dataset(rule_numbers: List[int],
                            width: int = 51,
                            steps: int = 20,
                            n_ic_samples: int = 5) -> Tuple[List[Data], Dict]:
    """
    Build dataset for evolution graphs with multiple initial conditions
    
    Args:
        rule_numbers: List of rule numbers
        width: Number of cells
        steps: Number of time steps
        n_ic_samples: Number of different initial conditions per rule
    
    Returns:
        dataset: List of PyG Data objects
        ic_mapping: Dict mapping rule_number -> list of sample indices
    """
    dataset = []
    ic_mapping = {}
    
    initial_densities = np.linspace(0.2, 0.8, n_ic_samples)
    
    for rule_num in rule_numbers:
        rule = ECARule(rule_num)
        sample_indices = []
        
        for ic_idx, density in enumerate(initial_densities):
            graph_builder = EvolutionGraph(
                rule,
                width=width,
                steps=steps,
                initial_density=float(density),
                seed=42 + ic_idx
            )
            graph_data = graph_builder.build()
            
            pyg_data = to_pyg_data(graph_data)
            pyg_data.rule_number = torch.tensor([rule_num])
            pyg_data.ic_index = torch.tensor([ic_idx])
            
            sample_indices.append(len(dataset))
            dataset.append(pyg_data)
        
        ic_mapping[rule_num] = sample_indices
    
    return dataset, ic_mapping


def train_model(model: nn.Module,
                dataset: List[Data],
                epochs: int = 200,
                batch_size: int = 16,
                learning_rate: float = 0.001,
                device: str = 'cpu',
                verbose: bool = True) -> List[float]:
    """
    Train the autoencoder model
    
    Returns:
        losses: List of training losses per epoch
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            z, x_recon = model(batch)
            loss = F.mse_loss(x_recon, batch.x)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def extract_embeddings(model: nn.Module,
                      dataset: List[Data],
                      device: str = 'cpu') -> Dict[int, np.ndarray]:
    """
    Extract embeddings for all samples
    
    Returns:
        embeddings: Dict mapping rule_number -> embedding array
    """
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    embeddings = {}
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            z = model.encode(data)
            
            rule_num = data.rule_number.item()
            embeddings[rule_num] = z.cpu().numpy().flatten()
    
    return embeddings


def aggregate_evolution_embeddings(raw_embeddings: Dict[int, np.ndarray],
                                  ic_mapping: Dict[int, List[int]],
                                  dataset: List[Data]) -> Dict[int, np.ndarray]:
    """
    Aggregate embeddings from multiple ICs into single embedding per rule
    
    Uses mean pooling across all IC samples
    """
    aggregated = {}
    
    for rule_num, sample_indices in ic_mapping.items():
        ic_embeddings = []
        
        for idx in sample_indices:
            rule_num_sample = dataset[idx].rule_number.item()
            ic_embeddings.append(raw_embeddings[rule_num_sample])
        
        aggregated[rule_num] = np.mean(ic_embeddings, axis=0)
    
    return aggregated


def train_truth_table_representation(rule_numbers: List[int],
                                     output_dir: Path,
                                     device: str = 'cpu') -> Dict:
    """Train Truth-Table graph representation"""
    print("\n" + "="*60)
    print("Training Truth-Table Representation")
    print("="*60)
    
    # Build dataset
    print("Building truth-table graphs...")
    dataset = build_truth_table_dataset(rule_numbers)
    print(f"  Created {len(dataset)} graphs")
    
    # Model configuration
    config = {
        'input_dim': 4,
        'hidden_dims': [32, 16],
        'latent_dim': 8,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 0.001
    }
    
    # Create model
    model = GraphAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    )
    
    # Train
    print("Training model...")
    losses = train_model(
        model, dataset,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        device=device,
        verbose=True
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, dataset, device=device)
    
    # Save results
    output_file = output_dir / 'truth_table_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'config': config,
            'losses': losses,
            'model_state': model.state_dict()
        }, f)
    
    print(f"Saved to {output_file}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    return embeddings


def train_dependency_representation(rule_numbers: List[int],
                                   output_dir: Path,
                                   device: str = 'cpu') -> Dict:
    """Train Dependency graph representation"""
    print("\n" + "="*60)
    print("Training Dependency Representation")
    print("="*60)
    
    # Build dataset
    print("Building dependency graphs...")
    dataset = build_dependency_dataset(rule_numbers)
    print(f"  Created {len(dataset)} graphs")
    
    # Model configuration
    config = {
        'input_dim': 2,
        'hidden_dims': [16, 8],
        'latent_dim': 8,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 0.001
    }
    
    # Create model
    model = GraphAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    )
    
    # Train
    print("Training model...")
    losses = train_model(
        model, dataset,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        device=device,
        verbose=True
    )
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, dataset, device=device)
    
    # Save results
    output_file = output_dir / 'dependency_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'config': config,
            'losses': losses,
            'model_state': model.state_dict()
        }, f)
    
    print(f"Saved to {output_file}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    return embeddings


def train_evolution_representation(rule_numbers: List[int],
                                  output_dir: Path,
                                  device: str = 'cpu') -> Dict:
    """Train Evolution graph representation with IC control"""
    print("\n" + "="*60)
    print("Training Evolution Representation (with IC control)")
    print("="*60)
    
    # Build dataset with multiple ICs
    print("Building evolution graphs with multiple ICs...")
    config = {
        'width': 51,
        'steps': 20,
        'n_ic_samples': 5,
        'input_dim': 3,
        'hidden_dims': [24, 12],
        'latent_dim': 8,
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.001
    }
    
    dataset, ic_mapping = build_evolution_dataset(
        rule_numbers,
        width=config['width'],
        steps=config['steps'],
        n_ic_samples=config['n_ic_samples']
    )
    print(f"  Created {len(dataset)} graphs ({len(rule_numbers)} rules x {config['n_ic_samples']} ICs)")
    
    # Create model
    model = GraphAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    )
    
    # Train on all IC samples
    print("Training model on all IC samples...")
    losses = train_model(
        model, dataset,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        device=device,
        verbose=True
    )
    
    # Extract raw embeddings (one per IC sample)
    print("Extracting embeddings...")
    raw_embeddings = {}
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data = data.to(device)
            z = model.encode(data)
            raw_embeddings[i] = z.cpu().numpy().flatten()
    
    # Aggregate embeddings by rule (mean pooling across ICs)
    print("Aggregating embeddings across ICs...")
    aggregated_embeddings = {}
    for rule_num, sample_indices in ic_mapping.items():
        ic_embs = [raw_embeddings[idx] for idx in sample_indices]
        aggregated_embeddings[rule_num] = np.mean(ic_embs, axis=0)
    
    # Save results
    output_file = output_dir / 'evolution_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': aggregated_embeddings,
            'raw_embeddings': raw_embeddings,
            'ic_mapping': ic_mapping,
            'config': config,
            'losses': losses,
            'model_state': model.state_dict()
        }, f)
    
    print(f"Saved to {output_file}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    return aggregated_embeddings


def main():
    """Main training pipeline"""
    
    # Setup
    output_dir = Path('../outputs/embeddings')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use CPU (can change to 'cuda' if available)
    device = 'cpu'
    
    # Get all 88 classified rules
    rule_numbers = sorted(list(WOLFRAM_CLASSES.keys()))
    print(f"Training on {len(rule_numbers)} rules")
    
    # Train all three representations
    results = {}
    
    # 1. Truth-Table
    results['truth_table'] = train_truth_table_representation(
        rule_numbers, output_dir, device
    )
    
    # 2. Dependency
    results['dependency'] = train_dependency_representation(
        rule_numbers, output_dir, device
    )
    
    # 3. Evolution
    results['evolution'] = train_evolution_representation(
        rule_numbers, output_dir, device
    )
    
    # Save summary
    summary = {
        'rule_numbers': rule_numbers,
        'n_rules': len(rule_numbers),
        'device': device,
        'representations': list(results.keys())
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    

    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - truth_table_embeddings.pkl")
    print("  - dependency_embeddings.pkl")
    print("  - evolution_embeddings.pkl")
    print("  - training_summary.json")
    
    print("\nEmbedding dimensions:")
    for name, emb_dict in results.items():
        sample_emb = next(iter(emb_dict.values()))
        print(f"  {name}: {len(rule_numbers)} rules x {len(sample_emb)} dims")


if __name__ == '__main__':
    main()