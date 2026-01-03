"""Training script for ECA graph embeddings"""

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
from datetime import datetime

sys.path.append('..')
from src.rule2graph import ECARule, TruthTableGraph, DependencyGraph, EvolutionGraph, PatternVocabularyGraph
from src.utils import WOLFRAM_CLASSES, to_pyg_data, normalize_dataset_edge_attr


class GraphAutoencoder(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            layers.append(GCNConv(dims[i], dims[i+1]))
        self.encoder_layers = nn.ModuleList(layers)
        
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
        self.decoder_layers = nn.ModuleList(decoder_layers)
    
    def encode(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, edge_index)
            if i < len(self.encoder_layers) - 1:
                x = F.relu(x)
        
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        embedding = global_mean_pool(x, batch)
        
        return embedding
    
    def decode(self, z: torch.Tensor, num_nodes: int) -> torch.Tensor:
        z_expanded = z.repeat(num_nodes, 1)
        
        x = z_expanded
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i < len(self.decoder_layers) - 1:
                x = F.relu(x)
                if self.dropout > 0 and self.training:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(data)
        
        if hasattr(data, 'batch') and data.batch is not None:

            x_recon_list = []
            unique_batches = torch.unique(data.batch, sorted=True)
            
            for graph_idx in unique_batches:
                
                node_mask = (data.batch == graph_idx)
                num_nodes = node_mask.sum().item()

                graph_idx_int = graph_idx.item()
                graph_embedding = z[graph_idx_int:graph_idx_int+1]  # [1, latent_dim]
                
                x_recon_graph = self.decode(graph_embedding, num_nodes)
                x_recon_list.append(x_recon_graph)
            
            x_recon = torch.cat(x_recon_list, dim=0)
        else:
            x_recon = self.decode(z, data.x.size(0))
        
        return z, x_recon


def build_truth_table_dataset(rule_numbers: List[int]) -> List[Data]:
    dataset = []
    
    for rule_num in rule_numbers:
        rule = ECARule(rule_num)
        graph_builder = TruthTableGraph(rule)
        graph_data = graph_builder.build()
        
        pyg_data = to_pyg_data(graph_data)
        pyg_data.rule_number = torch.tensor([rule_num])
        dataset.append(pyg_data)
    
    # 根据图类型统一处理 edge_attr（truth_table 不需要处理，但保持一致性）
    dataset = normalize_dataset_edge_attr(dataset, graph_type='truth_table')
    
    return dataset


def build_dependency_dataset(rule_numbers: List[int]) -> List[Data]:
    dataset = []
    
    for rule_num in rule_numbers:
        rule = ECARule(rule_num)
        graph_builder = DependencyGraph(rule)
        graph_data = graph_builder.build()
        
        pyg_data = to_pyg_data(graph_data)
        pyg_data.rule_number = torch.tensor([rule_num])
        dataset.append(pyg_data)
    
    # 根据图类型统一处理 edge_attr（dependency 不需要处理，但保持一致性）
    dataset = normalize_dataset_edge_attr(dataset, graph_type='dependency')
    
    return dataset


def build_pattern_vocabulary_dataset(rule_numbers: List[int], pattern_size: int = 3) -> List[Data]:
    dataset = []
    
    for rule_num in rule_numbers:
        rule = ECARule(rule_num)
        graph_builder = PatternVocabularyGraph(rule, pattern_size=pattern_size)
        graph_data = graph_builder.build()
        
        pyg_data = to_pyg_data(graph_data)
        pyg_data.rule_number = torch.tensor([rule_num])
        dataset.append(pyg_data)
    
    # 根据图类型统一处理 edge_attr
    dataset = normalize_dataset_edge_attr(dataset, graph_type='pattern_vocabulary')
    
    return dataset


def build_evolution_dataset(rule_numbers: List[int],
                            width: int = 51,
                            steps: int = 50,
                            n_ic_samples: int = 5) -> Tuple[List[Data], Dict]:
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
    
    # 根据图类型统一处理 edge_attr（evolution 不需要处理，但保持一致性）
    dataset = normalize_dataset_edge_attr(dataset, graph_type='evolution')
    
    return dataset, ic_mapping


def train_model(model: nn.Module,
                dataset: List[Data],
                epochs: int = 200,
                batch_size: int = 16,
                learning_rate: float = 0.001,
                weight_decay: float = 0.0,
                device: str = 'cpu',
                verbose: bool = True) -> List[float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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


def train_truth_table_representation(rule_numbers: List[int],
                                     output_dir: Path,
                                     device: str = 'cpu') -> Dict:
    print("\nTraining truth-table representation...")
    dataset = build_truth_table_dataset(rule_numbers)
    print(f"Built {len(dataset)} graphs")
    
    config = {
        'input_dim': 4,
        'hidden_dims': [32, 16],
        'latent_dim': 8,
        'batch_size': 16,
        'epochs': 125,
        'learning_rate': 0.001,
        'dropout': 0.1,
        'weight_decay': 1e-4
    }
    
    model = GraphAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout']
    )
    
    losses = train_model(model, dataset, epochs=config['epochs'],
                        batch_size=config['batch_size'],
                        learning_rate=config['learning_rate'],
                        weight_decay=config['weight_decay'],
                        device=device, verbose=True)
    
    embeddings = extract_embeddings(model, dataset, device=device)
    
    output_file = output_dir / 'truth_table_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'config': config,
            'losses': losses,
            'model_state': model.state_dict()
        }, f)
    
    print(f"Saved to {output_file}, final loss: {losses[-1]:.6f}")
    
    return embeddings


def train_dependency_representation(rule_numbers: List[int],
                                   output_dir: Path,
                                   device: str = 'cpu') -> Dict:
    print("\nTraining dependency representation...")
    dataset = build_dependency_dataset(rule_numbers)
    print(f"Built {len(dataset)} graphs")
    
    config = {
        'input_dim': 2,
        'hidden_dims': [32, 16],
        'latent_dim': 8,
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4
    }
    
    model = GraphAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    )
    
    losses = train_model(model, dataset, epochs=config['epochs'],
                        batch_size=config['batch_size'],
                        learning_rate=config['learning_rate'],
                        weight_decay=config['weight_decay'],
                        device=device, verbose=True)
    
    embeddings = extract_embeddings(model, dataset, device=device)
    
    output_file = output_dir / 'dependency_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'config': config,
            'losses': losses,
            'model_state': model.state_dict()
        }, f)
    
    print(f"Saved to {output_file}, final loss: {losses[-1]:.6f}")
    
    return embeddings


def train_pattern_vocabulary_representation(rule_numbers: List[int],
                                           output_dir: Path,
                                           device: str = 'cpu') -> Dict:
    print("\nTraining pattern vocabulary representation...")
    dataset = build_pattern_vocabulary_dataset(rule_numbers, pattern_size=3)
    print(f"Built {len(dataset)} graphs")
    
    config = {
        'pattern_size': 3,
        'input_dim': 4,
        'hidden_dims': [32, 16],
        'latent_dim': 8,
        'batch_size': 16,
        'epochs': 200,
        'learning_rate': 0.001
    }
    
    model = GraphAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    )
    
    losses = train_model(model, dataset, epochs=config['epochs'],
                        batch_size=config['batch_size'],
                        learning_rate=config['learning_rate'],
                        device=device, verbose=True)
    
    embeddings = extract_embeddings(model, dataset, device=device)
    
    output_file = output_dir / 'pattern_vocabulary_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'config': config,
            'losses': losses,
            'model_state': model.state_dict()
        }, f)
    
    print(f"Saved to {output_file}, final loss: {losses[-1]:.6f}")
    
    return embeddings


def train_evolution_representation(rule_numbers: List[int],
                                  output_dir: Path,
                                  device: str = 'cpu') -> Dict:
    print("\nTraining evolution representation...")
    config = {
        'width': 51,
        'steps': 50,
        'n_ic_samples': 32,
        'input_dim': 3,
        'hidden_dims': [64, 32],
        'latent_dim': 8,
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.0005,
        'weight_decay': 1e-4
    }
    
    dataset, ic_mapping = build_evolution_dataset(
        rule_numbers,
        width=config['width'],
        steps=config['steps'],
        n_ic_samples=config['n_ic_samples']
    )
    print(f"Built {len(dataset)} graphs ({len(rule_numbers)} rules x {config['n_ic_samples']} ICs)")
    
    model = GraphAutoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    )
    
    losses = train_model(model, dataset, epochs=config['epochs'],
                        batch_size=config['batch_size'],
                        learning_rate=config['learning_rate'],
                        weight_decay=config['weight_decay'],
                        device=device, verbose=True)
    
    raw_embeddings = {}
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data = data.to(device)
            z = model.encode(data)
            raw_embeddings[i] = z.cpu().numpy().flatten()
    aggregated_embeddings_mean = {}
    aggregated_embeddings_std = {}
    
    for rule_num, sample_indices in ic_mapping.items():
        ic_embs = np.array([raw_embeddings[idx] for idx in sample_indices])
        aggregated_embeddings_mean[rule_num] = np.mean(ic_embs, axis=0)
        aggregated_embeddings_std[rule_num] = np.std(ic_embs, axis=0)
    
    output_file = output_dir / 'evolution_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings_mean': aggregated_embeddings_mean,
            'embeddings_std': aggregated_embeddings_std,
            'raw_embeddings': raw_embeddings,
            'ic_mapping': ic_mapping,
            'config': config,
            'losses': losses,
            'model_state': model.state_dict()
        }, f)
    
    print(f"Saved to {output_file}, final loss: {losses[-1]:.6f}")
    
    return aggregated_embeddings_mean


def main():
    # 获取版本名（从环境变量或使用时间戳）
    version_name = os.environ.get('VERSION_NAME', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    # 获取输出基础目录（从环境变量或使用默认路径）
    if 'OUTPUT_BASE_DIR' in os.environ:
        # 如果从环境变量获取，使用绝对路径
        output_base = Path(os.environ['OUTPUT_BASE_DIR']).resolve()
    else:
        # 否则使用相对路径（相对于脚本位置）
        script_dir = Path(__file__).parent
        output_base = (script_dir / '../outputs/embeddings' / version_name).resolve()
    
    output_dir = output_base
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保使用 GPU（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rule_numbers = sorted(list(WOLFRAM_CLASSES.keys()))
    
    print("=" * 50)
    print(f"Training ECA Graph Embeddings")
    print(f"Version: {version_name}")
    print(f"Output Directory: {output_dir}")
    print(f"Training on {len(rule_numbers)} rules using {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("=" * 50)
    
    results = {}
    
    results['truth_table'] = train_truth_table_representation(
        rule_numbers, output_dir, device
    )
    
    results['dependency'] = train_dependency_representation(
        rule_numbers, output_dir, device
    )
    
    results['pattern_vocabulary'] = train_pattern_vocabulary_representation(
        rule_numbers, output_dir, device
    )
    
    results['evolution'] = train_evolution_representation(
        rule_numbers, output_dir, device
    )
    
    summary = {
        'version': version_name,
        'rule_numbers': rule_numbers,
        'n_rules': len(rule_numbers),
        'device': device,
        'representations': list(results.keys()),
        'output_directory': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }
    
    # 添加 GPU 信息（如果使用 GPU）
    if device == 'cuda':
        summary['gpu_name'] = torch.cuda.get_device_name(0)
        summary['cuda_version'] = torch.version.cuda
        summary['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"Training completed successfully!")
    print(f"Version: {version_name}")
    print(f"Results saved to: {output_dir}")
    print("\nEmbedding dimensions:")
    for name, emb_dict in results.items():
        sample_emb = next(iter(emb_dict.values()))
        print(f"  {name}: {len(rule_numbers)} rules x {len(sample_emb)} dims")
    print("=" * 50)


if __name__ == '__main__':
    main()