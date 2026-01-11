import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse

from gnn.dataset import CADebruijnDataset, create_train_val_test_splits


class GINEncoder(nn.Module):
    def __init__(self, input_dim=1, edge_dim=1, hidden_dim=128, num_layers=4, embedding_dim=128, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.edge_encoders = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            
            if edge_dim > 1:
                self.edge_encoders.append(nn.Linear(edge_dim, hidden_dim))
                mlp_in_dim = in_dim + hidden_dim
            else:
                mlp_in_dim = in_dim
            
            mlp = nn.Sequential(
                nn.Linear(mlp_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for i in range(self.num_layers):
            if self.edge_dim > 1 and edge_attr is not None:
                edge_emb = self.edge_encoders[i](edge_attr)
                
                row, col = edge_index
                edge_features = edge_emb
                aggregated = torch.zeros(x.size(0), edge_emb.size(1), device=x.device)
                aggregated.index_add_(0, col, edge_features)
                
                x_with_edges = torch.cat([x, aggregated], dim=1)
                x = self.convs[i](x_with_edges, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.project(x)
        
        return x


class GraphAutoencoder(nn.Module):
    def __init__(self, encoder, decoder_hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        
        embedding_dim = encoder.project[-1].out_features
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 32)
        )
    
    def forward(self, data):
        z = self.encoder(data)
        node_reconstruction = self.decoder(z)
        return z, node_reconstruction
    
    def encode(self, data):
        return self.encoder(data)


def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.mm(representations, representations.t())
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    positives = torch.cat([
        torch.diag(similarity_matrix, batch_size),
        torch.diag(similarity_matrix, -batch_size)
    ], dim=0)
    
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
    
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    logits = logits / temperature
    
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
    
    loss = F.cross_entropy(logits, labels)
    
    return loss


def augment_graph(data):
    num_edges = data.edge_index.shape[1]
    mask = torch.rand(num_edges) > 0.1
    
    augmented_data = data.clone()
    augmented_data.edge_index = data.edge_index[:, mask]
    if data.edge_attr is not None:
        augmented_data.edge_attr = data.edge_attr[mask]
    
    return augmented_data


def train_contrastive(model, loader, optimizer, device, temperature=0.5):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        
        data_aug = augment_graph(data)
        
        z_original = model.encode(data)
        z_augmented = model.encode(data_aug)
        
        loss = contrastive_loss(z_original, z_augmented, temperature)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def train_autoencoder(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        
        z, node_recon = model(data)
        
        target = data.x.squeeze()
        batch_idx = data.batch
        
        node_recon_expanded = node_recon[batch_idx]
        
        loss = F.mse_loss(node_recon_expanded, target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    rule_ids = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model.encode(data)
            embeddings.append(z.cpu().numpy())
            rule_ids.extend(data.y.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    rule_ids = np.array(rule_ids)
    
    return embeddings, rule_ids


def main():
    parser = argparse.ArgumentParser(description='Train De Bruijn GNN')
    parser.add_argument('--mode', type=str, default='deb', choices=['deb', '01'],
                        help='Graph representation: deb (32 nodes) or 01 (2 nodes)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--learning_mode', type=str, default='contrastive', 
                        choices=['contrastive', 'autoencoder'])
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (0 = single process)')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                        help='Pin memory for faster GPU transfer')
    args = parser.parse_args()
    
    if args.mode == 'deb':
        from ca.debruijn import rule_id_to_debruijn_graph as graph_builder
        input_dim = 1
        edge_dim = 1
        print("Using state graph representation (32 nodes, 64 edges)")
    else:
        from ca.simple01 import rule_id_to_simple_graph as graph_builder
        input_dim = 1
        edge_dim = 6
        print("Using symbol graph representation (2 nodes, 32 edges)")
    
    from gnn import dataset
    dataset.rule_id_to_debruijn_graph = graph_builder
    
    csv_path = "data/features/cheap_features.csv"
    output_dir = f"data/gnn/{args.mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_idx, val_idx, test_idx = create_train_val_test_splits(csv_path, seed=42)
    
    train_dataset = CADebruijnDataset(csv_path, split_indices=train_idx)
    val_dataset = CADebruijnDataset(csv_path, split_indices=val_idx)
    test_dataset = CADebruijnDataset(csv_path, split_indices=test_idx)
    
    # Configure DataLoader for efficient multiprocessing
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory and torch.cuda.is_available(),
        'persistent_workers': args.num_workers > 0,  # Keep workers alive between epochs
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    print(f"DataLoader config: num_workers={args.num_workers}, pin_memory={loader_kwargs['pin_memory']}, persistent_workers={loader_kwargs['persistent_workers']}")
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    encoder = GINEncoder(
        input_dim=input_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
        dropout=0.1
    ).to(device)
    
    model = GraphAutoencoder(encoder, decoder_hidden_dim=args.hidden_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining with {args.learning_mode} learning...")
    
    for epoch in range(args.epochs):
        if args.learning_mode == "contrastive":
            train_loss = train_contrastive(model, train_loader, optimizer, device, temperature=0.5)
            val_loss = train_contrastive(model, val_loader, optimizer, device, temperature=0.5)
        else:
            train_loss = train_autoencoder(model, train_loader, optimizer, device)
            val_loss = train_autoencoder(model, val_loader, optimizer, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  Model saved (val_loss improved to {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    print("\nExtracting embeddings...")
    train_emb, train_ids = extract_embeddings(model, train_loader, device)
    val_emb, val_ids = extract_embeddings(model, val_loader, device)
    test_emb, test_ids = extract_embeddings(model, test_loader, device)
    
    all_emb = np.vstack([train_emb, val_emb, test_emb])
    all_ids = np.concatenate([train_ids, val_ids, test_ids])
    
    emb_df = pd.DataFrame(all_emb)
    emb_df['rule_id'] = all_ids
    emb_df.to_csv(os.path.join(output_dir, 'debruijn_embeddings.csv'), index=False)
    
    print(f"Saved embeddings: {all_emb.shape}")
    print(f"Output: {os.path.join(output_dir, 'debruijn_embeddings.csv')}")


if __name__ == "__main__":
    main()