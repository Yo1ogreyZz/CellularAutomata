import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATEncoder(nn.Module):
    """GAT encoder for single view."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        # input -> hidden*heads -> hidden -> output
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch, return_attention=False):
        attention_dict = {}

        if return_attention:
            x, (edge_index1, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
            attention_dict['layer1'] = (edge_index1, alpha1)
        else:
            x = self.conv1(x, edge_index)
        
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, (edge_index2, alpha2) = self.conv2(x, edge_index, return_attention_weights=True)
            attention_dict['layer2'] = (edge_index2, alpha2)
        else:
            x = self.conv2(x, edge_index)
        x = F.elu(x)

        graph_emb = global_mean_pool(x, batch)
        out = self.project(graph_emb)

        if return_attention:
            return out, attention_dict
        return out


class MultiViewGAT(nn.Module):
    """Multi-view GAT: Symbol + De Bruijn + Dependency (no Lattice)."""
    
    def __init__(self, hidden_dim=64, embedding_dim=32, num_classes=5, heads=4, dropout=0.1):
        super().__init__()
        
        # symbol: 2 nodes, feature dim=1
        self.encoder_symbol = GATEncoder(
            input_dim=1, hidden_dim=hidden_dim, 
            output_dim=embedding_dim, heads=heads, dropout=dropout
        )
        
        # debruijn: 16 nodes, feature dim=4
        self.encoder_debruijn = GATEncoder(
            input_dim=4, hidden_dim=hidden_dim, 
            output_dim=embedding_dim, heads=heads, dropout=dropout
        )
        
        # dependency: T*W nodes, feature dim=2
        self.encoder_dependency = GATEncoder(
            input_dim=2, hidden_dim=hidden_dim, 
            output_dim=embedding_dim, heads=heads, dropout=dropout
        )
        
        # 3 views -> 3 * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch_data, return_attention=False):
        out_sym = self.encoder_symbol(
            batch_data['symbol'].x, 
            batch_data['symbol'].edge_index, 
            batch_data['symbol'].batch,
            return_attention
        )
        
        out_deb = self.encoder_debruijn(
            batch_data['debruijn'].x, 
            batch_data['debruijn'].edge_index, 
            batch_data['debruijn'].batch,
            return_attention
        )
        
        out_dep = self.encoder_dependency(
            batch_data['dependency'].x, 
            batch_data['dependency'].edge_index, 
            batch_data['dependency'].batch,
            return_attention
        )

        if return_attention:
            emb_sym, att_sym = out_sym
            emb_deb, att_deb = out_deb
            emb_dep, att_dep = out_dep
            
            z_all = torch.cat([emb_sym, emb_deb, emb_dep], dim=1)
            logits = self.classifier(z_all)
            
            return logits, {
                'symbol': att_sym,
                'debruijn': att_deb,
                'dependency': att_dep
            }
        
        z_all = torch.cat([out_sym, out_deb, out_dep], dim=1)
        return self.classifier(z_all)
