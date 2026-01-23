"""Attention weight analysis and visualization."""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from gnn.dataset import CAMultiViewDataset, custom_collate_fn, CLASS_NAMES
from gnn.model import MultiViewGAT


def visualize_attention(edge_index, attention_weights, title="Attention", save_path=None):
    num_nodes = edge_index.max().item() + 1
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=False)
    
    weights = attention_weights.mean(dim=1).detach().cpu().numpy()
    
    # normalize weights to [0, 1] for colormap
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        weights_norm = (weights - w_min) / (w_max - w_min)
    else:
        weights_norm = np.ones_like(weights) * 0.5
    
    # convert to RGBA colors
    cmap = plt.cm.Reds
    edge_colors = [cmap(w) for w in weights_norm]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue', edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Attention")
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def analyze_attention(attentions, view_name):
    print(f"\n{view_name}:")
    for layer, (edge_index, alpha) in attentions[view_name].items():
        w = alpha.detach().cpu().numpy()
        print(f"  {layer}: shape={w.shape}, mean={w.mean():.4f}, std={w.std():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='outputs/best_model.pt')
    parser.add_argument('--data', default='data/dataset.csv')
    parser.add_argument('--idx', type=int, default=0, help='Sample index')
    parser.add_argument('--output_dir', default='outputs/attention')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--heads', type=int, default=2)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MultiViewGAT(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_classes=len(CLASS_NAMES),
        heads=args.heads
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    dataset = CAMultiViewDataset(args.data)
    sample = dataset[args.idx]
    
    batch = custom_collate_fn([sample])
    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    
    rule_id = batch['rule_id'].item()
    true_label = batch['label'].item()
    
    with torch.no_grad():
        logits, attentions = model(batch, return_attention=True)
        pred = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    print(f"Rule {rule_id}: true={CLASS_NAMES[true_label]}, pred={CLASS_NAMES[pred]}")
    print(f"Probs: {dict(zip(CLASS_NAMES, probs.round(3)))}")
    
    for view in ['symbol', 'debruijn', 'dependency']:
        analyze_attention(attentions, view)
        edge_index, alpha = attentions[view]['layer1']
        save_path = os.path.join(args.output_dir, f'{view}_rule{rule_id}.png')
        visualize_attention(edge_index, alpha, f'{view} (Rule {rule_id})', save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
