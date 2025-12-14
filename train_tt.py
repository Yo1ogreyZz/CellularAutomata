# train_gnn.py
import sys
sys.path.append('src')

import pickle
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from gnn_models import GraphAutoencoder, GNNTrainer
import utils

print("="*60)
print("Training GNN on Truth-Table Graphs")
print("="*60)


print("\n[1] Loading dataset...")
with open('data/eca_graphs_truth_table.pkl', 'rb') as f:
    dataset_dict = pickle.load(f)
print(f"✓ Loaded {len(dataset_dict)} rules")


print("\n[2] Converting to PyG format...")
pyg_dataset = []
for rule_num, graph_data in dataset_dict.items():
    
    x = torch.tensor(graph_data['node_features'], dtype=torch.float)
    edges = graph_data['edges']
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([utils.get_wolfram_class_id(rule_num)], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        rule_number=rule_num
    )
    pyg_dataset.append(data)

print(f"✓ Created {len(pyg_dataset)} PyG graphs")
print(f"  Node features: {pyg_dataset[0].x.shape}")
print(f"  Edge index: {pyg_dataset[0].edge_index.shape}")


print("\n[3] Creating DataLoader...")
loader = DataLoader(pyg_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(pyg_dataset, batch_size=len(pyg_dataset), shuffle=False)


print("\n[4] Initializing model...")
model = GraphAutoencoder(
    input_dim=4,           # [left, center, right, output]
    hidden_dims=[32, 16],
    latent_dim=8
)
print(f"✓ Model created")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")


print("\n[5] Training...")
trainer = GNNTrainer(model, learning_rate=0.001, device='cpu')
losses = trainer.train(loader, epochs=200, verbose=True)


np.savetxt('training_loss.txt', losses)
print(f"✓ Loss saved to training_loss.txt")


print("\n[6] Extracting embeddings...")
embeddings_dict = trainer.extract_embeddings(eval_loader)


all_rules = sorted(embeddings_dict.keys())
embedding_matrix = np.array([embeddings_dict[r] for r in all_rules])


np.save('gnn_embeddings.npy', embedding_matrix)
np.save('rule_numbers.npy', np.array(all_rules))

print(f"\n✓ Embeddings saved!")
print(f"  Shape: {embedding_matrix.shape}")
print(f"  Rules: {len(all_rules)}")


print("\n[7] Quick verification...")
controversial = [18, 54, 62, 73, 110, 126]
print("Controversial rules embeddings:")
for rule in controversial:
    if rule in embeddings_dict:
        emb = embeddings_dict[rule]
        norm = np.linalg.norm(emb)
        print(f"  Rule {rule:3d}: norm={norm:.3f}")

print("\n" + "="*60)
print("✓ Training Complete!")
print("="*60)
print(f"\nOutputs:")
print(f"  - gnn_embeddings.npy  ({embedding_matrix.shape})")
print(f"  - rule_numbers.npy    ({len(all_rules)} rules)")
print(f"  - training_loss.txt   ({len(losses)} epochs)")