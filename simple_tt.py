# train_now.py - 立即可用
import sys
sys.path.append('src')
import pickle
import numpy as np

with open('data/eca_graphs_truth_table.pkl', 'rb') as f:
    dataset = pickle.load(f)

all_rules = sorted(dataset.keys())
embeddings = []

for rule_num in all_rules:
    feat = dataset[rule_num]['node_features']
    emb = [feat.mean(), feat.std(), feat.max(), feat.min(),
           feat[:, 0].mean(), feat[:, 1].mean(), 
           feat[:, 2].mean(), feat[:, 3].mean()]
    embeddings.append(emb)

np.save('outputs/gnn_embeddings_tt.npy', np.array(embeddings))
np.save('outputs/rule_numbers_tt.npy', np.array(all_rules))
print(f"Done: {len(all_rules)} rules")