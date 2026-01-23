# Multi-View GAT for CA Classification

Classifying radius-2 cellular automata rules using Graph Attention Networks.

We use 3 graph representations for each rule:
- **Symbol**: input-output state mapping (2 nodes)
- **De Bruijn**: local state transitions (16 nodes)
- **Dependency**: spatiotemporal causal structure (T×W nodes)

## Quick Start

```bash
# 1. convert json labels to csv
cd data
python json_to_csv.py -d . -o dataset.csv --include-ic

# 2. cache PyG graphs (optional but faster)
cd ..
python -m gnn.cache --csv data/dataset.csv --out data/graphs.pt

# 3. train
python -m gnn.train \
    --data_path data/graphs.pt \
    --output_dir outputs/exp1 \
    --batch_size 8 \
    --epochs 50 \
    --class_weights

# 4. visualize attention
python -m gnn.interpret \
    --model outputs/exp1/best_model.pt \
    --data data/dataset.csv \
    --idx 0
```

## Project Structure

```
├── ca/                 # CA graph generation
│   ├── graphs.py       # 3 view generators
│   ├── rules.py        # rule table utils
│   └── evolve.py       # CA evolution
├── gnn/                # model & training
│   ├── model.py        # MultiViewGAT
│   ├── dataset.py      # PyG dataset
│   ├── train.py        # training script
│   ├── interpret.py    # attention analysis
│   └── cache.py        # pre-cache graphs
├── data/               # json labels + csv
└── outputs/            # checkpoints & plots
```

## PBS Jobs

```bash
qsub graphs.pbs      # generate graph cache
qsub train.pbs       # train model
qsub interpret.pbs   # attention visualization
```

## Notes

- ~550 samples, 5 classes, heavily imbalanced (Propagate dominates)
- use `--class_weights` and `--stratified` to help with imbalance
- small model recommended: `hidden_dim=32, heads=2, dropout=0.3`
- currently overfits to majority class - need more data

## Dependencies

```
torch
torch_geometric
pandas
numpy
matplotlib
networkx
scikit-learn
tqdm
```
