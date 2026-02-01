import argparse, random
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool


# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_list_pt(path: str) -> List[Data]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Expected list[Data] in {path}, got {type(obj)}")

def normalize_graph(g: Data) -> Data:
    # y: tensor([k]) -> tensor(k)
    if hasattr(g, "y") and isinstance(g.y, torch.Tensor) and g.y.numel() == 1:
        g.y = g.y.long().view(1).squeeze(0)
    # rule_id: tensor -> int
    if hasattr(g, "rule_id") and isinstance(g.rule_id, torch.Tensor) and g.rule_id.numel() == 1:
        g.rule_id = int(g.rule_id.item())
    return g

def group_split_by_rule_id(graphs: List[Data], seed=42, test_ratio=0.2, val_ratio=0.1):
    rule_to_indices: Dict[int, List[int]] = {}
    for i, g in enumerate(graphs):
        rid = int(g.rule_id)
        rule_to_indices.setdefault(rid, []).append(i)

    rules = list(rule_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(rules)

    n = len(rules)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))

    test_rules = set(rules[:n_test])
    val_rules  = set(rules[n_test:n_test+n_val])
    train_rules= set(rules[n_test+n_val:])

    def collect(rs):
        idx = []
        for r in rs:
            idx.extend(rule_to_indices[r])
        return idx

    return collect(train_rules), collect(val_rules), collect(test_rules)

@torch.no_grad()
def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, C: int):
    cm = torch.zeros(C, C, dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm

@torch.no_grad()
def macro_f1(cm: torch.Tensor, eps=1e-9):
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return f1.mean().item()

@torch.no_grad()
def bal_acc(cm: torch.Tensor, eps=1e-9):
    tp = cm.diag().float()
    fn = cm.sum(1).float() - tp
    rec = tp / (tp + fn + eps)
    return rec.mean().item()

def compute_class_weights(graphs: List[Data], C: int):
    counts = torch.zeros(C, dtype=torch.float)
    for g in graphs:
        counts[int(g.y)] += 1
    present = counts > 0
    w = torch.ones(C, dtype=torch.float)
    w[present] = 1.0 / counts[present]
    w = w / w.sum() * C
    return w

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

# -------------------------
# Model
# -------------------------
class GATClassifier(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, heads=4, dropout=0.2, num_classes=8):
        super().__init__()
        self.dropout = dropout
        self.use_edge = edge_dim is not None and edge_dim > 0

        self.gat1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=dropout,
                              edge_dim=edge_dim if self.use_edge else None)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=1, dropout=dropout,
                              edge_dim=edge_dim if self.use_edge else None)

        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)

    def forward(self, data: Data):
        x, ei = data.x, data.edge_index
        ea = getattr(data, "edge_attr", None)

        if x.dtype != torch.float32:
            x = x.float()
        if ea is not None and ea.dtype != torch.float32:
            ea = ea.float()

        x = self.gat1(x, ei, ea if self.use_edge else None)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, ei, ea if self.use_edge else None)
        x = F.elu(x)

        g = global_mean_pool(x, data.batch)
        g = F.relu(self.lin1(g))
        g = F.dropout(g, p=self.dropout, training=self.training)
        return self.lin2(g)

# -------------------------
# Train/Eval
# -------------------------
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += loss.item() * data.num_graphs
        n += data.num_graphs
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device, C):
    model.eval()
    ys, ps = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data)
        pred = logits.argmax(dim=-1).cpu()
        ys.append(data.y.cpu())
        ps.append(pred)
    y = torch.cat(ys)
    p = torch.cat(ps)
    cm = confusion_matrix(y, p, C)
    acc = (cm.diag().sum().float() / cm.sum().float()).item()
    return {
        "acc": acc,
        "bal_acc": bal_acc(cm),
        "macro_f1": macro_f1(cm),
        "cm": cm
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True, help="path to ONE pt file (e.g., debruijn)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imbalance", type=str, default="class_weight",
                    choices=["class_weight", "sampler", "focal"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = [normalize_graph(g) for g in load_list_pt(args.pt)]
    C = 8  # from your view_labelmap.py
    # quick sanity check
    max_y = max(int(g.y) for g in graphs)
    if max_y >= C:
        raise ValueError(f"Found y={max_y} but C={C}")

    tr_idx, va_idx, te_idx = group_split_by_rule_id(graphs, seed=args.seed)
    train_set = [graphs[i] for i in tr_idx]
    val_set   = [graphs[i] for i in va_idx]
    test_set  = [graphs[i] for i in te_idx]

    class_w = compute_class_weights(train_set, C).to(device)

    if args.imbalance == "sampler":
        from torch.utils.data import WeightedRandomSampler
        weights = [class_w[int(g.y)].item() for g in train_set]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
        criterion = nn.CrossEntropyLoss()  # sampling already balances
    elif args.imbalance == "focal":
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        criterion = FocalLoss(gamma=2.0, weight=class_w)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss(weight=class_w)

    val_loader  = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    in_dim = train_set[0].x.size(-1)
    ea = getattr(train_set[0], "edge_attr", None)
    edge_dim = ea.size(-1) if ea is not None else None

    model = GATClassifier(in_dim, edge_dim, hidden=args.hidden, heads=args.heads,
                          dropout=args.dropout, num_classes=C).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, criterion, device)
        val_m = evaluate(model, val_loader, device, C)

        if val_m["macro_f1"] > best:
            best = val_m["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 20 == 0:
            print(f"ep={ep:03d} loss={loss:.4f}  "
                  f"val_acc={val_m['acc']:.3f} val_bal={val_m['bal_acc']:.3f} val_f1={val_m['macro_f1']:.3f}")

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, device, C)
    print("TEST:", {k: v for k, v in test_m.items() if k != "cm"})
    print("Confusion matrix:\n", test_m["cm"])

if __name__ == "__main__":
    main()