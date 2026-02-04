import argparse, random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_list_pt(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list[Data], got {type(obj)}")
    return obj

def normalize_y(g):
    # y: tensor([k]) -> tensor(k)
    if hasattr(g, "y") and isinstance(g.y, torch.Tensor) and g.y.numel() == 1:
        g.y = g.y.long().view(1).squeeze(0)
    return g

def get_Xy(graphs, use_rule_id=False):
    """
    X: [N, 32]
    y: [N]
    """
    X_list, y_list = [], []
    for g in graphs:
        g = normalize_y(g)
        y_list.append(int(g.y))

        if use_rule_id:
            # decode 32-bit truth table from rule_id
            rid = int(g.rule_id)
            bits = [(rid >> i) & 1 for i in range(32)]  # i=0..31
            X_list.append(torch.tensor(bits, dtype=torch.float))
        else:
            ea = g.edge_attr  # [32,1] uint8
            X_list.append(ea.view(-1).float())  # [32]

    X = torch.stack(X_list, dim=0)  # [N,32]
    y = torch.tensor(y_list, dtype=torch.long)  # [N]
    return X, y

def stratified_split(X, y, seed=42, test_ratio=0.2, val_ratio=0.1):
    """
    Simple stratified split without sklearn.
    Works well for your N=449.
    """
    rng = random.Random(seed)
    indices_by_class = {}
    for i, yi in enumerate(y.tolist()):
        indices_by_class.setdefault(yi, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for c, idxs in indices_by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_ratio))
        n_val  = int(round(n * val_ratio))
        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test:n_test+n_val])
        train_idx.extend(idxs[n_test+n_val:])

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

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

def compute_class_weights(y: torch.Tensor, C: int):
    counts = torch.zeros(C, dtype=torch.float)
    for k in y.tolist():
        counts[k] += 1
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
# Model: MLP(32)->8
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=32, hidden=128, num_classes=8, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Train/Eval
# -------------------------
def train_epoch(model, X, y, optimizer, criterion, batch_size, device, sampler=None):
    model.train()
    N = X.size(0)

    if sampler is None:
        idx = torch.randperm(N)
    else:
        idx = torch.tensor(list(sampler), dtype=torch.long)

    total = 0.0
    for start in range(0, N, batch_size):
        b = idx[start:start+batch_size]
        xb = X[b].to(device)
        yb = y[b].to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total += loss.item() * xb.size(0)

    return total / N

@torch.no_grad()
def evaluate(model, X, y, device, C):
    model.eval()
    logits = model(X.to(device)).cpu()
    pred = logits.argmax(dim=-1)
    cm = confusion_matrix(y, pred, C)
    acc = (cm.diag().sum().float() / cm.sum().float()).item()
    return {"acc": acc, "bal_acc": bal_acc(cm), "macro_f1": macro_f1(cm), "cm": cm}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imbalance", type=str, default="class_weight",
                    choices=["class_weight", "sampler", "focal"])
    ap.add_argument("--use_rule_id", action="store_true",
                    help="Use rule_id to decode 32 bits instead of edge_attr")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs = load_list_pt(args.pt)
    X, y = get_Xy(graphs, use_rule_id=args.use_rule_id)

    C = 8
    tr, va, te = stratified_split(X, y, seed=args.seed, test_ratio=0.2, val_ratio=0.1)
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]

    class_w = compute_class_weights(ytr, C).to(device)

    model = MLP(in_dim=32, hidden=args.hidden, num_classes=C, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.imbalance == "focal":
        criterion = FocalLoss(gamma=2.0, weight=class_w)
        sampler = None
    elif args.imbalance == "sampler":
        # build a simple weighted sampler index list
        weights = class_w.cpu()[ytr].numpy()
        # sample N indices with replacement
        import numpy as np
        np.random.seed(args.seed)
        sampler = np.random.choice(len(ytr), size=len(ytr), replace=True, p=weights/weights.sum())
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w)
        sampler = None

    best = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, Xtr, ytr, opt, criterion, args.batch_size, device,
                           sampler=sampler if args.imbalance == "sampler" else None)
        val_m = evaluate(model, Xva, yva, device, C)

        if val_m["macro_f1"] > best:
            best = val_m["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 50 == 0:
            print(f"ep={ep:03d} loss={loss:.4f}  "
                  f"val_acc={val_m['acc']:.3f} val_bal={val_m['bal_acc']:.3f} val_f1={val_m['macro_f1']:.3f}")

    model.load_state_dict(best_state)
    test_m = evaluate(model, Xte, yte, device, C)
    print("TEST:", {k: v for k, v in test_m.items() if k != "cm"})
    print("Confusion matrix:\n", test_m["cm"])

if __name__ == "__main__":
    main()