import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


# -------------------------
# 1) 读取你已经生成好的 PyG InMemoryDataset (symbol01)
# -------------------------
class LoadProcessedSymbol01(InMemoryDataset):
    """
    读取 output/pyg_radius2_symbol01_std/processed 下的 (data, slices)
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        meta_path = self.processed_paths[1]
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False)
            self.classes = meta.get("classes", None)      # list[str]
            self.label_map = meta.get("label_map", None)  # dict[str,int]
        else:
            self.classes = None
            self.label_map = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt", "meta.pt"]


# -------------------------
# 2) 模型：GINEConv 图分类（edge_attr 会参与 message passing）
# -------------------------
class GraphClassifier(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, num_classes=8, dropout=0.2):
        super().__init__()

        def mlp(in_ch, out_ch):
            return nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(),
                nn.Linear(out_ch, out_ch),
            )

        self.node_encoder = nn.Linear(in_dim, hidden)
        self.edge_encoder = nn.Linear(edge_dim, hidden)

        self.conv1 = GINEConv(nn=mlp(hidden, hidden), edge_dim=hidden)
        self.conv2 = GINEConv(nn=mlp(hidden, hidden), edge_dim=hidden)
        self.conv3 = GINEConv(nn=mlp(hidden, hidden), edge_dim=hidden)

        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        x = self.conv1(x, edge_index, e)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, e)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, e)
        x = F.relu(x)

        g = global_mean_pool(x, batch)
        out = self.classifier(g)
        return out


# -------------------------
# 3) 分层切分工具
# -------------------------
def stratified_split_indices(
    y: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    g = torch.Generator()
    g.manual_seed(seed)

    train_ids, val_ids, test_ids = [], [], []
    classes = torch.unique(y).tolist()

    for c in classes:
        idx_c = torch.nonzero(y == c, as_tuple=False).view(-1)
        idx_c = idx_c[torch.randperm(idx_c.numel(), generator=g)]

        n_c = idx_c.numel()
        n_train = int(round(train_ratio * n_c))
        n_val = int(round(val_ratio * n_c))

        n_train = min(n_train, n_c)
        n_val = min(n_val, n_c - n_train)

        train_ids.append(idx_c[:n_train])
        val_ids.append(idx_c[n_train:n_train + n_val])
        test_ids.append(idx_c[n_train + n_val:])

    train_idx = torch.cat(train_ids, dim=0)
    val_idx = torch.cat(val_ids, dim=0)
    test_idx = torch.cat(test_ids, dim=0)

    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=g)]

    return train_idx, val_idx, test_idx


def count_by_class(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.bincount(y, minlength=num_classes)


# -------------------------
# 4) confusion matrix + macro-F1
# -------------------------
@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def macro_f1_from_cm(cm: torch.Tensor) -> float:
    C = cm.size(0)
    f1s = []
    for k in range(C):
        support = cm[k, :].sum().item()
        if support == 0:
            continue

        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return float(sum(f1s) / max(len(f1s), 1))


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    all_pred, all_true = [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, data.y.view(-1))
        loss_sum += loss.detach().item() * data.num_graphs

        pred = logits.argmax(dim=-1)
        y = data.y.view(-1)

        correct += int((pred == y).sum())
        total += data.num_graphs

        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    if total == 0:
        return 0.0, 0.0, 0.0, torch.zeros((num_classes, num_classes), dtype=torch.long)

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)

    cm = confusion_matrix(all_pred, all_true, num_classes)
    mf1 = macro_f1_from_cm(cm)

    return loss_sum / total, correct / total, mf1, cm


def pretty_print_cm(cm: torch.Tensor, class_names=None):
    C = cm.size(0)
    header = ["true\\pred"] + [str(i) for i in range(C)]
    if class_names is not None and len(class_names) == C:
        header = ["true\\pred"] + [f"{i}:{class_names[i]}" for i in range(C)]

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(" | ".join(header))
    for i in range(C):
        row_name = str(i)
        if class_names is not None and len(class_names) == C:
            row_name = f"{i}:{class_names[i]}"
        row = [row_name] + [str(int(cm[i, j].item())) for j in range(C)]
        print(" | ".join(row))


# -------------------------
# 5) main：训练 symbol01
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/pyg_radius2_symbol01_std", help="symbol01 dataset root (contains processed/)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    ds = LoadProcessedSymbol01(root=args.root)
    print("graphs:", len(ds), flush=True)

    sample = ds[0]
    in_dim = sample.x.size(-1)             # symbol01: 2
    edge_dim = sample.edge_attr.size(-1)   # symbol01: 5 (neighbors + dir)

    all_y = ds.y.view(-1).cpu()
    num_classes = int(torch.unique(all_y).numel())
    class_names = ds.classes if (getattr(ds, "classes", None) is not None and len(ds.classes) == num_classes) else None

    print("in_dim:", in_dim, "edge_dim:", edge_dim, "num_classes:", num_classes, flush=True)

    # stratified split
    train_idx, val_idx, test_idx = stratified_split_indices(
        y=all_y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    train_ds = ds[train_idx]
    val_ds = ds[val_idx]
    test_ds = ds[test_idx]

    print("split sizes:", len(train_ds), len(val_ds), len(test_ds), flush=True)

    train_counts = count_by_class(train_ds.y.view(-1).cpu(), num_classes)
    val_counts = count_by_class(val_ds.y.view(-1).cpu(), num_classes)
    test_counts = count_by_class(test_ds.y.view(-1).cpu(), num_classes)

    print("class counts (train):", train_counts.tolist(), flush=True)
    print("class counts (val)  :", val_counts.tolist(), flush=True)
    print("class counts (test) :", test_counts.tolist(), flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GraphClassifier(in_dim=in_dim, edge_dim=edge_dim, hidden=args.hidden, num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, total = 0.0, 0

        for data in train_loader:
            data = data.to(device)
            optim.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, data.y.view(-1))
            loss.backward()
            optim.step()

            loss_sum += loss.detach().item() * data.num_graphs
            total += data.num_graphs

        train_loss = loss_sum / max(total, 1)
        val_loss, val_acc, val_mf1, _ = evaluate(model, val_loader, device, num_classes)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_macroF1 {val_mf1:.3f}",
                flush=True
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_mf1, test_cm = evaluate(model, test_loader, device, num_classes)

    print(f"\n✅ best_val_acc : {best_val_acc:.3f}", flush=True)
    print(f"✅ test_loss    : {test_loss:.4f}", flush=True)
    print(f"✅ test_acc     : {test_acc:.3f}", flush=True)
    print(f"✅ test_macroF1 : {test_mf1:.3f}", flush=True)

    pretty_print_cm(test_cm, class_names=class_names)


if __name__ == "__main__":
    main()