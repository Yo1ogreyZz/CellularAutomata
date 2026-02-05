import os
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool


# -------------------------
# 0) 通用 loader：读取 processed/data.pt + meta.pt
# -------------------------
class LoadProcessed(InMemoryDataset):
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
# 1) 把 DataList 按 rule_id 建索引
# -------------------------
def build_rule_index(ds: InMemoryDataset) -> Dict[int, Data]:
    m = {}
    for g in ds:
        rid = int(g.rule_id.item()) if torch.is_tensor(g.rule_id) else int(g.rule_id)
        m[rid] = g
    return m


def intersect_rule_ids(*maps: Dict[int, Data]) -> List[int]:
    s = set(maps[0].keys())
    for mp in maps[1:]:
        s &= set(mp.keys())
    return sorted(list(s))


# -------------------------
# 2) 分层切分（按 y）
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
# 3) 多视图 Dataset：每个样本是同一个 rule_id 的多张图
# -------------------------
class MultiViewRuleDataset(Dataset):
    def __init__(self, rule_ids: List[int], view_maps: List[Dict[int, Data]]):
        self.rule_ids = rule_ids
        self.view_maps = view_maps

    def __len__(self):
        return len(self.rule_ids)

    def __getitem__(self, idx: int):
        rid = self.rule_ids[idx]
        views = [view_map[rid] for view_map in self.view_maps]
        y = int(views[0].y.item())
        return rid, views, y


def collate_multiview(batch):
    rids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)

    num_views = len(batch[0][1])
    view_lists = [[] for _ in range(num_views)]
    for _, views, _ in batch:
        for i, g in enumerate(views):
            view_lists[i].append(g)

    view_batches = [Batch.from_data_list(v) for v in view_lists]
    return rids, view_batches, y


# -------------------------
# 4) Encoder：GINEConv + pooling -> 图 embedding
# -------------------------
class GINEEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, dropout=0.2):
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

    def forward(self, batch_data: Batch):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch

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
        return g


# -------------------------
# 5) Fusion 模型：多 encoder -> concat -> MLP 分类
# -------------------------
class MultiViewFusionClassifier(nn.Module):
    def __init__(self, view_dims: List[Tuple[int, int]], hidden=64, num_classes=8, dropout=0.2):
        super().__init__()

        self.encoders = nn.ModuleList([
            GINEEncoder(in_dim, edge_dim, hidden=hidden, dropout=dropout)
            for in_dim, edge_dim in view_dims
        ])

        fusion_dim = hidden * len(view_dims)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, view_batches: List[Batch]):
        embeddings = [enc(batch) for enc, batch in zip(self.encoders, view_batches)]
        h = torch.cat(embeddings, dim=-1)
        return self.classifier(h)


# -------------------------
# 6) 指标：confusion matrix + macro-F1
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
def evaluate(model, loader, device, num_classes: int, class_weight=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_pred, all_true = [], []

    for _, view_batches, y in loader:
        view_batches = [vb.to(device) for vb in view_batches]
        y = y.to(device)

        logits = model(view_batches)
        loss = F.cross_entropy(logits, y, weight=class_weight)
        loss_sum += loss.detach().item() * y.size(0)

        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.size(0)

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
# 7) main：多视图融合训练（balanced sampler + macro-F1 early stop）
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view-roots",
        nargs="+",
        default=[
            "output/pyg_radius2_debruijn_std",
            "output/pyg_radius2_symbol01_std",
            "output/pyg_radius2_dependency_std",
        ],
    )
    parser.add_argument(
        "--view-names",
        nargs="+",
        default=["dbg", "sym", "dep"],
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class-weight-power", type=float, default=0.4)
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--lr-factor", type=float, default=0.6)
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    args = parser.parse_args()

    if args.view_names and len(args.view_names) != len(args.view_roots):
        raise ValueError("--view-names must have the same length as --view-roots")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    datasets = [LoadProcessed(root) for root in args.view_roots]
    view_names = args.view_names or [os.path.basename(root) for root in args.view_roots]

    class_names = datasets[0].classes

    view_maps = [build_rule_index(ds) for ds in datasets]
    rule_ids = intersect_rule_ids(*view_maps)
    print("aligned rules:", len(rule_ids), flush=True)

    y_all = torch.tensor([int(view_maps[0][r].y.item()) for r in rule_ids], dtype=torch.long)
    num_classes = int(torch.unique(y_all).numel())
    print("num_classes:", num_classes, flush=True)

    train_idx, val_idx, test_idx = stratified_split_indices(
        y=y_all, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    train_rules = [rule_ids[i] for i in train_idx.tolist()]
    val_rules = [rule_ids[i] for i in val_idx.tolist()]
    test_rules = [rule_ids[i] for i in test_idx.tolist()]

    print("split sizes:", len(train_rules), len(val_rules), len(test_rules), flush=True)

    train_counts = count_by_class(y_all[train_idx], num_classes)
    val_counts = count_by_class(y_all[val_idx], num_classes)
    test_counts = count_by_class(y_all[test_idx], num_classes)
    print("class counts (train):", train_counts.tolist(), flush=True)
    print("class counts (val)  :", val_counts.tolist(), flush=True)
    print("class counts (test) :", test_counts.tolist(), flush=True)

    counts = train_counts.float()
    class_weight = (counts.sum() / (counts + 1e-6)) ** args.class_weight_power
    class_weight = class_weight / class_weight.mean()
    class_weight = class_weight.to(device)
    print("class_weight(invfreq^power):", [round(x, 3) for x in class_weight.detach().cpu().tolist()], flush=True)

    train_set = MultiViewRuleDataset(train_rules, view_maps)
    val_set = MultiViewRuleDataset(val_rules, view_maps)
    test_set = MultiViewRuleDataset(test_rules, view_maps)

    sampler = None
    shuffle = True
    if args.balanced_sampler:
        sample_weights = class_weight.detach().cpu()[y_all[train_idx]].double()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
        print("balanced sampler: on", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_multiview
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiview
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiview
    )

    r0 = rule_ids[0]
    view_dims = []
    for name, view_map in zip(view_names, view_maps):
        in_dim = view_map[r0].x.size(-1)
        edge_dim = view_map[r0].edge_attr.size(-1)
        view_dims.append((in_dim, edge_dim))
        print(f"dims({name}) in/edge:", in_dim, edge_dim)

    model = MultiViewFusionClassifier(
        view_dims,
        hidden=args.hidden,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr
    )

    best_val_mf1 = -1.0
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, total = 0.0, 0

        for _, view_batches, y in train_loader:
            view_batches = [vb.to(device) for vb in view_batches]
            y = y.to(device)

            optim.zero_grad()
            logits = model(view_batches)
            loss = F.cross_entropy(logits, y, weight=class_weight)
            loss.backward()
            optim.step()

            loss_sum += loss.detach().item() * y.size(0)
            total += y.size(0)

        train_loss = loss_sum / max(total, 1)
        val_loss, val_acc, val_mf1, _ = evaluate(model, val_loader, device, num_classes, class_weight)
        scheduler.step(val_mf1)

        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        elif epoch >= args.min_epochs:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_macroF1 {val_mf1:.3f}",
                flush=True
            )

        if epoch >= args.min_epochs and patience_count >= args.patience:
            print(f"[Early stop] epoch {epoch} (patience={args.patience}, min_epochs={args.min_epochs})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_mf1, test_cm = evaluate(model, test_loader, device, num_classes, class_weight)

    print(f"\n✅ best_val_macroF1 : {best_val_mf1:.3f}", flush=True)
    print(f"✅ test_loss        : {test_loss:.4f}", flush=True)
    print(f"✅ test_acc         : {test_acc:.3f}", flush=True)
    print(f"✅ test_macroF1     : {test_mf1:.3f}", flush=True)

    pretty_print_cm(test_cm, class_names=class_names)


if __name__ == "__main__":
    main()
