import os
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool


# -------------------------
# 0) Loader
# -------------------------
class LoadProcessed(InMemoryDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        meta_path = self.processed_paths[1]
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False)
            self.classes = meta.get("classes", None)
            self.label_map = meta.get("label_map", None)
        else:
            self.classes = None
            self.label_map = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt", "meta.pt"]


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


def stratified_split_indices(
    y: torch.Tensor, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    g = torch.Generator().manual_seed(seed)

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
# Dataset
# -------------------------
class MultiViewRuleDataset(Dataset):
    def __init__(self, rule_ids: List[int], dbg_map: Dict[int, Data], sym_map: Dict[int, Data], dep_map: Dict[int, Data]):
        self.rule_ids = rule_ids
        self.dbg_map = dbg_map
        self.sym_map = sym_map
        self.dep_map = dep_map

    def __len__(self):
        return len(self.rule_ids)

    def __getitem__(self, idx: int):
        rid = self.rule_ids[idx]
        g_dbg = self.dbg_map[rid]
        g_sym = self.sym_map[rid]
        g_dep = self.dep_map[rid]
        y = int(g_dbg.y.item())
        return rid, g_dbg, g_sym, g_dep, y


def collate_multiview(batch):
    rids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    dbg_list = [b[1] for b in batch]
    sym_list = [b[2] for b in batch]
    dep_list = [b[3] for b in batch]
    y = torch.tensor([b[4] for b in batch], dtype=torch.long)

    dbg_batch = Batch.from_data_list(dbg_list)
    sym_batch = Batch.from_data_list(sym_list)
    dep_batch = Batch.from_data_list(dep_list)
    return rids, dbg_batch, sym_batch, dep_batch, y


# -------------------------
# Encoder
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
# Hierarchical Model
# -------------------------
class HierFusionClassifier(nn.Module):
    """
    Stage1: 4-way: 0,1,2,34
    Stage2: binary: 3 vs 4
    """
    def __init__(self, dbg_in_dim, dbg_edge_dim, sym_in_dim, sym_edge_dim, dep_in_dim, dep_edge_dim,
                 hidden=64, dropout=0.2):
        super().__init__()
        self.enc_dbg = GINEEncoder(dbg_in_dim, dbg_edge_dim, hidden=hidden, dropout=dropout)
        self.enc_sym = GINEEncoder(sym_in_dim, sym_edge_dim, hidden=hidden, dropout=dropout)
        self.enc_dep = GINEEncoder(dep_in_dim, dep_edge_dim, hidden=hidden, dropout=dropout)

        fusion_dim = hidden * 3
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_stage1 = nn.Linear(fusion_dim, 4)  # 0,1,2,34
        self.head_stage2 = nn.Linear(fusion_dim, 2)  # 3 vs 4

    def embed(self, dbg_batch, sym_batch, dep_batch):
        h_dbg = self.enc_dbg(dbg_batch)
        h_sym = self.enc_sym(sym_batch)
        h_dep = self.enc_dep(dep_batch)
        h = torch.cat([h_dbg, h_sym, h_dep], dim=-1)
        h = self.fuse_mlp(h)
        return h

    def forward(self, dbg_batch, sym_batch, dep_batch):
        h = self.embed(dbg_batch, sym_batch, dep_batch)
        logits1 = self.head_stage1(h)
        logits2 = self.head_stage2(h)
        return logits1, logits2


# -------------------------
# Metrics / helpers
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


def map_stage1_target(y: torch.Tensor) -> torch.Tensor:
    # 0->0, 1->1, 2->2, 3/4->3
    y1 = y.clone()
    y1 = torch.where(y1 == 3, torch.tensor(3, device=y.device), y1)
    y1 = torch.where(y1 == 4, torch.tensor(3, device=y.device), y1)
    return y1


@torch.no_grad()
def predict_hier(model: HierFusionClassifier, dbg_b, sym_b, dep_b) -> torch.Tensor:
    logits1, logits2 = model(dbg_b, sym_b, dep_b)
    p1 = logits1.argmax(dim=-1)  # 0/1/2/3(=34)
    pred = p1.clone()
    mask = (p1 == 3)
    if mask.any():
        p2 = logits2.argmax(dim=-1)  # 0=>class3, 1=>class4
        pred = pred.clone()
        pred[mask] = torch.where(
            p2[mask] == 0,
            torch.tensor(3, device=pred.device),
            torch.tensor(4, device=pred.device)
        )
    return pred


@torch.no_grad()
def eval_full(model, loader, device, num_classes: int):
    model.eval()
    all_pred, all_true = [], []
    correct, total = 0, 0

    for _, dbg_b, sym_b, dep_b, y in loader:
        dbg_b = dbg_b.to(device)
        sym_b = sym_b.to(device)
        dep_b = dep_b.to(device)
        y = y.to(device)

        pred = predict_hier(model, dbg_b, sym_b, dep_b)
        correct += int((pred == y).sum())
        total += y.size(0)

        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    cm = confusion_matrix(all_pred, all_true, num_classes)
    mf1 = macro_f1_from_cm(cm)
    acc = correct / max(total, 1)
    return acc, mf1, cm


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
# main (2-phase)
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbg-root", default="output/pyg_radius2_debruijn_std")
    parser.add_argument("--sym-root", default="output/pyg_radius2_symbol01_std")
    parser.add_argument("--dep-root", default="output/pyg_radius2_dependency_std")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ckpt", default="hier_stage1.pt")
    parser.add_argument("--patience", type=int, default=15)  # early stopping
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    ds_dbg = LoadProcessed(args.dbg_root)
    ds_sym = LoadProcessed(args.sym_root)
    ds_dep = LoadProcessed(args.dep_root)
    class_names = ds_dbg.classes

    dbg_map = build_rule_index(ds_dbg)
    sym_map = build_rule_index(ds_sym)
    dep_map = build_rule_index(ds_dep)

    rule_ids = intersect_rule_ids(dbg_map, sym_map, dep_map)
    print("aligned rules:", len(rule_ids), flush=True)

    y_all = torch.tensor([int(dbg_map[r].y.item()) for r in rule_ids], dtype=torch.long)
    num_classes = int(torch.unique(y_all).numel())
    print("num_classes:", num_classes, flush=True)

    train_idx, val_idx, test_idx = stratified_split_indices(
        y=y_all, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )

    train_rules = [rule_ids[i] for i in train_idx.tolist()]
    val_rules   = [rule_ids[i] for i in val_idx.tolist()]
    test_rules  = [rule_ids[i] for i in test_idx.tolist()]
    print("split sizes:", len(train_rules), len(val_rules), len(test_rules), flush=True)

    train_counts = count_by_class(y_all[train_idx], num_classes)
    print("class counts (train):", train_counts.tolist(), flush=True)

    # build datasets
    train_set = MultiViewRuleDataset(train_rules, dbg_map, sym_map, dep_map)
    val_set   = MultiViewRuleDataset(val_rules, dbg_map, sym_map, dep_map)
    test_set  = MultiViewRuleDataset(test_rules, dbg_map, sym_map, dep_map)

    # dims
    r0 = rule_ids[0]
    dbg_in_dim = dbg_map[r0].x.size(-1)
    dbg_edge_dim = dbg_map[r0].edge_attr.size(-1)
    sym_in_dim = sym_map[r0].x.size(-1)
    sym_edge_dim = sym_map[r0].edge_attr.size(-1)
    dep_in_dim = dep_map[r0].x.size(-1)
    dep_edge_dim = dep_map[r0].edge_attr.size(-1)
    print("dims:")
    print("  dbg in/edge:", dbg_in_dim, dbg_edge_dim)
    print("  sym in/edge:", sym_in_dim, sym_edge_dim)
    print("  dep in/edge:", dep_in_dim, dep_edge_dim, flush=True)

    model = HierFusionClassifier(
        dbg_in_dim, dbg_edge_dim,
        sym_in_dim, sym_edge_dim,
        dep_in_dim, dep_edge_dim,
        hidden=args.hidden,
        dropout=0.2
    ).to(device)

    # -----------------------
    # Phase 1: train Stage1 only (NO sampler)
    # -----------------------
    if args.phase == 1:
        # Stage1 class weight (on stage1 labels: 0,1,2,34)
        y1_train = torch.tensor([int(dbg_map[r].y.item()) for r in train_rules], dtype=torch.long)
        y1_train = map_stage1_target(y1_train)
        stage1_counts = torch.bincount(y1_train, minlength=4).float()
        w1 = (stage1_counts.sum() / (stage1_counts + 1e-6))
        w1 = (w1 / w1.mean()).to(device)
        print("stage1_weight:", [round(x, 3) for x in w1.detach().cpu().tolist()], flush=True)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multiview
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiview
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiview
        )

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_val_mf1 = -1.0
        best_state = None
        bad = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, total = 0.0, 0

            for _, dbg_b, sym_b, dep_b, y in train_loader:
                dbg_b = dbg_b.to(device)
                sym_b = sym_b.to(device)
                dep_b = dep_b.to(device)
                y = y.to(device)

                optim.zero_grad()
                logits1, logits2 = model(dbg_b, sym_b, dep_b)
                y1 = map_stage1_target(y)
                loss = F.cross_entropy(logits1, y1, weight=w1)  # ✅ stage1 only
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                loss_sum += loss.detach().item() * y.size(0)
                total += y.size(0)

            train_loss = loss_sum / max(total, 1)
            val_acc, val_mf1, _ = eval_full(model, val_loader, device, num_classes)

            if val_mf1 > best_val_mf1:
                best_val_mf1 = val_mf1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_acc {val_acc:.3f} | val_macroF1 {val_mf1:.3f}", flush=True)

            if bad >= args.patience:
                print(f"Early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(model.state_dict(), args.ckpt)
        print("saved ckpt:", args.ckpt, flush=True)

        test_acc, test_mf1, test_cm = eval_full(model, test_loader, device, num_classes)
        print(f"\n✅ phase1 best_val_macroF1 : {best_val_mf1:.3f}", flush=True)
        print(f"✅ phase1 test_acc         : {test_acc:.3f}", flush=True)
        print(f"✅ phase1 test_macroF1      : {test_mf1:.3f}", flush=True)
        pretty_print_cm(test_cm, class_names)

        return

    # -----------------------
    # Phase 2: train Stage2 only (freeze encoder+stage1), only y in {3,4}
    # -----------------------
    if args.phase == 2:
        assert os.path.exists(args.ckpt), f"ckpt not found: {args.ckpt}"
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.to(device)

        # freeze everything except stage2 head
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head_stage2.parameters():
            p.requires_grad = True

        # build dataset of only 3/4
        train_rules_34 = [r for r in train_rules if int(dbg_map[r].y.item()) in (3, 4)]
        val_rules_34   = [r for r in val_rules   if int(dbg_map[r].y.item()) in (3, 4)]
        print("phase2 train_34:", len(train_rules_34), "val_34:", len(val_rules_34), flush=True)

        train_set_34 = MultiViewRuleDataset(train_rules_34, dbg_map, sym_map, dep_map)
        val_set_34   = MultiViewRuleDataset(val_rules_34, dbg_map, sym_map, dep_map)
        test_loader  = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiview)

        # sampler to balance 3 vs 4
        y34 = torch.tensor([int(dbg_map[r].y.item()) for r in train_rules_34], dtype=torch.long)
        w34 = torch.where(y34 == 3, torch.tensor(1.0), torch.tensor(1.0)).double()
        gen = torch.Generator().manual_seed(args.seed)
        sampler34 = WeightedRandomSampler(w34, num_samples=len(w34), replacement=True, generator=gen)

        train_loader_34 = torch.utils.data.DataLoader(
            train_set_34, batch_size=args.batch_size, shuffle=False, sampler=sampler34, collate_fn=collate_multiview
        )
        val_loader_34 = torch.utils.data.DataLoader(
            val_set_34, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiview
        )

        optim = torch.optim.AdamW(model.head_stage2.parameters(), lr=args.lr, weight_decay=1e-4)

        best_val_mf1 = -1.0
        best_state = None
        bad = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, total = 0.0, 0

            for _, dbg_b, sym_b, dep_b, y in train_loader_34:
                dbg_b = dbg_b.to(device)
                sym_b = sym_b.to(device)
                dep_b = dep_b.to(device)
                y = y.to(device)

                optim.zero_grad()
                logits1, logits2 = model(dbg_b, sym_b, dep_b)

                mask = (y == 3) | (y == 4)
                y2 = torch.where(y[mask] == 3, torch.tensor(0, device=device), torch.tensor(1, device=device))
                loss = F.cross_entropy(logits2[mask], y2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head_stage2.parameters(), 1.0)
                optim.step()

                loss_sum += loss.detach().item() * y.size(0)
                total += y.size(0)

            # evaluate full task (important) + optional evaluate on 3/4 only
            val_acc_full, val_mf1_full, _ = eval_full(model, test_loader if len(val_rules_34) == 0 else test_loader, device, num_classes)
            # better: evaluate on val_loader_34 by checking if stage1 routes to 34, but we keep it simple and use full-macroF1 on val_set (small anyway)
            # We'll compute full val on original val_set for early stopping:
            val_loader_full = torch.utils.data.DataLoader(
                val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiview
            )
            val_acc, val_mf1, _ = eval_full(model, val_loader_full, device, num_classes)

            if val_mf1 > best_val_mf1:
                best_val_mf1 = val_mf1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if epoch % 10 == 0 or epoch == 1:
                train_loss = loss_sum / max(total, 1)
                print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_acc {val_acc:.3f} | val_macroF1 {val_mf1:.3f}", flush=True)

            if bad >= args.patience:
                print(f"Early stop at epoch {epoch} (patience={args.patience})", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # final test
        test_acc, test_mf1, test_cm = eval_full(model, test_loader, device, num_classes)
        print(f"\n✅ phase2 best_val_macroF1 : {best_val_mf1:.3f}", flush=True)
        print(f"✅ phase2 test_acc         : {test_acc:.3f}", flush=True)
        print(f"✅ phase2 test_macroF1      : {test_mf1:.3f}", flush=True)
        pretty_print_cm(test_cm, class_names)


if __name__ == "__main__":
    main()
