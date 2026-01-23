import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from gnn.dataset import (
    CAMultiViewDataset, create_splits, create_stratified_splits,
    custom_collate_fn, get_class_weights, CLASS_NAMES
)
from gnn.model import MultiViewGAT


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-View GAT (3 views)')
    
    # Data - supports both CSV and .pt cache
    parser.add_argument('--data_path', type=str, required=True, help='CSV or .pt file')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stratified', action='store_true', help='Stratified split')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Graph params (only for CSV, ignored for .pt)
    parser.add_argument('--dependency_T', type=int, default=4)
    parser.add_argument('--dependency_W', type=int, default=7)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--class_weights', action='store_true', help='Use class weights')
    
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=15)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_every', type=int, default=10)

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch in tqdm(loader, desc='Train', leave=False):
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        logits = model(batch)
        
        if class_weights is not None:
            loss = F.cross_entropy(logits, batch['label'], weight=class_weights)
        else:
            loss = F.cross_entropy(logits, batch['label'])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch['label'].size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == batch['label']).sum().item()
        total += batch['label'].size(0)
    
    return {'loss': total_loss / total, 'accuracy': correct / total}


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        logits = model(batch)
        
        if class_weights is not None:
            loss = F.cross_entropy(logits, batch['label'], weight=class_weights)
        else:
            loss = F.cross_entropy(logits, batch['label'])
        
        total_loss += loss.item() * batch['label'].size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == batch['label']).sum().item()
        total += batch['label'].size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())
    
    return {
        'loss': total_loss / total, 
        'accuracy': correct / total,
        'preds': np.array(all_preds),
        'labels': np.array(all_labels)
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    
    # For .pt files, we need a CSV for splitting (or split the .pt directly)
    is_cached = args.data_path.endswith('.pt')
    
    if is_cached:
        samples = torch.load(args.data_path, weights_only=False)
        n = len(samples)
        indices = np.arange(n)
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    else:
        if args.stratified:
            train_idx, val_idx, test_idx = create_stratified_splits(
                args.data_path, args.train_ratio, args.val_ratio, args.seed
            )
        else:
            train_idx, val_idx, test_idx = create_splits(
                args.data_path, args.train_ratio, args.val_ratio, args.seed
            )
    
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    ds_kwargs = {'dependency_T': args.dependency_T, 'dependency_W': args.dependency_W}
    
    train_dataset = CAMultiViewDataset(args.data_path, split_indices=train_idx, **ds_kwargs)
    val_dataset = CAMultiViewDataset(args.data_path, split_indices=val_idx, **ds_kwargs)
    test_dataset = CAMultiViewDataset(args.data_path, split_indices=test_idx, **ds_kwargs)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=custom_collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=custom_collate_fn, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=args.num_workers
    )
    
    class_weights = None
    if args.class_weights:
        class_weights = get_class_weights(args.data_path).to(device)
        print(f"Class weights: {class_weights.cpu().numpy().round(3)}")
    
    model = MultiViewGAT(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_classes=len(CLASS_NAMES),
        heads=args.heads,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0
    patience = 0
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = evaluate(model, val_loader, device, class_weights)
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_metrics['loss']:.4f} / {train_metrics['accuracy']:.4f} | "
              f"Val: {val_metrics['loss']:.4f} / {val_metrics['accuracy']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            patience += 1
            if patience >= args.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'ckpt_e{epoch}.pt'))
    
    # Final test
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n=== Results ===")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_metrics['accuracy']:.4f}")
    
    preds, labels = test_metrics['preds'], test_metrics['labels']
    print("\nPer-class:")
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
            print(f"  {name}: {acc:.4f} (n={mask.sum()})")
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))


if __name__ == "__main__":
    main()
