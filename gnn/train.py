"""
Training script for Multi-View GNN on CA classification.

Supports:
- Standard supervised learning on labeled data
- Optional consistency regularization
- Checkpoint saving and resuming
- Detailed logging and evaluation
"""

import os
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from gnn.dataset import CAMultiViewDataset, create_splits, custom_collate_fn
from gnn.model import MultiViewGNN, MultiViewGNN_WithConsistency, consistency_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-View GNN for CA Classification')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                        default='data/benchmark/generated_dataset_100rules_10seeds.csv',
                        help='Path to CSV file')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for GNN layers')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension for each view')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Graph parameters
    parser.add_argument('--lattice_N', type=int, default=8,
                        help='Lattice size for view 1 (default 8)')
    parser.add_argument('--dependency_T', type=int, default=4,
                        help='Time steps for dependency graph (default 4)')
    parser.add_argument('--dependency_W', type=int, default=7,
                        help='Width for dependency graph (default 7)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Consistency regularization
    parser.add_argument('--use_consistency', action='store_true',
                        help='Use consistency regularization')
    parser.add_argument('--consistency_weight', type=float, default=0.1,
                        help='Weight for consistency loss')
    parser.add_argument('--consistency_warmup', type=int, default=10,
                        help='Epochs before applying consistency loss')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience')
    
    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs/multi_view',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging
    parser.add_argument('--log_every', type=int, default=5,
                        help='Log every N epochs')
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device, args, epoch):
    """Train for one epoch."""
    from typing import Dict, Any
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_cons_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.use_consistency:
            logits, individual_logits = model(batch, return_individual=True)
            
            # Classification loss
            cls_loss = F.cross_entropy(logits, batch['label'])
            
            # Consistency loss (with warmup)
            if epoch >= args.consistency_warmup:
                cons_loss = consistency_loss(individual_logits)
                loss = cls_loss + args.consistency_weight * cons_loss
                # cons_loss is already a scalar tensor or float
                cons_loss_value = cons_loss.item() if isinstance(cons_loss, torch.Tensor) else cons_loss
                total_cons_loss += cons_loss_value * batch['label'].size(0)
            else:
                loss = cls_loss
                cons_loss = torch.tensor(0.0)
        else:
            logits = model(batch)
            cls_loss = F.cross_entropy(logits, batch['label'])
            loss = cls_loss
            cons_loss = torch.tensor(0.0)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * batch['label'].size(0)
        total_cls_loss += cls_loss.item() * batch['label'].size(0)
        
        pred = logits.argmax(dim=1)
        correct += (pred == batch['label']).sum().item()
        total += batch['label'].size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    metrics: Dict[str, float] = {
        'loss': total_loss / total,
        'cls_loss': total_cls_loss / total,
        'cons_loss': total_cons_loss / total if args.use_consistency else 0.0,
        'accuracy': correct / total
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, args, return_predictions=False):
    """Evaluate on validation/test set."""
    from typing import Dict, Any, Tuple, List, Optional
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_rule_ids: List[int] = []
    
    for batch in tqdm(loader, desc='Evaluating'):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        if args.use_consistency:
            logits, _ = model(batch, return_individual=False)
        else:
            logits = model(batch)
        
        loss = F.cross_entropy(logits, batch['label'])
        
        # Statistics
        total_loss += loss.item() * batch['label'].size(0)
        
        pred = logits.argmax(dim=1)
        correct += (pred == batch['label']).sum().item()
        total += batch['label'].size(0)
        
        # Collect predictions
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch['label'].cpu().numpy())
        all_rule_ids.extend(batch['rule_id'].cpu().numpy())
    
    metrics: Dict[str, float] = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    if return_predictions:
        return metrics, all_preds, all_labels, all_rule_ids
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, filename='checkpoint.pt'):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'args': vars(args)
    }
    
    filepath = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Resumed from epoch {checkpoint['epoch']}")
    print(f"Previous metrics: {checkpoint['metrics']}")
    
    return checkpoint['epoch'], checkpoint['metrics']


def print_classification_report(labels, preds, class_names):
    """Print detailed classification metrics."""
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    
    # Get unique labels present in the data
    unique_labels = np.unique(np.concatenate([labels, preds]))
    present_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    
    print(f"Present classes: {present_class_names}")
    print(f"Unique labels in data: {unique_labels}")
    
    # Use labels parameter to specify which classes to include
    report = classification_report(
        labels, preds, 
        labels=unique_labels,
        target_names=present_class_names,
        digits=4
    )
    print(report)
    
    print("\nCONFUSION MATRIX")
    print("-"*80)
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    
    # Print header
    true_pred_label = 'True\\Pred'
    print(f"{true_pred_label:<15}", end='')
    for name in present_class_names:
        print(f"{name[:10]:>12}", end='')
    print()
    
    # Print matrix
    for i, label_idx in enumerate(unique_labels):
        name = class_names[label_idx] if label_idx < len(class_names) else f"Class {label_idx}"
        print(f"{name[:15]:<15}", end='')
        for j, _ in enumerate(unique_labels):
            print(f"{cm[i,j]:>12}", end='')
        print()
    print("="*80 + "\n")


def main():
    args = parse_args()
    
    # Set up
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    args_file = os.path.join(args.output_dir, 'args.json')
    with open(args_file, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)  # type: ignore
    
    # Load data
    print("\nLoading data...")
    train_idx, val_idx, test_idx = create_splits(
        args.data_path, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    train_dataset = CAMultiViewDataset(
        args.data_path, 
        split_indices=train_idx,
        lattice_N=args.lattice_N,
        dependency_T=args.dependency_T,
        dependency_W=args.dependency_W
    )
    
    val_dataset = CAMultiViewDataset(
        args.data_path, 
        split_indices=val_idx,
        lattice_N=args.lattice_N,
        dependency_T=args.dependency_T,
        dependency_W=args.dependency_W
    )
    
    test_dataset = CAMultiViewDataset(
        args.data_path, 
        split_indices=test_idx,
        lattice_N=args.lattice_N,
        dependency_T=args.dependency_T,
        dependency_W=args.dependency_W
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    if args.use_consistency:
        model = MultiViewGNN_WithConsistency(
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_classes=len(CAMultiViewDataset.CLASS_NAMES),
            dropout=args.dropout
        )
    else:
        model = MultiViewGNN(
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_classes=len(CAMultiViewDataset.CLASS_NAMES),
            dropout=args.dropout
        )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    else:
        scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0
    
    if args.resume:
        start_epoch, prev_metrics = load_checkpoint(model, optimizer, scheduler, args.resume)
        best_val_acc = prev_metrics.get('val_accuracy', 0)
        start_epoch += 1
    
    # Training loop
    print("\nStarting training...")
    print(f"Consistency regularization: {args.use_consistency}")
    if args.use_consistency:
        print(f"  Weight: {args.consistency_weight}, Warmup: {args.consistency_warmup} epochs")
    
    patience_counter = 0
    history: dict[str, list[float]] = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, args)
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau' and isinstance(scheduler, ReduceLROnPlateau):
                # ReduceLROnPlateau.step() expects the metric value
                scheduler.step(val_metrics['accuracy'])  # type: ignore
            elif isinstance(scheduler, (CosineAnnealingLR, ReduceLROnPlateau)):
                # CosineAnnealingLR.step() takes no arguments
                scheduler.step()  # type: ignore
            else:
                scheduler.step()  # type: ignore
        
        # Log
        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss: float = train_metrics['loss']  # type: ignore
            train_acc: float = train_metrics['accuracy']  # type: ignore
            val_loss: float = val_metrics['loss']  # type: ignore
            val_acc: float = val_metrics['accuracy']  # type: ignore
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            if args.use_consistency and epoch >= args.consistency_warmup:
                print(f"    Cls Loss: {train_metrics['cls_loss']:.4f}, Cons Loss: {train_metrics['cons_loss']:.4f}")  # type: ignore
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])  # type: ignore
        history['train_acc'].append(train_metrics['accuracy'])  # type: ignore
        history['val_loss'].append(val_metrics['loss'])  # type: ignore
        history['val_acc'].append(val_metrics['accuracy'])  # type: ignore
        
        # Save best model
        val_acc_value: float = val_metrics['accuracy']  # type: ignore
        if val_acc_value > best_val_acc:
            best_val_acc = val_acc_value
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                args, filename='best_model.pt'
            )
            print(f"  ★ New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                args, filename=f'checkpoint_epoch{epoch}.pt'
            )
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        {'train': train_metrics, 'val': val_metrics},
        args, filename='final_model.pt'
    )
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_preds, test_labels, test_rule_ids = evaluate(
        model, test_loader, device, args, return_predictions=True
    )
    
    test_acc: float = test_metrics['accuracy']  # type: ignore
    test_loss: float = test_metrics['loss']  # type: ignore
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Detailed classification report
    print_classification_report(
        test_labels, test_preds, 
        CAMultiViewDataset.CLASS_NAMES
    )
    
    # Save predictions
    predictions = {
        'rule_ids': [int(x) for x in test_rule_ids],
        'true_labels': [int(x) for x in test_labels],
        'predictions': [int(x) for x in test_preds],
        'class_names': CAMultiViewDataset.CLASS_NAMES
    }
    
    with open(os.path.join(args.output_dir, 'test_predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\nAll results saved to {args.output_dir}")
    print("\nTraining complete! ✓")


if __name__ == "__main__":
    main()