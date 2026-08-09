"""
Two-phase fine-tuning training script for DR classification.

Phase 1 (warm-up): freeze the pretrained backbone, train only the new
    classifier head. This lets the randomly-initialized head learn
    sensible weights WITHOUT sending destructive gradients back into
    the pretrained features.

Phase 2 (fine-tune): unfreeze the whole network and train at a LOW
    learning rate so the pretrained features adapt gently to fundus images.

The best checkpoint (by validation quadratic weighted kappa) across BOTH
phases is saved to checkpoints/best_model.pt.

Usage:
    python -m src.train --backbone convnextv2_tiny
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score

from src.dataset import load_splits, CLASS_NAMES
from src.model import build_model, set_backbone_trainable, count_trainable_params

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

def evaluate(model, loader, device, criterion):
    """Run the model over a loader and compute loss + metrics (no grad)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return {"loss": avg_loss, "kappa": kappa, "macro_f1": macro_f1, "accuracy": acc}

def run_epoch(model, loader, device, criterion, optimizer):
    """One full training pass over the training set."""
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def save_checkpoint(model, args, best_kappa):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": args.backbone,
            "image_size": args.image_size,
            "class_names": CLASS_NAMES,
            "val_kappa": best_kappa,
        },
        CHECKPOINT_DIR / "best_model.pt",
    )

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, class_weights = load_splits(image_size=args.image_size, seed=args.seed)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(backbone=args.backbone, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_kappa = -1.0
    history = []

    # ---------- PHASE 1: warm-up (head only) ----------
    set_backbone_trainable(model, trainable=False)
    print(f"=== PHASE 1: warm-up (head only) === trainable params: {count_trainable_params(model):,}")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.head_lr, weight_decay=args.weight_decay,
    )
    for epoch in range(1, args.phase1_epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - start
        print(f"[P1 {epoch:02d}/{args.phase1_epochs}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_kappa={val_metrics['kappa']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} {elapsed:.1f}s")
        history.append({"phase": 1, "epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["kappa"] > best_kappa:
            best_kappa = val_metrics["kappa"]
            save_checkpoint(model, args, best_kappa)
            print(f"  -> saved best (val_kappa={best_kappa:.4f})")

    # ---------- PHASE 2: full fine-tune (low LR) ----------
    set_backbone_trainable(model, trainable=True)
    print(f"=== PHASE 2: full fine-tune (low LR) === trainable params: {count_trainable_params(model):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    for epoch in range(1, args.phase2_epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["kappa"])
        elapsed = time.time() - start
        print(f"[P2 {epoch:02d}/{args.phase2_epochs}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_kappa={val_metrics['kappa']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} {elapsed:.1f}s")
        history.append({"phase": 2, "epoch": epoch, "train_loss": train_loss, **val_metrics})
        if val_metrics["kappa"] > best_kappa:
            best_kappa = val_metrics["kappa"]
            save_checkpoint(model, args, best_kappa)
            print(f"  -> saved best (val_kappa={best_kappa:.4f})")

    with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best val kappa: {best_kappa:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR / 'best_model.pt'}")

def parse_args():
    p = argparse.ArgumentParser(description="Two-phase fine-tuning for DR classification")
    p.add_argument("--backbone", type=str, default="convnextv2_tiny")
    p.add_argument("--phase1-epochs", type=int, default=3)
    p.add_argument("--phase2-epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)
