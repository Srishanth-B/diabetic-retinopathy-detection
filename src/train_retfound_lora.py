"""
LoRA (parameter-efficient) fine-tuning for RETFound on APTOS 2019.

Unlike the linear probe (frozen backbone + trainable head only), this injects
low-rank adapters into every attention qkv/proj layer of the ViT-Large
backbone. The pretrained weights stay frozen; only the LoRA adapters (~1.18M)
and the 5-class head (5,125) are trained.

Key difference from train_retfound.py:
  - The backbone runs WITH gradients (no torch.no_grad), so the adapters can
    learn. We therefore call model(images) directly instead of caching frozen
    features. This is heavier on VRAM -> use --batch-size 4 on an 8 GB GPU.

Model selection uses validation quadratic weighted kappa. The test set is
never touched during training.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CLASS_NAMES, load_splits
from src.retfound_model import (
    DEFAULT_CHECKPOINT,
    build_retfound_model,
    count_parameters,
)
# Reuse the exact helpers from the linear-probe script (DRY).
from src.train_retfound import set_seed, calculate_metrics
from src.lora import (
    inject_lora,
    mark_only_lora_and_head_trainable,
    lora_state_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "retfound_lora"

def train_one_epoch(model, loader, device, criterion, optimizer, scaler, use_amp):
    """Train LoRA adapters + head for one epoch (gradients flow through backbone)."""
    model.train()  # activates LoRA dropout + stochastic depth

    total_loss = 0.0
    seen = 0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)              # full compat forward, WITH grad
            loss = criterion(logits, labels)

        # Under fp16 AMP an occasional non-finite loss is normal; the GradScaler
        # skips those steps. We just avoid polluting metrics/backward with them.
        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = calculate_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics

def evaluate(model, loader, device, criterion, use_amp):
    """Evaluate without parameter updates."""
    model.eval()

    total_loss = 0.0
    seen = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite validation logits detected.")

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = calculate_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics

def save_lora_checkpoint(model, args, epoch, validation_metrics, trainable_parameters):
    """Save ONLY the LoRA adapters + head (a few MB), never the 4 GB backbone."""
    state = {k: v.detach().cpu() for k, v in lora_state_dict(model).items()}

    checkpoint = {
        "lora_state_dict": state,
        "model": "RETFound_mae_natureCFP",
        "adaptation": "lora",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "include_mlp": args.include_mlp,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "image_size": args.image_size,
        "epoch": epoch,
        "trainable_parameters": trainable_parameters,
        "validation_metrics": validation_metrics,
        "seed": args.seed,
    }
    torch.save(checkpoint, OUTPUT_DIR / "best_lora.pt")

def save_history(history):
    with (OUTPUT_DIR / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RETFound LoRA training requires CUDA for this project.")

    use_amp = not args.no_amp

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision: {use_amp}")

    train_dataset, val_dataset, test_dataset, class_weights = load_splits(
        image_size=args.image_size,
        seed=args.seed,
    )
    print(
        f"Dataset sizes: train={len(train_dataset)}, "
        f"validation={len(val_dataset)}, test={len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Build with full_finetune so no params are pre-frozen; LoRA injection +
    # mark_only_lora_and_head_trainable then decide exactly what trains.
    model, load_info = build_retfound_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(CLASS_NAMES),
        adaptation="full_finetune",
        image_size=args.image_size,
        drop_path_rate=args.drop_path,
    )

    targets = ["qkv", "proj"]
    if args.include_mlp:
        targets += ["fc1", "fc2"]

    print(f"Injecting LoRA into targets: {targets}")
    inject_lora(
        model,
        target_modules=tuple(targets),
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        verbose=False,  # set True once if you want the full per-layer list
    )
    mark_only_lora_and_head_trainable(model, extra_trainable=("head",))

    model = model.to(device)

    total_parameters, trainable_parameters = count_parameters(model)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,} "
          f"({100 * trainable_parameters / total_parameters:.3f}%)")
    print(f"Missing checkpoint keys: {load_info['missing_keys']}")
    print(f"Unexpected checkpoint key count: {len(load_info['unexpected_keys'])}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    trainable_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_kappa = float("-inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, device, criterion, optimizer, scaler, use_amp,
        )
        validation_metrics = evaluate(model, val_loader, device, criterion, use_amp)

        scheduler.step(validation_metrics["kappa"])

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": elapsed,
        })
        save_history(history)

        print(f" Epoch {epoch:02d}/{args.epochs} | time={elapsed:.1f}s | lr={current_lr:.2e}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"kappa={train_metrics['kappa']:.4f}, "
              f"accuracy={train_metrics['accuracy']:.4f}, "
              f"macro_f1={train_metrics['macro_f1']:.4f}")
        print(f"  Valid: loss={validation_metrics['loss']:.4f}, "
              f"kappa={validation_metrics['kappa']:.4f}, "
              f"accuracy={validation_metrics['accuracy']:.4f}, "
              f"macro_f1={validation_metrics['macro_f1']:.4f}")

        if validation_metrics["kappa"] > best_kappa:
            best_kappa = validation_metrics["kappa"]
            epochs_without_improvement = 0
            save_lora_checkpoint(model, args, epoch, validation_metrics, trainable_parameters)
            print(f"  Saved new best LoRA checkpoint: validation kappa={best_kappa:.4f}")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
            break

    print("RETFound LoRA fine-tuning completed.")
    print(f"Best validation kappa: {best_kappa:.4f}")
    print(f"Best checkpoint: {OUTPUT_DIR / 'best_lora.pt'}")
    print(f"Training history: {OUTPUT_DIR / 'training_history.json'}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of RETFound on APTOS."
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    # LoRA hyperparameters
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--include-mlp", action="store_true",
                        help="Also adapt MLP fc1/fc2 (more capacity, more VRAM).")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed-precision execution.")
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
