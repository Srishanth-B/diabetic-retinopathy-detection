"""
Linear-probe training for RETFound on APTOS 2019.

The RETFound ViT-Large backbone remains frozen. Only the newly initialized
five-class classifier head is trained. The frozen backbone still processes
randomly augmented fundus images each epoch, providing different features
without updating its pretrained retinal representations.

The best classifier head is selected using validation quadratic weighted
kappa. The held-out test set is not used during training.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score
from torch.utils.data import DataLoader

from src.dataset import CLASS_NAMES, load_splits
from src.retfound_model import (
    DEFAULT_CHECKPOINT,
    build_retfound_model,
    count_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "retfound_linear_probe"

def set_seed(seed: int) -> None:
    """Configure reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_features(model, images, device, use_amp):
    """Run the frozen RETFound backbone without creating gradients."""
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            features = model.forward_features(images)

    # RETFound global pooling returns [batch, 1, 1024].
    if features.ndim == 3:
        if features.shape[1] != 1:
            raise RuntimeError(
                f"Unexpected RETFound feature shape: {tuple(features.shape)}"
            )

        features = features[:, 0, :]

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2D features, received: {tuple(features.shape)}"
        )

    if not torch.isfinite(features).all():
        raise FloatingPointError(
            "RETFound produced non-finite features."
        )

    return features

def calculate_metrics(labels, predictions):
    """Calculate classification metrics from complete epoch outputs."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    accuracy = float(np.mean(labels == predictions))

    macro_f1 = float(
        f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    kappa = float(
        cohen_kappa_score(
            labels,
            predictions,
            weights="quadratic",
        )
    )

    # This can occur if every prediction belongs to one class.
    if not np.isfinite(kappa):
        kappa = 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "kappa": kappa,
    }

def train_one_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scaler,
    use_amp,
):
    """Train only the RETFound classification head for one epoch."""
    # Keep the frozen backbone deterministic while allowing the head to train.
    model.eval()
    model.head.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        features = extract_features(
            model,
            images,
            device,
            use_amp,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model.head(features)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss detected: {loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics

def evaluate(
    model,
    loader,
    device,
    criterion,
    use_amp,
):
    """Evaluate the linear probe without parameter updates."""
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            features = extract_features(
                model,
                images,
                device,
                use_amp,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model.head(features)
                loss = criterion(logits, labels)

            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "Non-finite validation logits detected."
                )

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics

def save_head_checkpoint(
    model,
    args,
    epoch,
    validation_metrics,
):
    """Save only the small trained classifier, not the 4 GB backbone."""
    head_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.head.state_dict().items()
    }

    checkpoint = {
        "head_state_dict": head_state,
        "model": "RETFound_mae_natureCFP",
        "adaptation": "linear_probe",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "image_size": args.image_size,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "seed": args.seed,
    }

    torch.save(
        checkpoint,
        OUTPUT_DIR / "best_head.pt",
    )

def save_history(history):
    """Persist training history after every epoch."""
    history_path = OUTPUT_DIR / "training_history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)

def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "RETFound training requires CUDA for this project."
        )

    use_amp = not args.no_amp

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision: {use_amp}")

    train_dataset, val_dataset, test_dataset, class_weights = (
        load_splits(
            image_size=args.image_size,
            seed=args.seed,
        )
    )

    print(
        f"Dataset sizes: train={len(train_dataset)}, "
        f"validation={len(val_dataset)}, "
        f"test={len(test_dataset)}"
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

    model, load_info = build_retfound_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(CLASS_NAMES),
        adaptation="linear_probe",
        image_size=args.image_size,
    )

    model = model.to(device)

    total_parameters, trainable_parameters = count_parameters(model)

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Missing checkpoint keys: {load_info['missing_keys']}")
    print(
        f"Unexpected checkpoint key count: "
        f"{len(load_info['unexpected_keys'])}"
    )

    if trainable_parameters != 5_125:
        raise RuntimeError(
            "Expected 5,125 trainable classifier parameters, "
            f"but found {trainable_parameters:,}."
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    trainable_parameters_list = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_kappa = float("-inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            use_amp,
        )

        validation_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            use_amp,
        )

        scheduler.step(validation_metrics["kappa"])

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": elapsed,
        }

        history.append(record)
        save_history(history)

        print(
            f" Epoch {epoch:02d}/{args.epochs} | "
            f"time={elapsed:.1f}s | lr={current_lr:.2e}"
        )

        print(f"  Train: "
            f"loss={train_metrics['loss']:.4f}, "
            f"kappa={train_metrics['kappa']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}, "
            f"macro_f1={train_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Valid: "
            f"loss={validation_metrics['loss']:.4f}, "
            f"kappa={validation_metrics['kappa']:.4f}, "
            f"accuracy={validation_metrics['accuracy']:.4f}, "
            f"macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["kappa"] > best_kappa:
            best_kappa = validation_metrics["kappa"]
            epochs_without_improvement = 0

            save_head_checkpoint(
                model,
                args,
                epoch,
                validation_metrics,
            )

            print(
                f"  Saved new best head: "
                f"validation kappa={best_kappa:.4f}"
            )
        else:
            epochs_without_improvement += 1

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                f"Early stopping after "
                f"{epochs_without_improvement} epochs "
                f"without improvement."
            )
            break

    print(
        "RETFound linear-probe training completed."
    )
    print(f"Best validation kappa: {best_kappa:.4f}")
    print(
        f"Best classifier: "
        f"{OUTPUT_DIR / 'best_head.pt'}"
    )
    print(
        f"Training history: "
        f"{OUTPUT_DIR / 'training_history.json'}"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a linear classifier on frozen RETFound features."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed-precision execution.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
