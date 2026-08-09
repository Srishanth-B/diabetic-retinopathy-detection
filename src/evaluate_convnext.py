"""
Held-out test evaluation for ConvNeXt V2-Tiny.

Uses the same APTOS split, class order, metrics, and artifact format as
the RETFound evaluation to enable a controlled comparison.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.dataset import CLASS_NAMES, load_splits
from src.model import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "checkpoints" / "convnextv2_tiny"
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_inference(model, loader, device, use_amp):
    """Evaluate ConvNeXt over the held-out test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "ConvNeXt produced non-finite logits."
                )

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )
            predictions = probabilities.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
        total_loss / len(loader.dataset),
    )

def save_confusion_matrices(
    labels,
    predictions,
    output_path,
):
    """Save raw and row-normalized confusion matrices."""
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=range(len(CLASS_NAMES)),
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title(
        "ConvNeXt V2-Tiny Test Confusion Matrix"
    )
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")

    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title(
        "ConvNeXt V2-Tiny Normalized Confusion Matrix"
    )
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("True class")

    for axis in axes:
        axis.tick_params(
            axis="x",
            rotation=30,
        )
        axis.tick_params(
            axis="y",
            rotation=0,
        )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    return matrix, normalized

def save_predictions(
    dataset_indices,
    labels,
    predictions,
    probabilities,
    output_path,
):
    """Save per-image predictions for error analysis."""
    probability_columns = [
        "prob_no_dr",
        "prob_mild",
        "prob_moderate",
        "prob_severe",
        "prob_proliferative_dr",
    ]

    records = []

    for position in range(len(labels)):
        true_id = int(labels[position])
        predicted_id = int(predictions[position])

        record = {
            "dataset_index": int(
                dataset_indices[position]
            ),
            "true_class_id": true_id,
            "true_class_name": CLASS_NAMES[true_id],
            "predicted_class_id": predicted_id,
            "predicted_class_name": CLASS_NAMES[
                predicted_id
            ],
            "confidence": float(
                probabilities[position].max()
            ),
            "correct": bool(
                true_id == predicted_id
            ),
        }

        for name, probability in zip(
            probability_columns,
            probabilities[position],
        ):
            record[name] = float(probability)

        records.append(record)

    pd.DataFrame(records).to_csv(
        output_path,
        index=False,
    )

def main(args):
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"ConvNeXt checkpoint not found: {checkpoint_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "backbone",
        "image_size",
        "class_names",
    }

    missing_checkpoint_fields = (
        required_keys - set(checkpoint.keys())
    )

    if missing_checkpoint_fields:
        raise KeyError(
            "ConvNeXt checkpoint is missing fields: "
            f"{sorted(missing_checkpoint_fields)}"
        )

    if list(checkpoint["class_names"]) != CLASS_NAMES:
        raise ValueError(
            "Checkpoint class order does not match dataset class order."
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"Mixed precision: {use_amp}")
    print(f"Backbone: {checkpoint['backbone']}")
    print(f"Image size: {checkpoint['image_size']}")
    print(
        "Validation kappa stored in checkpoint: "
        f"{checkpoint.get('val_kappa')}"
    )

    _, _, test_dataset, _ = load_splits(
        image_size=int(checkpoint["image_size"]),
        seed=args.seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        backbone=checkpoint["backbone"],
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone=False,
    )

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    print(f"State loading result: {load_result}")

    model = model.to(device)

    labels, predictions, probabilities, test_loss = (
        run_inference(
            model,
            test_loader,
            device,
            use_amp,
        )
    )

    accuracy = float(
        accuracy_score(labels, predictions)
    )
    balanced_accuracy = float(
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )
    kappa = float(
        cohen_kappa_score(
            labels,
            predictions,
            weights="quadratic",
        )
    )
    macro_f1 = float(
        f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    report_text = classification_report(
        labels,
        predictions,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_dictionary = classification_report(
        labels,
        predictions,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix, normalized_matrix = (
        save_confusion_matrices(
            labels,
            predictions,
            output_dir
            / "test_confusion_matrix.png",
        )
    )

    save_predictions(
        test_dataset.indices,
        labels,
        predictions,
        probabilities,
        output_dir
        / "test_predictions.csv",
    )

    results = {
        "model": checkpoint["backbone"],
        "validation_kappa": checkpoint.get(
            "val_kappa"
        ),
        "seed": args.seed,
        "test_samples": len(test_dataset),
        "test_loss": float(test_loss),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "quadratic_weighted_kappa": kappa,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dictionary,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": (
            normalized_matrix.tolist()
        ),
    }

    with (
        output_dir / "test_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("=== CONVNEXT V2-TINY HELD-OUT TEST RESULTS ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )
    print(
        f"Quadratic weighted kappa: "
        f"{kappa:.4f}"
    )
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print()
    print(report_text)

    print(
        "Saved results: "
        f"{output_dir / 'test_results.json'}"
    )
    print(
        "Saved predictions: "
        f"{output_dir / 'test_predictions.csv'}"
    )
    print(
        "Saved confusion matrix: "
        f"{output_dir / 'test_confusion_matrix.png'}"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ConvNeXt V2-Tiny on "
            "the held-out APTOS test split."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
