"""
Held-out test evaluation for the RETFound APTOS linear probe.

This script:
1. Rebuilds RETFound from the local Nature CFP checkpoint.
2. Loads the best validation-selected classifier head.
3. Evaluates once on the untouched test split.
4. Reports classification metrics.
5. Saves raw and normalized confusion matrices.
6. Saves per-image predictions for later error analysis.
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
from src.retfound_model import (
    DEFAULT_CHECKPOINT,
    build_retfound_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "retfound_linear_probe"
)
DEFAULT_HEAD_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best_head.pt"

def set_seed(seed: int) -> None:
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(
    model,
    loader,
    device,
    criterion,
    use_amp,
):
    """Run inference over the held-out test set."""
    model.eval()

    total_loss = 0.0
    labels_all = []
    predictions_all = []
    probabilities_all = []

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
                    "RETFound produced non-finite test logits."
                )

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )

            predictions = probabilities.argmax(dim=1)

            total_loss += (
                loss.item() * images.size(0)
            )

            labels_all.extend(
                labels.cpu().tolist()
            )

            predictions_all.extend(
                predictions.cpu().tolist()
            )

            probabilities_all.extend(
                probabilities.cpu().tolist()
            )

    average_loss = total_loss / len(loader.dataset)

    return (
        np.asarray(labels_all),
        np.asarray(predictions_all),
        np.asarray(probabilities_all),
        float(average_loss),
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

    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
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
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )

    axes[0].set_title(
        "RETFound Test Confusion Matrix"
    )
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")

    sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )

    axes[1].set_title(
        "RETFound Normalized Confusion Matrix"
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

    return matrix, normalized_matrix

def save_predictions(
    dataset_indices,
    labels,
    predictions,
    probabilities,
    output_path,
):
    """Save per-image predictions and class probabilities."""
    records = []

    probability_columns = [
        "prob_no_dr",
        "prob_mild",
        "prob_moderate",
        "prob_severe",
        "prob_proliferative_dr",
    ]

    for position in range(len(labels)):
        predicted_class = int(
            predictions[position]
        )

        record = {
            "dataset_index": int(
                dataset_indices[position]
            ),
            "true_class_id": int(
                labels[position]
            ),
            "true_class_name": CLASS_NAMES[
                int(labels[position])
            ],
            "predicted_class_id": predicted_class,
            "predicted_class_name": CLASS_NAMES[
                predicted_class
            ],
            "confidence": float(
                probabilities[position].max()
            ),
            "correct": bool(
                labels[position]
                == predictions[position]
            ),
        }

        for column, probability in zip(
            probability_columns,
            probabilities[position],
        ):
            record[column] = float(probability)

        records.append(record)

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        output_path,
        index=False,
    )

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    head_checkpoint_path = Path(
        args.head_checkpoint
    )

    if not head_checkpoint_path.exists():
        raise FileNotFoundError(
            "Best RETFound head was not found: "
            f"{head_checkpoint_path}"
        )

    head_checkpoint = torch.load(
        head_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    seed = int(
        head_checkpoint.get("seed", 42)
    )

    image_size = int(
        head_checkpoint.get(
            "image_size",
            224,
        )
    )

    set_seed(seed)

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
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Mixed precision: {use_amp}")
    print(
        f"Selected head epoch: "
        f"{head_checkpoint['epoch']}"
    )
    print(
        "Validation metrics used for selection: "
        f"{head_checkpoint['validation_metrics']}"
    )

    _, _, test_dataset, _ = load_splits(
        image_size=image_size,
        seed=seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model, load_info = build_retfound_model(
        checkpoint_path=args.retfound_checkpoint,
        num_classes=len(CLASS_NAMES),
        adaptation="linear_probe",
        image_size=image_size,
    )

    model.head.load_state_dict(
        head_checkpoint["head_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Test samples: {len(test_dataset)}"
    )
    print(
        f"Pretraining missing keys: "
        f"{load_info['missing_keys']}"
    )
    print(
        "Pretraining unexpected key count: "
        f"{len(load_info['unexpected_keys'])}"
    )

    criterion = nn.CrossEntropyLoss()

    labels, predictions, probabilities, test_loss = (
        evaluate_model(
            model,
            test_loader,
            device,
            criterion,
            use_amp,
        )
    )

    accuracy = float(
        accuracy_score(
            labels,
            predictions,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            labels,
            predictions,
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

    kappa = float(
        cohen_kappa_score(
            labels,
            predictions,
            weights="quadratic",
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
        "model": "RETFound_mae_natureCFP",
        "adaptation": "linear_probe",
        "head_checkpoint": str(
            head_checkpoint_path
        ),
        "selected_epoch": int(
            head_checkpoint["epoch"]
        ),
        "validation_metrics": (
            head_checkpoint[
                "validation_metrics"
            ]
        ),
        "test_samples": len(test_dataset),
        "test_loss": test_loss,
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

    print("=== RETFound HELD-OUT TEST RESULTS ===")
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
            "Evaluate the selected RETFound "
            "linear probe on the held-out test set."
        )
    )

    parser.add_argument(
        "--retfound-checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--head-checkpoint",
        type=str,
        default=str(
            DEFAULT_HEAD_CHECKPOINT
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
