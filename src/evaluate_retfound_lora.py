"""
Held-out test evaluation for the RETFound + LoRA model on APTOS 2019.

Produces TWO views of performance:
  1. Strict 5-class grading: accuracy, balanced accuracy, quadratic weighted
     kappa (QWK), macro/weighted F1, per-class precision/recall/F1, confusion
     matrix.
  2. Clinical referable-DR screening (grade >= 2 = "needs a specialist"):
     sensitivity, specificity, PPV, NPV, binary accuracy. This is how
     FDA-cleared DR systems are actually measured.

The test split is generated with the SAME seed as training, so it is identical
to the split used by the linear-probe and ConvNeXt evaluators. The test set is
evaluated exactly once.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.dataset import CLASS_NAMES, load_splits
from src.retfound_model import DEFAULT_CHECKPOINT, build_retfound_model
from src.train_retfound import set_seed
from src.lora import inject_lora

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "retfound_lora"
DEFAULT_LORA_CKPT = OUTPUT_DIR / "best_lora.pt"

# Clinical definition: grade >= 2 (Moderate, Severe, PDR) is "referable DR".
REFERABLE_THRESHOLD = 2
REFERABLE_NAMES = ["Non-referable (0-1)", "Referable (2-4)"]

def load_lora_model(retfound_ckpt, lora_ckpt, device, image_size):
    """Rebuild RETFound, inject LoRA with the SAVED config, load adapters+head."""
    print(f"Loading LoRA checkpoint: {lora_ckpt}")
    ckpt = torch.load(lora_ckpt, map_location="cpu", weights_only=False)

    rank = ckpt["lora_rank"]
    alpha = ckpt["lora_alpha"]
    dropout = ckpt["lora_dropout"]
    include_mlp = ckpt.get("include_mlp", False)

    targets = ["qkv", "proj"] + (["fc1", "fc2"] if include_mlp else [])
    print(f"Reconstructing LoRA: targets={targets}, r={rank}, alpha={alpha}, "
          f"dropout={dropout}")

    # Build with full_finetune so no params are pre-frozen, then inject LoRA
    # identically to training so the checkpoint keys line up exactly.
    model, _ = build_retfound_model(
        checkpoint_path=retfound_ckpt,
        num_classes=len(CLASS_NAMES),
        adaptation="full_finetune",
        image_size=image_size,
        drop_path_rate=0.0,  # irrelevant in eval mode, kept clean
    )
    inject_lora(model, target_modules=tuple(targets), r=rank, alpha=alpha,
                dropout=dropout, verbose=False)

    missing, unexpected = model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    if len(unexpected) > 0:
        raise RuntimeError(
            f"Unexpected keys when loading LoRA state dict: {unexpected[:5]} ..."
        )
    loaded = len(ckpt["lora_state_dict"])
    print(f"Loaded {loaded} LoRA/head tensors "
          f"(missing backbone keys, as expected: {len(missing)}).")

    provenance = {
        "best_epoch": ckpt.get("epoch"),
        "validation_metrics": ckpt.get("validation_metrics"),
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "include_mlp": include_mlp,
        "trainable_parameters": ckpt.get("trainable_parameters"),
    }
    return model.to(device).eval(), provenance

@torch.no_grad()
def run_inference(model, loader, device, use_amp):
    """Return (labels, preds, probs) over the whole loader in fixed order."""
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def compute_multiclass_metrics(labels, preds):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0,
    )
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, preds, weights="quadratic")
        ),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
    }

def compute_referable_metrics(labels, preds):
    """Collapse to the clinical binary decision: grade >= 2 == referable."""
    true_ref = (labels >= REFERABLE_THRESHOLD).astype(int)
    pred_ref = (preds >= REFERABLE_THRESHOLD).astype(int)

    tp = int(np.sum((true_ref == 1) & (pred_ref == 1)))
    fn = int(np.sum((true_ref == 1) & (pred_ref == 0)))
    tn = int(np.sum((true_ref == 0) & (pred_ref == 0)))
    fp = int(np.sum((true_ref == 0) & (pred_ref == 1)))

    def safe_div(a, b):
        return float(a / b) if b else 0.0

    return {
        "definition": "referable = predicted grade >= 2 (Moderate/Severe/PDR)",
        "sensitivity_recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv_precision": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "binary_accuracy": safe_div(tp + tn, tp + fn + tn + fp),
        "counts": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "confusion_matrix": [[tn, fp], [fn, tp]],  # rows=true[non-ref,ref]
    }

def plot_confusion(cm, class_names, out_path, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 1.6 * len(class_names) + 1))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_predictions_csv(path, labels, preds, probs):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["dataset_index", "true_label", "predicted_label",
                  "true_class_name", "predicted_class_name", "correct",
                  "true_referable", "predicted_referable"]
        header += [f"prob_{i}" for i in range(len(CLASS_NAMES))]
        writer.writerow(header)

        for idx, (t, p, pr) in enumerate(zip(labels, preds, probs)):
            row = [
                idx, int(t), int(p),
                CLASS_NAMES[int(t)], CLASS_NAMES[int(p)],
                int(t == p),
                int(t >= REFERABLE_THRESHOLD), int(p >= REFERABLE_THRESHOLD),
            ]
            row += [f"{x:.6f}" for x in pr]
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RETFound+LoRA on the APTOS test set.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--lora-checkpoint", type=str, default=str(DEFAULT_LORA_CKPT))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    print(f"Device: {device}")

    # Identical split to training / other evaluators.
    _, _, test_dataset, _ = load_splits(image_size=args.image_size, seed=args.seed)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,               # fixed order -> stable dataset_index
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model, provenance = load_lora_model(args.checkpoint, args.lora_checkpoint, device, args.image_size)

    labels, preds, probs = run_inference(model, test_loader, device, use_amp)

    multiclass = compute_multiclass_metrics(labels, preds)
    referable = compute_referable_metrics(labels, preds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Save artifacts ----
    results = {
        "model": "RETFound_mae_natureCFP + LoRA",
        "adaptation": "lora",
        "num_test_samples": int(len(labels)),
        "class_names": CLASS_NAMES,
        "provenance": provenance,
        "multiclass_metrics": multiclass,
        "referable_dr_metrics": referable,
    }
    with (OUTPUT_DIR / "test_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_predictions_csv(OUTPUT_DIR / "test_predictions.csv", labels, preds, probs)

    plot_confusion(multiclass["confusion_matrix"], CLASS_NAMES,
                   OUTPUT_DIR / "test_confusion_matrix.png",
                   "RETFound+LoRA - 5-class confusion (test)")
    plot_confusion(referable["confusion_matrix"], REFERABLE_NAMES,
                   OUTPUT_DIR / "test_referable_confusion_matrix.png",
                   "RETFound+LoRA - referable DR (test)")

    # ---- Console summary ----
    print("================ 5-CLASS GRADING (strict) ================")
    print(f"Accuracy           : {multiclass['accuracy']:.4f}")
    print(f"Balanced accuracy  : {multiclass['balanced_accuracy']:.4f}")
    print(f"Quadratic W. Kappa : {multiclass['quadratic_weighted_kappa']:.4f}")
    print(f"Macro F1           : {multiclass['macro_f1']:.4f}")
    print(f"Weighted F1        : {multiclass['weighted_f1']:.4f}")
    print("Per-class (P / R / F1 / support):")
    for name, m in multiclass["per_class"].items():
        print(f"  {name:16s}: {m['precision']:.4f} / {m['recall']:.4f} / "
              f"{m['f1']:.4f} / {m['support']}")

    print("=========== REFERABLE DR (clinical binary) ===============")
    print(f"Sensitivity (recall): {referable['sensitivity_recall']:.4f}  "
          f"<- fraction of sick patients caught")
    print(f"Specificity         : {referable['specificity']:.4f}")
    print(f"PPV (precision)     : {referable['ppv_precision']:.4f}")
    print(f"NPV                 : {referable['npv']:.4f}")
    print(f"Binary accuracy     : {referable['binary_accuracy']:.4f}")
    print(f"Counts (TP/FN/TN/FP): {referable['counts']}")

    print("Artifacts written to:", OUTPUT_DIR)
    print("  - test_results.json")
    print("  - test_predictions.csv")
    print("  - test_confusion_matrix.png")
    print("  - test_referable_confusion_matrix.png")

if __name__ == "__main__":
    main()
