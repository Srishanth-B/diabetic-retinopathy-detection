"""
3-way comparison: RETFound Linear Probe vs RETFound LoRA vs ConvNeXt V2-Tiny.

All metrics are derived from each model's 5x5 confusion matrix so the
methodology is identical across models (schemas of the three result files
differ, but the confusion matrix is a common, unambiguous source).

Outputs (in reports/):
  - overall_comparison.csv / .md
  - per_class_recall.csv
  - referable_comparison.csv
  - fig_overall_metrics.png
  - fig_per_class_recall.png
  - fig_referable.png
  - fig_confusion_matrices.png
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Each model -> its test_results.json
MODELS = {
    "Linear Probe": PROJECT_ROOT / "checkpoints" / "retfound_linear_probe" / "test_results.json",
    "LoRA":         PROJECT_ROOT / "checkpoints" / "retfound_lora" / "test_results.json",
    "ConvNeXt":     PROJECT_ROOT / "checkpoints" / "convnextv2_tiny" / "test_results.json",
}

def collect_5x5(obj, found):
    """Recursively collect every 5x5 numeric matrix in a JSON object."""
    if isinstance(obj, list):
        is_5x5 = (
            len(obj) == 5
            and all(isinstance(r, list) and len(r) == 5
                    and all(isinstance(x, (int, float)) for x in r) for r in obj)
        )
        if is_5x5:
            found.append(obj)
        else:
            for item in obj:
                collect_5x5(item, found)
    elif isinstance(obj, dict):
        for value in obj.values():
            collect_5x5(value, found)

def load_confusion_matrix(path):
    """Return the raw-count 5x5 confusion matrix from a result file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidates = []
    collect_5x5(data, candidates)
    if not candidates:
        raise ValueError(f"No 5x5 confusion matrix found in {path}")
    # The counts matrix has the largest total (~550); normalized ones sum to ~5.
    best = max(candidates, key=lambda m: np.array(m).sum())
    return np.array(best, dtype=float)

def qwk_from_cm(cm):
    """Quadratic weighted kappa computed directly from a confusion matrix."""
    n = cm.shape[0]
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weights[i, j] = (i - j) ** 2 / (n - 1) ** 2
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    expected = np.outer(row, col) / cm.sum()
    observed_num = (weights * cm).sum()
    expected_num = (weights * expected).sum()
    return 1.0 - observed_num / expected_num

def metrics_from_cm(cm):
    total = cm.sum()
    diag = np.diag(cm)
    row = cm.sum(axis=1)   # true support per class
    col = cm.sum(axis=0)   # predicted per class
    recall = diag / np.where(row == 0, 1, row)
    precision = diag / np.where(col == 0, 1, col)
    denom = np.where((precision + recall) == 0, 1, precision + recall)
    f1 = 2 * precision * recall / denom
    return {
        "accuracy": diag.sum() / total,
        "balanced_accuracy": recall.mean(),
        "qwk": qwk_from_cm(cm),
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * row).sum() / total,
        "recall_per_class": recall,
    }

def referable_from_cm(cm, threshold=2):
    """Binary referable-DR metrics: grade >= threshold is 'referable'."""
    tp = cm[threshold:, threshold:].sum()
    fn = cm[threshold:, :threshold].sum()
    tn = cm[:threshold, :threshold].sum()
    fp = cm[:threshold, threshold:].sum()

    def sd(a, b):
        return a / b if b else 0.0

    return {
        "sensitivity": sd(tp, tp + fn),
        "specificity": sd(tn, tn + fp),
        "ppv": sd(tp, tp + fp),
        "npv": sd(tn, tn + fn),
        "binary_accuracy": sd(tp + tn, total := cm.sum()),
        "missed_sick_fn": int(fn),
    }

def grouped_bar(ax, categories, series, title, ylabel, ylim=None):
    """series = dict{model_name: [values aligned with categories]}."""
    n_models = len(series)
    x = np.arange(len(categories))
    width = 0.8 / n_models
    for idx, (name, values) in enumerate(series.items()):
        offset = (idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cms, overall, referable = {}, {}, {}
    for name, path in MODELS.items():
        cm = load_confusion_matrix(path)
        cms[name] = cm
        overall[name] = metrics_from_cm(cm)
        referable[name] = referable_from_cm(cm)

    names = list(MODELS.keys())

    # ---- CSV: overall ----
    metric_keys = ["accuracy", "balanced_accuracy", "qwk", "macro_f1", "weighted_f1"]
    with open(REPORTS_DIR / "overall_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + names)
        for mk in metric_keys:
            w.writerow([mk] + [f"{overall[n][mk]:.4f}" for n in names])

    # ---- Markdown: overall ----
    with open(REPORTS_DIR / "overall_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Overall test comparison")
        f.write("| Metric | " + " | ".join(names) + " |")
        f.write("|" + "---|" * (len(names) + 1) + "")
        for mk in metric_keys:
            f.write(f"| {mk} | " + " | ".join(f"{overall[n][mk]:.4f}" for n in names) + " |")

    # ---- CSV: per-class recall ----
    with open(REPORTS_DIR / "per_class_recall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class"] + names)
        for i, cls in enumerate(CLASS_NAMES):
            w.writerow([cls] + [f"{overall[n]['recall_per_class'][i]:.4f}" for n in names])

    # ---- CSV: referable ----
    ref_keys = ["sensitivity", "specificity", "ppv", "npv", "binary_accuracy", "missed_sick_fn"]
    with open(REPORTS_DIR / "referable_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + names)
        for rk in ref_keys:
            w.writerow([rk] + [referable[n][rk] for n in names])

    # ---- Figure 1: overall metrics ----
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = ["Accuracy", "BalancedAcc", "QWK", "Macro-F1", "Weighted-F1"]
    series = {n: [overall[n][k] for k in metric_keys] for n in names}
    grouped_bar(ax, cats, series, "Overall test metrics", "Score", ylim=(0, 1.0))
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "fig_overall_metrics.png", dpi=150); plt.close(fig)

    # ---- Figure 2: per-class recall ----
    fig, ax = plt.subplots(figsize=(9, 5))
    series = {n: list(overall[n]["recall_per_class"]) for n in names}
    grouped_bar(ax, CLASS_NAMES, series, "Per-class recall", "Recall", ylim=(0, 1.05))
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "fig_per_class_recall.png", dpi=150); plt.close(fig)

    # ---- Figure 3: referable-DR ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["Sensitivity", "Specificity", "Binary Acc"]
    series = {n: [referable[n]["sensitivity"], referable[n]["specificity"],
                  referable[n]["binary_accuracy"]] for n in names}
    grouped_bar(ax, cats, series, "Referable-DR (clinical) comparison", "Score", ylim=(0, 1.05))
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "fig_referable.png", dpi=150); plt.close(fig)

    # ---- Figure 4: confusion matrices side by side ----
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    for ax, n in zip(axes, names):
        cm = cms[n].astype(int)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(n)
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(range(5)); ax.set_yticklabels(range(5))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        thresh = cm.max() / 2
        for i in range(5):
            for j in range(5):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "fig_confusion_matrices.png", dpi=150); plt.close(fig)

    # ---- Console summary ----
    print("Overall metrics:")
    for n in names:
        o = overall[n]
        print(f"  {n:14s} acc={o['accuracy']:.4f} qwk={o['qwk']:.4f} "
              f"macroF1={o['macro_f1']:.4f} balAcc={o['balanced_accuracy']:.4f}")
    print("Referable-DR:")
    for n in names:
        r = referable[n]
        print(f"  {n:14s} sens={r['sensitivity']:.4f} spec={r['specificity']:.4f} "
              f"binAcc={r['binary_accuracy']:.4f} missed={r['missed_sick_fn']}")
    print(f"Artifacts written to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()