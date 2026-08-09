"""External validation: run the APTOS-trained ConvNeXt on the EyePACS test set."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, confusion_matrix)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"   # APTOS-trained model
OUTPUT_DIR = PROJECT_ROOT / "reports"
EYEPACS_DIR = Path(r"C:\Users\Srishanth\archive")
IMAGES_DIR = EYEPACS_DIR / "Images"
TEST_CSV = EYEPACS_DIR / "original_test_labels.csv"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_image(name):
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG"]:
        p = IMAGES_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = timm.create_model(ckpt.get("backbone", "convnextv2_tiny"),
                              pretrained=False, num_classes=5)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"Loaded APTOS-trained model on {device}.")

    df = pd.read_csv(TEST_CSV)
    if args.limit > 0:
        df = df.sample(min(len(df), args.limit), random_state=args.seed)
    print(f"Evaluating {len(df)} EyePACS test images...")

    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    y_true, y_pred, missing = [], [], 0
    bx, by = [], []

    def flush():
        if not bx:
            return
        x = torch.stack(bx).to(device)
        with torch.no_grad():
            p = model(x).argmax(1).cpu().numpy()
        y_pred.extend(p.tolist()); y_true.extend(by)
        bx.clear(); by.clear()

    for i, row in enumerate(df.itertuples(index=False)):
        path = find_image(row.image)
        if path is None:
            missing += 1; continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            missing += 1; continue
        bx.append(tf(img)); by.append(int(row.level))
        if len(bx) >= args.batch_size:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(df)}")
    flush()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(f"Done. Evaluated {len(y_true)} images ({missing} missing/skipped).")

    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    tr, pr = (y_true >= 2).astype(int), (y_pred >= 2).astype(int)
    tp = int(((tr == 1) & (pr == 1)).sum()); fn = int(((tr == 1) & (pr == 0)).sum())
    tn = int(((tr == 0) & (pr == 0)).sum()); fp = int(((tr == 0) & (pr == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    binacc = (tp + tn) / len(y_true)

    print("===== EyePACS external validation (APTOS-trained ConvNeXt) =====")
    print(f"Accuracy          : {acc:.4f}   (APTOS test: 0.8109)")
    print(f"Balanced accuracy : {bal:.4f}   (APTOS test: 0.6426)")
    print(f"QWK               : {qwk:.4f}   (APTOS test: 0.8906)")
    print(f"Macro-F1          : {mf1:.4f}   (APTOS test: 0.6536)")
    print(f"Referable sens.   : {sens:.4f}   (APTOS test: 0.9552)")
    print(f"Referable spec.   : {spec:.4f}")
    print(f"Referable bin-acc : {binacc:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "eyepacs_external_validation.json").open("w", encoding="utf-8") as f:
        json.dump({"n_images": int(len(y_true)), "missing": missing, "accuracy": acc,
                   "balanced_accuracy": bal, "qwk": qwk, "macro_f1": mf1,
                   "referable": {"sensitivity": sens, "specificity": spec,
                                 "binary_accuracy": binacc, "tp": tp, "fn": fn, "tn": tn, "fp": fp},
                   "confusion_matrix": cm.tolist()}, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'eyepacs_external_validation.json'}")

if __name__ == "__main__":
    main()
