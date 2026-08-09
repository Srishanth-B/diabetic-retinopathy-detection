"""
Inference + Grad-CAM for the ConvNeXt V2-Tiny diabetic retinopathy model.
Predicts the 5-class grade, a clinical referable-DR decision, and a
Grad-CAM heatmap showing where the model looked.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
 # adjust if needed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model, _class_names, _image_size = None, None, 224

def load_model(ckpt_path=CKPT_PATH):
    global _model, _class_names, _image_size
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
    _class_names = ckpt.get("class_names",
                            ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"])
    _image_size = ckpt.get("image_size", 224)
    backbone = ckpt.get("backbone", "convnextv2_tiny")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(_class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(_device)
    _model = model
    print(f"Loaded {backbone} | classes: {_class_names} | image_size: {_image_size}")
    return model

def _preprocess(image: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((_image_size, _image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(image.convert("RGB")).unsqueeze(0).to(_device)

def predict(image: Image.Image):
    if _model is None:
        load_model()
    x = _preprocess(image)
    with torch.no_grad():
        probs = F.softmax(_model(x), dim=1)[0].cpu().numpy()
    grade = int(probs.argmax())
    return {
        "grade": grade,
        "grade_name": _class_names[grade],
        "confidence": float(probs[grade]),
        "referable": grade >= 2,               # grade >= 2 => see a doctor
        "probs": probs,
        "class_names": _class_names,
    }

def gradcam(image: Image.Image, class_idx=None):
    """Grad-CAM heatmap (HxW, values 0..1) via timm forward_features/forward_head."""
    if _model is None:
        load_model()
    x = _preprocess(image)
    feats = _model.forward_features(x)       # [1, C, H, W]
    feats.retain_grad()
    logits = _model.forward_head(feats)
    if class_idx is None:
        class_idx = int(logits.argmax(dim=1).item())
    _model.zero_grad()
    logits[0, class_idx].backward()
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(_image_size, _image_size),
                        mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def overlay_heatmap(image: Image.Image, cam, alpha=0.45):
    img = np.array(image.convert("RGB").resize((_image_size, _image_size))).astype(np.float32) / 255.0
    heat = cm.jet(cam)[..., :3]
    out = ((1 - alpha) * img + alpha * heat) * 255
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

if __name__ == "__main__":
    import sys
    load_model()
    if len(sys.argv) > 1:                     # optional: pass an image path to test
        img = Image.open(sys.argv[1])
        r = predict(img)
        print(f"Grade: {r['grade']} ({r['grade_name']}) | conf {r['confidence']:.2%} "
              f"| Referable: {r['referable']}")
