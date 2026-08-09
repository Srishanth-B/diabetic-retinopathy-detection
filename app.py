"""
Diabetic Retinopathy Screening — self-contained Gradio app.
Works locally (python app.py) and on Hugging Face Spaces.
"""
import io, urllib.request
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, timm
import matplotlib.cm as cm, gradio as gr
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
CKPT_PATH = next((p for p in [HERE / "best_model.pt", HERE / "checkpoints" / "best_model.pt"]
                  if p.exists()), HERE / "best_model.pt")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
CLASS_NAMES = ckpt.get("class_names", ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"])
IMG = ckpt.get("image_size", 224)
model = timm.create_model(ckpt.get("backbone", "convnextv2_tiny"),
                          pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(DEVICE)

_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(),
                          transforms.Normalize(MEAN, STD)])

def _pre(image): return _tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)

def _gradcam(x):
    feats = model.forward_features(x); feats.retain_grad()
    logits = model.forward_head(feats); idx = int(logits.argmax(1).item())
    model.zero_grad(); logits[0, idx].backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, (IMG, IMG), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def _overlay(image, cam, a=0.45):
    img = np.array(image.convert("RGB").resize((IMG, IMG))).astype(np.float32) / 255
    out = ((1 - a) * img + a * cm.jet(cam)[..., :3]) * 255
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

def _from_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")

def analyze(image, url):
    if image is None and url:
        try:
            image = _from_url(url.strip())
        except Exception as e:
            return f"Could not load image from URL: {e}", {}, None
    if image is None:
        return "Please upload an image or paste a URL.", {}, None
    with torch.no_grad():
        probs = F.softmax(model(_pre(image)), 1)[0].cpu().numpy()
    g = int(probs.argmax())
    conf = float(probs[g])
    heat = _overlay(image, _gradcam(_pre(image)))
    pd = {n: float(p) for n, p in zip(CLASS_NAMES, probs)}
    if g >= 2:
        v = f"REFER TO AN OPHTHALMOLOGIST -- Predicted: {CLASS_NAMES[g]} (grade {g}), confidence {conf:.1%}. Signs consistent with referable diabetic retinopathy were detected."
    else:
        v = f"NO REFERRAL NEEDED (routine follow-up) -- Predicted: {CLASS_NAMES[g]} (grade {g}), confidence {conf:.1%}. No referable diabetic retinopathy detected."
    return v, pd, heat


DESC = """# 👁️ Diabetic Retinopathy Screening
Upload a **retinal fundus photograph** (or paste an image URL) to get a DR severity grade and a
**referral decision**. The heatmap shows where the model focused.

⚠️ *Research/educational demo only — NOT a medical device.*"""

with gr.Blocks(title="Diabetic Retinopathy Screening") as demo:
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Upload fundus image")
            url_in = gr.Textbox(label="...or paste an image URL", placeholder="https://...")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            v_out = gr.Textbox(label="Result", lines=6)
            p_out = gr.Label(label="Grade probabilities", num_top_classes=5)
            h_out = gr.Image(type="pil", label="Grad-CAM (where the model looked)")
    btn.click(analyze, [img_in, url_in], [v_out, p_out, h_out])

if __name__ == "__main__":
    demo.launch()
