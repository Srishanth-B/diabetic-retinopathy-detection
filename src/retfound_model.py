"""
Local RETFound-MAE model integration for APTOS DR classification.

This module:
1. Imports the official RETFound ViT-Large/16 architecture.
2. Loads the local RETFound Nature colour-fundus checkpoint.
3. Replaces the original classifier with a five-class APTOS head.
4. Supports linear probing or full fine-tuning.

The pretrained checkpoint is loaded entirely from local storage. No
Hugging Face token or inference API is required at runtime.
"""
from types import MethodType

import argparse
import gc
import sys
from pathlib import Path

import torch
from timm.layers import trunc_normal_
from torchgen import model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETFOUND_SOURCE_DIR = PROJECT_ROOT / "external" / "RETFound"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "weights" / "RETFound_mae_natureCFP.pth"
)

NUM_CLASSES = 5

def import_official_retfound():
    """Import RETFound architecture and position-embedding utility."""
    if not RETFOUND_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"RETFound source directory not found: {RETFOUND_SOURCE_DIR}"
        )

    source_path = str(RETFOUND_SOURCE_DIR)

    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    import models_vit
    from util.pos_embed import interpolate_pos_embed

    return models_vit, interpolate_pos_embed

def _retfound_forward_compat(self, images):
    """Reproduce RETFound's original forward path with modern timm.

    The official RETFound forward_features() already performs global
    pooling and fc_norm. Modern timm's inherited forward_head() tries
    to perform pooling and normalization again, so we bypass it and
    pass the prepared features directly to the classifier.
    """
    features = self.forward_features(images)

    # RETFound global pooling returns [batch, 1, embedding_dimension].
    # The classifier expects [batch, embedding_dimension].
    if features.ndim == 3:
        if features.shape[1] != 1:
            raise RuntimeError(
                f"Unexpected RETFound feature shape: {tuple(features.shape)}"
            )
        features = features[:, 0, :]

    return self.head(features)

def build_retfound_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    num_classes: int = NUM_CLASSES,
    adaptation: str = "linear_probe",
    image_size: int = 224,
    drop_path_rate: float = 0.2,
):
    """Build RETFound MAE and load the local colour-fundus checkpoint.

    Args:
        checkpoint_path: local path to RETFound_mae_natureCFP.pth.
        num_classes: number of downstream APTOS classes.
        adaptation:
            "linear_probe" freezes everything except the classifier head.
            "full_finetune" makes the entire model trainable.
        image_size: model input size.
        drop_path_rate: stochastic-depth regularization rate.

    Returns:
        model: initialized RETFound model.
        load_info: missing and unexpected checkpoint keys.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"RETFound checkpoint not found: {checkpoint_path}"
        )

    if adaptation not in {"linear_probe", "full_finetune"}:
        raise ValueError(
            "adaptation must be 'linear_probe' or 'full_finetune'"
        )

    models_vit, interpolate_pos_embed = import_official_retfound()

    print("Creating RETFound MAE ViT-Large/16 architecture...")

    model = models_vit.RETFound_mae(
        img_size=image_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool=True,
    )
    
 # RETFound uses a Boolean flag inside its custom forward_features().
    model.global_pool = True

# Bypass timm 1.x forward_head(), which would pool and normalize twice.
    model.forward = MethodType(_retfound_forward_compat, model)

    print("RETFound compatibility forward installed.")


    print(f"Loading local checkpoint: {checkpoint_path}")

    # mmap avoids eagerly copying the complete ~4 GB checkpoint into RAM.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain the expected 'model' key."
        )

    checkpoint_model = checkpoint["model"]

    # Remove any original classifier whose output shape differs from ours.
    model_state = model.state_dict()

    for key in ("head.weight", "head.bias"):
        if (
            key in checkpoint_model
            and key in model_state
            and checkpoint_model[key].shape != model_state[key].shape
        ):
            print(
                f"Removing incompatible checkpoint key {key}: "
                f"{tuple(checkpoint_model[key].shape)} -> "
                f"{tuple(model_state[key].shape)}"
            )
            del checkpoint_model[key]

    # Required if checkpoint and downstream input resolutions differ.
    interpolate_pos_embed(model, checkpoint_model)

    load_result = model.load_state_dict(
        checkpoint_model,
        strict=False,
    )

    # Initialize the new five-class classifier.
    trunc_normal_(model.head.weight, std=2e-5)

    if model.head.bias is not None:
        torch.nn.init.zeros_(model.head.bias)

    # Release the large checkpoint dictionary as soon as possible.
    del checkpoint_model
    del checkpoint
    gc.collect()

    if adaptation == "linear_probe":
        # This matches the official repository's LP implementation:
        # only parameter names containing "head" remain trainable.
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "head" in name
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True

    load_info = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }

    return model, load_info

def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable

def main():
    parser = argparse.ArgumentParser(
        description="Build and verify the local RETFound model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--adaptation",
        choices=["linear_probe", "full_finetune"],
        default="linear_probe",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    args = parser.parse_args()

    model, load_info = build_retfound_model(
        checkpoint_path=args.checkpoint,
        adaptation=args.adaptation,
        image_size=args.image_size,
    )
 


    total, trainable = count_parameters(model)

    print("RETFound model created successfully.")
    print(f"Adaptation: {args.adaptation}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(
        f"Trainable percentage: "
        f"{100 * trainable / total:.6f}%"
    )
    print(f"Missing checkpoint keys: {load_info['missing_keys']}")
    print(
        f"Unexpected checkpoint keys: "
        f"{load_info['unexpected_keys']}"
    )

if __name__ == "__main__":
    main()
