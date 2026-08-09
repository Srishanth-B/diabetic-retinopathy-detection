"""
Model definition for diabetic retinopathy severity classification.

Backbone: ConvNeXt V2-Tiny (via the `timm` library).
ConvNeXt V2 (Woo et al., 2023) is a modern convolutional architecture
pretrained with a fully-convolutional masked autoencoder.

The `freeze_backbone` flag lets us do TWO-PHASE fine-tuning:
  - Phase 1 (warm-up): freeze_backbone=True  -> train only the head
  - Phase 2 (fine-tune): freeze_backbone=False -> train all layers (low LR)

Regularization for our small dataset (~2,500 images):
  - data augmentation (dataset.py)
  - dropout (drop_rate) + stochastic depth (drop_path_rate) below
  - weight decay + low LR (train.py)
  - best-checkpoint selection on validation
"""

import timm

NUM_CLASSES = 5
DEFAULT_BACKBONE = "convnextv2_tiny"

def build_model(
    backbone: str = DEFAULT_BACKBONE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.1,
):
    """Build a ConvNeXt V2 (or any timm) model for DR classification.

    Args:
        backbone: timm model name. Default "convnextv2_tiny".
        num_classes: number of output classes (5 for DR severity).
        pretrained: load pretrained weights.
        freeze_backbone: if True, freeze feature extractor and train only
            the classifier head (used for Phase 1 warm-up).
        drop_rate: dropout probability on the classifier head.
        drop_path_rate: stochastic depth rate (regularization).
    """
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model

def set_backbone_trainable(model, trainable: bool = True):
    """Freeze/unfreeze the whole backbone (used to switch between phases)."""
    for param in model.parameters():
        param.requires_grad = trainable
    # The head always stays trainable.
    for param in model.get_classifier().parameters():
        param.requires_grad = True

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Full fine-tuning view (Phase 2)
    m_full = build_model(freeze_backbone=False)
    total = sum(p.numel() for p in m_full.parameters())
    trainable_full = count_trainable_params(m_full)
    print(f"Backbone: {DEFAULT_BACKBONE}")
    print(f"[Phase 2 - full] Total: {total:,}  Trainable: {trainable_full:,} ({100*trainable_full/total:.1f}%)")

    # Warm-up view (Phase 1)
    m_head = build_model(freeze_backbone=True)
    trainable_head = count_trainable_params(m_head)
    print(f"[Phase 1 - head only] Trainable: {trainable_head:,} ({100*trainable_head/total:.1f}%)")
