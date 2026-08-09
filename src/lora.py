"""
Lightweight LoRA (Low-Rank Adaptation) implementation for parameter-efficient
fine-tuning of the frozen RETFound ViT-Large backbone.

Why LoRA here:
- The 300M-param ViT-Large backbone stays FROZEN.
- We inject small low-rank matrices (A, B) into the attention linear layers.
- Only these adapters + the classifier head are trained (~1M params),
  so it fits an 8 GB GPU and resists overfitting on ~2.5k images.
"""

import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear. The original weights stay frozen; we add a
    trainable low-rank update:  delta = (x @ A^T @ B^T) * (alpha / r)

    B is initialized to zero so that, at the start of training, the module
    behaves EXACTLY like the original pretrained layer (delta = 0). This is
    what makes LoRA safe to attach to a strong pretrained model.
    """

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), "LoRALinear expects an nn.Linear"

        self.base = base_linear
        # Freeze the pretrained projection.
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Low-rank trainable matrices.
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # start as a no-op

    def forward(self, x):
        base_out = self.base(x)                         # frozen path
        lora_out = self.dropout(x) @ self.lora_A.t()    # [..., r]
        lora_out = lora_out @ self.lora_B.t()           # [..., out]
        return base_out + lora_out * self.scaling

def inject_lora(model, target_modules=("qkv", "proj"), r=8, alpha=16, dropout=0.05, verbose=True):
    """
    Replace every nn.Linear whose path component matches target_modules with a
    LoRALinear. For a timm ViT this hits attention 'qkv' and 'proj' layers in
    all transformer blocks.

    We collect targets first, THEN replace, to avoid mutating modules while
    iterating over them.
    """
    to_replace = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                parts = full_name.split(".")
                if any(t in parts for t in target_modules):
                    to_replace.append((module, child_name, child, full_name))

    for module, child_name, child, full_name in to_replace:
        setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        if verbose:
            print(f"  LoRA -> {full_name}  (in={child.in_features}, out={child.out_features})")

    if verbose:
        print(f"Injected LoRA into {len(to_replace)} linear layers "
              f"(r={r}, alpha={alpha}, dropout={dropout}).")
    return model

def mark_only_lora_and_head_trainable(model, extra_trainable=("head",)):
    """
    Freeze everything except LoRA adapters and the classifier head
    (and anything else listed in extra_trainable).
    """
    extra = set(extra_trainable)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif set(name.split(".")) & extra:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def enable_grad_checkpointing(model, verbose=True):
    """
    Trade compute for memory: activations are recomputed in the backward pass
    instead of stored. This is what lets ViT-Large + LoRA train on 8 GB.
    Tries the timm API first, then a common fallback.
    """
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        if verbose:
            print("Gradient checkpointing enabled via set_grad_checkpointing(True).")
    elif hasattr(model, "grad_checkpointing"):
        model.grad_checkpointing = True
        if verbose:
            print("Gradient checkpointing enabled via model.grad_checkpointing = True.")
    else:
        if verbose:
            print("WARNING: could not enable gradient checkpointing on this model.")
    return model

def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def lora_state_dict(model):
    """
    Extract ONLY the trainable pieces (LoRA adapters + head) so checkpoints stay
    tiny (~a few MB) instead of saving the whole 300M-param model.
    """
    keep = {}
    for k, v in model.state_dict().items():
        if ("lora_A" in k) or ("lora_B" in k) or k.startswith("head.") or ".head." in f".{k}":
            keep[k] = v
    return keep

def load_lora_state_dict(model, state_dict, strict=False):
    """Load a checkpoint produced by lora_state_dict back into a LoRA-injected model."""
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return missing, unexpected
