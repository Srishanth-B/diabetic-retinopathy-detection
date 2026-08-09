"""
Data loading and preparation for the APTOS 2019 Diabetic Retinopathy dataset.

Source: Aravind Eye Hospital (India) / Kaggle APTOS 2019 Blindness Detection
Mirrored on Hugging Face at: bumbledeep/aptos (224x224, MIT license)
Original: https://www.kaggle.com/competitions/aptos2019-blindness-detection

Labels (Davis / ICDR-style severity scale):
    0 - No DR
    1 - Mild
    2 - Moderate
    3 - Severe
    4 - Proliferative DR

The upstream dataset ships as a single split, so we create our own
stratified train/val/test split here and keep the split reproducible
via a fixed random seed.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_DATASET_ID = "bumbledeep/aptos"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(image_size: int = 224, train: bool = True):
    """Return torchvision transforms for train or eval mode.

    Train transforms include mild augmentation appropriate for fundus
    images (random flips/rotation, slight color jitter). Fundus images
    are roughly rotationally symmetric so flips/rotation are safe
    augmentations, unlike for e.g. natural scene photos with a horizon.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

class DRDataset(Dataset):
    """Wraps a Hugging Face dataset split as a PyTorch Dataset."""

    def __init__(self, hf_dataset, indices, transform=None):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        example = self.hf_dataset[real_idx]
        image = example["image"].convert("RGB")
        label = example["label_code"]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_splits(image_size: int = 224, seed: int = 42, val_size: float = 0.15, test_size: float = 0.15):
    """Load APTOS dataset from Hugging Face and create stratified splits.

    Returns:
        train_ds, val_ds, test_ds: DRDataset instances
        class_weights: torch.Tensor of shape (5,) for use in a weighted
            loss to counteract class imbalance (~49% of images are
            "No DR" vs ~5% "Severe").
    """
    raw = load_dataset(HF_DATASET_ID)["train"]
    labels = np.array(raw["label_code"])
    all_indices = np.arange(len(raw))

    # First split off test set, then split remainder into train/val.
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=test_size, random_state=seed, stratify=labels
    )
    train_val_labels = labels[train_val_idx]
    val_fraction_of_remainder = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_remainder,
        random_state=seed,
        stratify=train_val_labels,
    )

    train_ds = DRDataset(raw, train_idx, transform=build_transforms(image_size, train=True))
    val_ds = DRDataset(raw, val_idx, transform=build_transforms(image_size, train=False))
    test_ds = DRDataset(raw, test_idx, transform=build_transforms(image_size, train=False))

    # Inverse-frequency class weights computed from the training split only.
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return train_ds, val_ds, test_ds, class_weights

if __name__ == "__main__":
    train_ds, val_ds, test_ds, weights = load_splits()
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print("class weights:", weights)
