# src/data/dataTransforms.py
from __future__ import annotations

from typing import Dict
from torchvision import transforms

# ImageNet stats (for pretrained MobileNetV2/EfficientNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def buildTransforms(inputSize: int = 224) -> Dict[str, transforms.Compose]:
    """
    Return dict: {"train": ..., "val": ..., "test": ...}
    Using safe augmentations for leaf disease classification.
    """
    trainTf = transforms.Compose([
        transforms.RandomResizedCrop(inputSize, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    evalTf = transforms.Compose([
        transforms.Resize(int(inputSize * 1.14)),  # ~256 when inputSize=224
        transforms.CenterCrop(inputSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {"train": trainTf, "val": evalTf, "test": evalTf}