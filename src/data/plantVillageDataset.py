# src/data/plantVillageDataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

@dataclass
class SampleItem:
    imagePath: str
    labelId: int

class PlantVillageDataset:
    def __init__(
        self,
        rootDir: str,
        samples: Optional[List[SampleItem]] = None,
        transform: Optional[Callable] = None,
        returnPath: bool = False,
    ) -> None:
        """
        Args:
            rootDir: The exact absolute or relative path to the dataset.
                     If samples is None, this directory MUST directly contain class folders.
            samples: Pre-defined list of SampleItems. Often loaded from a split CSV.
            transform: Transformations to apply to the images.
            returnPath: If True, returns (img, label, path).
        """
        self.rootDir = os.path.abspath(rootDir)
        if not os.path.isdir(self.rootDir):
            raise FileNotFoundError(f"Requested rootDir does not exist or is not a directory: {self.rootDir}")
            
        self.transform = transform
        self.returnPath = returnPath

        if samples is None:
            self.classToId = self._scanClasses(self.rootDir)
            self.idToClass = {v: k for k, v in self.classToId.items()}
            self.samples = self._scanSamples(self.rootDir, self.classToId)
        else:
            self.samples = samples
            self.classToId = {}
            self.idToClass = {}

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in rootDir='{self.rootDir}'. "
                "Ensure that the path directly contains class folders with images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        imagePath = item.imagePath
        
        # Ensure path is absolute for deterministic loading
        if not os.path.isabs(imagePath):
            imagePath = os.path.join(self.rootDir, imagePath)

        try:
            with Image.open(imagePath) as img:
                # Convert to RGB to ensure 3 channels
                img = img.convert("RGB")
        except Exception as e:
            # Fail fast if image is corrupted or missing
            raise RuntimeError(f"Failed to load image at {imagePath}. Error: {e}")

        if self.transform is not None:
            img = self.transform(img)

        if self.returnPath:
            return img, item.labelId, imagePath
        return img, item.labelId

    @staticmethod
    def _scanClasses(rootDir: str) -> Dict[str, int]:
        classNames = []
        for name in os.listdir(rootDir):
            full = os.path.join(rootDir, name)
            if os.path.isdir(full) and not name.startswith("."):
                classNames.append(name)

        classNames.sort()
        if len(classNames) == 0:
            raise ValueError(f"No class folders found directly under: {rootDir}")

        return {c: i for i, c in enumerate(classNames)}

    @staticmethod
    def _scanSamples(rootDir: str, classToId: Dict[str, int]) -> List[SampleItem]:
        samples: List[SampleItem] = []
        for className, labelId in classToId.items():
            classDir = os.path.join(rootDir, className)
            for dirpath, _, filenames in os.walk(classDir):
                for fn in filenames:
                    if fn.lower().endswith(IMG_EXTS):
                        fullPath = os.path.join(dirpath, fn)
                        relPath = os.path.relpath(fullPath, rootDir)
                        samples.append(SampleItem(imagePath=relPath, labelId=labelId))
        return samples

    def getClassMapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        return self.classToId, self.idToClass