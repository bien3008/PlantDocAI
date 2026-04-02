# src/data/plantVillageDataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

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
        autoResolveRoot: bool = True,
        maxResolveDepth: int = 3,
    ) -> None:
        self.originalRootDir = os.path.abspath(rootDir)
        self.rootDir = self.originalRootDir
        self.transform = transform
        self.returnPath = returnPath

        if samples is None:
            if autoResolveRoot:
                self.rootDir = self._resolveDatasetRoot(self.originalRootDir, maxDepth=maxResolveDepth)
            self.classToId = self._scanClasses(self.rootDir)
            self.idToClass = {v: k for k, v in self.classToId.items()}
            self.samples = self._scanSamples(self.rootDir, self.classToId)
        else:
            self.samples = samples
            self.classToId = {}
            self.idToClass = {}

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found. rootDir={self.rootDir}. Check dataset structure."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        imagePath = item.imagePath
        if not os.path.isabs(imagePath):
            imagePath = os.path.join(self.rootDir, imagePath)

        try:
            with Image.open(imagePath) as img:
                img = img.convert("RGB")
        except Exception:
            return None

        if self.transform is not None:
            img = self.transform(img)

        if self.returnPath:
            return img, item.labelId, imagePath
        return img, item.labelId

    @staticmethod
    def _isClassFolder(dirPath: str):
        if not os.path.isdir(dirPath):
            return False
        for dp, _, fns in os.walk(dirPath):
            for fn in fns:
                if fn.lower().endswith(IMG_EXTS):
                    return True
        return False

    @classmethod
    def _looksLikeClassLevel(cls, rootDir: str, minClasses: int = 2):
        if not os.path.isdir(rootDir):
            return False
        subDirs = []
        for name in os.listdir(rootDir):
            full = os.path.join(rootDir, name)
            if os.path.isdir(full) and not name.startswith("."):
                subDirs.append(full)

        classLike = 0
        for d in subDirs:
            if cls._isClassFolder(d):
                classLike += 1
        return classLike >= minClasses

    @classmethod
    def _resolveDatasetRoot(cls, startDir: str, maxDepth: int = 3):
        startDir = os.path.abspath(startDir)
        if not os.path.isdir(startDir):
            raise FileNotFoundError(f"Dataset rootDir not found: {startDir}")

        if cls._looksLikeClassLevel(startDir):
            return startDir

        queue = [(startDir, 0)]
        candidates = []

        while queue:
            cur, depth = queue.pop(0)
            if depth >= maxDepth:
                continue

            try:
                names = os.listdir(cur)
            except Exception:
                continue

            for name in names:
                full = os.path.join(cur, name)
                if os.path.isdir(full) and not name.startswith("."):
                    if cls._looksLikeClassLevel(full):
                        candidates.append(full)
                    queue.append((full, depth + 1))

        if candidates:
            def score(path: str):
                cnt = 0
                for name in os.listdir(path):
                    full = os.path.join(path, name)
                    if os.path.isdir(full) and cls._isClassFolder(full):
                        cnt += 1
                return cnt

            candidates.sort(key=score, reverse=True)
            return candidates[0]

        topEntries = []
        try:
            topEntries = os.listdir(startDir)[:20]
        except Exception:
            pass
        raise ValueError(
            "No class folders found under: "
            f"{startDir}\n"
            f"Top entries: {topEntries}\n"
            "Hint: point --dataDir to the folder that directly contains class folders, e.g. .../color/"
        )

    @staticmethod
    def _scanClasses(rootDir: str):
        classNames = []
        for name in os.listdir(rootDir):
            full = os.path.join(rootDir, name)
            if os.path.isdir(full) and not name.startswith("."):
                classNames.append(name)

        classNames.sort()
        if len(classNames) == 0:
            raise ValueError(f"No class folders found under: {rootDir}")

        return {c: i for i, c in enumerate(classNames)}

    @staticmethod
    def _scanSamples(rootDir: str, classToId: Dict[str, int]):
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