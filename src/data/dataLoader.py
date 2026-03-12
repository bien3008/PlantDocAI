# src/data/dataLoader.py
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any, List

import torch
from torch.utils.data import DataLoader

from .plantVillageDataset import PlantVillageDataset
from .dataSplit import loadSplitCsv
from .dataTransforms import buildTransforms

def _safeCollate(batch: List[Any]):
    """
    Filter out None samples (e.g., broken images) so training doesn't crash.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def buildDataLoaders(
    dataDir: str,
    splitDir: str,
    inputSize: int = 224,
    batchSize: int = 32,
    numWorkers: int = 2,
    pinMemory: Optional[bool] = None,
    returnPath: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """
    Build dataloaders from split CSVs in splitDir.
    Returns (loaders, classToId).
    """
    tf = buildTransforms(inputSize=inputSize)

    trainSamples = loadSplitCsv(f"{splitDir}/train.csv")
    valSamples = loadSplitCsv(f"{splitDir}/val.csv")
    testSamples = loadSplitCsv(f"{splitDir}/test.csv")

    trainDs = PlantVillageDataset(rootDir=dataDir, samples=trainSamples, transform=tf["train"], returnPath=returnPath)
    valDs = PlantVillageDataset(rootDir=dataDir, samples=valSamples, transform=tf["val"], returnPath=returnPath)
    testDs = PlantVillageDataset(rootDir=dataDir, samples=testSamples, transform=tf["test"], returnPath=returnPath)

 
    if pinMemory is None:
        pinMemory = bool(torch.cuda.is_available())

    trainLoader = DataLoader(
        trainDs, batch_size=batchSize, shuffle=True,
        num_workers=numWorkers, pin_memory=pinMemory,
        collate_fn=_safeCollate,
        drop_last=False,
    )
    valLoader = DataLoader(
        valDs, batch_size=batchSize, shuffle=False,
        num_workers=numWorkers, pin_memory=pinMemory,
        collate_fn=_safeCollate,
        drop_last=False,
    )
    testLoader = DataLoader(
        testDs, batch_size=batchSize, shuffle=False,
        num_workers=numWorkers, pin_memory=pinMemory,
        collate_fn=_safeCollate,
        drop_last=False,
    )

    scanDs = PlantVillageDataset(rootDir=dataDir, samples=None, transform=None)
    classToId, _ = scanDs.getClassMapping()

    return {"train": trainLoader, "val": valLoader, "test": testLoader}, classToId