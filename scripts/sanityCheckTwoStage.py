"""Quick sanity check: verify Stage 1 and Stage 2 data pipelines work."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataLoader import buildDataLoaders

print("=" * 60)
print("  Stage 1 DataLoader Check")
print("=" * 60)
loaders, classToId = buildDataLoaders(
    dataDir="data/extended/plantVillage/train",
    splitDir="data/splits/two_stage/stage1",
    inputSize=224, batchSize=32, numWorkers=0,
)
idToClass = {v: k for k, v in classToId.items()}
classNames = [idToClass[i] for i in range(len(classToId))]
print(f"Classes: {len(classNames)}")
print(f"Train batches: {len(loaders['train'])}")
print(f"Val batches:   {len(loaders['val'])}")
print(f"Test batches:  {len(loaders['test'])}")

# Quick batch check
batch = next(iter(loaders["train"]))
images, labels = batch
print(f"Batch shape: {images.shape}, Labels range: [{labels.min().item()}, {labels.max().item()}]")

print()
print("=" * 60)
print("  Stage 2 DataLoader Check")
print("=" * 60)
loaders2, classToId2 = buildDataLoaders(
    dataDir="data/extended/PlantDoc_Dataset_master/train",
    splitDir="data/splits/two_stage/stage2",
    inputSize=224, batchSize=16, numWorkers=0,
)
idToClass2 = {v: k for k, v in classToId2.items()}
classNames2 = [idToClass2[i] for i in range(len(classToId2))]
print(f"Classes: {len(classNames2)}")
print(f"Train batches: {len(loaders2['train'])}")
print(f"Val batches:   {len(loaders2['val'])}")

batch2 = next(iter(loaders2["train"]))
images2, labels2 = batch2
print(f"Batch shape: {images2.shape}, Labels range: [{labels2.min().item()}, {labels2.max().item()}]")

# Verify class consistency
assert classNames == classNames2, f"Class name mismatch between Stage 1 and Stage 2!"
print()
print("[OK] Class names match between Stage 1 and Stage 2!")
print(f"[OK] Both stages use {len(classNames)} classes: {classNames[:3]}...")
