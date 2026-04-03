# scripts/smokeTestData.py
import argparse
import os

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.seedUtils import setGlobalSeed
from src.data.dataSplit import SplitConfig, createSplits
from src.data.dataLoader import buildDataLoaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, required=True, help="Path to PlantVillage root folder")
    parser.add_argument("--outDir", type=str, required=True, help="Output dir for split CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inputSize", type=int, default=224)
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--numWorkers", type=int, default=2)
    args = parser.parse_args()

    setGlobalSeed(args.seed)

    splitCfg = SplitConfig(trainRatio=0.8, valRatio=0.1, testRatio=0.1, seed=args.seed)
    paths = createSplits(dataDir=args.dataDir, outDir=args.outDir, splitConfig=splitCfg)
    print("✅ Splits created:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    loaders, classToId = buildDataLoaders(
        dataDir=args.dataDir,
        splitDir=args.outDir,
        inputSize=args.inputSize,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers,
        returnPath=True,
    )

    print(f"\n✅ Found {len(classToId)} classes.")
    # show first 5 classes
    firstFew = list(classToId.items())[:5]
    print("   sample class mapping:", firstFew)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device check: {device}")

    batch = next(iter(loaders["train"]))
    if batch is None:
        raise RuntimeError("All samples in the first batch are invalid (broken images?).")

    images, labels, paths = batch
    print("\n✅ Batch OK")
    print("   images:", tuple(images.shape))  # (B, 3, H, W)
    print("   labels:", tuple(labels.shape))
    print("   sample path:", paths[0])

if __name__ == "__main__":
    main()
    