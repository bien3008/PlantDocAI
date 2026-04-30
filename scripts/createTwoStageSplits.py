"""
scripts/createTwoStageSplits.py
================================
Tạo split CSV cho chiến lược huấn luyện 2 giai đoạn.

Stage 1 splits: Từ PlantVillage, lọc theo core classes, split 80/10/10.
Stage 2 splits: Từ PlantDoc train, lọc theo core classes, split 80/20/0.
                (không tách test từ PlantDoc train — dùng PlantDoc test riêng)
PlantDoc test:  Dùng nguyên folder test của PlantDoc (đã lọc core classes) làm benchmark.

Cách chạy:
  python scripts/createTwoStageSplits.py

Outputs:
  data/splits/two_stage/stage1/train.csv, val.csv, test.csv, classes.csv
  data/splits/two_stage/stage2/train.csv, val.csv, classes.csv
  data/splits/two_stage/plantdoc_test/test.csv, classes.csv
"""
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split
from src.data.plantVillageDataset import IMG_EXTS, SampleItem

import argparse

# ── Đường dẫn mặc định ────────────────────────────────────────────────────────
DEFAULT_PV_DIR = str(ROOT / "data" / "extended" / "plantVillage" / "train")
DEFAULT_PD_TRAIN_DIR = str(ROOT / "data" / "extended" / "PlantDoc_Dataset_master" / "train")
DEFAULT_PD_TEST_DIR = str(ROOT / "data" / "extended" / "PlantDoc_Dataset_master" / "test")
DEFAULT_CORE_CLASSES_CSV = str(ROOT / "data" / "splits" / "two_stage" / "core_classes.csv")
DEFAULT_OUTPUT_BASE = str(ROOT / "data" / "splits" / "two_stage")

SEED = 42

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create splits for Two-Stage Training")
    parser.add_argument("--pvDir", type=str, default=DEFAULT_PV_DIR, help="Path to PlantVillage training data")
    parser.add_argument("--pdTrainDir", type=str, default=DEFAULT_PD_TRAIN_DIR, help="Path to PlantDoc training data")
    parser.add_argument("--pdTestDir", type=str, default=DEFAULT_PD_TEST_DIR, help="Path to PlantDoc test data")
    parser.add_argument("--coreClassesCsv", type=str, default=DEFAULT_CORE_CLASSES_CSV, help="Path to core_classes.csv")
    parser.add_argument("--outBase", type=str, default=DEFAULT_OUTPUT_BASE, help="Base output directory for splits")
    return parser.parse_args()


def _loadCoreClasses(csvPath: Path) -> Dict[str, int]:
    """Đọc core_classes.csv → {className: labelId}."""
    if not csvPath.exists():
        raise FileNotFoundError(
            f"core_classes.csv not found at {csvPath}. "
            "Run buildCoreClassMapping.py first."
        )
    mapping = {}
    with open(csvPath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["className"]] = int(row["labelId"])
    return mapping


def _scanAndFilter(dataDir: Path, coreMapping: Dict[str, int]) -> List[SampleItem]:
    """Quét tất cả ảnh trong dataDir, chỉ giữ lại class thuộc coreMapping, dùng labelId mới."""
    samples = []
    for d in sorted(dataDir.iterdir()):
        if not d.is_dir():
            continue
        if d.name not in coreMapping:
            continue
        labelId = coreMapping[d.name]
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                relPath = f.relative_to(dataDir)
                samples.append(SampleItem(imagePath=str(relPath), labelId=labelId))
    return samples


def _writeSamplesCsv(outPath: Path, samples: List[SampleItem]):
    outPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["imagePath", "labelId"])
        writer.writeheader()
        for s in samples:
            writer.writerow({"imagePath": s.imagePath, "labelId": s.labelId})


def _writeClassesCsv(outPath: Path, coreMapping: Dict[str, int]):
    outPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["className", "labelId"])
        writer.writeheader()
        for className, labelId in sorted(coreMapping.items(), key=lambda x: x[1]):
            writer.writerow({"className": className, "labelId": labelId})


def _printDistribution(samples: List[SampleItem], coreMapping: Dict[str, int], label: str):
    idToClass = {v: k for k, v in coreMapping.items()}
    counts = Counter(s.labelId for s in samples)
    print(f"\n  [{label}] Total: {len(samples)} samples")
    for lid in sorted(counts.keys()):
        print(f"    {lid:>2}  {idToClass[lid]:<50} {counts[lid]:>5}")


def main():
    args = parseArgs()
    pv_dir = Path(args.pvDir)
    pd_train_dir = Path(args.pdTrainDir)
    pd_test_dir = Path(args.pdTestDir)
    core_classes_csv = Path(args.coreClassesCsv)
    out_base = Path(args.outBase)

    print("=" * 70)
    print("  PlantDocAI — Two-Stage Split Generator")
    print("=" * 70)

    # 1. Load core classes
    coreMapping = _loadCoreClasses(core_classes_csv)
    numClasses = len(coreMapping)
    print(f"\n[INFO] Loaded {numClasses} core classes from {core_classes_csv}")

    # STAGE 1 -- PlantVillage (core classes), split 80/10/10
    print("\n" + "-" * 70)
    print("  STAGE 1: PlantVillage (core classes)")
    print("-" * 70)

    pvSamples = _scanAndFilter(pv_dir, coreMapping)
    pvLabels = [s.labelId for s in pvSamples]
    print(f"[INFO] PlantVillage core samples: {len(pvSamples)}")

    # Split 80/10/10
    pvTrain, pvTemp, _, tempLabels = train_test_split(
        pvSamples, pvLabels, test_size=0.20, random_state=SEED, stratify=pvLabels
    )
    pvVal, pvTest, _, _ = train_test_split(
        pvTemp, tempLabels, test_size=0.50, random_state=SEED, stratify=tempLabels
    )

    s1Dir = out_base / "stage1"
    _writeSamplesCsv(s1Dir / "train.csv", pvTrain)
    _writeSamplesCsv(s1Dir / "val.csv", pvVal)
    _writeSamplesCsv(s1Dir / "test.csv", pvTest)
    _writeClassesCsv(s1Dir / "classes.csv", coreMapping)

    _printDistribution(pvTrain, coreMapping, "Stage1 Train")
    _printDistribution(pvVal, coreMapping, "Stage1 Val")
    _printDistribution(pvTest, coreMapping, "Stage1 Test")

    print(f"\n[OK] Stage 1 splits saved to: {s1Dir}")

    # STAGE 2 -- PlantDoc train (core classes), split 80/20/0
    print("\n" + "-" * 70)
    print("  STAGE 2: PlantDoc train (core classes)")
    print("-" * 70)

    pdTrainSamples = _scanAndFilter(pd_train_dir, coreMapping)
    pdLabels = [s.labelId for s in pdTrainSamples]
    print(f"[INFO] PlantDoc train core samples: {len(pdTrainSamples)}")

    # Split 80/20 (train/val only — test is separate)
    pdTrain, pdVal, _, _ = train_test_split(
        pdTrainSamples, pdLabels, test_size=0.20, random_state=SEED, stratify=pdLabels
    )

    s2Dir = out_base / "stage2"
    _writeSamplesCsv(s2Dir / "train.csv", pdTrain)
    _writeSamplesCsv(s2Dir / "val.csv", pdVal)
    # Stage 2 cần một test.csv (dù rỗng hoặc dùng val) vì buildDataLoaders luôn đọc test.csv
    # → Dùng val làm test placeholder để pipeline không crash
    _writeSamplesCsv(s2Dir / "test.csv", pdVal)
    _writeClassesCsv(s2Dir / "classes.csv", coreMapping)

    _printDistribution(pdTrain, coreMapping, "Stage2 Train")
    _printDistribution(pdVal, coreMapping, "Stage2 Val")

    print(f"\n[OK] Stage 2 splits saved to: {s2Dir}")

    # PLANTDOC TEST -- Final real-world benchmark
    print("\n" + "-" * 70)
    print("  PLANTDOC TEST: Final real-world benchmark")
    print("-" * 70)

    pdTestSamples = _scanAndFilter(pd_test_dir, coreMapping)
    print(f"[INFO] PlantDoc test core samples: {len(pdTestSamples)}")

    pdTestDir = out_base / "plantdoc_test"
    _writeSamplesCsv(pdTestDir / "test.csv", pdTestSamples)
    # Cũng cần train/val placeholders cho buildDataLoaders nếu cần
    _writeSamplesCsv(pdTestDir / "train.csv", [])
    _writeSamplesCsv(pdTestDir / "val.csv", [])
    _writeClassesCsv(pdTestDir / "classes.csv", coreMapping)

    _printDistribution(pdTestSamples, coreMapping, "PlantDoc Test")

    print(f"\n[OK] PlantDoc test benchmark saved to: {pdTestDir}")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Core classes:      {numClasses}")
    print(f"  Stage 1 (PV):      {len(pvTrain)} train / {len(pvVal)} val / {len(pvTest)} test")
    print(f"  Stage 2 (PD):      {len(pdTrain)} train / {len(pdVal)} val")
    print(f"  PlantDoc Test:     {len(pdTestSamples)} (real-world benchmark)")
    print("=" * 70)


if __name__ == "__main__":
    main()
