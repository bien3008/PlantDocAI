"""
scripts/buildCoreClassMapping.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tính giao tập class giữa PlantVillage và PlantDoc Dataset.
Tạo file core_classes.csv chứa class mapping thống nhất cho chiến lược 2 giai đoạn.

Cách chạy:
  python scripts/buildCoreClassMapping.py

Outputs:
  data/splits/two_stage/core_classes.csv
"""
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.plantVillageDataset import IMG_EXTS

import argparse

# ── Đường dẫn mặc định ───────────────────────────────────────────────────────
DEFAULT_PV_DIR = str(ROOT / "data" / "extended" / "plantVillage" / "train")
DEFAULT_PD_TRAIN_DIR = str(ROOT / "data" / "extended" / "PlantDoc_Dataset_master" / "train")
DEFAULT_PD_TEST_DIR = str(ROOT / "data" / "extended" / "PlantDoc_Dataset_master" / "test")
DEFAULT_OUTPUT_DIR = str(ROOT / "data" / "splits" / "two_stage")

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Core Class Mapping for Two-Stage Training")
    parser.add_argument("--pvDir", type=str, default=DEFAULT_PV_DIR, help="Path to PlantVillage training data")
    parser.add_argument("--pdTrainDir", type=str, default=DEFAULT_PD_TRAIN_DIR, help="Path to PlantDoc training data")
    parser.add_argument("--pdTestDir", type=str, default=DEFAULT_PD_TEST_DIR, help="Path to PlantDoc test data")
    parser.add_argument("--outDir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for core_classes.csv")
    return parser.parse_args()


def _scanClassFolders(rootDir: Path) -> dict[str, int]:
    """Quét thư mục con chứa ảnh có tên dạng `___`. Trả về {className: imageCount}."""
    result = {}
    if not rootDir.exists():
        return result
    for d in sorted(rootDir.iterdir()):
        if not d.is_dir():
            continue
        if "___" not in d.name:
            continue
        count = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
        if count > 0:
            result[d.name] = count
    return result


def main():
    args = parseArgs()
    pv_dir = Path(args.pvDir)
    pd_train_dir = Path(args.pdTrainDir)
    pd_test_dir = Path(args.pdTestDir)
    out_dir = Path(args.outDir)

    print("=" * 70)
    print("  PlantDocAI — Core Class Mapping Builder")
    print("=" * 70)

    # 1. Quét cả hai dataset
    pvClasses = _scanClassFolders(pv_dir)
    pdTrainClasses = _scanClassFolders(pd_train_dir)
    pdTestClasses = _scanClassFolders(pd_test_dir)

    print(f"\n[INFO] PlantVillage classes found: {len(pvClasses)}")
    print(f"[INFO] PlantDoc train classes found: {len(pdTrainClasses)}")
    print(f"[INFO] PlantDoc test classes found:  {len(pdTestClasses)}")

    # 2. Tính giao tập — class phải có mặt ở CẢ PV, PlantDoc train, VÀ PlantDoc test
    coreClassNames = sorted(
        set(pvClasses.keys()) & set(pdTrainClasses.keys()) & set(pdTestClasses.keys())
    )

    if len(coreClassNames) == 0:
        print("[ERROR] Không tìm thấy class chung giữa hai dataset!")
        sys.exit(1)

    print(f"\n[INFO] Core classes (intersection): {len(coreClassNames)}")
    print("-" * 70)
    print(f"{'#':<4} {'Class Name':<50} {'PV':>6} {'PD-Tr':>6} {'PD-Te':>6}")
    print("-" * 70)
    for i, name in enumerate(coreClassNames):
        pv = pvClasses.get(name, 0)
        pdTr = pdTrainClasses.get(name, 0)
        pdTe = pdTestClasses.get(name, 0)
        print(f"{i:<4} {name:<50} {pv:>6} {pdTr:>6} {pdTe:>6}")
    print("-" * 70)

    # 3. Classes chỉ có ở PV (bị loại)
    pvOnly = sorted(set(pvClasses.keys()) - set(coreClassNames))
    if pvOnly:
        print(f"\n[INFO] PV-only classes (excluded from core): {len(pvOnly)}")
        for name in pvOnly:
            print(f"  - {name} ({pvClasses[name]} images)")

    # 4. Lưu core_classes.csv
    out_dir.mkdir(parents=True, exist_ok=True)
    outPath = out_dir / "core_classes.csv"
    with open(outPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["className", "labelId"])
        writer.writeheader()
        for i, name in enumerate(coreClassNames):
            writer.writerow({"className": name, "labelId": i})

    print(f"\n[OK] Saved core class mapping to: {outPath}")
    print(f"     Total core classes: {len(coreClassNames)}")


if __name__ == "__main__":
    main()
