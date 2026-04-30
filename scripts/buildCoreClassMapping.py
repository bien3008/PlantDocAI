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

# ── Đường dẫn mặc định ───────────────────────────────────────────────────────
PV_DIR = ROOT / "data" / "extended" / "plantVillage" / "train"
PD_TRAIN_DIR = ROOT / "data" / "extended" / "PlantDoc_Dataset_master" / "train"
PD_TEST_DIR = ROOT / "data" / "extended" / "PlantDoc_Dataset_master" / "test"
OUTPUT_DIR = ROOT / "data" / "splits" / "two_stage"


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
    print("=" * 70)
    print("  PlantDocAI — Core Class Mapping Builder")
    print("=" * 70)

    # 1. Quét cả hai dataset
    pvClasses = _scanClassFolders(PV_DIR)
    pdTrainClasses = _scanClassFolders(PD_TRAIN_DIR)
    pdTestClasses = _scanClassFolders(PD_TEST_DIR)

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outPath = OUTPUT_DIR / "core_classes.csv"
    with open(outPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["className", "labelId"])
        writer.writeheader()
        for i, name in enumerate(coreClassNames):
            writer.writerow({"className": name, "labelId": i})

    print(f"\n[OK] Saved core class mapping to: {outPath}")
    print(f"     Total core classes: {len(coreClassNames)}")


if __name__ == "__main__":
    main()
