"""
scripts/smokeTestAug.py
━━━━━━━━━━━━━━━━━━━━━━
Kiểm tra nhanh pipeline augmentation mà không cần chạy toàn bộ vòng lặp training.

Nội dung kiểm tra:
  1. AugConfig parse đúng từ baseline.yaml.
  2. buildTransforms() trả về transform callable cho train / val / test.
  3. Một ảnh đơn từ train split có thể được load và transform không có lỗi.
  4. Chạy một mini-batch qua DataLoader (4 ảnh) — kiểm tra shape tensor và chuẩn hóa.
  5. In PASS / FAIL rõ ràng với lý do cho mỗi kiểm tra.

Cách chạy:
  python scripts/smokeTestAug.py
  python scripts/smokeTestAug.py --config configs/baseline.yaml
  python scripts/smokeTestAug.py --saveGrid  # lưu grid ảnh augmented để kiểm tra bằng mắt
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Ép output UTF-8 để ký tự Unicode hiển thị đúng trên Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Thêm thư mục gốc project vào path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from src.utils.configUtils import loadTrainingConfig
from src.data.dataTransforms import buildTransforms, AugConfig
from src.data.dataLoader import buildDataLoaders
from src.data.dataSplit import loadSplitCsv

# ─────────────────────────────────────────────────────────────────────────────
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    detail_str = f"  → {detail}" if detail else ""
    print(f"  {status}  {label}{detail_str}")
    return condition


# ─────────────────────────────────────────────────────────────────────────────

def runSmokeTest(configPath: str, saveGrid: bool = False) -> bool:
    print(f"\n{'='*60}")
    print(f"  PlantDocAI  -  Kiểm tra nhanh Augmentation")
    print(f"  Config: {configPath}")
    print(f"{'='*60}\n")

    all_pass = True

    # ── 1. Parse config ───────────────────────────────────────────────────────
    try:
        config = loadTrainingConfig(configPath)
        aug = config.augConfig
        all_pass &= check(
            "Config load và augConfig được điền đúng",
            isinstance(aug, AugConfig),
            f"rotationDegrees={aug.rotationDegrees}, hueJitter={aug.hueJitter} "
            f"(phải là 0.0), elasticEnabled={aug.elasticEnabled}",
        )
        all_pass &= check(
            "hueJitter bằng 0.0 (màu bệnh KHÔNG bị thay đổi)",
            aug.hueJitter == 0.0,
            "QUAN TRỌNG: hue khác 0 phá hủy tín hiệu màu sắc bệnh lá",
        )
        all_pass &= check(
            "cropScaleMin >= 0.60 (không crop mất tổn thương)",
            aug.cropScaleMin >= 0.60,
            f"cropScaleMin={aug.cropScaleMin}",
        )
        all_pass &= check(
            "eraseScaleMax <= 0.15 (vùng xóa giữ rất nhỏ)",
            aug.eraseScaleMax <= 0.15,
            f"eraseScaleMax={aug.eraseScaleMax}",
        )
    except Exception as exc:
        print(f"  {FAIL}  Parse config thất bại: {exc}")
        return False

    # ── 2. Kiểm tra buildTransforms ──────────────────────────────────────────
    try:
        tf = buildTransforms(inputSize=config.imageSize, aug=aug)
        all_pass &= check(
            "buildTransforms() trả về đủ keys train/val/test",
            all(k in tf for k in ("train", "val", "test")),
        )
        all_pass &= check(
            "Train transform có thể gọi được (callable)",
            callable(tf["train"]),
        )
        all_pass &= check(
            "Val transform có thể gọi được (callable)",
            callable(tf["val"]),
        )
    except Exception as exc:
        print(f"  {FAIL}  buildTransforms thất bại: {exc}")
        return False

    # ── 3. Kiểm tra biến đổi một ảnh đơn ────────────────────────────────────
    pil_img = None   # giữ biến để dùng cho --saveGrid
    try:
        train_csv = Path(config.splitDir) / "train.csv"
        if not train_csv.exists():
            print(f"  [BỎ QUA] Kiểm tra ảnh đơn: không tìm thấy split CSV tại {train_csv}")
            print("           Chạy `python scripts/train.py` một lần để tự động tạo splits, rồi chạy lại test này.")
        else:
            samples = loadSplitCsv(str(train_csv))
            sample = samples[0]

            img_path = sample.imagePath
            if not Path(img_path).is_absolute():
                img_path = str(Path(config.dataDir) / img_path)

            with Image.open(img_path) as _img:
                pil_img = _img.convert("RGB")
                tensor = tf["train"](pil_img)

            all_pass &= check(
                "Ảnh đơn biến đổi thành tensor không có lỗi",
                isinstance(tensor, torch.Tensor),
                f"shape={tuple(tensor.shape)}",
            )
            expected_shape = (3, config.imageSize, config.imageSize)
            all_pass &= check(
                f"Shape tensor đầu ra đúng {expected_shape}",
                tuple(tensor.shape) == expected_shape,
                f"thực tế={tuple(tensor.shape)}",
            )
            # Sau khi normalize, giá trị pixel nên nằm trong khoảng [-3, 3]
            in_range = tensor.min().item() > -4.0 and tensor.max().item() < 4.0
            all_pass &= check(
                "Giá trị pixel nằm trong dải đã chuẩn hóa [-4, 4]",
                in_range,
                f"min={tensor.min():.3f}  max={tensor.max():.3f}",
            )
    except Exception as exc:
        print(f"  [THẤT BẠI] Kiểm tra ảnh đơn lỗi: {exc}")
        all_pass = False

    # ── 4. Kiểm tra mini-batch từ DataLoader ─────────────────────────────────
    train_csv = Path(config.splitDir) / "train.csv"
    if not train_csv.exists():
        print("  [BỎ QUA] Kiểm tra DataLoader batch: không tìm thấy split CSV (xem gợi ý bên trên).")
    else:
        try:
            loaders, classToId = buildDataLoaders(
                dataDir=config.dataDir,
                splitDir=config.splitDir,
                inputSize=config.imageSize,
                batchSize=4,
                numWorkers=0,   # 0 workers để tránh vấn đề multiprocessing trên Windows
                aug=aug,
            )
            batch_imgs, batch_labels = next(iter(loaders["train"]))
            all_pass &= check(
                "DataLoader trả về batch không có lỗi",
                batch_imgs.ndim == 4,
                f"shape ảnh={tuple(batch_imgs.shape)}  shape nhãn={tuple(batch_labels.shape)}",
            )
            all_pass &= check(
                "Batch ảnh có số kênh màu đúng (C=3)",
                batch_imgs.shape[1] == 3,
                f"C={batch_imgs.shape[1]}",
            )
            all_pass &= check(
                f"Chiều không gian batch khớp imageSize={config.imageSize}",
                batch_imgs.shape[2] == config.imageSize and batch_imgs.shape[3] == config.imageSize,
                f"H={batch_imgs.shape[2]}  W={batch_imgs.shape[3]}",
            )
        except Exception as exc:
            print(f"  [THẤT BẠI] Kiểm tra DataLoader batch lỗi: {exc}")
            all_pass = False

    # ── 5. Grid ảnh augmented để kiểm tra bằng mắt (tùy chọn) ───────────────
    if saveGrid and pil_img is not None:
        _saveAugGrid(pil_img, tf["train"], config, n=8)
    elif saveGrid:
        print("  [BỎ QUA] --saveGrid bị bỏ qua: không có ảnh nào được load (splits chưa có).")

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    overall = "TẤT CẢ KIỂM TRA ĐỀU ĐẠT" if all_pass else "MỘT SỐ KIỂM TRA THẤT BẠI"
    print(f"  Kết quả: {overall}")
    print(f"{'='*60}\n")

    return all_pass


def _saveAugGrid(pil_img: Image.Image, trainTf, config, n: int = 8):
    """
    Lưu grid gồm n phiên bản augmented của pil_img để kiểm tra bằng mắt
    xem augmentation có hợp lệ về mặt sinh học không.
    """
    try:
        import torchvision.utils as vutils

        # Giá trị để de-normalize ảnh trước khi lưu
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tensors = []
        for _ in range(n):
            t = trainTf(pil_img)                      # tensor đã normalize
            t = t * IMAGENET_STD + IMAGENET_MEAN      # de-normalize để hiển thị
            t = t.clamp(0, 1)
            tensors.append(t)

        grid = vutils.make_grid(tensors, nrow=4, padding=4)
        from torchvision.transforms.functional import to_pil_image
        grid_pil = to_pil_image(grid)

        out_path = Path(config.outputDir) / "aug_smoke_grid.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        grid_pil.save(str(out_path))
        print(f"  [INFO] Grid augmentation đã lưu → {out_path}")
    except Exception as exc:
        print(f"  [CẢNH BÁO] Không thể lưu grid: {exc}")


# ─────────────────────────────────────────────────────────────────────────────

def parseArgs():
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh pipeline augmentation")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Đường dẫn tới file config YAML")
    parser.add_argument(
        "--saveGrid",
        action="store_true",
        help="Lưu grid ảnh augmented vào outputDir/aug_smoke_grid.png để kiểm tra bằng mắt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    ok = runSmokeTest(configPath=args.config, saveGrid=args.saveGrid)
    sys.exit(0 if ok else 1)
