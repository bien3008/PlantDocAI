 
# PlantDoc AI 🌿🩺
Hệ thống nhận diện bệnh cây trồng từ ảnh lá cây, tích hợp **Explainable AI (Grad-CAM)** và **khuyến nghị xử lý**. Dự án phục vụ học phần **“Thực tập cơ sở”** (GVHD: **Thầy Nguyễn Xuân Đức**).

---

## 1) Demo nhanh (Quickstart)
### 1.1 Cài môi trường
> Khuyến nghị Python **3.9+** (3.10/3.11 đều ổn).

```bash
# Clone repo
git clone https://github.com/bien3008/PlantDoc_AI.git
cd PlantDoc_AI
# Tạo môi trường ảo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt


### 1.2 Chạy Web App (Streamlit)
 
streamlit run app/main.py
 

### 1.3 Pipeline đầy đủ (tùy chọn)
  
# 1) Chuẩn bị/split dữ liệu
python scripts/prepare_data.py --config configs/default.yaml

# 2) Train model
python scripts/train.py --config configs/default.yaml

# 3) Evaluate
python scripts/eval.py --config configs/default.yaml

# 4) Demo Grad-CAM ảnh đơn
python scripts/demo_gradcam.py --config configs/default.yaml --image data/samples/leaf.jpg
 
````
---

## 2) Tính năng chính (Features)

* ✅ **Phân loại đa lớp**: nhận diện *cây + bệnh* (PlantVillage: ~54k ảnh, 14 loài, 38 lớp).
* ✅ **Giải thích mô hình (XAI)**: **Grad-CAM heatmap** chồng lên ảnh gốc.
* ✅ **Khuyến nghị xử lý**: gợi ý nguyên nhân & cách xử lý theo bệnh.
* ✅ **Kiểm soát đầu vào**: cảnh báo nếu ảnh **không phải lá cây** để giảm “black-box”.
* ✅ **Local-first**: chạy local, dễ demo trực tiếp.

---

## 3) Kiến trúc tổng quan

**User (Upload ảnh)** → **Preprocess** → **Model (MobileNetV2/EfficientNet-B0)** →
(1) **Prediction Top-k** + confidence → (2) **Grad-CAM** → **Overlay Heatmap** →
**Recommendation** + **Input Validation message** → Hiển thị trên Streamlit.

---

## 4) Tech Stack

* Python 3.9+
* Deep Learning: **PyTorch** hoặc **TensorFlow/Keras** *(chọn 1 stack cho repo của bạn)*
* Xử lý ảnh: OpenCV, PIL
* Web: Streamlit
* Quản lý: Git/GitHub (commit đều theo tuần)

---

## 5) Cấu trúc thư mục

```text
PlantDoc_AI/
├── app/                      # Streamlit UI
│   ├── main.py
│   └── components/           # (optional) UI components
├── src/                      # Core logic
│   ├── config.py             # config loader / constants
│   ├── data/                 # dataset, transforms, split
│   ├── models/               # backbone loader, checkpoints
│   ├── train/                # trainer, loss, optimizer, scheduler
│   ├── eval/                 # metrics, confusion matrix
│   ├── infer/                # inference pipeline (predict top-k)
│   ├── xai/                  # grad-cam + visualization overlay
│   ├── validators/           # non-leaf check (input validation)
│   └── utils/                # common utilities (seed, io, logging)
├── scripts/                  # runnable scripts: train/eval/demo/prepare_data
├── notebooks/                # EDA & experiments (ipynb)
├── data/                     # dataset (ignored by git) + sample images
│   ├── samples/
│   ├── raw/                  # PlantVillage downloaded/extracted
│   ├── processed/            # resized/standardized (optional)
│   └── splits/               # train/val/test split files
├── runs/                     # outputs: checkpoints, logs, gradcam images
├── docs/
│   ├── TODO.md
│   ├── proposal.md
│   ├── architecture.md
│   ├── evaluation.md
│   └── weekly-log/
├── requirements.txt
└── README.md
```
---

## 6) Dataset (PlantVillage)

* Nguồn phổ biến: PlantVillage (Kaggle)
* Đầu vào: ảnh lá cây (RGB)
* Đầu ra: nhãn đa lớp (cây + bệnh)

### 6.1 Gợi ý tổ chức dữ liệu

* `data/raw/PlantVillage/` giữ nguyên cấu trúc thư mục của dataset (theo lớp).
* `data/splits/` chứa file split (CSV/JSON) để **tái lập** train/val/test.

---

## 7) Training & Evaluation

### 7.1 Baseline (Week 3)

* Backbone: `MobileNetV2` hoặc `EfficientNet-B0`
* Transfer learning từ ImageNet
* Loss: CrossEntropy
* Metrics tối thiểu:

  * Accuracy
  * F1-macro (khuyến nghị vì data có thể lệch lớp)

### 7.2 Fine-tuning (Week 4)

* Freeze head giai đoạn 1 → unfreeze một phần/ toàn bộ backbone giai đoạn 2
* Tuning:

  * learning rate
  * weight decay
  * scheduler (cosine/onecycle)
  * augmentation

### 7.3 Output sau train

* `runs/checkpoints/best.*` (best model)
* `runs/logs/train_log.csv` (loss/metric theo epoch)
* `runs/eval/confusion_matrix.png` (nếu có)

---

## 8) Explainable AI — Grad-CAM

### 8.1 Mục tiêu

* Tạo **heatmap** cho vùng ảnh mà mô hình “chú ý” khi dự đoán lớp bệnh.
* Chồng heatmap lên ảnh gốc để người dùng hiểu lý do dự đoán.

### 8.2 Output

* `runs/gradcam_samples/<image_name>_cam.png`
* Streamlit hiển thị:

  * ảnh gốc
  * ảnh overlay heatmap
  * top-k dự đoán

---

## 9) Input Validation (Non-leaf warning)

### 9.1 Vì sao cần?

Nếu user upload ảnh không phải lá (ví dụ: người, đồ vật), model vẫn “bắt” một lớp bệnh → dễ gây hiểu lầm.

### 9.2 Chiến lược (2 hướng)

* **Option A (A+)**: mô hình nhị phân `leaf vs non-leaf`
* **Option B (MVP)**: rule-based + confidence threshold:

  * Nếu max confidence < threshold → cảnh báo “Ảnh không rõ/không phải lá”

> Repo nên ghi rõ bạn dùng option nào trong `docs/architecture.md`.

---

## 10) Streamlit UI (App)

### 10.1 Luồng người dùng

1. Upload ảnh
2. App kiểm tra input
3. Predict top-k
4. Sinh Grad-CAM & overlay
5. Hiển thị recommendations

### 10.2 Lưu ý hiệu năng

* Cache model load:

  * `st.cache_resource` (khuyến nghị)
* Tránh load checkpoint lại mỗi lần user bấm nút.

---

## 11) Reproducibility

* Set seed (random/np/torch/tf)
* Ghi config chạy vào file `runs/config_used.yaml`
* Lưu version model + dataset split

---

## 12) Nhật ký tiến độ & Quy tắc commit

### 12.1 Weekly log

* Mỗi tuần cập nhật file: `docs/weekly-log/week-0X.md`
* Nội dung: việc đã làm, evidence (lệnh chạy, screenshot), issue, plan tuần sau.

### 12.2 Commit đều (gợi ý)

* Mỗi task nhỏ = 1 commit
* Ví dụ:

  * `feat: add dataset loader`
  * `feat: implement grad-cam`
  * `fix: correct label mapping`
  * `docs: update week-03 log`
  * `test: add preprocessing tests`

---

## 13) Roadmap 8 tuần

Chi tiết xem: `docs/TODO.md`
Tóm tắt:

* Week 1: proposal + setup
* Week 2: data pipeline
* Week 3: baseline train
* Week 4: fine-tune + improve metrics
* Week 5: Grad-CAM
* Week 6: Streamlit app
* Week 7: input validation + tests + UX
* Week 8: finalize report + demo video

---

## 14) Troubleshooting (các lỗi hay gặp)

* **Thiếu CUDA / chạy CPU chậm**: kiểm tra phiên bản torch/tf phù hợp, hoặc giảm batch size.
* **Mismatch số lớp**: đảm bảo mapping label2idx thống nhất giữa train và infer.
* **Ảnh lỗi/đọc không được**: thêm try/except trong dataset class + log file lỗi.
* **Grad-CAM ra heatmap đen**: chọn đúng layer conv cuối + normalize + tránh hook sai.

---

## 15) License & Disclaimer

* Dự án phục vụ học tập.
* Kết quả dự đoán chỉ mang tính tham khảo, **không thay thế chẩn đoán chuyên gia nông nghiệp**.

---

## 16) Acknowledgements

* PlantVillage dataset
* Pretrained models (ImageNet)
* Streamlit community

 