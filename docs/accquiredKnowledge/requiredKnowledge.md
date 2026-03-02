 
## 1) Nền tảng Deep Learning cho Computer Vision

**Cần nắm:**

* CNN hoạt động thế nào: convolution, pooling, receptive field, feature maps.
* Overfitting/underfitting, regularization (dropout, weight decay).
* Loss cho classification: **CrossEntropy**, softmax, logits.

**Đủ để làm dự án:**
Giải thích được “vì sao CNN phân loại ảnh”, “vì sao overfit”, “vì sao dùng CE loss”.

---

## 2) Transfer Learning & Fine-tuning 

**Cần nắm:**

* Pretrained ImageNet: dùng như feature extractor.
* Chiến lược 2 phase:

  1. **Freeze backbone** → train classifier head
  2. **Unfreeze một phần/tất cả** → fine-tune với LR nhỏ
* LR schedules (cosine/onecycle), early stopping, checkpoint “best”.

**Đủ để làm dự án:**
làm được baseline + improved, có giải thích rõ “freeze/unfreeze” trong report.

---

## 3) Data Pipeline & Image Preprocessing 

**Cần nắm:**

* Resize/crop/normalize (đặc biệt normalize theo ImageNet).
* Augmentation đúng cách: flip, rotation nhẹ, color jitter vừa, random resized crop.
* Split train/val/test **có seed** + (nếu được) stratified theo lớp.
* Data imbalance: lớp ít ảnh → F1-macro quan trọng hơn accuracy.

**Đủ để làm dự án:**
Có script smoke test dataloader + mô tả augmentation trong docs.

---

## 4) Đánh giá mô hình (Metrics & Error Analysis)

**Cần nắm:**

* Accuracy vs F1-macro (vì sao F1-macro cần khi lệch lớp).
* Confusion matrix: đọc và phân tích cặp lớp hay nhầm.
* Top-k accuracy (tuỳ chọn, hợp UI).
* Lập bảng so sánh baseline vs improved.

**Đủ để làm dự án:**
Bạn tạo được bảng metrics + confusion matrix ảnh và viết nhận xét 5–10 dòng.

---

## 5) Grad-CAM & Explainable AI 

**Cần nắm:**

* Grad-CAM concept: gradient của class score lên feature map conv cuối.
* Chọn đúng layer conv cuối, normalize heatmap, overlay alpha blend.
* Interpret đúng: heatmap là “vùng ảnh mô hình chú ý”, không phải “mask bệnh tuyệt đối”.

**Đủ để làm dự án:**
Demo ra heatmap đẹp trên 5–10 ảnh + viết phần “giải thích” ngắn gọn trong docs.

---

## 6) “Non-leaf input validation” 

**Cần nắm:**

* Distribution shift: ảnh ngoài miền → model tự tin sai.
* 2 hướng local:

  * MVP: threshold confidence + heuristic nhẹ
  * A+: binary model leaf-vs-nonleaf (nếu có data)
* UX: cảnh báo rõ ràng, hướng dẫn chụp ảnh đúng.

**Đủ để làm dự án:**
Có test case non-leaf + blur + kết quả Pass/Fail trong evaluation.

---

## 7) Streamlit 

**Cần nắm:**

* Upload file, hiển thị ảnh, button predict.
* `st.cache_resource` để cache model.
* Layout: input panel, output panel, top-k, heatmap, recommendation.
* Xử lý lỗi thân thiện (try/except, message rõ).

**Đủ để làm dự án:**
App chạy ổn định, không crash, có “happy path” và “error path”.

---

## 8) Clean Code + Repo Engineering 

**Cần nắm:**

* Tách module: data / models / train / eval / xai / infer / validators / app.
* Config-driven (yaml hoặc config.py) để đổi model/hyperparams.
* Logging + output artifacts có cấu trúc (`runs/`).
* Git workflow: commit nhỏ, message rõ.

**Đủ để làm dự án:**
Repo nhìn “pro”: chạy được bằng vài lệnh, docs rõ, commit đều.

---

## 9) Debugging thực chiến  

**Cần nắm:**

* Lỗi hay gặp: mismatch num_classes, label mapping lệch, device cpu/cuda, dtype, shape.
* Cách “sanity check”: overfit 1 batch nhỏ, train 1 epoch, kiểm tra pipeline trước khi train lâu.
* Reproducibility: seed, deterministic options (mức vừa).

**Đủ để làm dự án:**
Bạn có checklist debug và biết cách chứng minh “pipeline đúng”.

---

## 10) Kiến thức domain (thực vật/bệnh lá) — không cần sâu nhưng nên có

**Cần nắm:**

* Một vài mô tả triệu chứng bệnh phổ biến (đốm lá, mốc, cháy lá…).
* Viết recommendation theo kiểu “triệu chứng → nguyên nhân → xử lý”.

**Đủ để làm dự án:**
Recommendation có vẻ “thật”, không chung chung, 3–6 bullet.

---

# Lộ trình học “đúng tuần” để tối ưu kết quả

* **Week 1–2:** (3) Data pipeline + (8) repo engineering cơ bản
* **Week 3–4:** (2) Transfer learning + (4) Evaluation + (9) Debug sanity checks
* **Week 5:** (5) Grad-CAM
* **Week 6:** (7) Streamlit + (6) validation cơ bản
* **Week 7–8:** (6) validation hoàn thiện + (4) report + polish

 