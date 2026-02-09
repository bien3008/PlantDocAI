# ĐỀ XUẤT ĐỒ ÁN (Proposal) — PlantDoc AI
**Học phần:** Thực tập cơ sở  
**Giảng viên hướng dẫn:** Nguyễn Xuân Đức  
**Sinh viên:**  Nguyễn Hoàng Biên — B23DCKH007 — D23CQKH01-B  

---


## 1. Tóm tắt 
PlantDoc AI là hệ thống nhận diện bệnh cây trồng dựa trên ảnh lá cây, sử dụng mô hình học sâu theo hướng transfer learning nhằm phân loại đa lớp (cây + bệnh). Bên cạnh việc dự đoán nhãn, hệ thống tích hợp cơ chế giải thích mô hình (Explainable AI) bằng Grad-CAM để trực quan hóa vùng ảnh đóng góp mạnh nhất vào quyết định của mạng. Dự án được triển khai dưới dạng ứng dụng web local-first bằng Streamlit, hỗ trợ người dùng tải ảnh, nhận kết quả dự đoán top-k, hiển thị bản đồ nhiệt và cung cấp khuyến nghị xử lý bệnh. Ngoài ra, hệ thống có kiểm soát đầu vào nhằm cảnh báo trường hợp ảnh không phải lá cây, giảm rủi ro suy luận sai trong thực tế.

---

## 2. Đặt vấn đề & Động cơ
Trong thực tế sản xuất nông nghiệp, việc phát hiện sớm bệnh trên lá giúp giảm thiểu thiệt hại năng suất và chi phí phòng trị. Tuy nhiên, chẩn đoán thủ công phụ thuộc kinh nghiệm, dễ sai lệch và tốn thời gian. Các mô hình học sâu có thể đạt độ chính xác cao nhưng thường thiếu khả năng giải thích, khiến người dùng khó tin tưởng (“black-box”). Do đó, dự án hướng tới một hệ thống vừa dự đoán tốt vừa có minh chứng trực quan (heatmap) và thông tin khuyến nghị đi kèm.

**Bài toán:**  
Cho ảnh đầu vào \( x \in \mathbb{R}^{H \times W \times 3} \), mô hình dự đoán nhãn \( \hat{y} \in \{1,\dots,38\} \) (38 lớp PlantVillage). Đồng thời xuất bản đồ nhiệt Grad-CAM \( M \in \mathbb{R}^{H \times W} \) thể hiện vùng ảnh quan trọng cho lớp dự đoán.

---

## 3. Mục tiêu và phạm vi 
### 3.1 Mục tiêu chính
1. Xây dựng mô hình phân loại bệnh lá cây đa lớp dựa trên dataset PlantVillage.
2. Tích hợp Grad-CAM để giải thích quyết định dự đoán.
3. Xây dựng ứng dụng web demo (Streamlit) để người dùng tương tác trực tiếp.
4. Tích hợp cơ chế kiểm soát đầu vào (cảnh báo ảnh không phải lá) nhằm giảm suy luận sai.

### 3.2 Phạm vi chức năng 
- Upload ảnh → predict top-k + confidence.
- Hiển thị ảnh gốc + heatmap overlay (Grad-CAM).
- Hiển thị thông tin bệnh + khuyến nghị xử lý (knowledge base nội bộ).
- Cảnh báo non-leaf bằng chiến lược local (heuristic/threshold hoặc mô hình nhị phân).

### 3.3 Phạm vi nâng cao
- Fine-tune mạnh hơn (scheduler/augment nâng cao, class-weight).
- Đánh giá latency và tối ưu tốc độ suy luận (cache model, batch size).
- Non-leaf bằng mô hình nhị phân leaf-vs-nonleaf (nếu có dữ liệu bổ sung).
- Xuất mô hình dạng ONNX để tối ưu inference (tuỳ thời gian).

---

## 4. Dataset và tiền xử lý 
### 4.1 Dataset
- **PlantVillage**: ~54,000 ảnh, 14 loài cây, 38 lớp (bệnh + khỏe).
- Dữ liệu có cấu trúc theo thư mục lớp.

### 4.2 Tiền xử lý
- Resize về kích thước chuẩn (mặc định 224×224).
- Chuẩn hóa theo mean/std ImageNet (tương thích pretrained backbone).
- Augmentation cho train: random resized crop, horizontal flip, color jitter (mức vừa), rotation nhỏ.

### 4.3 Chia tập
- Split **Train/Val/Test = 80/10/10** (stratified theo lớp nếu áp dụng).
- Seed cố định để tái lập thí nghiệm.

---

## 5. Phương pháp đề xuất 
### 5.1 Kiến trúc mô hình
- Transfer learning với backbone nhẹ:
  - **MobileNetV2** hoặc **EfficientNet-B0** (ưu tiên tốc độ inference).
- Thay thế classifier head để phù hợp 38 lớp.
- Fine-tuning theo 2 giai đoạn:
  1) Freeze backbone, train head.
  2) Unfreeze một phần backbone, giảm learning rate để fine-tune.

### 5.2 Explainable AI 
- Lấy feature map từ lớp convolution cuối cùng.
- Tính gradient của logit lớp mục tiêu theo feature map.
- Tạo heatmap theo trọng số gradient và chồng lên ảnh gốc.
- Mục tiêu: cung cấp bằng chứng trực quan vùng bệnh/hư hại mô hình tập trung.

### 5.3 Input Validation 
- Hướng local-first:
  - **MVP**: dùng ngưỡng confidence + kiểm tra đơn giản (ví dụ: độ sắc nét, màu xanh chiếm ưu thế không đủ, v.v.) để cảnh báo.
  - **A+**: mô hình nhị phân leaf-vs-nonleaf nếu có dữ liệu bổ sung.

---

## 6. Kiến trúc triển khai 
- Python 3.9+
- Deep learning: **PyTorch** (mặc định chọn để dễ debug và tích hợp Grad-CAM)
- UI: Streamlit
- Xử lý ảnh: PIL/OpenCV
- Quản lý mã: Git/GitHub, commit đều theo tuần.

---

## 7. Kế hoạch đánh giá 
### 7.1 Chỉ số đánh giá
- **Accuracy** trên tập test.
- **F1-macro** (quan trọng khi dữ liệu lệch lớp).
- Confusion matrix để phân tích nhầm lẫn theo lớp.
- (Optional) Latency inference trên CPU (ms/ảnh).

### 7.2 Thiết kế thí nghiệm
- Baseline: backbone A (MobileNetV2 hoặc EfficientNet-B0) + cấu hình chuẩn.
- Improved: fine-tune + augment/scheduler tối ưu.
- Báo cáo so sánh: baseline vs improved.

### 7.3 Tiêu chí hoàn thành
- Có mô hình chạy end-to-end + demo UI.
- Có kết quả metric tối thiểu (Accuracy, F1-macro).
- Có minh chứng heatmap Grad-CAM.
- Có weekly logs và lịch sử commit đều.

--- 