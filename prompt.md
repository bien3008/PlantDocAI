# ROLE
Bạn là Senior ML Engineer. Bạn phải giúp tôi implement module Debug & QA (fix lỗi, tests) cho PlantDoc AI theo tiêu chuẩn clean code + dễ demo.

# CONTEXT (PlantDoc AI)
- Transfer learning (MobileNetV2/EfficientNet-B0) trên PlantVillage (38 classes).
- Có Grad-CAM explainability.
- Streamlit app demo.
- Cần commit đều: mỗi task nhỏ = 1 commit.

# MODULE REQUEST
- Module đang làm: Debug & QA (fix lỗi, tests) 
- 
# GOAL (cụ thể)
Tôi muốn bạn làm: "Thêm input validation non-leaf (threshold hoặc binary model)"

# ASSUMPTIONS (nếu thiếu)
- Framework: [PyTorch / TF-Keras] (nếu tôi chưa chọn, bạn tự chọn 1 cái dễ demo và ghi rõ)
- Input size: [224]
- Train/Val/Test split: [80/10/10]
- Top-k: [3]

# REQUIREMENTS
1) Thiết kế file/module theo repo structure:
   - Liệt kê rõ những file tạo/sửa (đường dẫn)
2) Code phải chạy được (không pseudo), có comment vừa đủ.
3) Kèm "Smoke test" / lệnh chạy kiểm tra.
4) Tách thành 3–8 commits nhỏ:
   - Mỗi commit: mục tiêu + file touched + commit message.
5) Nếu có edge cases (ảnh lỗi, mismatch labels, device cpu/cuda), phải xử lý hoặc note rõ.

# OUTPUT FORMAT (bắt buộc)
1) Plan (5–10 bullet)
2) Files to create/edit (path list)
3) Code blocks theo từng file (đủ nội dung)
4) How to run (commands) + expected output
5) Commit plan (3–8 commits)

# STYLE
- Ngắn gọn nhưng đủ hành động.
- Không lan man lý thuyết; giải thích chỉ khi cần để code đúng.