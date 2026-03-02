# Bộ câu hỏi kiểm tra hiểu sâu — DeepLearning.md 

## Phần A — Tự luận (10 câu)

### Câu 1

**Đề:** CNN giải quyết “vấn đề gì” khi làm việc với ảnh, và vì sao cách nhìn “mỗi pixel là một feature độc lập” dễ dẫn đến kết quả kém?
**Đáp án:**

* Ảnh là ma trận điểm ảnh rất lớn → số chiều cực cao nếu coi mỗi pixel là 1 feature.
* Số chiều quá lớn làm mô hình rất dễ **overfit** (học thuộc).
* Cách nhìn pixel độc lập làm **mất cấu trúc không gian** (pixel gần nhau tạo cạnh/hoa văn).
* CNN học **kernel/filter** để phát hiện **mẫu cục bộ** lặp lại ở nhiều vị trí.
* Lớp đầu học mẫu đơn giản (cạnh/góc), lớp sau tổ hợp thành mẫu phức tạp.
* Cuối cùng tạo đặc trưng cấp cao đủ phân biệt lớp để phân loại. 
---

### Câu 2

**Đề:** Hãy giải thích “feature map” là gì và vì sao 1 convolution layer thường tạo ra *nhiều* feature maps.
**Đáp án:**

* Feature map/activation map là kết quả sau convolution (thường qua kích hoạt), thể hiện **mức kích hoạt của một mẫu** theo vị trí.
* Mỗi kernel học một kiểu mẫu (ví dụ biên dọc), quét khắp ảnh để xem mẫu xuất hiện ở đâu mạnh/yếu.
* 1 conv layer thường có **nhiều kernels** → mỗi kernel tạo ra 1 feature map.
* Nhiều feature maps giúp học **nhiều loại mẫu** song song (cạnh/góc/hoa văn khác nhau).
* Không nên nhầm feature map là “ảnh gốc”; nó là “bản đồ mức xuất hiện của mẫu”. 

---

### Câu 3

**Đề:** Pooling (max pooling) “giữ lại cái gì” và “đánh đổi cái gì”? Hãy giải thích trực giác bằng ví dụ một vùng 2×2.
**Đáp án:**

* Pooling nhằm **giảm kích thước không gian (H×W)** để mô hình gọn và chạy nhanh hơn.
* Max pooling **giữ lại đặc trưng nổi bật nhất** trong mỗi vùng nhỏ.
* Ví dụ vùng 2×2: \[[1,3],[0,1]] → max = 3 (giữ “tín hiệu mạnh nhất”).
* Đánh đổi: mất một phần chi tiết vị trí chính xác trong vùng đó (vì chỉ lấy max).
* Giúp phần sau của mạng xử lý ít dữ liệu hơn nhưng vẫn giữ “điểm nổi bật”.

---

### Câu 4

**Đề:** Receptive field (RF) là gì theo trực giác, và vì sao “chồng nhiều conv + pooling/stride” làm RF tăng dần?
**Đáp án:**

* RF là “vùng nhìn hiệu quả”: một cell ở tầng sâu **phụ thuộc vào vùng bao nhiêu pixel** của ảnh gốc.
* Conv 3×3 stride 1: mỗi điểm output phụ thuộc patch 3×3 của input.
* Chồng 2 lớp conv 3×3: điểm tầng 2 phụ thuộc 3×3 điểm tầng 1; mỗi điểm tầng 1 lại phụ thuộc 3×3 pixel gốc → RF tầng 2 **lớn hơn 3×3**.
* Pooling hoặc stride > 1 làm “nhảy bước”, mỗi cell đại diện vùng rộng hơn → RF tăng nhanh hơn.
* RF quan trọng vì phân loại cần **ngữ cảnh rộng** (không chỉ một cạnh đơn lẻ).\
---

### Câu 5

**Đề:** Bạn có mô hình CNN mà train accuracy tăng cao, nhưng bạn nghi ngờ “mạng chưa học được ngữ cảnh đủ rộng”. Dựa trên tài liệu, bạn sẽ dùng khái niệm nào để giải thích nghi ngờ này, và bạn kỳ vọng điều gì khi mạng sâu hơn?
**Đáp án:**

* Dùng khái niệm **receptive field** để nói về “mức độ nhìn ngữ cảnh”.
* Nếu RF nhỏ, tầng sâu chỉ tổng hợp từ vùng hẹp → khó nắm được hình dạng/ngữ cảnh cho phân loại.
* Khi mạng sâu hơn (chồng conv, có pooling/stride), RF tăng dần → mỗi đặc trưng tầng sâu nhìn vùng rộng hơn.
* Kỳ vọng: đặc trưng tầng sâu chuyển từ cạnh/góc sang hoa văn phức tạp và đặc trưng cấp cao hơn.
* Mục tiêu cuối: đặc trưng đủ phân biệt lớp khi đi vào classifier tạo logits. 
---

### Câu 6

**Đề:** Cho 3 kịch bản train/val theo epoch. Hãy chẩn đoán (overfit hay underfit) và nêu hướng xử lý *chỉ trong phạm vi tài liệu*.
(1) Train loss ↓ đều, train acc ↑ cao; Val loss ↓ lúc đầu rồi ↑; Val acc đứng thấp hơn rõ.
(2) Train acc thấp, train loss cao và giảm rất chậm; Val cũng thấp tương tự, gap nhỏ.
(3) Thêm dropout xong train acc giảm nhưng val acc tăng rõ.
**Đáp án:**

* (1) Overfitting: “train tốt, val xấu”, gap lớn; hướng xử lý: **tăng dropout** và/hoặc **tăng weight decay (λ)**.
* (2) Underfitting: train đã kém thì val cũng kém; hướng xử lý: **giảm dropout/giảm weight decay** nếu đang quá mạnh, hoặc **train thêm epoch** (có thể chưa hội tụ).
* (3) Dấu hiệu generalization tốt hơn: dropout đang giúp giảm overfit; chấp nhận train acc giảm nhưng val cải thiện. 
---

### Câu 7

**Đề:** Dropout và weight decay đều nhằm giảm overfit nhưng “tác động lên chỗ nào” khác nhau. Hãy so sánh đúng theo tài liệu, và nêu điểm hay nhầm ở train vs inference.
**Đáp án:**

* Dropout: khi **train**, ngẫu nhiên “tắt” (set 0) một phần neuron/feature → tránh phụ thuộc vào vài đường tắt.
* Weight decay: thêm phạt L2 lên **trọng số W**:  Loss_total = Loss_data + $λ||W||^2$  → kéo trọng số nhỏ lại, mô hình “đơn giản” hơn.
* Khác nhau: dropout tác động lên **activations/feature** (trong quá trình train), weight decay tác động lên **weights** qua loss.
* Điểm hay nhầm: **dropout chỉ bật khi train**, **tắt khi inference**.
* λ quá lớn có thể làm mô hình quá bị “kìm” → dẫn đến underfit. 

---

### Câu 8

**Đề:** Vì sao “train acc thấp” chưa chắc là underfitting? Trình bày cách tránh kết luận vội theo đúng tài liệu.
**Đáp án:**

* Underfitting là khi model không học đủ: train/val đều kém và cải thiện chậm.
* Nhưng train acc thấp có thể do **mới train vài epoch** → chưa hội tụ, chưa thể kết luận underfit.
* Cần nhìn **đường cong theo epoch**: train loss/acc có giảm/tăng đều không.
* Nếu cả train/val đều thấp và loss giảm rất chậm → nghiêng về underfit.
* Trong phạm vi xử lý: có thể **train thêm epoch** trước khi quyết định giảm regularization. 
---

### Câu 9

**Đề:** Phân biệt rõ logits và probabilities. Vì sao nói “logits chưa phải xác suất”, và khi nào bạn cần softmax?
**Đáp án:**

* Logits là vector “điểm thô” cho từng lớp: có thể âm/dương, **không bị giới hạn 0..1** và **không cần tổng = 1**.
* Probabilities là kết quả sau softmax: mỗi phần tử trong (0..1) và tổng = 1.
* Dự đoán lớp có thể lấy  argmax(logits)  (tương đương argmax softmax).
* Softmax cần khi muốn **diễn giải xác suất** hoặc hiển thị **top-k** trong inference/demo.
* Trong train, nhiều framework cho CE nhận logits trực tiếp (không cần tự softmax). 
---

### Câu 10 (tổng hợp)

**Đề:** Hãy mô tả pipeline tối thiểu cho bài toán phân loại ảnh theo tài liệu (train vs inference). Sau đó, cho tình huống: “Train loss giảm mạnh, train acc tăng; val loss tăng từ epoch 6, val acc đứng”.
(1) Pipeline của bạn ở train/inference gồm những khối nào?
(2) Chẩn đoán gì và chỉnh gì trong phạm vi tài liệu?
**Đáp án:**

* (1) **Training pipeline:**

  * Input image → CNN backbone (conv/feature maps/pooling, RF tăng dần) → vector đặc trưng → classifier → **logits**
  * Tính **CrossEntropy(logits, label)** để tối ưu
* (1) **Inference pipeline:**

  * Input image → CNN → logits
  * Nếu cần xác suất/top-k:  softmax(logits) ; dự đoán lớp:  argmax(logits)  (hoặc argmax softmax)
* (2) Dấu hiệu: train tốt nhưng val xấu dần (val loss tăng, val acc đứng) → **overfitting**
* (2) Cách chỉnh trong phạm vi: **tăng dropout** (đúng train-only) và/hoặc **tăng weight decay (λ)**; theo dõi lại train/val curves. 
---

## Phần B — Trắc nghiệm (10 câu)

### Câu 1

**Đề:** Phát biểu nào đúng nhất về “feature map”?
A) Feature map là ảnh gốc sau khi resize.
B) Feature map là bản đồ mức kích hoạt của một kernel/mẫu theo vị trí.
C) Feature map luôn có cùng kích thước với ảnh gốc.
D) Feature map là ma trận trọng số của kernel.
**Đáp án:** B 

### Câu 2

**Đề:** Pooling (max pooling) chủ yếu nhằm mục tiêu nào theo tài liệu?
A) Tăng số tham số để model mạnh hơn.
B) Giảm kích thước H×W và giữ tín hiệu nổi bật.
C) Biến logits thành xác suất.
D) Làm kernel “cố định” như Sobel.
**Đáp án:** B 
### Câu 3

**Đề:** Receptive field (RF) được hiểu đúng nhất là:
A) Kích thước kernel của lớp hiện tại.
B) Số lượng kernel trong một conv layer.
C) Vùng ảnh gốc ảnh hưởng đến một điểm ở feature map tầng sâu.
D) Kích thước ảnh đầu vào H×W×C.
**Đáp án:** C 

### Câu 4

**Đề:** Dấu hiệu điển hình của **overfitting** là:
A) Train acc thấp, val acc thấp, gap nhỏ.
B) Train loss giảm đều, train acc cao; val loss tăng lại sau một thời điểm, val acc kém hơn rõ.
C) Train loss cao và giảm rất chậm, val loss thấp.
D) Train và val đều tốt như nhau ngay từ đầu.
**Đáp án:** B 

### Câu 5

**Đề:** Dấu hiệu điển hình của **underfitting** là:
A) Train tốt, val kém (gap lớn).
B) Val tốt, train kém (gap lớn).
C) Train/val đều thấp, loss cao và giảm chậm; gap không lớn.
D) Val loss tăng nhưng train loss cũng tăng.
**Đáp án:** C 
### Câu 6

**Đề:** Phát biểu nào đúng về **dropout** theo tài liệu?
A) Dropout chỉ dùng khi inference để ổn định dự đoán.
B) Dropout tắt ngẫu nhiên một phần neuron/feature khi train để giảm overfit.
C) Dropout là phạt L2 trên trọng số.
D) Dropout luôn làm train accuracy tăng.
**Đáp án:** B 

### Câu 7

**Đề:** Weight decay trong tài liệu được mô tả đúng nhất là:
A) Chuẩn hoá logits về (0..1).
B) Thêm phạt  $λ||W||^2$  để không khuyến khích trọng số quá lớn.
C) Tắt ngẫu nhiên một phần feature map.
D) Tăng stride để giảm kích thước.
**Đáp án:** B 

### Câu 8

**Đề:** Logits khác probabilities ở điểm nào?
A) Logits luôn nằm trong (0..1) và tổng bằng 1.
B) Probabilities có thể âm/dương.
C) Logits là điểm thô, không bị giới hạn 0..1 và không cần tổng bằng 1.
D) Probabilities không dùng được để lấy top-k.
**Đáp án:** C 
### Câu 9

**Đề:** Softmax được dùng khi nào theo tài liệu?
A) Luôn bắt buộc khi train trước CrossEntropy.
B) Chỉ để giảm overfitting.
C) Khi inference cần xác suất/top-k để hiển thị hoặc diễn giải.
D) Để tăng receptive field.
**Đáp án:** C 

### Câu 10

**Đề:** CrossEntropy (CE) cho one-hot label có trực giác cốt lõi là:
A) CE =  $log(p_y)$  nên p_y càng nhỏ càng tốt.
B) CE ≈  -$log(p_y)$  nên p_y càng cao thì loss càng nhỏ.
C) CE chỉ đo độ lớn của logits, không liên quan nhãn đúng.
D) CE thay thế cho pooling.
**Đáp án:** B 
 