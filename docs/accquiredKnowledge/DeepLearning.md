
# Learning Note: CNN – Overfitting/Regularization – CrossEntropy (Đủ để làm Project Phân Loại Ảnh)

## Mục lục
- [Phần A — CNN hoạt động thế nào](#phần-a--cnn-hoạt-động-thế-nào)
  - [0) Mục tiêu học phần](#a0-mục-tiêu-học-phần)
  - [1) Khái niệm & trực giác](#a1-khái-niệm--trực-giác)
  - [2) Cơ chế hoạt động (chi tiết vừa đủ)](#a2-cơ-chế-hoạt-động-chi-tiết-vừa-đủ)
  - [3) Thuật ngữ cốt lõi & điểm hay nhầm](#a3-thuật-ngữ-cốt-lõi--điểm-hay-nhầm)
  - [4) Ví dụ áp dụng trong dự án phân loại ảnh](#a4-ví-dụ-áp-dụng-trong-dự-án-phân-loại-ảnh)
  - [5) Checklist “đủ để làm dự án”](#a5-checklist-đủ-để-làm-dự-án)
  - [6) Bài tập luyện tập (kèm đáp án ngắn)](#a6-bài-tập-luyện-tập-kèm-đáp-án-ngắn)
- [Phần B — Overfitting/Underfitting & Regularization (Dropout, Weight Decay)](#phần-b--overfittingunderfitting--regularization-dropout-weight-decay)
  - [0) Mục tiêu học phần](#b0-mục-tiêu-học-phần)
  - [1) Khái niệm & trực giác](#b1-khái-niệm--trực-giác)
  - [2) Cơ chế hoạt động (chi tiết vừa đủ)](#b2-cơ-chế-hoạt-động-chi-tiết-vừa-đủ)
  - [3) Thuật ngữ cốt lõi & điểm hay nhầm](#b3-thuật-ngữ-cốt-lõi--điểm-hay-nhầm)
  - [4) Ví dụ áp dụng trong dự án phân loại ảnh](#b4-ví-dụ-áp-dụng-trong-dự-án-phân-loại-ảnh)
  - [5) Checklist “đủ để làm dự án”](#b5-checklist-đủ-để-làm-dự-án)
  - [6) Bài tập luyện tập (kèm đáp án ngắn)](#b6-bài-tập-luyện-tập-kèm-đáp-án-ngắn)
- [Phần C — Loss cho Classification: Softmax, Logits, CrossEntropy](#phần-c--loss-cho-classification-softmax-logits-crossentropy)
  - [0) Mục tiêu học phần](#c0-mục-tiêu-học-phần)
  - [1) Khái niệm & trực giác](#c1-khái-niệm--trực-giác)
  - [2) Cơ chế hoạt động (chi tiết vừa đủ)](#c2-cơ-chế-hoạt-động-chi-tiết-vừa-đủ)
  - [3) Thuật ngữ cốt lõi & điểm hay nhầm](#c3-thuật-ngữ-cốt-lõi--điểm-hay-nhầm)
  - [4) Ví dụ áp dụng trong dự án phân loại ảnh](#c4-ví-dụ-áp-dụng-trong-dự-án-phân-loại-ảnh)
  - [5) Checklist “đủ để làm dự án”](#c5-checklist-đủ-để-làm-dự-án)
  - [6) Bài tập luyện tập (kèm đáp án ngắn)](#c6-bài-tập-luyện-tập-kèm-đáp-án-ngắn)

---

## Phần A — CNN hoạt động thế nào

### A0) Mục tiêu học phần
- Sau phần này, bạn có thể:
  - Giải thích vì sao CNN “nhìn ảnh” được và biến ảnh thành đặc trưng (features) để phân loại.
  - Hiểu convolution/pooling tạo ra feature maps như thế nào.
  - Hiểu receptive field tăng dần qua các lớp và vì sao nó quan trọng cho phân loại.

### A1) Khái niệm & trực giác (intuition)
**CNN giải quyết vấn đề gì trong CV?**  
Ảnh là ma trận điểm ảnh rất lớn. Nếu coi mỗi pixel là một “feature” độc lập thì:
- Quá nhiều chiều (rất dễ overfit).
- Mất thông tin cấu trúc không gian (pixel gần nhau tạo thành cạnh/hoa văn).

**Trực giác CNN:**  
CNN học các “bộ lọc” (filters/kernels) để phát hiện mẫu cục bộ:
- Lớp đầu: cạnh, góc, đường viền (pattern đơn giản).
- Lớp sau: hoa văn phức tạp hơn (tổ hợp của cạnh/góc).
- Cuối: đặc trưng cấp cao đủ phân biệt lớp.

**Từ khóa trực giác:**
- **Convolution**: “quét” kernel để tìm mẫu lặp lại ở nhiều vị trí.
- **Feature map**: bản đồ “mức độ xuất hiện” của một mẫu.
- **Pooling**: nén thông tin, giảm kích thước, giữ phần nổi bật.
- **Receptive field**: “vùng nhìn” hiệu quả của 1 điểm ở tầng sâu đối với ảnh gốc.

### A2) Cơ chế hoạt động (chi tiết vừa đủ)

#### (1) Convolution tạo feature maps
Giả sử ảnh xám nhỏ 4×4 (để dễ tính), kernel 3×3, stride = 1, không padding.

Ảnh `X`:
```

1  2  0  1
3  1  2  2
0  1  3  1
2  2  1  0

```

Kernel `K` (3×3):
```

1  0 -1
1  0 -1
1  0 -1

```

**Bước tính một ô output (góc trên-trái):**
Lấy patch 3×3 ở `X`:
```

1 2 0
3 1 2
0 1 3

```
Nhân từng phần tử rồi cộng:
- Hàng 1: `1*1 + 2*0 + 0*(-1) = 1`
- Hàng 2: `3*1 + 1*0 + 2*(-1) = 1`
- Hàng 3: `0*1 + 1*0 + 3*(-1) = -3`
Tổng: `1 + 1 + (-3) = -1`

=> Output tại vị trí đó = `-1`.

**Ý nghĩa:** kernel này giống “bộ lọc phát hiện biên dọc” (trái-phải khác nhau).  
Khi quét khắp ảnh, bạn nhận được **feature map**: nơi nào có biên dọc mạnh, giá trị sẽ lớn (dương/âm tùy hướng).

> Lưu ý: Thực tế CNN có nhiều kernels → nhiều feature maps → học nhiều loại mẫu cùng lúc.

#### (2) Pooling (max pooling) để giảm kích thước & giữ nổi bật
Giả sử feature map 4×4:
```

1  3  2  0
0  1  5  2
2  2  1  1
0  1  2  4

```
Max pooling 2×2, stride 2 → output 2×2:
- Ô (0,0) lấy max của:
```

1 3
0 1

```
=> 3
- Ô (0,1) lấy max của:
```

2 0
5 2

```
=> 5
- Ô (1,0) max của:
```

2 2
0 1

```
=> 2
- Ô (1,1) max của:
```

1 1
2 4

```
=> 4

Output:
```

3 5
2 4

```

**Ý nghĩa:** giữ đặc trưng mạnh nhất trong mỗi vùng, giảm kích thước (nhanh hơn, ít tham số về sau).

#### (3) Receptive field tăng dần qua các lớp (và tại sao quan trọng)
**Receptive field (RF)**: một điểm (một “cell”) ở feature map tầng sâu “phụ thuộc” vào một vùng bao nhiêu pixel ở ảnh gốc.

Trực giác:
- Lớp conv đầu nhìn vùng nhỏ (ví dụ 3×3) → phát hiện cạnh.
- Xếp nhiều conv + pooling → điểm ở tầng sâu “nhìn” vùng lớn hơn → hiểu hình dạng/đối tượng.

Ví dụ đơn giản (không cần công thức nặng):
- Conv 3×3 (stride 1) làm mỗi điểm output phụ thuộc patch 3×3 của input.
- Nếu bạn **chồng 2 lớp conv 3×3 (stride 1)**:
  - Điểm ở tầng 2 phụ thuộc vào 3×3 điểm của tầng 1,
  - Mỗi điểm tầng 1 lại phụ thuộc 3×3 pixel gốc,
  - Kết quả RF tầng 2 lớn hơn 3×3 (tăng “dần”).
- Pooling/stride > 1 làm “nhảy” bước, RF tăng nhanh hơn vì mỗi cell đại diện cho vùng rộng hơn.

**Vì sao RF quan trọng cho phân loại?**
- Phân loại cần “ngữ cảnh”: không chỉ nhìn một cạnh, mà là tổng hợp nhiều phần để nhận ra “lá có đốm bệnh” hay “mèo”…
- RF đủ lớn giúp tầng sâu gom thông tin rộng hơn, tạo đặc trưng phân biệt lớp.

### A3) Thuật ngữ cốt lõi & điểm hay nhầm
- **Kernel/Filter**: ma trận nhỏ học được để quét ảnh.  
  *Hay nhầm:* kernel là “cố định” như Sobel; trong CNN kernel được học từ dữ liệu.
- **Convolution layer**: áp nhiều kernels → nhiều feature maps.  
  *Hay nhầm:* 1 conv chỉ ra 1 map; thật ra thường ra nhiều kênh (channels).
- **Feature map / Activation map**: kết quả sau conv (thường qua hàm kích hoạt).  
  *Hay nhầm:* feature map là “ảnh”; đúng hơn là “mức kích hoạt của một mẫu”.
- **Pooling**: giảm kích thước không gian (H×W).  
  *Hay nhầm:* pooling “học tham số”; pooling thường không học (max/avg).
- **Stride**: bước trượt khi quét kernel/pooling.  
  *Hay nhầm:* stride chỉ có ở pooling; conv cũng có stride.
- **Padding**: thêm viền để giữ kích thước / kiểm soát biên.  
  *Hay nhầm:* padding luôn giữ size; tùy kernel/stride.
- **Receptive field**: vùng ảnh gốc ảnh hưởng đến một điểm tầng sâu.  
  *Hay nhầm:* RF = kernel size; thực tế RF tăng theo chiều sâu mạng.

### A4) Ví dụ áp dụng trong dự án phân loại ảnh
**Pipeline tối thiểu (tổng quan):**
1. Input image `H×W×C`
2. CNN backbone:
   - Conv → feature maps → (pooling) → deeper feature maps
3. (Cuối) ra một vector đặc trưng
4. Linear/Classifier → **logits**
5. Train: dùng **CrossEntropy** trên logits với nhãn đúng
6. Inference: lấy **softmax** (nếu cần xác suất) hoặc argmax trực tiếp từ logits

**Liên hệ thực nghiệm:**
- Nếu bạn thấy model “không học được cạnh/hoa văn” (accuracy thấp cả train/val), thường là kiến trúc quá yếu hoặc dữ liệu/tiền xử lý có vấn đề (nhưng trong phạm vi này, trọng tâm là: CNN cần đủ lớp/đủ RF để gom ngữ cảnh).

### A5) Checklist “đủ để làm dự án”
**(a) Vì sao CNN phân loại ảnh?**  
Vì CNN học các bộ lọc phát hiện mẫu cục bộ (cạnh/hoa văn), rồi chồng nhiều lớp để tổng hợp thành đặc trưng cấp cao, cuối cùng dùng đặc trưng đó để phân biệt lớp.

**Checklist tự rà (8–12 gạch):**
- [ ] Tôi giải thích được “kernel quét ảnh” và tạo feature map là gì.
- [ ] Tôi tính được 1 ô convolution bằng phép nhân-cộng trên patch nhỏ.
- [ ] Tôi hiểu pooling (max) làm gì và tính được ví dụ 2×2.
- [ ] Tôi phân biệt được stride và kernel size.
- [ ] Tôi hiểu CNN có nhiều kernels → nhiều kênh feature maps.
- [ ] Tôi hiểu receptive field tăng khi mạng sâu hơn (conv chồng + pooling).
- [ ] Tôi biết RF quan trọng vì phân loại cần ngữ cảnh rộng.
- [ ] Tôi mô tả được pipeline: image → CNN → logits → loss/inference.
- [ ] Tôi không nhầm feature map với ảnh gốc.

### A6) Bài tập luyện tập (kèm đáp án ngắn)
**Câu 1 (khái niệm):** Feature map là gì?  
**Đáp án:** Là bản đồ kích hoạt (activation) cho một kernel/mẫu, cho biết mẫu đó xuất hiện mạnh ở vị trí nào.

**Câu 2 (khái niệm):** Pooling giải quyết vấn đề gì?  
**Đáp án:** Giảm kích thước H×W, giữ thông tin nổi bật (max) và làm mô hình gọn/nhanh hơn.

**Câu 3 (khái niệm):** Receptive field tăng dần nghĩa là gì?  
**Đáp án:** 1 điểm ở tầng sâu phụ thuộc vào vùng lớn hơn của ảnh gốc, giúp model “nhìn ngữ cảnh” rộng hơn.

**Câu 4 (tính nhanh):** Max pooling 2×2 stride 2 trên block:
```

2 0
1 3

```
ra gì?  
**Đáp án:** 3.

**Câu 5 (tính nhanh):** Với kernel 3×3 stride 1, input 4×4, không padding → output size?  
**Đáp án:** 2×2 (vì 4-3+1 = 2).

**Câu 6 (áp dụng):** Vì sao CNN cần nhiều layer hơn 1 layer conv?  
**Đáp án:** 1 layer chỉ phát hiện mẫu đơn giản; nhiều layer tổng hợp thành đặc trưng phức tạp phục vụ phân loại.

---

## Phần B — Overfitting/Underfitting & Regularization (Dropout, Weight Decay)

### B0) Mục tiêu học phần
- Sau phần này, bạn có thể:
  - Nhìn train/val loss/acc để chẩn đoán overfitting vs underfitting.
  - Biết vì sao overfit xảy ra trong phân loại ảnh.
  - Biết áp dụng 2 regularization cơ bản: **dropout** và **weight decay** trong training.

### B1) Khái niệm & trực giác (intuition)
**Overfitting:** model học “quá khớp” dữ liệu train, nhớ chi tiết/ồn, nên ra val/test kém.  
**Underfitting:** model quá yếu hoặc học chưa đủ → train cũng kém.

**Nó giải quyết vấn đề gì trong CV?**  
Trong project phân loại ảnh, bạn thường gặp:
- Dataset không quá lớn hoặc nhiễu.
- Model có nhiều tham số.
=> Model dễ “học thuộc” train (overfit), nhưng không tổng quát hóa.

### B2) Cơ chế hoạt động (chi tiết vừa đủ)

#### (1) Dấu hiệu qua đường cong train/val (loss/accuracy)
Bạn theo dõi theo epoch:

- **Overfitting điển hình:**
  - Train loss ↓ đều, train acc ↑ cao
  - Val loss ↓ lúc đầu rồi ↑ lại (hoặc đứng), val acc đứng/yếu hơn rõ rệt

- **Underfitting điển hình:**
  - Train loss cao/giảm rất chậm, train acc thấp
  - Val cũng thấp tương tự (không có “khoảng cách lớn”)

Bạn có thể nhớ bằng câu:  
> Overfit = “train tốt, val xấu”; Underfit = “train đã xấu thì val cũng xấu”.

#### (2) Dropout: “tắt ngẫu nhiên” một phần mạng khi train
**Ý tưởng:** Khi train, ta ngẫu nhiên bỏ (set 0) một số neuron/đặc trưng → mạng không dựa vào một vài “đường tắt” (shortcut) → học phân tán hơn → giảm overfit.

Ví dụ số nhỏ:
- Giả sử vector đặc trưng trước classifier: `[2, 1, 3, 4]`
- Dropout p = 0.5 (bỏ 50% phần tử)  
  Một lần train có thể thành: `[2, 0, 3, 0]` (chỉ là ví dụ)
- Lần sau lại khác: `[0, 1, 0, 4]`
=> Model buộc phải dựa vào nhiều kết hợp khác nhau, không “phụ thuộc” một feature duy nhất.

**Quan trọng:** Dropout thường **chỉ bật khi train**, tắt khi inference.

#### (3) Weight decay: “phạt” trọng số quá lớn (L2 regularization)
**Trực giác:** Trọng số quá lớn → mô hình dễ tạo ranh giới quyết định “ngoằn ngoèo” để khớp train.  
Weight decay thêm một hình phạt vào mục tiêu tối ưu để “kéo” trọng số nhỏ lại.

Công thức ý nghĩa (giải thích ký hiệu):
- Loss tổng:  
  `Loss_total = Loss_data + λ * ||W||^2`
  - `Loss_data`: loss trên dữ liệu (ví dụ CrossEntropy)
  - `W`: tập trọng số của model
  - `||W||^2`: tổng bình phương trọng số (L2)
  - `λ` (lambda): mức phạt (hệ số weight decay)

**Ý nghĩa:** λ càng lớn → càng “ghét” trọng số lớn → mô hình đơn giản hơn → giảm overfit (nhưng quá lớn có thể underfit).

### B3) Thuật ngữ cốt lõi & điểm hay nhầm
- **Overfitting**: train tốt, val kém (gap lớn).  
  *Hay nhầm:* “val acc thấp” luôn là overfit; có thể là underfit nếu train cũng thấp.
- **Underfitting**: model không học đủ, train/val đều kém.  
  *Hay nhầm:* mới train vài epoch thấy train thấp → kết luận underfit; có thể chỉ là chưa train đủ.
- **Regularization**: kỹ thuật giảm overfit.  
  *Hay nhầm:* regularization luôn làm tốt hơn; thực tế có thể làm giảm train acc.
- **Dropout**: tắt ngẫu nhiên neuron/feature khi train.  
  *Hay nhầm:* dùng dropout khi inference; không, inference phải tắt.
- **Weight decay**: phạt L2 trên trọng số.  
  *Hay nhầm:* weight decay = dropout; khác bản chất (một cái trên activations, một cái trên weights).
- **Generalization**: khả năng làm tốt trên dữ liệu mới.  
  *Hay nhầm:* generalization = train acc cao; không, cần val/test tốt.

### B4) Ví dụ áp dụng trong dự án phân loại ảnh
**Pipeline tối thiểu (gắn với overfit/underfit):**
- Train:
  - Input → CNN → logits → CrossEntropy
  - Theo dõi: train loss/acc, val loss/acc theo epoch
- Nếu thấy overfit:
  - Thêm/ tăng dropout ở phần classifier (hoặc sau một số lớp)
  - Thêm/ tăng weight decay (λ)
- Nếu thấy underfit:
  - (Trong phạm vi này) giảm dropout/giảm weight decay nếu đang quá mạnh, hoặc train thêm epoch (vì có thể chưa hội tụ).

**Mẫu chẩn đoán nhanh bằng bảng**

| Hiện tượng | Train Acc | Val Acc | Train Loss | Val Loss | Chẩn đoán | Hướng xử lý (trong phạm vi) |
|---|---:|---:|---:|---:|---|---|
| Gap lớn, val tệ dần | cao | thấp/giảm | giảm | tăng | Overfit | tăng dropout / tăng weight decay |
| Cả hai đều thấp | thấp | thấp | cao | cao | Underfit | giảm regularization (dropout/WD), train thêm |
| Cả hai đều tốt | cao | cao | thấp | thấp | OK | giữ cấu hình |

### B5) Checklist “đủ để làm dự án”
**(b) Vì sao overfit?**  
Vì model có đủ “năng lực” để ghi nhớ chi tiết/ồn của tập train, nên tối ưu tốt trên train nhưng không tổng quát hóa sang dữ liệu mới (val/test), đặc biệt khi dữ liệu ít/nhiễu.

**Checklist tự rà (8–12 gạch):**
- [ ] Tôi phân biệt được overfit vs underfit bằng train/val curves.
- [ ] Tôi hiểu overfit thường có “gap” train-val lớn.
- [ ] Tôi biết dropout làm gì (tắt ngẫu nhiên khi train).
- [ ] Tôi nhớ dropout phải tắt khi inference.
- [ ] Tôi biết weight decay là phạt L2 trên trọng số.
- [ ] Tôi hiểu λ quá lớn có thể gây underfit.
- [ ] Tôi biết khi overfit thì train acc có thể vẫn tăng.
- [ ] Tôi biết khi underfit thì train acc cũng thấp.
- [ ] Tôi biết cách thử: tăng dropout/WD để giảm overfit (và xem lại val).

### B6) Bài tập luyện tập (kèm đáp án ngắn)
**Câu 1 (khái niệm):** Overfitting khác underfitting ở dấu hiệu nào?  
**Đáp án:** Overfit: train tốt, val kém (gap lớn). Underfit: train/val đều kém.

**Câu 2 (khái niệm):** Dropout giúp giảm overfit bằng cách nào?  
**Đáp án:** Buộc mạng không phụ thuộc vào một vài feature/neuron cố định; học phân tán hơn.

**Câu 3 (khái niệm):** Weight decay “phạt” điều gì?  
**Đáp án:** Phạt trọng số lớn (L2), khuyến khích mô hình đơn giản hơn.

**Câu 4 (tình huống):** Train acc 98%, val acc 70%. Train loss giảm, val loss tăng sau epoch 5.  
**Đáp án:** Overfitting → tăng dropout/weight decay.

**Câu 5 (tình huống):** Train acc 55%, val acc 52%, cả hai loss đều cao và giảm chậm.  
**Đáp án:** Underfitting → giảm regularization nếu đang dùng mạnh, hoặc train thêm.

**Câu 6 (tình huống):** Thêm dropout, train acc giảm từ 95% xuống 90% nhưng val acc tăng từ 70% lên 78%.  
**Đáp án:** Tốt hơn về generalization; dropout đang giúp giảm overfit.

---

## Phần C — Loss cho Classification: Softmax, Logits, CrossEntropy

### C0) Mục tiêu học phần
- Sau phần này, bạn có thể:
  - Phân biệt **logits** vs **probabilities**.
  - Hiểu **softmax** làm gì và dùng khi nào.
  - Hiểu vì sao dùng **CrossEntropy** để train phân loại ảnh.

### C1) Khái niệm & trực giác (intuition)
**Bài toán classification:** chọn 1 trong K lớp.

CNN cuối cùng tạo ra một vector K chiều:
- Mỗi chiều là “điểm” cho một lớp → đó là **logits** (chưa phải xác suất).
- Ta muốn:
  - Trong train: có một loss đo “đúng/sai” và đẩy lớp đúng lên.
  - Trong inference: có thể cần xác suất (softmax) để hiển thị top-k.

**Nó giải quyết vấn đề gì trong CV?**  
Giúp biến “đầu ra thô” của model thành:
- Một mục tiêu tối ưu (CrossEntropy) để học tốt.
- Một cách diễn giải (softmax) khi suy luận.

### C2) Cơ chế hoạt động (chi tiết vừa đủ)

#### (1) Logits là gì?
Giả sử K=3 lớp: `cat`, `dog`, `rabbit`.  
Model output logits: `z = [2.0, 1.0, 0.1]`

- Đây là **điểm thô** (có thể âm/dương, không cộng = 1).
- Lớp dự đoán (argmax) là `cat` vì 2.0 lớn nhất.

#### (2) Softmax: biến logits thành xác suất
Softmax biến `z` thành `p`:
$$
p_i = \frac{e^{Z_i}} { ∑_j Z_j}
$$

Tính nhanh (xấp xỉ):
- exp(2.0) ≈ 7.39
- exp(1.0) ≈ 2.72
- exp(0.1) ≈ 1.105
Tổng ≈ 11.215  
→ `p ≈ [0.659, 0.243, 0.099]`

**Ý nghĩa:** có xác suất (0..1), tổng = 1, dễ hiểu & dùng top-k.

> Trong nhiều framework, khi train bạn **không cần tự softmax** trước khi đưa vào CrossEntropy (vì CrossEntropy thường “gộp” softmax bên trong để ổn định số học).

#### (3) CrossEntropy (CE) cho classification
Nếu nhãn đúng là `cat` (lớp 0), ta muốn $p_{cat}$ càng gần 1 càng tốt.

CrossEntropy cho one-hot label thường có dạng:
- CE = $- log(p_y)$
  - y: chỉ số lớp đúng
  - $p_y$: xác suất của lớp đúng sau softmax

Ví dụ trên: $p_y = 0.659  $
→ $CE ≈ -log(0.659) ≈ 0.417$

Nếu model dự đoán đúng kém ($p_y$ nhỏ), $log(p_y)$ rất âm → CE lớn → bị phạt nặng.

**Trực giác “tại sao CE hợp lý”:**
- Nó thưởng mạnh khi model tự tin đúng ($p_y$ cao).
- Phạt rất mạnh khi model tự tin sai ($p_y$ thấp).
- Tạo gradient “đẩy” logits lớp đúng lên, kéo logits lớp sai xuống.

### C3) Thuật ngữ cốt lõi & điểm hay nhầm
- **Logits**: điểm thô trước softmax, không phải xác suất.  
  *Hay nhầm:* logits là probability.
- **Softmax**: biến logits → phân phối xác suất tổng = 1.  
  *Hay nhầm:* softmax “bắt buộc” khi train; thường không cần gọi thủ công nếu dùng CE chuẩn.
- **Probabilities**: kết quả sau softmax (0..1).  
  *Hay nhầm:* probabilities luôn cần trong training; thực tế loss có thể dùng logits trực tiếp.
- **CrossEntropy**: loss đo sai lệch giữa dự đoán và nhãn đúng (trong classification).  
  *Hay nhầm:* CE nhận probabilities hay logits “tùy framework”; nhiều framework nhận **logits**.
- **Train vs inference**:
  - Train: logits → CrossEntropy (thường CE tự làm softmax nội bộ)
  - Inference: logits → softmax (nếu cần xác suất), hoặc argmax logits

### C4) Ví dụ áp dụng trong dự án phân loại ảnh
**Pipeline tối thiểu đúng chuẩn:**
- **Training**
  1. Input image → CNN → logits (K chiều)
  2. Tính **CrossEntropy(logits, label)**
  3. Backprop cập nhật weights
- **Inference**
  1. Input image → CNN → logits
  2. Nếu cần hiển thị xác suất/top-k: `softmax(logits)`
  3. Dự đoán lớp: `argmax(logits)` (tương đương argmax softmax)

**Gắn với overfit/underfit (thực nghiệm):**
- Nếu train loss giảm mạnh nhưng val loss tăng → overfit (quay lại Phần B), CE đang tối ưu train quá tốt nhưng generalization kém.
- Nếu cả train/val loss cao → underfit.

### C5) Checklist “đủ để làm dự án”
**(c) Vì sao dùng CrossEntropy?**  
Vì CrossEntropy đo mức “sai” của phân phối dự đoán so với nhãn đúng trong bài toán phân loại, và tạo tín hiệu học hiệu quả: đẩy xác suất lớp đúng lên, kéo lớp sai xuống (thông qua logits/softmax).

**Checklist tự rà (8–12 gạch):**
- [ ] Tôi phân biệt được logits và probabilities.
- [ ] Tôi biết softmax biến logits thành xác suất tổng = 1.
- [ ] Tôi biết inference có thể dùng argmax logits hoặc softmax rồi argmax.
- [ ] Tôi hiểu CE có dạng cơ bản $-log(p_y)$ (one-hot).
- [ ] Tôi hiểu $p_y$ nhỏ → loss lớn (phạt mạnh).
- [ ] Tôi biết khi train thường đưa logits thẳng vào CE (không tự softmax nếu framework đã gộp).
- [ ] Tôi biết softmax hữu ích để báo cáo top-k/prob khi demo.
- [ ] Tôi đọc được pipeline: image → CNN → logits → CE (train) / softmax (inference).

### C6) Bài tập luyện tập (kèm đáp án ngắn)
**Câu 1 (khái niệm):** Logits là gì?  
**Đáp án:** Vector điểm thô cho từng lớp trước softmax; không bị giới hạn 0..1 và không cần tổng = 1.

**Câu 2 (khái niệm):** Softmax làm gì?  
**Đáp án:** Chuyển logits thành xác suất (0..1) và tổng các xác suất bằng 1.

**Câu 3 (khái niệm):** CrossEntropy “đo” cái gì?  
**Đáp án:** Mức phạt khi model gán xác suất thấp cho lớp đúng $(CE ≈ -log(p_y))$.

**Câu 4 (logits/softmax/CE):** Khi inference, muốn hiển thị top-3 xác suất, bạn làm gì?  
**Đáp án:** Tính softmax(logits) rồi lấy 3 lớp có xác suất cao nhất.

**Câu 5 (logits/softmax/CE):** Trong training, nếu framework CrossEntropy nhận logits, bạn có nên tự softmax trước không?  
**Đáp án:** Không (thường), vì CE đã gộp softmax nội bộ; tự softmax có thể gây sai/không ổn định.

**Câu 6 (tình huống thực nghiệm):** Train loss giảm đều, train acc tăng; val loss tăng từ epoch 6, val acc đứng.  
**Đáp án:** Overfit → tăng dropout/weight decay.

**Câu 7 (tình huống thực nghiệm):** Train acc 60% sau nhiều epoch, val acc 58%, cả hai cải thiện rất chậm.  
**Đáp án:** Underfit → giảm regularization nếu đang quá mạnh, hoặc train thêm.

**Câu 8 (tính nhanh CE):** Nếu $p_y$ = 0.1 thì CE ≈ -log(0.1) ≈ 2.30. Nếu $p_y = 0.9$ thì $CE ≈ 0.105$. Ý nghĩa?  
**Đáp án:** Model càng tự tin đúng ($p_y$ cao) thì loss nhỏ; tự tin sai ($p_y$ thấp) bị phạt lớn.

---

## Kết nối 3 câu bắt buộc (tóm tắt 1 trang)
- **(a) Vì sao CNN phân loại ảnh?**  
  CNN học bộ lọc phát hiện mẫu cục bộ → chồng nhiều lớp để tổng hợp đặc trưng cấp cao → dùng đặc trưng đó phân biệt lớp.
- **(b) Vì sao overfit?**  
  Model có thể học thuộc chi tiết/ ồn của train, nên train tốt nhưng val/test kém (gap lớn).
- **(c) Vì sao dùng CrossEntropy?**  
  CE phù hợp phân loại: phạt mạnh khi xác suất lớp đúng thấp, tạo tín hiệu học đẩy lớp đúng lên (qua logits/softmax) và kéo lớp sai xuống.
 
