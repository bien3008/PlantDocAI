# Transfer Learning & Fine-tuning cho PlantDoc AI

## Mục lục

* [0) Mở đầu: Bối cảnh PlantDoc AI và vì sao chọn Transfer Learning](#0-mở-đầu-bối-cảnh-plantdoc-ai-và-vì-sao-chọn-transfer-learning)
* [1) Level 1 — Nền tảng Transfer Learning cho PlantDoc AI](#1-level-1--nền-tảng-transfer-learning-cho-plantdoc-ai)
* [2) Level 2 — Pretrained ImageNet như Feature Extractor](#2-level-2--pretrained-imagenet-như-feature-extractor)
* [3) Level 3 — Chiến lược huấn luyện 2 phase cho PlantDoc AI](#3-level-3--chiến-lược-huấn-luyện-2-phase-cho-plantdoc-ai)

  * [3.1 Phase 1: Freeze backbone → train classifier head (Baseline)](#31-phase-1-freeze-backbone--train-classifier-head-baseline)
  * [3.2 Phase 2: Unfreeze một phần/tất cả → fine-tune với LR nhỏ (Improved)](#32-phase-2-unfreeze-một-phầntất-cả--fine-tune-với-lr-nhỏ-improved)
  * [3.3 Cách mô tả freeze/unfreeze trong report đồ án](#33-cách-mô-tả-freezeunfreeze-trong-report-đồ-án)
* [4) Level 4 — LR Schedules, Early Stopping, Checkpoint “Best” cho PlantDoc AI](#4-level-4--lr-schedules-early-stopping-checkpoint-best-cho-plantdoc-ai)

  * [4.1 Learning Rate Schedules](#41-learning-rate-schedules)
  * [4.2 Early Stopping](#42-early-stopping)
  * [4.3 Checkpoint “Best”](#43-checkpoint-best)
* [5) Level 5 — Chuẩn bị code và pipeline huấn luyện cho PlantDoc AI](#5-level-5--chuẩn-bị-code-và-pipeline-huấn-luyện-cho-plantdoc-ai)

  * [5.1 Chuẩn bị môi trường và thư viện](#51-chuẩn-bị-môi-trường-và-thư-viện)
  * [5.2 Skeleton code cho baseline (Phase 1: feature extractor)](#52-skeleton-code-cho-baseline-phase-1-feature-extractor)
  * [5.3 Skeleton code cho improved (Phase 2: fine-tuning)](#53-skeleton-code-cho-improved-phase-2-fine-tuning)
  * [5.4 Tổ chức thí nghiệm để so sánh công bằng baseline vs improved](#54-tổ-chức-thí-nghiệm-để-so-sánh-công-bằng-baseline-vs-improved)
  * [5.5 Inference & tích hợp vào Streamlit (mức đủ dùng)](#55-inference--tích-hợp-vào-streamlit-mức-đủ-dùng)
* [6) Level 6 — Lỗi thường gặp trong PlantDoc AI và cách tránh](#6-level-6--lỗi-thường-gặp-trong-plantdoc-ai-và-cách-tránh)
* [7) Level 7 — Mở rộng vừa đủ cho PlantDoc AI](#7-level-7--mở-rộng-vừa-đủ-cho-plantdoc-ai)
* [8) Capstone Mini Plan — Kế hoạch thí nghiệm cho PlantDoc AI (áp dụng ngay)](#8-capstone-mini-plan--kế-hoạch-thí-nghiệm-cho-plantdoc-ai-áp-dụng-ngay)
* [9) Tổng kết](#9-tổng-kết)

---

## 0) Mở đầu: Bối cảnh PlantDoc AI và vì sao chọn Transfer Learning

PlantDoc AI là bài toán phân loại bệnh lá cây từ ảnh. Về bản chất, mô hình nhận ảnh đầu vào và dự đoán nhãn như “lá khỏe”, “bệnh A”, “bệnh B”,… Đây là bài toán image classification (phân loại ảnh), nhưng trong thực tế dữ liệu thường không hoàn hảo: số ảnh chưa nhiều, nhãn có thể nhiễu, điều kiện chụp không đồng đều (ánh sáng, góc chụp, nền, độ nét).

Nếu train mô hình CNN từ đầu (train from scratch), cần nhiều dữ liệu sạch và thời gian huấn luyện hơn để mô hình học được các đặc trưng cơ bản của ảnh. Với đồ án sinh viên, đây thường là điểm nghẽn: mô hình khó hội tụ, tốn thời gian thử nghiệm, và khó có baseline ổn định để demo sớm.

Transfer learning là cách rất hợp lý trong bối cảnh này. Thay vì bắt đầu từ random weights, ta dùng mô hình đã được pretrained (huấn luyện trước) trên ImageNet. Mô hình này đã học được nhiều đặc trưng thị giác chung như cạnh, texture, pattern, shape ở các mức trừu tượng khác nhau. Với ảnh lá cây, những đặc trưng như vệt bệnh, đốm, vùng đổi màu, texture bất thường vẫn có thể tận dụng tốt các kiến thức nền này.

Tư duy làm đồ án nên đi theo hướng: **baseline → improved → so sánh → giải thích**. Nghĩa là trước tiên tạo baseline nhanh, ổn định bằng feature extractor (đóng băng backbone, chỉ học head). Sau đó mới fine-tuning (mở một phần hoặc toàn bộ backbone) để cải thiện. Cuối cùng, so sánh kết quả một cách công bằng và giải thích rõ trong report vì sao freeze/unfreeze, LR nào, checkpoint nào được dùng.

---

## 1) Level 1 — Nền tảng Transfer Learning cho PlantDoc AI

Transfer learning trong computer vision là việc lấy một mô hình đã học từ một bài toán/dataset lớn (ví dụ ImageNet) rồi tái sử dụng cho bài toán mới. Ý chính là: mô hình không cần học lại mọi thứ từ đầu, mà kế thừa tri thức thị giác chung đã học trước đó.

Pretrained model là mô hình đã được huấn luyện sẵn. Khi nói “pretrained trên ImageNet”, nghĩa là mô hình đã được train trên một tập ảnh rất lớn và đa dạng với nhiều lớp đối tượng. Dù ImageNet không phải dataset bệnh lá cây, nó vẫn giúp mô hình học được các đặc trưng nền tảng có tính tổng quát.

Trong các mô hình như MobileNetV2 hay EfficientNet-B0, ta thường tách thành hai phần: **backbone** (phần trích xuất đặc trưng) và **classifier head** (phần phân loại cuối cùng). Backbone biến ảnh đầu vào thành biểu diễn đặc trưng giàu thông tin. Head nhận đặc trưng đó và dự đoán xác suất lớp.

Trực giác rất quan trọng ở đây: backbone giống như “mắt + hệ thống cảm nhận pattern”, còn head giống như “bộ ra quyết định cho bài toán cụ thể”. Với PlantDoc AI, backbone đã biết nhìn texture, biên, vùng đổi màu; head sẽ học cách ánh xạ các pattern đó thành nhãn bệnh lá cây cụ thể.

Ví dụ gần gũi: lá khỏe thường có texture đều, màu ổn định; lá bệnh có thể xuất hiện đốm nâu, loang vàng, vùng cháy mép, hoặc pattern không đồng nhất. Những khác biệt này phần lớn là khác biệt về texture và màu sắc cục bộ — đúng kiểu đặc trưng mà CNN pretrained thường xử lý tốt.

Transfer learning có thể kém hiệu quả khi domain shift (khác biệt miền dữ liệu) quá lớn. Ví dụ ảnh lá cây chụp trong điều kiện rất đặc thù (camera hồng ngoại, ảnh cực mờ, nền phức tạp bất thường, ảnh macro quá gần), hoặc dữ liệu bị lệch mạnh so với kiểu ảnh mà ImageNet giúp được. Khi đó, chỉ dùng feature extractor có thể chưa đủ và fine-tuning hoặc cải thiện dữ liệu sẽ quan trọng hơn.

---

## 2) Level 2 — Pretrained ImageNet như Feature Extractor

Khi dùng pretrained CNN như **feature extractor** (bộ trích xuất đặc trưng), ta giữ nguyên phần backbone đã học từ ImageNet và chỉ thay/huấn luyện lại classifier head cho số lớp bệnh lá cây thực tế của dataset. Nói đơn giản: backbone “rút ra đặc trưng”, head “học phân loại theo bài toán PlantDoc AI”.

Trong PlantDoc AI, điều này thường được làm bằng cách lấy MobileNetV2/EfficientNet-B0 pretrained, thay lớp cuối cùng (fully connected/classifier) thành một head mới có đầu ra bằng số lớp của dataset (ví dụ 10 lớp bệnh). Head mới ban đầu là “chưa học”, nên ta train nó trên dữ liệu lá cây, còn backbone tạm thời giữ nguyên.

Vì sao đây là baseline mạnh và nhanh? Vì phần khó nhất của CNN — học đặc trưng thị giác hữu ích — đã được pretrained hỗ trợ rất nhiều. Chỉ cần tối ưu phần head để thích nghi với nhãn bệnh lá cây. Kết quả là mô hình thường hội tụ nhanh hơn, ổn định hơn, và ít cần dữ liệu hơn so với train from scratch.

Trong đồ án, lợi ích rất thực tế: giảm thời gian train, dễ ra kết quả sớm để demo, dễ debug pipeline (split, transform, inference, mapping class). Khi baseline chạy ổn, mới chuyển sang fine-tuning để cải thiện. Đây là cách làm “an toàn và có tiến độ”.

Tuy nhiên, feature extractor cũng có giới hạn. Nếu ảnh lá cây trong dữ liệu khác nhiều về điều kiện chụp (thực địa, ánh sáng gắt, nền nhiễu, lá bị che khuất, nhiều lá trong cùng ảnh), backbone pretrained có thể chưa thích nghi đủ. Khi đó head học tốt đến một mức nào đó sẽ chững lại, và fine-tuning backbone là bước tiếp theo hợp lý.

Ảnh nền nhiễu, ánh sáng khác nhau, góc chụp khác nhau ảnh hưởng trực tiếp đến phân phối đặc trưng đầu vào. Nếu train phase 1 mà validation metric thấp hoặc bị dao động, không nhất thiết do model “dở”; nhiều khi là do domain gap lớn, transform chưa phù hợp, hoặc cần phase 2 để backbone học lại một phần đặc trưng sát bài toán lá cây hơn.

### So sánh nhanh: Train from scratch vs Feature extractor

| Tiêu chí                | Train from scratch | Feature extractor (pretrained ImageNet) |
| ----------------------- | ------------------ | --------------------------------------- |
| Tốc độ hội tụ           | Chậm hơn           | Nhanh hơn                               |
| Nhu cầu dữ liệu         | Cao hơn            | Thấp hơn (tương đối)                    |
| Độ ổn định baseline     | Thường kém hơn     | Thường tốt hơn                          |
| Phù hợp đồ án sinh viên | Khó hơn            | Rất phù hợp                             |

---

## 3) Level 3 — Chiến lược huấn luyện 2 phase cho PlantDoc AI

Đây là lõi của dự án và cũng là phần quan trọng nhất để viết report rõ ràng. Ý tưởng của chiến lược 2 phase là không fine-tune toàn bộ mô hình ngay từ đầu. Ta đi từng bước: đầu tiên học head để có baseline ổn định, sau đó mới unfreeze để tinh chỉnh backbone bằng learning rate nhỏ.

Cách này giúp cân bằng giữa tốc độ, độ ổn định, và khả năng cải thiện. Với dữ liệu lá cây không quá lớn, nếu unfreeze toàn bộ ngay từ đầu với LR không phù hợp, mô hình có thể “phá” kiến thức pretrained và cho validation metric rất thất thường.

### 3.1 Phase 1: Freeze backbone → train classifier head (Baseline)

**Freeze** (đóng băng) nghĩa là khóa tham số, không cho cập nhật trong quá trình backprop. Ở mức code, thường là đặt `requires_grad = False` (PyTorch) cho các tham số backbone. Khi đó gradient không được dùng để cập nhật chúng.

Trong phase này, phần backbone giữ nguyên weights pretrained, còn classifier head là phần được học. Nghĩa là mô hình vẫn đi qua toàn bộ backbone để trích xuất đặc trưng, nhưng optimizer chỉ cập nhật head.

Đây là baseline rất hợp lý cho PlantDoc AI vì nó nhanh, ổn định và ít rủi ro. Có thể sớm kiểm tra được các thành phần quan trọng của pipeline như data split, transform, loss, metrics, checkpoint, class mapping, inference. Nếu baseline đã tệ bất thường, thường lỗi nằm ở pipeline hơn là ở “chưa fine-tune”.

Ưu điểm lớn của phase 1 là tránh làm hỏng knowledge pretrained quá sớm. Đặc biệt với dữ liệu ít, việc chỉ học head giúp mô hình tận dụng tri thức sẵn có mà không overfit quá nhanh ở tầng sâu. Nhược điểm là backbone chưa thích nghi sâu với đặc trưng bệnh lá cây cụ thể, nên trần hiệu năng có thể bị giới hạn.

### 3.2 Phase 2: Unfreeze một phần/tất cả → fine-tune với LR nhỏ (Improved)

**Unfreeze** (mở băng) nghĩa là cho phép cập nhật lại một phần hoặc toàn bộ tham số backbone. Sau phase 1, head đã học được phân loại cơ bản. Lúc này ta dùng checkpoint tốt nhất của phase 1 làm điểm bắt đầu và fine-tune để mô hình thích nghi hơn với domain lá cây.

Có hai mức phổ biến. Một là unfreeze một phần, thường là block cuối của backbone (các tầng trừu tượng cao hơn, gần head hơn). Hai là unfreeze toàn bộ backbone. Với đồ án sinh viên, cách thực dụng là thử **unfreeze block cuối trước**, vì ít rủi ro hơn, nhẹ hơn, dễ ổn định hơn.

Vì sao phase 2 cần learning rate nhỏ hơn phase 1? Bởi backbone pretrained đã chứa kiến thức hữu ích. Ta chỉ muốn tinh chỉnh nhẹ để thích nghi domain mới, không muốn cập nhật quá mạnh làm mất tri thức đã học. LR lớn trong phase 2 có thể gây **catastrophic forgetting** (quên kiến thức cũ), loss/metric dao động mạnh, hoặc validation giảm dù train accuracy tăng.

Một dấu hiệu điển hình của LR phase 2 quá lớn là train loss giảm nhanh nhưng val loss tăng, val accuracy giảm hoặc lên xuống thất thường. Khi gặp tình huống này, thay vì kết luận “fine-tune không hiệu quả”, cần giảm LR, giảm số tầng unfreeze, hoặc tăng regularization/early stopping phù hợp.

Trong bối cảnh đồ án, không cần làm chiến lược quá phức tạp. Một lộ trình thực dụng là:

1. Phase 1: freeze toàn bộ backbone, train head.
2. Phase 2 (phiên bản đơn giản): unfreeze toàn bộ backbone với LR nhỏ.
3. Nếu thiếu tài nguyên hoặc dễ mất ổn định: chỉ unfreeze block cuối.

Khi nào chỉ cần phase 1? Khi baseline đã đủ tốt cho mục tiêu demo/report, dataset nhỏ và noisy, hoặc phase 2 không cải thiện ổn định sau vài lần thử hợp lý. Khi nào nên phase 2? Khi baseline còn dư địa cải thiện, dữ liệu đủ đa dạng và muốn giải thích rõ chiến lược fine-tuning trong đồ án.

### 3.3 Cách mô tả freeze/unfreeze trong report đồ án

Trong report, cần tách rõ **Baseline model** và **Improved model**. Baseline nên được mô tả là mô hình pretrained ImageNet dùng như feature extractor, với backbone frozen (đóng băng) và chỉ train classifier head. Improved nên được mô tả là mô hình fine-tuned sau khi unfreeze (một phần hoặc toàn bộ) backbone, dùng learning rate nhỏ hơn.

Mục tiêu của từng phase phải nói rõ. Phase 1 nhằm tạo baseline nhanh, ổn định, kiểm chứng pipeline. Phase 2 nhằm thích nghi backbone với domain bệnh lá cây để cải thiện hiệu năng. Nếu chỉ viết “đã fine-tune mô hình” thì quá mơ hồ và khó đánh giá chất lượng thực nghiệm.

Nếu improved tăng ít hoặc không tăng, vẫn có thể giải thích hợp lý và chuyên nghiệp. Ví dụ: dữ liệu nhỏ/noisy, domain gap không quá lớn nên feature extractor đã đủ tốt; hoặc phase 2 chưa tối ưu (mức unfreeze, LR, scheduler); hoặc metric cải thiện không đáng kể nhưng inference vẫn ổn định hơn. Điều quan trọng là trình bày trung thực, có cấu hình và số liệu đi kèm.

Nếu improved giảm, cũng không phải thất bại nếu bạn giải thích được. Đây là cơ hội để thể hiện hiểu biết: fine-tuning nhạy với LR, số tầng unfreeze, thời lượng train, dữ liệu nhiễu. Một report tốt không chỉ khoe điểm cao, mà còn chứng minh bạn hiểu vì sao kết quả xảy ra.

### So sánh nhanh Phase 1 vs Phase 2

| Tiêu chí           | Phase 1 (Freeze backbone) | Phase 2 (Unfreeze fine-tune)     |
| ------------------ | ------------------------- | -------------------------------- |
| Mục tiêu           | Baseline nhanh, ổn định   | Cải thiện hiệu năng              |
| Tham số trainable  | Chủ yếu head              | Head + một phần/toàn bộ backbone |
| Learning rate      | Thường lớn hơn phase 2    | Nhỏ hơn để tinh chỉnh            |
| Thời gian train    | Nhanh hơn                 | Chậm hơn                         |
| Rủi ro mất ổn định | Thấp                      | Cao hơn nếu LR lớn               |

---

## 4) Level 4 — LR Schedules, Early Stopping, Checkpoint “Best” cho PlantDoc AI

Ba thành phần này rất quan trọng trong thực chiến vì chúng quyết định mô hình có train ổn định, dừng đúng lúc và chọn đúng phiên bản để báo cáo/demo hay không. Nhiều đồ án có code train được nhưng kết quả không đáng tin chỉ vì bỏ qua checkpoint best hoặc dùng cấu hình dừng không hợp lý.

### 4.1 Learning Rate Schedules

Learning rate (LR) quyết định “bước nhảy” khi cập nhật tham số. Trong transfer learning, đặc biệt ở phase 2, LR quá lớn dễ phá weights pretrained; LR quá nhỏ thì mô hình học quá chậm hoặc gần như không cải thiện. Vì vậy, scheduler (bộ điều chỉnh LR theo thời gian) giúp việc train mượt hơn và ít phụ thuộc vào một LR cố định.

Phase 2 cần LR nhỏ hơn phase 1 vì backbone đã có tri thức tốt. Trong phase 1, head mới khởi tạo ngẫu nhiên nên cần LR tương đối lớn hơn để học nhanh. Sang phase 2, ta tinh chỉnh toàn mô hình hoặc một phần backbone nên cần LR nhỏ để tránh cập nhật quá mạnh.

**Cosine schedule** (cosine annealing) có trực giác đơn giản: LR giảm dần theo đường cong cosine, thường giúp fine-tuning ổn định hơn về cuối quá trình train. Nó phù hợp khi muốn mô hình học tương đối “êm”, đặc biệt ở phase 2, nơi cần tinh chỉnh nhẹ.

**OneCycle schedule** có trực giác là LR tăng lên rồi giảm xuống trong một chu kỳ. Cách này có thể giúp học nhanh và đôi khi cho kết quả tốt trong thời gian ngắn, nhưng cần thiết lập theo số bước/epoch rõ ràng. Với sinh viên, OneCycle dùng được nhưng dễ cấu hình sai hơn Cosine nếu chưa quen.

Gợi ý thực dụng: nếu muốn đơn giản và an toàn, dùng Cosine cho phase 2. Nếu muốn thử nghiệm thêm và đã nắm train loop theo số step, OneCycle là lựa chọn tốt cho một run so sánh. Không cần tuyệt đối hóa một scheduler nào.

Trong report, nên ghi rõ scheduler theo kiểu đủ hiểu: “Phase 2 dùng CosineAnnealingLR (T_max = số epoch fine-tune), LR khởi đầu nhỏ hơn phase 1”. Chỉ cần nói rõ cấu hình chính, không cần sa đà công thức.

**Tình huống minh họa ngắn:** train accuracy tăng đều nhưng val accuracy tăng chậm và val loss dao động mạnh ở phase 2. Một nguyên nhân thường gặp là LR quá lớn hoặc giảm LR chưa đủ mượt. Dùng scheduler + LR thấp hơn có thể giúp ổn định hơn.

### 4.2 Early Stopping

Early stopping là cơ chế dừng train sớm khi mô hình không còn cải thiện trên validation trong một số epoch liên tiếp. Mục tiêu trong PlantDoc AI là tránh overfitting và tiết kiệm thời gian, nhất là khi fine-tuning dễ overfit dữ liệu nhỏ.

Bạn cần chọn metric để theo dõi. Thực dụng nhất là `val_loss` vì nhạy với việc overfit sớm. Nếu dataset mất cân bằng lớp, có thể cân nhắc `macro-F1` (nếu bạn có tính). `val_accuracy` dễ hiểu nhưng đôi khi che giấu việc mô hình đang trở nên tự tin quá mức sai cách.

**Patience** là số epoch cho phép mô hình “không cải thiện” trước khi dừng. Ví dụ patience = 5 nghĩa là nếu 5 epoch liên tiếp metric không tốt hơn mức tốt nhất, ta dừng. Với đồ án, chọn patience ở mức vừa phải (ví dụ 5–10) thường đủ thực dụng.

Dừng quá sớm có thể làm bỏ lỡ cải thiện thật, nhất là khi metric dao động. Dừng quá muộn thì tốn thời gian và tăng nguy cơ overfit. Vì vậy early stopping nên đi cùng checkpoint best: dù có dừng ở đâu, bạn vẫn giữ được mô hình tốt nhất theo metric đã chọn.

**Tình huống minh họa ngắn:** train acc tiếp tục tăng nhưng val loss tăng dần từ epoch 8. Nếu không early stopping, bạn có thể train đến epoch 30 và chọn nhầm epoch cuối, dẫn đến demo/inference kém hơn rõ rệt.

### 4.3 Checkpoint “Best”

Checkpoint là file lưu trạng thái mô hình (và thường cả optimizer, epoch, metric). “Checkpoint best” là checkpoint tại epoch có metric validation tốt nhất theo tiêu chí bạn chọn. Đây là checkpoint nên dùng để đánh giá test và demo Streamlit.

Không nên mặc định dùng epoch cuối cho báo cáo/inference. Epoch cuối chỉ là thời điểm training dừng, không đảm bảo là mô hình tốt nhất. Trong thực tế, metric validation thường đạt đỉnh ở giữa quá trình train rồi giảm do overfit.

“Best” phải được định nghĩa rõ và thống nhất. Ví dụ: “best theo val_loss thấp nhất” hoặc “best theo val_accuracy cao nhất”. Nếu bạn dùng `val_loss` để early stopping thì cũng nên cân nhắc lưu best theo cùng metric để logic nhất quán.

Early stopping và checkpoint best bổ trợ nhau. Early stopping quyết định khi nào dừng; checkpoint best quyết định phiên bản nào được dùng. Dù dừng ở epoch 15, checkpoint best có thể là epoch 10 — và đó mới là model bạn dùng cho test/demo.

Khi tích hợp Streamlit, cần load đúng checkpoint best (đúng phase bạn chọn để deploy), khôi phục class mapping và transform tương ứng. Rất nhiều lỗi demo xảy ra vì load nhầm checkpoint phase 1/phase 2 hoặc dùng sai preprocessing.

**Tình huống minh họa ngắn:** train acc epoch cuối cao nhất nhưng val accuracy epoch cuối thấp hơn 3 epoch trước. Nếu dùng epoch cuối để demo, bạn có thể thấy kết quả kém hơn so với “best checkpoint” dù quá trình train nhìn có vẻ thành công.

---

## 5) Level 5 — Chuẩn bị code và pipeline huấn luyện cho PlantDoc AI

Phần này tập trung vào skeleton code bằng **PyTorch** để bạn triển khai được baseline + improved theo hướng sạch, dễ debug, và dễ viết report. Code dưới đây không phải production hoàn chỉnh, nhưng đủ chắc để làm đồ án và mở rộng dần.

### 5.1 Chuẩn bị môi trường và thư viện

Bạn nên thống nhất một stack đơn giản: Python + PyTorch + torchvision, và có thể thêm `timm` nếu muốn dùng model pretrained tiện hơn. `scikit-learn` hữu ích để tính metrics hoặc split (nếu tự quản lý split), `matplotlib` để vẽ loss/accuracy curves.

Cấu trúc dữ liệu nên rõ từ đầu. Cách dễ dùng nhất với `ImageFolder` là tách thư mục thành `train/`, `val/`, `test/`, mỗi thư mục chứa các lớp con theo tên class. Nếu chưa có `test`, ít nhất cần `train/` và `val/`, sau đó giữ lại `test` để đánh giá cuối.

Bạn nên cố định seed để reproducibility ở mức cơ bản và xác định `device` (`cuda` nếu có GPU, ngược lại `cpu`). Reproducibility không bao giờ tuyệt đối 100% trong DL, nhưng làm tốt bước này giúp so sánh baseline vs improved công bằng hơn.

Về tổ chức file dự án, nên tách tối thiểu thành `dataset.py`, `train.py`, `evaluate.py`, `infer.py`, `utils.py`. Mục tiêu là dễ đọc, dễ debug và dễ mô tả trong report hơn một file notebook quá dài.

```python
# requirements gợi ý (không phải file bắt buộc)
# torch torchvision timm scikit-learn matplotlib streamlit pillow
```

Đoạn này chỉ là gợi ý môi trường. Quan trọng nhất là thống nhất framework và cách tổ chức ngay từ đầu để tránh “đến lúc demo mới vá code”.

---

### 5.2 Skeleton code cho baseline (Phase 1: feature extractor)

Dưới đây là skeleton theo hướng `torchvision` với MobileNetV2. Bạn có thể đổi sang EfficientNet-B0 tương tự.

```python
# train_phase1_baseline.py
import os
import copy
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights

# =========================
# 1) Config
# =========================
class CFG:
    data_dir = "data"  # data/train, data/val, data/test
    img_size = 224
    batch_size = 32
    num_workers = 2
    lr = 1e-3                 # Phase 1 LR (head only)
    epochs = 15
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "checkpoints"
    best_metric_name = "val_acc"   # hoặc "val_loss"
    maximize_metric = True         # True nếu best theo accuracy; False nếu best theo loss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 2) Data
# =========================
def build_dataloaders(data_dir, img_size, batch_size, num_workers):
    # Dùng normalize theo ImageNet vì backbone pretrained trên ImageNet
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # augmentation nhẹ (phục vụ trực tiếp transfer learning)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

# =========================
# 3) Model: pretrained + replace head + freeze backbone
# =========================
def build_model_mobilenetv2(num_classes):
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze backbone (features)
    for p in model.features.parameters():
        p.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

# =========================
# 4) Train / Val loops
# =========================
def run_one_epoch(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc

# =========================
# 5) Checkpoint utils
# =========================
def is_better(curr, best, maximize=True):
    return curr > best if maximize else curr < best

def save_checkpoint(path, model, optimizer, epoch, best_metric, class_names):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "class_names": class_names,
    }
    torch.save(ckpt, path)

# =========================
# 6) Main train (Phase 1)
# =========================
def main():
    cfg = CFG()
    set_seed(cfg.seed)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(
        cfg.data_dir, cfg.img_size, cfg.batch_size, cfg.num_workers
    )
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = build_model_mobilenetv2(num_classes).to(cfg.device)
    criterion = nn.CrossEntropyLoss()

    # IMPORTANT: optimizer chỉ nhận params trainable (head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=cfg.lr)

    best_metric = -float("inf") if cfg.maximize_metric else float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, is_train=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, cfg.device, is_train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_metric = val_acc if cfg.best_metric_name == "val_acc" else val_loss
        if is_better(current_metric, best_metric, cfg.maximize_metric):
            best_metric = current_metric
            save_checkpoint(
                os.path.join(cfg.save_dir, "best_phase1.pth"),
                model, optimizer, epoch, best_metric, class_names
            )

        print(f"[Phase1][Epoch {epoch}/{cfg.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    print(f"Done Phase 1. Best {cfg.best_metric_name} = {best_metric:.4f}")

if __name__ == "__main__":
    main()
```

Đoạn code này làm đúng lõi baseline cho PlantDoc AI: load pretrained model, thay head theo số lớp, freeze backbone, chỉ tối ưu head, train/validate và lưu checkpoint tốt nhất. Điểm quan trọng nhất là **optimizer chỉ nhận các tham số `requires_grad=True`**, nếu không bạn dễ “freeze giả” (tưởng freeze nhưng optimizer vẫn giữ params không cần thiết hoặc code khó kiểm soát).

---

### 5.3 Skeleton code cho improved (Phase 2: fine-tuning)

Phase 2 bắt đầu từ checkpoint tốt nhất của phase 1. Sau đó unfreeze một phần hoặc toàn bộ backbone, giảm LR, thêm scheduler và early stopping.

```python
# train_phase2_finetune.py
import os
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights

# =========================
# Config
# =========================
class CFG:
    data_dir = "data"
    img_size = 224
    batch_size = 32
    num_workers = 2
    lr = 1e-4              # Phase 2 LR nhỏ hơn phase 1
    epochs = 20
    patience = 5           # early stopping
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "checkpoints"
    phase1_ckpt = "checkpoints/best_phase1.pth"

    # chọn metric best
    best_metric_name = "val_acc"
    maximize_metric = True

    # unfreeze strategy: "all" hoặc "last_block"
    unfreeze_mode = "all"

def build_dataloaders(data_dir, img_size, batch_size, num_workers):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def build_model_mobilenetv2(num_classes):
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def load_phase1_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt

def freeze_all_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False

def unfreeze_all_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = True

def unfreeze_last_block_only(model):
    # MobileNetV2: features là Sequential; unfreeze vài layer cuối (ví dụ từ index 15 trở đi)
    for p in model.features.parameters():
        p.requires_grad = False

    for idx, block in enumerate(model.features):
        if idx >= 15:
            for p in block.parameters():
                p.requires_grad = True

def run_one_epoch(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples

def is_better(curr, best, maximize=True):
    return curr > best if maximize else curr < best

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, class_names):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "class_names": class_names,
    }
    torch.save(ckpt, path)

def main():
    cfg = CFG()
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(
        cfg.data_dir, cfg.img_size, cfg.batch_size, cfg.num_workers
    )
    num_classes = len(class_names)

    model = build_model_mobilenetv2(num_classes).to(cfg.device)

    # Load best checkpoint từ phase 1
    phase1_ckpt = load_phase1_checkpoint(model, cfg.phase1_ckpt, cfg.device)
    print(f"Loaded phase1 checkpoint from epoch {phase1_ckpt.get('epoch')}")

    # Unfreeze strategy
    if cfg.unfreeze_mode == "all":
        unfreeze_all_backbone(model)
    elif cfg.unfreeze_mode == "last_block":
        unfreeze_last_block_only(model)
    else:
        raise ValueError("Invalid unfreeze_mode")

    # Head luôn trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    # Option 1 (đơn giản): một LR chung nhỏ cho tất cả trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=cfg.lr)

    # Cosine scheduler (gợi ý an toàn cho fine-tuning)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_metric = -float("inf") if cfg.maximize_metric else float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, is_train=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, cfg.device, is_train=False
        )

        scheduler.step()

        current_metric = val_acc if cfg.best_metric_name == "val_acc" else val_loss

        if is_better(current_metric, best_metric, cfg.maximize_metric):
            best_metric = current_metric
            epochs_no_improve = 0
            save_checkpoint(
                os.path.join(cfg.save_dir, "best_phase2.pth"),
                model, optimizer, scheduler, epoch, best_metric, class_names
            )
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Phase2][Epoch {epoch}/{cfg.epochs}] "
              f"lr={current_lr:.6e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"no_improve={epochs_no_improve}")

        # Early stopping
        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping triggered. Best {cfg.best_metric_name}={best_metric:.4f}")
            break

    print(f"Done Phase 2. Best {cfg.best_metric_name} = {best_metric:.4f}")

if __name__ == "__main__":
    main()
```

Đoạn code này thể hiện đúng tinh thần improved model: bắt đầu từ baseline tốt nhất, unfreeze để fine-tune với LR nhỏ, dùng scheduler và early stopping, và tiếp tục lưu checkpoint best của phase 2. Đây là phần rất quan trọng để bạn giải thích trong report rằng “improved model” không chỉ là train lâu hơn, mà là thay đổi chiến lược huấn luyện có kiểm soát.

---

### 5.4 Tổ chức thí nghiệm để so sánh công bằng baseline vs improved

So sánh công bằng nghĩa là baseline và improved phải dùng cùng data split, cùng preprocessing/normalization, và càng ít thay đổi ngoài biến số nghiên cứu càng tốt. Nếu vừa unfreeze vừa đổi augmentation mạnh vừa đổi model, bạn sẽ khó biết cải thiện đến từ đâu.

Trong report, mỗi thí nghiệm nên có cấu hình rõ: model backbone (MobileNetV2/EfficientNet-B0), phase (freeze/unfreeze), LR, scheduler, epochs tối đa, early stopping patience, metric chọn best checkpoint. Chỉ cần bảng cấu hình gọn nhưng đủ tái lập.

Bạn nên log các metric cốt lõi như train/val loss, train/val accuracy, và nếu có thể thêm macro-F1 cho validation/test (đặc biệt khi dữ liệu mất cân bằng lớp). Với đồ án, việc ghi log rõ ràng quan trọng hơn thử quá nhiều biến thể mà không theo dõi được.

Cách lập bảng kết quả trong report nên tách “Baseline” và “Improved” với cột ghi rõ checkpoint metric. Nếu improved tăng ít, đừng overclaim kiểu “cải thiện đáng kể” khi chênh lệch rất nhỏ. Trình bày trung thực sẽ thuyết phục hơn.

---

### 5.5 Inference & tích hợp vào Streamlit (mức đủ dùng)

Khi demo Streamlit, điều quan trọng nhất là **load đúng checkpoint best**, chuyển model sang `eval()` và dùng preprocessing giống lúc train/val (đặc biệt resize + normalize). Rất nhiều lỗi demo xảy ra vì inference transform khác train transform.

Dưới đây là skeleton inference tối thiểu.

```python
# infer.py
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)  # inference load từ checkpoint
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_infer_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def load_model_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    class_names = ckpt["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

@torch.no_grad()
def predict_image(model, class_names, image_pil, topk=3):
    tfm = get_infer_transform()
    x = tfm(image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)

    top_probs, top_idxs = torch.topk(probs, k=min(topk, probs.shape[1]), dim=1)

    results = []
    for p, idx in zip(top_probs[0].cpu().tolist(), top_idxs[0].cpu().tolist()):
        results.append({
            "class_name": class_names[idx],
            "prob": float(p)
        })
    return results

if __name__ == "__main__":
    model, class_names = load_model_from_checkpoint("checkpoints/best_phase2.pth")
    img = Image.open("sample_leaf.jpg")
    results = predict_image(model, class_names, img, topk=3)
    print(results)
```

Đoạn code này đủ để tích hợp vào Streamlit: upload ảnh → `PIL.Image` → gọi `predict_image()` → hiển thị top-1 hoặc top-k. Điểm quan trọng là `class_names` được lưu trong checkpoint để tránh sai class mapping giữa train và demo.

Ví dụ skeleton Streamlit tối giản:

```python
# app.py (skeleton)
import streamlit as st
from PIL import Image
from infer import load_model_from_checkpoint, predict_image

st.title("PlantDoc AI - Plant Leaf Disease Classification")

CKPT_PATH = "checkpoints/best_phase2.pth"  # hoặc best_phase1.pth nếu deploy baseline
model, class_names = load_model_from_checkpoint(CKPT_PATH)

uploaded = st.file_uploader("Tải ảnh lá cây", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    results = predict_image(model, class_names, image, topk=3)

    st.subheader("Kết quả dự đoán")
    for i, r in enumerate(results, 1):
        st.write(f"Top-{i}: **{r['class_name']}** — {r['prob']:.4f}")
```

Skeleton Streamlit này giúp demo nhanh và đủ sạch. Khi mở rộng, bạn có thể thêm cảnh báo ảnh không phải lá, hiển thị Grad-CAM, hoặc gợi ý xử lý, nhưng lõi inference phải đúng trước.

---

## 6) Level 6 — Lỗi thường gặp trong PlantDoc AI và cách tránh

Phần này rất sát đồ án sinh viên vì nhiều lỗi không nằm ở mô hình, mà nằm ở cách tổ chức train/inference/report.

**1) Freeze nhưng optimizer vẫn đang cập nhật nhầm tham số**
Dấu hiệu là bạn nghĩ đang train head-only nhưng số tham số trainable quá lớn, train chậm bất thường, hoặc hành vi giống fine-tuning. Cách khắc phục là sau khi freeze, luôn tạo optimizer từ danh sách `[p for p in model.parameters() if p.requires_grad]` và in ra số tham số trainable để kiểm tra.

**2) Quên chuyển `model.train()` / `model.eval()` đúng chỗ**
Dấu hiệu là validation metric dao động lạ, dropout/batchnorm hoạt động sai chế độ, kết quả inference không ổn định giữa các lần chạy. Cách khắc phục là đặt rõ `model.train()` trong train loop và `model.eval()` trong val/infer; khi infer thêm `torch.no_grad()`.

**3) Fine-tune với LR quá lớn làm val metric giảm mạnh**
Dấu hiệu là phase 2 vừa bắt đầu thì train loss giảm nhưng val loss tăng nhanh, val accuracy tụt hoặc dao động mạnh. Cách khắc phục là giảm LR phase 2 (thường nhỏ hơn phase 1), unfreeze ít tầng hơn, dùng scheduler như cosine, và kiểm tra lại checkpoint khởi đầu từ phase 1.

**4) So sánh baseline vs improved không công bằng**
Dấu hiệu là baseline và improved dùng split khác nhau, transform khác nhau, hoặc thay nhiều thứ cùng lúc nên không biết nguyên nhân cải thiện. Cách khắc phục là giữ nguyên split và preprocessing khi mục tiêu là so sánh freeze vs unfreeze; chỉ thay biến số cần nghiên cứu.

**5) Dùng epoch cuối thay vì checkpoint best**
Dấu hiệu là kết quả test/demo tệ hơn những gì log validation từng đạt được. Cách khắc phục là luôn lưu checkpoint best theo metric đã chọn và dùng checkpoint đó cho đánh giá/test/Streamlit.

**6) Early stopping cấu hình không hợp lý**
Dấu hiệu dừng quá sớm: metric có xu hướng cải thiện chậm nhưng train bị cắt trước khi kịp tăng. Dấu hiệu dừng quá muộn: train kéo dài dù val metric không cải thiện từ lâu. Cách khắc phục là chọn patience vừa phải và xem learning curves để điều chỉnh.

**7) Inference dùng transform khác lúc train**
Dấu hiệu là model test nội bộ ổn nhưng demo ảnh ngoài dự đoán sai hàng loạt, đặc biệt khi quên normalize hoặc resize khác. Cách khắc phục là tách transform inference thành hàm rõ ràng và dùng cùng chuẩn normalize ImageNet như lúc train.

**8) Report ghi “fine-tune” nhưng không mô tả rõ phase, LR, phần unfreeze**
Dấu hiệu là report khó thuyết phục, người đọc không tái lập được, giảng viên khó đánh giá bạn hiểu hay chỉ chạy code mẫu. Cách khắc phục là mô tả rõ: freeze/unfreeze phần nào, LR từng phase, scheduler, metric chọn best, checkpoint nào dùng để báo cáo/test.

---

## 7) Level 7 — Mở rộng vừa đủ cho PlantDoc AI

Phần mở rộng này chỉ nên làm khi baseline + improved đã ổn, để tránh “đụng cái gì cũng thử” nhưng không kiểm soát được.

**Progressive unfreezing (unfreeze theo từng block)** là chiến lược mở dần các block của backbone theo từng giai đoạn thay vì unfreeze toàn bộ ngay. Nó hữu ích khi phase 2 dễ mất ổn định hoặc dataset nhỏ. Với đồ án, bạn có thể thử một phiên bản đơn giản: phase 2a unfreeze block cuối, phase 2b unfreeze toàn bộ nếu còn thời gian.

**Discriminative learning rates** là dùng LR khác nhau cho head và backbone (thường head LR > backbone LR). Trực giác là head cần học nhiều hơn, backbone chỉ tinh chỉnh nhẹ. Cách này thường hợp lý trong fine-tuning nhưng làm code phức tạp hơn một chút, nên chỉ thêm khi bạn muốn nâng chất lượng thực nghiệm/report.

Ví dụ optimizer với nhiều param groups (ý tưởng skeleton):

```python
optimizer = torch.optim.Adam([
    {"params": model.features.parameters(), "lr": 1e-5},   # backbone nhỏ hơn
    {"params": model.classifier.parameters(), "lr": 1e-4}, # head lớn hơn
])
```

**Chọn MobileNetV2 vs EfficientNet-B0** trong đồ án là trade-off giữa tốc độ, độ chính xác và kích thước model. MobileNetV2 thường nhẹ, nhanh, hợp demo CPU/Streamlit. EfficientNet-B0 thường có thể cho accuracy tốt hơn một chút trong nhiều trường hợp nhưng chậm hơn/nhạy cấu hình hơn. Không có lựa chọn “đúng tuyệt đối”; lựa chọn đúng là lựa chọn phù hợp mục tiêu demo + tài nguyên của bạn.

Cuối cùng, có những lúc transfer learning không đủ để cải thiện thêm. Khi đó, điểm nghẽn thường nằm ở dữ liệu: nhãn chưa sạch, lớp mất cân bằng nặng, ảnh quá nhiễu, không đủ đa dạng điều kiện chụp. Lúc này, thêm dữ liệu tốt hơn hoặc cải thiện chất lượng nhãn có thể hiệu quả hơn tiếp tục tinh chỉnh hyperparameters.

---

## 8) Capstone Mini Plan — Kế hoạch thí nghiệm cho PlantDoc AI (áp dụng ngay)

Mục tiêu của kế hoạch này là tạo được một pipeline rõ ràng để có **baseline + improved**, đủ dữ liệu cho báo cáo, và đủ giải thích freeze/unfreeze một cách thuyết phục. Không cần quá nhiều run, nhưng mỗi run phải có mục đích.

Bạn có thể bắt đầu với 3–5 run tối thiểu. Cách tổ chức tốt là mỗi run chỉ thay đổi một yếu tố chính để dễ diễn giải.

### Mục tiêu thí nghiệm

Mục tiêu chính là:

* Xây baseline nhanh, ổn định bằng feature extractor (Phase 1).
* Cải thiện bằng fine-tuning (Phase 2) với LR nhỏ hơn.
* So sánh công bằng và giải thích rõ kết quả.

### Gợi ý danh sách thí nghiệm tối thiểu (3–5 run)

1. **Run A (Baseline):** MobileNetV2, freeze backbone, train head (Phase 1).
2. **Run B (Improved-1):** Từ best Run A, unfreeze toàn bộ backbone, LR nhỏ, cosine scheduler.
3. **Run C (Improved-2):** Từ best Run A, unfreeze block cuối, LR nhỏ, cosine scheduler.
4. *(Tùy chọn)* **Run D:** EfficientNet-B0 baseline (feature extractor).
5. *(Tùy chọn)* **Run E:** EfficientNet-B0 fine-tune để so với MobileNetV2 nếu còn thời gian/tài nguyên.

### Bảng cấu hình thí nghiệm mẫu (đưa vào report)

| Run | Model           | Phase | Freeze/Unfreeze     |   LR | Scheduler            | Epochs max | Early stopping  | Checkpoint best theo |
| --- | --------------- | ----- | ------------------- | ---: | -------------------- | ---------: | --------------- | -------------------- |
| A   | MobileNetV2     | 1     | Freeze backbone     | 1e-3 | None / Step đơn giản |         15 | Có (patience=5) | val_acc              |
| B   | MobileNetV2     | 2     | Unfreeze all        | 1e-4 | Cosine               |         20 | Có (patience=5) | val_acc              |
| C   | MobileNetV2     | 2     | Unfreeze last block | 1e-4 | Cosine               |         20 | Có (patience=5) | val_acc              |
| D   | EfficientNet-B0 | 1     | Freeze backbone     | 1e-3 | None / Cosine        |         15 | Có              | val_acc              |

Bạn không cần chạy đủ hết nếu thiếu thời gian. Chỉ cần A + B + C là đã đủ để giải thích freeze/unfreeze rất tốt.

### Metrics nên báo cáo

Nên báo cáo tối thiểu:

* Validation accuracy (best)
* Test accuracy (trên checkpoint best)
* Validation/test loss
* (Nếu có) macro-F1, đặc biệt khi dữ liệu mất cân bằng lớp

Nếu có thêm confusion matrix cho test thì rất tốt, nhưng chỉ nên thêm nếu bạn đã hiểu và giải thích được các lớp dễ nhầm.

### Cách lập bảng kết quả và kết luận trung thực

Bảng kết quả nên ghi rõ metric, checkpoint và cấu hình run. Nếu improved chỉ tăng nhẹ (ví dụ +0.5% đến +2%), vẫn có giá trị nếu bạn mô tả đúng điều kiện và độ ổn định. Nếu improved không tăng, hãy nói rõ rằng baseline feature extractor đã mạnh trong bối cảnh dataset hiện tại hoặc phase 2 cần tuning thêm.

Một kết luận thuyết phục thường có dạng: “Phase 1 cho baseline ổn định, hội tụ nhanh và phù hợp demo. Phase 2 với unfreeze + LR nhỏ giúp (hoặc chưa giúp) cải thiện validation/test metric trong cấu hình hiện tại. Kết quả cho thấy fine-tuning cần kiểm soát LR và mức unfreeze để tránh mất ổn định.”

### Checklist “đủ để làm dự án” và đủ để viết report freeze/unfreeze

* [ ] Có baseline feature extractor chạy ổn, lưu checkpoint best.
* [ ] Có ít nhất 1 run fine-tuning (phase 2) từ checkpoint phase 1.
* [ ] Ghi rõ freeze/unfreeze phần nào và LR từng phase.
* [ ] Có early stopping + checkpoint best, không dùng epoch cuối mặc định.
* [ ] So sánh công bằng cùng split/preprocessing.
* [ ] Inference/Streamlit load đúng checkpoint best và đúng class mapping.
* [ ] Report mô tả rõ baseline vs improved, không viết mơ hồ “đã fine-tune”.

---

## 9) Tổng kết

Trong PlantDoc AI, pretrained ImageNet giúp bạn tạo baseline mạnh và nhanh vì backbone đã học được các đặc trưng thị giác tổng quát như cạnh, texture, pattern — những thứ vẫn rất hữu ích cho ảnh lá cây và biểu hiện bệnh trên lá. Nhờ vậy, bạn không cần train từ đầu với dữ liệu thường còn hạn chế của đồ án.

Chiến lược 2 phase **freeze → unfreeze** là cách làm thực dụng và dễ giải thích. Phase 1 (freeze backbone, train head) giúp có baseline ổn định và kiểm chứng pipeline. Phase 2 (unfreeze để fine-tune) giúp mô hình thích nghi sâu hơn với domain lá cây, nhưng cần kiểm soát bằng learning rate nhỏ để tránh làm hỏng kiến thức pretrained.

Phase 2 cần LR nhỏ hơn phase 1 vì bạn đang tinh chỉnh weights đã tốt sẵn, không phải học từ đầu. Nếu LR quá lớn, mô hình dễ mất ổn định, val metric dao động hoặc giảm. Đây là lý do cần scheduler, đặc biệt như cosine hoặc onecycle, để điều chỉnh LR hợp lý trong quá trình fine-tune.

Scheduler, early stopping và checkpoint best là bộ ba quan trọng để train ổn định và đánh giá đúng mô hình. Scheduler giúp tối ưu mượt hơn, early stopping giúp tránh overfit và tiết kiệm thời gian, còn checkpoint best đảm bảo bạn dùng đúng phiên bản tốt nhất cho test và demo Streamlit thay vì epoch cuối.

### Checklist ngắn để triển khai và báo cáo đồ án

* [ ] Dùng pretrained ImageNet (MobileNetV2/EfficientNet-B0) làm baseline feature extractor.
* [ ] Phase 1: freeze backbone, train classifier head, lưu best checkpoint.
* [ ] Phase 2: load best phase 1, unfreeze (một phần/toàn bộ), LR nhỏ hơn, dùng scheduler.
* [ ] Thêm early stopping + checkpoint best theo metric thống nhất.
* [ ] So sánh baseline vs improved trên cùng split và preprocessing.
* [ ] Inference/Streamlit dùng đúng transform + class mapping + best checkpoint.
* [ ] Report ghi rõ freeze/unfreeze, LR, scheduler, early stopping, metric chọn best.
 