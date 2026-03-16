# Cách làm việc với model trong PlantDoc AI

## Mục lục

* [Mở đầu](#mở-đầu)
* [Level 0 — Tổng quan: “Làm việc với model” trong một dự án AI nghĩa là gì?](#level-0--tổng-quan-làm-việc-với-model-trong-một-dự-án-ai-nghĩa-là-gì)
* [Level 1 — Chọn model cho PlantDoc AI: bắt đầu từ đâu?](#level-1--chọn-model-cho-plantdoc-ai-bắt-đầu-từ-đâu)
* [Level 2 — Hiểu cấu trúc model khi dùng transfer learning](#level-2--hiểu-cấu-trúc-model-khi-dùng-transfer-learning)
* [Level 3 — Cách train model đúng trong PlantDoc AI](#level-3--cách-train-model-đúng-trong-plantdoc-ai)
* [Level 4 — Làm việc với kết quả của model: đọc gì, hiểu gì, tin gì?](#level-4--làm-việc-với-kết-quả-của-model-đọc-gì-hiểu-gì-tin-gì)
* [Level 5 — Lưu, tải và quản lý phiên bản model](#level-5--lưu-tải-và-quản-lý-phiên-bản-model)
* [Level 6 — Inference: dùng model để dự đoán ảnh mới như thế nào?](#level-6--inference-dùng-model-để-dự-đoán-ảnh-mới-như-thế-nào)
* [Level 7 — So sánh và cải thiện model một cách có hệ thống](#level-7--so-sánh-và-cải-thiện-model-một-cách-có-hệ-thống)
* [Level 8 — Phân tích lỗi và hiểu giới hạn của model](#level-8--phân-tích-lỗi-và-hiểu-giới-hạn-của-model)
* [Level 9 — Đưa model vào ứng dụng PlantDoc AI](#level-9--đưa-model-vào-ứng-dụng-plantdoc-ai)
* [Level 10 — Viết phần “làm việc với model” vào báo cáo đồ án](#level-10--viết-phần-làm-việc-với-model-vào-báo-cáo-đồ-án)
* [Level 11 — Capstone mini-plan cho PlantDoc AI](#level-11--capstone-mini-plan-cho-plantdoc-ai)
* [Kết luận](#kết-luận)

---

## Mở đầu

Khi mới làm một dự án AI, nhiều sinh viên thường nghĩ phần model chỉ là chọn một mạng nào đó, viết `train.py`, chạy vài epoch rồi lấy accuracy cao nhất để đưa vào báo cáo. Cách nghĩ đó rất dễ làm cho dự án thiếu chiều sâu, thiếu tính kiểm soát, và đến lúc demo hoặc viết báo cáo thì không giải thích được vì sao model lại tốt hoặc vì sao model lại sai.

Trong PlantDoc AI, “làm việc với model” phải được hiểu như một quá trình đầy đủ. Ta không chỉ quan tâm model là MobileNetV2 hay EfficientNet-B0, mà còn phải quan tâm model đó được chọn vì lý do gì, được train theo chiến lược nào, được đánh giá ra sao, checkpoint nào mới thật sự là phiên bản tốt nhất, pipeline inference có khớp với lúc train hay không, và nếu model sai thì phải phân tích sai do đâu. Khi nhìn model theo tư duy vòng đời thay vì tư duy “chạy cho ra kết quả”, em sẽ làm dự án chắc tay hơn rất nhiều.

Tài liệu này được viết để giúp em hình dung rõ cách làm việc với model trong bối cảnh PlantDoc AI, tức là một bài toán phân loại bệnh lá cây bằng PyTorch, dùng transfer learning từ ImageNet, có train/val/test rõ ràng, có best checkpoint, có inference để demo bằng Streamlit, và có thể có thêm Grad-CAM để hỗ trợ giải thích mô hình khi cần.

---

## Level 0 — Tổng quan: “Làm việc với model” trong một dự án AI nghĩa là gì?

### Mục tiêu của level

Mục tiêu của level này là giúp em nhìn được bức tranh toàn cảnh. Sau phần này, em phải hiểu rằng model trong PlantDoc AI không chỉ là một file `.pt` hay một kiến trúc CNN, mà là trung tâm của một vòng đời kỹ thuật hoàn chỉnh.

### Giải thích

Trong PlantDoc AI, model là thành phần nhận ảnh lá cây đã qua tiền xử lý và đưa ra dự đoán lớp bệnh tương ứng. Nhưng nếu chỉ dừng ở định nghĩa đó thì vẫn còn quá hẹp. Trong một dự án thật, model không sống một mình. Nó gắn với dữ liệu, cách chia train/val/test, cách huấn luyện, cách đo chất lượng, cách lưu checkpoint, cách tải lại để suy luận, cách tích hợp vào ứng dụng, và cả cách giải thích trong báo cáo.

Nói cách khác, làm việc với model là quản lý toàn bộ vòng đời của nó. Em bắt đầu từ việc chọn một model hợp lý để làm baseline. Sau đó em train baseline để có một cột mốc đầu tiên. Tiếp theo em dùng validation để xem model có học đúng hướng không. Khi đã có baseline, em mới cải tiến bằng fine-tuning, augmentation hoặc đổi backbone. Sau đó em đánh giá trên test, lưu best checkpoint, dùng checkpoint đó cho inference và demo, rồi phân tích lỗi để quyết định vòng cải tiến tiếp theo.

Tư duy này rất quan trọng trong PlantDoc AI vì bài toán ảnh lá cây dễ tạo ra ảo giác rằng model “đã ổn” chỉ vì accuracy cao trên một tập nào đó. Thực tế, một model chỉ đáng tin khi em biết rõ nó được train thế nào, được chọn thế nào, được lưu bản nào, và nó sai trong những trường hợp nào.

Một cách hình dung đơn giản là vòng đời model trong dự án như sau:

`chọn model -> train baseline -> validate -> improve -> test -> save best -> inference -> demo -> phân tích lỗi -> lặp lại`

Nếu em bỏ qua một mắt xích trong chuỗi này, dự án vẫn có thể chạy được, nhưng rất dễ yếu ở phần kỹ thuật hoặc phần trình bày. Ví dụ, train xong mà không lưu đúng checkpoint thì đến lúc demo em có thể dùng nhầm model. Hoặc inference dùng transform khác lúc train thì model dự đoán sai nhưng em không biết lỗi nằm ở đâu.

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử em chọn MobileNetV2 làm baseline cho PlantDoc AI. Em thay classifier head để số đầu ra bằng số lớp bệnh trong dataset. Em freeze backbone, train head vài epoch, thấy val accuracy đạt mức ổn định. Sau đó em unfreeze một phần backbone và fine-tune với learning rate nhỏ hơn. Trong quá trình train, em lưu best checkpoint theo `val_f1`. Cuối cùng em lấy đúng checkpoint đó để đưa vào app Streamlit.

Nếu sau này em đổi sang EfficientNet-B0 và kết quả tốt hơn một chút, điều quan trọng không chỉ là “model mới cao hơn”. Điều quan trọng là em biết mình đã đổi gì, giữ nguyên gì, checkpoint nào được chọn, và vì sao em quyết định dùng phiên bản nào cho demo. Đó chính là làm việc với model theo kiểu dự án thật.

### Lỗi hay gặp

* Nghĩ rằng làm việc với model chỉ là train.
* Chỉ quan tâm accuracy cuối cùng.
* Không tách rõ train, validation và test.
* Không lưu best checkpoint.
* Train một kiểu, inference một kiểu.
* Không ghi lại cấu hình của model đã dùng.

### Checklist áp dụng

* Em có mô tả được vòng đời model trong PlantDoc AI không?
* Em có biết model hiện tại là baseline hay improved version không?
* Em có biết checkpoint nào đang được dùng cho demo không?
* Em có thể giải thích model được đánh giá bằng metric nào không?

---

## Level 1 — Chọn model cho PlantDoc AI: bắt đầu từ đâu?

### Mục tiêu của level

Sau level này, em phải biết cách chọn một model khởi đầu hợp lý cho bài toán lá cây, thay vì chọn theo cảm tính hoặc chạy theo model nặng.

### Giải thích

Với một đồ án như PlantDoc AI, lựa chọn model đầu tiên nên dựa trên tính thực dụng. Mục tiêu ban đầu không phải là săn mô hình mạnh nhất thị trường, mà là có một baseline đủ tốt, dễ train, dễ debug, dễ demo và dễ giải thích. Đây là lý do vì sao không nên bắt đầu bằng model quá nặng.

Model nặng thường có nhiều tham số hơn, thời gian train lâu hơn, tiêu tốn tài nguyên hơn, khó fine-tune hơn và khó đưa vào ứng dụng demo hơn. Với sinh viên mới làm đồ án, model quá nặng còn làm tăng khả năng gặp lỗi do batch size nhỏ, out-of-memory, thời gian thử nghiệm kéo dài, và khó so sánh nhiều phiên bản. Kết quả là em mất nhiều thời gian cho việc “đấu phần cứng” hơn là học được tư duy làm việc với model.

Trong bối cảnh PlantDoc AI, MobileNetV2 và EfficientNet-B0 là hai lựa chọn rất hợp lý. Cả hai đều đã có pretrained ImageNet, đủ mạnh để xử lý bài toán classification ảnh lá cây ở mức đồ án, lại tương đối nhẹ so với nhiều backbone lớn khác. Điều này giúp em train nhanh hơn, thử nghiệm nhiều hơn, và dễ tích hợp vào Streamlit hơn.

Khi chọn model, em cần nhìn trade-off giữa accuracy, tốc độ train, tốc độ inference, kích thước model và độ dễ deploy. Một model có accuracy nhỉnh hơn một chút nhưng chạy chậm gấp nhiều lần chưa chắc là lựa chọn tốt cho demo app. Ngược lại, một model nhẹ hơn, infer nhanh hơn, load nhanh hơn và vẫn đủ chính xác có thể là lựa chọn thực tế hơn rất nhiều.

Ở đây cũng cần phân biệt model “hợp để nghiên cứu” và model “hợp để demo/app”. Model hợp để nghiên cứu là model em dùng để khám phá xem bài toán có thể được cải thiện đến đâu. Còn model hợp để demo/app là model cho trải nghiệm ổn định, chạy nhanh, file nhỏ, ít lỗi thực tế hơn. Trong một đồ án như PlantDoc AI, hai vai trò này có thể trùng nhau, nhưng cũng có thể khác nhau. Em có thể dùng EfficientNet-B0 để đạt kết quả tốt hơn trong thí nghiệm, nhưng lại chọn MobileNetV2 để demo nếu nó nhẹ và ổn định hơn trong Streamlit.

Baseline model nên là model đủ đơn giản để em kiểm soát được toàn bộ pipeline. Điều em cần ở baseline là một điểm xuất phát rõ ràng, không quá phức tạp, dễ lặp lại, dễ so sánh. Một baseline tốt sẽ giúp mọi cải tiến sau này có ý nghĩa, vì em biết mình đang cải thiện từ đâu.

### So sánh ngắn: MobileNetV2 vs EfficientNet-B0 trong PlantDoc AI

| Tiêu chí           | MobileNetV2          | EfficientNet-B0                           |
| ------------------ | -------------------- | ----------------------------------------- |
| Độ nhẹ             | Rất nhẹ              | Nhẹ nhưng thường nặng hơn MobileNetV2     |
| Tốc độ train/infer | Nhanh                | Thường chậm hơn một chút                  |
| Độ dễ deploy       | Rất tốt cho app/demo | Tốt, nhưng file và compute thường cao hơn |
| Hiệu năng kỳ vọng  | Ổn định, thực dụng   | Có thể nhỉnh hơn nếu fine-tune tốt        |
| Phù hợp vai trò    | Baseline, demo       | Improved model, so sánh nâng cấp          |

### Ví dụ/mini case bám sát PlantDoc AI

Nếu em mới bắt đầu phần model của PlantDoc AI, cách hợp lý là chọn MobileNetV2 làm baseline. Lý do không phải vì nó chắc chắn tốt nhất, mà vì em có thể nhanh chóng dựng xong pipeline train/val/test, kiểm tra checkpoint, viết inference và đẩy lên Streamlit. Sau đó em mới dùng EfficientNet-B0 như một phiên bản cải tiến để xem liệu độ chính xác có tăng đủ đáng kể để đáng đổi lấy chi phí cao hơn hay không.

### Lỗi hay gặp

* Chọn model quá nặng ngay từ đầu.
* Chọn model theo trào lưu thay vì theo nhu cầu dự án.
* Không xác định rõ baseline model.
* So sánh model nhưng không xét tốc độ inference và deploy.
* Kết luận model tốt chỉ dựa trên một metric.

### Checklist áp dụng

* Em đã chọn được một baseline model rõ ràng chưa?
* Model đó có phù hợp với tài nguyên và mục tiêu demo không?
* Em có giải thích được vì sao chọn MobileNetV2 hoặc EfficientNet-B0 không?
* Em có tách được tiêu chí “nghiên cứu” và “demo/app” không?

---

## Level 2 — Hiểu cấu trúc model khi dùng transfer learning

### Mục tiêu của level

Sau level này, em phải đọc được mô hình mình đang dùng, biết backbone là gì, head là gì, và hiểu khi thay head thì thực chất mình đang thay đổi phần nào của model.

### Giải thích

Khi dùng transfer learning trong PyTorch, cách phổ biến là lấy một model pretrained trên ImageNet rồi sửa phần cuối để phù hợp với bài toán mới. Muốn làm việc với model có ý thức, em phải nhìn model thành hai phần chính: backbone và classifier head.

Backbone là phần trích xuất đặc trưng. Nó đi qua nhiều tầng convolution để học ra các pattern từ ảnh. Với ảnh lá cây, backbone có thể học được các dấu hiệu như viền lá, vết đốm, vùng úa vàng, texture bất thường, sự thay đổi màu sắc và cấu trúc bề mặt. Backbone không trực tiếp kết luận ảnh thuộc bệnh nào, mà nó tạo ra biểu diễn đặc trưng đủ giàu để phần sau sử dụng.

Classifier head là phần nhận đặc trưng từ backbone và đưa ra dự đoán cuối cùng theo số lớp của bài toán. Với PlantDoc AI, nếu dataset có 38 lớp bệnh thì head phải có đầu ra 38 neuron. Khi em thay head, em đang giữ lại khả năng trích xuất đặc trưng chung từ pretrained model nhưng thay phần ra quyết định cuối để phù hợp với bộ lớp mới.

Điều này rất quan trọng. Nó giải thích vì sao pretrained ImageNet vẫn hữu ích cho bài toán lá cây. Dù ImageNet không được huấn luyện riêng cho bệnh cây, các tầng đầu và tầng giữa của backbone vẫn học được những đặc trưng thị giác rất nền tảng như cạnh, texture, vùng màu, hình dạng cục bộ. Những đặc trưng này vẫn có giá trị khi chuyển sang ảnh lá cây. Em không bắt đầu từ con số 0, mà đang tận dụng một nền tảng thị giác đã được học trước.

Một trực giác dễ hiểu là backbone giống như một người đã học cách quan sát ảnh nói chung. Còn classifier head giống như phần được dạy thêm để phân biệt đúng các lớp bệnh trong dataset của em. Vì thế, nếu em chỉ thay head thì em đang nói với model: “Cách nhìn ảnh chung của anh khá ổn rồi, giờ tôi chỉ cần anh học lại cách phân loại cho đúng tập lớp của dự án tôi.”

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử em dùng `torchvision.models.mobilenet_v2(pretrained=True)`. Phần features của model đóng vai trò backbone. Phần classifier cuối cùng ban đầu được thiết kế cho 1000 lớp ImageNet. Trong PlantDoc AI, em sửa classifier đó thành một tầng Linear có số đầu ra bằng số lớp bệnh của dataset.

Lúc này, nếu dataset của em có các lớp như healthy, early blight, late blight, bacterial spot..., thì classifier head mới chính là phần học cách map các đặc trưng từ backbone thành các nhãn bệnh cụ thể đó.

### Lỗi hay gặp

* Không biết phần nào là backbone, phần nào là head.
* Thay sai số đầu ra của classifier.
* Tưởng pretrained ImageNet là “không liên quan” đến ảnh lá cây.
* Freeze hoặc unfreeze mà không hiểu mình đang tác động lên phần nào.

### Checklist áp dụng

* Em có xác định được backbone trong model của mình không?
* Em có biết classifier head đang có bao nhiêu đầu ra không?
* Số đầu ra có khớp với số lớp trong dataset không?
* Em có hiểu vì sao pretrained ImageNet vẫn dùng được cho PlantDoc AI không?

---

## Level 3 — Cách train model đúng trong PlantDoc AI

### Mục tiêu của level

Sau level này, em phải hiểu một quy trình train đúng, dễ kiểm soát và phù hợp với PlantDoc AI, thay vì train theo kiểu “chạy thật lâu rồi hy vọng kết quả tăng”.

### Giải thích

Một chiến lược rất thực dụng cho transfer learning là chia quá trình huấn luyện thành hai phase. Ở phase 1, em freeze backbone và chỉ train classifier head. Ở phase 2, em unfreeze một phần hoặc toàn bộ backbone để fine-tune với learning rate nhỏ hơn.

Cách làm này có ý nghĩa rất rõ trong PlantDoc AI. Ở phase 1, em tận dụng đặc trưng đã học từ ImageNet và chỉ dạy model cách phân loại các lớp bệnh của dataset mới. Việc này thường ổn định, train nhanh và ít rủi ro hơn. Nó giúp em có baseline sạch, dễ quan sát xem head mới có học được hay không.

Sau khi classifier head đã học tương đối ổn, em bước sang phase 2. Lúc này em cho phép một phần hoặc toàn bộ backbone cập nhật trọng số, nhưng với learning rate nhỏ. Mục tiêu là tinh chỉnh lại các đặc trưng để chúng phù hợp hơn với ảnh lá cây thay vì chỉ dựa trên đặc trưng tổng quát từ ImageNet. Đây là fine-tuning thật sự.

Tư duy baseline trước, improved sau rất quan trọng. Em không nên ngay từ đầu unfreeze toàn bộ model, thêm đủ loại augmentation, scheduler phức tạp và nhiều trick khác cùng lúc. Làm vậy có thể cho kết quả tốt, nhưng em sẽ không biết điều gì thật sự tạo ra cải thiện. Trong dự án học thuật, khả năng giải thích quy trình thường quan trọng không kém kết quả.

Về optimizer, learning rate, scheduler, early stopping và checkpoint best, em nên hiểu vai trò thay vì chỉ “gắn vào cho có”. Optimizer quyết định cách model cập nhật trọng số. Learning rate quyết định bước đi lớn hay nhỏ khi học. Scheduler giúp điều chỉnh learning rate theo thời gian khi mô hình bắt đầu chững lại. Early stopping giúp tránh train quá lâu khi val metric không còn cải thiện. Còn best checkpoint giúp em giữ lại phiên bản tốt nhất theo tiêu chí đã chọn, thay vì mặc định lấy epoch cuối.

Train nhiều epoch không đồng nghĩa với tốt hơn. Nếu train loss tiếp tục giảm nhưng val loss tăng hoặc val metric dừng cải thiện, đó có thể là dấu hiệu model đang overfit. Lúc này, train thêm chỉ làm model học quá sát tập train chứ không giúp tổng quát hóa tốt hơn.

Validation set là cơ chế bảo vệ em khỏi việc bị train accuracy đánh lừa. Train accuracy chỉ nói model học tốt trên dữ liệu đã thấy. Validation mới giúp em ước lượng model có khả năng hoạt động ra sao trên dữ liệu chưa gặp trong quá trình cập nhật trọng số. Trong PlantDoc AI, đây là điều đặc biệt quan trọng vì ảnh lá cây ngoài thực tế rất dễ khác dữ liệu train về ánh sáng, góc chụp và nền.

### Một quy trình train thực dụng cho PlantDoc AI

Trình tự hợp lý thường là như sau. Đầu tiên, em dựng baseline với MobileNetV2 pretrained. Sau đó thay head theo số lớp bệnh. Tiếp theo freeze backbone và train classifier head trong một số epoch để model học nhanh phần phân loại. Sau khi val metric ổn định, em unfreeze một phần hoặc toàn bộ backbone rồi fine-tune với learning rate thấp hơn. Trong toàn bộ quá trình, em log train loss, val loss, train accuracy, val accuracy và nếu cần thì val F1-macro. Em lưu best checkpoint theo val metric phù hợp, thường là `val_f1` nếu dữ liệu có lệch lớp.

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử phase 1 em train MobileNetV2 với backbone freeze trong 5 epoch. Val accuracy tăng khá nhanh từ 70% lên 88%. Sau đó em unfreeze các block cuối và fine-tune thêm 10 epoch với learning rate nhỏ hơn 10 lần. Val accuracy tăng lên 91%, val F1 cũng cải thiện. Ở đây em có một câu chuyện kỹ thuật rất rõ: head học nhanh phần phân loại, sau đó fine-tuning giúp backbone thích nghi hơn với ảnh lá cây.

### Lỗi hay gặp

* Train cả model ngay từ đầu với learning rate lớn.
* Không tách phase 1 và phase 2.
* Chỉ nhìn train accuracy.
* Không lưu best checkpoint.
* Train quá nhiều epoch dù val metric không cải thiện.
* Dùng quá nhiều kỹ thuật cùng lúc nên không biết cái nào hiệu quả.

### Checklist áp dụng

* Em đã có baseline rõ ràng trước khi cải tiến chưa?
* Em có tách phase freeze và fine-tune không?
* Em có theo dõi validation metric trong lúc train không?
* Em có lưu best checkpoint thay vì chỉ epoch cuối không?
* Em có lý do rõ ràng cho learning rate và scheduler đang dùng không?

---

## Level 4 — Làm việc với kết quả của model: đọc gì, hiểu gì, tin gì?

### Mục tiêu của level

Sau level này, em phải biết đọc log huấn luyện một cách tỉnh táo. Em không chỉ nhìn một con số đẹp, mà phải hiểu model đang học như thế nào và có đáng tin hay không.

### Giải thích

Trong PlantDoc AI, bốn tín hiệu cơ bản mà em thường theo dõi là train loss, val loss, train accuracy và val accuracy. Loss phản ánh model dự đoán sai đến mức nào theo hàm mục tiêu. Accuracy phản ánh tỷ lệ dự đoán đúng. Khi theo dõi cùng lúc train và validation, em sẽ thấy được xu hướng học của model thay vì chỉ một ảnh chụp rời rạc.

Nếu train loss giảm và val loss cũng giảm, đó là dấu hiệu tốt. Nếu train accuracy tăng và val accuracy cũng tăng tương ứng, mô hình đang học đúng hướng. Nhưng nếu train loss tiếp tục giảm còn val loss tăng, hoặc train accuracy tăng mạnh còn val accuracy đứng yên, đó thường là dấu hiệu overfitting. Model đang trở nên quá giỏi trên dữ liệu train nhưng không cải thiện khả năng tổng quát hóa.

Trong các bài toán có thể lệch lớp, F1-macro là metric rất đáng chú ý. Accuracy có thể cao vì model làm tốt trên các lớp nhiều mẫu, trong khi bỏ qua các lớp ít mẫu hơn. F1-macro buộc em nhìn đều hơn giữa các lớp. Với PlantDoc AI, nếu một số bệnh có ít ảnh hơn, F1-macro sẽ giúp em tránh ảo giác rằng model đang “rất tốt” dù thật ra nó vẫn yếu ở vài lớp quan trọng.

Confusion matrix là công cụ cực kỳ hữu ích để hiểu model đang nhầm những lớp nào với nhau. Nó đặc biệt có giá trị trong ảnh lá cây, nơi nhiều bệnh có biểu hiện trực quan khá giống nhau. Nếu model thường xuyên nhầm giữa hai lớp có triệu chứng tương đồng, đó là tín hiệu để em xem lại dữ liệu, augmentation, hoặc chiến lược fine-tuning. Confusion matrix biến kết quả từ một con số chung thành một bản đồ lỗi cụ thể.

Điều rất quan trọng là accuracy cao chưa chắc model thật sự tốt. Model có thể đạt accuracy cao trên test set nhưng vẫn thất bại trên ảnh thực tế người dùng tải lên Streamlit, do nền khác, ánh sáng khác, lá chụp quá xa hoặc không rõ triệu chứng. Vì vậy, hiệu năng trong notebook là cần thiết, nhưng chưa đủ. Em phải đọc kết quả với tư duy kiểm chứng, không phải tư duy tự thuyết phục.

### Dấu hiệu underfitting và overfitting

Underfitting thường xảy ra khi cả train và validation đều thấp. Model học chưa đủ, hoặc kiến trúc quá yếu, hoặc train chưa đúng cách. Overfitting thường xuất hiện khi train rất đẹp nhưng validation không tương xứng, thậm chí xấu đi. Trong PlantDoc AI, underfitting có thể xảy ra nếu em freeze quá nhiều và train quá ít. Overfitting có thể xảy ra nếu em fine-tune lâu quá, dữ liệu chưa đủ đa dạng, hoặc augmentation chưa hợp lý.

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử em thấy train accuracy đạt 98% nhưng val accuracy chỉ 87%, val loss bắt đầu tăng từ epoch 9. Nếu chỉ nhìn train accuracy, em sẽ nghĩ model rất mạnh. Nhưng nếu nhìn đầy đủ các tín hiệu, em phải kết luận rằng model đang overfit. Lúc này, best checkpoint có thể nằm ở epoch 7 hoặc 8 chứ không phải epoch cuối.

Một trường hợp khác là accuracy tổng thể 92% nhưng confusion matrix cho thấy model thường nhầm bệnh A sang bệnh B. Điều này cho thấy model tốt ở mức tổng thể nhưng chưa chắc tốt ở mức từng lớp. Nếu bệnh A và B có ý nghĩa quan trọng trong demo hoặc báo cáo, em phải nêu điều này thay vì chỉ khoe accuracy.

### Lỗi hay gặp

* Chỉ nhìn accuracy.
* Không theo dõi val loss.
* Không dùng F1-macro khi dữ liệu lệch lớp.
* Không xem confusion matrix.
* Chọn epoch cuối vì “mới nhất” thay vì epoch tốt nhất.

### Checklist áp dụng

* Em có log cả train và validation không?
* Em có biết model đang underfit hay overfit không?
* Em có xem confusion matrix không?
* Em có cân nhắc F1-macro khi lớp không cân bằng không?
* Em có tránh kết luận chỉ dựa trên accuracy không?

---

## Level 5 — Lưu, tải và quản lý phiên bản model

### Mục tiêu của level

Sau level này, em phải hiểu checkpoint là gì, vì sao quản lý version model là việc nghiêm túc, và cách phân biệt load để inference với load để resume training.

### Giải thích

Checkpoint là trạng thái được lưu lại của model tại một thời điểm trong quá trình huấn luyện. Nó không chỉ là “cân nặng của model”, mà là một mốc kỹ thuật giúp em quay lại đúng phiên bản cần dùng. Trong PlantDoc AI, checkpoint tốt thường là checkpoint có metric validation tốt nhất, chứ không phải checkpoint của epoch cuối cùng.

Vì sao phải lưu best checkpoint thay vì chỉ lưu epoch cuối? Bởi vì huấn luyện không phải lúc nào cũng cải thiện đều đến cuối. Có những giai đoạn model đạt tốt nhất ở giữa quá trình rồi bắt đầu overfit. Nếu em chỉ giữ epoch cuối, em có thể vô tình dùng phiên bản kém hơn cho test hoặc cho app demo.

Một checkpoint tốt trong dự án nên lưu nhiều hơn `state_dict`. Ngoài trọng số model, em nên lưu optimizer state nếu cần resume training, lưu epoch hiện tại, best metric, class mapping, và config quan trọng. Class mapping đặc biệt quan trọng trong PlantDoc AI vì nếu index lớp bị lệch so với tên bệnh, inference sẽ cho ra kết quả rất nguy hiểm: model có thể dự đoán đúng về mặt số học nhưng hiển thị sai tên bệnh.

Load model để inference khác với load model để resume training. Khi inference, em chủ yếu cần cấu trúc model đúng, trọng số đúng, class mapping đúng, transform đúng, rồi chuyển sang `eval()`. Khi resume training, em còn cần optimizer state, epoch trước đó, scheduler state nếu có, và bối cảnh cấu hình để tiếp tục train một cách nhất quán. Nhiều bạn lưu checkpoint quá sơ sài nên sau này không thể resume hoặc không biết checkpoint đó thuộc cấu hình nào.

Đặt tên version model có ý nghĩa là một thói quen rất đáng giá. Tên file nên cho thấy backbone, kiểu train hoặc phase, metric chính, hoặc version thí nghiệm. Ví dụ `best_mobilenetv2_valf1.pt` hay `efficientnetb0_finetune_v2.pt` sẽ dễ quản lý hơn rất nhiều so với `model_new.pt` hay `final_final2.pt`.

Quản lý model kém có thể phá hỏng cả demo lẫn báo cáo. Em có thể train đúng, metric đẹp, nhưng đến lúc demo lại load nhầm file. Hoặc trong báo cáo nói demo dùng EfficientNet-B0, nhưng app thực tế lại đang dùng MobileNetV2 cũ. Đây là lỗi rất thường gặp và làm giảm độ tin cậy của cả đồ án.

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử em có hai file:

* `best_mobilenetv2_valf1.pt`
* `mobilenetv2_last_epoch.pt`

Trong log train, epoch 12 cho `val_f1` tốt nhất, nhưng em train tới epoch 20. Nếu app Streamlit load `last_epoch`, kết quả demo có thể kém hơn bản best. Nếu em không đặt tên rõ và không lưu metadata, rất dễ nhầm giữa hai file này.

### Nên lưu gì cùng model

* `model_state_dict`
* `optimizer_state_dict`
* `epoch`
* `best_metric`
* `class_to_idx` hoặc `idx_to_class`
* `config` quan trọng như model name, input size, normalization, số lớp

### Lỗi hay gặp

* Chỉ lưu model cuối.
* Không lưu class mapping.
* Không biết checkpoint nào đang được demo sử dụng.
* Tên file checkpoint mơ hồ.
* Muốn resume training nhưng checkpoint không có optimizer state.

### Checklist áp dụng

* Em có best checkpoint riêng không?
* Em có lưu class mapping và config không?
* Em có phân biệt checkpoint dùng cho inference và resume không?
* Tên model version của em có rõ nghĩa không?
* Em có biết app đang load file nào không?

---

## Level 6 — Inference: dùng model để dự đoán ảnh mới như thế nào?

### Mục tiêu của level

Sau level này, em phải hiểu inference không chỉ là “đưa ảnh vào model”, mà là một pipeline cần nhất quán chặt với lúc train.

### Giải thích

Trong PlantDoc AI, inference là quá trình dùng model đã train để dự đoán ảnh lá cây mới. Về mặt kỹ thuật, pipeline inference thường gồm các bước: load đúng model, load đúng checkpoint, chuyển model sang chế độ `eval()`, áp dụng preprocessing giống lúc train, đưa ảnh vào model, lấy logits, chuyển sang probabilities nếu cần, chọn lớp dự đoán, rồi map index sang tên bệnh.

Điểm quan trọng nhất ở đây là tính nhất quán. Nếu lúc train em resize ảnh theo một cách, normalize theo ImageNet mean/std, nhưng lúc inference lại bỏ normalize hoặc dùng transform khác, chất lượng dự đoán có thể giảm mạnh. Nhiều bạn thấy model chạy được nên tưởng pipeline ổn, nhưng kết quả sai lệch thực ra đến từ mismatch giữa training pipeline và inference pipeline.

Khi model xuất ra logits, em có thể dùng softmax để chuyển thành xác suất tương đối. Sau đó em lấy top-1 nếu chỉ muốn một dự đoán chính, hoặc top-k nếu muốn hiển thị thêm vài khả năng gần nhất. Trong demo Streamlit, top-k thường hữu ích vì nó giúp người dùng thấy model đang phân vân giữa những lớp nào. Nó cũng giúp em debug trực quan hơn.

Map index sang tên bệnh là bước nhỏ nhưng rất dễ sai. Nếu lúc train em có `class_to_idx`, lúc inference em phải dùng đúng mapping đảo ngược tương ứng. Nếu mapping lệch, model có thể dự đoán index 5 đúng nhưng em lại hiển thị tên bệnh của index 5 trong một mapping cũ khác. Đây là lỗi rất nguy hiểm vì nhìn bề ngoài app vẫn chạy bình thường.

### Vì sao inference pipeline phải nhất quán với training pipeline

Model không học trên ảnh thô theo nghĩa đời thường. Nó học trên ảnh sau khi đã qua resize, tensor hóa, normalize, và đôi khi các xử lý khác. Vì vậy, lúc inference em phải tái tạo đúng “ngôn ngữ đầu vào” mà model đã quen lúc train. Nếu không, em đang yêu cầu model hiểu một kiểu dữ liệu khác với thứ nó từng học.

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử lúc train em dùng ảnh 224x224, normalize theo ImageNet, và class mapping được sinh từ thư mục dataset. Đến lúc viết app Streamlit, em cũng phải resize về 224x224, normalize y hệt, load đúng checkpoint MobileNetV2 có head 38 lớp, rồi dùng đúng `idx_to_class` đã lưu. Nếu em resize khác, quên normalize hoặc dùng mapping cũ của một lần train khác, kết quả demo có thể sai dù model thật ra không tệ.

### Lỗi hay gặp

* Không gọi `model.eval()`.
* Dùng sai transform lúc inference.
* Load nhầm checkpoint.
* Class mapping bị lệch.
* Dự đoán được nhưng hiển thị sai tên bệnh.
* Lấy output trực tiếp mà không hiểu logits là gì.

### Checklist áp dụng

* Em có load đúng model architecture và checkpoint không?
* Em có dùng đúng preprocessing như lúc train không?
* Em có chuyển model sang `eval()` không?
* Em có dùng đúng class mapping không?
* Em có kiểm tra top-1 hoặc top-k trên ảnh thực tế chưa?

---

## Level 7 — So sánh và cải thiện model một cách có hệ thống

### Mục tiêu của level

Sau level này, em phải biết cách cải thiện model bằng tư duy thí nghiệm có kiểm soát, thay vì thay nhiều thứ cùng lúc rồi kết luận theo cảm giác.

### Giải thích

Trong PlantDoc AI, baseline là điểm xuất phát. Improved model là bất kỳ phiên bản nào được thay đổi có chủ đích để kiểm tra xem hiệu năng có tăng hay không. Điều quan trọng không phải là em có bao nhiêu phiên bản model, mà là em có so sánh chúng một cách công bằng hay không.

So sánh công bằng nghĩa là giữ nguyên những điều kiện quan trọng và chỉ thay đổi một vài yếu tố có chủ đích. Nếu em đổi backbone, đổi augmentation, đổi learning rate, đổi scheduler, đổi số epoch cùng lúc, rồi thấy metric tăng, em sẽ không biết yếu tố nào thực sự tạo ra khác biệt. Trong đồ án, điều này làm phần phân tích rất yếu.

Một nguyên tắc rất tốt là mỗi lần thí nghiệm chỉ thay đổi một nhóm nhỏ biến số. Ví dụ, em giữ nguyên toàn bộ pipeline và chỉ đổi backbone từ MobileNetV2 sang EfficientNet-B0. Hoặc giữ nguyên backbone, chỉ thêm phase fine-tuning. Hoặc giữ nguyên mọi thứ, chỉ đổi augmentation. Khi đó em mới kể được câu chuyện kỹ thuật rõ ràng.

Log cấu hình thí nghiệm là việc bắt buộc nếu em muốn làm việc với model chuyên nghiệp hơn. Em nên ghi lại backbone nào, input size bao nhiêu, batch size, learning rate, scheduler, số epoch, phase freeze/unfreeze, metric chọn best checkpoint, và kết quả tương ứng. Không cần hệ thống quá phức tạp, chỉ cần em ghi nhất quán vào file log, YAML config hoặc bảng so sánh là đã hơn rất nhiều so với làm theo trí nhớ.

Cũng cần tránh overclaim. Nếu kết quả tăng nhẹ, em không nên vội viết rằng “mô hình cải thiện rõ rệt” nếu chênh lệch không đáng kể hoặc chưa được kiểm chứng bằng phân tích lỗi. Trong PlantDoc AI, tăng 0.5% accuracy chưa chắc đã có ý nghĩa thực tế nếu đổi lại inference chậm hơn đáng kể hoặc model nặng hơn nhiều.

Ablation đơn giản là cách rất tốt để học tư duy nghiên cứu ứng dụng. Em thay từng thành phần một để xem nó ảnh hưởng thế nào. Trong PlantDoc AI, các ablation hợp lý có thể là đổi backbone, thêm fine-tuning, đổi augmentation, hoặc đổi learning rate schedule. Mỗi thay đổi đều phải có lý do và có phần phân tích kết quả.

### Ví dụ/mini case bám sát PlantDoc AI

Một chuỗi thử nghiệm đẹp có thể là thế này. Thí nghiệm 1: MobileNetV2, chỉ train head. Thí nghiệm 2: MobileNetV2, thêm fine-tuning. Thí nghiệm 3: EfficientNet-B0, cùng chiến lược train như thí nghiệm 2. Thí nghiệm 4: giữ EfficientNet-B0 nhưng thêm augmentation tốt hơn. Khi trình bày theo thứ tự này, em vừa cho thấy quá trình cải tiến có logic, vừa biết thay đổi nào thực sự có ích.

### Lỗi hay gặp

* Thay quá nhiều yếu tố cùng lúc.
* Không lưu config thí nghiệm.
* So sánh model không công bằng.
* Kết luận quá mạnh khi metric chỉ tăng rất nhẹ.
* Không phân tích vì sao model cải thiện hoặc không cải thiện.

### Checklist áp dụng

* Em có baseline rõ ràng không?
* Mỗi lần cải tiến em có biết mình đang đổi gì không?
* Em có log cấu hình từng thí nghiệm không?
* Em có so sánh công bằng giữa các model không?
* Em có tránh overclaim trong kết luận không?

---

## Level 8 — Phân tích lỗi và hiểu giới hạn của model

### Mục tiêu của level

Sau level này, em phải hiểu rằng model sai là nguồn thông tin rất quý. Phân tích lỗi giúp em hiểu giới hạn thực sự của PlantDoc AI và biết nên cải thiện ở đâu.

### Giải thích

Một model trong bài toán lá cây có thể sai vì rất nhiều lý do. Có thể dữ liệu gốc chưa tốt, ảnh bị mờ, góc chụp khó, nền rối, ánh sáng không đều, vùng bệnh quá nhỏ, nhiều lớp có biểu hiện trực quan gần giống nhau, hoặc ảnh thực tế khác xa dữ liệu train. Cũng có thể lỗi nằm ở preprocessing hoặc inference pipeline chứ không phải do năng lực của model.

Điều cần nhớ là model sai không đồng nghĩa model vô dụng. Trong một đồ án tốt, em không cố chứng minh model hoàn hảo. Em chứng minh rằng em hiểu model mạnh ở đâu, yếu ở đâu, và có tư duy phân tích giới hạn một cách trung thực. Chính phần này làm đồ án có chiều sâu.

Một cách thực dụng là phân loại lỗi thành bốn nhóm. Lỗi dữ liệu là khi ảnh gốc có vấn đề hoặc nhãn chưa tốt. Lỗi mô hình là khi model chưa đủ khả năng phân biệt các lớp khó. Lỗi preprocessing là khi pipeline chuẩn hóa, resize, crop hoặc transform làm mất thông tin quan trọng. Lỗi inference là khi checkpoint, mapping hoặc transform trong app bị sai so với lúc train.

Confusion matrix và tập ảnh dự đoán sai là hai nguồn cực kỳ quan trọng. Confusion matrix cho em thấy model hay nhầm những lớp nào với nhau. Ảnh sai dự đoán cho em thấy bối cảnh cụ thể của lỗi. Khi xem ảnh sai, em nên tự hỏi: lá có rõ không, bệnh có rõ không, nền có gây nhiễu không, ánh sáng có bất thường không, và mô hình có thể đã tập trung vào vùng nào.

Nếu dùng Grad-CAM, em chỉ nên xem nó như công cụ hỗ trợ hiểu mô hình đang chú ý vào đâu. Trong PlantDoc AI, Grad-CAM có thể giúp em kiểm tra xem model đang nhìn vào vùng tổn thương trên lá hay bị nền ảnh đánh lạc hướng. Nhưng nó không thay thế cho đánh giá định lượng, cũng không nên biến tài liệu về model thành bài giảng explainability.

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử model thường nhầm giữa hai bệnh có triệu chứng đốm nâu khá giống nhau. Khi xem ảnh sai, em nhận ra nhiều ảnh bị nền phức tạp hoặc vùng bệnh chỉ chiếm một phần nhỏ của lá. Lúc đó, em có thể kết luận hợp lý rằng lỗi không hoàn toàn do model “dở”, mà có thể do dữ liệu thực tế khó, cần crop tốt hơn hoặc augmentation phù hợp hơn.

Một tình huống khác là model trong notebook hoạt động ổn nhưng trên Streamlit lại sai nhiều. Sau khi kiểm tra, em phát hiện app dùng transform thiếu normalization. Đây là lỗi inference pipeline, không phải lỗi bản thân model.

### Lỗi hay gặp

* Gặp lỗi là kết luận model yếu mà không phân tích nguyên nhân.
* Không xem ảnh dự đoán sai.
* Không tách lỗi dữ liệu với lỗi model.
* Dùng Grad-CAM như bằng chứng tuyệt đối.
* Che giấu giới hạn của model trong báo cáo.

### Checklist áp dụng

* Em có phân loại lỗi thành từng nhóm nguyên nhân không?
* Em có xem confusion matrix và ảnh sai dự đoán không?
* Em có kiểm tra pipeline inference khi kết quả demo bất thường không?
* Em có nêu giới hạn model một cách trung thực không?

---

## Level 9 — Đưa model vào ứng dụng PlantDoc AI

### Mục tiêu của level

Sau level này, em phải hiểu rằng model tốt trong notebook chưa chắc đã tốt trong ứng dụng. Em cần biết tư duy chọn model để deploy cho Streamlit.

### Giải thích

Khi đưa model vào Streamlit, bài toán không còn chỉ là đạt metric tốt. Em phải quan tâm model load có nhanh không, file checkpoint có quá lớn không, tốc độ dự đoán có mượt không, và model có ổn định trên ảnh người dùng thật tải lên không. Đây là lúc những tiêu chí “app-friendly” trở nên quan trọng.

Một model rất mạnh nhưng load chậm, infer chậm hoặc dễ lỗi do thiếu tài nguyên có thể làm trải nghiệm demo kém. Với PlantDoc AI, nếu mục tiêu có phần trình diễn trước giảng viên hoặc trước hội đồng, tốc độ phản hồi và độ ổn định rất quan trọng. Đó là lý do MobileNetV2 nhiều khi là lựa chọn demo rất hợp lý, kể cả khi EfficientNet-B0 có thể nhỉnh hơn một ít về metric.

Đưa model vào app cũng buộc em chuẩn hóa lại inference pipeline. Em phải đóng gói rõ architecture, checkpoint, class mapping, transform, logic top-k và cách hiển thị kết quả. Nếu có cảnh báo ảnh không phù hợp, ví dụ ảnh không phải lá hoặc lá quá mờ, em cũng nên nghĩ đến điều này từ góc độ triển khai chứ không chỉ từ góc độ train model.

Một tư duy rất tốt là chọn “model deploy” như một quyết định riêng. Model deploy là model dùng cho app. Nó có thể trùng với model tốt nhất trên benchmark, nhưng cũng có thể là model cân bằng tốt hơn giữa chất lượng và trải nghiệm. Trong đồ án, em nên trình bày rõ quyết định này để người đọc thấy em không chỉ biết train mà còn biết triển khai có trách nhiệm.

### Ví dụ/mini case bám sát PlantDoc AI

Giả sử EfficientNet-B0 đạt val F1 cao hơn MobileNetV2 khoảng 1%, nhưng thời gian load và inference trên máy demo chậm hơn rõ rệt. Nếu Streamlit app có phản hồi chậm, người dùng dễ nghĩ app “bị đơ”. Trong trường hợp đó, dùng MobileNetV2 cho demo có thể là quyết định hợp lý hơn, miễn là em giải thích rõ trong báo cáo rằng đây là lựa chọn cân bằng giữa hiệu năng và khả năng triển khai.

### Lỗi hay gặp

* Chọn model deploy chỉ theo accuracy.
* Không đo thời gian load/inference.
* Không kiểm tra app trên ảnh thực tế ngoài dataset.
* Dùng notebook pipeline khác với app pipeline.
* Không biết model nào đang chạy trong Streamlit.

### Checklist áp dụng

* Em đã chọn model deploy riêng cho app chưa?
* Em có kiểm tra tốc độ load và dự đoán chưa?
* Em có test trên ảnh thực tế chưa?
* Pipeline trong app có khớp với pipeline inference chuẩn không?
* Em có thể giải thích vì sao chọn model đó cho Streamlit không?

---

## Level 10 — Viết phần “làm việc với model” vào báo cáo đồ án

### Mục tiêu của level

Sau level này, em phải biết cách viết phần model trong báo cáo sao cho rõ ràng, đủ kỹ thuật, có logic và thuyết phục.

### Giải thích

Phần “làm việc với model” trong báo cáo không nên chỉ liệt kê tên model và kết quả accuracy. Em cần trình bày model như một quá trình kỹ thuật có quyết định, có thí nghiệm và có lý do.

Trước hết, em nên mô tả model theo cách đủ cụ thể nhưng không sa vào chi tiết không cần thiết. Ví dụ, em có thể viết rằng dự án sử dụng transfer learning với MobileNetV2 hoặc EfficientNet-B0 pretrained trên ImageNet, thay classifier head để phù hợp với số lớp bệnh của dataset PlantVillage. Cách viết này cho thấy em hiểu mô hình ở mức cấu trúc và mục đích sử dụng.

Tiếp theo, em nên trình bày baseline và improved model tách bạch. Baseline là phiên bản khởi đầu để thiết lập mốc hiệu năng. Improved model là phiên bản có thêm fine-tuning, backbone khác, hoặc chiến lược augmentation khác. Khi viết như vậy, người đọc sẽ thấy quá trình làm việc của em có logic phát triển chứ không phải thử ngẫu nhiên.

Về metric, em nên báo cáo không chỉ accuracy mà cả val loss, test accuracy, và nếu dữ liệu không cân bằng thì thêm F1-macro. Nếu có confusion matrix hoặc phân tích lỗi, đó là điểm cộng lớn vì nó cho thấy em không dừng ở một con số tổng quát.

Khi giải thích freeze/unfreeze, em nên viết theo ngôn ngữ kỹ thuật rõ nhưng dễ hiểu. Ví dụ, giai đoạn đầu đóng băng backbone để chỉ huấn luyện classifier head, sau đó mở khóa một phần hoặc toàn bộ backbone để fine-tune với learning rate nhỏ hơn. Cách trình bày này vừa đúng bản chất, vừa thể hiện em biết vì sao lại huấn luyện theo hai phase.

Checkpoint cũng nên được mô tả ngắn gọn nhưng rõ. Em cần nói model nào được chọn làm best checkpoint, dựa trên metric nào, và model nào được dùng cho inference/demo. Đây là điểm nhiều báo cáo bỏ qua, khiến phần mô tả và phần sản phẩm thực tế không khớp nhau.

Ngoài ra, em nên mô tả inference pipeline ở mức đủ dùng: ảnh đầu vào được resize và normalize giống lúc train, model trả về xác suất các lớp, sau đó lấy lớp dự đoán cao nhất hoặc top-k để hiển thị. Nếu dự án có Grad-CAM, chỉ nên nhắc như công cụ hỗ trợ giải thích mô hình trong một số trường hợp.

Cuối cùng, phần giới hạn model là thứ làm báo cáo trưởng thành hơn. Em nên nêu rõ model có thể sai trong các trường hợp ánh sáng kém, nền phức tạp, ảnh mờ, hoặc các lớp bệnh có triệu chứng gần giống nhau. Một báo cáo trung thực về giới hạn thường đáng tin hơn nhiều so với một báo cáo chỉ khoe thành tích.

### Những lỗi viết báo cáo thường gặp

* Mô tả model quá mơ hồ.
* Chỉ khoe accuracy mà không nói bối cảnh.
* Không nêu cấu hình model hoặc chiến lược train.
* Không nói checkpoint nào dùng cho demo.
* Không phân biệt baseline và improved model.
* Không đề cập giới hạn và phân tích lỗi.

### Ví dụ/mini case bám sát PlantDoc AI

Một đoạn viết tốt có thể là: dự án chọn MobileNetV2 pretrained ImageNet làm baseline do mô hình nhẹ, tốc độ huấn luyện và suy luận nhanh, phù hợp cho giai đoạn xây dựng pipeline và demo ứng dụng. Classifier head được thay đổi để phù hợp với số lớp bệnh của bộ dữ liệu. Quá trình huấn luyện gồm hai giai đoạn: huấn luyện head khi đóng băng backbone, sau đó fine-tune một phần backbone với learning rate nhỏ. Checkpoint tốt nhất được chọn theo `val_f1` và được dùng trong pipeline inference của ứng dụng Streamlit.

Đoạn này ngắn nhưng có đủ tư duy kỹ thuật, có lý do chọn model, có chiến lược train, có checkpoint, và có liên hệ tới triển khai.

### Checklist áp dụng

* Em đã mô tả rõ model và lý do chọn model chưa?
* Em có tách baseline và improved model không?
* Em có báo cáo metric phù hợp ngoài accuracy không?
* Em có nói rõ checkpoint nào dùng cho demo không?
* Em có nêu giới hạn model và phân tích lỗi không?

---

## Level 11 — Capstone mini-plan cho PlantDoc AI

### Mục tiêu của level

Sau level này, em phải có một kế hoạch hành động thực tế để áp dụng ngay vào dự án PlantDoc AI.

### Giải thích

Một dự án tốt không chỉ cần hiểu đúng mà còn cần kế hoạch làm việc rõ. Với PlantDoc AI, em có thể triển khai phần model theo một mini roadmap rất thực dụng. Điểm hay của roadmap này là mỗi bước đều gắn với một mục tiêu kỹ thuật cụ thể, nên em không bị rơi vào trạng thái train lung tung rồi mới nghĩ cách giải thích sau.

Bước đầu tiên là chọn baseline model. Trong hầu hết trường hợp, MobileNetV2 là lựa chọn hợp lý để dựng pipeline ban đầu. Em thay head theo số lớp, kiểm tra dataloader, transform, loss function và log training.

Tiếp theo là train phase 1. Em freeze backbone, train classifier head, theo dõi train/val loss và accuracy. Mục tiêu ở giai đoạn này không phải tối đa hóa kết quả ngay, mà là xác nhận pipeline hoạt động đúng và model bắt đầu học được.

Sau đó em chuyển sang fine-tune phase 2. Em unfreeze một phần hoặc toàn bộ backbone, giảm learning rate, tiếp tục train và quan sát validation metric. Đây là lúc em kiểm tra xem fine-tuning có thật sự giúp model thích nghi tốt hơn với ảnh lá cây không.

Trong quá trình đó, em phải chọn best checkpoint theo metric hợp lý, thường là val accuracy hoặc val F1-macro. Checkpoint tốt nhất này sẽ là ứng viên chính cho test và inference.

Khi đã có checkpoint tốt, em đánh giá trên val/test để có kết quả chính thức hơn. Sau đó em không dừng lại ở con số, mà phân tích confusion matrix và các ảnh dự đoán sai để hiểu giới hạn của model.

Sau bước phân tích lỗi, em mới chốt model deploy cho Streamlit. Quyết định này có thể dựa trên cả metric lẫn tốc độ load/inference. Cuối cùng, em viết phần model vào báo cáo theo đúng logic đã làm: chọn model, train, fine-tune, checkpoint, đánh giá, phân tích lỗi, triển khai.

### Mini roadmap hành động

**Bước 1.** Chọn MobileNetV2 làm baseline để dựng pipeline nhanh và ổn định.
**Bước 2.** Thay classifier head theo số lớp bệnh của dataset.
**Bước 3.** Train phase 1 với backbone freeze, log train/val metric đầy đủ.
**Bước 4.** Train phase 2 với fine-tuning và learning rate nhỏ hơn.
**Bước 5.** Lưu `best checkpoint` theo validation metric đã chọn.
**Bước 6.** Đánh giá trên val/test, xem confusion matrix và ảnh dự đoán sai.
**Bước 7.** Thử improved model như EfficientNet-B0 hoặc thay chiến lược augmentation.
**Bước 8.** So sánh công bằng giữa baseline và improved model.
**Bước 9.** Chọn model phù hợp nhất để deploy cho Streamlit.
**Bước 10.** Viết lại câu chuyện kỹ thuật đó thành phần báo cáo rõ ràng, nhất quán.

### Ví dụ/mini case bám sát PlantDoc AI

Một kế hoạch thực tế có thể là tuần đầu em hoàn tất MobileNetV2 baseline và inference local. Tuần tiếp theo em fine-tune và lưu best checkpoint. Sau đó em thử EfficientNet-B0 như improved model. Cuối cùng em chốt model deploy cho Streamlit dựa trên cả val F1 và tốc độ phản hồi của app. Như vậy, mỗi bước đều có sản phẩm đầu ra rõ ràng và dễ commit theo từng task nhỏ.

### Lỗi hay gặp

* Không có roadmap nên làm việc rời rạc.
* Muốn tối ưu quá sớm khi baseline chưa ổn.
* Chốt model deploy trước khi phân tích lỗi.
* Đến cuối mới nhớ viết log và báo cáo.
* Không nhất quán giữa phần train, phần app và phần báo cáo.

### Checklist áp dụng

* Em đã có roadmap rõ cho baseline và improved model chưa?
* Em có biết checkpoint nào là bản chốt không?
* Em đã phân tích lỗi trước khi chốt deploy chưa?
* Em có thể biến toàn bộ quá trình thành câu chuyện kỹ thuật trong báo cáo không?

---

## Kết luận

“Làm việc với model” trong PlantDoc AI không phải là chuyện chọn một kiến trúc rồi train cho ra accuracy cao. Đó là tư duy quản lý toàn bộ vòng đời của model: chọn mô hình, thiết kế baseline, huấn luyện đúng cách, theo dõi validation, lưu best checkpoint, suy luận nhất quán, phân tích lỗi, so sánh phiên bản và triển khai vào ứng dụng.

Trong bối cảnh PlantDoc AI, điều quan trọng không chỉ là train được model, mà là hiểu model đang làm gì, kiểm soát được model ở từng giai đoạn, đánh giá model đúng cách và dùng model đúng phiên bản cho đúng mục đích. Một model tốt cho đồ án không nhất thiết là model nặng nhất hay có con số đẹp nhất, mà là model đủ tốt, đủ ổn định, đủ giải thích được, và đủ thuyết phục khi đi từ notebook đến Streamlit và đến cả báo cáo cuối cùng.

Nếu em giữ được tư duy này, phần model của PlantDoc AI sẽ không còn là một “hộp đen biết đoán ảnh”, mà trở thành một thành phần kỹ thuật em thật sự hiểu, thật sự làm chủ, và có thể bảo vệ tự tin trong đồ án.

Nếu bạn muốn, mình có thể chuyển ngay tài liệu này thành **file `.md` hoàn chỉnh theo format đẹp để bạn copy vào repo PlantDoc AI**.
