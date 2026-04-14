    
# Metrics & Error Analysis cho PlantDoc AI: học từ nền tảng đến thực chiến

## 1. Mở đầu: đánh giá mô hình thật ra là đang trả lời câu hỏi gì?

Khi xây dựng PlantDoc AI, mục tiêu không chỉ là “train được model” hay “ra một con số accuracy đẹp”. Điều quan trọng hơn là phải biết mô hình đang hiểu ảnh lá cây đến đâu, đang mạnh ở đâu, yếu ở đâu, và liệu kết quả đó có đáng tin khi đem vào demo hoặc dùng trên ảnh thực tế hay không.

Trong Machine Learning, **evaluation** là quá trình đánh giá chất lượng của mô hình sau khi huấn luyện. Nói đơn giản, evaluation giúp ta trả lời những câu hỏi như: mô hình dự đoán có đúng không, đúng theo kiểu nào, sai theo kiểu nào, sai tập trung ở nhóm ảnh nào, và liệu những lỗi đó có chấp nhận được với mục tiêu của sản phẩm hay không.

Một **metric** là một thước đo định lượng dùng để tóm tắt hiệu năng của mô hình. Ví dụ quen thuộc nhất là accuracy. Nhưng trong PlantDoc AI, nếu chỉ nhìn accuracy tổng thể thì rất dễ bị đánh lừa. Lý do là bài toán phân loại bệnh lá cây không hề “đều” và “sạch” như trên lý thuyết. Có lớp bệnh rất giống nhau về triệu chứng. Có lớp ít ảnh hơn hẳn các lớp khác. Có ảnh sạch nền trắng như trong PlantVillage, nhưng cũng có ảnh thực địa bị mờ, lệch sáng, nền rối, lá bị che khuất hoặc bệnh còn ở giai đoạn rất sớm. Một con số accuracy gộp lại toàn bộ những tình huống đó có thể che mất nhiều vấn đề nghiêm trọng.

Hãy tưởng tượng mô hình của bạn đạt accuracy 92%. Nghe có vẻ tốt. Nhưng nếu 92% đó chủ yếu đến từ việc dự đoán đúng các lớp phổ biến và ảnh sạch, còn những lớp bệnh hiếm hoặc ảnh ngoài thực địa thì sai rất nhiều, mô hình này vẫn chưa thực sự tốt cho dự án. Vì thế, học Metrics & Error Analysis là học cách **đọc kết quả một cách có tư duy**, thay vì nhìn một con số rồi kết luận vội.

---

## 2. Mức 1: nền tảng nhập môn — vì sao accuracy chưa đủ?

### Accuracy là gì?

**Accuracy** là tỷ lệ dự đoán đúng trên tổng số mẫu. Nếu mô hình dự đoán đúng 900 ảnh trên 1000 ảnh thì accuracy là 90%.

Đây là metric dễ hiểu nhất và rất hữu ích ở giai đoạn đầu. Khi bạn vừa train baseline cho PlantDoc AI, accuracy giúp kiểm tra nhanh xem pipeline có đang chạy đúng hướng hay không. Nếu accuracy quá thấp, có thể pipeline có lỗi ở dataset, transform, mapping class, hoặc training loop. Nếu accuracy tăng dần theo epoch và validation accuracy cũng ổn, đó là tín hiệu tốt ban đầu.

Nhưng accuracy có một giới hạn rất lớn: nó không nói cho bạn biết mô hình đúng ở lớp nào và sai ở lớp nào. Trong một bài toán nhiều lớp như phân loại bệnh lá cây, đây là điểm yếu cực kỳ quan trọng.

### Vì sao accuracy có thể đánh lừa trong PlantDoc AI?

Giả sử bạn có 10 lớp bệnh, trong đó 3 lớp rất nhiều ảnh và 7 lớp còn lại ít ảnh hơn. Nếu mô hình học rất tốt 3 lớp lớn nhưng học kém các lớp nhỏ, accuracy tổng vẫn có thể khá cao. Tức là mô hình “trông có vẻ tốt” nhưng thực ra đang bỏ rơi những lớp khó hơn hoặc ít dữ liệu hơn.

Một ví dụ khác: mô hình hoạt động rất tốt trên ảnh PlantVillage vì nền sạch, lá rõ và điều kiện chụp ổn định. Nhưng khi chuyển sang ảnh gần kiểu PlantDoc, accuracy giảm mạnh. Nếu bạn chỉ nhìn accuracy trên validation set “quá sạch”, bạn sẽ tưởng mô hình đã sẵn sàng, trong khi thực tế nó chưa chịu được dữ liệu ngoài đời.

Một hiểu sai phổ biến của người mới là nghĩ rằng accuracy cao đồng nghĩa mô hình tốt toàn diện. Thực ra accuracy chỉ là một bức ảnh chụp rất tổng quát. Nó có ích, nhưng không đủ để chẩn đoán chất lượng mô hình.

### Cách áp dụng vào PlantDoc AI

Ở giai đoạn baseline, bạn vẫn nên nhìn accuracy đầu tiên vì nó giúp xác nhận mô hình có học được gì không. Nhưng ngay sau đó, bạn phải chuyển sang các metric chi tiết hơn, đặc biệt là precision, recall, F1-score theo từng lớp và confusion matrix. PlantDoc AI là một bài toán mà “sai kiểu gì” quan trọng gần như ngang với “sai bao nhiêu”.

---

## 3. Mức 2: hiểu đúng các metric cốt lõi

## Precision, Recall và F1-score

Ba metric này xuất phát từ cách nhìn vào đúng-sai chi tiết hơn, đặc biệt hữu ích khi muốn phân tích chất lượng dự đoán của từng lớp bệnh.

### Precision là gì?

**Precision** trả lời câu hỏi: trong những ảnh mà mô hình dự đoán là một lớp bệnh nào đó, có bao nhiêu ảnh thực sự đúng là lớp đó?

Trong PlantDoc AI, nếu precision của lớp “Early Blight” thấp, điều đó nghĩa là mô hình đang khá hay gắn nhãn “Early Blight” cho những ảnh không thực sự là Early Blight. Nói cách khác, model có nhiều **false positives** cho lớp này.

Điều này quan trọng khi bạn không muốn mô hình báo nhầm quá nhiều. Ví dụ lá khỏe bị dự đoán thành lá bệnh là một kiểu false positive gây khó chịu trong trải nghiệm người dùng. Nếu người dùng chụp một lá bình thường mà app liên tục cảnh báo bệnh, họ sẽ mất niềm tin vào hệ thống.

### Recall là gì?

**Recall** trả lời câu hỏi: trong tất cả các ảnh thực sự thuộc một lớp bệnh nào đó, mô hình bắt được bao nhiêu ảnh?

Nếu recall của lớp “Late Blight” thấp, điều đó nghĩa là nhiều ảnh Late Blight thật đang bị mô hình bỏ sót, tức có nhiều **false negatives**.

Trong bối cảnh PlantDoc AI, recall thấp ở một bệnh quan trọng có thể đáng lo hơn precision thấp. Vì precision thấp là báo nhầm, còn recall thấp là bỏ sót bệnh thật. Nếu mục tiêu sản phẩm thiên về hỗ trợ phát hiện sớm bệnh, bỏ sót có thể là vấn đề nghiêm trọng hơn.

### F1-score là gì?

**F1-score** là trung bình điều hòa giữa precision và recall. Nó hữu ích khi bạn muốn một metric cân bằng hơn, không thiên quá nhiều về một phía.

Nếu một lớp có precision cao nhưng recall thấp, nghĩa là mô hình chỉ dám dự đoán lớp đó khi rất chắc, nên ít báo nhầm nhưng bỏ sót nhiều. Ngược lại, recall cao nhưng precision thấp nghĩa là mô hình đoán lớp đó khá nhiều, bắt được nhiều mẫu thật nhưng cũng báo nhầm khá nhiều. F1-score giúp tóm lại mức cân bằng giữa hai mặt này.

### Ví dụ rất sát với PlantDoc AI

Giả sử lớp “healthy leaf” có recall cao nhưng precision thấp. Điều đó có thể nghĩa là mô hình thường dự đoán “healthy” khá thoải mái, nên bắt được nhiều lá khỏe thật, nhưng đồng thời cũng lỡ cho cả vài lá bệnh vào nhóm “healthy”. Đây là kiểu lỗi nguy hiểm vì nó tạo cảm giác an toàn giả.

Ngược lại, nếu lớp “disease X” có precision cao nhưng recall thấp, mô hình chỉ gọi “disease X” khi rất chắc chắn. Điều đó làm nó ít báo nhầm, nhưng nhiều ảnh của disease X thật lại bị đẩy sang lớp khác.

### Lỗi hiểu sai phổ biến

Nhiều người đọc classification report thấy một con số F1 đẹp rồi yên tâm, nhưng không nhìn precision và recall riêng lẻ. Điều này dễ làm mất thông tin quan trọng. Hai mô hình có cùng F1 nhưng một mô hình thiên về precision, một mô hình thiên về recall. Trong thực tế sản phẩm, đó là hai hành vi rất khác nhau.

### Cách áp dụng vào PlantDoc AI

Khi đánh giá model, đừng chỉ ghi accuracy. Hãy luôn xem precision, recall, F1 của từng lớp bệnh. Sau đó tự hỏi: lớp nào precision thấp, lớp nào recall thấp, lớp nào cân bằng ổn. Chính những câu hỏi đó mở đầu cho error analysis.

---

## 4. Macro, Micro và Weighted average: vì sao cùng là F1 mà có nhiều kiểu tính?

Trong bài toán nhiều lớp như PlantDoc AI, bạn sẽ thường thấy các dòng như **macro avg**, **micro avg**, **weighted avg** trong classification report. Nếu không hiểu, rất dễ đọc nhầm.

### Macro average

**Macro average** tính metric cho từng lớp trước, rồi lấy trung bình đều giữa các lớp. Mỗi lớp có trọng số như nhau, dù lớp đó có 50 ảnh hay 5000 ảnh.

Điều này rất quan trọng khi dữ liệu bị mất cân bằng. Nếu PlantDoc AI có một số lớp bệnh hiếm, macro F1 giúp bạn nhìn xem mô hình có đối xử tử tế với các lớp nhỏ hay không. Đây thường là metric đáng quan tâm khi bạn muốn đánh giá công bằng hơn trên toàn bộ lớp.

### Micro average

**Micro average** gộp tất cả dự đoán của mọi lớp lại rồi tính. Nó phản ánh hiệu năng tổng thể theo mức từng mẫu, nên thường gần với accuracy trong bài toán single-label multiclass.

Micro hữu ích khi bạn muốn nhìn bức tranh tổng quát, nhưng nó dễ bị lớp lớn chi phối.

### Weighted average

**Weighted average** cũng tính metric từng lớp rồi lấy trung bình, nhưng mỗi lớp được nhân với số lượng mẫu của lớp đó. Vì thế nó nằm ở giữa macro và micro: có xét đến per-class, nhưng lớp đông vẫn ảnh hưởng nhiều hơn.

### Diễn giải trong PlantDoc AI

Nếu accuracy và weighted F1 đều cao nhưng macro F1 thấp, đó là dấu hiệu rất rõ rằng mô hình đang làm tốt trên lớp lớn nhưng làm tệ trên lớp nhỏ. Đây là một tình huống rất hay gặp.

Nếu macro F1 tăng sau khi fine-tuning, dù accuracy không tăng nhiều, đó vẫn có thể là một cải thiện thực chất. Nó cho thấy mô hình đang phân biệt các lớp đồng đều hơn, thay vì chỉ giỏi vài lớp dễ.

### Cách áp dụng

Trong báo cáo PlantDoc AI, nên ưu tiên trình bày cả accuracy lẫn macro F1. Accuracy cho cái nhìn tổng quát. Macro F1 cho biết mô hình có cân bằng giữa các lớp không. Nếu thêm weighted F1, bạn sẽ có một bức tranh tròn hơn.

---

## 5. Support và per-class performance: đừng đọc metric mà quên kích thước dữ liệu

Trong classification report, **support** là số lượng mẫu thật của từng lớp trong tập đánh giá. Đây là thông tin cực kỳ quan trọng khi diễn giải metric.

Một lớp có F1 thấp nhưng support chỉ có 8 ảnh thì cần cẩn thận khi kết luận. Có thể lớp đó thực sự khó. Có thể dữ liệu quá ít. Có thể vài mẫu bị gán nhãn chưa ổn đã làm metric dao động mạnh. Ngược lại, nếu một lớp có support hàng trăm ảnh mà F1 vẫn thấp, đó là tín hiệu đáng quan tâm hơn.

Trong PlantDoc AI, support giúp bạn tránh rơi vào bẫy “thấy một lớp điểm thấp là hoảng”. Ta phải luôn hỏi thêm: lớp đó có bao nhiêu mẫu? Có đại diện đủ các điều kiện chụp không? Có quá ít ảnh thực địa không? Có bị trùng hình hoặc mất đa dạng không?

**Per-class performance** nghĩa là nhìn hiệu năng theo từng lớp thay vì chỉ nhìn trung bình toàn cục. Đây là nơi bạn bắt đầu thật sự hiểu mô hình. Một mô hình dùng được cho PlantDoc AI phải được phân tích theo từng lớp bệnh, không thể chỉ chốt bằng một con số gộp.

---

## 6. Mức 3: biết đọc kết quả mô hình trong dự án thực tế

## Confusion matrix: nơi mô hình “lộ bài”

**Confusion matrix** là ma trận cho thấy ảnh của lớp thật bị dự đoán thành lớp nào. Đây là một trong những công cụ mạnh nhất cho PlantDoc AI, vì nó không chỉ cho biết model sai nhiều hay ít, mà cho biết **sai theo cặp lớp nào**.

Nếu bạn thấy mô hình thường nhầm giữa hai bệnh có triệu chứng hình ảnh giống nhau, confusion matrix sẽ làm điều đó lộ ra ngay. Ví dụ một bệnh tạo đốm nâu nhỏ và một bệnh khác cũng gây tổn thương lá tương tự ở giai đoạn đầu. Accuracy chỉ nói “có sai”. Confusion matrix nói “sai giữa hai lớp này”.

### Cách đọc confusion matrix trong PlantDoc AI

Nếu một hàng của lớp A có nhiều dự đoán chảy sang lớp B, nghĩa là ảnh thật của A thường bị nhầm thành B. Lúc này bạn cần hỏi tiếp: vì A và B thật sự giống nhau về mặt hình ảnh, hay vì dữ liệu của A kém đa dạng, hay vì preprocessing làm mất đặc trưng quan trọng?

Nếu nhiều lớp bệnh bị nhầm thành “healthy”, đây là dấu hiệu rất đáng lo. Có thể mô hình chưa học được đặc trưng bệnh sớm. Có thể augmentation làm mờ tín hiệu bệnh quá mức. Có thể ảnh bệnh trong dataset quá “nặng triệu chứng”, nên model không học tốt giai đoạn nhẹ.

Nếu “healthy” bị nhầm thành nhiều lớp bệnh, mô hình có thể đang quá nhạy với các tín hiệu nền, ánh sáng, hoặc các vết không liên quan đến bệnh.

### Lỗi hiểu sai phổ biến

Nhiều người nhìn confusion matrix rồi chỉ nói chung chung là “model nhầm nhiều giữa vài lớp”. Cách dùng tốt hơn là phải nối ma trận với giả thuyết nguyên nhân. Confusion matrix không chỉ để mô tả, mà để đặt câu hỏi kỹ thuật.

### Cách áp dụng

Sau mỗi lần evaluate, hãy lấy ra 3–5 cặp lớp bị nhầm nhiều nhất. Với mỗi cặp, kiểm tra ảnh thật. Khi xem ảnh, tự hỏi: triệu chứng có giống nhau không, có vấn đề dữ liệu không, ảnh của lớp này có thiên về nền nào đó không, có nhiều ảnh bị mờ hay góc chụp lạ không. Đây chính là cầu nối giữa metric và error analysis.

---

## 7. Top-k accuracy: rất hợp với PlantDoc AI, đặc biệt trong UI

Trong phân loại nhiều lớp, **top-k accuracy** đo xem nhãn đúng có nằm trong top k dự đoán xác suất cao nhất hay không. Với PlantDoc AI, **top-3 accuracy** thường rất đáng quan tâm.

Lý do là một số bệnh lá cây rất giống nhau. Mô hình có thể sai ở top-1 nhưng vẫn đưa đúng bệnh vào top-3. Trong ngữ cảnh hỗ trợ người dùng, điều này vẫn có giá trị. Nếu app hiển thị ba khả năng cao nhất kèm xác suất và hình giải thích, trải nghiệm có thể hữu ích hơn nhiều so với chỉ ép mô hình đưa ra một nhãn duy nhất.

### Ví dụ

Một ảnh thực tế bị dự đoán top-1 là “Septoria leaf spot”, top-2 là “Early blight”, top-3 là “Late blight”, trong khi nhãn thật là “Early blight”. Nếu top-1 accuracy coi đây là sai, điều đó đúng. Nhưng về mặt hỗ trợ chẩn đoán, mô hình vẫn có một mức hiểu nhất định vì bệnh thật nằm trong top-3.

### Khi nào top-k hữu ích?

Top-k rất hữu ích khi:

* số lớp nhiều,
* nhiều lớp có triệu chứng gần nhau,
* app cần hỗ trợ gợi ý hơn là khẳng định tuyệt đối,
* bạn muốn đánh giá chất lượng xếp hạng dự đoán thay vì chỉ đúng-sai cứng.

### Cách áp dụng vào PlantDoc AI

Trong training log, có thể lưu top-3 accuracy. Trong demo, có thể hiển thị top-3 predictions thay vì chỉ top-1. Nếu top-3 tốt nhưng top-1 chưa đủ mạnh, bạn vẫn có thể biến mô hình thành công cụ hỗ trợ khá ổn bằng thiết kế UI đúng cách.

---

## 8. Validation và Test: cùng là metric nhưng ý nghĩa khác nhau

Trong pipeline thực tế, bạn thường có train, validation và test.

**Validation metrics** dùng để theo dõi trong quá trình huấn luyện, so sánh các checkpoint, chọn siêu tham số, chọn epoch tốt nhất, và quyết định dừng sớm. Đây là bộ số liệu phục vụ phát triển mô hình.

**Test metrics** dùng để đánh giá cuối cùng sau khi bạn đã chốt lựa chọn. Đây là bộ số liệu gần hơn với “kết quả báo cáo”.

Một lỗi rất phổ biến là nhìn test quá sớm quá nhiều lần rồi vô tình tối ưu theo test. Khi đó test không còn là đánh giá khách quan nữa. Với PlantDoc AI, bạn nên dùng validation để ra quyết định trong lúc phát triển, và giữ test như bước kiểm tra cuối cùng.

### Chọn checkpoint tốt nhất theo gì?

Nếu dataset khá cân bằng và mục tiêu đơn giản, có thể chọn theo validation accuracy. Nhưng với PlantDoc AI, thường nên cân nhắc chọn theo **validation macro F1** hoặc một metric phù hợp hơn với mục tiêu của dự án. Nếu bạn rất quan tâm việc không bỏ sót vài bệnh quan trọng, có thể theo dõi recall của các lớp đó nữa.

Một checkpoint có accuracy cao hơn rất ít nhưng macro F1 thấp hơn có thể là checkpoint tệ hơn về mặt thực tế.

---

## 9. Overfitting và underfitting nhìn qua metric như thế nào?

### Underfitting

**Underfitting** là khi mô hình chưa học đủ. Dấu hiệu thường là train metrics và validation metrics đều thấp. Mô hình nhìn đâu cũng chưa hiểu rõ.

Trong PlantDoc AI, underfitting có thể xuất hiện khi backbone bị freeze quá nhiều, số epoch quá ít, learning rate chưa phù hợp, head quá đơn giản, hoặc dữ liệu preprocessing làm mất nhiều thông tin.

### Overfitting

**Overfitting** là khi mô hình học quá sát train set nhưng không tổng quát hóa tốt. Dấu hiệu thường là train metrics rất cao, còn validation metrics kém hơn rõ rệt hoặc bắt đầu giảm sau vài epoch.

Trong PlantDoc AI, overfitting dễ xảy ra nếu dataset nhỏ, augmentation chưa đủ, fine-tuning quá mạnh, hoặc có background shortcut khiến model học mẹo từ train set nhưng fail trên ảnh thực tế.

### Cách đọc thực tế

Đừng chỉ nhìn accuracy. Hãy nhìn cả train/val loss, train/val macro F1, và nếu được thì per-class recall trên validation. Có lúc accuracy validation giữ ổn nhưng macro F1 bắt đầu xấu đi, nghĩa là model đang tốt lên trên lớp lớn nhưng xấu đi trên lớp nhỏ.

---

## 10. Mức 4: error analysis là gì và vì sao nó quan trọng hơn việc chỉ xem điểm số?

**Error analysis** là quá trình phân tích có hệ thống các dự đoán sai để hiểu nguyên nhân gốc của lỗi. Nếu metric nói cho bạn biết “mô hình đang tệ ở đâu đó”, thì error analysis giúp trả lời “vì sao lại tệ” và “nên sửa cái gì”.

Điểm khác nhau cốt lõi là thế này: metric tổng cho bạn tín hiệu, còn error analysis cho bạn hướng hành động.

Một mô hình có macro F1 thấp có thể vì nhiều nguyên nhân rất khác nhau: thiếu dữ liệu ở vài lớp, nhãn bẩn, augmentation chưa phù hợp, tiền xử lý sai, domain shift giữa ảnh sạch và ảnh thực địa, hoặc đơn giản là hai lớp quá khó tách bằng hình ảnh hiện tại. Nếu không làm error analysis, bạn rất dễ sửa sai chỗ. Có khi vấn đề là nhãn, nhưng bạn lại đi đổi model. Có khi vấn đề là domain shift, nhưng bạn lại chỉ tăng epoch.

---

## 11. Cách làm error analysis có hệ thống cho PlantDoc AI

Cách tốt nhất là xem error analysis như một quy trình lặp lại sau mỗi lần train, chứ không phải làm một lần cho có.

### Bước đầu: nhìn số liệu để chọn chỗ cần đào sâu

Bắt đầu bằng các metric tổng như accuracy, macro F1, weighted F1, top-3 accuracy. Sau đó đi vào classification report và confusion matrix để xác định các lớp yếu, các cặp lớp hay nhầm, và kiểu lỗi nổi bật.

### Bước tiếp theo: kéo ảnh sai ra xem bằng mắt

Đây là bước cực kỳ quan trọng trong Computer Vision. Bạn nên xem các ảnh false positives và false negatives của từng lớp. Khi xem ảnh, đừng chỉ hỏi “sai hay đúng”. Hãy hỏi:

Mô hình nhầm class nào với class nào?
Ảnh sai có hay xuất hiện ở điều kiện chụp nào không?
Có phải nhiều ảnh sai là ảnh mờ, nhiễu, tối, quá sáng, nền rối, hoặc lá bị che khuất không?
Có phải bệnh còn ở giai đoạn rất sớm nên tín hiệu yếu không?
Có phải một số ảnh có label đáng nghi không?
Có phải model đang chú ý vào nền thay vì vùng bệnh?

### Nhóm lỗi thành cụm có ý nghĩa

Nếu chỉ xem từng ảnh lẻ, bạn sẽ dễ bị ngợp. Hãy nhóm lỗi thành các loại.

Một nhóm có thể là **lỗi theo class**, ví dụ nhầm A sang B rất nhiều. Một nhóm khác là **lỗi theo chất lượng ảnh**, ví dụ ảnh mờ, thiếu sáng, bóng đổ, góc chụp xiên. Một nhóm nữa là **lỗi theo bối cảnh dữ liệu**, ví dụ ảnh ngoài thực địa sai nhiều hơn ảnh sạch. Cũng có thể có nhóm **lỗi nghi do nhãn**, nơi ảnh trông không giống class được gán.

### Từ nhóm lỗi đến giả thuyết nguyên nhân

Nếu lỗi tập trung ở ảnh nền rối, có thể model đang chưa robust với background. Nếu lỗi tập trung ở ảnh thực địa, có thể đang có domain shift. Nếu lỗi tập trung ở vài lớp support thấp, có thể dữ liệu chưa đủ. Nếu nhiều ảnh “sai” nhưng nhìn kỹ label không chắc chắn, có thể đang có annotation ambiguity hoặc label noise.

Error analysis tốt là khi bạn không chỉ đếm lỗi, mà biến lỗi thành giả thuyết có thể kiểm chứng.

---

## 12. Phân tích lỗi theo class

Đây là kiểu phân tích gần như bắt buộc trong PlantDoc AI.

Nếu một lớp có recall thấp, hỏi xem model đang bỏ sót bệnh đó vào lớp nào. Nếu nhiều ảnh của lớp bệnh bị đẩy sang “healthy”, hãy kiểm tra xem những ảnh đó có phải bệnh nhẹ, dấu hiệu mờ nhạt, hay điều kiện chụp quá khó.

Nếu một lớp có precision thấp, hãy xem model đang “lôi” những lớp nào vào nó. Có thể model đang dùng một đặc trưng quá chung, ví dụ chỉ cần thấy vài đốm nâu là kết luận cùng một bệnh, trong khi thực ra nhiều bệnh khác cũng có đốm tương tự.

Khi làm per-class analysis, đừng dừng ở metric. Phải mở ảnh ra xem. Computer Vision mà không xem ảnh sai thì phân tích sẽ rất hời hợt.

---

## 13. Phân tích false positives và false negatives

### False positives

False positive là khi mô hình dự đoán một lớp bệnh nhưng thực tế ảnh không thuộc lớp đó.

Trong PlantDoc AI, false positives gây ra báo động nhầm. Ví dụ lá khỏe bị gọi là bệnh. Hoặc bệnh A bị gọi nhầm thành bệnh B. Khi precision thấp, false positives thường nhiều.

Hãy kiểm tra xem false positives có liên quan đến nền, ánh sáng, hoặc những pattern không thực sự là triệu chứng bệnh không. Đôi khi model học shortcut từ màu nền, bóng đổ, hoặc texture của nền đất phía sau.

### False negatives

False negative là khi ảnh thực sự thuộc lớp bệnh nhưng mô hình không nhận ra.

Đây là kiểu lỗi thường gắn với recall thấp. Trong PlantDoc AI, false negatives đặc biệt đáng chú ý nếu hệ thống mang ý nghĩa hỗ trợ phát hiện bệnh.

Ảnh false negative thường đáng xem kỹ: bệnh có quá nhẹ không, vùng tổn thương có quá nhỏ không, ảnh có bị mất nét không, hay label có quá mơ hồ không? Nhiều khi false negative nói với bạn rằng dữ liệu huấn luyện chưa có đủ ví dụ “ca khó”.

---

## 14. Phân tích lỗi theo nhóm điều kiện ảnh

Một mô hình ảnh không chỉ sai theo class, mà còn sai theo **điều kiện ảnh**. Đây là điểm rất thực chiến.

Bạn có thể chia ảnh thành các nhóm như: ảnh sạch nền đơn giản, ảnh ngoài thực địa, ảnh thiếu sáng, ảnh quá sáng, ảnh mờ, ảnh chụp gần, ảnh chụp xa, ảnh chỉ có một lá, ảnh nhiều lá chồng lên nhau. Sau đó xem metric của từng nhóm.

Nếu model tốt trên ảnh sạch nhưng kém trên ảnh thực địa, đó là dấu hiệu domain shift rõ. Nếu model kém trên ảnh mờ, có thể cần tăng robustness qua augmentation phù hợp hoặc bổ sung dữ liệu khó. Nếu model kém khi nền rối, có thể cần crop lá tốt hơn, segmentation, hoặc dữ liệu đa dạng nền hơn.

Đây là kiểu phân tích rất có giá trị khi bạn muốn PlantDoc AI không chỉ đẹp trong notebook mà còn usable trong demo thật.

---

## 15. Lỗi do dữ liệu, do label, do preprocessing, do augmentation, do model hay do domain shift?

Một kỹ năng quan trọng là phân biệt lỗi đến từ đâu.

### Lỗi do dữ liệu

Có thể do thiếu dữ liệu ở một số lớp, dữ liệu không đa dạng, ảnh quá sạch so với thực tế, hoặc phân bố train/val/test không phản ánh tình huống sử dụng thật.

### Lỗi do label

Một số ảnh có thể bị gán nhãn sai, hoặc bản thân triệu chứng ở ảnh rất mơ hồ. Trong bệnh lá cây, có những trường hợp ranh giới giữa hai lớp không sắc nét, đặc biệt ở giai đoạn sớm hoặc ảnh chất lượng thấp.

### Lỗi do preprocessing

Resize, crop, normalize, color conversion, hoặc đọc ảnh sai kênh màu đều có thể làm mất tín hiệu quan trọng. Ví dụ crop không khéo có thể cắt mất vùng bệnh.

### Lỗi do augmentation

Augmentation là con dao hai lưỡi. Nếu quá mạnh hoặc không phù hợp, bạn có thể vô tình phá hỏng dấu hiệu bệnh. Ví dụ color jitter quá mạnh có thể làm biến đổi màu triệu chứng vốn là tín hiệu quan trọng.

### Lỗi do model capacity

Model quá nhỏ có thể chưa đủ sức phân biệt các lớp khó. Nhưng đổi model không phải lúc nào cũng là câu trả lời đầu tiên.

### Lỗi do domain shift

Đây là vấn đề rất hay gặp giữa PlantVillage và ảnh thực tế kiểu PlantDoc. Model học tốt trên ảnh sạch nhưng fail khi dữ liệu ngoài đời khác hẳn về nền, ánh sáng, góc chụp, độ nhiễu.

Một người làm project tốt phải tránh phản xạ “thấy lỗi là đổi model”. Rất nhiều lỗi thật ra nằm ở dữ liệu và bài toán, không nằm ở backbone.

---

## 16. Phân biệt “model weakness” với “dataset/problem definition weakness”

Đây là góc nhìn trưởng thành hơn trong ML.

**Model weakness** là khi dữ liệu và định nghĩa bài toán tương đối ổn, nhưng mô hình chưa khai thác tốt. Ví dụ cùng dữ liệu đó, một mô hình khác hoặc cách fine-tune khác có thể cải thiện rõ.

**Dataset/problem definition weakness** là khi bản thân dữ liệu hoặc cách đặt bài toán đã có vấn đề. Ví dụ nhãn quá mơ hồ, lớp chồng lấn mạnh, train set quá sạch so với deployment, hoặc số lượng dữ liệu ở lớp quan trọng quá ít.

Nếu lỗi đến từ dataset/problem definition mà bạn cứ tiếp tục tinh chỉnh model, bạn sẽ bị mắc kẹt trong vòng lặp tối ưu sai chỗ.

---

## 17. Mức 5: từ phát hiện lỗi đến quyết định cải thiện

Đây là phần nhiều người học thiếu nhất. Biết metric là một chuyện. Biết **ra quyết định từ metric** mới là chuyện quan trọng.

### Khi recall thấp ở một bệnh quan trọng

Điều này gợi ý mô hình đang bỏ sót nhiều ca thật. Bạn nên nghĩ tới việc bổ sung dữ liệu cho lớp đó, đặc biệt là các ca khó và đa dạng điều kiện chụp. Ngoài ra, hãy xem model đang nhầm bệnh đó sang lớp nào để biết có cần cải thiện khả năng phân biệt cặp lớp hay không. Nếu hệ thống ưu tiên phát hiện bệnh hơn là tránh báo nhầm, recall thấp là tín hiệu cần ưu tiên xử lý.

### Khi precision thấp

Mô hình đang báo nhầm khá nhiều cho lớp đó. Bạn nên kiểm tra false positives xem có chung pattern gì không. Có thể model đang bị background shortcut, hoặc đang dựa vào đặc trưng quá chung. Lúc này có thể cần dữ liệu âm tính tốt hơn, cải thiện augmentations, hoặc thiết kế loss/training để mô hình phân biệt tinh hơn.

### Khi confusion tập trung giữa vài class cụ thể

Đây thường là một gợi ý rất mạnh. Có thể hai lớp thật sự giống nhau về triệu chứng, nên cần thêm dữ liệu phân biệt. Có thể cần phân tích bằng Grad-CAM để xem model đang nhìn vào đâu. Cũng có thể nhãn giữa hai lớp này bản thân đã khó tách. Nếu vậy, bạn nên cân nhắc lại định nghĩa bài toán hoặc cách trình bày kết quả trong app.

### Khi top-k tốt nhưng top-1 chưa tốt

Đây không hẳn là thất bại. Nếu top-3 tốt, bạn có thể tận dụng trong UI bằng cách hiển thị ba khả năng cao nhất, kèm xác suất và giải thích. Với người dùng cuối, điều đó thường hữu ích hơn một dự đoán top-1 đầy tự tin nhưng sai.

### Khi accuracy tăng nhưng macro F1 giảm

Đây là tình huống rất đáng chú ý. Nó gợi ý mô hình đang cải thiện trên các lớp lớn hoặc dễ, nhưng lại tệ đi ở lớp nhỏ hoặc khó. Nếu không để ý macro F1, bạn rất dễ hiểu nhầm rằng mô hình đang tốt lên toàn diện.

---

## 18. Confidence, calibration và chuyện “tự tin nhưng sai”

Nhiều mô hình deep learning có thể dự đoán với xác suất rất cao nhưng vẫn sai. Đây là hiện tượng “**tự tin nhưng sai**”. Trong demo PlantDoc AI, điều này rất nguy hiểm vì người dùng thường tin vào con số xác suất.

### Confidence analysis

Hãy xem những mẫu sai nhưng có confidence cao. Nếu nhiều ảnh sai mà model vẫn rất tự tin, đó là dấu hiệu mô hình chưa được calibration tốt hoặc đang học shortcut.

### Calibration là gì?

**Calibration** nói về việc xác suất dự đoán có phản ánh đúng mức độ tin cậy hay không. Nếu model nói “90% chắc chắn”, ta kỳ vọng khoảng 90% những dự đoán ở mức đó là đúng. Trong thực tế, nhiều model không calibrated tốt.

### Áp dụng vào PlantDoc AI

Nếu app hiển thị confidence score, bạn nên cẩn thận. Đừng xem nó như “xác suất thật” tuyệt đối. Trong báo cáo hoặc demo, có thể nói rõ đây là mức tin cậy của mô hình, không phải xác nhận y học. Nếu có thời gian, bạn có thể làm phân tích calibration cơ bản hoặc ít nhất kiểm tra các trường hợp high-confidence wrong.

---

## 19. Threshold thinking: không phải lúc nào cũng chỉ lấy top-1

Trong bài toán multiclass single-label, thường ta lấy lớp có xác suất cao nhất làm dự đoán. Nhưng về tư duy sản phẩm, bạn vẫn có thể nghĩ theo hướng threshold.

Ví dụ, nếu xác suất cao nhất quá thấp hoặc chênh lệch giữa top-1 và top-2 quá nhỏ, app có thể hiển thị thông báo như “kết quả chưa đủ chắc chắn” thay vì khẳng định mạnh. Hoặc nếu top-3 có xác suất khá sát nhau, UI có thể khuyến khích người dùng chụp ảnh rõ hơn.

Trong PlantDoc AI, threshold thinking không nhất thiết là đổi thuật toán phân loại, mà là cách dùng confidence để thiết kế trải nghiệm an toàn và trung thực hơn.

---

## 20. Distribution shift và domain shift: vấn đề sống còn giữa PlantVillage và ảnh thực tế

Một trong những rủi ro lớn nhất của PlantDoc AI là model học rất tốt trên dữ liệu sạch nhưng yếu trên dữ liệu ngoài đời. Đây là **distribution shift** hoặc cụ thể hơn là **domain shift**.

PlantVillage thường có ảnh tương đối sạch, nền gọn, lá rõ. Ảnh thực tế có thể khác hẳn. Nếu bạn chỉ báo cáo metric trên dữ liệu “đẹp”, bạn dễ tự tin quá sớm.

### Dấu hiệu

Validation trên dữ liệu sạch rất tốt, nhưng khi thử vài ảnh tự chụp hoặc dữ liệu thực địa thì kết quả tệ hơn hẳn. Error analysis cho thấy model chú ý vào nền hoặc vùng không liên quan. Confusion tăng mạnh ở ảnh tối, mờ, nhiều lá, hoặc góc chụp lạ.

### Hành động

Lúc này, cải thiện dữ liệu thường quan trọng hơn cải thiện model. Bổ sung dữ liệu thực tế, augment hợp lý, hoặc chia evaluation theo domain là những bước rất đáng làm.

---

## 21. Grad-CAM hỗ trợ error analysis như thế nào?

Grad-CAM không phải metric, nhưng là công cụ giải thích rất hữu ích trong PlantDoc AI.

Nếu model dự đoán đúng, Grad-CAM giúp bạn xem nó có thật sự nhìn vào vùng bệnh hay chỉ trùng hợp đúng nhờ nền. Nếu model sai, Grad-CAM càng hữu ích hơn: nó cho thấy model đang bị thu hút bởi đâu.

Ví dụ, nếu mô hình nhầm vì tập trung vào nền đất, bóng đổ, hoặc mép ảnh thay vì vùng lá bị tổn thương, bạn có cơ sở để nghi ngờ shortcut learning. Nếu model tập trung đúng vùng bệnh nhưng vẫn nhầm giữa hai bệnh tương tự, có thể vấn đề nằm ở độ khó phân biệt của lớp chứ không hẳn do model nhìn sai chỗ.

Grad-CAM vì thế là cầu nối rất tốt giữa error analysis định lượng và phân tích trực quan.

---

## 22. Mức 6: góc nhìn nâng cao và thực chiến

## Khi nào nên cải thiện dữ liệu thay vì cố đổi model?

Đây là câu hỏi cực kỳ thực chiến. Nếu error analysis cho thấy:

* lỗi tập trung ở lớp ít dữ liệu,
* lỗi tập trung ở ảnh thực địa,
* có nhiều label đáng nghi,
* model đang học shortcut từ nền,
* hoặc khác biệt train-deployment quá lớn,

thì đổi backbone từ MobileNetV2 sang EfficientNet-B0 có thể chưa phải cú đấm mạnh nhất. Nhiều khi bổ sung dữ liệu đúng kiểu, làm sạch nhãn, hoặc cải thiện cách split dữ liệu sẽ hiệu quả hơn.

Ngược lại, nếu dữ liệu đã khá ổn nhưng model vẫn underfit hoặc không tách nổi vài lớp khó, lúc đó mới đáng cân nhắc tăng capacity, fine-tune sâu hơn, đổi loss, hoặc thêm kỹ thuật mạnh hơn.

## Label noise và annotation ambiguity

Trong bệnh lá cây, không phải ảnh nào cũng “sạch nhãn” như bài toán textbook. Có thể có ảnh gán nhãn sai. Cũng có thể có ảnh mà ngay cả con người cũng khó phân biệt rõ. Nếu bạn thấy nhiều mẫu “sai” nhưng nhìn ảnh lại thấy hợp lý, có thể vấn đề nằm ở nhãn hoặc định nghĩa lớp.

Một project trưởng thành phải dám thừa nhận có những lỗi không phải do model ngu, mà do bài toán vốn mơ hồ.

## Fairness theo nhóm dữ liệu

Fairness trong PlantDoc AI không phải fairness theo nhân khẩu học như nhiều bài toán khác, nhưng vẫn có thể hiểu theo nhóm dữ liệu: mô hình có công bằng giữa ảnh sạch và ảnh thực địa không, giữa ánh sáng tốt và kém không, giữa góc chụp chuẩn và lệch không. Nếu model chỉ tốt ở điều kiện thuận lợi, chất lượng thật của hệ thống vẫn còn hạn chế.

## Thay đổi objective/metric theo mục tiêu sản phẩm

Nếu mục tiêu của PlantDoc AI là demo học thuật, accuracy và macro F1 có thể là trọng tâm hợp lý. Nếu mục tiêu là hỗ trợ người dùng phát hiện bệnh sớm, recall của các lớp bệnh quan trọng có thể cần được ưu tiên hơn. Nếu sản phẩm thiên về gợi ý hỗ trợ, top-3 accuracy và UI hiển thị đa phương án có thể rất giá trị.

Metric tốt không phải metric “nghe xịn”, mà là metric phản ánh đúng mục tiêu sử dụng.

---

## 23. So sánh tình huống: cùng số đẹp nhưng chất lượng thực tế khác nhau

### Tình huống 1: hai model có accuracy gần nhau nhưng chất lượng khác nhau

Model A đạt accuracy 91%, model B đạt 90.5%. Nếu nhìn thoáng, A có vẻ tốt hơn. Nhưng nếu model B có macro F1 cao hơn rõ và confusion matrix cân bằng hơn giữa các lớp hiếm, B có thể là lựa chọn tốt hơn cho PlantDoc AI.

### Tình huống 2: accuracy tăng nhưng macro F1 giảm

Điều này thường có nghĩa mô hình đang tối ưu thêm cho lớp lớn nhưng làm xấu đi lớp nhỏ. Nếu project cần chất lượng đồng đều giữa các lớp bệnh, đây là một bước lùi chứ không phải tiến.

### Tình huống 3: confusion matrix lộ ra vấn đề mà accuracy che mất

Accuracy có thể vẫn đẹp, nhưng confusion matrix cho thấy hai bệnh quan trọng bị nhầm qua lại rất nhiều. Nếu đó là hai lớp mà người dùng thực sự cần phân biệt, vấn đề này quan trọng hơn con số accuracy tổng.

### Tình huống 4: background shortcut hoặc lighting bias

Model đạt điểm cao trên validation nhưng sai mạnh ở ảnh nền rối hoặc thiếu sáng. Grad-CAM cho thấy nó chú ý vào nền nhiều hơn vùng bệnh. Đây là ví dụ điển hình của việc metric tổng đẹp nhưng khả năng tổng quát hóa yếu.

### Tình huống 5: top-1 sai nhưng top-3 đúng

Về metric top-1, đây là một lỗi. Nhưng về thiết kế sản phẩm, đây vẫn có thể là kết quả hữu ích. Nếu UI hiển thị top-3 và khuyến khích người dùng chụp lại ảnh rõ hơn, hệ thống vẫn mang lại giá trị.

---

## 24. Kết nối trực tiếp với pipeline PlantDoc AI

### Khi train baseline nên nhìn gì trước?

Ban đầu, hãy nhìn train/validation loss và accuracy để kiểm tra pipeline có hoạt động đúng không. Sau đó chuyển rất nhanh sang macro F1, classification report và confusion matrix để tránh bị accuracy đánh lừa.

### Khi fine-tune nên so sánh gì?

Khi fine-tune, đừng chỉ so accuracy. Hãy so cả macro F1, per-class recall, top-3 accuracy, và xem confusion matrix có bớt nhầm ở các cặp lớp khó không. Fine-tuning tốt là fine-tuning tạo ra cải thiện thực chất, không chỉ tăng nhẹ một con số tổng.

### Khi chọn checkpoint tốt nhất nên dựa trên gì?

Nên chọn theo metric phản ánh đúng mục tiêu. Với PlantDoc AI, validation macro F1 thường là một lựa chọn hợp lý hơn accuracy nếu dữ liệu mất cân bằng hoặc muốn hiệu năng đồng đều giữa lớp. Nếu app cần gợi ý top-3, bạn cũng có thể theo dõi top-3 accuracy song song.

### Khi viết báo cáo đồ án nên trình bày ra sao?

Đừng chỉ đưa một bảng accuracy. Nên có bảng gồm accuracy, macro F1, weighted F1, top-3 accuracy. Nên có confusion matrix và một đoạn phân tích 5–10 dòng nêu rõ model mạnh ở đâu, yếu ở đâu, cặp lớp nào hay nhầm, và nguyên nhân khả dĩ. Nếu có Grad-CAM, nên thêm vài ví dụ đúng/sai tiêu biểu để làm phần phân tích lỗi giàu sức thuyết phục hơn.

### Khi làm demo/app nên hiển thị gì?

Nên hiển thị top-3 predictions, confidence tương đối, và nếu có thể thì kèm Grad-CAM hoặc một vùng giải thích trực quan. Nếu confidence thấp hoặc dự đoán không chắc, UI nên thể hiện sự bất định thay vì tỏ ra chắc chắn thái quá.

---

## 25. Những sai lầm phổ biến khi làm Metrics & Error Analysis cho PlantDoc AI

Sai lầm phổ biến nhất là chỉ nhìn accuracy. Ngay sau đó là nhìn classification report nhưng không xem ảnh sai. Một sai lầm khác là thấy model yếu rồi mặc định lỗi do kiến trúc mạng, trong khi vấn đề thật nằm ở dữ liệu hoặc domain shift.

Nhiều người cũng chọn checkpoint theo validation accuracy vì tiện, trong khi mục tiêu dự án lại hợp hơn với macro F1. Một lỗi khác nữa là dùng confidence như chân lý, dù model có thể tự tin nhưng sai. Và cuối cùng, rất nhiều project làm confusion matrix chỉ để “cho có”, không biến nó thành hành động cải thiện cụ thể.

---

## 26. Tổng kết thực hành: quy trình đọc metric và làm error analysis cho PlantDoc AI

Sau mỗi lần train/evaluate, bạn có thể đi theo thứ tự sau:

1. Xem train loss, validation loss, accuracy để kiểm tra bức tranh tổng quát.
2. Xem macro F1, weighted F1, top-3 accuracy để hiểu chất lượng sâu hơn.
3. Đọc classification report theo từng lớp, đặc biệt để ý precision, recall, F1 và support.
4. Xem confusion matrix để tìm các cặp lớp nhầm nhiều nhất.
5. Kéo các mẫu dự đoán sai ra xem ảnh thật, chia theo false positives và false negatives.
6. Nhóm lỗi theo class, theo điều kiện ảnh, theo domain, theo mức độ bệnh, theo nghi ngờ label.
7. Đặt giả thuyết nguyên nhân: dữ liệu, nhãn, preprocessing, augmentation, model capacity hay domain shift.
8. Đề xuất hành động cụ thể cho vòng train tiếp theo: thêm dữ liệu, sửa split, tinh chỉnh augmentation, đổi checkpoint criterion, cải thiện UI, hoặc xem lại định nghĩa bài toán.
9. Sau lần train sau, so sánh lại đúng các điểm yếu cũ để xem cải thiện có thật không.

### Checklist ngắn cho lần evaluate tiếp theo của PlantDoc AI

* Có xem cả accuracy và macro F1 chưa?
* Có đọc per-class metrics chưa?
* Có nhìn support trước khi kết luận không?
* Có xem confusion matrix chưa?
* Có mở ảnh sai ra xem thật không?
* Có nhóm lỗi thành các cụm có ý nghĩa không?
* Có phân biệt lỗi do model với lỗi do dữ liệu không?
* Có chuyển phân tích thành hành động cụ thể cho vòng sau không?
* Có nghĩ đến top-3 và confidence cho UI/demo không?
* Có kiểm tra model trên ảnh gần điều kiện thực tế chưa?

 