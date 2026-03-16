# PyTorch cốt lõi cho PlantDoc AI

## Mục lục

- [0) Mở đầu: 3 khối PyTorch quan trọng nhất trong PlantDoc AI](#0-mở-đầu-3-khối-pytorch-quan-trọng-nhất-trong-plantdoc-ai)
- [1) Dataset, DataLoader và Transforms trong PlantDoc AI](#1-dataset-dataloader-và-transforms-trong-plantdoc-ai)
- [2) Training loop và Validation loop trong PlantDoc AI](#2-training-loop-và-validation-loop-trong-plantdoc-ai)
- [3) Checkpoint, load model và inference trong PlantDoc AI](#3-checkpoint-load-model-và-inference-trong-plantdoc-ai)
- [4) Kết nối 3 khối thành một workflow hoàn chỉnh cho PlantDoc AI](#4-kết-nối-3-khối-thành-một-workflow-hoàn-chỉnh-cho-plantdoc-ai)
- [5) Lỗi thường gặp nhất khi dùng PyTorch trong PlantDoc AI](#5-lỗi-thường-gặp-nhất-khi-dùng-pytorch-trong-plantdoc-ai)
- [6) Mini roadmap áp dụng ngay vào PlantDoc AI](#6-mini-roadmap-áp-dụng-ngay-vào-plantdoc-ai)
- [7) Kết luận](#7-kết-luận)

---

## 0) Mở đầu: 3 khối PyTorch quan trọng nhất trong PlantDoc AI

### Mục tiêu

Phần mở đầu này giúp người đọc nhìn ra bức tranh lớn của pipeline PyTorch trong PlantDoc AI trước khi đi sâu vào từng đoạn code. Mục tiêu không phải là nhớ thật nhiều API, mà là hiểu model đang sống ở đâu trong toàn bộ luồng xử lý dữ liệu.

### Giải thích

Trong PlantDoc AI, PyTorch không chỉ là nơi bạn khai báo model rồi bấm train. Thực tế, model chỉ là một mắt xích nằm ở giữa pipeline. Trước khi model học được, ảnh lá cây phải được đọc đúng cách, biến đổi đúng cách và đóng gói thành batch đúng cách. Sau khi model học xong, bạn còn phải lưu lại phiên bản tốt nhất, load lại đúng cấu hình và dùng nó để dự đoán trong ứng dụng demo.

Vì vậy, khi làm một dự án image classification bằng PyTorch, có 3 khối quan trọng nhất mà bạn phải nắm thật chắc. Khối thứ nhất là đưa dữ liệu vào đúng cách, tức Dataset, DataLoader và Transforms. Khối thứ hai là train và evaluate đúng cách, tức training loop và validation loop. Khối thứ ba là lưu và dùng lại model đúng cách, tức checkpoint, load model và inference. Nếu hỏng một trong ba khối này, toàn bộ dự án sẽ dễ bị lỗi hoặc cho kết quả không đáng tin.

Cách nhìn đúng là: ảnh lá cây đi từ ổ đĩa vào Dataset, qua Transforms để thành tensor, được DataLoader gom thành batch, đưa vào model trong training loop để học, sau đó validation loop dùng để chọn ra checkpoint tốt nhất, rồi checkpoint đó được load lại để chạy inference trong Streamlit. Khi bạn hiểu được luồng này, code PyTorch sẽ bớt cảm giác rời rạc và bắt đầu có logic rất rõ ràng.

### Ví dụ bám sát PlantDoc AI

Hãy hình dung repo PlantDoc AI có thư mục dữ liệu được chia thành `train/`, `val/`, `test/`, trong mỗi thư mục lại có các thư mục con như `Tomato_Early_blight`, `Tomato_healthy`, `Potato_late_blight`... Ảnh được đọc từ đó, resize về 224x224, normalize theo ImageNet, rồi đóng gói thành các batch để model MobileNetV2 hoặc EfficientNet-B0 học. Sau mỗi epoch, bạn chạy validation để xem model nào đang tốt nhất. Cuối cùng, bản tốt nhất được lưu lại thành checkpoint và dùng trong app dự đoán bệnh từ ảnh người dùng upload.

### Lỗi hay gặp

- Chỉ tập trung vào model mà bỏ qua pipeline dữ liệu.
- Train xong nhưng không biết lưu gì để demo lại.
- Dùng được trong notebook nhưng tách sang app thì inference sai vì pipeline không khớp.

### Checklist áp dụng rất ngắn

- Hiểu ảnh đi từ đâu đến đâu trong pipeline.
- Xác định rõ 3 khối: dữ liệu, train/eval, checkpoint/inference.
- Luôn nghĩ theo luồng đầy đủ, không nghĩ từng file rời rạc.

---

## 1) Dataset, DataLoader và Transforms trong PlantDoc AI

### Mục tiêu

Phần này giúp bạn hiểu rõ ảnh lá cây trên đĩa đi vào PyTorch như thế nào để trở thành batch tensor cho model. Sau khi học xong phần này, bạn phải đọc được pipeline dữ liệu của repo và tự viết được phiên bản cơ bản của nó.

### Giải thích

Trong PyTorch, `Dataset` là lớp đại diện cho tập dữ liệu. Nói đơn giản, nó trả lời hai câu hỏi: tập dữ liệu có bao nhiêu mẫu, và khi lấy một mẫu thì nhận được cái gì. Với bài toán phân loại bệnh lá cây, mỗi mẫu thường gồm một ảnh và một nhãn lớp. Ảnh ban đầu nằm trên đĩa ở dạng file `.jpg` hoặc `.png`, nhưng model không làm việc trực tiếp với file ảnh. Model cần tensor số học. Nhiệm vụ của `Dataset` là nối khoảng cách đó.

`DataLoader` là lớp giúp lấy dữ liệu từ `Dataset` theo từng batch. Nó lo việc trộn dữ liệu, chia batch, và có thể đọc dữ liệu song song bằng nhiều worker. Nếu không có `DataLoader`, bạn sẽ phải tự viết vòng lặp đọc từng ảnh, ghép từng tensor, xử lý batch thủ công. Cách đó vừa dài, vừa khó bảo trì, vừa dễ sai. Trong dự án thật như PlantDoc AI, bạn không nên tự đọc từng ảnh bằng tay trong training loop, vì loop huấn luyện nên chỉ tập trung vào học, không nên gánh thêm việc tổ chức dữ liệu đầu vào.

`Transforms` là chuỗi các phép biến đổi áp dụng lên ảnh trước khi nó thành tensor cuối cùng. Với transfer learning dùng pretrained ImageNet, phần này đặc biệt quan trọng vì model backbone đã quen với định dạng đầu vào nhất định. Ảnh cần được resize về kích thước phù hợp, thường là 224x224, rồi chuyển sang tensor và normalize theo mean, std của ImageNet. Nếu bạn bỏ qua normalize hoặc normalize sai, model vẫn chạy nhưng chất lượng thường giảm rõ rệt.

Điểm rất quan trọng là train transforms và val/test transforms không giống nhau. Với train, bạn thường thêm augmentation vừa phải như lật ngang, xoay nhẹ, random resized crop hoặc color jitter ở mức an toàn để model bớt học thuộc dữ liệu. Với val/test, mục tiêu không phải làm đa dạng dữ liệu mà là đo đúng năng lực thực sự của model trên đầu vào ổn định. Vì vậy, val/test transforms thường đơn giản và nhất quán hơn, ví dụ resize, center crop nếu cần, rồi tensor và normalize.

Trong PlantDoc AI, luồng dữ liệu diễn ra như sau: ảnh được lưu trong các thư mục lớp, `Dataset` đọc đường dẫn ảnh và xác định nhãn, `Transforms` biến ảnh sang tensor đúng chuẩn, `DataLoader` gom nhiều mẫu thành batch, và cuối cùng batch đó được đưa vào model. Đây là một đường ống rõ ràng. Nếu bạn hiểu từng đoạn, bạn sẽ dễ debug hơn rất nhiều.

### Ví dụ bám sát PlantDoc AI

Giả sử dữ liệu được tổ chức như sau:

```text
plantvillage_splits/
├── train/
│   ├── Tomato_Early_blight/
│   ├── Tomato_healthy/
│   └── Potato_Late_blight/
├── val/
│   ├── Tomato_Early_blight/
│   ├── Tomato_healthy/
│   └── Potato_Late_blight/
└── test/
    ├── Tomato_Early_blight/
    ├── Tomato_healthy/
    └── Potato_Late_blight/
```

Một cách đơn giản để đọc dữ liệu kiểu này là dùng `torchvision.datasets.ImageFolder`.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

train_dataset = datasets.ImageFolder("plantvillage_splits/train", transform=train_transform)
val_dataset = datasets.ImageFolder("plantvillage_splits/val", transform=val_transform)
test_dataset = datasets.ImageFolder("plantvillage_splits/test", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(train_dataset.classes)
```

Đoạn code trên đang làm một việc rất quan trọng. `ImageFolder` tự suy ra tên lớp từ tên thư mục con. Ví dụ thư mục `Tomato_healthy` sẽ trở thành một class. Khi lấy một mẫu, dataset sẽ mở file ảnh, áp dụng transform, rồi trả ra một cặp `(image_tensor, label_index)`.

Bạn có thể kiểm tra một batch đầu ra như sau:

```python
images, labels = next(iter(train_loader))
print(images.shape)  # ví dụ: torch.Size([32, 3, 224, 224])
print(labels.shape)  # ví dụ: torch.Size([32])
```

`images.shape = [32, 3, 224, 224]` nghĩa là batch có 32 ảnh, mỗi ảnh có 3 kênh màu RGB, cao 224 và rộng 224. `labels.shape = [32]` nghĩa là có 32 nhãn số nguyên tương ứng với 32 ảnh trong batch. Đây chính là định dạng mà model classification mong đợi.

Nếu bạn muốn hiểu ảnh đi từ đĩa vào batch như thế nào, hãy đọc theo thứ tự sau. Đầu tiên `ImageFolder` quét thư mục và lập danh sách file ảnh cùng nhãn. Khi DataLoader gọi đến một chỉ số nào đó, dataset mở ảnh bằng PIL, áp transform để biến nó từ ảnh thành tensor, rồi trả về cho DataLoader. DataLoader gom nhiều mẫu như vậy thành một batch duy nhất. Batch đó mới được chuyển sang GPU và đưa vào model.

### Lỗi hay gặp

- Dùng cùng một transform augmentation cho cả train và val/test làm validation thiếu ổn định.
- Quên normalize theo ImageNet khi dùng pretrained model.
- Để `shuffle=False` cho train loader khiến model học kém linh hoạt hơn.
- Tưởng `num_workers` càng lớn càng tốt, nhưng trên một số máy Windows có thể gây lỗi hoặc chậm hơn.
- Không kiểm tra shape của batch nên đến lúc đưa vào model mới phát hiện sai.

### Checklist áp dụng rất ngắn

- Ảnh train/val/test đã tách rõ.
- Train transform có augmentation vừa phải.
- Val/test transform ổn định, không augmentation ngẫu nhiên.
- Có normalize theo ImageNet.
- Kiểm tra được shape của một batch mẫu.

---

## 2) Training loop và Validation loop trong PlantDoc AI

### Mục tiêu

Phần này giúp bạn hiểu training loop không phải phép màu, mà là một chuỗi thao tác logic rõ ràng. Sau phần này, bạn phải đọc được từng khối lệnh trong vòng lặp huấn luyện và biết vì sao nó tồn tại.

### Giải thích

Trong bài toán classification của PlantDoc AI, ba thành phần trung tâm của giai đoạn học là `model`, `loss function` và `optimizer`. Model nhận batch ảnh đầu vào và sinh ra logits. Loss function so sánh logits đó với nhãn thật để đo xem model đang sai nhiều hay ít. Optimizer dùng gradient để cập nhật trọng số model theo hướng giảm loss.

`model.train()` là lệnh chuyển model sang chế độ huấn luyện. Điều này đặc biệt quan trọng nếu model có các lớp như Dropout hoặc BatchNorm. Ở chế độ train, Dropout sẽ hoạt động để regularize, còn BatchNorm sẽ cập nhật thống kê theo batch hiện tại. Nếu bạn quên gọi `model.train()`, model vẫn chạy nhưng hành vi của một số lớp sẽ không đúng như mong muốn trong huấn luyện.

Một epoch train thường gồm các bước lặp lại trên từng batch. Đầu tiên lấy batch ảnh và nhãn từ `DataLoader`. Sau đó chuyển dữ liệu sang device như CPU hoặc GPU. Tiếp theo gọi `optimizer.zero_grad()` để xóa gradient cũ còn sót lại. Sau đó chạy forward pass, tức đưa batch ảnh qua model để lấy đầu ra. Dùng đầu ra đó tính loss. Tiếp theo gọi `loss.backward()` để tính gradient cho các tham số. Cuối cùng gọi `optimizer.step()` để cập nhật trọng số. Đó là trục xương sống của training loop.

Nhiều sinh viên mới học thường thấy vòng lặp này như một công thức phải thuộc lòng. Cách hiểu tốt hơn là xem nó như một quy trình logic. Model đoán, loss chấm điểm sai, backward truy ngược xem sai vì đâu, optimizer sửa trọng số. Toàn bộ vòng lặp chỉ là lặp lại quy trình đó trên nhiều batch và nhiều epoch.

Validation loop tồn tại vì bạn không thể chỉ nhìn train accuracy rồi kết luận model tốt. Một model có thể đạt train accuracy rất cao nhưng chỉ đang học thuộc dữ liệu huấn luyện. Validation là tập dữ liệu không dùng để cập nhật trọng số, chỉ dùng để quan sát khả năng tổng quát hóa. Chính vì vậy, validation mới là nơi quyết định early stopping và best checkpoint.

`model.eval()` chuyển model sang chế độ đánh giá. Với BatchNorm và Dropout, hành vi lúc này sẽ khác train mode. `torch.no_grad()` giúp tắt việc lưu đồ thị gradient, làm quá trình validate và inference nhẹ hơn, nhanh hơn và ít tốn bộ nhớ hơn. Nếu bạn validate mà quên `eval()` hoặc `no_grad()`, kết quả có thể lệch hoặc tài nguyên bị lãng phí.

Trong PlantDoc AI, ngoài `val_loss` và `val_accuracy`, bạn nên chú ý cả `F1-macro`, đặc biệt khi dữ liệu giữa các lớp không cân bằng. Accuracy có thể cao vì model làm tốt ở lớp đông mẫu, nhưng F1-macro mới cho bạn cảm giác công bằng hơn giữa các lớp. Đây là lý do nhiều dự án classification thực tế không chỉ nhìn mỗi accuracy.

### Ví dụ bám sát PlantDoc AI

Dưới đây là một mini training loop theo phong cách PlantDoc AI.

```python
import torch
import torch.nn as nn
from torchvision import models

num_classes = 38
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

Đoạn này khởi tạo model pretrained, thay classifier cuối để phù hợp với số lớp bệnh lá cây, rồi tạo loss và optimizer. Từ đây trở đi, model đã sẵn sàng để đi vào training loop.

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc
```

Bây giờ hãy đọc loop này đúng kiểu mentor đọc code. `model.train()` bật chế độ train. `running_loss`, `running_correct`, `total_samples` là các biến để cộng dồn thống kê trong cả epoch. Trong mỗi batch, dữ liệu được chuyển sang device. `optimizer.zero_grad()` xóa gradient cũ. `outputs = model(images)` là forward pass. `loss = criterion(outputs, labels)` tính mức sai. `loss.backward()` tính gradient. `optimizer.step()` cập nhật trọng số. Sau đó ta tính prediction bằng `argmax`, so sánh với nhãn thật để lấy accuracy.

Phần validation nên tách riêng như sau:

```python
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc
```

Loop validation gần giống loop train, nhưng có hai khác biệt sống còn. Thứ nhất là `model.eval()`. Thứ hai là không có backward và optimizer step, đồng thời toàn bộ được bọc trong `torch.no_grad()`. Điều đó nhắc bạn rằng validation chỉ để đo chất lượng chứ không phải để học.

Bạn có thể nối hai hàm này trong vòng lặp epoch tổng như sau:

```python
num_epochs = 10
best_val_loss = float("inf")

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
```

Nếu muốn sát hơn với PlantDoc AI, bạn có thể tính thêm F1-macro ở validation bằng `sklearn.metrics.f1_score`. Điều quan trọng là metric nào được chọn làm chuẩn để lưu best checkpoint phải nhất quán và có ý nghĩa với bài toán.

### Lỗi hay gặp

- Quên `model.train()` nên huấn luyện không đúng chế độ.
- Quên `model.eval()` khi validate làm kết quả dao động khó hiểu.
- Quên `optimizer.zero_grad()` khiến gradient bị cộng dồn ngoài ý muốn.
- Tính accuracy sai do chia sai mẫu số hoặc lấy nhầm shape.
- Nhìn train accuracy cao rồi nghĩ model tốt, trong khi val kém.
- Data leakage vì cách split dữ liệu không chuẩn hoặc transform áp sai vào val/test.

### Checklist áp dụng rất ngắn

- Train loop có đủ: zero_grad, forward, loss, backward, step.
- Validation loop có: eval và no_grad.
- Theo dõi ít nhất val loss và val accuracy.
- Nếu dữ liệu lệch lớp, theo dõi thêm F1-macro.
- Chọn metric validation để quyết định best checkpoint.

---

## 3) Checkpoint, load model và inference trong PlantDoc AI

### Mục tiêu

Phần này giúp bạn hiểu rằng train xong chưa phải là xong. Muốn làm được dự án thật, bạn phải biết lưu đúng, load đúng và dùng lại model đúng trong môi trường inference.

### Giải thích

`Checkpoint` là trạng thái đã lưu của quá trình huấn luyện tại một thời điểm nào đó. Nó giống như việc bạn chụp lại toàn bộ những gì cần thiết để có thể quay lại hoặc dùng model về sau. Trong dự án thật, đặc biệt là PlantDoc AI, bạn gần như luôn cần checkpoint.

Vì sao phải lưu best checkpoint thay vì chỉ lưu epoch cuối? Vì epoch cuối chưa chắc là tốt nhất trên validation. Có những giai đoạn model bắt đầu overfit: train accuracy tiếp tục tăng nhưng val loss xấu đi hoặc F1-macro giảm. Nếu bạn chỉ giữ epoch cuối, bạn có thể vô tình dùng bản model tệ hơn bản ở vài epoch trước. Lưu best checkpoint theo metric validation là cách thực tế hơn rất nhiều.

Một checkpoint tốt thường không chỉ chứa `model_state_dict`. Nếu bạn muốn resume training, nên lưu cả `optimizer_state_dict`, `epoch`, `best_metric`. Với bài toán classification như PlantDoc AI, bạn còn nên lưu `class_to_idx` hoặc mapping lớp và một phần config cơ bản như tên model, input size, mean/std dùng cho preprocessing. Điều này giúp inference về sau không bị lệch pipeline.

Load model để resume training khác với load model để inference. Khi resume training, bạn cần khôi phục cả model lẫn optimizer, biết đang ở epoch nào và metric tốt nhất hiện tại là bao nhiêu. Khi inference, bạn chỉ cần dựng lại đúng kiến trúc model, load trọng số, chuyển sang `eval()`, rồi dùng đúng preprocessing để dự đoán.

Inference pipeline trong PlantDoc AI nhìn đơn giản nhưng rất dễ sai. Quy trình đúng là: load checkpoint, dựng lại model đúng kiến trúc, load weight, gọi `model.eval()`, preprocess ảnh đúng như lúc train, thêm batch dimension vì model luôn mong đầu vào theo batch, chạy forward, lấy class dự đoán và map về tên bệnh. Rất nhiều lỗi demo Streamlit thực ra không đến từ model, mà đến từ inference pipeline khác với training pipeline.

### Ví dụ bám sát PlantDoc AI

Một ví dụ mini để save best checkpoint theo validation loss:

```python
import torch

best_val_loss = float("inf")

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "class_to_idx": train_dataset.class_to_idx,
            "model_name": "mobilenet_v2",
            "input_size": 224,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        }
        torch.save(checkpoint, "checkpoints/best_model.pth")
        print("Saved best checkpoint")
```

Đoạn này nói lên một tư duy rất quan trọng. Bạn không chỉ lưu weight, mà còn lưu thông tin để giải thích cách model đã được train. Điều đó giúp việc resume hoặc inference về sau đáng tin cậy hơn.

Bây giờ là ví dụ load model để inference:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_model_for_inference(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((checkpoint["input_size"], checkpoint["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["imagenet_mean"],
            std=checkpoint["imagenet_std"]
        )
    ])

    return model, transform, idx_to_class
```

Ở đây có ba thứ được trả về cùng nhau: model, transform và mapping từ index sang tên lớp. Đây là cách tổ chức rất hợp lý cho app inference, vì ba phần này luôn đi cùng nhau.

Hàm inference mini cho ảnh người dùng upload có thể viết như sau:

```python
def predict_image(image_path, model, transform, idx_to_class, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # thêm batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        pred_idx = outputs.argmax(dim=1).item()

    pred_class = idx_to_class[pred_idx]
    return pred_class
```

Điểm đáng chú ý nhất ở đây là `unsqueeze(0)`. Một ảnh đơn lẻ ban đầu có shape `[3, 224, 224]`, nhưng model classification thường mong đầu vào dạng batch là `[N, 3, 224, 224]`. Thêm chiều batch biến nó thành `[1, 3, 224, 224]`.

Nếu đặt vào bối cảnh Streamlit, luồng có thể như sau: người dùng upload ảnh lá cà chua, app nhận file, mở ảnh bằng PIL, áp transform giống lúc train, gọi model để dự đoán và trả ra kết quả như `Tomato_Early_blight`. Nếu muốn hiển thị đẹp hơn, bạn có thể lấy thêm softmax probability để hiện top-3 dự đoán.

Ví dụ mini theo kiểu pseudo-flow trong Streamlit:

```python
uploaded_file = st.file_uploader("Upload ảnh lá cây", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh người dùng upload")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        pred_idx = outputs.argmax(dim=1).item()

    pred_label = idx_to_class[pred_idx]
    st.write(f"Kết quả dự đoán: {pred_label}")
```

### Lỗi hay gặp

- Load sai file checkpoint hoặc nhầm giữa checkpoint và raw state_dict.
- Dựng sai kiến trúc model trước khi `load_state_dict()`.
- `class_to_idx` lúc inference không khớp với lúc train.
- Transform ở app khác transform lúc train, đặc biệt là normalize.
- Quên `model.eval()` nên kết quả dự đoán thiếu ổn định.
- Quên thêm batch dimension trước khi forward.

### Checklist áp dụng rất ngắn

- Lưu best checkpoint theo metric validation.
- Checkpoint có đủ weight và class mapping.
- Inference dùng đúng kiến trúc model đã train.
- Preprocess ở app phải nhất quán với train.
- Gọi `model.eval()` trước khi dự đoán.

---

## 4) Kết nối 3 khối thành một workflow hoàn chỉnh cho PlantDoc AI

### Mục tiêu

Phần này giúp bạn nối ba khối đã học thành một workflow duy nhất, để nhìn rõ luồng từ dữ liệu đến dự đoán thay vì xem từng phần rời rạc.

### Giải thích

Toàn bộ pipeline của PlantDoc AI có thể nhìn như một đường đi liên tục. Đầu tiên, ảnh lá cây nằm trong thư mục dữ liệu. `Dataset` biết cách tìm chúng và gắn nhãn cho chúng. `Transforms` biến chúng thành tensor đúng chuẩn của pretrained ImageNet. `DataLoader` gom các tensor đó thành batch để model học hiệu quả.

Sau đó batch đi vào training loop. Model thực hiện forward pass, loss đo độ sai, backward tạo gradient và optimizer cập nhật trọng số. Sau mỗi epoch, validation loop kiểm tra xem model đang tổng quát hóa tốt đến đâu trên tập val. Metric validation sẽ quyết định checkpoint nào là tốt nhất.

Cuối cùng, checkpoint tốt nhất được load lại ở pha inference. Ảnh người dùng upload trong Streamlit cũng phải đi qua preprocessing tương thích, rồi mới được đưa vào model để dự đoán. Như vậy, đầu vào lúc demo thực ra vẫn đang đi lại một phần của pipeline cũ, chỉ khác là không còn backward và optimizer nữa.

Điều rất đáng nhớ là ba khối này không độc lập. Nếu transform sai, training học trên dữ liệu méo. Nếu validation sai, best checkpoint bị chọn nhầm. Nếu inference pipeline lệch, app demo cho kết quả sai dù model đã train tốt. Chỉ cần một mắt xích lỗi, cả pipeline trở nên thiếu đáng tin.

### Ví dụ bám sát PlantDoc AI

Một cách hình dung ngắn gọn là:

```text
Folder ảnh
→ Dataset đọc từng ảnh và nhãn
→ Transforms biến ảnh thành tensor chuẩn
→ DataLoader tạo batch
→ Training loop cập nhật trọng số
→ Validation loop chọn model tốt nhất
→ Save best checkpoint
→ Load checkpoint trong Streamlit
→ Inference trên ảnh người dùng upload
→ Trả ra tên bệnh lá cây
```

### Lỗi hay gặp

- Hiểu từng file riêng lẻ nhưng không hiểu chúng nối với nhau thế nào.
- Train ổn nhưng app demo sai vì inference pipeline không đồng bộ.
- Không lưu class mapping nên giai đoạn cuối bị lệch nhãn.

### Checklist áp dụng rất ngắn

- Kiểm tra dữ liệu, train, checkpoint và inference có cùng logic.
- Dùng một nguồn config nhất quán cho input size và normalize.
- Luôn test toàn pipeline bằng vài ảnh mẫu sau khi train xong.

---

## 5) Lỗi thường gặp nhất khi dùng PyTorch trong PlantDoc AI

### Mục tiêu

Phần này tập trung vào những lỗi cốt lõi nhất mà sinh viên rất hay gặp khi làm project classification bằng PyTorch. Mục tiêu là giúp bạn nhận ra lỗi nhanh và sửa đúng chỗ.

### Giải thích

Những lỗi nguy hiểm nhất trong PlantDoc AI thường không phải lỗi syntax, mà là lỗi logic pipeline. Code vẫn chạy, nhưng kết quả sai hoặc thiếu ổn định. Đây là kiểu lỗi khiến nhiều bạn mất nhiều thời gian nhất vì nhìn bề ngoài mọi thứ có vẻ bình thường.

### Các lỗi cốt lõi nhất

- **Sai transform giữa train và inference**  
  Dấu hiệu là validation khá ổn nhưng demo dự đoán lung tung. Nguyên nhân là ảnh ở app đi qua một pipeline preprocess khác lúc train, ví dụ khác resize hoặc thiếu normalize. Cách kiểm tra nhanh là in ra transform của train/val và transform của inference để so từng bước. Cách sửa là gom preprocess vào cùng một nơi cấu hình hoặc lưu thông số preprocess vào checkpoint.

- **Quên normalize theo ImageNet**  
  Dấu hiệu là model học chậm, accuracy thấp bất thường dù code không báo lỗi. Nguyên nhân là pretrained backbone mong chờ đầu vào đã được normalize theo thống kê ImageNet. Cách kiểm tra nhanh là xem transform có `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` hay không. Cách sửa là thêm normalize nhất quán ở train, val và inference.

- **Quên `model.train()` hoặc `model.eval()`**  
  Dấu hiệu là metric dao động khó hiểu, đặc biệt khi model có BatchNorm hoặc Dropout. Nguyên nhân là mode của model không đúng với mục đích hiện tại. Cách kiểm tra nhanh là nhìn thẳng vào đầu mỗi hàm `train_one_epoch()` và `validate_one_epoch()`. Cách sửa là luôn đặt `model.train()` trong train loop và `model.eval()` trong validation/inference.

- **Quên `torch.no_grad()` khi validate hoặc inference**  
  Dấu hiệu là validate tốn bộ nhớ hơn dự kiến hoặc chậm không cần thiết. Nguyên nhân là PyTorch vẫn lưu đồ thị gradient dù bạn không dùng để backward. Cách kiểm tra nhanh là xem loop validate/inference có được bọc bởi `with torch.no_grad():` không. Cách sửa là thêm `torch.no_grad()` cho mọi đoạn chỉ dùng để dự đoán.

- **Quên `optimizer.zero_grad()`**  
  Dấu hiệu là loss học rất lạ, gradient bị tích lũy ngoài ý muốn. Nguyên nhân là gradient của batch trước chưa được xóa. Cách kiểm tra nhanh là xem trong mỗi vòng lặp train có gọi `optimizer.zero_grad()` trước forward/backward không. Cách sửa là đặt nó ngay đầu mỗi iteration.

- **Batch tensor sai shape**  
  Dấu hiệu là model báo lỗi kích thước đầu vào, hoặc đôi khi không lỗi nhưng metric sai. Nguyên nhân có thể là ảnh chưa đúng số kênh, chưa resize, hoặc quên thêm batch dimension ở inference. Cách kiểm tra nhanh là in `images.shape` trước khi đưa vào model. Cách sửa là chuẩn hóa shape về `[batch_size, 3, 224, 224]` cho train và `[1, 3, 224, 224]` cho ảnh đơn.

- **Lưu checkpoint không đủ thông tin**  
  Dấu hiệu là load lại được weight nhưng không biết map nhãn hoặc không resume train được. Nguyên nhân là chỉ lưu `model_state_dict` mà quên những phần phụ trợ. Cách kiểm tra nhanh là mở checkpoint và xem các key đang có. Cách sửa là lưu thêm `epoch`, `best_metric`, `optimizer_state_dict`, `class_to_idx`, `input_size`, `mean`, `std`, `model_name`.

- **Load checkpoint đúng model nhưng sai class mapping**  
  Dấu hiệu là model dự đoán ra index hợp lý nhưng tên bệnh hiển thị sai. Nguyên nhân là thứ tự lớp ở inference khác với lúc train. Cách kiểm tra nhanh là in `class_to_idx` từ checkpoint và so với mapping hiện tại của app. Cách sửa là dùng mapping lưu sẵn trong checkpoint thay vì tự tạo lại bằng tay.

### Checklist áp dụng rất ngắn

- Khi kết quả lạ, kiểm tra lại pipeline trước khi nghi ngờ model.
- In shape, in mapping lớp, in transform để debug nhanh.
- Đừng chỉ kiểm tra code có chạy hay không, hãy kiểm tra logic có khớp hay không.

---

## 6) Mini roadmap áp dụng ngay vào PlantDoc AI

### Mục tiêu

Phần này biến toàn bộ nội dung vừa học thành một kế hoạch hành động ngắn gọn để bạn bắt đầu code ngay trong repo PlantDoc AI.

### Giải thích

Cách tốt nhất để học PyTorch trong đồ án không phải là đọc hết tài liệu tổng quát, mà là tự tay dựng pipeline tối thiểu chạy được. Khi bạn đi từng bước nhỏ, mỗi bước đều có đầu ra rõ ràng, bạn sẽ dễ kiểm soát code hơn và cũng dễ commit hơn.

### Kế hoạch hành động

- **Bước 1:** Chuẩn bị folder dữ liệu theo train/val/test và viết train transform, val/test transform với resize 224x224 và normalize theo ImageNet.
- **Bước 2:** Tạo `Dataset` và `DataLoader`, sau đó in thử một batch để kiểm tra shape và label.
- **Bước 3:** Viết `train_one_epoch()` và `validate_one_epoch()` chạy được với một model pretrained đơn giản như MobileNetV2.
- **Bước 4:** Thêm logic lưu best checkpoint dựa trên val loss hoặc F1-macro.
- **Bước 5:** Viết `load_model_for_inference()` và `predict_image()` cho ảnh đơn.
- **Bước 6:** Nối hàm inference vào Streamlit để người dùng upload ảnh và xem kết quả.
- **Bước 7:** Kiểm tra lại toàn pipeline để chắc chắn train transform, val transform, checkpoint info và inference preprocess không bị lệch nhau.

### Ví dụ bám sát PlantDoc AI

Một cách triển khai thực tế trong repo có thể là tách thành các file như `data/transforms.py`, `data/datasets.py`, `training/trainer.py`, `training/checkpoint.py`, `inference/predict.py`, `app.py`. Cách tách này giúp bạn nhìn đúng ba khối lớn và giữ code dễ đọc khi đồ án bắt đầu lớn hơn.

### Lỗi hay gặp

- Nhảy ngay vào viết app khi phần train và checkpoint chưa ổn định.
- Không test từng bước nhỏ nên đến cuối mới phát hiện lỗi pipeline.
- Code được nhưng không tách module, dẫn đến khó sửa và khó mở rộng.

### Checklist áp dụng rất ngắn

- Mỗi bước phải chạy được trước khi sang bước sau.
- Mỗi phần nên có test nhỏ hoặc print kiểm tra.
- Ưu tiên pipeline đúng trước, tối ưu sau.

---

## 7) Kết luận

### Mục tiêu

Phần cuối cùng chốt lại điều cốt lõi nhất mà bạn cần nhớ khi dùng PyTorch cho PlantDoc AI.

### Giải thích

Nếu nắm chắc `Dataset/DataLoader/Transforms`, `training loop/validation loop`, và `checkpoint/load model/inference`, bạn đã nắm được phần lõi nhất của PyTorch trong PlantDoc AI. Đây là phần tạo ra khả năng làm việc thật với dự án, chứ không chỉ dừng ở mức đọc ví dụ rời rạc.

Điều quan trọng không phải là nhớ thật nhiều API. Điều quan trọng hơn là hiểu dữ liệu đi như thế nào, model học ra sao, và kết quả được lưu rồi dùng lại thế nào. Khi bạn hiểu ba điều đó, bạn sẽ bớt sợ repo PyTorch, bớt sợ training loop và cũng bớt cảm giác mọi thứ quá nhiều.

Trong đồ án thật, người làm tốt thường không phải người thuộc nhiều lệnh nhất, mà là người giữ được pipeline nhất quán từ đầu đến cuối. Với PlantDoc AI, sự nhất quán đó thể hiện ở dữ liệu đầu vào, cách train, cách chọn checkpoint và cách inference trong app. Khi làm tốt ba phần này, bạn có thể tự tin đọc repo, sửa code, debug lỗi và mở rộng dự án theo hướng chuyên nghiệp hơn.

### Tóm tắt cuối tài liệu

- PyTorch trong PlantDoc AI là một pipeline, không chỉ là model.
- Ba khối cốt lõi nhất là dữ liệu, train/eval và checkpoint/inference.
- Hiểu đúng workflow quan trọng hơn nhớ nhiều API.
- Muốn demo tốt, inference pipeline phải bám sát training pipeline.
- Nắm chắc ba phần này là đủ để bắt đầu viết và mở rộng PlantDoc AI một cách tự tin.
