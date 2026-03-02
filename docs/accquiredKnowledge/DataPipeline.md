# Data Pipeline & Image Preprocessing cho PlantDoc AI (PyTorch + torchvision)

## Mục lục (TOC)

* [Level 0 — Tổng quan pipeline dữ liệu trong PlantDoc AI](#level-0--tổng-quan-pipeline-dữ-liệu-trong-plantdoc-ai)
* [Level 1 — Resize, Crop, Normalize (TRỌNG TÂM)](#level-1--resize-crop-normalize-trọng-tâm)
* [Level 2 — Augmentation “vừa đủ” cho bệnh lá (TRỌNG TÂM)](#level-2--augmentation-vừa-đủ-cho-bệnh-lá-trọng-tâm)
* [Level 3 — Split train/val/test có seed + stratified (TRỌNG TÂM)](#level-3--split-trainvaltest-có-seed--stratified-trọng-tâm)
* [Level 4 — Data imbalance & metric: vì sao ưu tiên F1-macro](#level-4--data-imbalance--metric-vì-sao-ưu-tiên-f1-macro)
* [Level 5 — Code template “project-ready” cho PlantDoc AI (PyTorch)](#level-5--code-template-project-ready-cho-plantdoc-ai-pytorch)
* [Level 6 — Lỗi thường gặp trong PlantDoc AI và cách debug nhanh](#level-6--lỗi-thường-gặp-trong-plantdoc-ai-và-cách-debug-nhanh)
* [Level 7 — “Data & Preprocessing” viết vào report (đúng kiểu đồ án)](#level-7--data--preprocessing-viết-vào-report-đúng-kiểu-đồ-án)

---

## Level 0 — Tổng quan pipeline dữ liệu trong PlantDoc AI

**(1) Mục tiêu level**
Hiểu pipeline dữ liệu end-to-end để training/inference đúng chuẩn, tránh leakage, và tái lập kết quả cho báo cáo đồ án.

**(2) Diễn giải**
Trong PlantDoc AI, pipeline dữ liệu thường đi theo chuỗi: nạp ảnh → chia train/val/test → áp transforms → DataLoader tạo batch → train/val/test → tính metrics. Điểm “hay sai” nhất là dùng chung transform cho mọi split, khiến val/test bị “augment” và metric bị ảo. Một yêu cầu quan trọng của report là kết quả tái lập: cùng seed + cùng split + cùng transform phải ra kết quả gần như y hệt.

**(3) Ví dụ/mini case**
Bạn dùng EfficientNet-B0 pretrained ImageNet để phân loại ~38 lớp như PlantVillage, sau đó test trên ảnh thực địa kiểu PlantDoc (nền nhiễu). Nếu bạn vô tình dùng RandomResizedCrop cho val/test, accuracy/F1 có thể tăng giả vì ảnh val/test “dễ” theo kiểu crop ngẫu nhiên giống train. Khi demo Streamlit, model lại “rớt” mạnh vì inference không giống pipeline val/test.

**(4) Lỗi hay gặp**

* Dùng augmentation cho val/test → đánh giá sai, chọn model sai.
* Không lưu split → mỗi lần chạy chia khác nhau, report không tái lập.
* Normalize sai → pretrained backbone nhận ảnh “lệch phân phối”, học chậm hoặc tụt hiệu năng.

**(5) Checklist áp dụng**

* Train transform **khác** val/test transform.
* Split có **seed + stratified**, và **lưu ra file**.
* Pretrained dùng **ImageNet normalize** đúng mean/std.

---

## Level 1 — Resize, Crop, Normalize 
**(1) Mục tiêu level**
Dùng đúng Resize/Crop/Normalize (đặc biệt ImageNet normalize) để tận dụng pretrained backbone mà không làm mất vùng bệnh quan trọng.

**(2) Diễn giải**
Resize giúp mọi ảnh về cùng kích thước để tạo batch và phù hợp input model (thường 224×224). Crop là thao tác “cắt vùng” để lấy đúng size; CenterCrop ổn cho ảnh chụp tương đối trung tâm, nhưng có rủi ro cắt mất vết bệnh nếu bệnh nằm rìa lá. Normalize theo ImageNet (mean/std) là bước đưa ảnh về phân phối mà backbone đã học; nếu bỏ qua hoặc dùng sai, model có thể “nhìn màu” sai, giảm khả năng transfer learning.

**(3) Ví dụ/mini case**
Với PlantVillage, lá thường chiếm phần lớn khung hình và khá trung tâm, pipeline `Resize(256) + CenterCrop(224)` thường ổn. Với PlantDoc ảnh thực địa, lá có thể lệch, nhỏ, hoặc bị che; CenterCrop có thể cắt nhầm nền và mất vùng bệnh, khiến model học sai tín hiệu. Khi đó, train nên dùng `RandomResizedCrop(224)` (giới hạn hợp lý) để tăng robust, còn val/test dùng cách ổn định (Resize + CenterCrop) để đo lường công bằng.

**(4) Minh họa 2 pipeline phổ biến**

* (A) **Resize(256) + CenterCrop(224) + Normalize**: ổn định, thường dùng cho **val/test** (và có thể cho train nếu dữ liệu sạch, trung tâm).
* (B) **RandomResizedCrop(224) + Normalize**: tăng đa dạng góc nhìn, chỉ dùng cho **train**.

**Code transform tối thiểu (train/val/test) + giải thích ngay sau code**

```python
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tf = val_tf
```

Đoạn `ToTensor()` chuyển ảnh về tensor [0..1] theo kênh (C,H,W), sau đó `Normalize` chuẩn hóa theo ImageNet để backbone pretrained “nhận đúng kiểu ảnh” như lúc pretrain. `RandomResizedCrop` đặt scale tối thiểu 0.7 để giảm nguy cơ crop quá gắt làm mất vùng bệnh (đây là trade-off quan trọng cho ảnh lá). Val/test dùng pipeline ổn định để so sánh mô hình công bằng; nếu bạn ngẫu nhiên hóa val/test, bạn sẽ khó biết model thật sự tốt hay chỉ “ăn may” theo crop.

**(4) Lỗi hay gặp**

* Quên normalize hoặc normalize sai mean/std → pretrained bị lệch, học lâu, F1 giảm.
* CenterCrop cắt mất vùng bệnh → model học nền hoặc học “lá sạch”.
* RandomResizedCrop scale quá nhỏ (vd 0.2) → crop cực gắt, nhãn bị “phá” vì bệnh không còn trong ảnh.

**(5) Checklist áp dụng**

* Input size thống nhất: **224×224** (hoặc Resize 256 + crop 224 cho val/test).
* Dùng đúng **ImageNet mean/std** khi pretrained ImageNet.
* Kiểm soát crop để tránh cắt mất vùng bệnh (đặc biệt dữ liệu thực địa).

---

## Level 2 — Augmentation “vừa đủ” cho bệnh lá

**(1) Mục tiêu level**
Thiết kế augmentation tăng khả năng tổng quát (đặc biệt PlantDoc ảnh thực địa) nhưng không làm sai nhãn hoặc tạo tín hiệu giả.

**(2) Diễn giải**
Augmentation đúng giúp model bền vững trước thay đổi góc chụp, ánh sáng, và khung hình. Nhưng với bệnh lá, màu sắc và texture là tín hiệu quan trọng; augmentation quá tay (đặc biệt color jitter mạnh) có thể biến bệnh thật thành “trông như” bệnh khác hoặc mất dấu hiệu bệnh. Quy tắc thực dụng: augment “vừa đủ”, rồi sanity check bằng mắt để chắc chắn không phá nhãn.

**(3) Giải thích theo từng augmentation **
Horizontal flip thường an toàn vì bệnh trên lá không phụ thuộc trái/phải, nên giúp tăng dữ liệu hiệu quả. Ngoại lệ hiếm là ảnh có chữ/nhãn/marker định hướng (ví dụ tên giống, nhãn dán) hoặc bài toán nhãn phụ thuộc hướng chụp; nếu có, flip có thể tạo dữ liệu “không có thật”.

Rotation nhẹ (±10°~±20°) phản ánh rung tay hoặc góc chụp lệch nhẹ ngoài thực tế. Quay quá mạnh dễ tạo ảnh “phi thực” (lá bị xoay cực đoan, viền cắt xấu) và làm model học các artifact do xoay thay vì học bệnh.

Color jitter vừa (nhẹ) giúp model chịu được thay đổi ánh sáng, nhưng bệnh lá nhiều khi là “vấn đề màu” (vàng lá, đốm nâu, mốc trắng). Nếu jitter mạnh, bạn vô tình làm bệnh “biến màu” và model học sai mối liên hệ giữa màu và nhãn.

Random resized crop giúp model robust khi lá chiếm tỉ lệ khác nhau trong khung hình (zoom in/out). Nhưng nếu crop quá gắt, bạn có thể cắt mất vùng bệnh, khiến ảnh mang nhãn bệnh nhưng không còn dấu hiệu bệnh, làm label noise tăng mạnh.

**(3) Ví dụ/mini case**
Bạn train trên PlantVillage khá sạch, test trên PlantDoc ảnh thực địa nền cỏ/đất. Nếu không có crop + jitter nhẹ, model có thể overfit nền “trơn” của PlantVillage và rớt F1 khi gặp nền phức tạp. Ngược lại, nếu bạn dùng jitter mạnh, model lại nhầm các bệnh có dấu hiệu màu gần nhau (ví dụ các dạng đốm nâu vs cháy lá) vì màu đã bị biến dạng.

**Sanity check augmentation: visualize 16 ảnh sau augment (kèm code ngắn)**

```python
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_augmented_samples(dataset, n=16):
    # dataset: nên là ImageFolder với train_tf
    idx = torch.randperm(len(dataset))[:n].tolist()
    imgs = [dataset[i][0] for i in idx]  # lấy image sau transform
    grid = make_grid(imgs, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
    plt.axis("off")
    plt.show()
```

Sanity check này giúp bạn phát hiện nhanh các lỗi “phá nhãn”: crop cắt mất bệnh, rotation làm lá bị cắt cụt, hoặc màu bị lệch quá mức. Nếu 16 ảnh nhìn “không còn giống lá bệnh thật”, metric tốt đến mấy cũng dễ là ảo và khi chạy Streamlit inference sẽ tụt.

**(4) Lỗi hay gặp**

* Color jitter quá mạnh → mất tín hiệu bệnh hoặc tạo tín hiệu giả.
* RandomResizedCrop scale nhỏ → crop mất vùng bệnh.
* Augment áp vào val/test → metric không phản ánh thực tế.

**(5) Checklist áp dụng**

* Flip: OK đa số trường hợp, nhưng kiểm tra ngoại lệ có chữ/marker.
* Rotation: giữ **nhẹ** (±10–20°).
* Color jitter: **vừa** (nhỏ) vì bệnh phụ thuộc màu.
* Luôn sanity check bằng cách visualize ảnh sau augment.

---

## Level 3 — Split train/val/test có seed + stratified (TRỌNG TÂM)

**(1) Mục tiêu level**
Chia dữ liệu đúng chuẩn đồ án: tránh leakage, có seed để tái lập, và stratified để val/test không thiếu lớp hiếm.

**(2) Diễn giải**
Train dùng để học, val dùng để tune (chọn epoch, chọn hyperparam), và test chỉ dùng để báo cáo cuối cùng. Nếu bạn “nhìn” test trong quá trình tune (dù vô tình), kết quả test không còn khách quan. Seed là chìa khóa để tái lập: cùng split và seed giúp bạn giải thích rõ vì sao model A tốt hơn B trong report. Stratified split giữ tỉ lệ lớp tương đối giống nhau giữa train/val/test; nếu không, val/test có thể thiếu lớp hiếm và F1-macro bị méo.

**(3) Ví dụ/mini case**
Dataset có lớp “healthy” 3000 ảnh, lớp bệnh hiếm chỉ 80 ảnh. Nếu bạn split ngẫu nhiên không stratify, hoàn toàn có thể xảy ra val chỉ còn 2 ảnh của lớp hiếm, hoặc test không có ảnh nào của lớp đó. Khi đó accuracy vẫn cao nhưng model thực chất không biết lớp hiếm; đến lúc demo gặp đúng lớp hiếm là fail ngay.

**(4) Code ví dụ tạo split từ folder gốc bằng sklearn + lưu file (CSV) để tái lập**

Giả sử bạn có folder gốc: `data_all/class_x/*.jpg` (chưa chia train/val/test). Bạn tạo danh sách file + label, sau đó stratified split và lưu ra CSV để lần sau load đúng.

```python
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def build_index(root_dir: str):
    root = Path(root_dir)
    paths, labels = [], []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                paths.append(str(img_path))
                labels.append(class_dir.name)
    return pd.DataFrame({"path": paths, "label": labels})

def make_splits(df, seed=42, val_ratio=0.1, test_ratio=0.1):
    # tách test trước
    df_trainval, df_test = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df["label"]
    )
    # tách val từ trainval
    val_size = val_ratio / (1 - test_ratio)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_size, random_state=seed, stratify=df_trainval["label"]
    )
    return df_train, df_val, df_test

df = build_index("data_all")
train_df, val_df, test_df = make_splits(df, seed=2026)

train_df.to_csv("splits_train.csv", index=False)
val_df.to_csv("splits_val.csv", index=False)
test_df.to_csv("splits_test.csv", index=False)
```

Lưu CSV danh sách `path,label` giúp bạn chạy lại training bất cứ lúc nào mà split không đổi, rất quan trọng cho report (tái lập). Stratify theo `label` giúp các lớp hiếm có cơ hội xuất hiện ở cả val/test, từ đó F1-macro mới phản ánh đúng.

**(4) Lỗi hay gặp**

* Dùng test để “chọn model tốt nhất” → leakage, report thiếu tin cậy.
* Không seed / không lưu split → mỗi lần chạy mỗi kết quả, khó giải thích.
* Không stratify → val/test lệch lớp hoặc thiếu lớp.

**(5) Checklist áp dụng**

* Train/val/test tách rõ; **test chỉ dùng một lần** khi chốt.
* Dùng **seed cố định** và ghi seed trong report.
* Split **stratified theo lớp**, nhất là khi có lớp hiếm.
* **Lưu split** (CSV/JSON) để chạy lại đúng dữ liệu.

---

## Level 4 — Data imbalance & metric: vì sao ưu tiên F1-macro

**(1) Mục tiêu level**
Hiểu vì sao accuracy dễ “ảo” khi lệch lớp trong PlantDoc AI và biết cách báo cáo metric phù hợp (ưu tiên F1-macro).

**(2) Diễn giải**
Khi lớp đa số áp đảo, model đoán “toàn lớp đa số” vẫn đạt accuracy cao, tạo cảm giác model tốt dù thực ra bỏ qua lớp hiếm. F1-macro tính F1 cho từng lớp rồi lấy trung bình, nên mỗi lớp được “đối xử công bằng”, lớp hiếm có trọng lượng tương đương lớp nhiều. Với dữ liệu PlantDoc thực địa, lệch lớp thường rõ hơn, nên F1-macro thường phản ánh chất lượng mô hình tốt hơn accuracy.

**(3) Ví dụ/mini case**
Giả sử 70% ảnh là “healthy”. Model A đoán healthy cho mọi ảnh: accuracy = 70% nhưng F1-macro rất thấp vì các lớp bệnh có recall gần 0. Model B accuracy chỉ 75% nhưng F1-macro cao hơn rõ vì nó bắt được nhiều lớp hiếm; model B thường là lựa chọn đúng cho app thực tế vì người dùng quan tâm “bắt đúng bệnh”, không phải “đoán healthy cho chắc”.

**(4) Confusion matrix để phát hiện model bỏ qua lớp hiếm**
Confusion matrix cho bạn thấy model hay nhầm lớp nào sang lớp nào, và có “cột/ hàng” nào gần như trống (recall = 0). Nếu bạn thấy nhiều lớp hiếm bị dồn vào một lớp phổ biến, đó là dấu hiệu model học thiên lệch hoặc dữ liệu/augment/split có vấn đề. Trong báo cáo PlantDoc AI, confusion matrix giúp bạn giải thích cụ thể thay vì chỉ nêu một con số.
 
**Hướng xử lý imbalance ở mức pipeline**
Bạn có thể cân nhắc `WeightedRandomSampler` để oversample lớp hiếm trong train, hoặc dùng `class weights` trong loss để phạt sai lớp hiếm nặng hơn. Nên cân nhắc khi bạn thấy F1-macro thấp trong khi accuracy ổn, và confusion matrix cho thấy model “bỏ rơi” các lớp hiếm.

**(6) Lỗi hay gặp**

* Chỉ báo cáo accuracy → che giấu việc fail lớp hiếm.
* Không xem confusion matrix → không biết model sai kiểu gì để sửa pipeline.
* Oversample quá tay mà không sanity check → dễ overfit vài ảnh hiếm.

**(7) Checklist áp dụng**

* Ưu tiên **F1-macro** khi lệch lớp.
* Luôn xem **confusion matrix** để chẩn đoán.
* Báo cáo tối thiểu: accuracy + F1-macro + confusion matrix.

---