# Legal Document Retrieval - SoICT Hackathon 2024

Đây là solution đạt **Top 3** tại cuộc thi [Legal Document Retrieval - SoICT Hackathon 2024](https://aihub.ml/competitions/715#results), với **MRR@10 = 0.7754** trên tập **private test**.

## 🧾 Nhiệm vụ

Truy vấn và tìm kiếm thông tin pháp luật từ các văn bản tiếng Việt.

## 📦 Dữ liệu

Dữ liệu được cung cấp bởi ban tổ chức bao gồm 3 tập:

- **Training data**: 119,456 cặp (truy vấn, văn bản liên quan) — dùng để huấn luyện mô hình.
- **Public test**: 10,000 truy vấn — dùng để đánh giá công khai.
- **Private test**: 50,000 truy vấn — dùng để đánh giá cuối cùng trên hệ thống.

> **Tiêu chí đánh giá**: MRR@10

## ⚙️ Phương pháp

Pipeline của chúng tôi gồm 2 bước:

1. **Retrieval** — sử dụng Bi-Encoder: [`vietnamese-bi-encoder`](https://huggingface.co/models)
2. **Re-ranking** — sử dụng Cross-Encoder: [`itdainb/PhoRanker`](https://huggingface.co/itdainb/PhoRanker)

### Training 

![Training workflow](images/training.jpg)

### Inference

![Inference](images/inference-1.png)


### Chi tiết:

- Vì dữ liệu chỉ có dạng Question-Answer, việc fine-tune dễ gây **bias**.
- Với **Bi-Encoder**, chúng tôi sử dụng **MultiNegativeRanking loss**.
- Với **Cross-Encoder**, chúng tôi áp dụng **negative mining** để tăng chất lượng mô hình.

### Lưu ý:

- Tập training được chia nhỏ thành `train` và `eval` để tự đánh giá do hạn chế số lần nộp bài.
- Sự khác biệt giữa các tập `eval`, `public`, `private` là **không đáng kể**.
- Phương pháp **không dùng ensemble** nhưng vẫn đạt hiệu quả cao.
- Dễ dàng **mở rộng** cho các dataset khác chỉ có dạng QA.

Do kích thước mô hình và cơ sở dữ liệu khá lớn, bạn cần tải thủ công các tệp từ liên kết sau:  
🔗 [Tải xuống tại đây](https://drive.google.com/drive/folders/1pWYtYJBIAoI6O_LrThFVANYQQs8a7W7O?usp=sharing)

Sau khi tải về, vui lòng thay thế các thư mục gốc của dự án bằng các thư mục tương ứng:  
- `data`  
- `result`  
- `saved_model`

## 🚀 Reproduce

### 1. Data processing:

```bash
$python data_processing.py 
``` 

### 2. Train BiEncoder: 
```bash
$python train_bi.py
#$python bm25.py (Optinal) Thử nghiệm BM25:
``` 
### 3. Retrieval candiates: 

```bash
$python predict_bi.py --train
```
### 4. Get negative examples for CrossEncoder training: 

```bash
$python negative_mining.py 
``` 

### 5. Train CrossEncoder

```bash
$python train_cross.py
``` 

### 6. Re-rank candidates by CrossEncoder: 

```bash
$python predict_cross.py 
``` 
## 🚀 Hướng dẫn sử dụng

Bạn có thể đặt câu hỏi liên quan đến pháp luật Việt Nam bằng cách sử dụng dòng lệnh như sau:

```bash
$ python run.py --question "Tội bán hàng giả bị xử lý như thế nào?"
``` 
## 📬 Liên hệ

Nếu bạn có bất kỳ thắc mắc hoặc góp ý nào, vui lòng liên hệ qua email:  
📧 [22520465@gm.uit.edu.vn](mailto:22520465@gm.uit.edu.vn)
