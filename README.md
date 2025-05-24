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

![Pipeline](docs/workflow.drawio.pdf)

### Chi tiết:

- Vì dữ liệu chỉ có dạng Question-Answer, việc fine-tune dễ gây **bias**.
- Với **Bi-Encoder**, chúng tôi sử dụng **MultiNegativeRanking loss**.
- Với **Cross-Encoder**, chúng tôi áp dụng **negative mining** để tăng chất lượng mô hình.

### Lưu ý:

- Tập training được chia nhỏ thành `train` và `eval` để tự đánh giá do hạn chế số lần nộp bài.
- Sự khác biệt giữa các tập `eval`, `public`, `private` là **không đáng kể**.
- Phương pháp **không dùng ensemble** nhưng vẫn đạt hiệu quả cao.
- Dễ dàng **mở rộng** cho các dataset khác chỉ có dạng QA.

## 🚀 Hướng dẫn Reproduce

### 1. Xử lý dữ liệu:

```bash
$python data_processing.py
``` 

### 2. Huấn luyện bước Retrieval:
```bash
$python train_bi.py
#$python bm25.py (Tuỳ chọn) Thử nghiệm BM25:
``` 
### 3. Đánh giá Retrieval model:

```bash
$python predict_bi.py --train
$python evaluation.py 
```
