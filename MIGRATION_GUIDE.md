# Migration Guide - Cấu trúc mới

Dự án đã được restructure để có cấu trúc rõ ràng và dễ maintain hơn. Đây là hướng dẫn migration.

## Thay đổi chính

### 1. Package structure mới

Code đã được tổ chức thành package `legal_retrieval` với các submodules:
- `legal_retrieval.models` - Model wrappers
- `legal_retrieval.data` - Data processing
- `legal_retrieval.training` - Training utilities
- `legal_retrieval.inference` - Inference pipeline
- `legal_retrieval.evaluation` - Evaluation metrics
- `legal_retrieval.utils` - Utility functions

### 2. Scripts mới

Tất cả các scripts đã được di chuyển vào thư mục `scripts/`:
- `scripts/process_data.py` (thay cho `data_processing.py`)
- `scripts/train_bi.py` (giữ nguyên tên)
- `scripts/train_cross.py` (giữ nguyên tên)
- `scripts/predict_bi.py` (giữ nguyên tên)
- `scripts/predict_cross.py` (giữ nguyên tên)
- `scripts/negative_mining.py` (giữ nguyên tên)
- `scripts/evaluate.py` (thay cho `evaluation.py`)
- `scripts/run_inference.py` (thay cho `run.py`)

### 3. Thư mục đổi tên

**Lưu ý**: Bạn cần đổi tên các thư mục sau (nếu chúng tồn tại):

```bash
# Windows PowerShell
Rename-Item -Path "saved_model" -NewName "saved_models"
Rename-Item -Path "result" -NewName "results"

# Linux/Mac
mv saved_model saved_models
mv result results
```

### 4. Import statements

Nếu bạn có code custom sử dụng các module cũ, cần cập nhật imports:

**Trước:**
```python
from src.utils import get_candidate, mrr_m
from src.get_negative import ran_negative
```

**Sau:**
```python
from legal_retrieval.utils import get_candidate
from legal_retrieval.evaluation import mrr_m
from legal_retrieval.data import NegativeMiner
```

### 5. Cài đặt package

Để sử dụng như một package:

```bash
pip install -e .
```

Sau đó bạn có thể import:

```python
from legal_retrieval.inference import InferencePipeline
from legal_retrieval.training import BiEncoderTrainer
```

## Backward Compatibility

Các file cũ vẫn còn trong thư mục gốc để đảm bảo backward compatibility. Tuy nhiên, khuyến nghị sử dụng cấu trúc mới.

## Checklist Migration

- [ ] Đổi tên `saved_model` → `saved_models`
- [ ] Đổi tên `result` → `results`
- [ ] Cài đặt package: `pip install -e .`
- [ ] Cập nhật các script custom (nếu có)
- [ ] Test lại các scripts mới

## Hỗ trợ

Nếu gặp vấn đề trong quá trình migration, vui lòng kiểm tra:
1. Đã cài đặt tất cả dependencies: `pip install -r requirements.txt`
2. Đã cài đặt package: `pip install -e .`
3. Đã đổi tên các thư mục cần thiết

