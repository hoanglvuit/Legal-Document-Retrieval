# Tóm tắt cấu trúc mới

## Cấu trúc thư mục

```
Legal-Document-Retrieval/
├── legal_retrieval/          # Package chính
│   ├── models/               # Model wrappers
│   ├── data/                 # Data processing
│   ├── training/            # Training utilities
│   ├── inference/           # Inference pipeline
│   ├── evaluation/          # Evaluation metrics
│   └── utils/               # Utilities
├── scripts/                  # Entry point scripts
├── configs/                 # Configuration files
├── data/                    # Data directories
├── saved_models/            # Trained models (renamed from saved_model)
├── results/                 # Results (renamed from result)
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Các thay đổi chính

1. **Package structure**: Code được tổ chức thành package `legal_retrieval` với các submodules rõ ràng
2. **Scripts**: Tất cả scripts được di chuyển vào `scripts/`
3. **Configs**: Thêm thư mục `configs/` cho các file cấu hình
4. **Naming**: `saved_model` → `saved_models`, `result` → `results`
5. **Documentation**: Thêm `docs/` và `tests/` directories

## Cách sử dụng

### Cài đặt
```bash
pip install -e .
```

### Import modules
```python
from legal_retrieval.models import BiEncoderModel, CrossEncoderModel
from legal_retrieval.inference import InferencePipeline
from legal_retrieval.training import BiEncoderTrainer
```

### Chạy scripts
```bash
python scripts/train_bi.py --data_folder data/processed
python scripts/predict_bi.py --model_path saved_models/BiEncoder/model1/best
```

## Lợi ích

1. **Tổ chức rõ ràng**: Code được phân chia theo chức năng
2. **Dễ maintain**: Mỗi module có trách nhiệm riêng
3. **Dễ test**: Có thể test từng module độc lập
4. **Dễ mở rộng**: Có thể thêm features mới dễ dàng
5. **Reusable**: Có thể import và sử dụng như một package

