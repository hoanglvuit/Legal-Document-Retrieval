# API Documentation

## Models

### BiEncoderModel

Wrapper for Bi-Encoder model.

```python
from legal_retrieval.models import BiEncoderModel

model = BiEncoderModel(model_path="path/to/model")
embeddings = model.encode(["text1", "text2"])
```

### CrossEncoderModel

Wrapper for Cross-Encoder model.

```python
from legal_retrieval.models import CrossEncoderModel

model = CrossEncoderModel(model_path="path/to/model")
scores = model.predict([["query", "document"]])
```

## Data Processing

### DataProcessor

Process raw data into training format.

```python
from legal_retrieval.data import DataProcessor

processor = DataProcessor()
processor.process_pipeline(
    raw_path="data/raw",
    processed_path="data/processed"
)
```

### NegativeMiner

Generate negative examples for training.

```python
from legal_retrieval.data import NegativeMiner

neg_cids = NegativeMiner.random_negative(pred_cids, true_cids, neg_num=3)
```

## Training

### BiEncoderTrainer

Train Bi-Encoder model.

```python
from legal_retrieval.training import BiEncoderTrainer

trainer = BiEncoderTrainer()
trainer.train(
    data_folder="data/processed",
    output_folder="saved_models/BiEncoder/model1"
)
```

### CrossEncoderTrainer

Train Cross-Encoder model.

```python
from legal_retrieval.training import CrossEncoderTrainer

trainer = CrossEncoderTrainer()
trainer.train(
    pos_path="data/processed/train.csv",
    neg_path="data/processed/3_moderateneg_ex.csv",
    output_folder="saved_models/CrossEncoder/model1"
)
```

## Inference

### Retriever

Retrieve candidates using Bi-Encoder.

```python
from legal_retrieval.inference import Retriever

retriever = Retriever(model_path="path/to/bi/model")
predictions = retriever.predict(
    data_path="data/processed/eval.csv",
    corpus_path="data/processed/corpus.csv"
)
```

### Reranker

Rerank candidates using Cross-Encoder.

```python
from legal_retrieval.inference import Reranker

reranker = Reranker(model_path="path/to/cross/model")
reranked = reranker.predict(
    pred_path="results/BiEncoder/model1/outputeval.txt",
    eval_path="data/processed/eval.csv",
    corpus_path="data/processed/corpus.csv"
)
```

### InferencePipeline

End-to-end inference pipeline.

```python
from legal_retrieval.inference import InferencePipeline

pipeline = InferencePipeline(
    bi_model_path="saved_models/BiEncoder/model1/best",
    cross_model_path="saved_models/CrossEncoder/model1"
)
pipeline.load_corpus("data/processed/corpus.csv")
results = pipeline.predict("Câu hỏi pháp luật?")
```

## Evaluation

### Evaluator

Evaluate model predictions.

```python
from legal_retrieval.evaluation import Evaluator

evaluator = Evaluator()
scores = evaluator.evaluate(
    pred_path="results/CrossEncoder/model1/output.txt",
    true_path="data/processed/eval.csv"
)
```

