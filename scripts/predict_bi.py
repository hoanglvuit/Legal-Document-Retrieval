"""Run Bi-Encoder prediction"""

import argparse
import os
from legal_retrieval.inference.retriever import Retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction using the Bi Encoder model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, default="data/processed/corpus.csv")
    parser.add_argument("--train", action="store_true", help="Whether to evaluate on training data")
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv")
    parser.add_argument("--eval_path", type=str, default="data/processed/eval.csv")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output_folder", type=str, default='results/BiEncoder/model1')

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    retriever = Retriever(args.model_path)

    # Load corpus once
    documents, cids, answer_embeddings = retriever.load_corpus(args.corpus_path)

    # Predict on train if requested
    if args.train:
        retriever.predict(
            data_path=args.train_path,
            corpus_path=args.corpus_path,
            top_k=args.top_k,
            output_folder=args.output_folder,
            name='train'
        )

    # Predict on eval
    retriever.predict(
        data_path=args.eval_path,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
        output_folder=args.output_folder,
        name='eval'
    )

