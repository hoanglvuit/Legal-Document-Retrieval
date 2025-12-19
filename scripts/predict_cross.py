"""Run Cross-Encoder re-ranking"""

import argparse
from legal_retrieval.inference.reranker import Reranker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reranking by Cross Encoder")
    parser.add_argument('--model', type=str, default='saved_models/CrossEncoder/model1')
    parser.add_argument('--pred_path', type=str, default='results/BiEncoder/model1/outputeval.txt')
    parser.add_argument('--eval_path', type=str, default='data/processed/eval.csv')
    parser.add_argument('--corpus_path', type=str, default='data/processed/corpus.csv')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--output_folder', type=str, default='results/CrossEncoder/model1')

    args = parser.parse_args()

    reranker = Reranker(args.model)
    reranker.predict(
        pred_path=args.pred_path,
        eval_path=args.eval_path,
        corpus_path=args.corpus_path,
        top_k=args.num,
        output_folder=args.output_folder
    )

