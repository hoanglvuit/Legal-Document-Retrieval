"""Interactive inference with custom questions"""

import argparse
from legal_retrieval.inference.pipeline import InferencePipeline
import yaml

with open('configs/inference_config.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run with a legal question")
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--database', type=str, default='data/database')
    parser.add_argument('--bi_path', type=str, default=data['bi_model_path'])
    parser.add_argument('--cross_path', type=str, default=data['cross_model_path'])
    parser.add_argument('--corpus_path', type=str, default=data['corpus_path'])
    parser.add_argument('--top_k_retrieval', type=int, default=data['retrieval']['top_k'])
    parser.add_argument('--top_k_rerank', type=int, default=data['reranking']['top_k'])

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = InferencePipeline(
        bi_model_path=args.bi_path,
        cross_model_path=args.cross_path,
        database_path=args.database
    )

    # Load corpus
    pipeline.load_corpus(args.corpus_path)

    # Predict
    final_answers = pipeline.predict(
        question=args.question,
        top_k_retrieval=args.top_k_retrieval,
        top_k_rerank=args.top_k_rerank
    )

    # Print results
    for ind, answer in enumerate(final_answers):
        print(f"Top {ind+1}: {answer}")
        print('-----------------------------------')

