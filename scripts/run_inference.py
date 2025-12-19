"""Interactive inference with custom questions"""

import argparse
from legal_retrieval.inference.pipeline import InferencePipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run with a legal question")
    parser.add_argument('--question', type=str, required=True)
    parser.add_argument('--database', type=str, default='data/database')
    parser.add_argument('--bi_path', type=str, default='saved_models/BiEncoder/model1/best')
    parser.add_argument('--cross_path', type=str, default='saved_models/CrossEncoder/model1')
    parser.add_argument('--corpus_path', type=str, default='data/processed/corpus.csv')
    parser.add_argument('--top_k_retrieval', type=int, default=50)
    parser.add_argument('--top_k_rerank', type=int, default=5)

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

