"""Evaluate model predictions"""

import argparse
from legal_retrieval.evaluation.evaluator import Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate result.txt on exist and mrr metrics")
    parser.add_argument('--pred_path', type=str, required=True, help="Path to file .txt")
    parser.add_argument('--true_path', type=str, required=True, help="Path to file .csv")
    parser.add_argument('--top_e', type=int, default=90,
                       help='The number of top candidates to evaluate by exist_m')
    parser.add_argument('--top_m', type=int, default=10,
                       help='The number of top candidates to evaluate by mrr_m')

    args = parser.parse_args()

    scores = Evaluator.evaluate(
        pred_path=args.pred_path,
        true_path=args.true_path,
        top_e=args.top_e,
        top_m=args.top_m
    )
    
    print("Evaluation completed!")
    print(f"Results: {scores}")

