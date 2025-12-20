"""Process raw data into training format"""

import argparse
from legal_retrieval.data.processor import DataProcessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data")
    parser.add_argument("--raw_path", type=str, default="data/raw",
                       help="Folder contains raw data")
    parser.add_argument("--processed_path", type=str, default="data/processed",
                       help="Folder to save processed data")
    parser.add_argument("--eval_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=28)

    args = parser.parse_args()
    
    processor = DataProcessor()
    processor.process_pipeline(
        raw_path=args.raw_path,
        processed_path=args.processed_path,
        eval_size=args.eval_size,
        random_state=args.random_state
    )

