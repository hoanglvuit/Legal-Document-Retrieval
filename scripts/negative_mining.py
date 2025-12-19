"""Generate negative examples for Cross-Encoder training"""

import argparse
import os
import pandas as pd
import ast
from legal_retrieval.data.negative_mining import NegativeMiner


def load_data(data_folder):
    """Load training data and corpus"""
    train_df = None
    corpus_df = None
    
    for dirpath, dirname, filenames in os.walk(data_folder):
        if 'train.csv' in filenames:
            train_df = pd.read_csv(os.path.join(dirpath, 'train.csv'), encoding='utf-8')
        if 'corpus.csv' in filenames:
            corpus_df = pd.read_csv(os.path.join(dirpath, 'corpus.csv'), encoding='utf-8')
    
    if train_df is None or corpus_df is None:
        raise FileNotFoundError("train.csv or corpus.csv not found")
    
    train_question = train_df['question'].tolist()
    train_cids = train_df['cid'].apply(ast.literal_eval).tolist()
    documents = corpus_df['text'].tolist()
    cids = corpus_df['cid'].tolist()
    
    return train_question, train_cids, documents, cids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Negative example mining, create negative csv for cross train")
    parser.add_argument('--type', type=str, default='moderate', choices=['moderate', 'hard', 'easy'])
    parser.add_argument('--data_folder', type=str, default='data/processed')
    parser.add_argument('--pred_file', type=str, default='results/BiEncoder/model1/outputtrain.txt')
    parser.add_argument('--neg_num', type=int, default=3, help="The number of negative example need")
    parser.add_argument('--seed', type=int, default=22520465)

    args = parser.parse_args()

    # Load data
    train_question, train_cids, documents, cids = load_data(args.data_folder)

    # Generate negative examples
    if args.type == 'easy':
        neg_cids = NegativeMiner.random_negative(cids, train_cids, args.neg_num, seed=args.seed)
    else:
        # Load predictions
        with open(args.pred_file, 'r') as file:
            pred_cids = []
            for line in file:
                pred_cids.append([int(x) for x in line.strip().split()])
        
        if args.type == 'hard':
            neg_cids = NegativeMiner.hard_negative(pred_cids, train_cids, args.neg_num)
        else:  # moderate
            neg_cids = NegativeMiner.random_negative(pred_cids, train_cids, args.neg_num, args.seed)

    # Create negative example pairs
    negative_examples = NegativeMiner.create_negative_examples(
        train_question, neg_cids, documents, cids
    )

    # Save
    df = pd.DataFrame(negative_examples, columns=['question', 'answer'])
    out_dir = os.path.join(args.data_folder, f'{args.neg_num}_{args.type}neg_ex.csv')
    df.to_csv(out_dir, index=False)
    print(f"Saved negative examples to {out_dir}")

