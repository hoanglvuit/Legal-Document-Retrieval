import argparse
import pandas as pd
import os
import ast
from src.get_negative import ran_negative, hard_negative


def load_data(data_folder): 
    for dirpath, dirname, filenames in os.walk(data_folder):
        if 'train.csv' in filenames: 
            train_df = pd.read_csv(os.path.join(args.data_folder, 'train.csv'), encoding='utf-8') 
            train_question = train_df['question'].tolist() 
            train_cids = train_df['cid'].apply(ast.literal_eval).tolist() 
        if 'corpus.csv' in filenames: 
            corpus_df = pd.read_csv(os.path.join(args.data_folder, 'corpus.csv'), encoding='utf-8')
            document = corpus_df['text'].tolist() 
            cids = corpus_df['cid'].tolist() 
    return train_question, train_cids, document, cids 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Negative example mining, create negative csv fro cross train") 
    parser.add_argument('--type', type=str, default='moderate', choices=['moderate', 'hard', 'easy']) 
    parser.add_argument('--data_folder', type=str, default='data/processed') 
    parser.add_argument('--pred_file', type=str, default='result/BiEncoder/model1/outputtrain.txt')
    parser.add_argument('--neg_num', type=int, default=3,help= "The number of negative example need")
    parser.add_argument('--seed', type=int, default=22520465)

    args = parser.parse_args()

    # load_data
    train_question, train_cids, document, cids = load_data(args.data_folder) 

    if args.type == 'easy': 
        neg_cids = ran_negative(cids, train_cids, args.neg_num, seed=args.seed)
    else: 
        with open(args.pred_file, 'r') as file: 
            pred_cids = [] 
            for line in file: 
                pred_cids.append([int(x) for x in line.strip().split()])
        if args.type == 'hard': 
            neg_cids = hard_negative(pred_cids, train_cids, args.neg_num) 
        else: 
            neg_cids = ran_negative(pred_cids, train_cids, args.neg_num, args.seed) 
    
    # create negative example
    assert len(train_question) == len(neg_cids), "Must same length" 
    cid_to_index = {cid: ind for ind,cid in enumerate(cids)}
    negative_example = [] 
    for train_ques, neg_cid in zip(train_question, neg_cids): 
        for n_cid in neg_cid: 
            negative_example.append([train_ques,document[cid_to_index[n_cid]]]) 
    
    # Save
    df = pd.DataFrame(negative_example, columns=['question', 'answer'])
    out_dir = os.path.join(args.data_folder, f'{args.neg_num}_{args.type}neg_ex.csv')
    df.to_csv(out_dir, index=False) 