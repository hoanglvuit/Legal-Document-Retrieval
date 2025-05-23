import os 
import argparse 
import pandas as pd
import ast
import json
from src.utils import exist_m, mrr_m

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate result.txt on exist and mrr metrics")
    parser.add_argument('--pred_path', type=str, help="Path to file .txt") 
    parser.add_argument('--true_path', type=str, help="Path to file .csv")
    parser.add_argument('--top_e', type=int, default=90, help='The number of top candidates to evaluate by exist_m')
    parser.add_argument('--top_m', type=int, default=10, help='The number of top candidates to evaluate by mrr_m')

    args = parser.parse_args() 
    
    # load prediction
    prediction = []
    with open(args.pred_path, 'r') as file: 
        for line in file: 
            preds = list(map(int, line.strip().split()))
            prediction.append(preds) 
    
    #load true label
    df = pd.read_csv(args.true_path, encoding='utf-8') 
    true_cids = df['cid'].apply(ast.literal_eval).tolist() 

    # evaluate
    exist_score = exist_m(prediction, true_cids, args.top_e)
    mrr_m = mrr_m(prediction, true_cids, args.top_m)

    # save 
    dict_score = {
        f'exist@{args.top_e}': exist_score,
        f'mrr@{args.top_m}': mrr_m
    }

    path = os.path.join(os.path.dirname(os.path.abspath(args.pred_path)), 'score.json')

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            old = json.load(f)
    else:
        old = {}

    old.update(dict_score)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(old, f, ensure_ascii=False, indent=2)