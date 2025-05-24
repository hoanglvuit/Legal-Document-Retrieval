from sentence_transformers import CrossEncoder
import os
import argparse 
import pandas as pd 
from tqdm import tqdm
from src.utils import get_top_cids

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Reranking by Cross Encoder") 
    parser.add_argument('--model', type=str, default= 'saved_model/CrossEncoder/model1')
    parser.add_argument('--pred_path', type=str, default= 'result/BiEncoder/model1/outputeval.txt')
    parser.add_argument('--eval_path', type=str, default='data/processed/eval.csv')
    parser.add_argument('--corpus_path', type=str, default='data/processed/corpus.csv')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--output_folder', type=str, default= 'result/CrossEncoder/model1') 

    args = parser.parse_args() 
    os.makedirs(args.output_folder, exist_ok=True) 

    # load data 
    with open(args.pred_path, 'r') as file: 
        pred_cids = [] 
        for line in file: 
            pred_cids.append([int(x) for x in line.strip().split()][:args.num])
    
    questions  = pd.read_csv(args.eval_path, encoding= 'utf-8')['question'].tolist() 
    corpus_df = pd.read_csv(args.corpus_path, encoding= 'utf-8') 
    documents = corpus_df['text'].tolist() 
    cids = corpus_df['cid'].tolist() 
    cid_to_index = {cid:ind for ind,cid in enumerate(cids)} 

    # load model 
    model = CrossEncoder(args.model, max_length=256) 
    model.model.half()

    # implement 
    score_pred = []
    assert len(questions) == len(pred_cids), "Must same length" 
    for question, pred_cid in tqdm(zip(questions, pred_cids), total=len(questions)):
        docs = [documents[cid_to_index[i]] for i in pred_cid] 
        pairs = [[question, doc] for doc in docs] 
        scores = model.predict(pairs) 
        score_pred.append(scores) 
    
    top_cids = get_top_cids(score_pred, args.num, cids) 
    
    # save 
    with open(os.path.join(args.output_folder,'output.txt'), "w") as f:
        for sublist in top_cids:
            line = ' '.join(map(str, sublist)) 
            f.write(line + '\n')  

