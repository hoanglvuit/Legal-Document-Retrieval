import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from src.utils import get_top_cids
from rank_bm25 import BM25Okapi,BM25L,BM25Plus


def load_data(data_folder): 
    for dirpath, dirname, filenames in os.walk(data_folder): 
        if 'corpus.csv' in filenames: 
            corpus_df = pd.read_csv(os.path.join(dirpath, 'corpus.csv'), encoding='utf-8') 
            document = corpus_df['text'].tolist()
            cids = corpus_df['cid'].tolist()
        if 'eval.csv' in filenames: 
            eval_df = pd.read_csv(os.path.join(dirpath, 'eval.csv'), encoding = 'utf-8', nrows=2000) 
            eval_question = eval_df['question'].tolist()
            eval_cid = eval_df['cid'].tolist()
    return document, cids, eval_question, eval_cid

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Use BM25 for retrieval') 
    parser.add_argument("--model", type=str, default="bm25o", choices=["bm25o", "bm25l", "bm25plus"])  
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75) 
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.5) 
    parser.add_argument("--data_folder", type=str, default='data/processed')
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args() 
    os.makedirs(args.output_folder, exist_ok=True)

    # load data
    document, cids, eval_question, eval_cid = load_data(args.data_folder) 

    # load model 
    tokenized_corpus = [[token.lower() for token in doc.split(" ")] for doc in document] 
    if args.model == 'bm25o': 
        model = BM25Okapi(tokenized_corpus, k1= args.k1, b= args.b, epsilon= args.epsilon) 
    elif args.model == 'bm25l': 
        model = BM25L(tokenized_corpus, k1= args.k1, b= args.b, delta= args.delta)
    elif args.model == 'bm25plus': 
        model = BM25Plus(tokenized_corpus, k1= args.k1, b= args.b, delta= args.delta) 
    
    # query 
    prediction = []
    for query in tqdm(eval_question, desc="Processing queries"):
        tokenized_query = [token.lower() for token in query.split(" ")] 
        scores = list(model.get_scores(tokenized_query))
        prediction.append(scores)

    # save
    top_cids = get_top_cids(prediction, args.top_k, cids) 
    output_path = os.path.join(args.output_folder, 'output.txt')
    with open(output_path, "w") as file:
        file.writelines(" ".join(map(str, sublist)) + "\n" for sublist in top_cids)

    config_path = os.path.join(args.output_folder, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


