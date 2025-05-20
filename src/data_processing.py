import pandas as pd
import os 
import argparse
from pyvi.ViTokenizer import tokenize
import logging
from sklearn.model_selection import train_test_split


def process_context(context) : 
    return context.replace("['",'').replace("']",'')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data")
    parser.add_argument("--raw_path", type=str, default="data/raw", help="Folder contains raw data")
    parser.add_argument("--processed_path", type=str, default="data/processed", help="Folder to save processed data")
    parser.add_argument("--eval_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=28)

    args = parser.parse_args()
    raw_path = args.raw_path 
    proc_path = args.processed_path 
    eval_size = args.eval_size 
    random_state = args.random_state

    # load data 
    for dirpath, _, filenames in os.walk(raw_path): 
        if "train.csv" in filenames: 
            train_path = os.path.join(dirpath, 'train.csv')
        if "corpus.csv" in filenames: 
            corpus_path = os.path.join(dirpath, 'corpus.csv') 
        train_df = pd.read_csv(train_path, encoding= 'utf-8') 
        corpus_df = pd.read_csv(corpus_path, encoding= 'utf-8') 
        
    # processing train dataframe 
    train_question = train_df['question'].tolist() 
    train_answer = train_df['context'].tolist() 
    raw_cid = train_df['cid'].tolist() 
    text = corpus_df['text'].tolist()
    corpus_cid = corpus_df['cid'].tolist() 
    train_cid = [] 

    for cids in raw_cid: 
        cids = cids.strip('[]').split() 
        train_cid.append([int(cid) for cid in cids])

    processed_train = [] 
    cid_to_index = {cid: ind for ind,cid in enumerate(corpus_cid)}
    for ind, cids in enumerate(train_cid): 
        for cid in cids: 
            if cid in corpus_cid: 
                processed_train.append([train_question[ind], text[cid_to_index[cid]], cids]) 
            else: 
                processed_train.append([train_question[ind], process_context(train_answer[ind], cids)])
    seg_processed_train = [[tokenize(ques),tokenize(answer),cid] for ques,answer,cid in processed_train]
    train, eval = train_test_split(seg_processed_train, test_size=eval_size, random_state=random_state) 

    # save train and eval data 
    train_df = pd.DataFrame(train,columns=['question','answer','cid'])
    saved_train_path = os.path.join(proc_path, 'train.csv')
    train_df.to_csv(saved_train_path, index=False)
    saved_eval_path = os.path.join(proc_path, 'eval.csv')
    eval_df = pd.DataFrame(eval,columns=['question','answer','cid'])
    eval_df.to_csv(saved_eval_path,index=False)

    # processing corpus dataframe
    corpus_df['text'] = corpus_df['text'].apply(tokenize)
    saved_corpus_path = os.path.join(proc_path, 'corpus.csv')
    corpus_df.to_csv(saved_corpus_path, index=False)




      