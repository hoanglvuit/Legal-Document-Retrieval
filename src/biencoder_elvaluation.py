from sentence_transformers import SentenceTransformer
import argparse
import torch
import pandas as pd
from .utils im

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluation for Bi Encoder model") 
    parser.add_argument("--model_path", type=str, default="../saved_model/BiEncoder/model1/best")
    parser.add_argument("--corpus_path", type=str, default="../data/processed/corpus.csv") 
    parser.add_argument("--train", action= "store_true", help="Whether evaluate for train csv") 
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default="../data/processed/eval.csv")


args = parser.parse_args() 

#load model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(args.model_path).to(device) 

#load data 
corpus_df = pd.read_csv(args.corpus_path, encoding='utf-8') 
document = corpus_df['text'].tolist() 
cids = corpus_df['cid'].tolist() 

if args.train: 
    train_df = pd.read_csv(args.train_path, encoding='utf-8') 
    train_question = train_df['question'].tolist() 
    train_cid = train_df['cid'].tolist()
eval_df = pd.read_csv(args.eval_path, encoding='utf-8') 
eval_question = eval_df['question'].tolist()
eval_cid = eval_df['cid'].tolist() 

# encoding
train_question_embeddings = model.encode(train_question,show_process_bar=True,convert_to_tensor=True,device=device)
eval_question_embeddings = model.encode(eval_question,show_process_bar=True,convert_to_tensor=True,device=device)
answer_embeddings = model.encode(document,show_process_bar=True,convert_to_tensor=True,device=device)

# 

