import argparse
import os
import ast
import json
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils import get_candidate


def load_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    questions = df['question'].tolist()
    cids = df['cid'].apply(ast.literal_eval).tolist()
    return questions, cids


def encode_texts(model, texts, device):
    return model.encode(texts, show_progress_bar=True, convert_to_tensor=True, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction using the Bi Encoder model")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--corpus_path", type=str, default="data/processed/corpus.csv")
    parser.add_argument("--train", action="store_true", help="Whether to evaluate on training data")
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv")
    parser.add_argument("--eval_path", type=str, default="data/processed/eval.csv")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--saved_folder", type=str, default='result/BiEncoder/model1')

    args = parser.parse_args()
    os.makedirs(args.saved_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model_path).to(device)

    # Load and encode corpus
    corpus_df = pd.read_csv(args.corpus_path, encoding='utf-8')
    documents = corpus_df['text'].tolist()
    cids = corpus_df['cid'].tolist()
    answer_embeddings = encode_texts(model, documents, device)

    results = {}

    # Load and encode train
    if args.train:
        train_questions, train_cids = load_data(args.train_path)
        train_embedding = encode_texts(model, train_questions, device)
        train_pred = get_candidate(train_embedding, answer_embeddings, cids, args.top_k, args.saved_folder, 'train')

    # Load and encode eval
    eval_questions, eval_cids = load_data(args.eval_path)
    eval_embedding = encode_texts(model, eval_questions, device) 
    eval_pred = get_candidate(eval_embedding, answer_embeddings, cids, args.top_k, args.saved_folder, 'eval')