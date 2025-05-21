import argparse
import os
import ast
import json
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from .utils import get_candidate, exist_m, mrr_m


def load_data(path):
    df = pd.read_csv(path, encoding='utf-8')
    questions = df['question'].tolist()
    cids = df['cid'].apply(ast.literal_eval).tolist()
    return questions, cids


def encode_texts(model, texts, device):
    return model.encode(texts, show_progress_bar=True, convert_to_tensor=True, device=device)


def evaluate(model, questions, answer_embeddings, cids, true_cids, saved_folder, tag, k_exist, k_mrr, top_k):
    embeddings = encode_texts(model, questions, device)
    predictions = get_candidate(embeddings, answer_embeddings, cids, top_k, saved_folder, tag)
    exist_score = exist_m(predictions, true_cids, k_exist)
    mrr_score = mrr_m(predictions, true_cids, k_mrr)
    return exist_score, mrr_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for Bi Encoder model")
    parser.add_argument("--model_path", type=str, default="../saved_model/BiEncoder/model1/best")
    parser.add_argument("--corpus_path", type=str, default="../data/processed/corpus.csv")
    parser.add_argument("--train", action="store_true", help="Whether to evaluate on training data")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_data", type=str, default="../data/processed/eval.csv")
    parser.add_argument("--k_exist", type=int, default=90)
    parser.add_argument("--k_mrr", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("saved_folder", type=str, default='../result/BiEncoder/model1')

    args = parser.parse_args()
    os.makedirs(args.saved_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(args.model_path).to(device)

    # Load corpus
    corpus_df = pd.read_csv(args.corpus_path, encoding='utf-8')
    documents = corpus_df['text'].tolist()
    cids = corpus_df['cid'].tolist()
    answer_embeddings = encode_texts(model, documents, device)

    results = {}

    # Evaluate train
    if args.train:
        train_questions, train_cids = load_data(args.train_path)
        train_exist, train_mrr = evaluate(
            model, train_questions, answer_embeddings, cids, train_cids,
            args.saved_folder, 'train', args.k_exist, args.k_mrr, args.top_k
        )
        results['train'] = {"exist@{}".format(args.k_exist): train_exist,
                            "mrr@{}".format(args.k_mrr): train_mrr}

    # Evaluate eval
    eval_questions, eval_cids = load_data(args.eval_data)
    eval_exist, eval_mrr = evaluate(
        model, eval_questions, answer_embeddings, cids, eval_cids,
        args.saved_folder, 'eval', args.k_exist, args.k_mrr, args.top_k
    )
    results['eval'] = {"exist@{}".format(args.k_exist): eval_exist,
                       "mrr@{}".format(args.k_mrr): eval_mrr}

    # Save results to result.json
    result_path = os.path.join(args.saved_folder, "result.json")
    with open(result_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Evaluation results saved to", result_path)
