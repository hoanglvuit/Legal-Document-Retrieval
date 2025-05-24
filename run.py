from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch 
import pandas as pd 
import argparse 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Run with a legal question")
    parser.add_argument('--question', type=str) 
    parser.add_argument('--database', type=str, default= 'data/database') 
    parser.add_argument('--bi_path', type=str, default= 'saved_model/BiEncoder/model1/best')
    parser.add_argument('--cross_path', type=str, default= 'saved_model/CrossEncoder/model1')
    parser.add_argument('--corpus_path', type=str, default= 'data/processed/corpus.csv')
    args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
biencoder = SentenceTransformer('saved_model/BiEncoder/model1/best')
corpus_df = pd.read_csv('data/processed/corpus.csv', encoding='utf-8')
documents = corpus_df['text'].tolist()
answer_embeddings = biencoder.encode(documents, show_progress_bar=True, convert_to_tensor=True, device=device)
question_embedding = biencoder.encode(args.question, show_progress_bar=True, convert_to_tensor=True, device=device)
sim = util.cos_sim(question_embedding, answer_embeddings)
sim_cpu = sim.cpu().tolist() 
top_90_indices = np.argsort(sim_cpu)[-90:][::-1]






