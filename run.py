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
    parser.add_argument('corpus_path', type=str, default= 'data/processed/corpus.csv')
    args = parser.parse_args()

    # load BiEncoder 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bi_model = SentenceTransformer(args.bi_path)

    # load data 
    corpus = pd.read_csv(args.corpus_path, encoding='utf-8') 
    documents = corpus['text'].tolist()

    # database 
    if os.path.isdir(args.database): 
        answer_embeddings = np.load(os.path.join(args.database, 'database.npy'))
    else: 
        os.makedirs(args.database) 
        answer_embeddings = bi_model.encode(documents, show_progress_bar=True, convert_to_tensor=True, device=device)
        np.save(os.path.join(args.database, 'database.npy'), answer_embeddings)

    query_embedding = bi_model.encode([args.question]) 
    similarities = cosine_similarity(query_embedding, answer_embeddings)[0]
    top_inds = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:50]
    retrieval_docs = [documents[i] for i in top_inds]
    
    # load Cross Encoder 
    cross_model = CrossEncoder(args.cross_path, max_length=256)
    cross_model.model.half()
    
    pairs = [[args.question, doc] for doc in retrieval_docs] 
    scores = cross_model.predict(pairs)

    top_5inds = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5] 
    final_answers = [retrieval_docs[i] for i in top_5inds] 

    for ind, answer in enumerate(final_answers): 
        print(f"Top {ind+1}: {answer}")
        print('-----------------------------------')