"""End-to-end inference pipeline"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
from ..models.bi_encoder import BiEncoderModel
from ..models.cross_encoder import CrossEncoderModel


class InferencePipeline:
    """End-to-end inference pipeline combining Bi-Encoder and Cross-Encoder"""
    
    def __init__(self, bi_model_path: str, cross_model_path: str,
                 database_path: Optional[str] = None, device: Optional[torch.device] = None):
        """
        Initialize inference pipeline
        
        Args:
            bi_model_path: Path to Bi-Encoder model
            cross_model_path: Path to Cross-Encoder model
            database_path: Path to embeddings database (optional)
            device: Device to run on
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.bi_model = BiEncoderModel(bi_model_path, device)
        self.cross_model = CrossEncoderModel(cross_model_path, max_length=256, use_half=True)
        self.database_path = database_path
        self.corpus = None
        self.documents = None
        self.answer_embeddings = None
    
    def load_corpus(self, corpus_path: str):
        """
        Load corpus and optionally load/save embeddings database
        
        Args:
            corpus_path: Path to corpus CSV file
        """
        corpus = pd.read_csv(corpus_path, encoding='utf-8')
        self.corpus = corpus
        self.documents = corpus['text'].tolist()
        
        # Load or create embeddings database
        if self.database_path and os.path.isdir(self.database_path):
            db_file = os.path.join(self.database_path, 'database.npy')
            if os.path.exists(db_file):
                self.answer_embeddings = np.load(db_file)
                print(f"Loaded embeddings database from {db_file}")
            else:
                self._create_database()
        else:
            self._create_database()
    
    def _create_database(self):
        """Create embeddings database"""
        print("Creating embeddings database...")
        embeddings = self.bi_model.encode(self.documents)
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        self.answer_embeddings = embeddings
        
        # Save if database path provided
        if self.database_path:
            os.makedirs(self.database_path, exist_ok=True)
            db_file = os.path.join(self.database_path, 'database.npy')
            np.save(db_file, embeddings)
            print(f"Saved embeddings database to {db_file}")
    
    def predict(self, question: str, top_k_retrieval: int = 50, top_k_rerank: int = 5) -> List[str]:
        """
        Predict top documents for a question
        
        Args:
            question: Question string
            top_k_retrieval: Number of candidates to retrieve
            top_k_rerank: Number of final results to return
        
        Returns:
            List of top document texts
        """
        if self.documents is None or self.answer_embeddings is None:
            raise ValueError("Corpus not loaded. Call load_corpus() first.")
        
        # Retrieve with Bi-Encoder
        query_embedding = self.bi_model.encode([question])
        
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        similarities = cosine_similarity(query_embedding, self.answer_embeddings)[0]
        top_inds = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k_retrieval]
        retrieval_docs = [self.documents[i] for i in top_inds]
        
        # Rerank with Cross-Encoder
        pairs = [[question, doc] for doc in retrieval_docs]
        scores = self.cross_model.predict(pairs)
        
        top_5inds = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_rerank]
        final_answers = [retrieval_docs[i] for i in top_5inds]
        
        return final_answers
    
    def predict_batch(self, questions: List[str], top_k_retrieval: int = 50,
                     top_k_rerank: int = 5) -> List[List[str]]:
        """
        Predict for multiple questions
        
        Args:
            questions: List of question strings
            top_k_retrieval: Number of candidates to retrieve
            top_k_rerank: Number of final results to return
        
        Returns:
            List of top document texts for each question
        """
        results = []
        for question in questions:
            results.append(self.predict(question, top_k_retrieval, top_k_rerank))
        return results

