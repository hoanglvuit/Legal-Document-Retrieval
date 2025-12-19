"""Bi-Encoder retriever"""

import torch
import pandas as pd
import ast
from typing import List, Optional
from ..models.bi_encoder import BiEncoderModel
from ..utils.helpers import get_candidate
from ..utils.embeddings import encode_texts


class Retriever:
    """Retriever using Bi-Encoder"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialize retriever
        
        Args:
            model_path: Path to Bi-Encoder model
            device: Device to run on
        """
        self.model = BiEncoderModel(model_path, device)
        self.device = self.model.device
    
    def load_corpus(self, corpus_path: str):
        """
        Load and encode corpus
        
        Args:
            corpus_path: Path to corpus CSV file
        
        Returns:
            documents, cids, answer_embeddings
        """
        corpus_df = pd.read_csv(corpus_path, encoding='utf-8')
        documents = corpus_df['text'].tolist()
        cids = corpus_df['cid'].tolist()
        answer_embeddings = encode_texts(self.model.model, documents, device=self.device)
        
        return documents, cids, answer_embeddings
    
    def retrieve(self, questions: List[str], answer_embeddings: torch.Tensor,
                cids: List[int], top_k: int = 100, output_folder: Optional[str] = None,
                name: str = "", batch_size: int = 512) -> List[List[int]]:
        """
        Retrieve top-k candidates for questions
        
        Args:
            questions: List of questions
            answer_embeddings: Pre-computed answer embeddings
            cids: List of context IDs
            top_k: Number of top candidates to retrieve
            output_folder: Folder to save results (optional)
            name: Suffix for output filename
            batch_size: Batch size for processing
        
        Returns:
            List of top-k candidate CIDs for each question
        """
        question_embeddings = encode_texts(self.model.model, questions, device=self.device)
        
        if output_folder:
            return get_candidate(
                question_embeddings, answer_embeddings, cids, top_k,
                output_folder, name, batch_size
            )
        else:
            # Return without saving
            from ..utils.helpers import get_top_cids
            from sentence_transformers import util
            
            similarities = util.cos_sim(question_embeddings, answer_embeddings)
            similarities_list = similarities.cpu().tolist()
            return get_top_cids(similarities_list, top_k, cids)
    
    def predict(self, data_path: str, corpus_path: str, top_k: int = 100,
               output_folder: Optional[str] = None, name: str = ""):
        """
        Predict on a dataset
        
        Args:
            data_path: Path to CSV file with questions
            corpus_path: Path to corpus CSV file
            top_k: Number of top candidates
            output_folder: Folder to save results
            name: Suffix for output filename
        
        Returns:
            List of predictions
        """
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        questions = df['question'].tolist()
        
        # Load corpus
        documents, cids, answer_embeddings = self.load_corpus(corpus_path)
        
        # Retrieve
        return self.retrieve(questions, answer_embeddings, cids, top_k, output_folder, name)

