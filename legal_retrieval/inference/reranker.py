"""Cross-Encoder reranker"""

import os
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
import logging
from ..models.cross_encoder import CrossEncoderModel


class Reranker:
    """Reranker using Cross-Encoder"""
    
    def __init__(self, model_path: str, max_length: int = 256, use_half: bool = True):
        """
        Initialize reranker
        
        Args:
            model_path: Path to Cross-Encoder model
            max_length: Maximum sequence length
            use_half: Whether to use half precision
        """
        self.model = CrossEncoderModel(model_path, max_length=max_length, use_half=use_half)
        logging.disable(logging.WARNING)
    
    def rerank(self, questions: List[str], pred_cids: List[List[int]],
              documents: List[str], cids: List[int], top_k: int = 100) -> List[List[int]]:
        """
        Rerank candidates using Cross-Encoder
        
        Args:
            questions: List of questions
            pred_cids: List of predicted context IDs for each question
            documents: List of all documents
            cids: List of all context IDs
            top_k: Number of top results to return
        
        Returns:
            List of reranked context IDs for each question
        """
        cid_to_index = {cid: ind for ind, cid in enumerate(cids)}
        
        score_pred = []
        assert len(questions) == len(pred_cids), "Must same length"
        
        for question, pred_cid_list in tqdm(zip(questions, pred_cids), total=len(questions)):
            docs = [documents[cid_to_index[i]] for i in pred_cid_list]
            pairs = [[question, doc] for doc in docs]
            scores = self.model.predict(pairs)
            score_pred.append(scores)
        
        # Sort by scores
        sorted_cids = []
        for cid_list, score_list in zip(pred_cids, score_pred):
            sorted_pair = sorted(zip(cid_list, score_list), key=lambda x: x[1], reverse=True)
            sorted_cids.append([cid for cid, score in sorted_pair[:top_k]])
        
        return sorted_cids
    
    def predict(self, pred_path: str, eval_path: str, corpus_path: str,
               top_k: int = 100, output_folder: Optional[str] = None):
        """
        Rerank predictions from file
        
        Args:
            pred_path: Path to prediction file from Bi-Encoder
            eval_path: Path to evaluation CSV with questions
            corpus_path: Path to corpus CSV
            top_k: Number of top results
            output_folder: Folder to save results
        
        Returns:
            List of reranked predictions
        """
        # Load predictions
        with open(pred_path, 'r') as file:
            pred_cids = []
            for line in file:
                pred_cids.append([int(x) for x in line.strip().split()])
        
        # Load questions
        questions = pd.read_csv(eval_path, encoding='utf-8')['question'].tolist()
        
        # Load corpus
        corpus_df = pd.read_csv(corpus_path, encoding='utf-8')
        documents = corpus_df['text'].tolist()
        cids = corpus_df['cid'].tolist()
        
        # Rerank
        sorted_cids = self.rerank(questions, pred_cids, documents, cids, top_k)
        
        # Save if output folder provided
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            with open(os.path.join(output_folder, 'output.txt'), "w") as f:
                for sublist in sorted_cids:
                    line = ' '.join(map(str, sublist))
                    f.write(line + '\n')
        
        return sorted_cids

