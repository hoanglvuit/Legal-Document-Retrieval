"""Evaluation runner"""

import os
import json
import pandas as pd
import ast
from typing import List
from .metrics import exist_m, mrr_m


class Evaluator:
    """Evaluate model predictions"""
    
    @staticmethod
    def load_predictions(pred_path: str) -> List[List[int]]:
        """
        Load predictions from text file
        
        Args:
            pred_path: Path to prediction file
        
        Returns:
            List of predicted context ID lists
        """
        prediction = []
        with open(pred_path, 'r') as file:
            for line in file:
                preds = list(map(int, line.strip().split()))
                prediction.append(preds)
        return prediction
    
    @staticmethod
    def load_true_labels(true_path: str) -> List[List[int]]:
        """
        Load true labels from CSV file
        
        Args:
            true_path: Path to CSV file with true labels
        
        Returns:
            List of true context ID lists
        """
        df = pd.read_csv(true_path, encoding='utf-8')
        true_cids = df['cid'].apply(ast.literal_eval).tolist()
        return true_cids
    
    @staticmethod
    def evaluate(pred_path: str, true_path: str, top_e: int = 90, top_m: int = 10,
                output_path: str = None) -> dict:
        """
        Evaluate predictions and return scores
        
        Args:
            pred_path: Path to prediction file
            true_path: Path to true labels file
            top_e: Number of top candidates for Exist@k
            top_m: Number of top candidates for MRR@k
            output_path: Path to save scores (optional)
        
        Returns:
            Dictionary with evaluation scores
        """
        prediction = Evaluator.load_predictions(pred_path)
        true_cids = Evaluator.load_true_labels(true_path)
        
        exist_score = exist_m(prediction, true_cids, top_e)
        mrr_score = mrr_m(prediction, true_cids, top_m)
        
        dict_score = {
            f'exist@{top_e}': exist_score,
            f'mrr@{top_m}': mrr_score
        }
        
        if output_path is None:
            output_path = os.path.join(os.path.dirname(os.path.abspath(pred_path)), 'score.json')
        
        # Load existing scores if file exists
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                old = json.load(f)
        else:
            old = {}
        
        old.update(dict_score)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(old, f, ensure_ascii=False, indent=2)
        
        return dict_score

