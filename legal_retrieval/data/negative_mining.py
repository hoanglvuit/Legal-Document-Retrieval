"""Negative example mining for Cross-Encoder training"""

import random
from typing import List


class NegativeMiner:
    """Generate negative examples for training"""
    
    @staticmethod
    def random_negative(pred_cids: List[List[int]], true_cids: List[List[int]],
                       neg_num: int, seed: int = 28) -> List[List[int]]:
        """
        Generate random negative examples from predictions
        
        Args:
            pred_cids: Predicted context IDs for each question
            true_cids: True context IDs for each question
            neg_num: Number of negative examples per question
            seed: Random seed
        
        Returns:
            List of negative context IDs for each question
        """
        random.seed(seed)
        assert len(pred_cids) == len(true_cids), "Must same length"
        
        neg_cids = []
        for pred_cid, true_cid in zip(pred_cids, true_cids):
            neg_cid = [cid for cid in pred_cid if cid not in true_cid]
            neg_cid = random.sample(neg_cid, min(neg_num, len(neg_cid)))
            neg_cids.append(neg_cid)
        
        return neg_cids
    
    @staticmethod
    def hard_negative(pred_cids: List[List[int]], true_cids: List[List[int]],
                     neg_num: int) -> List[List[int]]:
        """
        Generate hard negative examples (top-ranked false positives)
        
        Args:
            pred_cids: Predicted context IDs for each question
            true_cids: True context IDs for each question
            neg_num: Number of negative examples per question
        
        Returns:
            List of hard negative context IDs for each question
        """
        assert len(pred_cids) == len(true_cids), "Must same length"
        
        neg_cids = []
        for pred_cid, true_cid in zip(pred_cids, true_cids):
            neg_cid = [cid for cid in pred_cid if cid not in true_cid]
            neg_cid = neg_cid[:neg_num]
            neg_cids.append(neg_cid)
        
        return neg_cids
    
    @staticmethod
    def create_negative_examples(questions: List[str], neg_cids: List[List[int]],
                                documents: List[str], cids: List[int]) -> List[List[str]]:
        """
        Create negative example pairs from negative CIDs
        
        Args:
            questions: List of questions
            neg_cids: List of negative context IDs for each question
            documents: List of all documents
            cids: List of all context IDs
        
        Returns:
            List of [question, answer] pairs
        """
        assert len(questions) == len(neg_cids), "Must same length"
        
        cid_to_index = {cid: ind for ind, cid in enumerate(cids)}
        negative_examples = []
        
        for question, neg_cid_list in zip(questions, neg_cids):
            for n_cid in neg_cid_list:
                if n_cid in cid_to_index:
                    negative_examples.append([
                        question,
                        documents[cid_to_index[n_cid]]
                    ])
        
        return negative_examples

