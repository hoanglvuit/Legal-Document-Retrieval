"""Cross-Encoder model wrapper"""

import torch
from sentence_transformers import CrossEncoder
from typing import List, Union


class CrossEncoderModel:
    """Wrapper for Cross-Encoder model"""
    
    def __init__(self, model_path: str, max_length: int = 256, use_half: bool = True):
        """
        Initialize Cross-Encoder model
        
        Args:
            model_path: Path to the model or HuggingFace model name
            max_length: Maximum sequence length
            use_half: Whether to use half precision (FP16)
        """
        self.model = CrossEncoder(model_path, max_length=max_length)
        if use_half:
            self.model.model.half()
    
    def predict(self, pairs: List[List[str]], **kwargs):
        """
        Predict scores for query-document pairs
        
        Args:
            pairs: List of [query, document] pairs
            **kwargs: Additional arguments for CrossEncoder.predict()
        
        Returns:
            List of scores
        """
        return self.model.predict(pairs, **kwargs)
    
    def save(self, path: str):
        """Save the model to a path"""
        self.model.save_pretrained(path)

