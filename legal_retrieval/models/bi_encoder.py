"""Bi-Encoder model wrapper"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union


class BiEncoderModel:
    """Wrapper for Bi-Encoder model"""
    
    def __init__(self, model_path: str, device: Union[str, torch.device] = None):
        """
        Initialize Bi-Encoder model
        
        Args:
            model_path: Path to the model or HuggingFace model name
            device: Device to run the model on (cuda/cpu)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.model = SentenceTransformer(model_path).to(device)
    
    def encode(self, texts: List[str], **kwargs):
        """
        Encode texts to embeddings
        
        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments for SentenceTransformer.encode()
        
        Returns:
            Encoded embeddings
        """
        default_kwargs = {
            'show_progress_bar': True,
            'convert_to_tensor': True,
            'device': self.device
        }
        default_kwargs.update(kwargs)
        return self.model.encode(texts, **default_kwargs)
    
    def save(self, path: str):
        """Save the model to a path"""
        self.model.save_pretrained(path)

