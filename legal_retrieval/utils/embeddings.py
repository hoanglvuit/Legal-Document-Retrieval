"""Embedding utility functions"""

import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer


def encode_texts(model: Union[SentenceTransformer, object], texts: List[str],
                device: Union[str, torch.device] = None, **kwargs) -> torch.Tensor:
    """
    Encode texts to embeddings
    
    Args:
        model: SentenceTransformer model or model with encode method
        texts: List of texts to encode
        device: Device to run encoding on
        **kwargs: Additional arguments for encode method
    
    Returns:
        Encoded embeddings tensor
    """
    default_kwargs = {
        'show_progress_bar': True,
        'convert_to_tensor': True,
        'device': device
    }
    default_kwargs.update(kwargs)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    default_kwargs['device'] = device
    
    return model.encode(texts, **default_kwargs)

