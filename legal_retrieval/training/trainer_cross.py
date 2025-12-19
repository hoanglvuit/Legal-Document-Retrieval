"""Cross-Encoder trainer"""

import os
import json
import logging
import pandas as pd
from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader


class CrossEncoderTrainer:
    """Trainer for Cross-Encoder model"""
    
    def __init__(self, model_name: str = 'itdainb/PhoRanker', max_length: int = 256):
        """
        Initialize Cross-Encoder trainer
        
        Args:
            model_name: HuggingFace model name or path
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model = CrossEncoder(model_name, max_length=max_length)
        logging.disable(logging.WARNING)
    
    def load_data(self, pos_path: str, neg_path: str):
        """
        Load positive and negative examples
        
        Args:
            pos_path: Path to positive examples CSV
            neg_path: Path to negative examples CSV
        
        Returns:
            pos_question, pos_answer, neg_question, neg_answer
        """
        pos_df = pd.read_csv(pos_path, encoding='utf-8')
        neg_df = pd.read_csv(neg_path, encoding='utf-8')
        
        pos_question = pos_df['question'].tolist()
        pos_answer = pos_df['answer'].tolist()
        neg_question = neg_df['question'].tolist()
        neg_answer = neg_df['answer'].tolist()
        
        return pos_question, pos_answer, neg_question, neg_answer
    
    def train(self, pos_path: str, neg_path: str, output_folder: str,
             num_epochs: int = 2, batch_size: int = 64, learning_rate: float = 2e-5):
        """
        Train the Cross-Encoder model
        
        Args:
            pos_path: Path to positive examples CSV
            neg_path: Path to negative examples CSV
            output_folder: Folder to save the trained model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Load data
        pos_question, pos_answer, neg_question, neg_answer = self.load_data(pos_path, neg_path)
        
        # Create dataset
        pos_dataset = [
            InputExample(texts=[q, a], label=1)
            for q, a in zip(pos_question, pos_answer)
        ]
        neg_dataset = [
            InputExample(texts=[q, a], label=0)
            for q, a in zip(neg_question, neg_answer)
        ]
        
        dataset = pos_dataset + neg_dataset
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        
        # Train
        self.model.fit(
            train_dataloader=dataloader,
            epochs=num_epochs,
            optimizer_params={'lr': learning_rate},
            warmup_steps=0,
            output_path=output_folder
        )
        
        # Save model
        self.model.save_pretrained(output_folder)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        with open(os.path.join(output_folder, 'config_args.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Training completed. Model saved to {output_folder}")

