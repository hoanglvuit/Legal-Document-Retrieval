"""Bi-Encoder trainer"""

import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import losses
from datasets import Dataset
from typing import Optional


class BiEncoderTrainer:
    """Trainer for Bi-Encoder model"""
    
    def __init__(self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
                 device: Optional[torch.device] = None):
        """
        Initialize Bi-Encoder trainer
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to train on
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model = SentenceTransformer(model_name)
        print(f"Using device: {device}")
    
    def load_data(self, data_folder: str):
        """
        Load training and evaluation data
        
        Args:
            data_folder: Folder containing train.csv and eval.csv
        
        Returns:
            train_question, train_answer, eval_question, eval_answer
        """
        train_df, eval_df = None, None
        
        for dirpath, dirname, filenames in os.walk(data_folder):
            if "train.csv" in filenames:
                train_df = pd.read_csv(os.path.join(dirpath, "train.csv"), encoding="utf-8")
            if "eval.csv" in filenames:
                eval_df = pd.read_csv(os.path.join(dirpath, "eval.csv"), encoding="utf-8")
        
        if train_df is None or eval_df is None:
            raise FileNotFoundError("train.csv or eval.csv not found in data_folder")
        
        train_question = train_df["question"].tolist()
        train_answer = train_df["answer"].tolist()
        eval_question = eval_df["question"].tolist()
        eval_answer = eval_df['answer'].tolist()
        
        return train_question, train_answer, eval_question, eval_answer
    
    def train(self, data_folder: str, output_folder: str, num_epochs: int = 3,
             batch_size: int = 32, learning_rate: float = 2e-5, weight_decay: float = 0.01, fp16: bool = True, bf16: bool = False, eval_strategy: str = "epoch", metric_for_best_model: str = "eval_loss"):
        """
        Train the Bi-Encoder model
        """
        # Load data
        train_question, train_answer, eval_question, eval_answer = self.load_data(data_folder)
        
        # Define dataset
        train_data = {'query': train_question, 'answer': train_answer}
        eval_data = {'query': eval_question, 'answer': eval_answer}
        train_dataset = Dataset.from_dict(train_data).shuffle(seed=28)
        eval_dataset = Dataset.from_dict(eval_data)
        
        # Define loss
        loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Define training arguments
        from sentence_transformers import SentenceTransformerTrainingArguments
        
        train_args = SentenceTransformerTrainingArguments(
            output_dir=output_folder,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=bf16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy=eval_strategy,
            eval_steps=1,
            save_strategy="epoch",
            save_steps=1,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=False,
            logging_strategy='epoch',
            logging_steps=1,
            weight_decay=weight_decay,
            report_to="none",
            log_level='error'
        )
        
        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss
        )
        
        # Train
        trainer.train()
        
        # Save best model
        self.model.save(os.path.join(output_folder, 'best')) 
        print(f"Training completed. Model saved to {output_folder}")

