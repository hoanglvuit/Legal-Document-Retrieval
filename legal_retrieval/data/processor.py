"""Data processing utilities"""

import pandas as pd
import os
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm

tqdm.pandas()


class DataProcessor:
    """Process raw data into training format"""
    
    @staticmethod
    def process_context(context: str) -> str:
        """Clean context string"""
        return context.replace("['", '').replace("']", '')
    
    @staticmethod
    def load_raw_data(raw_path: str):
        """
        Load raw data from directory
        
        Args:
            raw_path: Path to directory containing train.csv and corpus.csv
        
        Returns:
            train_df, corpus_df
        """
        train_path = None
        corpus_path = None
        
        for dirpath, _, filenames in os.walk(raw_path):
            if "train.csv" in filenames:
                train_path = os.path.join(dirpath, 'train.csv')
            if "corpus.csv" in filenames:
                corpus_path = os.path.join(dirpath, 'corpus.csv')
        
        if train_path is None or corpus_path is None:
            raise FileNotFoundError("train.csv or corpus.csv not found in raw_path")
        
        train_df = pd.read_csv(train_path, encoding='utf-8')
        corpus_df = pd.read_csv(corpus_path, encoding='utf-8')
        
        return train_df, corpus_df
    
    @staticmethod
    def process_train_data(train_df: pd.DataFrame, corpus_df: pd.DataFrame):
        """
        Process training data into question-answer pairs
        
        Args:
            train_df: Training dataframe
            corpus_df: Corpus dataframe
        
        Returns:
            List of [question, answer, cid] tuples
        """
        train_question = train_df['question'].tolist()
        train_answer = train_df['context'].tolist()
        raw_cid = train_df['cid'].tolist()
        text = corpus_df['text'].tolist()
        corpus_cid = corpus_df['cid'].tolist()
        
        train_cid = []
        for cids in raw_cid:
            cids = cids.strip('[]').split()
            train_cid.append([int(cid) for cid in cids])
        
        processed_train = []
        cid_to_index = {cid: ind for ind, cid in enumerate(corpus_cid)}
        
        for ind, cids in enumerate(train_cid):
            for cid in cids:
                if cid in corpus_cid:
                    processed_train.append([
                        train_question[ind],
                        text[cid_to_index[cid]],
                        cids
                    ])
                else:
                    processed_train.append([
                        train_question[ind],
                        DataProcessor.process_context(train_answer[ind]),
                        cids
                    ])
        
        return processed_train
    
    @staticmethod
    def tokenize_data(data: List[list], desc: str = "Tokenizing"):
        """
        Tokenize questions and answers
        
        Args:
            data: List of [question, answer, cid] tuples
            desc: Description for progress bar
        
        Returns:
            Tokenized data
        """
        return [
            [tokenize(ques), tokenize(answer), cid]
            for ques, answer, cid in tqdm(data, desc=desc)
        ]
    
    @staticmethod
    def split_train_eval(data: List[list], eval_size: float = 0.1, random_state: int = 28):
        """
        Split data into train and eval sets
        
        Args:
            data: List of data samples
            eval_size: Proportion of eval set
            random_state: Random seed
        
        Returns:
            train_data, eval_data
        """
        return train_test_split(data, test_size=eval_size, random_state=random_state)
    
    @staticmethod
    def process_corpus(corpus_df: pd.DataFrame):
        """
        Process corpus data
        
        Args:
            corpus_df: Corpus dataframe
        
        Returns:
            Processed corpus dataframe
        """
        corpus_df['text'] = corpus_df['text'].progress_apply(tokenize)
        return corpus_df
    
    @staticmethod
    def save_processed_data(train_data: List[list], eval_data: List[list],
                           corpus_df: pd.DataFrame, output_path: str):
        """
        Save processed data to CSV files
        
        Args:
            train_data: Training data
            eval_data: Evaluation data
            corpus_df: Processed corpus dataframe
            output_path: Output directory path
        """
        os.makedirs(output_path, exist_ok=True)
        
        train_df = pd.DataFrame(train_data, columns=['question', 'answer', 'cid'])
        train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
        
        eval_df = pd.DataFrame(eval_data, columns=['question', 'answer', 'cid'])
        eval_df.to_csv(os.path.join(output_path, 'eval.csv'), index=False)
        
        corpus_df.to_csv(os.path.join(output_path, 'corpus.csv'), index=False)
    
    @classmethod
    def process_pipeline(cls, raw_path: str, processed_path: str,
                        eval_size: float = 0.1, random_state: int = 28):
        """
        Complete data processing pipeline
        
        Args:
            raw_path: Path to raw data directory
            processed_path: Path to save processed data
            eval_size: Proportion of eval set
            random_state: Random seed
        """
        # Load data
        train_df, corpus_df = cls.load_raw_data(raw_path)
        
        # Process training data
        processed_train = cls.process_train_data(train_df, corpus_df)
        seg_processed_train = cls.tokenize_data(processed_train, desc="Tokenizing train set")
        
        # Split train/eval
        train, eval = cls.split_train_eval(seg_processed_train, eval_size, random_state)
        
        # Process corpus
        processed_corpus = cls.process_corpus(corpus_df)
        
        # Save
        cls.save_processed_data(train, eval, processed_corpus, processed_path)
        
        print(f"Data processing completed. Saved to {processed_path}")

