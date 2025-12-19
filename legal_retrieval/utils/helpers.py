"""Helper utility functions"""

import torch
import os
from sentence_transformers import util
from typing import List


def get_top_cids(score_pred: List[List[float]], num: int, cids: List[int]) -> List[List[int]]:
    """
    Get top N context IDs based on scores
    
    Args:
        score_pred: List of score lists for each query
        num: Number of top results to return
        cids: List of context IDs
    
    Returns:
        List of top N context IDs for each query
    """
    top_indices_per_row = []
    for row in score_pred:
        top_indices = [
            i for i, _ in sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:num]
        ]
        top_indices_per_row.append(top_indices)
    top_cids = [[cids[i] for i in pred] for pred in top_indices_per_row]
    return top_cids


def get_candidate(question_embedding, answer_embedding, cids: List[int], num: int,
                 saved_folder: str, name: str, batch_size: int = 512) -> List[List[int]]:
    """
    Batched version of candidate retrieval to avoid CUDA OOM.
    Computes cosine similarity in batches.
    
    Args:
        question_embedding: Tensor of question embeddings
        answer_embedding: Tensor of answer embeddings
        cids: List of context IDs
        num: Number of top candidates to retrieve
        saved_folder: Folder to save output
        name: Suffix for output filename
        batch_size: Batch size for processing
    
    Returns:
        List of top candidate CIDs for each question
    """
    all_top_cids = []
    output_path = os.path.join(saved_folder, f"output{name}.txt")
    
    # Move answer_embedding to CPU if too big for GPU or keep on GPU if you want speed
    answer_embedding = answer_embedding.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(saved_folder, exist_ok=True)
    
    with open(output_path, "w") as file:
        for start_idx in range(0, question_embedding.size(0), batch_size):
            end_idx = min(start_idx + batch_size, question_embedding.size(0))
            batch_questions = question_embedding[start_idx:end_idx].to(answer_embedding.device)

            # Compute similarity for batch
            sim = util.cos_sim(batch_questions, answer_embedding)  # shape: (batch_size, num_answers)
            
            # Move to CPU and convert to list for get_top_cids
            sim_cpu = sim.cpu().tolist()

            # Get top cids for this batch
            batch_top_cids = get_top_cids(sim_cpu, num, cids)
            all_top_cids.extend(batch_top_cids)

            # Write batch results to file
            for sublist in batch_top_cids:
                file.write(" ".join(map(str, sublist)) + "\n")

            # Free GPU cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_top_cids

