"""Evaluation metrics"""

from typing import List


def exist_m(prediction: List[List[int]], true_cids: List[List[int]], m: int = 10) -> float:
    """
    Calculate Exist@m metric
    
    Args:
        prediction: List of predicted context ID lists
        true_cids: List of true context ID lists
        m: Number of top predictions to consider
    
    Returns:
        Exist@m score
    """
    assert len(prediction) == len(true_cids), "Must same length"
    num_exist = 0
    
    for pred_cids, true_cids_list in zip(prediction, true_cids):
        pred_cids = pred_cids[:m]
        num_exist += any(item in set(true_cids_list) for item in pred_cids)
    
    exist_score = num_exist / len(prediction)
    print(f"Exist@{m} = {exist_score}")
    return exist_score


def mrr_m(prediction: List[List[int]], true_cids: List[List[int]], m: int = 10) -> float:
    """
    Calculate MRR@m metric (Mean Reciprocal Rank)
    
    Args:
        prediction: List of predicted context ID lists
        true_cids: List of true context ID lists
        m: Number of top predictions to consider
    
    Returns:
        MRR@m score
    """
    assert len(prediction) == len(true_cids), "Must same length"
    mrr_num = 0
    
    for pred_cids, true_cids_list in zip(prediction, true_cids):
        pred_cids = pred_cids[:m]
        for ind, cid in enumerate(pred_cids):
            if cid in true_cids_list:
                mrr_num += 1 / (ind + 1)
                break
    
    mrr_score = mrr_num / len(prediction)
    print(f"MRR@{m} = {mrr_score}")
    return mrr_score

