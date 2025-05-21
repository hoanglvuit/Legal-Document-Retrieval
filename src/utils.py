import torch
import os
from sentence_transformers import SentenceTransformer, util
from typing import List


def get_top_cids(score_pred: List[List[float]], num:int, cids: List[int]): 
    top_indices_per_row = []
    for row in score_pred:
        top_indices = [i for i, _ in sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:num]]
        top_indices_per_row.append(top_indices)
    top_cids = [[cids[i] for i in pred] for pred in top_indices_per_row]
    return top_cids


def get_candidate(question_embedding, answer_embedding, cids, num, saved_folder, name):
    '''
        Description: Get num candidates for BiEncoder from embedding and save result in saved_folder with name
    '''
    tensor = util.cos_sim(question_embedding, answer_embedding) 
    top_cids = get_top_cids(tensor.cpu().tolist(), num, cids)
    
    output_path = os.path.join(saved_folder, f"output{name}.txt")
    with open(output_path, "w") as file:
        file.writelines(" ".join(map(str, sublist)) + "\n" for sublist in top_cids)
    return top_cids

#test get_candidate
# sentence1 = ["Cô ấy là người tốt", "Cô ấy rất xinh đẹp", "Tôi yêu cô ấy"] 
# sentence2 = ["Cô ấy tệ", "Cô ấy như thần tiên"]

# model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
# embeddings1 = model.encode(sentence1, convert_to_tensor=True)
# embeddings2 = model.encode(sentence2, convert_to_tensor=True)
# get_candidate(embeddings1, embeddings2, [1,2,3,4,5,6,7,8], 1, '.')

def exist_m(prediction: List[List[int]], true_cids: List[List[int]], m: int=10) -> float: 
    assert len(prediction) == len(true_cids), "Must same length"
    num_exist = 0 
    for pred_cids, true_cids in zip(prediction, true_cids): 
        pred_cids = pred_cids[:m] 
        num_exist += any(item in set(true_cids) for item in pred_cids) 
    exist_score = num_exist/len(prediction)
    print(f"Exist@{m} = {exist_score}") 
    return exist_score

def mrr_m(prediction: List[List[int]], true_cids: List[List[int]], m: int=10) -> float: 
    assert len(prediction) == len(true_cids), "Must same length"
    mrr_num = 0 
    for pred_cids, true_cids in zip(prediction, true_cids): 
        pred_cids = pred_cids[:m] 
        for ind, cid in enumerate(pred_cids): 
            if cid in true_cids: 
                mrr_num += 1/(ind+1) 
                break
    mrr_score = mrr_num / len(prediction)
    print(f"MRR@{m} = {mrr_score}") 
    return mrr_score

# test 
# A = [[1,2,3,4],[3,4,2,1],[10,4,2,1,4,5]] 
# B = [[5,2],[0,6],[10,5]] 
# exist_m(A,B) 
# mrr_m(A,B)