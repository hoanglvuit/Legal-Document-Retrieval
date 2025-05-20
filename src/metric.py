def exist_m(prediction, true_cids, m=10): 
    assert len(prediction) == len(true_cids), "Must same length"
    num_exist = 0 
    for pred_cids, true_cids in zip(prediction, true_cids): 
        pred_cids = pred_cids[:m] 
        num_exist += any(item in set(true_cids) for item in pred_cids) 
    exist_score = num_exist/len(prediction)
    print(f"Exist@{m} = {exist_score}") 
    return exist_score

def mrr_m(prediction, true_cids, m=10): 
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