import random 


def ran_negative(pred_cids, true_cids, neg_num, seed=28): 
    random.seed(seed)
    assert len(pred_cids) == len(true_cids), "Must same length" 
    neg_cids = [] 
    for pred_cid, true_cid in zip(pred_cids, true_cids): 
        neg_cid = [cid for cid in pred_cid if cid not in true_cid]
        neg_cid = random.sample(neg_cid, neg_num) 
        neg_cids.append(neg_cid)
    return neg_cids

def hard_negative(pred_cids, true_cids, neg_num): 
    assert len(pred_cids) == len(true_cids), "Must same length" 
    neg_cids = [] 
    for pred_cid, true_cid in zip(pred_cids, true_cids): 
        neg_cid = [cid for cid in pred_cid if cid not in true_cid]
        neg_cid = neg_cid[:neg_num] 
        neg_cids.append(neg_cid) 
    return neg_cids


