import torch
import os
from sentence_transformers import SentenceTransformer, util


def get_candidate(question_embedding, answer_embedding, cids, num, saved_folder):
    tensor = util.cos_sim(question_embedding, answer_embedding) 
    _, top_indices = torch.topk(tensor, num, dim=1)
    top_cids = [[cids[i] for i in indices] for indices in top_indices.cpu().tolist()]
    output_path = os.path.join(saved_folder, "output.txt")
    with open(output_path, "w") as file:
        file.writelines(" ".join(map(str, sublist)) + "\n" for sublist in top_cids)

#test get_candidate
# sentence1 = ["Cô ấy là người tốt", "Cô ấy rất xinh đẹp", "Tôi yêu cô ấy"] 
# sentence2 = ["Cô ấy tệ", "Cô ấy như thần tiên"]

# model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
# embeddings1 = model.encode(sentence1, convert_to_tensor=True)
# embeddings2 = model.encode(sentence2, convert_to_tensor=True)
# get_candidate(embeddings1, embeddings2, [1,2,3,4,5,6,7,8], 1, '.')

