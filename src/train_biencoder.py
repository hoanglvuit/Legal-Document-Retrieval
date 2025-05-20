import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,SentenceTransformerTrainer,SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset,load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
train_df = pd.read_csv('../data/processed/train.csv',encoding='utf-8') 
train_ques = train_df['question'].tolist() 
train_ans = train_df['answer'].tolist() 
eval_df = pd.read_csv('../data/processed/train.csv',encoding='utf-8') 
eval_ques = eval_df['question'].tolist() 
eval_ans = eval_df['answer'].tolist() 

# Define data
train_data = {'query': train_ques,
                'answer': train_ans} 
eval_data = {'query': eval_ques, 
                'answer': eval_ans} 
train_dataset = Dataset.from_dict(train_data).shuffle(seed=28)
eval_dataset = Dataset.from_dict(eval_data)

# Bi-encoder
model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

loss = losses.MultipleNegativesRankingLoss(model)


args = SentenceTransformerTrainingArguments(
    output_dir="models/bi_encoder", 
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0, 
    fp16=True,  
    bf16=False,  
    batch_sampler=BatchSamplers.NO_DUPLICATES,  
    eval_strategy="steps",
    eval_steps=750,
    save_strategy="steps",
    save_steps = 750,
    save_total_limit = 3,
    load_best_model_at_end=True,                
    metric_for_best_model="eval_loss",       
    greater_is_better=False,  
    logging_steps=200,
    weight_decay =0.01,
)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()
model.save_pretrained('models/bi_encoder/best')