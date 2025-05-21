import pandas as pd
import torch
from sentence_transformers import SentenceTransformer,SentenceTransformerTrainer,SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset,load_dataset
import argparse
import os
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(input_dir): 
    train_df, eval_df = None, None
    for dirpath, dirname, filenames in os.walk(input_dir): 
        if "train.csv" in filenames: 
            train_df = pd.read_csv(os.path.join(dirpath, "train.csv"), encoding="utf-8") 
        if "eval.csv" in filenames: 
            eval_df = pd.read_csv(os.path.join(dirpath, "eval.csv"), encoding="utf-8") 
    train_question, train_answer = train_df["question"].tolist(), train_df["answer"].tolist()
    eval_question, eval_answer = eval_df["question"].tolist(), eval_df['answer'].tolist() 
    return train_question, train_answer, eval_question, eval_answer


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train Bi Encoder") 
    parser.add_argument("--input_dir", type=str, default="../data/processed", help="Folder contain train.csv and eval.csv") 
    parser.add_argument("--model", type=str, default= "bkai-foundation-models/vietnamese-bi-encoder") 
    parser.add_argument("--num_epochs", type=int, default=3) 
    parser.add_argument("--batch", type=int, default=32) 
    parser.add_argument("--lr", type=float, default=2e-5) 
    parser.add_argument('--weight_decay', type=float, default=0.01) 
    parser.add_argument("--output_dir", type=str, default="../saved_model/BiEncoder/model1")

    args = parser.parse_args()
    args_dict = vars(args)

    # Load data
    train_question, train_answer, eval_question, eval_answer = load_data(args.input_dir)

    # Define dataset
    train_data = {'query': train_question, 'answer': train_answer} 
    eval_data = {'query': eval_question, 'answer': eval_answer} 
    train_dataset = Dataset.from_dict(train_data).shuffle(seed=28)
    eval_dataset = Dataset.from_dict(eval_data)

    # Define model and loss
    model = SentenceTransformer(args.model)
    loss = losses.MultipleNegativesRankingLoss(model)

    # Define hyperparameters
    train_args = SentenceTransformerTrainingArguments(
        output_dir= args.output_dir, 
        num_train_epochs= args.num_epochs,
        per_device_train_batch_size= args.batch,
        per_device_eval_batch_size= args.batch,
        learning_rate= args.lr, 
        fp16=True,  
        bf16=False,  
        batch_sampler=BatchSamplers.NO_DUPLICATES,  
        eval_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        save_steps = 1,
        save_total_limit = 3,
        load_best_model_at_end=True,                
        metric_for_best_model="eval_loss",       
        greater_is_better=False, 
        logging_strategy ='epoch',
        logging_steps=1,
        weight_decay = args.weight_decay,
        report_to= "none",
        log_level = 'error'
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss
    )

    trainer.train()

    # save
    model.save_pretrained(os.path.join(args.output_dir, 'best'))
    with open(os.path.join(args.output_dir,'config.json'), 'w') as f: 
        json.dump(args_dict, f, indent=4)
