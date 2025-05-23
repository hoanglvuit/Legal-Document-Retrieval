import os
import json
import logging
import argparse 
import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Train Cross model by positive and negative examples!") 
    parser.add_argument('--model', type=str, default='itdainb/PhoRanker')
    parser.add_argument('--num_epoch', type=int, default=2) 
    parser.add_argument('--lr', type=float, default=2e-5) 
    parser.add_argument('--pos_path', type=str, default= 'data/processed/train.csv')
    parser.add_argument('--neg_path', type=str, default= 'data/processed/3_moderateneg_ex.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_folder', type=str, default='saved_model/CrossEncoder/model1')

    args = parser.parse_args() 
    os.makedirs(args.output_folder, exist_ok=True) 

    # load positive examples
    pos_df = pd.read_csv(args.pos_path, encoding='utf-8') 
    pos_question = pos_df['question'].tolist() 
    pos_answer = pos_df['answer'].tolist() 

    # load negative examples
    neg_df = pd.read_csv(args.neg_path, encoding='utf-8') 
    neg_question = neg_df['question'].tolist() 
    neg_answer = neg_df['answer'].tolist() 

    # create dataset 

    pos_dataset = [InputExample(texts=[q,a], label = 1) for q, a in zip(pos_question, pos_answer)]
    neg_dataset = [InputExample(texts=[q,a], label = 0) for q, a in zip(neg_question, neg_answer)]

    dataset = pos_dataset + neg_dataset
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)


    model = CrossEncoder(args.model, max_length=256)
    logging.disable(logging.WARNING)
    model.fit(
            train_dataloader=dataloader,
            epochs=args.num_epoch,
            optimizer_params = {'lr' : args.lr},
            warmup_steps=0,
            output_path= args.output_folder
        )
    
    # save model and config
    model.save_pretrained(args.output_folder)
    args_dict = vars(args) 
    with open(os.path.join(args.output_folder, 'config.json'), 'w') as f: 
        json.dump(args_dict, f, indent=4)


    

