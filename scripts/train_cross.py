"""Train Cross-Encoder model"""

import argparse
from legal_retrieval.training.trainer_cross import CrossEncoderTrainer
import yaml

with open('configs/cross_encoder_config.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cross model by positive and negative examples!")
    parser.add_argument('--model', type=str, default=data['model_name'])
    parser.add_argument('--num_epoch', type=int, default=data['training']['num_epochs'])
    parser.add_argument('--lr', type=float, default=data['training']['learning_rate'])
    parser.add_argument('--pos_path', type=str, default=data['pos_path'])
    parser.add_argument('--neg_path', type=str, default=data['neg_path'])
    parser.add_argument('--batch_size', type=int, default=data['training']['batch_size'])
    parser.add_argument('--output_folder', type=str, default=data['output_folder'])

    args = parser.parse_args()

    trainer = CrossEncoderTrainer(model_name=args.model)
    trainer.train(
        pos_path=args.pos_path,
        neg_path=args.neg_path,
        output_folder=args.output_folder,
        num_epochs=args.num_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

