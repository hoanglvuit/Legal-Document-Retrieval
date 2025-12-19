"""Train Cross-Encoder model"""

import argparse
from legal_retrieval.training.trainer_cross import CrossEncoderTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cross model by positive and negative examples!")
    parser.add_argument('--model', type=str, default='itdainb/PhoRanker')
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--pos_path', type=str, default='data/processed/train.csv')
    parser.add_argument('--neg_path', type=str, default='data/processed/3_moderateneg_ex.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_folder', type=str, default='saved_models/CrossEncoder/model1')

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

