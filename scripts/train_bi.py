"""Train Bi-Encoder model"""

from legal_retrieval.training.trainer_bi import BiEncoderTrainer
import yaml
import argparse

# load config 
with open('configs/bi_encoder_config.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Bi-Encoder model")
    parser.add_argument('--model_name', type=str, default=data['model_name'])
    parser.add_argument('--data_folder', type=str, default=data['data_folder'])
    parser.add_argument('--output_folder', type=str, default=data['output_folder'])
    parser.add_argument('--num_epochs', type=int, default=float(data['training']['num_epochs']))
    parser.add_argument('--batch_size', type=int, default=float(data['training']['batch_size']))
    parser.add_argument('--learning_rate', type=float, default=float(data['training']['learning_rate']))
    parser.add_argument('--weight_decay', type=float, default=float(data['training']['weight_decay']))
    parser.add_argument('--fp16', type=bool, default=bool(data['hardware']['fp16']))
    parser.add_argument('--bf16', type=bool, default=bool(data['hardware']['bf16']))
    parser.add_argument('--eval_strategy', type=str, default=data['evaluation']['eval_strategy'])
    parser.add_argument('--metric_for_best_model', type=str, default=data['evaluation']['metric_for_best_model'])
    args = parser.parse_args()

    trainer = BiEncoderTrainer(model_name=args.model_name)
    trainer.train(
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        metric_for_best_model=args.metric_for_best_model
    )
