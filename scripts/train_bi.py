"""Train Bi-Encoder model"""

from legal_retrieval.training.trainer_bi import BiEncoderTrainer
import yaml

# load config 
with open('configs/bi_encoder_config.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

trainer = BiEncoderTrainer(model_name=data['model_name'])
trainer.train(
    data_folder=data['data_folder'],
    output_folder=data['output_folder'],
    num_epochs=data['training']['num_epochs'],
    batch_size=data['training']['batch_size'],
    learning_rate=data['training']['learning_rate'],
    weight_decay=data['training']['weight_decay'],
    fp16=data['hardware']['fp16'],
    bf16=data['hardware']['bf16'],
    eval_strategy=data['evaluation']['eval_strategy'],
    metric_for_best_model=data['evaluation']['metric_for_best_model']
)
