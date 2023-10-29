import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import argparse

import dataset
import model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('mode', help="Choose train, eval, or test.")
args = parser.parse_args()

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

go_emo_dataset = load_dataset("go_emotions")
# go_emo_dataset.save_to_disk("datasets/go_emotions")
# go_emo_dataset = load_from_disk("datasets/go_emotions")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
roberta_config = RobertaConfig(hidden_dropout_prob=0.5)
t_model = RobertaModel.from_pretrained('roberta-base', roberta_config).to(device)

data = go_emo_dataset.data

model_config = {
    'model': t_model,
    'n_classes': 28,
    'p_drop': 0.5,
}

max_len = 512

train = data["train"].to_pandas()
dev = data["validation"].to_pandas()
test = data["test"].to_pandas()

train['one_hot'] = train['labels'].apply(utils.labels_to_one_hot)
dev['one_hot'] = dev['labels'].apply(utils.labels_to_one_hot)
test['one_hot'] = test['labels'].apply(utils.labels_to_one_hot)

train_dataset = dataset.GoEmotionsDataset(train.text.tolist(), train.one_hot.tolist(), tokenizer, max_len)
dev_dataset = dataset.GoEmotionsDataset(dev.text.tolist(), train.one_hot.tolist(), tokenizer, max_len)
test_dataset = dataset.GoEmotionsDataset(test.text.tolist(), train.one_hot.tolist(), tokenizer, max_len)

t_model = model.MultLabEmoClassifier(**model_config)

training_args = TrainingArguments(
    output_dir="trainer_outputs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    optim="adamw_torch",
    load_best_model_at_end=True,
    seed=266
)

trainer = Trainer(
    model=t_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=utils.compute_metrics
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
trainer.add_callback(utils.LoggingCallback("roberta_trainer/log.jsonl"))


def main():
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        trainer.evaluate()
    elif args.mode == 'test':
        trainer.args.output_dir = "trainer_outputs/test"
        trainer.evaluate(test_dataset)
    else:
        print("Invalid mode.")

if __name__ == '__main__':
    main()