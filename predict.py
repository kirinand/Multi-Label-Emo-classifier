import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    RobertaConfig,
)
from .model import MultLabEmoClassifier

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
model_path = "saved_model"

tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(BASE_DIR, model_path, 'roberta_tokenizer'))
roberta_config = RobertaConfig(hidden_dropout_prob=0.5)
base_model = RobertaModel.from_pretrained(os.path.join(BASE_DIR, model_path, 'roberta_model'), roberta_config)

model_config = {
    'model': base_model,
    'n_classes': 28,
    'p_drop': 0.5,
}

max_len = 512

t_model = MultLabEmoClassifier(**model_config)
t_model.load_state_dict(torch.load(os.path.join(BASE_DIR, model_path, 'pytorch_model.bin'), map_location=torch.device(device)), strict=False)
t_model.to(device)

t_model.eval()

sample = "How can I divide 35 in a half, a third and a ninth, and get an even number? It seems impossible! Yet I know there must be a way..."

def predict(text):
    """
    Predicts the labels for a given text or list of texts
    :param text: string, or a list of strings
    :return: binary tensor of shape (number of strings, 28)
    """
    input = tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True, padding=False).to(device)
    logits = t_model(**input)
    output = torch.sigmoid(logits)
    output = (output > 0.1).int()
    return output.tolist()


def main():
    print(predict(sample))

if __name__ == '__main__':
    main()