from torch.utils.data import Dataset
import torch

class GoEmotionsDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len, padding=False)
        return {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": self.labels[idx]
            }
