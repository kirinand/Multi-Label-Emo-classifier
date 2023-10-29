import torch.nn as nn

class MultLabEmoClassifier(nn.Module):
    def __init__(self, model, p_drop, n_classes):
        super().__init__()
        self.transformer = model
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(768, n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask)["pooler_output"]
        drop_output = self.dropout(transformer_output)
        logits = self.classifier(drop_output)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            return loss, logits
        
        return logits
