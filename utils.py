import json
import numpy as np
from transformers import TrainerCallback

n_classes = 28

def labels_to_one_hot(labels):
    one_hot = [0] * n_classes
    for lb in labels:
        one_hot[lb] = 1
    return one_hot


def compute_metrics(eval_pred):
    logits, one_hot_labels = eval_pred
    probabilities = 1 / (1 + np.exp(-np.array(logits)))
    predictions = (probabilities > 0.5).astype(int)

    return {"accuracy": np.mean(predictions == np.array(one_hot_labels))}


class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


# def pad_tokenized_data(batch):
#     ids = [item["ids"] for item in batch]
#     mask = [item["mask"] for item in batch]
#     labels = [item["labels"] for item in batch]
#
#     pad_sequence(ids, batch_first=True, padding_value=0)
#     pad_sequence(mask, batch_first=True, padding_value=0)
#     pad_sequence(labels, batch_first=True, padding_value=0)
#
#     return {
#         "ids": ids,
#         "mask": mask,
#         "labels": labels
#     }
