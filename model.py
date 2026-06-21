import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# ---------------- GRU MODEL ----------------
class GRUModel(nn.Module):

    def __init__(self, vocab_size=50000, emb=128, hidden=256, out=6):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h[-1])


# ---------------- TRANSFORMER MODEL ----------------
def load_transformer():

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=6,
        problem_type="multi_label_classification"
    )

    return model


# ---------------- ENSEMBLE ----------------
def ensemble_predict(gru_logits, transformer_logits):

    return (gru_logits + transformer_logits) / 2