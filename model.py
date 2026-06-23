from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# =====================================================
# LABEL SPACE
# =====================================================

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

DEFAULT_TRANSFORMER_MODEL = "microsoft/deberta-v3-base"


# =====================================================
# GRU MODEL
# =====================================================

class GRUModel(nn.Module):

    def __init__(
        self,
        vocab_size=2,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        output_dim=len(LABELS),
        dropout=0.3,
        bidirectional=True,
        padding_idx=0,
    ):
        super().__init__()

        self.config = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "padding_idx": padding_idx,
        }

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        direction_multiplier = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * direction_multiplier, output_dim)

    def forward(self, x):
        x = x.long()
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)

        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        return self.classifier(hidden)


# =====================================================
# TRANSFORMER MODEL LOADER
# =====================================================

def load_transformer(model_name=DEFAULT_TRANSFORMER_MODEL):

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )

    return model.float()


# =====================================================
# BEST MODEL LOADER (USED BY API)
# =====================================================

def load_best_model(path="models/best_model.pt"):

    checkpoint_path = Path(path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_type = checkpoint["model_type"]

        if model_type == "gru":
            model_config = checkpoint.get("model_config", {})
            model = GRUModel(**model_config)
        else:
            transformer_name = checkpoint.get(
                "transformer_model_name",
                DEFAULT_TRANSFORMER_MODEL,
            )
            model = load_transformer(transformer_name)

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model, model_type

    except Exception:
        return GRUModel(), "gru_fallback"

