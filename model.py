import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

# Resolve project root dynamically so relative paths work in Cloud environments
BASE_DIR = Path(__file__).resolve().parent

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

def load_transformer(model_name=DEFAULT_TRANSFORMER_MODEL, quiet=True):

    def build_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            problem_type="multi_label_classification"
        )

    if quiet:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                model = build_model()
    else:
        model = build_model()

    return model.float()


# =====================================================
# BEST MODEL LOADER
# =====================================================

def load_best_model(path="models/best_model.pt"):
    checkpoint_path = BASE_DIR / path if not Path(path).is_absolute() else Path(path)

    if not checkpoint_path.exists():
        print(f"WARNING: Model file not found at {checkpoint_path}. Falling back to untrained GRU.")
        return GRUModel(), "gru_fallback"

    try:
        # Streamlit cloud supports weights_only=False for custom dictionary checkpoints
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_type = checkpoint.get("model_type", "gru")

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

    except Exception as e:
        print(f"ERROR loading checkpoint from {checkpoint_path}: {e}")
        return GRUModel(), "gru_fallback"