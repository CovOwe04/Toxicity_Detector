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

# =====================================================
# GRU MODEL (STRUCTURE ONLY)
# =====================================================

class GRUModel(nn.Module):

    def __init__(self):
        super().__init__()

        # TODO:
        # - embedding layer
        # - GRU layer
        # - classification head

        self.dummy = nn.Linear(10, 6)

    def forward(self, x):
        return self.dummy(x)


# =====================================================
# TRANSFORMER MODEL LOADER
# =====================================================

def load_transformer():

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=6,
        problem_type="multi_label_classification"
    )

    return model


# =====================================================
# BEST MODEL LOADER (USED BY API)
# =====================================================

def load_best_model():

    path = "models/best_model.pt"

    try:
        checkpoint = torch.load(path, map_location="cpu")

        model_type = checkpoint["model_type"]

        if model_type == "gru":
            model = GRUModel()
        else:
            model = load_transformer()

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model, model_type

    except:

        # fallback (no trained model yet)
        return GRUModel(), "gru_fallback"