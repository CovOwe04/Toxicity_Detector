from fastapi import FastAPI
from pydantic import BaseModel

import json
import numpy as np
import torch
from pathlib import Path

from transformers import AutoTokenizer

from preprocess import TextPreprocessor
from model import load_best_model, LABELS


# =====================================================
# APP SETUP
# =====================================================

app = FastAPI()

preprocessor = TextPreprocessor()

model, model_type = load_best_model()


class TextRequest(BaseModel):
    text: str


MAX_SEQ_LEN = 150


# =====================================================
# LOAD PRODUCTION VOCAB (FROM TRAINING PIPELINE)
# =====================================================

def load_vocab(path="models/vocab.json"):
    vocab_path = Path(path)

    if not vocab_path.exists():
        return {"<PAD>": 0, "<UNK>": 1}

    with vocab_path.open("r") as f:
        return json.load(f)


vocab = load_vocab()


# =====================================================
# GRU TOKENIZATION (PRODUCTION SAFE)
# =====================================================

def tokenize_gru(text: str):

    tokens = [vocab.get(w, 1) for w in text.split()]  # 1 = <UNK>

    tokens = tokens[:MAX_SEQ_LEN]
    tokens += [0] * (MAX_SEQ_LEN - len(tokens))       # 0 = <PAD>

    return np.array(tokens)


# =====================================================
# TRANSFORMER TOKENIZER (DEBERTA)
# =====================================================

def load_transformer_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-base",
            local_files_only=True
        )
    except Exception:
        return None


transformer_tokenizer = load_transformer_tokenizer()


def tokenize_transformer(text: str):
    if transformer_tokenizer is None:
        return None

    return transformer_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )


def predict_gru(tokens):
    """
    Run GRU inference and return toxicity probabilities.
    """

    device = next(model.parameters()).device

    input_tensor = torch.tensor(
        tokens,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    return {
        label: float(score)
        for label, score in zip(LABELS, probabilities)
    }


# =====================================================
# ROUTES
# =====================================================

@app.get("/")
def health_check():
    return {
        "status": "active",
        "model_loaded": model_type,
        "vocab_size": len(vocab)
    }


@app.post("/predict")
def predict(req: TextRequest):

    cleaned_text = preprocessor.normalize(req.text)

    # =================================================
    # GRU TOKENIZATION (FROM TRAINED VOCAB)
    # =================================================

    gru_tokens = tokenize_gru(cleaned_text)

    # =================================================
    # TRANSFORMER TOKENIZATION
    # =================================================

    transformer_tokens = tokenize_transformer(cleaned_text)
    transformer_input_ids_sample = []
    attention_mask_sample = []

    if transformer_tokens is not None:
        transformer_input_ids_sample = transformer_tokens["input_ids"][0][:20].tolist()
        attention_mask_sample = transformer_tokens["attention_mask"][0][:20].tolist()

    # =================================================
    # MODEL INFERENCE
    # =================================================

    if model_type == "gru":
        toxicity_scores = predict_gru(gru_tokens)

    else:
        raise RuntimeError(
            f"Unsupported deployed model type: {model_type}"
        )


    return {
        "input": cleaned_text,
        "model_used": model_type,

        # DEBUG / QA OUTPUT
        "gru_tokens_sample": gru_tokens[:20].tolist(),

        "transformer_input_ids_sample": transformer_input_ids_sample,
        "attention_mask_sample": attention_mask_sample,

        "toxicity_scores": toxicity_scores
    }