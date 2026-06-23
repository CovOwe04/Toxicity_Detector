from fastapi import FastAPI
from pydantic import BaseModel

import json
import numpy as np
from pathlib import Path

from transformers import AutoTokenizer

from preprocess import TextPreprocessor
from model import load_best_model


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

    # -------------------------------------------------
    # NOTE:
    # MODEL INFERENCE STILL NOT IMPLEMENTED (EXPECTED)
    # -------------------------------------------------

    return {
        "input": cleaned_text,
        "model_used": model_type,

        # DEBUG / QA OUTPUT
        "gru_tokens_sample": gru_tokens[:20].tolist(),

        "transformer_input_ids_sample": transformer_input_ids_sample,
        "attention_mask_sample": attention_mask_sample,

        "toxicity_scores": {
            "toxic": 0.0,
            "severe_toxic": 0.0,
            "obscene": 0.0,
            "threat": 0.0,
            "insult": 0.0,
            "identity_hate": 0.0
        }
    }