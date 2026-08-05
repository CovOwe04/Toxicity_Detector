import os
import json
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from preprocess import TextPreprocessor
from model import load_best_model, LABELS

app = FastAPI(title="Toxicity Detector API")

preprocessor = TextPreprocessor()
model, model_type = load_best_model()

class TextRequest(BaseModel):
    text: str

MAX_SEQ_LEN = 150

def load_vocab(path="models/vocab.json"):
    vocab_path = Path(path)
    if not vocab_path.exists():
        return {"<PAD>": 0, "<UNK>": 1}
    with vocab_path.open("r") as f:
        return json.load(f)

vocab = load_vocab()

def tokenize_gru(text: str):
    tokens = [vocab.get(w, 1) for w in text.split()]
    tokens = tokens[:MAX_SEQ_LEN]
    tokens += [0] * (MAX_SEQ_LEN - len(tokens))
    return np.array(tokens)

def load_transformer_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
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
    device = next(model.parameters()).device
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()
    return {label: float(score) for label, score in zip(LABELS, probabilities)}

def predict_transformer(inputs):
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()
    return {label: float(score) for label, score in zip(LABELS, probabilities)}

@app.get("/")
def health_check():
    return {
        "status": "active",
        "model_loaded": model_type,
        "vocab_size": len(vocab)
    }

@app.post("/predict")
def predict(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    cleaned_text = preprocessor.normalize(req.text)
    
    gru_tokens = tokenize_gru(cleaned_text)
    transformer_tokens = tokenize_transformer(cleaned_text)
    
    transformer_input_ids_sample = []
    attention_mask_sample = []

    if transformer_tokens is not None:
        transformer_input_ids_sample = transformer_tokens["input_ids"][0][:20].tolist()
        attention_mask_sample = transformer_tokens["attention_mask"][0][:20].tolist()

    if model_type == "gru":
        toxicity_scores = predict_gru(gru_tokens)
    elif model_type in ["transformer", "deberta"]:
        if transformer_tokens is None:
            raise HTTPException(status_code=500, detail="Transformer tokenizer is uninitialized.")
        toxicity_scores = predict_transformer(transformer_tokens)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported model type: {model_type}")

    return {
        "input": cleaned_text,
        "model_used": model_type,
        "gru_tokens_sample": gru_tokens[:20].tolist(),
        "transformer_input_ids_sample": transformer_input_ids_sample,
        "attention_mask_sample": attention_mask_sample,
        "toxicity_scores": toxicity_scores
    }