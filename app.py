import os
import json
import numpy as np
import torch
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from transformers import AutoTokenizer

from preprocess import TextPreprocessor
from model import load_best_model, LABELS

# Resolve project root dynamically for Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MAX_SEQ_LEN = 64

# Page setup
st.set_page_config(page_title="Toxicity Detector", layout="wide")

# =====================================================
# CACHED MODEL & TOKENIZER LOADERS
# =====================================================

@st.cache_resource
def load_app_resources():
    preprocessor = TextPreprocessor()
    model, model_type = load_best_model()
    return preprocessor, model, model_type

@st.cache_data
def load_vocab(path="models/vocab.json"):
    vocab_path = BASE_DIR / path if not Path(path).is_absolute() else Path(path)
    if not vocab_path.exists():
        return {"<PAD>": 0, "<UNK>": 1}
    with vocab_path.open("r") as f:
        return json.load(f)

@st.cache_resource
def load_transformer_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    except Exception:
        return None

# Load resources
preprocessor, model, model_type = load_app_resources()
MAX_SEQ_LEN = getattr(model, "max_seq_len", DEFAULT_MAX_SEQ_LEN)
vocab = load_vocab()
transformer_tokenizer = load_transformer_tokenizer()

# =====================================================
# INFERENCE HELPERS
# =====================================================

def tokenize_gru(text: str):
    tokens = [vocab.get(w, 1) for w in text.split()]
    tokens = tokens[:MAX_SEQ_LEN]
    tokens += [0] * (MAX_SEQ_LEN - len(tokens))
    return np.array(tokens)

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

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("Multi-Platform Toxicity Detection System")
st.write("Enter a comment to analyze toxicity across 6 categories.")

text = st.text_area("Input Comment", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text.")
        st.stop()

    with st.spinner("Analyzing text..."):
        try:
            cleaned_text = preprocessor.normalize(text)

            if model_type in ["gru", "gru_fallback"]:
                gru_tokens = tokenize_gru(cleaned_text)
                toxicity_scores = predict_gru(gru_tokens)
            elif model_type in ["transformer", "deberta"]:
                transformer_tokens = tokenize_transformer(cleaned_text)
                if transformer_tokens is None:
                    st.error("Transformer tokenizer failed to load.")
                    st.stop()
                toxicity_scores = predict_transformer(transformer_tokens)
            else:
                st.error(f"Unsupported model type loaded: {model_type}")
                st.stop()

            # Display Results Chart
            df = pd.DataFrame({
                "Label": list(toxicity_scores.keys()),
                "Score": list(toxicity_scores.values())
            })

            fig = px.bar(
                df,
                x="Label",
                y="Score",
                range_y=[0, 1],
                title="Toxicity Breakdown"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display Raw Scores JSON
            st.subheader("Raw Output")
            st.json({
                "input": cleaned_text,
                "model_used": model_type,
                "toxicity_scores": toxicity_scores
            })

        except Exception as e:
            st.error("An error occurred during prediction.")
            st.code(str(e))

# Sidebar Information
st.sidebar.title("System Info")
st.sidebar.write("Pure Streamlit Standalone Application")
st.sidebar.write(f"Model Engine Active: **{model_type.upper()}**")
st.sidebar.write("6-class multi-label toxicity detection")

