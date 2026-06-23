import pandas as pd
import torch
import json
from pathlib import Path

from preprocess import TextPreprocessor
from metrics import compute_roc_auc
from model import GRUModel, load_transformer


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "data/train.csv"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# =====================================================
# DATA LOADING LAYER
# =====================================================

def load_dataset(path=DATA_PATH):
    """
    Load raw Jigsaw dataset
    """
    df = pd.read_csv(path)

    # Standardize Jigsaw format
    df = df[["comment_text"] + LABELS]

    return df


# =====================================================
# PREPROCESSING LAYER
# =====================================================

def preprocess_dataset(df):
    """
    Apply text normalization + cleaning pipeline
    """

    processor = TextPreprocessor()

    # Create a clean working column
    df["clean_text"] = df["comment_text"].astype(str).apply(
        lambda x: processor.normalize(x)
    )

    # Optional: drop empty rows after cleaning
    df = df[df["clean_text"].str.len() > 0]

    return df


# =====================================================
# FEATURE ENGINEERING LAYER
# =====================================================

def build_features(df):
    """
    Convert text into model-ready format
    """

    from collections import Counter
    import numpy as np
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    MAX_VOCAB_SIZE = 50000
    MAX_SEQ_LEN = 150

    # =================================================
    # LABELS
    # =================================================

    y = df[LABELS].values

    # =================================================
    # GRU TOKENIZATION (CUSTOM VOCAB)
    # =================================================

    print("Building GRU vocabulary...")

    counter = Counter()
    for text in df["clean_text"]:
        counter.update(text.split())

    most_common = counter.most_common(MAX_VOCAB_SIZE - 2)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i

    def encode_gru(text):
        tokens = [vocab.get(w, 1) for w in text.split()]
        tokens = tokens[:MAX_SEQ_LEN]
        tokens += [0] * (MAX_SEQ_LEN - len(tokens))
        return tokens

    print("Tokenizing GRU inputs...")

    X_gru = np.array([
        encode_gru(text) for text in df["clean_text"]
    ])

    # =================================================
    # TRANSFORMER TOKENIZATION (DEBERTA)
    # =================================================

    print("Loading transformer tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base"
    )

    transformer_inputs = tokenizer(
        df["clean_text"].tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )

    # =================================================
    # TRAIN / VAL SPLIT (GRU ONLY SPLIT HERE)
    # =================================================

    X_train, X_val, y_train, y_val = train_test_split(
        X_gru,
        y,
        test_size=0.2,
        random_state=42
    )

    train_data = {
        "gru": (X_train, y_train),
        "transformer": transformer_inputs
    }

    val_data = {
        "gru": (X_val, y_val)
    }

    def save_vocab(vocab, path="models/vocab.json"):
        vocab_path = Path(path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        with vocab_path.open("w") as f:
            json.dump(vocab, f)

    save_vocab(vocab)

    return train_data, val_data


# =====================================================
# MODEL TRAINING LAYER (PLACEHOLDER ONLY)
# =====================================================

def train_gru(train_data, val_data):
    """
    Train GRU baseline model (to be implemented)
    """

    model = GRUModel()

    # TODO:
    # - training loop
    # - optimizer setup
    # - loss function (Focal Loss)
    # - validation step

    return model


def train_transformer(train_data, val_data):
    """
    Fine-tune transformer model (to be implemented)
    """

    model = load_transformer()

    # TODO:
    # - HuggingFace Trainer or custom loop
    # - multi-label training
    # - evaluation hooks

    return model


# =====================================================
# EVALUATION LAYER
# =====================================================

def evaluate(model, val_data, model_name="model"):
    """
    Evaluate model using ROC-AUC metric
    """

    # TODO:
    # - generate predictions
    # - compare against ground truth
    # - compute ROC-AUC per label

    scores = compute_roc_auc([], [])

    print(f"{model_name} evaluation results:", scores)

    return scores


# =====================================================
# MODEL SELECTION / EXPERIMENT TRACKING
# =====================================================

def select_best_model(gru_scores, transformer_scores):
    """
    Choose best performing model
    """

    if transformer_scores["mean_auc"] > gru_scores["mean_auc"]:
        return "transformer"
    else:
        return "gru"


# =====================================================
# MODEL SAVING LAYER
# =====================================================

def save_model(model, model_name):
    """
    Save trained model to disk for deployment
    """

    # TODO:
    # torch.save(model.state_dict(), "models/best_model.pt")
    pass


# =====================================================
# PIPELINE ORCHESTRATOR
# =====================================================

def run_pipeline():

    print("Loading dataset...")
    df = load_dataset()

    print("Preprocessing dataset...")
    df = preprocess_dataset(df)

    print("Sample cleaned text:")
    print(df["clean_text"].head())

    print("Building features...")
    train_data, val_data = build_features(df)

    print("Training GRU model...")
    gru_model = train_gru(train_data, val_data)

    print("Training Transformer model...")
    transformer_model = train_transformer(train_data, val_data)

    print("Evaluating models...")
    gru_scores = evaluate(gru_model, val_data, "GRU")
    transformer_scores = evaluate(transformer_model, val_data, "Transformer")

    print("Selecting best model...")
    best = select_best_model(gru_scores, transformer_scores)

    print("Saving best model:", best)
    save_model(
        gru_model if best == "gru" else transformer_model,
        best
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_pipeline()