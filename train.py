import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer

from loss import FocalLoss
from metrics import compute_roc_auc
from model import DEFAULT_TRANSFORMER_MODEL, GRUModel, load_transformer
from preprocess import TextPreprocessor


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "data/train.csv"
MODEL_DIR = Path("models")
VOCAB_PATH = MODEL_DIR / "vocab.json"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"

MAX_VOCAB_SIZE = 50000
MAX_SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-3
DEFAULT_MAX_ROWS = 5000

TRANSFORMER_BATCH_SIZE = 4
TRANSFORMER_EPOCHS = 1
TRANSFORMER_LEARNING_RATE = 1e-6
TRANSFORMER_MAX_BATCHES = 50

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
    Load raw Jigsaw dataset.
    """
    df = pd.read_csv(path)
    return df[["comment_text"] + LABELS]


# =====================================================
# PREPROCESSING LAYER
# =====================================================

def preprocess_dataset(df):
    """
    Apply text normalization + cleaning pipeline.
    """

    processor = TextPreprocessor()

    df = df.copy()
    df["clean_text"] = df["comment_text"].astype(str).apply(processor.normalize)
    df = df[df["clean_text"].str.len() > 0]

    return df


# =====================================================
# FEATURE ENGINEERING LAYER
# =====================================================

def save_vocab(vocab, path=VOCAB_PATH):
    vocab_path = Path(path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with vocab_path.open("w") as f:
        json.dump(vocab, f)


def build_vocab(texts, max_vocab_size=MAX_VOCAB_SIZE):
    counter = Counter()

    for text in texts:
        counter.update(text.split())

    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i

    return vocab


def encode_gru(text, vocab, max_seq_len=MAX_SEQ_LEN):
    tokens = [vocab.get(word, 1) for word in text.split()]
    tokens = tokens[:max_seq_len]
    tokens += [0] * (max_seq_len - len(tokens))
    return tokens


def build_features(df):
    """
    Convert cleaned text into model-ready train/validation payloads.
    """

    print("Building GRU vocabulary...")
    vocab = build_vocab(df["clean_text"])
    save_vocab(vocab)

    print("Tokenizing GRU inputs...")
    X_gru = np.array(
        [encode_gru(text, vocab) for text in df["clean_text"]],
        dtype=np.int64,
    )
    y = df[LABELS].values.astype(np.float32)
    texts = df["clean_text"].tolist()

    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
    )

    train_data = {
        "gru": {
            "inputs": torch.tensor(X_gru[train_idx], dtype=torch.long),
            "labels": torch.tensor(y[train_idx], dtype=torch.float32),
            "texts": [texts[i] for i in train_idx],
            "vocab_size": len(vocab),
        },
        "transformer": {
            "texts": [texts[i] for i in train_idx],
            "labels": torch.tensor(y[train_idx], dtype=torch.float32),
        },
    }

    val_data = {
        "gru": {
            "inputs": torch.tensor(X_gru[val_idx], dtype=torch.long),
            "labels": torch.tensor(y[val_idx], dtype=torch.float32),
            "texts": [texts[i] for i in val_idx],
            "vocab_size": len(vocab),
        },
        "transformer": {
            "texts": [texts[i] for i in val_idx],
            "labels": torch.tensor(y[val_idx], dtype=torch.float32),
        },
    }

    return train_data, val_data


class TransformerTextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        return item

def print_sample_probabilities(model_name, sample_text, probabilities):
    print(f"{model_name} sample text:")
    print(sample_text)
    print(f"{model_name} sample probabilities:")
    print("  Label           Probability")
    print("  --------------  -----------")

    for label, score in zip(LABELS, probabilities):
        print(f"  {label:<14}  {score * 100:>10.2f}%")

# =====================================================
# MODEL TRAINING LAYER
# =====================================================

def train_gru(
    train_data,
    val_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    max_batches=None,
):
    """
    Train the GRU baseline model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru_train = train_data["gru"]

    dataset = TensorDataset(gru_train["inputs"], gru_train["labels"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUModel(vocab_size=gru_train["vocab_size"]).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen_examples = 0

        for batch_idx, (batch_inputs, batch_labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size_seen = batch_inputs.size(0)
            total_loss += loss.item() * batch_size_seen
            seen_examples += batch_size_seen

        avg_loss = total_loss / max(seen_examples, 1)
        print(f"GRU epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    print_gru_sample_output(model, val_data, device)

    return model

def print_gru_sample_output(model, val_data, device):
    """
    Print one validation probability vector for the exact shown sample text.
    """

    gru_val = val_data["gru"]

    if len(gru_val["inputs"]) == 0:
        return

    model.eval()
    sample_input = gru_val["inputs"][0].unsqueeze(0).to(device)
    sample_text = gru_val.get("texts", [""])[0]

    with torch.no_grad():
        logits = model(sample_input)

        if not torch.isfinite(logits).all():
            raise RuntimeError("GRU produced non-finite sample logits.")

        probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    print_sample_probabilities("GRU", sample_text, probabilities)

def train_transformer(
    train_data,
    val_data,
    model_name=DEFAULT_TRANSFORMER_MODEL,
    epochs=TRANSFORMER_EPOCHS,
    batch_size=TRANSFORMER_BATCH_SIZE,
    learning_rate=TRANSFORMER_LEARNING_RATE,
    max_seq_len=MAX_SEQ_LEN,
    max_batches=None,
):
    """
    Fine-tune the transformer classification head/model on the toxicity labels.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_train = train_data["transformer"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TransformerTextDataset(
        transformer_train["texts"],
        transformer_train["labels"],
        tokenizer,
        max_seq_len=max_seq_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = load_transformer(model_name).float().to(device)
    model.transformer_model_name = model_name
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen_examples = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Transformer loss became non-finite. "
                    "Try a lower --transformer-lr, smaller --max-seq-len, or fewer rows."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_seen = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size_seen
            seen_examples += batch_size_seen

        avg_loss = total_loss / max(seen_examples, 1)
        print(f"Transformer epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    print_transformer_sample_output(
        model,
        tokenizer,
        val_data["transformer"],
        device,
        max_seq_len=max_seq_len,
    )

    return model


def print_transformer_sample_output(model, tokenizer, transformer_val, device, max_seq_len=MAX_SEQ_LEN):
    """
    Print one validation probability vector so the workflow has visible output.
    """

    if not transformer_val["texts"]:
        return

    model.eval()
    text = transformer_val["texts"][0]
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits

        if not torch.isfinite(logits).all():
            raise RuntimeError("Transformer produced non-finite sample logits.")

        probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    print_sample_probabilities("Transformer", text, probabilities)


# =====================================================
# EVALUATION LAYER
# =====================================================

def evaluate(model, val_data, model_name="model", batch_size=BATCH_SIZE):
    """
    Evaluate a trained GRU model using ROC-AUC.
    """

    device = next(model.parameters()).device
    gru_val = val_data["gru"]

    dataset = TensorDataset(gru_val["inputs"], gru_val["labels"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            predictions.append(probabilities)
            targets.append(batch_labels.numpy())

    y_pred = np.vstack(predictions)
    y_true = np.vstack(targets)

    scores = compute_roc_auc(y_true, y_pred)
    print(f"{model_name} evaluation results:", scores)

    return scores


def evaluate_transformer(
    model,
    val_data,
    model_name="Transformer",
    batch_size=TRANSFORMER_BATCH_SIZE,
    max_seq_len=MAX_SEQ_LEN,
):
    """
    Evaluate the Transformer model using ROC-AUC.
    """

    device = next(model.parameters()).device

    tokenizer = AutoTokenizer.from_pretrained(
        getattr(model, "transformer_model_name", DEFAULT_TRANSFORMER_MODEL)
    )

    dataset = TransformerTextDataset(
        val_data["transformer"]["texts"],
        val_data["transformer"]["labels"],
        tokenizer,
        max_seq_len=max_seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits

            probabilities = torch.sigmoid(logits).cpu().numpy()

            predictions.append(probabilities)
            targets.append(labels.numpy())

    y_pred = np.vstack(predictions)
    y_true = np.vstack(targets)

    scores = compute_roc_auc(y_true, y_pred)

    print(f"{model_name} evaluation results:")
    print(scores)

    return scores


# =====================================================
# MODEL SELECTION / EXPERIMENT TRACKING
# =====================================================

def select_best_model(gru_scores, transformer_scores=None):
    """
    Choose best performing model.
    """

    if (
        transformer_scores is not None
        and transformer_scores["mean_auc"] > gru_scores["mean_auc"]
    ):
        return "transformer"

    return "gru"


# =====================================================
# MODEL SAVING LAYER
# =====================================================

def save_model(model, model_name, path=BEST_MODEL_PATH):
    """
    Save trained model checkpoint for later workflow testing.
    """

    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Refusing to save {model_name}: model contains non-finite weights.")

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_type": model_name,
        "state_dict": model.cpu().state_dict(),
        "labels": LABELS,
    }

    if model_name == "gru" and hasattr(model, "config"):
        checkpoint["model_config"] = model.config

    if model_name == "transformer":
        checkpoint["transformer_model_name"] = getattr(
            model,
            "transformer_model_name",
            DEFAULT_TRANSFORMER_MODEL,
        )

    torch.save(checkpoint, model_path)
    print(f"Saved {model_name} checkpoint to {model_path}")


def save_workflow_model(model, model_name):
    path = MODEL_DIR / f"{model_name}_model.pt"
    save_model(model, model_name, path)


# =====================================================
# PIPELINE ORCHESTRATOR
# =====================================================

def run_pipeline(args=None):
    args = args or parse_args()

    print("Loading dataset...")
    df = load_dataset(args.data_path)

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Using first {len(df)} rows for this run.")

    print("Preprocessing dataset...")
    df = preprocess_dataset(df)

    print("Sample cleaned text:")
    print(df["clean_text"].head())

    print("Building features...")
    train_data, val_data = build_features(df)

    trained_models = {}

    if args.model in ("gru", "both"):
        print("Training GRU model...")
        gru_model = train_gru(
            train_data,
            val_data,
            epochs=args.gru_epochs,
            batch_size=args.gru_batch_size,
            learning_rate=args.gru_lr,
            max_batches=args.gru_max_batches,
        )
        trained_models["gru"] = gru_model
        save_workflow_model(gru_model, "gru")

    if args.model in ("transformer", "both"):
        print("Training Transformer model...")
        transformer_model = train_transformer(
            train_data,
            val_data,
            model_name=args.transformer_model,
            epochs=args.transformer_epochs,
            batch_size=args.transformer_batch_size,
            learning_rate=args.transformer_lr,
            max_seq_len=args.max_seq_len,
            max_batches=args.transformer_max_batches,
        )
        trained_models["transformer"] = transformer_model
        save_workflow_model(transformer_model, "transformer")


    # =================================================
    # MODEL EVALUATION
    # =================================================

    scores = {}

    if "gru" in trained_models:
        print("\nEvaluating GRU model...")
        scores["gru"] = evaluate(trained_models["gru"], val_data, model_name="GRU")

    if "transformer" in trained_models:
        print("\nEvaluating Transformer model...")
        scores["transformer"] = evaluate_transformer(trained_models["transformer"], val_data, model_name="Transformer")


    # =================================================
    # MODEL SELECTION
    # =================================================

    if "gru" in scores and "transformer" in scores:
        best_model_name = select_best_model(scores["gru"], scores["transformer"])
    elif "gru" in scores:
        best_model_name = "gru"
    else:
        best_model_name = "transformer"

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")

    for name, result in scores.items():
        print(name.upper())
        print(result)
        print()

    print("BEST MODEL:", best_model_name)


    # =================================================
    # SAVE BEST MODEL
    # =================================================

    save_model(trained_models[best_model_name], best_model_name, BEST_MODEL_PATH)
    print("\nTraining workflow complete.")


# =====================================================
# CLI
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train toxicity detection models.")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--model", choices=["gru", "transformer", "both"], default="both")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)

    parser.add_argument("--gru-epochs", type=int, default=EPOCHS)
    parser.add_argument("--gru-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gru-lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--gru-max-batches", type=int, default=None)

    parser.add_argument("--transformer-model", default=DEFAULT_TRANSFORMER_MODEL)
    parser.add_argument("--transformer-epochs", type=int, default=TRANSFORMER_EPOCHS)
    parser.add_argument("--transformer-batch-size", type=int, default=TRANSFORMER_BATCH_SIZE)
    parser.add_argument("--transformer-lr", type=float, default=TRANSFORMER_LEARNING_RATE)
    parser.add_argument("--transformer-max-batches", type=int, default=TRANSFORMER_MAX_BATCHES)

    args = parser.parse_args()

    if args.max_batches is not None:
        args.gru_max_batches = args.max_batches
        args.transformer_max_batches = args.max_batches

    return args


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_pipeline()






