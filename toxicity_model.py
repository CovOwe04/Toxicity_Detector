import re

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

def preprocess(text):

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    return text.strip()

def load_models():

    # TODO
    # Load GRU model
    # Load Transformer model

    return None, None

gru_model, transformer_model = load_models()

def predict(text):

    text = preprocess(text)

    # TODO
    # Actual inference

    return {
        "comment": text,
        "toxicity_score": 0.35,
        "labels": ["toxic"]
    }