import pandas as pd
import torch
from model import GRUModel, load_transformer
from preprocess import TextPreprocessor
from loss import FocalLoss

def train_gru():

    model = GRUModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = FocalLoss()

    print("Training GRU baseline...")

    # TODO:
    # - DataLoader
    # - batching
    # - training loop
    # - validation

    return model


def train_transformer():

    model = load_transformer()

    print("Training Transformer model...")

    # TODO:
    # HuggingFace Trainer API
    # tokenization
    # evaluation using ROC-AUC

    return model


if __name__ == "__main__":
    train_gru()
    train_transformer()