import pandas as pd

def train():

    df = pd.read_csv(
        "data/train.csv"
    )

    print(
        f"Loaded {len(df)} records"
    )

    # TODO
    # preprocessing
    # GRU training
    # Transformer training
    # save models

if __name__ == "__main__":
    train()