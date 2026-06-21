import numpy as np
from sklearn.metrics import roc_auc_score

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def compute_roc_auc(y_true, y_pred):

    scores = {}

    for i, label in enumerate(LABELS):

        try:
            scores[label] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            scores[label] = 0.0

    scores["mean_auc"] = np.mean(list(scores.values()))

    return scores