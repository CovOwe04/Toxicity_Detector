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
            if len(np.unique(y_true[:, i])) < 2:
                scores[label] = 0.0
                continue

            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            scores[label] = 0.0 if np.isnan(score) else score
        except Exception:
            scores[label] = 0.0

    scores["mean_auc"] = float(np.mean(list(scores.values())))

    return scores
