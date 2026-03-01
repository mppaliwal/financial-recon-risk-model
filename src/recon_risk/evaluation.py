from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    """Capacity-aware thresholding and metric reporting."""

    @staticmethod
    def choose_threshold(y_prob: np.ndarray, k_frac: float) -> float:
        n = len(y_prob)
        if n == 0:
            raise ValueError("Cannot choose threshold on empty probability array.")
        k = max(1, int(np.ceil(k_frac * n)))
        return float(np.partition(y_prob, -k)[-k])

    @staticmethod
    def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
        if len(y_true) == 0:
            raise ValueError("Cannot compute metrics on empty evaluation set.")

        y_pred = (y_prob >= threshold).astype(int)
        roc_auc = 0.0
        if np.unique(y_true).shape[0] > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))

        return {
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "roc_auc": roc_auc,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "threshold": float(threshold),
            "flag_rate": float(y_pred.mean()),
        }
