from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def cosine_alignment_metrics(z_a: np.ndarray, y_a: np.ndarray, z_p: np.ndarray, y_p: np.ndarray) -> Dict[str, float]:
    if len(z_a) == 0 or len(z_p) == 0:
        return {"same_label_cosine": 0.0, "different_label_cosine": 0.0}
    z_a = z_a / (np.linalg.norm(z_a, axis=1, keepdims=True) + 1e-8)
    z_p = z_p / (np.linalg.norm(z_p, axis=1, keepdims=True) + 1e-8)
    sim = z_a @ z_p.T
    same = y_a[:, None] == y_p[None, :]
    diff = ~same
    return {
        "same_label_cosine": float(sim[same].mean()) if same.any() else 0.0,
        "different_label_cosine": float(sim[diff].mean()) if diff.any() else 0.0,
    }
