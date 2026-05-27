"""Evaluation metrics focused on minimizing false negatives.

The wildfire setting is heavily imbalanced and the operational cost of a
missed fire (FN) is far higher than a false alarm (FP). We therefore lead
with recall, F-beta (beta=2), PR-AUC, and a threshold sweep.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0):
    """Return per-threshold metrics; pick threshold maximizing F-beta."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one extra precision/recall entry; align.
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        fbeta = np.nan_to_num(fbeta, nan=0.0)
    best_idx = int(np.argmax(fbeta))
    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "best_threshold": float(thresholds[best_idx]),
        "best_fbeta": float(fbeta[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
    }


def evaluate(y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0,
             fixed_threshold: float | None = None) -> dict:
    """Return a flat dict of metrics suitable for wandb logging."""
    sweep = threshold_sweep(y_true, y_score, beta=beta)
    thr = fixed_threshold if fixed_threshold is not None else sweep["best_threshold"]
    y_pred = (y_score >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(thr),
        f"f{int(beta)}": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fn_rate": float(fn / max(fn + tp, 1)),  # miss rate; lower is better
        "best_fbeta_on_sweep": sweep["best_fbeta"],
        "best_threshold_on_sweep": sweep["best_threshold"],
    }
