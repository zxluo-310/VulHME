from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    y_pred = probs.argmax(axis=1)
    result = {
        "accuracy": accuracy_score(y_true, y_pred) * 100.0,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100.0,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100.0,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100.0,
    }
    try:
        result["auc"] = roc_auc_score(y_true, probs[:, 1]) * 100.0
    except ValueError:
        result["auc"] = float("nan")
    return result


def average_metric_dict(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys() if items else []
    return {key: float(np.nanmean([item[key] for item in items])) for key in keys}
