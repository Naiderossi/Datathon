"""Utility functions to evaluate model predictions."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        'threshold': float(threshold),
        'f1': float(f1_score(y_true, y_pred)),
        'classification_report': report,
        'confusion_matrix': matrix,
    }


__all__ = ['compute_metrics']
