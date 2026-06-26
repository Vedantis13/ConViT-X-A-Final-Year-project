"""
metrics.py — Evaluation metrics for multi-label classification.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve
)
import config


def compute_metrics(all_labels, all_probs, threshold=0.5):
    """
    Compute per-class and mean metrics.

    all_labels: [N, C] numpy array (binary)
    all_probs:  [N, C] numpy array (sigmoid probabilities)
    Returns dict with per-class and mean metrics.
    """
    results = {
        "auroc":    [],
        "auprc":    [],
        "f1":       [],
        "precision": [],
        "recall":   [],
    }

    for i, disease in enumerate(config.DISEASES):
        y_true = all_labels[:, i]
        y_prob  = all_probs[:, i]
        y_pred  = (y_prob >= threshold).astype(int)

        # Skip if only one class present (AUROC undefined)
        if len(np.unique(y_true)) < 2:
            results["auroc"].append(np.nan)
            results["auprc"].append(np.nan)
        else:
            results["auroc"].append(roc_auc_score(y_true, y_prob))
            results["auprc"].append(average_precision_score(y_true, y_prob))

        results["f1"].append(f1_score(y_true, y_pred, zero_division=0))
        results["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        results["recall"].append(recall_score(y_true, y_pred, zero_division=0))

    # Mean (ignoring NaN)
    summary = {}
    for key, vals in results.items():
        arr = np.array(vals, dtype=float)
        summary[f"mean_{key}"] = float(np.nanmean(arr))
        for i, disease in enumerate(config.DISEASES):
            summary[f"{disease}_{key}"] = float(arr[i])

    return summary


def format_metrics(summary):
    """Return a human-readable string of key metrics."""
    lines = [
        f"{'Disease':<22} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'Prec':>6} {'Rec':>6}",
        "─" * 60,
    ]
    for d in config.DISEASES:
        lines.append(
            f"{d:<22} "
            f"{summary.get(f'{d}_auroc', float('nan')):7.4f} "
            f"{summary.get(f'{d}_auprc', float('nan')):7.4f} "
            f"{summary.get(f'{d}_f1', float('nan')):6.4f} "
            f"{summary.get(f'{d}_precision', float('nan')):6.4f} "
            f"{summary.get(f'{d}_recall', float('nan')):6.4f}"
        )
    lines += [
        "─" * 60,
        f"{'MEAN':<22} "
        f"{summary.get('mean_auroc', float('nan')):7.4f} "
        f"{summary.get('mean_auprc', float('nan')):7.4f} "
        f"{summary.get('mean_f1', float('nan')):6.4f} "
        f"{summary.get('mean_precision', float('nan')):6.4f} "
        f"{summary.get('mean_recall', float('nan')):6.4f}",
    ]
    return "\n".join(lines)
