"""
metrics.py — Complete evaluation metrics for retinal vessel segmentation.

All metrics are computed ONLY inside the FOV mask (as per DRIVE standard).

Metrics:
  • Dice / F1
  • IoU (Jaccard)
  • Accuracy
  • Sensitivity (Recall / TPR)
  • Specificity (TNR)
  • Precision
  • AUC-ROC
  • AUC-PR
  • Confusion matrix (TP, FP, TN, FN)

Also includes:
  • find_best_threshold() — grid search over Dice
  • MetricAccumulator     — running average across batches
"""

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

import config as C


# ──────────────────────────────────────────────────────────────────────────────
# Core per-image metric function
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(probs: np.ndarray,
                    targets: np.ndarray,
                    fov: Optional[np.ndarray] = None,
                    threshold: float = 0.5) -> dict:
    """
    Compute all segmentation metrics.

    Args:
        probs   : probability map (H, W) float32 in [0,1]
        targets : ground truth binary mask (H, W) uint8 {0,1}
        fov     : FOV mask (H, W) {0,1}. If None, use all pixels.
        threshold: binarisation threshold

    Returns:
        dict of metric name → float value
    """
    # Flatten and apply FOV
    if fov is not None:
        mask = fov.flatten().astype(bool)
    else:
        mask = np.ones(probs.size, dtype=bool)

    p  = probs.flatten()[mask]
    t  = targets.flatten()[mask].astype(np.float32)

    # Binarise
    pred = (p >= threshold).astype(np.float32)

    # ── Confusion matrix ──────────────────────────────────────────────────
    tp = ((pred == 1) & (t == 1)).sum()
    fp = ((pred == 1) & (t == 0)).sum()
    tn = ((pred == 0) & (t == 0)).sum()
    fn = ((pred == 0) & (t == 1)).sum()

    eps = 1e-8

    dice        = (2 * tp) / (2 * tp + fp + fn + eps)
    iou         = tp / (tp + fp + fn + eps)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)             # recall
    specificity = tn / (tn + fp + eps)
    precision   = tp / (tp + fp + eps)
    f1          = (2 * precision * sensitivity) / (precision + sensitivity + eps)

    # ── AUC scores (need probabilities, not binary) ───────────────────────
    try:
        auc_roc = roc_auc_score(t.astype(int), p)
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = average_precision_score(t.astype(int), p)
    except ValueError:
        auc_pr = float("nan")

    return {
        "dice":        float(dice),
        "iou":         float(iou),
        "accuracy":    float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "f1":          float(f1),
        "auc_roc":     float(auc_roc),
        "auc_pr":      float(auc_pr),
        "tp":          int(tp),
        "fp":          int(fp),
        "tn":          int(tn),
        "fn":          int(fn),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Threshold search
# ──────────────────────────────────────────────────────────────────────────────

def find_best_threshold(probs_list: list[np.ndarray],
                        targets_list: list[np.ndarray],
                        fov_list: list[np.ndarray],
                        lo: float = C.THRESH_LO,
                        hi: float = C.THRESH_HI,
                        step: float = C.THRESH_STEP) -> dict:
    """
    Grid-search threshold on a list of probability maps.
    Returns dict of threshold → mean Dice, and the best threshold.
    """
    thresholds = np.arange(lo, hi + step, step)
    results    = {}

    for thr in thresholds:
        dices = []
        for p, t, fov in zip(probs_list, targets_list, fov_list):
            m = compute_metrics(p, t, fov, threshold=thr)
            dices.append(m["dice"])
        results[float(round(thr, 4))] = float(np.mean(dices))

    best_thr = max(results, key=results.__getitem__)
    return {"scores": results, "best_threshold": best_thr,
            "best_dice": results[best_thr]}


# ──────────────────────────────────────────────────────────────────────────────
# ROC / PR curve data
# ──────────────────────────────────────────────────────────────────────────────

def compute_roc_data(probs_list, targets_list, fov_list):
    """
    Aggregate all pixels (inside FOV) from multiple images and
    return (fpr, tpr, thresholds, auc_roc).
    """
    all_p, all_t = [], []
    for p, t, fov in zip(probs_list, targets_list, fov_list):
        if fov is not None:
            m = fov.flatten().astype(bool)
        else:
            m = np.ones(p.size, dtype=bool)
        all_p.append(p.flatten()[m])
        all_t.append(t.flatten()[m])

    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t).astype(int)

    fpr, tpr, thr = roc_curve(all_t, all_p)
    auc            = roc_auc_score(all_t, all_p)
    return fpr, tpr, thr, auc


def compute_pr_data(probs_list, targets_list, fov_list):
    """Return (precision, recall, thresholds, auc_pr)."""
    all_p, all_t = [], []
    for p, t, fov in zip(probs_list, targets_list, fov_list):
        if fov is not None:
            m = fov.flatten().astype(bool)
        else:
            m = np.ones(p.size, dtype=bool)
        all_p.append(p.flatten()[m])
        all_t.append(t.flatten()[m])

    all_p = np.concatenate(all_p)
    all_t = np.concatenate(all_t).astype(int)

    prec, rec, thr = precision_recall_curve(all_t, all_p)
    auc             = average_precision_score(all_t, all_p)
    return prec, rec, thr, auc


# ──────────────────────────────────────────────────────────────────────────────
# Accumulator (used during training for running mean)
# ──────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """Running sum / count for online mean tracking during an epoch."""

    def __init__(self):
        self._sums   : dict[str, float] = {}
        self._counts : dict[str, int]   = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if np.isnan(float(v)):
                continue
            self._sums[k]   = self._sums.get(k, 0.0)   + float(v)
            self._counts[k] = self._counts.get(k, 0)    + 1

    def mean(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums if self._counts[k] > 0}

    def reset(self):
        self._sums   = {}
        self._counts = {}
