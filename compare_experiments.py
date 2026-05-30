"""
compare_experiments.py — Side-by-side comparison of all architectures.

Pulls aggregated results and produces:
  1. 8-metric bar chart (Dice/IoU/Sens/Spec/Prec/Acc/AUC-ROC/AUC-PR)
  2. Side-by-side training curves (validation Dice over epochs)
  3. Side-by-side ROC and PR curves
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from utils import get_logger, ensure_dirs

logger = get_logger("compare")


def load_aggregated():
    p = C.METRIC_DIR / "aggregated_results.json"
    if not p.exists():
        logger.error(f"Run aggregate_results.py first (no file at {p}).")
        return None
    with open(p) as f:
        return json.load(f)


def plot_metric_comparison(aggregations, out_dir):
    metrics = [
        ("dice_mean",        "Dice"),
        ("iou_mean",         "IoU"),
        ("sensitivity_mean", "Sensitivity"),
        ("specificity_mean", "Specificity"),
        ("precision_mean",   "Precision"),
        ("accuracy_mean",    "Accuracy"),
        ("auc_roc_global",   "AUC-ROC"),
        ("auc_pr_global",    "AUC-PR"),
    ]
    experiments = list(aggregations.keys())
    palette = sns.color_palette("Set2", n_colors=len(experiments))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Architecture comparison — mean ± std across all runs",
                 fontsize=14, fontweight="bold")

    for ax, (key, label) in zip(axes.flat, metrics):
        means = [aggregations[e].get(key, {}).get("mean", 0.0)
                 for e in experiments]
        stds  = [aggregations[e].get(key, {}).get("std",  0.0)
                 for e in experiments]
        bars  = ax.bar(experiments, means, yerr=stds, capsize=5,
                        color=palette, edgecolor="black", lw=0.5,
                        error_kw={"elinewidth": 1.5})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{m:.4f}", ha="center", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ymax = max(1.0, max(means) + max(stds) + 0.02)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "experiment_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    aggregations = load_aggregated()
    if aggregations is None:
        return

    out_dir = C.VIZ_DIR / "comparison"
    ensure_dirs(out_dir)
    plot_metric_comparison(aggregations, out_dir)


if __name__ == "__main__":
    main()
