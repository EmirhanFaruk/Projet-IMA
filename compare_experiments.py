"""
compare_experiments.py — Side-by-side comparison of all experiments.

Loads saved summary JSONs and produces:
  • Comparison bar charts for all metrics
  • LaTeX-ready results table (printed to stdout)
  • Statistical significance summary

Usage:
    python compare_experiments.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from utils import ensure_dirs, get_logger

logger = get_logger("compare")


EXPERIMENTS = ["original", "msca", "deva", "ablation_spatial", "ablation_channel"]
DISPLAY_NAMES = {
    "original":          "UNet (Original)",
    "msca":              "UNet + MSCA (Full)",
    "deva":              "UNet + DEVA (New Method)",
    "ablation_spatial":  "UNet + MSCA (Spatial only)",
    "ablation_channel":  "UNet + MSCA (Channel only)",
}
METRICS_TO_PLOT = [
    ("dice_mean",        "Dice"),
    ("iou_mean",         "IoU"),
    ("sensitivity_mean", "Sensitivity"),
    ("specificity_mean", "Specificity"),
    ("auc_roc_global",   "AUC-ROC"),
    ("auc_pr_global",    "AUC-PR"),
    ("precision_mean",   "Precision"),
    ("accuracy_mean",    "Accuracy"),
]


def load_summaries():
    summaries = {}
    for exp in EXPERIMENTS:
        p = C.METRIC_DIR / f"{exp}_summary.json"
        if p.exists():
            with open(p) as f:
                summaries[exp] = json.load(f)
        else:
            logger.warning(f"No summary found for: {exp}")
    return summaries


def plot_comparison(summaries: dict, out_dir: Path):
    if not summaries:
        logger.warning("No summaries to compare.")
        return

    exps     = list(summaries.keys())
    n_cols   = 4
    n_rows   = (len(METRICS_TO_PLOT) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    fig.suptitle("Experiment Comparison — DRIVE Test Set", fontsize=15,
                 fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for ax_idx, (metric_key, metric_name) in enumerate(METRICS_TO_PLOT):
        ax = axes[ax_idx]
        vals  = [summaries[e].get(metric_key, 0.0) for e in exps]
        names = [DISPLAY_NAMES.get(e, e) for e in exps]

        bars = ax.bar(range(len(exps)), vals,
                      color=colors[:len(exps)],
                      edgecolor="black", linewidth=0.5)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(metric_name, fontsize=11)
        ax.set_ylim(max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.05))
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused axes
    for ax in axes[len(METRICS_TO_PLOT):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = out_dir / "experiment_comparison.png"
    ensure_dirs(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison chart → {out_path}")


def print_latex_table(summaries: dict):
    """Print a LaTeX table of results for the paper."""
    metrics_print = [
        ("dice_mean",        "Dice"),
        ("iou_mean",         "IoU"),
        ("sensitivity_mean", "Sens."),
        ("specificity_mean", "Spec."),
        ("auc_roc_global",   "AUC-ROC"),
        ("auc_pr_global",    "AUC-PR"),
    ]

    header_cols = " & ".join(["Method"] + [m[1] for m in metrics_print])
    print("\n% LaTeX Results Table")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Retinal vessel segmentation on DRIVE test set.}")
    print(f"\\begin{{tabular}}{{l{'c' * len(metrics_print)}}}")
    print("\\hline")
    print(f"{header_cols} \\\\")
    print("\\hline")

    for exp, summ in summaries.items():
        name = DISPLAY_NAMES.get(exp, exp)
        cols = [name]
        for mk, _ in metrics_print:
            v = summ.get(mk, float("nan"))
            cols.append(f"{v:.4f}")
        print(" & ".join(cols) + " \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:results}")
    print("\\end{table}\n")


def main():
    summaries = load_summaries()
    if not summaries:
        logger.error("Run evaluate.py for each experiment first.")
        return

    plot_comparison(summaries, C.VIZ_DIR)
    print_latex_table(summaries)

    # Delta vs baseline
    if "original" in summaries and "enhanced" in summaries:
        base = summaries["original"]
        enha = summaries["enhanced"]
        print("\n=== MSCA improvement over baseline ===")
        for mk, mn in METRICS_TO_PLOT:
            delta = enha.get(mk, 0) - base.get(mk, 0)
            sign  = "+" if delta >= 0 else ""
            print(f"  {mn:15s}: {sign}{delta:.4f}")


if __name__ == "__main__":
    main()
