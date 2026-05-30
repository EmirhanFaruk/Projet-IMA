"""
hparam_search.py — Hyperparameter sensitivity study with full visualisation.

For each of 7 axes (optimizer, lr, batch_size, loss, preprocessing,
augmentation, patch_size), train each candidate config and produce:

  1. <axis>_results.json — per-config metrics
  2. <axis>_metrics_bar.png — 6-panel bar charts (Dice, IoU, Sens, Spec, Prec, AUC)
  3. <axis>_training_curves.png — training curves overlay
  4. hparam_summary_heatmap.png — final summary heatmap across all axes
  5. hparam_radar.png — radar chart of best config per axis

Each config is trained for HPARAM_FULL_EPOCHS (200 by default) on a single
fold/seed so the search runs in reasonable time.

Usage:
  python hparam_search.py --axes optimizer       # one axis
  python hparam_search.py --axes optimizer lr   # multiple
  python hparam_search.py                        # all 7 axes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import json
import os
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from utils import get_logger, ensure_dirs

logger = get_logger("hparam_search", C.LOG_DIR / "hparam_search.log")


HPARAM_AXES = {
    "optimizer": [
        {"name": "AdamW",        "OPTIMIZER": "adamw",   "LR": 1e-4},
        {"name": "Adam",         "OPTIMIZER": "adam",    "LR": 1e-4},
        {"name": "SGD+Nesterov", "OPTIMIZER": "sgd",     "LR": 1e-2},
        {"name": "RMSprop",      "OPTIMIZER": "rmsprop", "LR": 5e-4},
    ],
    "lr": [
        {"name": "LR=1e-2", "LR": 1e-2},
        {"name": "LR=5e-3", "LR": 5e-3},
        {"name": "LR=1e-3", "LR": 1e-3},
        {"name": "LR=5e-4", "LR": 5e-4},
        {"name": "LR=1e-4", "LR": 1e-4},
    ],
    "batch_size": [
        {"name": "Batch=16", "BATCH_SIZE": 16},
        {"name": "Batch=32", "BATCH_SIZE": 32},
        {"name": "Batch=64", "BATCH_SIZE": 64},
    ],
    "loss": [
        {"name": "BCE+Dice",   "LOSS_FN": "bce_dice"},
        {"name": "Focal+Dice", "LOSS_FN": "focal_dice"},
    ],
    "preprocessing": [
        {"name": "No CLAHE",    "USE_CLAHE": False},
        {"name": "CLAHE clip=1", "USE_CLAHE": True, "CLAHE_CLIP_LIMIT": 1.0},
        {"name": "CLAHE clip=2", "USE_CLAHE": True, "CLAHE_CLIP_LIMIT": 2.0},
        {"name": "CLAHE clip=4", "USE_CLAHE": True, "CLAHE_CLIP_LIMIT": 4.0},
    ],
    "augmentation": [
        {"name": "No aug", "AUG_LEVEL": "none"},
        {"name": "Light",  "AUG_LEVEL": "light"},
        {"name": "Medium", "AUG_LEVEL": "medium"},
        {"name": "Heavy",  "AUG_LEVEL": "heavy"},
    ],
    "patch_size": [
        {"name": "Patch=48", "PATCH_SIZE": 48},
        {"name": "Patch=64", "PATCH_SIZE": 64},
        {"name": "Patch=96", "PATCH_SIZE": 96},
    ],
}


# ─────────────────────────── Run helpers ─────────────────────────────────

def _run_one_config(axis: str, cfg: dict, epochs: int, fold: int, seed: int):
    """Train + evaluate a single hparam config. Returns the summary dict."""
    name = cfg["name"]
    tag  = f"hparam_{axis}_{name.replace(' ', '_').replace('=', '').replace('+','plus')}"

    overrides = {k: v for k, v in cfg.items() if k != "name"}
    env = os.environ.copy()
    for k, v in overrides.items():
        env[f"OVERRIDE_{k}"] = str(v)

    # Determine summary path BEFORE running
    sub = f"fold{fold}_seed{seed}_{tag}"
    summary_path = C.METRIC_DIR / "original" / sub / "summary.json"

    if summary_path.exists():
        logger.info(f"  [skip] {axis}/{name} — already done")
        with open(summary_path) as f:
            return json.load(f)

    # Train
    cmd_train = [
        sys.executable, "src/train.py",
        "--experiment", "original",
        "--fold",       str(fold),
        "--seed",       str(seed),
        "--epochs",     str(epochs),
        "--tag",        tag,
    ]
    logger.info(f"  Training: {axis}/{name} (overrides: {overrides})")
    subprocess.run(cmd_train, env=env, check=False)

    # Evaluate
    cmd_eval = [
        sys.executable, "src/evaluate.py",
        "--experiment", "original",
        "--fold",       str(fold),
        "--seed",       str(seed),
        "--tag",        tag,
        "--tta",
    ]
    logger.info(f"  Evaluating: {axis}/{name}")
    subprocess.run(cmd_eval, env=env, check=False)

    # Load summary
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


# ─────────────────────────── Per-axis runner ─────────────────────────────

def run_axis(axis: str, epochs: int, fold: int, seed: int):
    configs = HPARAM_AXES[axis]
    results = []
    for cfg in configs:
        s = _run_one_config(axis, cfg, epochs, fold, seed)
        if s:
            s["name"] = cfg["name"]
            s["overrides"] = {k: v for k, v in cfg.items() if k != "name"}
            results.append(s)

    out_dir = C.HPARAM_DIR
    ensure_dirs(out_dir)
    with open(out_dir / f"{axis}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {axis}_results.json ({len(results)} configs)")

    plot_axis_metrics_bar(axis, results, out_dir)
    plot_axis_training_curves(axis, results, fold, seed, out_dir)
    return results


# ─────────────────────────── Visualisations ──────────────────────────────

def plot_axis_metrics_bar(axis: str, results: list, out_dir: Path):
    """6-panel bar chart: Dice / IoU / Sens / Spec / Prec / AUC."""
    if not results:
        return
    names = [r["name"] for r in results]
    metrics = [
        ("dice_mean",        "Dice"),
        ("iou_mean",         "IoU"),
        ("sensitivity_mean", "Sensitivity"),
        ("specificity_mean", "Specificity"),
        ("precision_mean",   "Precision"),
        ("auc_roc_global",   "AUC-ROC"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Hyperparameter axis: {axis}", fontsize=14, fontweight="bold")

    palette = sns.color_palette("Set2", n_colors=len(names))
    for ax, (key, label) in zip(axes.flat, metrics):
        vals = [r.get(key, 0.0) for r in results]
        bars = ax.bar(names, vals, color=palette, edgecolor="black", lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, max(1.0, max(vals) * 1.10))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / f"{axis}_metrics_bar.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_axis_training_curves(axis: str, results: list, fold: int, seed: int,
                               out_dir: Path):
    """Overlay validation Dice training curves for each config."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Training curves — {axis} sweep")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.grid(True, alpha=0.3)

    palette = sns.color_palette("Set2", n_colors=len(results))
    for col, r in zip(palette, results):
        name = r["name"]
        tag  = f"hparam_{axis}_{name.replace(' ', '_').replace('=','').replace('+','plus')}"
        sub  = f"fold{fold}_seed{seed}_{tag}"
        hist_path = C.METRIC_DIR / "original" / sub / "original_history.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            h = json.load(f)
        epochs = list(range(1, len(h.get("val_dice", [])) + 1))
        ax.plot(epochs, h["val_dice"], label=name, lw=1.5, color=col)

    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f"{axis}_training_curves.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(out_dir: Path):
    """Heatmap of Dice across all axes and their configs."""
    rows = []
    for axis, configs in HPARAM_AXES.items():
        results_path = out_dir / f"{axis}_results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            results = json.load(f)
        for r in results:
            rows.append({
                "axis":   axis,
                "config": r["name"],
                "dice":   r.get("dice_mean", 0.0),
                "auc":    r.get("auc_roc_global", 0.0),
                "iou":    r.get("iou_mean", 0.0),
            })
    if not rows:
        return

    # Build a matrix: rows=configs (with axis prefix), cols=metrics
    labels = [f"[{r['axis']}] {r['config']}" for r in rows]
    matrix = np.array([[r["dice"], r["iou"], r["auc"]] for r in rows])

    fig, ax = plt.subplots(figsize=(8, max(6, len(labels) * 0.32)))
    sns.heatmap(matrix, annot=True, fmt=".4f",
                yticklabels=labels, xticklabels=["Dice", "IoU", "AUC-ROC"],
                cmap="YlGnBu", cbar=True, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Hyperparameter sensitivity — all axes")
    plt.tight_layout()
    fig.savefig(out_dir / "hparam_summary_heatmap.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary_radar(out_dir: Path):
    """Radar chart of best config per axis (multi-metric view)."""
    metrics = ["dice_mean", "iou_mean", "sensitivity_mean", "specificity_mean",
               "precision_mean", "auc_roc_global"]
    metric_labels = ["Dice", "IoU", "Sens", "Spec", "Prec", "AUC"]

    bests = {}
    for axis, _ in HPARAM_AXES.items():
        results_path = out_dir / f"{axis}_results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            results = json.load(f)
        if not results:
            continue
        best = max(results, key=lambda r: r.get("dice_mean", 0))
        bests[axis] = best

    if not bests:
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    palette = sns.color_palette("Set2", n_colors=len(bests))

    for col, (axis, best) in zip(palette, bests.items()):
        vals = [best.get(m, 0.0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, label=f"{axis}: {best['name']}", color=col)
        ax.fill(angles, vals, alpha=0.10, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Best config per axis — multi-metric comparison",
                 y=1.10, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.05), fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "hparam_radar.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────── Entry point ─────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axes", nargs="*", default=None,
                   help="Subset of axes to run (default: all 7)")
    p.add_argument("--epochs", type=int, default=C.HPARAM_FULL_EPOCHS)
    p.add_argument("--fold",   type=int, default=0)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    axes = args.axes or list(HPARAM_AXES.keys())
    out_dir = C.HPARAM_DIR
    ensure_dirs(out_dir)

    t0 = time.time()
    for axis in axes:
        logger.info(f"\n=== AXIS: {axis} ===")
        run_axis(axis, args.epochs, args.fold, args.seed)
        logger.info(f"  Elapsed total: {(time.time()-t0)/3600:.2f} h")

    plot_summary_heatmap(out_dir)
    plot_summary_radar(out_dir)
    logger.info(f"\nAll done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
