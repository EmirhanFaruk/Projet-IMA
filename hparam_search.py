"""
hparam_search.py — Systematic hyperparameter search with full visualisation.

Tests the following axes (each in isolation, one at a time):
  1. Optimizer         : adamw | adam | sgd | rmsprop
  2. Learning rate     : 1e-2 | 5e-3 | 1e-3 | 5e-4 | 1e-4
  3. Batch size        : 16 | 32 | 64
  4. Loss function     : bce_dice | focal_dice
  5. Preprocessing     : clahe | no_clahe | clahe_clip1 | clahe_clip4
  6. Augmentation level: none | light | medium | heavy
  7. Patch size        : 48 | 64 | 96

Each configuration trains for HPARAM_QUICK_EPOCHS epochs (default 25).
All metrics logged and saved. Full comparison plots generated per axis.

Usage:
    python hparam_search.py                      # run all axes
    python hparam_search.py --axis optimizer     # single axis
    python hparam_search.py --axis lr
    python hparam_search.py --axis batch_size
    python hparam_search.py --axis loss
    python hparam_search.py --axis preprocessing
    python hparam_search.py --axis augmentation
    python hparam_search.py --axis patch_size
    python hparam_search.py --plot_only          # just regenerate plots
"""

import sys
import json
import argparse
import copy
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import config as C
from src.utils import (set_seed, ensure_dirs, get_logger, MetricHistory, Timer)
from src.model import build_model
from src.losses import build_loss
from src.metrics import compute_metrics, MetricAccumulator
from src.optimizers import build_optimizer, build_scheduler
from src.infer import infer_full_image
from src.dataset import DRIVEFullImageDataset

logger = get_logger("hparam_search", C.LOG_DIR / "hparam_search.log")

# ─────────────────────────── Search grid definition ───────────────────────────

SEARCH_GRID = {
    "optimizer": [
        {"name": "AdamW",   "optimizer": "adamw",   "lr": 1e-3},
        {"name": "Adam",    "optimizer": "adam",    "lr": 1e-3},
        {"name": "SGD+Nesterov", "optimizer": "sgd", "lr": 1e-2},
        {"name": "RMSprop", "optimizer": "rmsprop", "lr": 5e-4},
    ],
    "lr": [
        {"name": "LR=1e-2",  "lr": 1e-2},
        {"name": "LR=5e-3",  "lr": 5e-3},
        {"name": "LR=1e-3",  "lr": 1e-3},
        {"name": "LR=5e-4",  "lr": 5e-4},
        {"name": "LR=1e-4",  "lr": 1e-4},
    ],
    "batch_size": [
        {"name": "Batch=16", "batch_size": 16},
        {"name": "Batch=32", "batch_size": 32},
        {"name": "Batch=64", "batch_size": 64},
    ],
    "loss": [
        {"name": "BCE+Dice",   "loss_fn": "bce_dice"},
        {"name": "Focal+Dice", "loss_fn": "focal_dice"},
    ],
    "preprocessing": [
        {"name": "CLAHE clip=2 (default)", "use_clahe": True,  "clahe_clip": 2.0},
        {"name": "No CLAHE",               "use_clahe": False, "clahe_clip": 2.0},
        {"name": "CLAHE clip=1 (mild)",    "use_clahe": True,  "clahe_clip": 1.0},
        {"name": "CLAHE clip=4 (strong)",  "use_clahe": True,  "clahe_clip": 4.0},
    ],
    "augmentation": [
        {"name": "No augmentation", "aug_level": "none"},
        {"name": "Light aug",       "aug_level": "light"},
        {"name": "Medium aug",      "aug_level": "medium"},
        {"name": "Heavy aug",       "aug_level": "heavy"},
    ],
    "patch_size": [
        {"name": "Patch=48", "patch_size": 48},
        {"name": "Patch=64", "patch_size": 64},
        {"name": "Patch=96", "patch_size": 96},
    ],
}

METRICS_TO_PLOT = [
    "val_dice", "val_iou", "val_sensitivity",
    "val_specificity", "val_auc_roc", "val_precision"
]

METRIC_LABELS = {
    "val_dice":        "Dice",
    "val_iou":         "IoU",
    "val_sensitivity": "Sensitivity",
    "val_specificity": "Specificity",
    "val_auc_roc":     "AUC-ROC",
    "val_precision":   "Precision",
}


# ─────────────────────────── Per-config training ──────────────────────────────

def _apply_cfg_to_module(cfg: dict):
    """Temporarily patch config module with values from a search config."""
    C.OPTIMIZER  = cfg.get("optimizer",   C.OPTIMIZER)
    C.LR         = cfg.get("lr",          C.LR)
    C.BATCH_SIZE = cfg.get("batch_size",  C.BATCH_SIZE)
    C.LOSS_FN    = cfg.get("loss_fn",     C.LOSS_FN)
    C.AUG_LEVEL  = cfg.get("aug_level",   C.AUG_LEVEL)
    C.PATCH_SIZE = cfg.get("patch_size",  C.PATCH_SIZE)
    if "use_clahe" in cfg:
        C.USE_CLAHE        = cfg["use_clahe"]
        C.CLAHE_CLIP_LIMIT = cfg.get("clahe_clip", C.CLAHE_CLIP_LIMIT)


def _reset_config():
    """Reset config to safe defaults after each run."""
    C.OPTIMIZER        = "adamw"
    C.LR               = 1e-3
    C.BATCH_SIZE       = 32
    C.LOSS_FN          = "bce_dice"
    C.AUG_LEVEL        = "medium"
    C.PATCH_SIZE       = 64
    C.USE_CLAHE        = True
    C.CLAHE_CLIP_LIMIT = 2.0


@torch.no_grad()
def _validate(model, device, threshold=0.5):
    """Full-image validation on val split. Returns dict of mean metrics."""
    ds  = DRIVEFullImageDataset("val")
    acc = MetricAccumulator()
    model.eval()
    for img_t, msk_t, fov_t, stem in ds:
        image = img_t.squeeze(0).numpy()
        gt    = msk_t.squeeze(0).numpy()
        fov   = fov_t.squeeze(0).numpy()
        prob  = infer_full_image(model, image, fov, device,
                                 patch_sz=C.PATCH_SIZE, stride=C.TEST_STRIDE)
        m = compute_metrics(prob, gt, fov, threshold)
        acc.update(**{k: v for k, v in m.items()
                      if k not in ("tp","fp","tn","fn")})
    return acc.mean()


def run_one_config(cfg: dict,
                   run_name: str,
                   device: torch.device,
                   experiment: str = "original") -> dict:
    """
    Train a model for HPARAM_QUICK_EPOCHS with the given config.
    Returns dict of per-epoch history + final val metrics.
    """
    set_seed(C.SEED)
    _apply_cfg_to_module(cfg)

    # ── Rebuild dataset with current config settings ─────────────────────
    # Import AFTER patching config so patch_size / aug_level take effect
    from src.dataset import DRIVEPatchDataset
    from src.dataset import get_train_transform

    # Patch augmentation transform into dataset module
    import src.dataset as ds_mod
    ds_mod._CURRENT_AUG_LEVEL = C.AUG_LEVEL

    train_ds = DRIVEPatchDataset("train")
    if len(train_ds) == 0:
        logger.error(f"[{run_name}] Empty training dataset — skipping")
        _reset_config()
        return {}

    train_ld = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                          num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY,
                          drop_last=True)

    # ── Model / optimizer / loss ──────────────────────────────────────────
    model    = build_model(experiment).to(device)
    crit     = build_loss(C.LOSS_FN)
    opt      = build_optimizer(model, name=C.OPTIMIZER, lr=C.LR)
    sched    = build_scheduler(opt)
    scaler   = GradScaler()

    history  = {m: [] for m in METRICS_TO_PLOT}
    history["train_loss"] = []
    history["val_loss"]   = []
    history["lr"]         = []

    best_dice   = 0.0
    no_improve  = 0
    epochs      = C.HPARAM_QUICK_EPOCHS
    patience    = C.HPARAM_PATIENCE

    logger.info(f"[{run_name}] Training {epochs} epochs …")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for imgs, masks, fovs in train_ld:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            with autocast():
                loss = crit(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt); scaler.update()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_ld)

        # ── Validate patch-level loss ──────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, fovs in train_ld:   # quick proxy: use 1 pass
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast():
                    val_loss += crit(model(imgs), masks).item()
                break   # just one batch for speed
        sched.step(val_loss)

        # ── Full-image validation (every 5 epochs for speed) ──────────────
        if epoch % 5 == 0 or epoch == epochs:
            vm = _validate(model, device)
            for k in METRICS_TO_PLOT:
                history[k].append(vm.get(k.replace("val_", ""), 0.0))
            cur_dice = vm.get("dice", 0.0)
            if cur_dice > best_dice:
                best_dice = cur_dice; no_improve = 0
            else:
                no_improve += 1
        else:
            for k in METRICS_TO_PLOT:
                history[k].append(history[k][-1] if history[k] else 0.0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        if no_improve >= patience:
            logger.info(f"  [{run_name}] Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0

    # Final metrics
    final = _validate(model, device)
    final["train_time_s"] = elapsed
    final["best_dice"]    = best_dice
    final["config"]       = cfg
    final["name"]         = run_name
    final["history"]      = history

    logger.info(f"  [{run_name}] Done — Dice={best_dice:.4f}  ({elapsed:.0f}s)")
    _reset_config()
    return final


# ─────────────────────────── Visualisation ────────────────────────────────────

COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E",
          "#639922", "#BA7517", "#7F77DD", "#888780"]


def _savefig(fig, path, dpi=150):
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_axis_curves(results: list[dict], axis_name: str, out_dir: Path):
    """Training curves for each config in an axis."""
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training curves — axis: {axis_name}", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    curve_keys = ["train_loss", "val_dice", "val_iou",
                  "val_auc_roc", "val_sensitivity", "lr"]
    curve_labels = ["Train Loss", "Val Dice", "Val IoU",
                    "Val AUC-ROC", "Val Sensitivity", "LR"]

    for ax, key, label in zip(axes, curve_keys, curve_labels):
        for i, r in enumerate(results):
            if not r or "history" not in r:
                continue
            h = r["history"]
            if key not in h or not h[key]:
                continue
            epochs = list(range(1, len(h[key]) + 1))
            ax.plot(epochs, h[key], label=r["name"],
                    color=COLORS[i % len(COLORS)], linewidth=1.8)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if key == "lr":
            ax.set_yscale("log")

    plt.tight_layout()
    _savefig(fig, out_dir / f"{axis_name}_training_curves.png")


def plot_axis_bar(results: list[dict], axis_name: str, out_dir: Path):
    """Final metric bar chart for all configs in an axis."""
    valid = [r for r in results if r and "dice" in r]
    if not valid:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Final metrics — axis: {axis_name}", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    metric_pairs = [
        ("dice",        "Dice"),
        ("iou",         "IoU"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("auc_roc",     "AUC-ROC"),
        ("precision",   "Precision"),
    ]

    names = [r["name"] for r in valid]
    x = np.arange(len(valid))

    for ax, (key, label) in zip(axes, metric_pairs):
        vals = [r.get(key, 0.0) for r in valid]
        bars = ax.bar(x, vals, color=COLORS[:len(valid)],
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11)
        lo = max(0, min(vals) - 0.03)
        hi = min(1.0, max(vals) + 0.03)
        ax.set_ylim(lo, hi)
        ax.grid(True, axis="y", alpha=0.3)

        # Highlight best
        best_idx = int(np.argmax(vals))
        axes[list(axes).index(ax)].get_children()[best_idx].set_edgecolor("gold")
        axes[list(axes).index(ax)].get_children()[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    _savefig(fig, out_dir / f"{axis_name}_metrics_bar.png")


def plot_radar(results: list[dict], axis_name: str, out_dir: Path):
    """Radar chart comparing all configs across 6 metrics."""
    valid = [r for r in results if r and "dice" in r]
    if len(valid) < 2:
        return

    metrics = ["dice", "iou", "sensitivity", "specificity", "auc_roc", "precision"]
    labels  = ["Dice", "IoU", "Sens", "Spec", "AUC-ROC", "Prec"]
    n_m     = len(metrics)
    angles  = [n / n_m * 2 * np.pi for n in range(n_m)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.7, 1.0)

    for i, r in enumerate(valid):
        vals  = [r.get(m, 0.0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=COLORS[i % len(COLORS)],
                label=r["name"])
        ax.fill(angles, vals, alpha=0.08, color=COLORS[i % len(COLORS)])

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title(f"Radar — axis: {axis_name}", fontsize=13, pad=20)
    plt.tight_layout()
    _savefig(fig, out_dir / f"{axis_name}_radar.png")


def plot_summary_heatmap(all_results: dict, out_dir: Path):
    """Heatmap of best Dice per axis / config."""
    rows, cols, vals = [], [], []
    for axis, results in all_results.items():
        for r in results:
            if r and "dice" in r:
                rows.append(axis)
                cols.append(r["name"])
                vals.append(r.get("dice", 0.0))

    if not rows:
        return

    axes_list = list(dict.fromkeys(rows))
    all_names = list(dict.fromkeys(cols))
    matrix    = np.full((len(axes_list), len(all_names)), np.nan)
    for ax, nm, v in zip(rows, cols, vals):
        i = axes_list.index(ax)
        j = all_names.index(nm)
        matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(max(12, len(all_names)), len(axes_list) + 1))
    import seaborn as sns
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlGn",
                xticklabels=all_names, yticklabels=axes_list,
                ax=ax, linewidths=0.3, cbar_kws={"label": "Dice"})
    ax.set_title("Dice scores across all hyperparameter axes", fontsize=13)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    _savefig(fig, out_dir / "hparam_summary_heatmap.png")
    logger.info("Summary heatmap saved.")


# ─────────────────────────── Main driver ──────────────────────────────────────

def run_axis(axis_name: str,
             device: torch.device,
             out_dir: Path,
             experiment: str = "original") -> list[dict]:
    """Run all configs for one axis and generate plots."""
    configs = SEARCH_GRID[axis_name]
    results = []

    logger.info(f"\n{'='*60}")
    logger.info(f"AXIS: {axis_name}  ({len(configs)} configs)")
    logger.info(f"{'='*60}")

    for cfg in configs:
        run_name = cfg["name"]
        result   = run_one_config(cfg, run_name, device, experiment)
        results.append(result)

        # Save intermediate JSON
        ensure_dirs(out_dir)
        with open(out_dir / f"{axis_name}_results.json", "w") as f:
            safe = [{k: v for k, v in r.items() if k != "history"}
                    for r in results if r]
            json.dump(safe, f, indent=2)

    # Generate plots
    plot_axis_curves(results, axis_name, out_dir)
    plot_axis_bar(results, axis_name, out_dir)
    plot_radar(results, axis_name, out_dir)
    logger.info(f"Plots saved for axis: {axis_name}")
    return results


def run_all(experiment: str = "original"):
    set_seed(C.SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = C.HPARAM_DIR
    ensure_dirs(out_dir, C.LOG_DIR)

    logger.info(f"Device: {device}")
    logger.info(f"Quick epochs per config: {C.HPARAM_QUICK_EPOCHS}")

    all_results = {}
    for axis_name in SEARCH_GRID:
        results = run_axis(axis_name, device, out_dir, experiment)
        all_results[axis_name] = results

    # Summary heatmap across all axes
    plot_summary_heatmap(all_results, out_dir)

    # Save full summary
    summary = {}
    for axis, results in all_results.items():
        valid = [r for r in results if r and "dice" in r]
        if valid:
            best = max(valid, key=lambda r: r.get("dice", 0))
            summary[axis] = {
                "best_config": best["name"],
                "best_dice":   best.get("dice", 0),
                "all": [{
                    "name":        r["name"],
                    "dice":        r.get("dice", 0),
                    "sensitivity": r.get("sensitivity", 0),
                    "auc_roc":     r.get("auc_roc", 0),
                } for r in valid]
            }

    with open(out_dir / "hparam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== HYPERPARAMETER SEARCH COMPLETE ===")
    logger.info(f"Results saved to: {out_dir}")
    for axis, s in summary.items():
        logger.info(f"  {axis:20s}: best={s['best_config']}  Dice={s['best_dice']:.4f}")

    return summary


# ─────────────────────────── Entry point ──────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axis", default="all",
                   choices=["all"] + list(SEARCH_GRID.keys()))
    p.add_argument("--experiment", default="original",
                   choices=["original", "msca", "deva"])
    p.add_argument("--plot_only", action="store_true",
                   help="Reload saved results and regenerate plots only")
    args = p.parse_args()

    if args.plot_only:
        # Reload all saved JSONs and replot
        out_dir = C.HPARAM_DIR
        all_results = {}
        for axis_name in SEARCH_GRID:
            path = out_dir / f"{axis_name}_results.json"
            if path.exists():
                with open(path) as f:
                    results = json.load(f)
                all_results[axis_name] = results
                plot_axis_bar(results, axis_name, out_dir)
                plot_radar(results, axis_name, out_dir)
        plot_summary_heatmap(all_results, out_dir)
        print("Plots regenerated.")

    elif args.axis == "all":
        run_all(args.experiment)

    else:
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out_dir = C.HPARAM_DIR
        ensure_dirs(out_dir, C.LOG_DIR)
        set_seed(C.SEED)
        run_axis(args.axis, device, out_dir, args.experiment)
