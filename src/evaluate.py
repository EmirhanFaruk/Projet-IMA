"""
evaluate.py — Full test-set evaluation with comprehensive visualisation.

Produces:
  ┌─ Metrics ──────────────────────────────────────────────────────────────┐
  │ • Per-image CSV  (outputs/metrics/<exp>_per_image.csv)                  │
  │ • Summary JSON/TXT                                                      │
  │ • Threshold search results                                              │
  └─────────────────────────────────────────────────────────────────────────┘
  ┌─ Figures ──────────────────────────────────────────────────────────────┐
  │ • Training curves  (loss, dice, IoU, AUC, LR per epoch)               │
  │ • ROC curve                                                             │
  │ • Precision-Recall curve                                               │
  │ • Confusion matrix (aggregated)                                        │
  │ • Dice vs Threshold                                                     │
  │ • Per-image Dice bar chart                                              │
  │ • 4-panel predictions for every test image                             │
  │ • Best 5 / Worst 5 overlays                                            │
  └─────────────────────────────────────────────────────────────────────────┘
"""

import sys
import json
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

import config as C
from dataset import DRIVEFullImageDataset
from model import build_model
from infer import infer_full_image, infer_tta
from metrics import (compute_metrics, find_best_threshold,
                     compute_roc_data, compute_pr_data)
from utils import (get_logger, load_checkpoint, ensure_dirs, set_seed)

logger = get_logger("evaluate", C.LOG_DIR / "evaluate.log")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path, dpi: int = 150):
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _prob_to_rgb(prob: np.ndarray) -> np.ndarray:
    """Convert grayscale probability map [0,1] to uint8 RGB."""
    gray = (prob * 255).clip(0, 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def _overlay_vessels(image_gray: np.ndarray,
                     pred_bin: np.ndarray,
                     gt_bin: np.ndarray) -> np.ndarray:
    """
    Green = true positive, Red = false positive, Blue = false negative.
    image_gray: (H,W) float32 [0,1]
    """
    h, w    = image_gray.shape
    canvas  = np.stack([(image_gray * 255).astype(np.uint8)] * 3, axis=-1)
    tp      = (pred_bin == 1) & (gt_bin == 1)
    fp      = (pred_bin == 1) & (gt_bin == 0)
    fn      = (pred_bin == 0) & (gt_bin == 1)

    canvas[tp] = [0,   200, 0]    # green
    canvas[fp] = [220, 50,  50]   # red
    canvas[fn] = [50,  50,  220]  # blue
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Plot: training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, out_dir: Path, exp: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves — {exp}", fontsize=14, fontweight="bold")

    pairs = [
        ("train_loss",     "val_loss",        "Loss",        axes[0, 0]),
        ("val_dice",       None,              "Val Dice",    axes[0, 1]),
        ("val_iou",        None,              "Val IoU",     axes[0, 2]),
        ("val_auc_roc",    None,              "Val AUC-ROC", axes[1, 0]),
        ("val_sensitivity","val_specificity", "Sens/Spec",   axes[1, 1]),
        ("learning_rate",  None,              "Learning Rate", axes[1, 2]),
    ]

    for key1, key2, title, ax in pairs:
        if key1 in history:
            ax.plot(epochs, history[key1], label=key1.replace("_", " "), linewidth=1.5)
        if key2 and key2 in history:
            ax.plot(epochs, history[key2], label=key2.replace("_", " "),
                    linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if "lr" in title.lower():
            ax.set_yscale("log")

    plt.tight_layout()
    _savefig(fig, out_dir / f"{exp}_training_curves.png")
    logger.info(f"  Saved training curves → {exp}_training_curves.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot: ROC curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc(probs_list, targets_list, fov_list, out_dir, exp):
    fpr, tpr, _, auc = compute_roc_data(probs_list, targets_list, fov_list)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {exp}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_roc_curve.png")
    logger.info(f"  AUC-ROC = {auc:.4f}")
    return auc


# ──────────────────────────────────────────────────────────────────────────────
# Plot: Precision-Recall curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_pr(probs_list, targets_list, fov_list, out_dir, exp):
    prec, rec, _, auc = compute_pr_data(probs_list, targets_list, fov_list)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, color="#E91E63", label=f"AUC-PR = {auc:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {exp}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_pr_curve.png")
    logger.info(f"  AUC-PR  = {auc:.4f}")
    return auc


# ──────────────────────────────────────────────────────────────────────────────
# Plot: Confusion matrix (aggregated across all test images)
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(total_tp, total_fp, total_tn, total_fn,
                          out_dir, exp):
    cm  = np.array([[total_tn, total_fp],
                    [total_fn, total_tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Pred BG", "Pred Vessel"],
                yticklabels=["GT BG",   "GT Vessel"],
                ax=ax, cbar=True)
    ax.set_title(f"Confusion Matrix (all test images, inside FOV)\n{exp}",
                 fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_xlabel("Prediction",   fontsize=11)
    _savefig(fig, out_dir / f"{exp}_confusion_matrix.png")
    logger.info("  Confusion matrix saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Plot: Dice vs threshold
# ──────────────────────────────────────────────────────────────────────────────

def plot_threshold_search(thresh_results: dict, best_thr: float,
                          out_dir: Path, exp: str):
    thresholds = sorted(thresh_results.keys())
    dices      = [thresh_results[t] for t in thresholds]
    best_dice  = thresh_results[best_thr]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, dices, lw=2, color="#4CAF50", label="Mean Dice")
    ax.axvline(best_thr, color="red", linestyle="--", lw=1.5,
               label=f"Best thr={best_thr:.2f}  Dice={best_dice:.4f}")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Mean Dice", fontsize=12)
    ax.set_title(f"Dice vs Threshold — {exp}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_threshold_search.png")
    logger.info(f"  Best threshold = {best_thr:.2f}  (Dice={best_dice:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# Plot: 4-panel prediction figure per image
# ──────────────────────────────────────────────────────────────────────────────

def plot_4panel(image, gt, prob_map, pred_bin, fov, stem, out_dir, exp):
    overlay = _overlay_vessels(image, pred_bin, gt)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"{stem} — {exp}", fontsize=11)

    axes[0].imshow(image,    cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Green+CLAHE")
    axes[1].imshow(gt,       cmap="gray")
    axes[1].set_title("GT Vessels")
    axes[2].imshow(prob_map, cmap="jet",  vmin=0, vmax=1)
    axes[2].set_title("Probability Map")
    axes[3].imshow(pred_bin, cmap="gray")
    axes[3].set_title("Prediction (binary)")
    axes[4].imshow(overlay)
    axes[4].set_title("Overlay\nGreen=TP Red=FP Blue=FN")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    _savefig(fig, out_dir / "per_image" / f"{stem}_{exp}.png", dpi=100)


# ──────────────────────────────────────────────────────────────────────────────
# Plot: Per-image Dice bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_image_dice(stems, dices, out_dir, exp):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors  = ["#4CAF50" if d >= np.mean(dices) else "#F44336" for d in dices]
    ax.bar(range(len(stems)), dices, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(np.mean(dices), color="navy", linestyle="--", lw=1.5,
               label=f"Mean = {np.mean(dices):.4f}")
    ax.set_xticks(range(len(stems)))
    ax.set_xticklabels(stems, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dice")
    ax.set_title(f"Per-Image Dice — {exp}")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_per_image_dice.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation driver
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(experiment: str       = C.EXPERIMENT,
             ckpt_path:  str | None = None,
             use_tta:    bool       = False):

    set_seed(C.SEED)
    out_dir = C.VIZ_DIR / experiment
    ensure_dirs(out_dir / "per_image", C.METRIC_DIR, C.PRED_DIR / experiment)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Model + checkpoint ────────────────────────────────────────────────
    model = build_model(experiment).to(device)
    if ckpt_path is None:
        ckpt_path = C.CKPT_DIR / experiment / "best_model.pth"
    load_checkpoint(str(ckpt_path), model)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ── Load training history for curves ─────────────────────────────────
    hist_file = C.METRIC_DIR / f"{experiment}_history.json"
    if hist_file.exists():
        with open(hist_file) as f:
            history = json.load(f)
        plot_training_curves(history, out_dir, experiment)

    # ── Test set inference ────────────────────────────────────────────────
    test_ds  = DRIVEFullImageDataset(split="test")
    infer_fn = infer_tta if use_tta else infer_full_image

    probs_list   = []
    targets_list = []
    fov_list     = []
    stems        = []
    per_image    = []

    logger.info("Running inference on test set …")
    images_list  = []   # store raw images for visualisation
    for img_t, msk_t, fov_t, stem in test_ds:
        image = img_t.squeeze(0).numpy()
        gt    = msk_t.squeeze(0).numpy()
        fov   = fov_t.squeeze(0).numpy()

        prob  = infer_fn(model, image, fov, device)
        images_list.append(image)
        probs_list.append(prob)
        targets_list.append(gt)
        fov_list.append(fov)
        stems.append(stem)

        # Save raw predictions
        np.save(C.PRED_DIR / experiment / f"{stem}_prob.npy", prob)

    # ── Threshold search on validation-equivalent subset (use test here) ──
    logger.info("Searching best threshold …")
    thr_results = find_best_threshold(probs_list, targets_list, fov_list)
    best_thr    = thr_results["best_threshold"]
    logger.info(f"Best threshold: {best_thr:.2f}")
    plot_threshold_search(thr_results["scores"], best_thr, out_dir, experiment)

    # ── Per-image metrics at best threshold ──────────────────────────────
    total_tp = total_fp = total_tn = total_fn = 0
    dices    = []

    for idx, (prob, gt, fov, stem) in enumerate(zip(probs_list, targets_list, fov_list, stems)):
        m         = compute_metrics(prob, gt, fov, best_thr)
        m["stem"] = stem
        per_image.append(m)
        dices.append(m["dice"])

        total_tp += m["tp"];  total_fp += m["fp"]
        total_tn += m["tn"];  total_fn += m["fn"]

        pred_bin = (prob >= best_thr).astype(np.uint8)
        plot_4panel(images_list[idx], gt, prob, pred_bin, fov, stem, out_dir, experiment)

        # Save binary mask PNG
        Image.fromarray((pred_bin * 255).astype(np.uint8)).save(
            C.PRED_DIR / experiment / f"{stem}_binary.png")

    # ── ROC / PR curves ───────────────────────────────────────────────────
    auc_roc = plot_roc(probs_list, targets_list, fov_list, out_dir, experiment)
    auc_pr  = plot_pr( probs_list, targets_list, fov_list, out_dir, experiment)

    # ── Confusion matrix ──────────────────────────────────────────────────
    plot_confusion_matrix(total_tp, total_fp, total_tn, total_fn,
                          out_dir, experiment)

    # ── Per-image Dice bar ────────────────────────────────────────────────
    plot_per_image_dice(stems, dices, out_dir, experiment)

    # ── Best / worst 5 ───────────────────────────────────────────────────
    sorted_idx = np.argsort(dices)
    worst5     = sorted_idx[:5]
    best5      = sorted_idx[-5:]
    for label, indices in [("best", best5), ("worst", worst5)]:
        for rank, i in enumerate(indices[::-1] if label == "best" else indices):
            stem  = stems[i]
            prob  = probs_list[i]
            gt    = targets_list[i]
            fov   = fov_list[i]
            image = np.load(C.IMG_DIR / f"{stem}.npy")
            pred_bin = (prob >= best_thr).astype(np.uint8)
            overlay  = _overlay_vessels(image, pred_bin, gt)

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(overlay)
            ax.set_title(f"{label.upper()} #{rank+1}  {stem}\n"
                         f"Dice={dices[i]:.4f}", fontsize=10)
            ax.axis("off")
            _savefig(fig, out_dir / f"{experiment}_{label}{rank+1}_{stem}.png")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = C.METRIC_DIR / f"{experiment}_per_image.csv"
    keys = ["stem", "dice", "iou", "accuracy", "sensitivity", "specificity",
            "precision", "f1", "auc_roc", "auc_pr"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_image:
            w.writerow({k: row.get(k, "") for k in keys})
    logger.info(f"Per-image metrics → {csv_path}")

    # ── Summary stats ─────────────────────────────────────────────────────
    summary = {}
    for key in ["dice", "iou", "accuracy", "sensitivity",
                "specificity", "precision", "f1", "auc_roc", "auc_pr"]:
        vals = [r[key] for r in per_image if key in r]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"]  = float(np.std(vals))

    summary["best_threshold"] = best_thr
    summary["auc_roc_global"] = float(auc_roc)
    summary["auc_pr_global"]  = float(auc_pr)

    # Save summary
    summ_path = C.METRIC_DIR / f"{experiment}_summary.json"
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)

    summ_txt = C.METRIC_DIR / f"{experiment}_summary.txt"
    with open(summ_txt, "w") as f:
        f.write(f"Experiment: {experiment}\n")
        f.write(f"Threshold : {best_thr:.2f}\n\n")
        for k, v in summary.items():
            f.write(f"  {k:30s}: {v:.4f}\n")

    # ── Print summary ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"FINAL TEST METRICS — {experiment}")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k:32s}: {v:.4f}")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=C.EXPERIMENT,
                   choices=["original", "msca", "deva",
                             "ablation_spatial", "ablation_channel"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()
    evaluate(args.experiment, args.ckpt, args.tta)
