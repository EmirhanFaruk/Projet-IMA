"""
evaluate.py — Full test-set evaluation with comprehensive visualisation.

Produces (under outputs/visualizations/<exp>/fold<F>_seed<S>[_<tag>]/):
  - Training curves (loss, dice, IoU, AUC, sens/spec, LR per epoch)
  - ROC + Precision-Recall curves
  - Confusion matrix (aggregated)
  - Dice vs Threshold plot
  - Per-image Dice bar chart
  - 5-panel prediction figure for every test image
  - Best 5 / Worst 5 overlays

And under outputs/metrics/<exp>/fold<F>_seed<S>[_<tag>]/:
  - per_image.csv
  - summary.json
  - summary.txt
"""

import sys
import json
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path, dpi: int = 150):
    ensure_dirs(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _overlay_vessels(image_gray, pred_bin, gt_bin):
    h, w   = image_gray.shape
    canvas = np.stack([(image_gray * 255).astype(np.uint8)] * 3, axis=-1)
    tp = (pred_bin == 1) & (gt_bin == 1)
    fp = (pred_bin == 1) & (gt_bin == 0)
    fn = (pred_bin == 0) & (gt_bin == 1)
    canvas[tp] = [0,   200, 0]
    canvas[fp] = [220, 50,  50]
    canvas[fn] = [50,  100, 220]
    return canvas


# ─────────────────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, out_dir, exp):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves — {exp}", fontsize=14, fontweight="bold")

    pairs = [
        ("train_loss",     "val_loss",        "Loss",          axes[0, 0]),
        ("val_dice",       None,              "Val Dice",      axes[0, 1]),
        ("val_iou",        None,              "Val IoU",       axes[0, 2]),
        ("val_auc_roc",    None,              "Val AUC-ROC",   axes[1, 0]),
        ("val_sensitivity","val_specificity", "Sens/Spec",     axes[1, 1]),
        ("learning_rate",  None,              "Learning Rate", axes[1, 2]),
    ]
    for k1, k2, title, ax in pairs:
        if k1 in history:
            ax.plot(epochs, history[k1], label=k1.replace("_", " "), lw=1.5)
        if k2 and k2 in history:
            ax.plot(epochs, history[k2], label=k2.replace("_", " "),
                    linestyle="--", lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if "lr" in title.lower():
            ax.set_yscale("log")
    plt.tight_layout()
    _savefig(fig, out_dir / f"{exp}_training_curves.png")


def plot_roc(probs_list, targets_list, fov_list, out_dir, exp):
    fpr, tpr, _, auc = compute_roc_data(probs_list, targets_list, fov_list)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"AUC-ROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {exp}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_roc_curve.png")
    return auc


def plot_pr(probs_list, targets_list, fov_list, out_dir, exp):
    prec, rec, _, auc = compute_pr_data(probs_list, targets_list, fov_list)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, color="#E91E63", label=f"AUC-PR = {auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {exp}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_pr_curve.png")
    return auc


def plot_confusion_matrix(tp, fp, tn, fn, out_dir, exp):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Pred BG", "Pred Vessel"],
                yticklabels=["GT BG",   "GT Vessel"],
                ax=ax, cbar=True)
    ax.set_title(f"Confusion Matrix (FOV pixels) — {exp}")
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Prediction")
    _savefig(fig, out_dir / f"{exp}_confusion_matrix.png")


def plot_threshold_search(scores, best_thr, out_dir, exp):
    thresholds = sorted(scores.keys())
    dices      = [scores[t] for t in thresholds]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, dices, lw=2, color="#4CAF50")
    ax.axvline(best_thr, color="red", ls="--", lw=1.5,
               label=f"Best τ={best_thr:.2f}  Dice={scores[best_thr]:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean Dice")
    ax.set_title(f"Dice vs Threshold — {exp}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_threshold_search.png")


def plot_5panel(image, gt, prob, pred_bin, fov, stem, out_dir, exp):
    overlay = _overlay_vessels(image, pred_bin, gt)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"{stem} — {exp}")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Green+CLAHE")
    axes[1].imshow(gt,    cmap="gray");                 axes[1].set_title("GT")
    axes[2].imshow(prob,  cmap="jet", vmin=0, vmax=1);  axes[2].set_title("Probability")
    axes[3].imshow(pred_bin, cmap="gray");              axes[3].set_title("Prediction")
    axes[4].imshow(overlay);                            axes[4].set_title("Overlay")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    _savefig(fig, out_dir / "per_image" / f"{stem}_{exp}.png", dpi=100)


def plot_per_image_dice(stems, dices, out_dir, exp):
    fig, ax = plt.subplots(figsize=(14, 5))
    mean_d = float(np.mean(dices))
    colors = ["#4CAF50" if d >= mean_d else "#F44336" for d in dices]
    ax.bar(range(len(stems)), dices, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(mean_d, color="navy", ls="--", lw=1.5,
               label=f"Mean = {mean_d:.4f}")
    ax.set_xticks(range(len(stems)))
    ax.set_xticklabels(stems, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dice")
    ax.set_title(f"Per-Image Dice — {exp}")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, out_dir / f"{exp}_per_image_dice.png")


# ─────────────────────────────────────────────────────────────────────────
# Main evaluation driver
# ─────────────────────────────────────────────────────────────────────────

def evaluate(experiment=None, tag=None, ckpt_path=None, use_tta=False):
    if experiment is None:
        experiment = C.EXPERIMENT

    set_seed(C.SEED)

    # Resolve fold/seed-namespaced directories
    sub        = f"fold{C.FOLD}_seed{C.SEED}" + (f"_{tag}" if tag else "")
    viz_dir    = C.VIZ_DIR    / experiment / sub
    metric_dir = C.METRIC_DIR / experiment / sub
    pred_dir   = C.PRED_DIR   / experiment / sub
    ckpt_dir   = C.CKPT_DIR   / experiment / sub
    ensure_dirs(viz_dir / "per_image", metric_dir, pred_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Resolve checkpoint
    if ckpt_path is None:
        ckpt_path = ckpt_dir / "best_model.pth"
    if not Path(ckpt_path).exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return None

    model = build_model(experiment).to(device)
    load_checkpoint(str(ckpt_path), model)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Plot training curves
    history_path = metric_dir / f"{experiment}_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, viz_dir, experiment)

    # Test-set inference
    test_ds  = DRIVEFullImageDataset(split="test")
    infer_fn = infer_tta if use_tta else infer_full_image

    images_list, probs_list, targets_list, fov_list, stems = [], [], [], [], []
    logger.info("Running inference on test set…")
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
        np.save(pred_dir / f"{stem}_prob.npy", prob)

    # Threshold search
    logger.info("Searching best threshold…")
    thr_results = find_best_threshold(probs_list, targets_list, fov_list)
    best_thr    = thr_results["best_threshold"]
    plot_threshold_search(thr_results["scores"], best_thr, viz_dir, experiment)

    # Per-image metrics at best threshold
    total_tp = total_fp = total_tn = total_fn = 0
    dices, per_image = [], []

    for idx, (prob, gt, fov, stem) in enumerate(
            zip(probs_list, targets_list, fov_list, stems)):
        m         = compute_metrics(prob, gt, fov, best_thr)
        m["stem"] = stem
        per_image.append(m)
        dices.append(m["dice"])

        total_tp += m["tp"]; total_fp += m["fp"]
        total_tn += m["tn"]; total_fn += m["fn"]

        pred_bin = (prob >= best_thr).astype(np.uint8)
        plot_5panel(images_list[idx], gt, prob, pred_bin, fov,
                    stem, viz_dir, experiment)
        Image.fromarray((pred_bin * 255).astype(np.uint8)).save(
            pred_dir / f"{stem}_binary.png")

    # ROC / PR curves
    auc_roc = plot_roc(probs_list, targets_list, fov_list, viz_dir, experiment)
    auc_pr  = plot_pr( probs_list, targets_list, fov_list, viz_dir, experiment)

    # Confusion matrix
    plot_confusion_matrix(total_tp, total_fp, total_tn, total_fn,
                          viz_dir, experiment)

    # Per-image Dice
    plot_per_image_dice(stems, dices, viz_dir, experiment)

    # Best/worst 5 overlays
    sorted_idx = np.argsort(dices)
    worst5 = sorted_idx[:min(5, len(sorted_idx))]
    best5  = sorted_idx[-min(5, len(sorted_idx)):]
    for label, indices in [("best", best5), ("worst", worst5)]:
        ranked = list(indices[::-1] if label == "best" else indices)
        for rank, i in enumerate(ranked):
            stem  = stems[i]
            prob  = probs_list[i]
            gt    = targets_list[i]
            image = np.load(C.IMG_DIR / f"{stem}.npy")
            pred_bin = (prob >= best_thr).astype(np.uint8)
            overlay  = _overlay_vessels(image, pred_bin, gt)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(overlay)
            ax.set_title(f"{label.upper()} #{rank+1}  {stem}\n"
                         f"Dice={dices[i]:.4f}", fontsize=10)
            ax.axis("off")
            _savefig(fig, viz_dir / f"{experiment}_{label}{rank+1}_{stem}.png")

    # CSV
    csv_path = metric_dir / "per_image.csv"
    keys = ["stem", "dice", "iou", "accuracy", "sensitivity", "specificity",
            "precision", "f1", "auc_roc", "auc_pr"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_image:
            w.writerow({k: row.get(k, "") for k in keys})

    # Summary
    summary = {"experiment": experiment, "fold": C.FOLD, "seed": C.SEED,
               "tag": tag, "use_tta": use_tta}
    for key in ["dice", "iou", "accuracy", "sensitivity", "specificity",
                "precision", "f1", "auc_roc", "auc_pr"]:
        vals = [r[key] for r in per_image if key in r]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"]  = float(np.std(vals))
    summary["best_threshold"] = float(best_thr)
    summary["auc_roc_global"] = float(auc_roc)
    summary["auc_pr_global"]  = float(auc_pr)

    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(metric_dir / "summary.txt", "w") as f:
        f.write(f"Experiment: {experiment}\n")
        f.write(f"Fold      : {C.FOLD}\n")
        f.write(f"Seed      : {C.SEED}\n")
        f.write(f"Tag       : {tag}\n")
        f.write(f"TTA       : {use_tta}\n")
        f.write(f"Threshold : {best_thr:.2f}\n\n")
        for k, v in summary.items():
            if isinstance(v, float):
                f.write(f"  {k:30s}: {v:.4f}\n")
            else:
                f.write(f"  {k:30s}: {v}\n")

    logger.info("=" * 60)
    logger.info(f"FINAL TEST METRICS — {experiment} (fold={C.FOLD}, seed={C.SEED})")
    logger.info("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k:32s}: {v:.4f}")
    return summary


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=C.EXPERIMENT,
                   choices=["original", "msca", "deva",
                             "ablation_spatial", "ablation_channel"])
    p.add_argument("--fold",   type=int, default=None)
    p.add_argument("--seed",   type=int, default=None)
    p.add_argument("--tag",    default=None)
    p.add_argument("--tta",    action="store_true")
    p.add_argument("--ckpt",   default=None)
    args = p.parse_args()

    if args.fold is not None: C.FOLD = args.fold
    if args.seed is not None: C.SEED = args.seed

    evaluate(experiment=args.experiment, tag=args.tag,
             ckpt_path=args.ckpt, use_tta=args.tta)
