"""
analyze_test_predictions.py — Self-consistency analysis of test-set predictions.

This script does NOT compute Dice, IoU, AUC, etc. — those require ground truth,
which the DRIVE test set does not have. Instead it produces *plausibility* and
*self-consistency* signals that sanity-checks the trained models on unseen data.

What it computes (per image, per model):
  • Vessel pixel %   inside the FOV
  • Probability map  mean, std, entropy
  • Connectivity:    number of connected components, largest component
                     fraction, skeleton length, branching points
  • Plausibility flag (1 if vessel % is inside the training-distribution
                       range 8.7 %–16.8 %, 0 otherwise)

Across-model signals:
  • Pairwise Dice / IoU between binary predictions (predictions agree or not)
  • Per-pixel probability std across all selected runs (uncertainty map)
  • Per-pixel disagreement count (binary, how many models predict vessel)

What this script writes (under <input_root>/_analysis/):
  • per_image_stats.csv               — long-format table, one row per
                                         (model, image)
  • per_image_stats.json              — same data, JSON
  • inter_model_agreement.csv         — pairwise Dice/IoU matrix
  • inter_model_agreement_heatmap.png
  • plausibility_summary.png          — vessel-% bar chart with the training
                                         distribution band shown
  • probability_stats.png             — mean/std/entropy comparison plot
  • per_image_uncertainty/<image>.png — pixel-wise uncertainty heatmap
  • per_image_disagreement/<image>.png — where models disagree

Usage (after running predict_test.py):
    python analyze_test_predictions.py \\
        --input outputs/predictions/test/

    # Or pick a specific subset of models:
    python analyze_test_predictions.py \\
        --input outputs/predictions/test/ \\
        --models original,msca,deva
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

import config as C
from utils import get_logger, ensure_dirs

logger = get_logger("analyze_test", C.LOG_DIR / "analyze_test.log")

# DRIVE training-set vessel-fraction range (inside FOV)
DRIVE_VESSEL_PCT_MIN  = 8.7
DRIVE_VESSEL_PCT_MAX  = 16.8
DRIVE_VESSEL_PCT_MEAN = 12.5


# ─────────────────────────────────────────────────────────────────────────
# Discovery: walk an outputs/predictions/test/ tree
# ─────────────────────────────────────────────────────────────────────────

def discover_predictions(root: Path):
    """Return {model_label: {stem: prob_path}}.

    A `model_label` is "<experiment>/<run_subdir>" (e.g.
    "original/fold2_seed42" or "msca/ensemble_fall_sall"), so multiple
    runs of the same architecture are kept distinct.
    """
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    layout = {}
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            label = f"{exp_dir.name}/{run_dir.name}"
            probs = {p.stem.replace("_prob", ""): p
                     for p in run_dir.glob("*_prob.npy")}
            if probs:
                layout[label] = probs
    return layout


def derive_fov_from_prob(prob: np.ndarray) -> np.ndarray:
    """Predictions outside FOV are written as exactly 0 by predict_test.
    The FOV is therefore wherever any > 0 value is *plausible*. This is
    a robust signal in practice."""
    return (prob > 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────
# Per-image, per-model statistics
# ─────────────────────────────────────────────────────────────────────────

def compute_image_stats(prob: np.ndarray, fov: np.ndarray, threshold: float):
    """Returns a dict of self-consistency stats for one (model, image)."""
    binary  = ((prob >= threshold) & (fov == 1)).astype(np.uint8)
    inside  = fov.astype(bool)

    vessel_pct = 100.0 * binary[inside].mean() if inside.any() else 0.0
    plausible  = bool(DRIVE_VESSEL_PCT_MIN <= vessel_pct <= DRIVE_VESSEL_PCT_MAX)

    p_in = prob[inside]
    if p_in.size > 0:
        p_clip = np.clip(p_in, 1e-7, 1 - 1e-7)
        entropy = float(-(p_clip * np.log(p_clip)
                          + (1 - p_clip) * np.log(1 - p_clip)).mean())
    else:
        entropy = 0.0

    # Connectivity on the binary mask
    labels, n_components = ndi.label(binary, structure=np.ones((3, 3),
                                                                dtype=int))
    if n_components > 0:
        comp_sizes = np.bincount(labels.ravel())[1:]
        largest_pct = 100.0 * comp_sizes.max() / max(1, comp_sizes.sum())
    else:
        largest_pct = 0.0

    # Skeleton-based length and branch points
    if binary.sum() > 0:
        skel = skeletonize(binary > 0).astype(np.uint8)
        skeleton_len = int(skel.sum())
        # Branch points: skeleton pixels with > 2 skeleton neighbours
        kernel = np.ones((3, 3), dtype=np.uint8); kernel[1, 1] = 0
        nbr = ndi.convolve(skel, kernel, mode="constant", cval=0)
        branch_points = int(((skel == 1) & (nbr >= 3)).sum())
    else:
        skeleton_len  = 0
        branch_points = 0

    return {
        "vessel_pct_fov":            float(vessel_pct),
        "plausibility_flag":         int(plausible),
        "prob_mean_in_fov":          float(p_in.mean()) if p_in.size else 0.0,
        "prob_std_in_fov":           float(p_in.std())  if p_in.size else 0.0,
        "prob_entropy_in_fov":       float(entropy),
        "n_connected_components":    int(n_components),
        "largest_component_pct":     float(largest_pct),
        "skeleton_length":           int(skeleton_len),
        "branch_points":             int(branch_points),
    }


# ─────────────────────────────────────────────────────────────────────────
# Inter-model agreement
# ─────────────────────────────────────────────────────────────────────────

def pairwise_dice_iou(b1: np.ndarray, b2: np.ndarray):
    inter = int(((b1 == 1) & (b2 == 1)).sum())
    s1, s2 = int(b1.sum()), int(b2.sum())
    union = s1 + s2 - inter
    dice  = (2 * inter / (s1 + s2)) if (s1 + s2) else 1.0
    iou   = (inter / union)         if union     else 1.0
    return dice, iou


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def plot_plausibility(per_image_rows, out_path: Path):
    """Bar chart of vessel % per (model, image) with the DRIVE training band shown."""
    by_model = defaultdict(list)
    for r in per_image_rows:
        by_model[r["model"]].append((r["image"], r["vessel_pct_fov"]))

    n_models = len(by_model)
    if n_models == 0:
        return
    fig, axes = plt.subplots(n_models, 1, figsize=(12, max(3, 2.2 * n_models)),
                              sharex=True, squeeze=False)
    palette = sns.color_palette("Set2", n_colors=n_models)

    for ax, (model, pairs), col in zip(axes[:, 0], by_model.items(), palette):
        pairs.sort()
        names = [p[0] for p in pairs]
        vals  = [p[1] for p in pairs]
        ax.bar(names, vals, color=col, edgecolor="black", lw=0.5)
        ax.axhspan(DRIVE_VESSEL_PCT_MIN, DRIVE_VESSEL_PCT_MAX,
                    alpha=0.15, color="green",
                    label=f"DRIVE training range "
                          f"[{DRIVE_VESSEL_PCT_MIN}%, {DRIVE_VESSEL_PCT_MAX}%]")
        ax.axhline(DRIVE_VESSEL_PCT_MEAN, color="green", ls="--", lw=1.0,
                    label=f"Mean ({DRIVE_VESSEL_PCT_MEAN}%)")
        ax.set_ylabel("Vessel %")
        ax.set_title(f"{model}", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Predicted vessel % per image vs DRIVE training distribution",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_probability_stats(per_image_rows, out_path: Path):
    """Mean / std / entropy of probability inside FOV, grouped by model."""
    by_model = defaultdict(lambda: {"mean": [], "std": [], "entropy": []})
    for r in per_image_rows:
        by_model[r["model"]]["mean"].append(r["prob_mean_in_fov"])
        by_model[r["model"]]["std"].append(r["prob_std_in_fov"])
        by_model[r["model"]]["entropy"].append(r["prob_entropy_in_fov"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, title in zip(axes,
                                ["mean", "std", "entropy"],
                                ["Probability mean inside FOV",
                                 "Probability std inside FOV",
                                 "Pixel-wise entropy inside FOV"]):
        labels = list(by_model.keys())
        data   = [by_model[m][key] for m in labels]
        # Compatible across matplotlib versions: 'tick_labels' (>=3.9) and 'labels' (older)
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
        palette = sns.color_palette("Set2", n_colors=len(labels))
        for patch, col in zip(bp["boxes"], palette):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Probability calibration signals (no ground truth needed)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_agreement_heatmap(agreement_matrix, labels, out_path: Path,
                            metric_name="Dice"):
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels)),
                                     max(5, 0.7 * len(labels))))
    sns.heatmap(agreement_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels,
                vmin=0.5, vmax=1.0, square=True, cbar=True, ax=ax,
                annot_kws={"size": 9})
    ax.set_title(f"Inter-model agreement ({metric_name})\n"
                 f"between binary predictions — no ground truth used",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_map(prob_stack, fov, stem, out_path: Path):
    """Per-pixel std across models / runs, with a histogram inset."""
    if prob_stack.shape[0] < 2:
        return
    pixel_std = prob_stack.std(axis=0)
    pixel_std[fov == 0] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(pixel_std, cmap="hot", vmin=0, vmax=0.5)
    axes[0].set_title(f"{stem} — per-pixel std across {prob_stack.shape[0]} model(s)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.045)

    vals = pixel_std[fov == 1]
    axes[1].hist(vals[~np.isnan(vals)], bins=40, color="#E91E63",
                  edgecolor="black", lw=0.3)
    axes[1].set_title("Distribution of per-pixel std inside FOV")
    axes[1].set_xlabel("Probability std")
    axes[1].set_ylabel("Pixel count (FOV only)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_disagreement_map(binary_stack, fov, stem, out_path: Path,
                           model_labels):
    """For each pixel, how many of the N models predict vessel."""
    if binary_stack.shape[0] < 2:
        return
    n = binary_stack.shape[0]
    vote = binary_stack.sum(axis=0).astype(np.float32)
    vote[fov == 0] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black")
    im = axes[0].imshow(vote, cmap=cmap, vmin=0, vmax=n)
    axes[0].set_title(f"{stem} — votes per pixel "
                       f"(0 = no model, {n} = unanimous)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.045, ticks=range(n + 1))

    # Bar chart of vote distribution
    counts = [int((vote == k).sum()) for k in range(n + 1)]
    labels = [f"{k}/{n}" for k in range(n + 1)]
    palette = sns.color_palette("viridis", n_colors=n + 1)
    axes[1].bar(labels, counts, color=palette,
                 edgecolor="black", lw=0.3)
    axes[1].set_title("Pixel count per agreement level (FOV only)")
    axes[1].set_ylabel("Pixels")
    axes[1].set_xlabel("Number of models predicting vessel")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Self-consistency analysis of test-set predictions "
                    "(no ground truth required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", "-i", required=True,
                    help="Root folder produced by predict_test.py "
                         "(default outputs/predictions/test/).")
    ap.add_argument("--output-dir", "-o", default=None,
                    help="Where to write the analysis. Defaults to "
                         "<input>/_analysis/.")
    ap.add_argument("--models", "-m", default=None,
                    help="Comma-separated list of model labels to include, "
                         "e.g. 'original/fold0_seed42,msca/ensemble_fall_sall'. "
                         "If omitted, every model found is included.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold used to binarise probability maps for "
                         "the per-image stats and inter-model agreement.")
    ap.add_argument("--no-perimage-maps", action="store_true",
                    help="Skip per-image uncertainty/disagreement heatmaps.")
    args = ap.parse_args()

    input_root = Path(args.input)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else input_root / "_analysis")
    ensure_dirs(output_dir)

    # ── Discover predictions ────────────────────────────────────────────
    layout = discover_predictions(input_root)
    if not layout:
        ap.error(f"No prediction folders found under {input_root}. "
                 f"Run predict_test.py first.")

    if args.models:
        wanted = set(m.strip() for m in args.models.split(","))
        layout = {k: v for k, v in layout.items() if k in wanted}
        if not layout:
            ap.error(f"None of the requested models were found. "
                     f"Available: {sorted(discover_predictions(input_root))}")

    # All image stems that appear in at least one model
    all_stems = sorted({stem for d in layout.values() for stem in d})
    n_models  = len(layout)
    n_images  = len(all_stems)

    logger.info("=" * 70)
    logger.info(f"Analysing {n_models} model(s) × {n_images} image(s) "
                f"from {input_root}")
    for label in layout:
        logger.info(f"  • {label} ({len(layout[label])} image(s))")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Threshold: {args.threshold:.2f}")
    logger.info("=" * 70)

    # ── Per-image, per-model stats ─────────────────────────────────────
    per_image_rows = []
    # Cache prob/binary/fov per (model, stem) so we reuse them later
    prob_cache, bin_cache, fov_cache = {}, {}, {}

    for label, stem_to_path in layout.items():
        for stem, prob_path in stem_to_path.items():
            prob = np.load(prob_path).astype(np.float32)
            fov  = derive_fov_from_prob(prob)
            stats = compute_image_stats(prob, fov, args.threshold)
            row = {"model": label, "image": stem, **stats}
            per_image_rows.append(row)
            prob_cache[(label, stem)] = prob
            bin_cache[(label, stem)]  = ((prob >= args.threshold) &
                                          (fov == 1)).astype(np.uint8)
            fov_cache[(label, stem)]  = fov
            logger.info(f"  {label:40s} {stem:20s} "
                         f"vessel%={stats['vessel_pct_fov']:5.2f} "
                         f"plausible={stats['plausibility_flag']} "
                         f"comps={stats['n_connected_components']}")

    # ── Save tables ────────────────────────────────────────────────────
    keys = ["model", "image", "vessel_pct_fov", "plausibility_flag",
            "prob_mean_in_fov", "prob_std_in_fov", "prob_entropy_in_fov",
            "n_connected_components", "largest_component_pct",
            "skeleton_length", "branch_points"]
    csv_path = output_dir / "per_image_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_image_rows:
            w.writerow({k: row.get(k, "") for k in keys})
    with open(output_dir / "per_image_stats.json", "w") as f:
        json.dump(per_image_rows, f, indent=2)

    # ── Plot plausibility & probability stats ──────────────────────────
    plot_plausibility(per_image_rows,
                       output_dir / "plausibility_summary.png")
    plot_probability_stats(per_image_rows,
                            output_dir / "probability_stats.png")

    # ── Inter-model agreement (Dice + IoU) ─────────────────────────────
    if n_models >= 2:
        labels = list(layout.keys())
        dice_mat = np.eye(n_models, dtype=np.float64)
        iou_mat  = np.eye(n_models, dtype=np.float64)

        # Average Dice/IoU per pair across the images both models cover
        for (i, l1), (j, l2) in combinations(enumerate(labels), 2):
            shared = sorted(set(layout[l1]) & set(layout[l2]))
            if not shared:
                continue
            dices, ious = [], []
            for stem in shared:
                d, u = pairwise_dice_iou(bin_cache[(l1, stem)],
                                          bin_cache[(l2, stem)])
                dices.append(d); ious.append(u)
            dice_mat[i, j] = dice_mat[j, i] = float(np.mean(dices))
            iou_mat[i, j]  = iou_mat[j, i]  = float(np.mean(ious))

        # Save CSV
        with open(output_dir / "inter_model_agreement.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", *labels])
            for i, lbl in enumerate(labels):
                w.writerow([f"{lbl} (dice)", *[f"{x:.4f}" for x in dice_mat[i]]])
            w.writerow([])
            for i, lbl in enumerate(labels):
                w.writerow([f"{lbl} (iou)",  *[f"{x:.4f}" for x in iou_mat[i]]])

        plot_agreement_heatmap(dice_mat, labels,
                                output_dir / "inter_model_agreement_dice.png",
                                metric_name="Dice")
        plot_agreement_heatmap(iou_mat,  labels,
                                output_dir / "inter_model_agreement_iou.png",
                                metric_name="IoU")
        logger.info(f"Inter-model Dice (mean off-diagonal): "
                    f"{(dice_mat.sum() - dice_mat.trace()) / max(1, n_models * (n_models - 1)):.4f}")

    # ── Per-image uncertainty / disagreement maps ──────────────────────
    if n_models >= 2 and not args.no_perimage_maps:
        unc_dir = output_dir / "per_image_uncertainty"
        dis_dir = output_dir / "per_image_disagreement"
        ensure_dirs(unc_dir, dis_dir)
        for stem in all_stems:
            probs, bins, fov = [], [], None
            for label in layout:
                if stem not in layout[label]:
                    continue
                probs.append(prob_cache[(label, stem)])
                bins.append(bin_cache[(label, stem)])
                if fov is None:
                    fov = fov_cache[(label, stem)]
            if len(probs) < 2:
                continue
            prob_stack = np.stack(probs, axis=0)
            bin_stack  = np.stack(bins,  axis=0)
            plot_uncertainty_map(prob_stack, fov, stem,
                                  unc_dir / f"{stem}_uncertainty.png")
            plot_disagreement_map(bin_stack, fov, stem,
                                   dis_dir / f"{stem}_disagreement.png",
                                   list(layout.keys()))
        logger.info(f"  per-image uncertainty maps  → {unc_dir}")
        logger.info(f"  per-image disagreement maps → {dis_dir}")

    # ── Top-level summary ──────────────────────────────────────────────
    summary = {
        "n_models":   n_models,
        "n_images":   n_images,
        "threshold":  args.threshold,
        "models":     list(layout.keys()),
        "warnings":   [],
        "per_model_summary": {},
    }
    for label in layout:
        rows = [r for r in per_image_rows if r["model"] == label]
        if not rows:
            continue
        v = [r["vessel_pct_fov"] for r in rows]
        summary["per_model_summary"][label] = {
            "mean_vessel_pct":      float(np.mean(v)),
            "std_vessel_pct":       float(np.std(v)),
            "n_implausible_images": int(sum(1 for r in rows
                                              if r["plausibility_flag"] == 0)),
            "median_n_components":  float(np.median([r["n_connected_components"]
                                                       for r in rows])),
        }
        if summary["per_model_summary"][label]["n_implausible_images"] > 0:
            summary["warnings"].append(
                f"{label}: {summary['per_model_summary'][label]['n_implausible_images']} "
                f"image(s) outside DRIVE training vessel-% range "
                f"[{DRIVE_VESSEL_PCT_MIN}, {DRIVE_VESSEL_PCT_MAX}].")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"Done. Analysis written to {output_dir}")
    if summary["warnings"]:
        logger.warning("Plausibility warnings:")
        for w in summary["warnings"]:
            logger.warning(f"  • {w}")


if __name__ == "__main__":
    main()