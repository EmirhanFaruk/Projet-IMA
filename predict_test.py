"""
predict_test.py — Inference on images without ground truth.

Supports inference with one, several, or all trained models in a single run.
For each (experiment, fold, seed) combination requested, the corresponding
best_model.pth is loaded and predictions are written to a separate folder.

Selection of (experiment, fold, seed) follows the same syntax for all three
arguments:

    "best"     → auto-pick the (fold, seed) with the highest val Dice
                 (only meaningful for --fold and --seed; not for --experiment)
    "all"      → use every available value
    "0,1,3"    → explicit comma-separated list
    "42"       → a single value

Ensemble mode (--ensemble) averages probability maps across all selected
(fold, seed) for each experiment, giving one set of predictions per
experiment under outputs/predictions/test/<experiment>/ensemble_<tag>/.

Usage:
    # Single experiment, best (fold, seed) — default behaviour
    python predict_test.py -i data/raw/DRIVE/test/images/

    # All five experiments, each with its own best run
    python predict_test.py -e all -i data/raw/DRIVE/test/images/

    # All experiments, all 15 runs each (75 checkpoints in total)
    python predict_test.py -e all --fold all --seed all \\
        -i data/raw/DRIVE/test/images/

    # All experiments, fold 0 only, seed 42 only
    python predict_test.py -e all --fold 0 --seed 42 \\
        -i data/raw/DRIVE/test/images/

    # Ensemble: 5-fold × 3-seed mean probability per experiment
    python predict_test.py -e all --fold all --seed all --ensemble \\
        -i data/raw/DRIVE/test/images/

    # Just msca and deva, both folds 0 and 2, all seeds
    python predict_test.py -e msca,deva --fold 0,2 --seed all \\
        -i data/raw/DRIVE/test/images/
"""

import sys
import json
import argparse
import time
from pathlib import Path

# Ensure src/ is on the path so we can import config, model, infer, ...
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config as C
from model import build_model
from infer import infer_full_image, infer_tta
from utils import get_logger, load_checkpoint, ensure_dirs, set_seed

logger = get_logger("predict_test", C.LOG_DIR / "predict_test.log")

ALL_EXPERIMENTS = ["original", "msca", "deva",
                   "ablation_spatial", "ablation_channel"]


# ─────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}
NPY_EXT  = ".npy"


def list_input_images(input_path: Path):
    """A single file → list of one path; a directory → its image files."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir()
                       if p.is_file()
                       and p.suffix.lower() in IMG_EXTS | {NPY_EXT})
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


# ─────────────────────────────────────────────────────────────────────────
# Preprocessing — mirrors prepare_data.preprocess_image
# ─────────────────────────────────────────────────────────────────────────

def extract_green(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 1]


def apply_clahe(img: np.ndarray, clip: float, tile_grid: tuple) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid).apply(img)


def preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    g = extract_green(rgb) if C.CHANNEL == "green" \
        else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if C.USE_CLAHE:
        g = apply_clahe(g, C.CLAHE_CLIP_LIMIT, C.CLAHE_TILE_GRID)
    return g.astype(np.float32) / 255.0


def derive_fov_mask(rgb: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Auto-derive a FOV mask from the dark border of a fundus image."""
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == biggest).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (mask > 0).astype(np.uint8)


def load_image_to_arrays(path: Path):
    """Load `path` → (rgb_uint8, preprocessed_float32, fov_uint8)."""
    if path.suffix.lower() == NPY_EXT:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            rgb = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            return rgb, preprocess_rgb(rgb), derive_fov_mask(rgb)
        preproc = arr if arr.max() <= 1.0 else arr / 255.0
        disp = (np.clip(preproc, 0, 1) * 255).astype(np.uint8)
        return (np.stack([disp]*3, axis=-1), preproc.astype(np.float32),
                (disp > 5).astype(np.uint8))

    rgb = np.array(Image.open(path).convert("RGB"))
    return rgb, preprocess_rgb(rgb), derive_fov_mask(rgb)


# ─────────────────────────────────────────────────────────────────────────
# Run discovery & selection
# ─────────────────────────────────────────────────────────────────────────

def discover_runs(experiment: str):
    """Scan the metric directory and return [(fold, seed, summary, run_dir, ckpt_path), ...]
    for every available run of `experiment`."""
    metric_dir = C.METRIC_DIR / experiment
    if not metric_dir.exists():
        return []
    out = []
    for run_dir in sorted(metric_dir.glob("fold*_seed*")):
        if "hparam" in run_dir.name:
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        fold = summary.get("fold")
        seed = summary.get("seed")
        if fold is None or seed is None:
            try:
                parts = run_dir.name.split("_")
                fold = int(parts[0].replace("fold", ""))
                seed = int(parts[1].replace("seed", ""))
            except Exception:
                continue
        ckpt_path = C.CKPT_DIR / experiment / run_dir.name / "best_model.pth"
        if not ckpt_path.exists():
            continue
        out.append((fold, seed, summary, run_dir, ckpt_path))
    return out


def parse_selection(value: str, allowed_values=None, support_best=True):
    """Parse a CLI selection string.

    Returns one of:
      ("all",  None)            → use every available
      ("best", None)            → pick the best one (only valid for fold/seed)
      ("explicit", [v1, v2,...]) → use exactly those values
    """
    v = (value or "best").strip().lower()
    if v == "all":
        return ("all", None)
    if v == "best":
        if not support_best:
            raise argparse.ArgumentTypeError(
                "'best' is not allowed for this argument.")
        return ("best", None)
    try:
        values = [int(x.strip()) for x in v.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Could not parse '{value}' as 'best', 'all', or a "
            f"comma-separated list of integers.")
    if not values:
        raise argparse.ArgumentTypeError(f"Empty selection: '{value}'")
    if allowed_values is not None:
        bad = [x for x in values if x not in allowed_values]
        if bad:
            raise argparse.ArgumentTypeError(
                f"Values {bad} not in allowed set {sorted(allowed_values)}.")
    return ("explicit", values)


def parse_experiment_selection(value: str):
    """Parse the --experiment argument: 'all' | comma-separated names | single name."""
    v = (value or "original").strip().lower()
    if v == "all":
        return list(ALL_EXPERIMENTS)
    names = [x.strip() for x in v.split(",") if x.strip()]
    bad = [n for n in names if n not in ALL_EXPERIMENTS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown experiment(s) {bad}. "
            f"Allowed: {ALL_EXPERIMENTS} or 'all'.")
    return names


def select_runs(experiment: str, fold_sel, seed_sel, metric_key="dice_mean"):
    """Select which runs of `experiment` to use given the parsed selections.

    Returns a list of (fold, seed, summary, run_dir, ckpt_path).
    """
    runs = discover_runs(experiment)
    if not runs:
        return []

    fold_kind, fold_vals = fold_sel
    seed_kind, seed_vals = seed_sel

    # Apply explicit fold filter
    if fold_kind == "explicit":
        runs = [r for r in runs if r[0] in fold_vals]
    # Apply explicit seed filter
    if seed_kind == "explicit":
        runs = [r for r in runs if r[1] in seed_vals]

    if not runs:
        return []

    # If neither dim is "best", return as-is
    if fold_kind != "best" and seed_kind != "best":
        return runs

    # Both "best" → single best run overall
    if fold_kind == "best" and seed_kind == "best":
        return [max(runs, key=lambda r: r[2].get(metric_key, -1))]

    # fold "best", seed "all"/explicit → best fold per seed
    if fold_kind == "best":
        by_seed = {}
        for r in runs:
            seed = r[1]
            if (seed not in by_seed
                    or r[2].get(metric_key, -1) > by_seed[seed][2].get(metric_key, -1)):
                by_seed[seed] = r
        return list(by_seed.values())

    # seed "best", fold "all"/explicit → best seed per fold
    by_fold = {}
    for r in runs:
        fold = r[0]
        if (fold not in by_fold
                or r[2].get(metric_key, -1) > by_fold[fold][2].get(metric_key, -1)):
            by_fold[fold] = r
    return list(by_fold.values())


# ─────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────

def save_visualization(rgb, preproc, prob, binary, out_path: Path,
                       stem: str, experiment: str, threshold: float,
                       run_label: str):
    """5-panel summary figure: RGB | preprocessed | probability | binary | overlay."""
    overlay = (np.stack([(preproc * 255).astype(np.uint8)] * 3, axis=-1)).copy()
    overlay[binary == 1] = [0, 200, 0]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    fig.suptitle(f"{stem} — {experiment} ({run_label}) "
                 f"— threshold = {threshold:.2f}",
                 fontsize=12, fontweight="bold")
    axes[0].imshow(rgb);                                axes[0].set_title("Original RGB")
    axes[1].imshow(preproc, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Preprocessed (Green + CLAHE)")
    axes[2].imshow(prob, cmap="jet", vmin=0, vmax=1);   axes[2].set_title("Probability map")
    axes[3].imshow(binary, cmap="gray");                axes[3].set_title(f"Binary @ τ={threshold:.2f}")
    axes[4].imshow(overlay);                            axes[4].set_title("Overlay")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Single-run inference
# ─────────────────────────────────────────────────────────────────────────

def run_single_checkpoint(experiment, fold, seed, summary, ckpt_path,
                          input_paths_loaded, output_dir,
                          use_tta, threshold_override,
                          save_npy, save_viz, device):
    """Run inference with one checkpoint and write per-image artefacts."""
    threshold = (threshold_override if threshold_override is not None
                 else float(summary.get("best_threshold", C.THRESHOLD)))
    run_label = f"fold{fold}_seed{seed}"
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    logger.info(f"  → {experiment}/{run_label}: ckpt={ckpt_path.name}, "
                f"threshold={threshold:.3f}")

    model = build_model(experiment).to(device)
    load_checkpoint(str(ckpt_path), model)
    model.eval()

    infer_fn = infer_tta if use_tta else infer_full_image
    image_results = []
    for img_path, rgb, preproc, fov in input_paths_loaded:
        stem = img_path.stem
        prob = infer_fn(model, preproc, fov, device)
        binary = ((prob >= threshold) & (fov == 1)).astype(np.uint8)
        vessel_pct = 100.0 * binary[fov == 1].mean() if fov.any() else 0.0

        if save_npy:
            np.save(output_dir / f"{stem}_prob.npy", prob.astype(np.float32))
        Image.fromarray((binary * 255).astype(np.uint8)).save(
            output_dir / f"{stem}_binary.png")
        if save_viz:
            save_visualization(rgb, preproc, prob, binary,
                                output_dir / f"{stem}_viz.png",
                                stem, experiment, threshold, run_label)

        image_results.append({
            "stem":           stem,
            "image":          str(img_path),
            "vessel_pct_fov": float(vessel_pct),
            "threshold":      float(threshold),
            "outputs": {
                "binary":      str(output_dir / f"{stem}_binary.png"),
                "probability": (str(output_dir / f"{stem}_prob.npy")
                                if save_npy else None),
                "visualization": (str(output_dir / f"{stem}_viz.png")
                                  if save_viz else None),
            },
        })

    return {
        "experiment":  experiment,
        "fold":        fold,
        "seed":        seed,
        "checkpoint":  str(ckpt_path),
        "threshold":   float(threshold),
        "use_tta":     use_tta,
        "val_dice":    summary.get("dice_mean"),
        "predictions": image_results,
    }


# ─────────────────────────────────────────────────────────────────────────
# Ensemble: average probability across selected runs
# ─────────────────────────────────────────────────────────────────────────

def run_ensemble(experiment, runs, input_paths_loaded, output_dir,
                 use_tta, threshold_override, save_npy, save_viz, device,
                 tag):
    """Average probability maps across all selected runs of one experiment."""
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)
    logger.info(f"  → ENSEMBLE {experiment} ({tag}): "
                f"averaging {len(runs)} run(s)")

    if threshold_override is not None:
        threshold = float(threshold_override)
    else:
        thrs = [float(r[2].get("best_threshold", C.THRESHOLD)) for r in runs]
        threshold = float(np.mean(thrs))
    logger.info(f"     threshold = {threshold:.3f}")

    infer_fn = infer_tta if use_tta else infer_full_image

    n_imgs = len(input_paths_loaded)
    prob_sum = [None] * n_imgs

    for ri, (fold, seed, summary, run_dir, ckpt_path) in enumerate(runs):
        logger.info(f"      [{ri+1}/{len(runs)}] fold{fold}_seed{seed} ...")
        model = build_model(experiment).to(device)
        load_checkpoint(str(ckpt_path), model)
        model.eval()
        for i, (_, _, preproc, fov) in enumerate(input_paths_loaded):
            prob = infer_fn(model, preproc, fov, device)
            prob_sum[i] = (prob if prob_sum[i] is None
                            else prob_sum[i] + prob)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    image_results = []
    for i, (img_path, rgb, preproc, fov) in enumerate(input_paths_loaded):
        prob = (prob_sum[i] / len(runs)).astype(np.float32)
        binary = ((prob >= threshold) & (fov == 1)).astype(np.uint8)
        vessel_pct = 100.0 * binary[fov == 1].mean() if fov.any() else 0.0
        stem = img_path.stem

        if save_npy:
            np.save(output_dir / f"{stem}_prob.npy", prob)
        Image.fromarray((binary * 255).astype(np.uint8)).save(
            output_dir / f"{stem}_binary.png")
        if save_viz:
            save_visualization(rgb, preproc, prob, binary,
                                output_dir / f"{stem}_viz.png",
                                stem, experiment, threshold,
                                f"ensemble_{tag}")

        image_results.append({
            "stem":           stem,
            "image":          str(img_path),
            "vessel_pct_fov": float(vessel_pct),
            "threshold":      float(threshold),
            "outputs": {
                "binary":      str(output_dir / f"{stem}_binary.png"),
                "probability": (str(output_dir / f"{stem}_prob.npy")
                                if save_npy else None),
                "visualization": (str(output_dir / f"{stem}_viz.png")
                                  if save_viz else None),
            },
        })

    return {
        "experiment":     experiment,
        "ensemble":       True,
        "ensemble_tag":   tag,
        "n_runs":         len(runs),
        "runs":           [{"fold": r[0], "seed": r[1],
                            "checkpoint": str(r[4])} for r in runs],
        "threshold":      float(threshold),
        "use_tta":        use_tta,
        "predictions":    image_results,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Predict vessel masks for unlabelled DRIVE-style images. "
                    "Supports running with one, several, or all trained "
                    "models in a single command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--experiment", "-e",
                    default="original",
                    type=parse_experiment_selection,
                    help="Architecture(s). One of "
                         "{original, msca, deva, ablation_spatial, "
                         "ablation_channel}, a comma-separated list, or "
                         "'all'.")
    ap.add_argument("--fold",       default="best",
                    help="Which fold(s): 'best' (auto-pick by val Dice), "
                         "'all', or comma-separated integers e.g. '0,2,4'.")
    ap.add_argument("--seed",       default="best",
                    help="Which seed(s): 'best', 'all', or comma-separated "
                         "integers e.g. '42,2026'.")
    ap.add_argument("--input", "-i", required=True,
                    help="Path to a single image or to a folder of images. "
                         "Supported: .tif/.tiff/.png/.jpg/.bmp/.npy.")
    ap.add_argument("--output-dir", "-o", default=None,
                    help="Destination root. Defaults to "
                         "outputs/predictions/test/. Each "
                         "(experiment, run) gets its own subfolder.")
    ap.add_argument("--ensemble",   action="store_true",
                    help="For each experiment, average probabilities across "
                         "all selected runs and write a single ensemble "
                         "prediction. Per-run outputs are still saved "
                         "unless --ensemble-only is passed.")
    ap.add_argument("--ensemble-only", action="store_true",
                    help="Only write the ensemble (skip per-run outputs). "
                         "Implies --ensemble.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Binarisation threshold. Defaults to each run's "
                         "best_threshold from summary.json.")
    ap.add_argument("--no-tta",    action="store_true",
                    help="Disable test-time augmentation (faster).")
    ap.add_argument("--no-npy",    action="store_true",
                    help="Do not save .npy probability maps.")
    ap.add_argument("--no-viz",    action="store_true",
                    help="Do not save 5-panel visualisation figures.")
    args = ap.parse_args()

    if args.ensemble_only:
        args.ensemble = True

    experiments = args.experiment
    try:
        fold_sel = parse_selection(args.fold,
                                    allowed_values=set(range(C.N_FOLDS)))
        seed_sel = parse_selection(args.seed,
                                    allowed_values=set(C.SEEDS))
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))

    # Locate inputs and load once (preprocessing is identical across runs)
    input_paths = list_input_images(Path(args.input))
    if not input_paths:
        ap.error(f"No images found at {args.input}")
    logger.info(f"Found {len(input_paths)} input image(s).")
    logger.info(f"Loading + preprocessing inputs ...")
    input_paths_loaded = []
    for p in input_paths:
        rgb, preproc, fov = load_image_to_arrays(p)
        if preproc.shape[:2] != fov.shape[:2]:
            logger.warning(f"  Shape mismatch for {p.name}; skipping.")
            continue
        input_paths_loaded.append((p, rgb, preproc, fov))
    logger.info(f"  loaded {len(input_paths_loaded)} image(s).")

    output_root = Path(args.output_dir) if args.output_dir \
                  else (C.PRED_DIR / "test")
    ensure_dirs(output_root)

    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    plan = []
    for exp in experiments:
        runs = select_runs(exp, fold_sel, seed_sel)
        if not runs:
            logger.warning(f"[{exp}] no runs available for the requested "
                           f"fold/seed selection — skipping.")
            continue
        plan.append((exp, runs))

    if not plan:
        logger.error("Nothing to do — every requested combination is missing. "
                     "Run training first.")
        return

    total_runs = sum(len(r) for _, r in plan)
    logger.info("=" * 70)
    logger.info(f"PLAN: {len(plan)} experiment(s) × runs as below "
                f"= {total_runs} checkpoint(s) total")
    for exp, runs in plan:
        run_labels = ", ".join(f"f{r[0]}s{r[1]}" for r in runs)
        logger.info(f"  {exp:20s} ({len(runs)} run(s)): {run_labels}")
    logger.info(f"  ensemble        : {args.ensemble}")
    logger.info(f"  ensemble-only   : {args.ensemble_only}")
    logger.info(f"  TTA             : {not args.no_tta}")
    logger.info(f"  output root     : {output_root}")
    logger.info("=" * 70)

    def _tag(kind, vals):
        if kind == "all":  return "all"
        if kind == "best": return "best"
        return ",".join(map(str, vals))
    ens_tag = f"f{_tag(*fold_sel)}_s{_tag(*seed_sel)}"

    overall = {
        "input":          str(args.input),
        "n_images":       len(input_paths_loaded),
        "ensemble":       args.ensemble,
        "ensemble_only":  args.ensemble_only,
        "use_tta":        not args.no_tta,
        "results":        [],
    }
    t_start = time.time()

    for exp, runs in plan:
        if not args.ensemble_only:
            for fold, seed, summary, run_dir, ckpt_path in runs:
                run_label = f"fold{fold}_seed{seed}"
                out_dir = output_root / exp / run_label
                t0 = time.time()
                summary_dict = run_single_checkpoint(
                    exp, fold, seed, summary, ckpt_path,
                    input_paths_loaded, out_dir,
                    use_tta=not args.no_tta,
                    threshold_override=args.threshold,
                    save_npy=not args.no_npy,
                    save_viz=not args.no_viz,
                    device=device)
                summary_dict["elapsed_sec"] = round(time.time() - t0, 2)
                summary_dict["output_dir"]  = str(out_dir)
                with open(out_dir / "manifest.json", "w") as f:
                    json.dump(summary_dict, f, indent=2)
                overall["results"].append(summary_dict)

        if args.ensemble:
            ens_dir = output_root / exp / f"ensemble_{ens_tag}"
            t0 = time.time()
            summary_dict = run_ensemble(
                exp, runs, input_paths_loaded, ens_dir,
                use_tta=not args.no_tta,
                threshold_override=args.threshold,
                save_npy=not args.no_npy,
                save_viz=not args.no_viz,
                device=device, tag=ens_tag)
            summary_dict["elapsed_sec"] = round(time.time() - t0, 2)
            summary_dict["output_dir"]  = str(ens_dir)
            with open(ens_dir / "manifest.json", "w") as f:
                json.dump(summary_dict, f, indent=2)
            overall["results"].append(summary_dict)

    overall["elapsed_total_sec"] = round(time.time() - t_start, 2)
    with open(output_root / "manifest.json", "w") as f:
        json.dump(overall, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"DONE. {len(overall['results'])} result group(s) written "
                f"in {overall['elapsed_total_sec']}s.")
    logger.info(f"Top-level manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()