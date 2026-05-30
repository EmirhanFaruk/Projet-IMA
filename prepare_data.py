"""
prepare_data.py — DRIVE preprocessing pipeline with full visualisation.

Pipeline:
  Raw RGB → Green channel → CLAHE → Normalise [0,1] → save as .npy

Outputs:
  data/processed/images/<stem>.npy       (float32 [0,1], shape (H,W))
  data/processed/masks/<stem>.npy        (uint8 {0,1})
  data/processed/fov/<stem>.npy          (uint8 {0,1})
  data/processed/splits/{train,val,test}_ids.txt
  outputs/visualizations/preprocessing/  (step-by-step subplots — REPORT-READY)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from utils import get_logger, ensure_dirs

logger = get_logger("prepare_data")


# ─────────────────────────── Preprocessing ───────────────────────────────

def extract_channel(rgb_image: np.ndarray, channel: str = "green") -> np.ndarray:
    if channel == "green":
        return rgb_image[..., 1]
    elif channel == "gray":
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unknown channel: {channel}")


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: tuple) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def normalize_image(image: np.ndarray, method: str = "0_1") -> np.ndarray:
    img = image.astype(np.float32)
    if method == "0_1":
        return img / 255.0
    elif method == "z_score":
        return (img - img.mean()) / (img.std() + 1e-8)
    return img


def preprocess_image(rgb_image: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline (final output only)."""
    gray = extract_channel(rgb_image, C.CHANNEL)
    if C.USE_CLAHE:
        gray = apply_clahe(gray, C.CLAHE_CLIP_LIMIT, C.CLAHE_TILE_GRID)
    return normalize_image(gray, C.NORMALIZATION)


# ─────────────────────────── Mask processing ─────────────────────────────

def load_vessel_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


def load_fov_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


# ─────────────────────────── Visualisation ───────────────────────────────

def visualize_preprocessing_steps(rgb_image, mask, fov, stem, viz_dir):
    """
    Save a multi-panel figure showing each preprocessing step on one image.
    Used for the report to demonstrate the pipeline visually.
    """
    # Step-by-step intermediates
    green     = extract_channel(rgb_image, "green")
    gray      = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    after_clahe = apply_clahe(green, C.CLAHE_CLIP_LIMIT, C.CLAHE_TILE_GRID)
    normalized = normalize_image(after_clahe, "0_1")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Preprocessing pipeline: {stem}\n"
                 f"CLAHE clip={C.CLAHE_CLIP_LIMIT}, "
                 f"tile_grid={C.CLAHE_TILE_GRID}, "
                 f"norm={C.NORMALIZATION}",
                 fontsize=14, fontweight="bold")

    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("1. Original RGB")
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("(alt) Grayscale")
    axes[0, 2].imshow(green, cmap="gray")
    axes[0, 2].set_title("2. Green Channel\n(highest vessel contrast)")
    axes[0, 3].imshow(after_clahe, cmap="gray")
    axes[0, 3].set_title(f"3. + CLAHE\n(clip={C.CLAHE_CLIP_LIMIT})")

    axes[1, 0].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("4. Normalised → [0,1]\n(input to model)")
    axes[1, 1].imshow(mask, cmap="gray")
    axes[1, 1].set_title("Vessel ground truth\n(1st_manual)")
    axes[1, 2].imshow(fov, cmap="gray")
    axes[1, 2].set_title("FOV mask\n(metric region)")

    # Histograms — green vs after CLAHE
    axes[1, 3].hist(green.ravel(),       bins=50, alpha=0.5, label="Green channel",
                     color="green", density=True)
    axes[1, 3].hist(after_clahe.ravel(), bins=50, alpha=0.5, label="After CLAHE",
                     color="orange", density=True)
    axes[1, 3].set_title("Intensity Histograms")
    axes[1, 3].set_xlabel("Intensity")
    axes[1, 3].set_ylabel("Density")
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax is not axes[1, 3]:
            ax.axis("off")

    plt.tight_layout()
    out = viz_dir / f"preprocessing_{stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_clahe_comparison(rgb_image, stem, viz_dir):
    """Compare different CLAHE clip limits to show their effect."""
    green = extract_channel(rgb_image, "green")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"CLAHE clip-limit comparison — {stem}",
                 fontsize=14, fontweight="bold")

    settings = [
        ("No CLAHE", green),
        ("clip=1 (mild)",   apply_clahe(green, 1.0, (8, 8))),
        ("clip=2 (default)", apply_clahe(green, 2.0, (8, 8))),
        ("clip=4 (strong)",  apply_clahe(green, 4.0, (8, 8))),
    ]
    for ax, (title, img) in zip(axes, settings):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out = viz_dir / f"clahe_comparison_{stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def visualize_dataset_summary(viz_dir):
    """Aggregate stats: vessel fractions across all images, etc."""
    fracs = []
    ids   = []
    for img_id in C.ALL_IDS:
        stem = f"{img_id:02d}_training"
        m = np.load(C.MASK_DIR / f"{stem}.npy")
        f = np.load(C.FOV_DIR  / f"{stem}.npy")
        frac = m[f == 1].mean() * 100
        fracs.append(frac)
        ids.append(img_id)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DRIVE dataset — vessel statistics inside FOV",
                 fontsize=14, fontweight="bold")

    colors = ["#4CAF50" if f >= np.mean(fracs) else "#F44336" for f in fracs]
    axes[0].bar(ids, fracs, color=colors, edgecolor="black", lw=0.5)
    axes[0].axhline(np.mean(fracs), color="navy", ls="--",
                    label=f"Mean = {np.mean(fracs):.1f}%")
    axes[0].set_xlabel("Image ID")
    axes[0].set_ylabel("Vessel pixel fraction inside FOV (%)")
    axes[0].set_title("Per-image vessel fraction")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].hist(fracs, bins=10, color="#2196F3",
                 edgecolor="black", lw=0.5)
    axes[1].axvline(np.mean(fracs), color="red", ls="--",
                    label=f"Mean = {np.mean(fracs):.1f}%")
    axes[1].set_xlabel("Vessel fraction (%)")
    axes[1].set_ylabel("Number of images")
    axes[1].set_title("Distribution across 20 annotated images")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out = viz_dir / "dataset_vessel_stats.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────── Main pipeline ───────────────────────────────

def process_set(set_name: str, viz_dir: Path, save_viz_for: list = None):
    """Process either 'training' or 'test' split.

    Note on DRIVE structure:
      training/{images,mask,1st_manual} - all 20 are labelled
      test/{images,mask}                 - no public ground truth

    We process both folders for image saving, but only training/ has
    vessel ground truth so only those can be evaluated.
    """
    img_dir = C.DATA_RAW / set_name / "images"
    fov_dir = C.DATA_RAW / set_name / "mask"
    gt_dir  = C.DATA_RAW / set_name / "1st_manual" if set_name == "training" else None

    if not img_dir.exists():
        logger.warning(f"Missing {img_dir} — skipping {set_name}")
        return

    image_files = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.tiff"))
    if not image_files:
        image_files = sorted(img_dir.iterdir())
    logger.info(f"Processing {set_name}: {len(image_files)} images")

    save_viz_for = save_viz_for or []

    for img_path in image_files:
        stem_raw = img_path.stem
        # DRIVE filenames: NN_training.tif or NN_test.tif
        # We standardize on the format <NN>_training (since that's what the
        # rest of our pipeline expects) — even for the test/ folder we
        # keep them under the unified name.
        out_stem = stem_raw if "_training" in stem_raw else stem_raw

        # Load RGB
        rgb = np.array(Image.open(img_path).convert("RGB"))

        # Preprocess
        proc = preprocess_image(rgb)

        # FOV mask
        fov_candidates = list(fov_dir.glob(f"{out_stem.split('_')[0]}_*"))
        if not fov_candidates:
            logger.warning(f"  No FOV for {out_stem}")
            continue
        fov = load_fov_mask(fov_candidates[0])

        # Vessel mask (only for training/)
        if gt_dir is not None:
            gt_candidates = list(gt_dir.glob(f"{out_stem.split('_')[0]}_*"))
            if gt_candidates:
                mask = load_vessel_mask(gt_candidates[0])
            else:
                logger.warning(f"  No GT for {out_stem}")
                continue
        else:
            # No ground truth available — we don't save these (can't evaluate)
            logger.info(f"  Skip {out_stem} (no ground truth in test/ folder)")
            continue

        # Save .npy
        np.save(C.IMG_DIR  / f"{out_stem}.npy", proc.astype(np.float32))
        np.save(C.MASK_DIR / f"{out_stem}.npy", mask.astype(np.uint8))
        np.save(C.FOV_DIR  / f"{out_stem}.npy", fov.astype(np.uint8))

        # Save preprocessing-pipeline visualization for selected images
        if out_stem in save_viz_for:
            visualize_preprocessing_steps(rgb, mask, fov, out_stem, viz_dir)
            visualize_clahe_comparison(rgb, out_stem, viz_dir)
            logger.info(f"  ✓ Saved preprocessing visualisations for {out_stem}")

        logger.info(f"  ✓ {out_stem}  (img={proc.shape}, "
                    f"vessel_frac={mask[fov==1].mean()*100:.1f}%)")


def main():
    ensure_dirs(C.IMG_DIR, C.MASK_DIR, C.FOV_DIR, C.SPLIT_DIR)
    viz_dir = C.VIZ_DIR / "preprocessing"
    ensure_dirs(viz_dir)

    # Select 3 representative images for detailed visualisation
    save_viz_for = ["21_training", "30_training", "40_training"]

    process_set("training", viz_dir, save_viz_for=save_viz_for)
    process_set("test",     viz_dir)   # will be skipped (no GT)

    # Save split files (legacy — CV is preferred)
    def _write_ids(name, ids):
        with open(C.SPLIT_DIR / name, "w") as f:
            f.write("\n".join(ids))

    _write_ids("train_ids.txt",
               [f"{i:02d}_training" for i in C.TRAIN_IDS_FIXED])
    _write_ids("val_ids.txt",
               [f"{i:02d}_training" for i in C.VAL_IDS_FIXED])
    _write_ids("test_ids.txt",
               [f"{i:02d}_training" for i in C.TEST_IDS_FIXED])

    # Generate CV split files now for transparency
    from cv_splits import generate_fold_splits
    folds = generate_fold_splits(seed=C.SEED, n_folds=C.N_FOLDS,
                                  val_frac=C.VAL_FRAC_OF_TRAIN)
    import json
    cv_path = C.SPLIT_DIR / f"cv_splits_n{C.N_FOLDS}_seed{C.SEED}.json"
    with open(cv_path, "w") as f:
        json.dump(folds, f, indent=2)
    logger.info(f"Saved CV splits → {cv_path}")
    for fold in folds:
        logger.info(f"  Fold {fold['fold']}: train={fold['train']}")
        logger.info(f"           val={fold['val']}, test={fold['test']}")

    # Dataset summary visualisation
    visualize_dataset_summary(viz_dir)
    logger.info(f"All visualisations → {viz_dir}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
