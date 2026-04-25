"""
prepare_data.py — One-time preprocessing of DRIVE dataset.

Reads raw DRIVE .tif / .gif files, applies:
  1. Green channel extraction
  2. CLAHE enhancement
  3. Normalisation to [0, 1]
  4. Save processed arrays as .npy

Run once before training:
    python prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import config as C
from utils import ensure_dirs, get_logger, set_seed

logger = get_logger("prepare_data", C.LOG_DIR / "prepare_data.log")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load any image (tif/gif/png) as uint8 numpy array."""
    img = np.array(Image.open(path).convert("RGB"))
    return img


def load_mask(path: Path) -> np.ndarray:
    """Load binary mask. Returns uint8 array with values {0, 1}."""
    m = np.array(Image.open(path).convert("L"))
    return (m > 127).astype(np.uint8)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to uint8 grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_image(rgb: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
      RGB → green channel → CLAHE → normalise to [0,1] → float32
    Returns shape (H, W) float32.
    """
    green = rgb[:, :, 1]                            # extract green channel
    enhanced = apply_clahe(green)                   # CLAHE
    normalised = enhanced.astype(np.float32) / 255.0  # [0, 1]
    return normalised


# ──────────────────────────────────────────────────────────────────────────────
# Sanity-check visual (optional)
# ──────────────────────────────────────────────────────────────────────────────

def save_sanity_check(img_rgb, vessel_mask, fov_mask, proc_img, out_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB")
    axes[1].imshow(vessel_mask, cmap="gray")
    axes[1].set_title("Vessel Mask GT")
    axes[2].imshow(fov_mask, cmap="gray")
    axes[2].set_title("FOV Mask")
    axes[3].imshow(proc_img, cmap="gray")
    axes[3].set_title("Processed (Green+CLAHE)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────────────

def process_split(split: str, n_sanity: int = 3):
    """Process all images in a DRIVE split ('training' or 'test')."""
    img_dir  = C.DATA_RAW / split / "images"
    mask_dir = C.DATA_RAW / split / "1st_manual"
    fov_dir  = C.DATA_RAW / split / "mask"

    if not img_dir.exists():
        logger.warning(f"Directory not found: {img_dir} — skipping {split}")
        return

    img_files = sorted(img_dir.glob("*.tif"))
    logger.info(f"[{split}] Found {len(img_files)} images")

    sanity_dir = C.VIZ_DIR / "sanity_checks" / split
    ensure_dirs(sanity_dir)

    for idx, img_path in enumerate(img_files):
        stem = img_path.stem                        # e.g. "01_training"
        img_id = stem.split("_")[0]                 # "01"

        # ── find matching mask and fov files ──────────────────────────────
        # DRIVE naming: mask -> 01_manual1.gif / 01_manual1.png  (varies)
        # Try multiple patterns
        mask_path = _find_file(mask_dir, img_id, ["gif", "png", "tif"])
        fov_path  = _find_file(fov_dir,  img_id, ["gif", "png", "tif"])

        if mask_path is None or fov_path is None:
            logger.warning(f"  [{stem}] Could not find mask or fov — skipping")
            continue

        # ── load ─────────────────────────────────────────────────────────
        rgb    = load_image(img_path)
        vessel = load_mask(mask_path)
        fov    = load_mask(fov_path)
        proc   = preprocess_image(rgb)

        # ── sanity check ─────────────────────────────────────────────────
        if idx < n_sanity:
            save_sanity_check(rgb, vessel, fov, proc,
                              sanity_dir / f"{stem}_check.png")

        # ── save processed arrays ─────────────────────────────────────────
        out_stem = f"{img_id}_{split}"
        np.save(C.IMG_DIR  / f"{out_stem}.npy", proc.astype(np.float32))
        np.save(C.MASK_DIR / f"{out_stem}.npy", vessel.astype(np.uint8))
        np.save(C.FOV_DIR  / f"{out_stem}.npy", fov.astype(np.uint8))

        # Stats
        vessel_frac = vessel[fov == 1].mean() * 100
        logger.info(f"  [{stem}] shape={proc.shape}  vessel_frac={vessel_frac:.1f}%")

    logger.info(f"[{split}] Done.\n")


def _find_file(directory: Path, img_id: str,
               extensions: list[str]) -> Path | None:
    """Try to locate a mask file with a given image ID in several formats."""
    for ext in extensions:
        candidates = sorted(directory.glob(f"{img_id}*.{ext}"))
        if candidates:
            return candidates[0]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(C.SEED)
    ensure_dirs(C.IMG_DIR, C.MASK_DIR, C.FOV_DIR, C.SPLIT_DIR,
                C.LOG_DIR, C.VIZ_DIR)

    logger.info("=" * 60)
    logger.info("DRIVE Data Preparation")
    logger.info("=" * 60)

    process_split("training")
    # NOTE: DRIVE test images (01-20) have no public ground truth labels.
    # We therefore evaluate on a held-out subset of the labeled training images.
    # process_split("test") is intentionally skipped.

    # ── write split ID files ─────────────────────────────────────────────
    # All IDs refer to the training folder (21-40).
    _write_ids("train_ids.txt", [f"{i:02d}_training" for i in C.TRAIN_IDS])
    _write_ids("val_ids.txt",   [f"{i:02d}_training" for i in C.VAL_IDS])
    _write_ids("test_ids.txt",  [f"{i:02d}_training" for i in C.TEST_IDS])

    logger.info("Split files written.")
    logger.info("All preprocessing complete.")


def _write_ids(filename: str, ids: list[str]):
    path = C.SPLIT_DIR / filename
    ensure_dirs(path.parent)
    with open(path, "w") as f:
        f.write("\n".join(ids) + "\n")


if __name__ == "__main__":
    main()
