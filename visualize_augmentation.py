"""
visualize_augmentation.py — Visualise the four augmentation levels on a sample patch.

Produces a grid showing the same vessel-centred patch transformed by each level
(none/light/medium/heavy), with multiple random samples per level to see
the variability the model trains on.

Output: outputs/visualizations/augmentation/aug_comparison.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from dataset import get_train_transform, balanced_sample_patches
from utils import get_logger, ensure_dirs

logger = get_logger("visualize_aug")


def main():
    out_dir = C.VIZ_DIR / "augmentation"
    ensure_dirs(out_dir)

    # Load a vessel-centred patch from image 21
    stem = "21_training"
    img  = np.load(C.IMG_DIR  / f"{stem}.npy")
    mask = np.load(C.MASK_DIR / f"{stem}.npy").astype(np.uint8)
    fov  = np.load(C.FOV_DIR  / f"{stem}.npy").astype(np.uint8)

    rng = np.random.default_rng(42)
    patches = balanced_sample_patches(img, mask, fov, C.PATCH_SIZE, 4, rng)
    if not patches:
        logger.error("Could not extract patches; run prepare_data.py first.")
        return
    p   = patches[0]
    img_p = p["img"].astype(np.float32)
    msk_p = p["mask"].astype(np.uint8)
    fov_p = p["fov"].astype(np.uint8)

    # Layout: 4 rows (levels) × 6 columns (1 image + 5 augmented samples)
    levels = ["none", "light", "medium", "heavy"]
    n_samples = 5

    fig, axes = plt.subplots(len(levels) * 2, n_samples + 1,
                              figsize=(3 * (n_samples + 1), 3 * len(levels) * 2))
    fig.suptitle(f"Augmentation comparison on patch from {stem}\n"
                 f"Top row of each pair = image, bottom row = vessel mask",
                 fontsize=14, fontweight="bold")

    for li, lvl in enumerate(levels):
        transform = get_train_transform(lvl)

        # Original (column 0)
        axes[2 * li,     0].imshow(img_p, cmap="gray", vmin=0, vmax=1)
        axes[2 * li,     0].set_title(f"{lvl}\n(original)", fontweight="bold")
        axes[2 * li,     0].axis("off")
        axes[2 * li + 1, 0].imshow(msk_p, cmap="gray")
        axes[2 * li + 1, 0].set_title("vessel mask")
        axes[2 * li + 1, 0].axis("off")

        # Augmented samples
        for s in range(n_samples):
            if lvl == "none":
                aug_img, aug_msk = img_p, msk_p
            else:
                out = transform(image=img_p, mask=msk_p, fov=fov_p)
                aug_img = out["image"]
                aug_msk = out["mask"]
            axes[2 * li,     s + 1].imshow(aug_img, cmap="gray", vmin=0, vmax=1)
            axes[2 * li,     s + 1].set_title(f"sample {s+1}")
            axes[2 * li,     s + 1].axis("off")
            axes[2 * li + 1, s + 1].imshow(aug_msk, cmap="gray")
            axes[2 * li + 1, s + 1].axis("off")

    plt.tight_layout()
    out = out_dir / "augmentation_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    main()
