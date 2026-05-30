"""
dataset.py — DRIVE patch dataset with configurable augmentation levels
and preprocessing options.

Uses cv_splits.get_split() to retrieve train/val/test IDs based on the active
fold (C.FOLD), seed (C.SEED), and N_FOLDS configuration.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

import config as C
from cv_splits import get_split
from utils import get_logger

logger = get_logger("dataset")

# Module-level override for hparam search
_CURRENT_AUG_LEVEL = None


# ─────────────────────────── Augmentation pipelines ──────────────────────

def get_train_transform(aug_level=None):
    level = aug_level or _CURRENT_AUG_LEVEL or C.AUG_LEVEL
    common = {"additional_targets": {"mask": "mask", "fov": "mask"}}

    if level == "none":
        return A.Compose([], **common)

    elif level == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3, border_mode=0),
        ], **common)

    elif level == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5, border_mode=0),
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                     scale=(0.90, 1.10), rotate=(-15, 15), p=0.5),
            A.ElasticTransform(alpha=1, sigma=10, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(std_range=(0.0, 0.05), p=0.2),
        ], **common)

    elif level == "heavy":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.6, border_mode=0),
            A.Affine(translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                     scale=(0.80, 1.20), rotate=(-20, 20), p=0.6),
            A.ElasticTransform(alpha=2, sigma=12, p=0.3),
            A.GridDistortion(num_steps=7, distort_limit=0.3, p=0.3),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.4),
            A.RandomGamma(gamma_limit=(60, 140), p=0.4),
            A.GaussNoise(std_range=(0.0, 0.08), p=0.3),
            A.Sharpen(alpha=(0.1, 0.3), p=0.2),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(4, 8),
                            hole_width_range=(4, 8), p=0.15),
        ], **common)

    else:
        raise ValueError(f"Unknown aug_level: {level}")


# ─────────────────────────── Patch utilities ─────────────────────────────

def extract_patches(image, mask, fov, patch_size, stride):
    H, W   = image.shape
    half   = patch_size // 2
    patches = []
    for r in range(half, H - half, stride):
        for c in range(half, W - half, stride):
            if fov[r, c] == 0:
                continue
            r0, r1 = r - half, r + half
            c0, c1 = c - half, c + half
            patches.append({
                "img":  image[r0:r1, c0:c1].copy(),
                "mask": mask[r0:r1, c0:c1].copy(),
                "fov":  fov[r0:r1, c0:c1].copy(),
            })
    return patches


def balanced_sample_patches(image, mask, fov, patch_size, n_patches, rng):
    H, W  = image.shape
    half  = patch_size // 2

    vy, vx = np.where((mask == 1) & (fov == 1))
    fy, fx = np.where(fov == 1)

    def in_bounds(ys, xs):
        ok = (ys >= half) & (ys < H - half) & (xs >= half) & (xs < W - half)
        return ys[ok], xs[ok]

    vy, vx = in_bounds(vy, vx)
    fy, fx = in_bounds(fy, fx)
    n_v, n_r = n_patches // 2, n_patches - n_patches // 2
    patches  = []

    if len(vy) > 0:
        idx = rng.choice(len(vy), size=n_v, replace=True)
        for i in idx:
            patches.append(_crop(image, mask, fov, vy[i], vx[i], half))

    if len(fy) > 0:
        idx = rng.choice(len(fy), size=n_r, replace=True)
        for i in idx:
            patches.append(_crop(image, mask, fov, fy[i], fx[i], half))

    return patches


def _crop(image, mask, fov, r, c, half):
    return {
        "img":  image[r - half:r + half, c - half:c + half].copy(),
        "mask": mask[r  - half:r + half, c - half:c + half].copy(),
        "fov":  fov[r   - half:r + half, c - half:c + half].copy(),
    }


# ─────────────────────────── Dataset classes ─────────────────────────────

def _get_active_ids(split):
    """Resolves train/val/test IDs based on current C.FOLD/C.N_FOLDS/C.SEED."""
    train_ids, val_ids, test_ids = get_split(
        fold=getattr(C, "FOLD", 0),
        n_folds=getattr(C, "N_FOLDS", 5),
        seed=getattr(C, "SEED", 42),
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}[split]


class DRIVEPatchDataset(Dataset):
    def __init__(self, split="train", aug_level=None):
        assert split in ("train", "val", "test")
        self.split   = split
        self.augment = get_train_transform(aug_level) if split == "train" else None
        self.patches = []
        self.rng     = np.random.default_rng(C.SEED)
        self._load()
        logger.info(f"[DRIVEPatchDataset:{split}] {len(self.patches)} patches "
                    f"(fold={C.FOLD}, seed={C.SEED})")

    def _load(self):
        ids = _get_active_ids(self.split)
        for img_id in ids:
            stem  = f"{img_id:02d}_training"
            img_p = C.IMG_DIR  / f"{stem}.npy"
            msk_p = C.MASK_DIR / f"{stem}.npy"
            fov_p = C.FOV_DIR  / f"{stem}.npy"
            if not img_p.exists():
                logger.warning(f"  Missing: {img_p}")
                continue
            img  = np.load(img_p)
            mask = np.load(msk_p).astype(np.uint8)
            fov  = np.load(fov_p).astype(np.uint8)
            ps   = C.PATCH_SIZE
            if self.split == "train":
                self.patches.extend(
                    balanced_sample_patches(img, mask, fov, ps,
                                            C.PATCHES_PER_IMG, self.rng))
            else:
                stride = C.VAL_STRIDE if self.split == "val" else C.TEST_STRIDE
                self.patches.extend(
                    extract_patches(img, mask, fov, ps, stride))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p   = self.patches[idx]
        img = p["img"].astype(np.float32)
        msk = p["mask"].astype(np.float32)
        fov = p["fov"].astype(np.float32)

        if self.augment is not None:
            aug = self.augment(image=img,
                               mask=msk.astype(np.uint8),
                               fov=fov.astype(np.uint8))
            img = aug["image"].astype(np.float32)
            msk = aug["mask"].astype(np.float32)
            fov = aug["fov"].astype(np.float32)

        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(msk).unsqueeze(0),
                torch.from_numpy(fov).unsqueeze(0))


class DRIVEFullImageDataset(Dataset):
    def __init__(self, split="test"):
        self.items = []
        ids = _get_active_ids(split)
        for img_id in ids:
            stem  = f"{img_id:02d}_training"
            img_p = C.IMG_DIR  / f"{stem}.npy"
            msk_p = C.MASK_DIR / f"{stem}.npy"
            fov_p = C.FOV_DIR  / f"{stem}.npy"
            if img_p.exists():
                self.items.append({"stem": stem,
                                   "img":  img_p,
                                   "mask": msk_p,
                                   "fov":  fov_p})
        logger.info(f"[DRIVEFullImageDataset:{split}] {len(self.items)} images "
                    f"(fold={C.FOLD}, seed={C.SEED})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it  = self.items[idx]
        img = np.load(it["img"])
        msk = np.load(it["mask"]).astype(np.float32)
        fov = np.load(it["fov"]).astype(np.float32)
        return (torch.from_numpy(img).unsqueeze(0).float(),
                torch.from_numpy(msk).unsqueeze(0),
                torch.from_numpy(fov).unsqueeze(0),
                it["stem"])
