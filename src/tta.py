"""
tta.py — Test-Time Augmentation for retinal vessel segmentation.

Standard TTA protocol for medical image segmentation:
  1. Predict on the original image
  2. Predict on horizontally-flipped image, flip prediction back
  3. Predict on vertically-flipped image, flip prediction back
  4. Predict on H+V flipped image, flip prediction back twice
  5. Average all 4 probability maps → final prediction

This typically improves Dice by +0.003 to +0.005 on DRIVE without retraining.
"""

import numpy as np
import torch

from infer import infer_full_image


@torch.no_grad()
def infer_with_tta(model, image: np.ndarray, fov: np.ndarray,
                   device: torch.device,
                   patch_sz: int = 96, stride: int = 16,
                   variants: list = None) -> np.ndarray:
    """
    Run inference with Test-Time Augmentation.

    Args:
        model     : trained segmentation model
        image     : (H, W) preprocessed input
        fov       : (H, W) FOV mask
        device    : torch device
        patch_sz  : patch size for sliding window
        stride    : stride for sliding window
        variants  : list of augmentations to apply, default 4 flips

    Returns:
        Averaged probability map (H, W) in [0, 1]
    """
    if variants is None:
        variants = ["original", "hflip", "vflip", "both"]

    accumulator = np.zeros_like(image, dtype=np.float64)

    for v in variants:
        # ── Apply augmentation ─────────────────────────────────
        if v == "original":
            img_aug = image
            fov_aug = fov
        elif v == "hflip":
            img_aug = np.fliplr(image).copy()
            fov_aug = np.fliplr(fov).copy()
        elif v == "vflip":
            img_aug = np.flipud(image).copy()
            fov_aug = np.flipud(fov).copy()
        elif v == "both":
            img_aug = np.flipud(np.fliplr(image)).copy()
            fov_aug = np.flipud(np.fliplr(fov)).copy()
        else:
            raise ValueError(f"Unknown TTA variant: {v}")

        # ── Run inference ─────────────────────────────────────
        prob = infer_full_image(model, img_aug, fov_aug, device,
                                patch_sz=patch_sz, stride=stride)

        # ── Reverse augmentation on prediction ────────────────
        if v == "hflip":
            prob = np.fliplr(prob)
        elif v == "vflip":
            prob = np.flipud(prob)
        elif v == "both":
            prob = np.flipud(np.fliplr(prob))
        # original needs no reversal

        accumulator += prob

    # ── Average across variants ─────────────────────────────────
    averaged = accumulator / len(variants)
    return averaged.astype(np.float32)
