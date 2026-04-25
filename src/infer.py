"""
infer.py — Full-image inference with overlapping-patch reconstruction.

Strategy (from recipe §19):
  1. Load full pre-processed image
  2. Extract overlapping patches (stride < patch_size for averaging)
  3. Predict logits for each patch
  4. Reconstruct probability map by summing predictions and counts
  5. Divide by count map (overlap averaging)
  6. Apply sigmoid → probability map in [0,1]
  7. Zero-out outside FOV
"""

import numpy as np
import torch
import torch.nn as nn

import config as C


# ──────────────────────────────────────────────────────────────────────────────
# Core: patch-based reconstruction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_full_image(model:    nn.Module,
                     image:    np.ndarray,   # (H, W) float32
                     fov:      np.ndarray,   # (H, W) uint8
                     device:   torch.device,
                     patch_sz: int   = C.PATCH_SIZE,
                     stride:   int   = C.TEST_STRIDE,
                     batch_sz: int   = 64) -> np.ndarray:
    """
    Returns probability map (H, W) float32 in [0, 1].
    Pixels outside FOV are set to 0.
    """
    model.eval()
    H, W   = image.shape
    half   = patch_sz // 2
    logit_sum = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)

    # ── collect patch centre coordinates ─────────────────────────────────
    centres = []
    for r in range(half, H - half + 1, stride):
        for c in range(half, W - half + 1, stride):
            r_  = min(r, H - half)
            c_  = min(c, W - half)
            centres.append((r_, c_))

    # ── process in mini-batches ───────────────────────────────────────────
    patches_np = []
    pos_list   = []
    for r, c in centres:
        r0, r1 = r - half, r + half
        c0, c1 = c - half, c + half
        patch = image[r0:r1, c0:c1]
        if patch.shape != (patch_sz, patch_sz):
            continue
        patches_np.append(patch)
        pos_list.append((r0, r1, c0, c1))

        if len(patches_np) == batch_sz:
            _infer_batch(model, device, patches_np, pos_list,
                         logit_sum, count_map)
            patches_np, pos_list = [], []

    if patches_np:
        _infer_batch(model, device, patches_np, pos_list,
                     logit_sum, count_map)

    # ── reconstruct probability map ───────────────────────────────────────
    safe_count = np.where(count_map == 0, 1, count_map)
    avg_logits = logit_sum / safe_count
    prob_map   = _sigmoid_np(avg_logits).astype(np.float32)

    # ── zero outside FOV ─────────────────────────────────────────────────
    prob_map[fov == 0] = 0.0
    return prob_map


def _infer_batch(model, device, patches_np, pos_list, logit_sum, count_map):
    x = np.stack(patches_np, axis=0)[:, np.newaxis, :, :]  # (B,1,H,W)
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    logits = model(x).cpu().numpy()[:, 0, :, :]             # (B,H,W)

    for i, (r0, r1, c0, c1) in enumerate(pos_list):
        logit_sum[r0:r1, c0:c1] += logits[i]
        count_map[r0:r1, c0:c1] += 1.0


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


# ──────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA) wrapper  (optional)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_tta(model:   nn.Module,
              image:   np.ndarray,
              fov:     np.ndarray,
              device:  torch.device,
              patch_sz: int = C.PATCH_SIZE,
              stride:   int = C.TEST_STRIDE) -> np.ndarray:
    """
    TTA: average predictions from 4 augmentations
    (original, h-flip, v-flip, both flips).
    """
    transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(np.flipud(x)),
    ]
    inverses = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(np.flipud(x)),
    ]

    prob_sum = np.zeros_like(image, dtype=np.float64)
    for tfm, inv in zip(transforms, inverses):
        img_t  = tfm(image).copy()
        fov_t  = tfm(fov).copy()
        prob   = infer_full_image(model, img_t, fov_t, device,
                                  patch_sz, stride)
        prob_sum += inv(prob)

    return (prob_sum / len(transforms)).astype(np.float32)
