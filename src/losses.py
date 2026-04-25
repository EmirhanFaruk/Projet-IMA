"""
losses.py — Combined loss functions for retinal vessel segmentation.

Primary:   0.5 * BCE + 0.5 * Dice
Optional:  0.5 * Focal + 0.5 * Dice   (for later experiments)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ──────────────────────────────────────────────────────────────────────────────
# Soft Dice Loss
# ──────────────────────────────────────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    """
    Soft (differentiable) Dice loss.
    Input:  logits  (B,1,H,W),  targets (B,1,H,W) in {0,1}
    Output: scalar loss in [0, 1]  (0 = perfect overlap)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        flat_p = probs.view(probs.size(0), -1)
        flat_t = targets.view(targets.size(0), -1)
        inter  = (flat_p * flat_t).sum(dim=1)
        union  = flat_p.sum(dim=1) + flat_t.sum(dim=1)
        dice   = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ──────────────────────────────────────────────────────────────────────────────
# BCE + Dice (primary training loss)
# ──────────────────────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """
    Combined loss = α * BCE + β * SoftDice

    pos_weight  handles class imbalance (vessels ≈ 10-15% of FOV pixels).
    """

    def __init__(self,
                 alpha:      float = C.BCE_WEIGHT,
                 beta:       float = C.DICE_WEIGHT,
                 pos_weight: float = C.POS_WEIGHT):
        super().__init__()
        self.alpha    = alpha
        self.beta     = beta
        pw            = torch.tensor([pos_weight])
        self.bce      = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice     = SoftDiceLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to correct device on first call
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.alpha * bce_loss + self.beta * dice_loss


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss helper
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for class imbalance.
    FL(p) = -(1-p_t)^γ * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets,
                                                   reduction="none")
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        a_t   = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl    = a_t * (1 - p_t) ** self.gamma * bce
        return fl.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Focal + Dice (alternative)
# ──────────────────────────────────────────────────────────────────────────────

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 gamma: float = 2.0, focal_alpha: float = 0.25):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.focal = FocalLoss(gamma, focal_alpha)
        self.dice  = SoftDiceLoss()

    def forward(self, logits, targets):
        return self.alpha * self.focal(logits, targets) + \
               self.beta  * self.dice(logits, targets)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_loss(name: str = "bce_dice") -> nn.Module:
    if name == "bce_dice":
        return BCEDiceLoss()
    elif name == "focal_dice":
        return FocalDiceLoss()
    else:
        raise ValueError(f"Unknown loss: {name}")
