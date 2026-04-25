"""
optimizers.py — Optimizer and scheduler factory.

Supported optimizers:
  adamw   — AdamW  (default, best for transformers/attention blocks)
  adam    — Adam   (no weight decay decoupling)
  sgd     — SGD with Nesterov momentum (strong baseline)
  rmsprop — RMSprop (good for noisy gradients)

Usage:
    from optimizers import build_optimizer, build_scheduler
    opt  = build_optimizer(model, name="sgd", lr=1e-2, wd=1e-4)
    sched = build_scheduler(opt)
"""

import torch
import torch.nn as nn
import torch.optim as optim

import config as C


def build_optimizer(model: nn.Module,
                    name: str   = C.OPTIMIZER,
                    lr:   float = C.LR,
                    wd:   float = C.WEIGHT_DECAY) -> optim.Optimizer:
    """
    Build an optimizer for the model.

    Args:
        model : PyTorch model
        name  : "adamw" | "adam" | "sgd" | "rmsprop"
        lr    : learning rate
        wd    : weight decay (L2 regularisation)
    """
    params = model.parameters()
    name   = name.lower()

    if name == "adamw":
        # AdamW decouples weight decay from gradient update — best for
        # models with attention/BN blocks.
        return optim.AdamW(params, lr=lr, weight_decay=wd,
                           betas=(0.9, 0.999), eps=1e-8)

    elif name == "adam":
        # Standard Adam — weight decay entangled with gradient.
        # Slightly worse regularisation than AdamW for deep nets.
        return optim.Adam(params, lr=lr, weight_decay=wd,
                          betas=(0.9, 0.999), eps=1e-8)

    elif name == "sgd":
        # SGD + Nesterov momentum — classic, can find sharper minima,
        # needs higher LR (typically 1e-2 to 1e-1 for image segmentation).
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd,
                         nesterov=True)

    elif name == "rmsprop":
        # RMSprop — adaptive LR, good for recurrent-like training dynamics.
        # alpha (smoothing constant) = 0.99 for stable gradient estimates.
        return optim.RMSprop(params, lr=lr, weight_decay=wd,
                             alpha=0.99, eps=1e-8, momentum=0.9)

    else:
        raise ValueError(f"Unknown optimizer: {name}. "
                         f"Choose: adamw | adam | sgd | rmsprop")


def build_scheduler(optimizer: optim.Optimizer,
                    factor:   float = C.SCHED_FACTOR,
                    patience: int   = C.SCHED_PATIENCE,
                    min_lr:   float = C.MIN_LR):
    """ReduceLROnPlateau — halves LR when val_loss stops improving."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor,
        patience=patience, min_lr=min_lr)
