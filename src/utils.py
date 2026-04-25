"""
utils.py — Seeds, checkpoint I/O, logging helpers, directory setup.
"""

import os
import random
import json
import time
import logging
from pathlib import Path

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cudnn (slight speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            ensure_dirs(Path(log_file).parent)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    ensure_dirs(Path(path).parent)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_score", 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Metric logger (CSV-style running history)
# ──────────────────────────────────────────────────────────────────────────────

class MetricHistory:
    """Accumulates per-epoch metrics and saves to JSON + NumPy arrays."""

    def __init__(self, save_dir: str, run_name: str):
        self.save_dir = Path(save_dir)
        self.run_name = run_name
        self.history: dict[str, list] = {}

    def update(self, epoch: int, **kwargs):
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))

    def save(self):
        ensure_dirs(self.save_dir)
        out = self.save_dir / f"{self.run_name}_history.json"
        with open(out, "w") as f:
            json.dump(self.history, f, indent=2)

    def get(self, key: str) -> list:
        return self.history.get(key, [])


# ──────────────────────────────────────────────────────────────────────────────
# Timer
# ──────────────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self._lap = self.start_time

    def lap(self) -> str:
        now = time.time()
        elapsed = now - self._lap
        self._lap = now
        return self._fmt(elapsed)

    def total(self) -> str:
        return self._fmt(time.time() - self.start_time)

    @staticmethod
    def _fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}h {m:02d}m {s:02d}s"


# ──────────────────────────────────────────────────────────────────────────────
# Model parameter counter
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {trainable:,}"


# ──────────────────────────────────────────────────────────────────────────────
# Save experiment config snapshot
# ──────────────────────────────────────────────────────────────────────────────

def save_config(cfg_module, save_path: str):
    """Dump config module attributes to JSON."""
    d = {k: str(v) for k, v in vars(cfg_module).items()
         if not k.startswith("_") and not callable(v)}
    ensure_dirs(Path(save_path).parent)
    with open(save_path, "w") as f:
        json.dump(d, f, indent=2)
