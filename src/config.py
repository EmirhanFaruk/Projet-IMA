"""
config.py — Central configuration for DRIVE retinal vessel segmentation.
All hyperparameters, paths, and experiment settings live here.
"""

from pathlib import Path

# ─────────────────────────── Paths ───────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT / "data/raw/DRIVE"
DATA_PROC     = ROOT / "data/processed"
IMG_DIR       = DATA_PROC / "images"
MASK_DIR      = DATA_PROC / "masks"
FOV_DIR       = DATA_PROC / "fov"
SPLIT_DIR     = DATA_PROC / "splits"
CKPT_DIR      = ROOT / "checkpoints"
LOG_DIR       = ROOT / "logs"
OUT_DIR       = ROOT / "outputs"
PRED_DIR      = OUT_DIR / "predictions"
METRIC_DIR    = OUT_DIR / "metrics"
VIZ_DIR       = OUT_DIR / "visualizations"
HPARAM_DIR    = OUT_DIR / "hparam_search"

# ─────────────────────────── Reproducibility ─────────────────────────────────
SEED = 42

# ─────────────────────────── Data / Patch ────────────────────────────────────
IMG_H, IMG_W    = 584, 565
PATCH_SIZE      = 96
TRAIN_STRIDE    = 32
VAL_STRIDE      = 32
TEST_STRIDE     = 16
MIN_VESSEL_FRAC = 0.01
PATCHES_PER_IMG = 600

# ── DRIVE IDs ─────────────────────────────────────────────────────────────────
# DRIVE training images numbered 21-40 (test images 01-20 have no public GT)
# Split: 12 train | 4 val | 4 test
TRAIN_IDS = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
VAL_IDS   = [33, 34, 35, 36]
TEST_IDS  = [37, 38, 39, 40]

# ─────────────────────────── Training ────────────────────────────────────────
EPOCHS               = 150
BATCH_SIZE           = 64
LR                   = 1e-4
WEIGHT_DECAY         = 1e-4
GRAD_CLIP            = 1.0
NUM_WORKERS          = 4
PIN_MEMORY           = True
OPTIMIZER            = "adam"       # adamw | adam | sgd | rmsprop
LOSS_FN              = "bce_dice"    # bce_dice | focal_dice

# ─────────────────────────── Model ───────────────────────────────────────────
IN_CHANNELS     = 1
OUT_CHANNELS    = 1
BASE_FILTERS    = 64
DROPOUT         = 0.2
REDUCTION_RATIO = 8      # SE channel reduction (MSCA + DEVA)

# ─────────────────────────── Loss ────────────────────────────────────────────
BCE_WEIGHT  = 0.5
DICE_WEIGHT = 0.5
POS_WEIGHT  = 2.5

# ─────────────────────────── Scheduler ───────────────────────────────────────
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 5
MIN_LR         = 1e-6

# ─────────────────────────── Early Stopping ──────────────────────────────────
EARLY_STOP_PATIENCE = 25

# ─────────────────────────── Threshold ───────────────────────────────────────
THRESHOLD   = 0.5
THRESH_LO   = 0.30
THRESH_HI   = 0.90
THRESH_STEP = 0.01

# ─────────────────────────── Preprocessing flags ─────────────────────────────
USE_CLAHE         = True
CLAHE_CLIP_LIMIT  = 4.0
CLAHE_TILE_GRID   = (8, 8)
CHANNEL           = "green"     # "green" | "gray" | "all3"

# ─────────────────────────── Augmentation level ──────────────────────────────
AUG_LEVEL = "light"   # "none" | "light" | "medium" | "heavy"

# ─────────────────────────── Experiment Modes ────────────────────────────────
# "original"         → standard UNet
# "msca"             → UNet + MSCA skip-calibration blocks
# "deva"             → UNet + DEVA novel blocks (new method)
# "ablation_spatial" → MSCA spatial only
# "ablation_channel" → MSCA channel only
EXPERIMENT = "original"

# ─────────────────────────── Hyperparameter search ───────────────────────────
HPARAM_QUICK_EPOCHS = 25   # epochs per config during hparam search
HPARAM_PATIENCE     = 12    # early stop patience during hparam search
