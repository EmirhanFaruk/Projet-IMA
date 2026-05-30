"""
config.py — DRIVE retinal vessel segmentation: rigorous research configuration.

DRIVE has TWO folders:
  • training/ (IDs 21-40, with ground truth in 1st_manual)
  • test/     (IDs 01-20, no public ground truth)

Since metrics require ground truth, we use only the 20 labelled images and
partition them with K-fold cross-validation (standard practice for DRIVE).

Run identification:
  EXPERIMENT × FOLD × SEED → unique results path:
  <type>/<experiment>/fold<F>_seed<S>/...
"""

from pathlib import Path
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════
#  PATHS
# ════════════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════
SEED                = 42
SEEDS               = [42, 123, 2026]
RUN_ID              = datetime.now().strftime("%Y%m%d_%H%M%S")
DETERMINISTIC_CUDA  = True
BENCHMARK_CUDA      = False

# ════════════════════════════════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════════════════════════════════
ALL_IDS             = list(range(21, 41))   # 20 labelled DRIVE images

# Cross-validation
USE_CROSS_VAL       = True
N_FOLDS             = 5
VAL_FRAC_OF_TRAIN   = 0.20
FOLD                = 0

# Fixed split (used only if USE_CROSS_VAL = False)
TRAIN_IDS_FIXED     = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
VAL_IDS_FIXED       = [33, 34, 35, 36]
TEST_IDS_FIXED      = [37, 38, 39, 40]

IMG_H               = 584
IMG_W               = 565

# ════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ════════════════════════════════════════════════════════════════════════
CHANNEL             = "green"
USE_CLAHE           = True
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)
NORMALIZATION       = "0_1"

# ════════════════════════════════════════════════════════════════════════
#  PATCH STRATEGY
# ════════════════════════════════════════════════════════════════════════
PATCH_SIZE          = 96
PATCH_STRATEGY      = "balanced_random"
PATCHES_PER_IMG     = 600
PATCH_VESSEL_FRAC   = 0.50
USE_FOV_MASK        = True
TRAIN_STRIDE        = 32
VAL_STRIDE          = 16
TEST_STRIDE         = 16
MIN_VESSEL_FRAC     = 0.01

# ════════════════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════════════════
EPOCHS              = 200
BATCH_SIZE          = 64
LR                  = 1e-4
WEIGHT_DECAY        = 1e-4
GRAD_CLIP           = 1.0
NUM_WORKERS         = 4
PIN_MEMORY          = True

OPTIMIZER           = "adam"
SGD_MOMENTUM        = 0.9
SGD_NESTEROV        = True
ADAM_BETAS          = (0.9, 0.999)
ADAM_EPS            = 1e-8
RMSPROP_ALPHA       = 0.99

LOSS_FN             = "bce_dice"
BCE_WEIGHT          = 0.5
DICE_WEIGHT         = 0.5
POS_WEIGHT          = 2.5
DICE_SMOOTH         = 1.0
FOCAL_GAMMA         = 2.0
FOCAL_ALPHA         = 0.25

USE_AMP             = True

# ════════════════════════════════════════════════════════════════════════
#  MODEL
# ════════════════════════════════════════════════════════════════════════
IN_CHANNELS         = 1
OUT_CHANNELS        = 1
BASE_FILTERS        = 64
DEPTH               = 4
DROPOUT             = 0.2
USE_BATCHNORM       = True
ACTIVATION          = "relu"
REDUCTION_RATIO     = 8

EXPERIMENT          = "original"

# ════════════════════════════════════════════════════════════════════════
#  SCHEDULER & EARLY STOPPING
# ════════════════════════════════════════════════════════════════════════
SCHED_FACTOR        = 0.5
SCHED_PATIENCE      = 8
MIN_LR              = 1e-6
EARLY_STOP_PATIENCE = 30
EARLY_STOP_METRIC   = "val_dice"
EARLY_STOP_MODE     = "max"

# ════════════════════════════════════════════════════════════════════════
#  CHECKPOINTING
# ════════════════════════════════════════════════════════════════════════
SAVE_BEST_METRIC    = "val_dice"
SAVE_PERIODIC_EVERY = 10
SAVE_LAST_ALWAYS    = True

# ════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ════════════════════════════════════════════════════════════════════════
THRESHOLD           = 0.5
THRESH_LO           = 0.30
THRESH_HI           = 0.90
THRESH_STEP         = 0.01
USE_TTA             = True
TTA_VARIANTS        = ["original", "hflip", "vflip", "both"]

# ════════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ════════════════════════════════════════════════════════════════════════
AUG_LEVEL           = "light"

# ════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER SEARCH
# ════════════════════════════════════════════════════════════════════════
HPARAM_FULL_EPOCHS  = 200
HPARAM_PATIENCE     = 25
HPARAM_FOLDS        = 1
HPARAM_SEEDS        = [42]


# ════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════
def get_run_dir(base, experiment=None, fold=None, seed=None, tag=None):
    """Returns base/<experiment>/fold<F>_seed<S>[_<tag>]/"""
    e = experiment if experiment is not None else EXPERIMENT
    f = fold       if fold       is not None else FOLD
    s = seed       if seed       is not None else SEED
    sub = f"fold{f}_seed{s}"
    if tag:
        sub += f"_{tag}"
    return Path(base) / e / sub


def get_config_dict():
    """Returns all config values as a JSON-serialisable dict."""
    import sys, json as _json
    cfg = {}
    for k in dir(sys.modules[__name__]):
        if k.isupper() and not k.startswith("_"):
            v = getattr(sys.modules[__name__], k)
            try:
                _json.dumps(v, default=str)
                cfg[k] = v if not isinstance(v, Path) else str(v)
            except (TypeError, ValueError):
                pass
    return cfg
