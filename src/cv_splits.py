"""
cv_splits.py — Reproducible K-fold cross-validation splits for DRIVE.

Each fold uses a different held-out test subset (4 images of 20).
Remaining 16 images are split into train (~13) + val (3) deterministically.

Saved as JSON to data/processed/splits/cv_splits_n<N>_seed<S>.json so subsequent
runs use the same partitions.
"""

import json
import numpy as np
from pathlib import Path

import config as C


def generate_fold_splits(all_ids=None, n_folds=5, val_frac=0.20, seed=42):
    """Generate deterministic K-fold splits."""
    if all_ids is None:
        all_ids = list(C.ALL_IDS)
    n = len(all_ids)
    rng = np.random.default_rng(seed)
    shuffled = list(all_ids)
    rng.shuffle(shuffled)

    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        test_start = f * fold_size
        test_end   = (f + 1) * fold_size if f < n_folds - 1 else n
        test_ids   = sorted(shuffled[test_start:test_end])
        train_val  = [x for x in shuffled if x not in test_ids]

        n_val      = max(1, int(round(len(train_val) * val_frac)))
        val_ids    = sorted(train_val[:n_val])
        train_ids  = sorted(train_val[n_val:])

        folds.append({
            "fold":     f,
            "train":    train_ids,
            "val":      val_ids,
            "test":     test_ids,
            "seed":     seed,
            "n_folds":  n_folds,
        })
    return folds


def get_fold_split(fold=None, n_folds=None, seed=None):
    """Return (train, val, test) IDs for a specific fold (cached on disk)."""
    fold     = fold     if fold     is not None else C.FOLD
    n_folds  = n_folds  if n_folds  is not None else C.N_FOLDS
    seed     = seed     if seed     is not None else C.SEED

    cache_path = C.SPLIT_DIR / f"cv_splits_n{n_folds}_seed{seed}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            folds = json.load(f)
    else:
        folds = generate_fold_splits(C.ALL_IDS, n_folds=n_folds, seed=seed,
                                      val_frac=C.VAL_FRAC_OF_TRAIN)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(folds, f, indent=2)

    chosen = folds[fold]
    return chosen["train"], chosen["val"], chosen["test"]


def get_split(fold=None, n_folds=None, seed=None):
    """Returns (train, val, test) IDs respecting USE_CROSS_VAL flag."""
    if C.USE_CROSS_VAL:
        return get_fold_split(fold, n_folds, seed)
    return (C.TRAIN_IDS_FIXED, C.VAL_IDS_FIXED, C.TEST_IDS_FIXED)


if __name__ == "__main__":
    folds = generate_fold_splits(seed=42)
    print(f"Generated {len(folds)} folds (seed=42):")
    for f in folds:
        print(f"  Fold {f['fold']}: train={f['train']}")
        print(f"           val={f['val']}, test={f['test']}")
