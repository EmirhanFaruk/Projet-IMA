"""
run_research.py — Master orchestrator for the full rigorous experiment.

Stages:
  main      — 3 architectures × N_FOLDS × SEEDS  (e.g. 3×5×3 = 45 runs)
  ablation  — 2 ablation variants × N_FOLDS × SEEDS
  hparam    — full-epoch hparam search across 7 axes
  aggregate — collect everything into mean ± std table
  all       — main + ablation + hparam + aggregate

Each individual run is identified by EXPERIMENT × FOLD × SEED, and is
idempotent: re-running skips already-completed configurations.

Usage:
  python run_research.py --stage main --epochs 200
  python run_research.py --stage all  --epochs 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import os
import subprocess
import time
from itertools import product

import config as C
from utils import get_logger, ensure_dirs

logger = get_logger("research", C.LOG_DIR / "research.log")

MAIN_EXPERIMENTS     = ["original", "msca", "deva"]
ABLATION_EXPERIMENTS = ["ablation_spatial", "ablation_channel"]


# ─────────────────────────── Subprocess helpers ──────────────────────────

def _run(cmd, env_overrides=None):
    env = os.environ.copy()
    if env_overrides:
        for k, v in env_overrides.items():
            env[k] = str(v)
    logger.info(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


def _train_eval(experiment, fold, seed, epochs, use_tta=True):
    train_cmd = [
        sys.executable, "src/train.py",
        "--experiment", experiment,
        "--fold",       str(fold),
        "--seed",       str(seed),
        "--epochs",     str(epochs),
    ]
    rc = _run(train_cmd)
    if rc != 0:
        logger.warning(f"Train returned {rc} for {experiment} f{fold} s{seed}")

    eval_cmd = [
        sys.executable, "src/evaluate.py",
        "--experiment", experiment,
        "--fold",       str(fold),
        "--seed",       str(seed),
    ]
    if use_tta:
        eval_cmd.append("--tta")
    rc = _run(eval_cmd)
    if rc != 0:
        logger.warning(f"Eval returned {rc} for {experiment} f{fold} s{seed}")


# ─────────────────────────── Stage runners ───────────────────────────────

def run_stage_main(experiments, folds, seeds, epochs):
    n = len(experiments) * len(folds) * len(seeds)
    logger.info(f"=== MAIN STAGE: {n} runs ===")
    logger.info(f"  Experiments: {experiments}")
    logger.info(f"  Folds:       {folds}")
    logger.info(f"  Seeds:       {seeds}")
    logger.info(f"  Epochs/run:  {epochs}")

    t0 = time.time()
    completed = 0
    for exp, fold, seed in product(experiments, folds, seeds):
        logger.info(f"\n[{completed+1}/{n}] {exp} fold={fold} seed={seed}")
        run_t0 = time.time()
        _train_eval(exp, fold, seed, epochs, use_tta=C.USE_TTA)
        elapsed = (time.time() - run_t0) / 60
        completed += 1
        logger.info(f"  Run finished in {elapsed:.1f} min")
        logger.info(f"  Total elapsed: {(time.time()-t0)/3600:.1f} h")
    logger.info(f"\n=== MAIN STAGE COMPLETE: {completed}/{n} ===")


def run_stage_hparam(axes, epochs, fold, seed):
    logger.info(f"=== HPARAM STAGE: axes={axes or 'ALL'} ===")
    cmd = [sys.executable, "hparam_search.py",
           "--epochs", str(epochs),
           "--fold",   str(fold),
           "--seed",   str(seed)]
    if axes:
        cmd += ["--axes"] + list(axes)
    _run(cmd)


def run_stage_aggregate():
    logger.info("=== AGGREGATE STAGE ===")
    _run([sys.executable, "aggregate_results.py"])


# ─────────────────────────── Entry point ─────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["main", "ablation", "hparam", "aggregate", "all"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--folds",  type=int, nargs="*", default=None)
    p.add_argument("--seeds",  type=int, nargs="*", default=None)
    p.add_argument("--axes",   nargs="*", default=None)
    args = p.parse_args()

    ensure_dirs(C.CKPT_DIR, C.LOG_DIR, C.METRIC_DIR, C.VIZ_DIR)

    epochs = args.epochs or C.EPOCHS
    folds  = args.folds  or list(range(C.N_FOLDS))
    seeds  = args.seeds  or list(C.SEEDS)

    if args.stage in ("main", "all"):
        run_stage_main(MAIN_EXPERIMENTS, folds, seeds, epochs)
    if args.stage in ("ablation", "all"):
        run_stage_main(ABLATION_EXPERIMENTS, folds, seeds, epochs)
    if args.stage in ("hparam", "all"):
        run_stage_hparam(args.axes, C.HPARAM_FULL_EPOCHS,
                         fold=folds[0], seed=seeds[0])
    if args.stage in ("aggregate", "all"):
        run_stage_aggregate()


if __name__ == "__main__":
    main()
