"""
run_all.py — Master script to run the full experiment pipeline.

Sequence:
  1. Prepare data         (prepare_data.py)
  2. Train original UNet  (experiment="original")
  3. Evaluate original
  4. Train enhanced UNet  (experiment="enhanced")
  5. Evaluate enhanced
  6. Ablation: spatial-only
  7. Ablation: channel-only
  8. Compare all          (compare_experiments.py)

Usage:
    # Full pipeline:
    python run_all.py

    # Single experiment:
    python run_all.py --exp original
    python run_all.py --exp enhanced
    python run_all.py --exp ablation_spatial
    python run_all.py --exp ablation_channel
    python run_all.py --compare_only
"""

import sys
import argparse
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config as C
from utils import ensure_dirs, get_logger, set_seed

logger = get_logger("run_all", C.LOG_DIR / "run_all.log")


def run_prepare():
    logger.info("=" * 70)
    logger.info("STEP 1 — Data Preparation")
    logger.info("=" * 70)
    from prepare_data import main as prep_main
    # Import at top-level to use project root
    import importlib, sys
    spec = importlib.util.spec_from_file_location(
        "prepare_data", Path(__file__).parent / "prepare_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def run_train(experiment: str):
    logger.info("=" * 70)
    logger.info(f"TRAINING — {experiment}")
    logger.info("=" * 70)
    from train import train
    best_ckpt = train(experiment)
    logger.info(f"Best checkpoint: {best_ckpt}")
    return best_ckpt


def run_evaluate(experiment: str, tta: bool = False):
    logger.info("=" * 70)
    logger.info(f"EVALUATION — {experiment}")
    logger.info("=" * 70)
    from evaluate import evaluate
    summary = evaluate(experiment, use_tta=tta)
    return summary


def run_compare():
    logger.info("=" * 70)
    logger.info("COMPARISON — all experiments")
    logger.info("=" * 70)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_experiments",
        Path(__file__).parent / "compare_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    parser = argparse.ArgumentParser(
        description="DRIVE retinal vessel segmentation pipeline")
    parser.add_argument("--exp", default="all",
                        choices=["all", "original", "enhanced",
                                 "ablation_spatial", "ablation_channel"])
    parser.add_argument("--skip_prep", action="store_true",
                        help="Skip data preparation (already done)")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--compare_only", action="store_true")
    parser.add_argument("--tta", action="store_true",
                        help="Use test-time augmentation during evaluation")
    args = parser.parse_args()

    set_seed(C.SEED)
    ensure_dirs(C.LOG_DIR)

    # ── Compare only ─────────────────────────────────────────────────────
    if args.compare_only:
        run_compare()
        return

    # ── Data prep ─────────────────────────────────────────────────────────
    if not args.skip_prep and not args.eval_only:
        run_prepare()

    # ── Experiments ───────────────────────────────────────────────────────
    exps_to_run = (
        ["original", "msca", "deva", "ablation_spatial", "ablation_channel"]
        if args.exp == "all" else [args.exp]
    )

    for exp in exps_to_run:
        if not args.eval_only:
            run_train(exp)
        run_evaluate(exp, tta=args.tta)

    # ── Compare (only if running all) ─────────────────────────────────────
    if args.exp == "all":
        run_compare()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
