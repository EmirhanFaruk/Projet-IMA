"""
train.py — Full training loop for DRIVE retinal vessel segmentation.

Features:
  • Mixed-precision (AMP) training
  • ReduceLROnPlateau scheduler
  • Early stopping (monitors val Dice)
  • Full per-epoch metric logging
  • Best + last + periodic checkpoint saving
  • Fold/seed-aware checkpoint directories
  • Env-var overrides for hparam search

Usage:
  python train.py --experiment original --fold 0 --seed 42 --epochs 200
  python train.py --experiment msca     --fold 1 --seed 123 --epochs 200
  OVERRIDE_LR=5e-4 python train.py --experiment original --tag hparam_lr_5e4
"""

# ────────────── Env-var overrides for hparam search ─────────────────
# Apply BEFORE other modules read C.* values.
import os as _os
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import config as C
for _k in dir(C):
    if _k.isupper() and not _k.startswith("_"):
        _ev = _os.environ.get(f"OVERRIDE_{_k}")
        if _ev is not None:
            _orig = getattr(C, _k)
            try:
                if isinstance(_orig, bool):
                    _new = _ev.lower() in ("true", "1", "yes")
                elif isinstance(_orig, int):
                    _new = int(_ev)
                elif isinstance(_orig, float):
                    _new = float(_ev)
                else:
                    _new = _ev
                setattr(C, _k, _new)
                print(f"[config override] {_k} = {_new}")
            except Exception as _e:
                print(f"[config override] FAILED {_k}: {_e}")

from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import DRIVEPatchDataset, DRIVEFullImageDataset
from model import build_model
from losses import build_loss
from metrics import compute_metrics, MetricAccumulator
from infer import infer_full_image
from optimizers import build_optimizer, build_scheduler
from utils import (get_logger, save_checkpoint, load_checkpoint,
                   MetricHistory, Timer, count_parameters, set_seed,
                   ensure_dirs, save_config)

logger = get_logger("train", C.LOG_DIR / "train.log")


# ─────────────────────────────────────────────────────────────────────────
# Validation on full images (patch reconstruction)
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_full_images(model, device, threshold=None):
    if threshold is None:
        threshold = C.THRESHOLD
    ds  = DRIVEFullImageDataset(split="val")
    acc = MetricAccumulator()
    for img_t, msk_t, fov_t, stem in ds:
        image = img_t.squeeze(0).numpy()
        gt    = msk_t.squeeze(0).numpy()
        fov   = fov_t.squeeze(0).numpy()
        prob  = infer_full_image(model, image, fov, device)
        m     = compute_metrics(prob, gt, fov, threshold)
        acc.update(**{k: v for k, v in m.items()
                      if k not in ("tp", "fp", "tn", "fn")})
    return acc.mean()


# ─────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser, scaler, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, fovs in tqdm(loader, desc="  train", leave=False, ncols=80):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimiser.zero_grad()
        with autocast(enabled=C.USE_AMP):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        scaler.step(optimiser)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────
# Main training driver
# ─────────────────────────────────────────────────────────────────────────

def train(experiment: str           = None,
          tag:        Optional[str] = None,
          resume_ckpt: Optional[str] = None):
    if experiment is None:
        experiment = C.EXPERIMENT

    set_seed(C.SEED)

    # Run-specific dirs: <type>/<experiment>/fold<F>_seed<S>[_<tag>]/
    sub = f"fold{C.FOLD}_seed{C.SEED}" + (f"_{tag}" if tag else "")
    ckpt_dir   = C.CKPT_DIR   / experiment / sub
    metric_dir = C.METRIC_DIR / experiment / sub
    viz_dir    = C.VIZ_DIR    / experiment / sub

    ensure_dirs(ckpt_dir, metric_dir, viz_dir, C.LOG_DIR)

    # Skip if already done (idempotency for run_research.py)
    summary_done = metric_dir / "training_complete.json"
    if summary_done.exists() and not resume_ckpt:
        logger.info(f"[skip] {experiment}/{sub} already complete")
        return ckpt_dir / "best_model.pth"

    # Save config snapshot
    save_config(C, metric_dir / "config.json")

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.deterministic = C.DETERMINISTIC_CUDA
        torch.backends.cudnn.benchmark     = C.BENCHMARK_CUDA

    # ── Data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading patch datasets… (fold={C.FOLD}, seed={C.SEED})")
    train_ds  = DRIVEPatchDataset("train")
    val_ds    = DRIVEPatchDataset("val")

    train_ld  = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                           num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY,
                           drop_last=True)
    val_ld    = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                           num_workers=C.NUM_WORKERS, pin_memory=C.PIN_MEMORY)

    logger.info(f"  Train patches : {len(train_ds)}")
    logger.info(f"  Val   patches : {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(experiment).to(device)
    logger.info(f"Model: {experiment}")
    logger.info(f"Params: {count_parameters(model)}")

    # ── Optimiser / Loss / Scheduler ─────────────────────────────────────
    criterion = build_loss(C.LOSS_FN)
    optimiser = build_optimizer(model, name=C.OPTIMIZER, lr=C.LR,
                                wd=C.WEIGHT_DECAY)
    scheduler = build_scheduler(optimiser)
    scaler    = GradScaler(enabled=C.USE_AMP)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_dice   = 0.0
    if resume_ckpt:
        start_epoch, best_dice = load_checkpoint(
            resume_ckpt, model, optimiser, scheduler, scaler)
        start_epoch += 1
        logger.info(f"Resumed from {resume_ckpt} "
                    f"(epoch {start_epoch}, best_dice={best_dice:.4f})")

    # ── History ───────────────────────────────────────────────────────────
    history = MetricHistory(metric_dir, experiment)
    timer   = Timer()
    no_improve = 0

    logger.info("=" * 70)
    logger.info(f"Training {experiment}  fold={C.FOLD}  seed={C.SEED}  "
                f"tag={tag}  epochs={C.EPOCHS}")
    logger.info("=" * 70)

    for epoch in range(start_epoch, C.EPOCHS + 1):
        t0 = timer.lap()
        train_loss = train_one_epoch(model, train_ld, criterion,
                                     optimiser, scaler, device)

        # Patch-level val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, _ in val_ld:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast(enabled=C.USE_AMP):
                    logits = model(imgs)
                    val_loss += criterion(logits, masks).item()
        val_loss /= max(1, len(val_ld))

        # Full-image val metrics
        val_metrics = validate_full_images(model, device)
        val_dice    = val_metrics.get("dice", 0.0)
        val_auc     = val_metrics.get("auc_roc", 0.0)
        val_iou     = val_metrics.get("iou", 0.0)
        val_sens    = val_metrics.get("sensitivity", 0.0)
        val_spec    = val_metrics.get("specificity", 0.0)
        cur_lr      = optimiser.param_groups[0]["lr"]

        scheduler.step(val_loss)

        lap = timer.lap()
        logger.info(
            f"[Ep {epoch:03d}/{C.EPOCHS}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"dice={val_dice:.4f} | iou={val_iou:.4f} | "
            f"AUC={val_auc:.4f} | sens={val_sens:.4f} | "
            f"spec={val_spec:.4f} | lr={cur_lr:.2e} | {lap}"
        )

        history.update(epoch,
                       train_loss=train_loss, val_loss=val_loss,
                       val_dice=val_dice, val_iou=val_iou,
                       val_auc_roc=val_auc, val_sensitivity=val_sens,
                       val_specificity=val_spec, learning_rate=cur_lr)
        history.save()

        # Checkpoints
        state = dict(epoch=epoch,
                     model_state=model.state_dict(),
                     optimizer_state=optimiser.state_dict(),
                     scheduler_state=scheduler.state_dict(),
                     scaler_state=scaler.state_dict(),
                     best_score=best_dice,
                     experiment=experiment,
                     fold=C.FOLD, seed=C.SEED, tag=tag)

        save_checkpoint(state, ckpt_dir / "last_model.pth")
        if val_dice > best_dice:
            best_dice  = val_dice
            no_improve = 0
            save_checkpoint(state, ckpt_dir / "best_model.pth")
            logger.info(f"  ★ New best Dice: {best_dice:.4f} — checkpoint saved")
        else:
            no_improve += 1

        if epoch % C.SAVE_PERIODIC_EVERY == 0:
            save_checkpoint(state, ckpt_dir / f"epoch_{epoch:03d}.pth")

        # Early stopping
        if no_improve >= C.EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(no improvement for {C.EARLY_STOP_PATIENCE} epochs)")
            break

    # Mark training complete (used for skip-already-done detection)
    import json
    with open(summary_done, "w") as f:
        json.dump({"best_dice": best_dice, "epochs_run": epoch,
                   "experiment": experiment,
                   "fold": C.FOLD, "seed": C.SEED, "tag": tag}, f, indent=2)

    logger.info(f"Training complete in {timer.total()}")
    logger.info(f"Best val Dice: {best_dice:.4f}")
    return ckpt_dir / "best_model.pth"


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=C.EXPERIMENT,
                   choices=["original", "msca", "deva",
                             "ablation_spatial", "ablation_channel"])
    p.add_argument("--resume",     default=None)
    p.add_argument("--fold",       type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--tag",        default=None)
    p.add_argument("--optimizer",  default=None,
                   choices=["adamw", "adam", "sgd", "rmsprop"])
    p.add_argument("--loss",       default=None,
                   choices=["bce_dice", "focal_dice", "dice_only"])
    p.add_argument("--aug",        default=None,
                   choices=["none", "light", "medium", "heavy"])
    p.add_argument("--lr",         default=None, type=float)
    args = p.parse_args()

    if args.fold      is not None: C.FOLD       = args.fold
    if args.seed      is not None: C.SEED       = args.seed
    if args.epochs    is not None: C.EPOCHS     = args.epochs
    if args.optimizer:             C.OPTIMIZER  = args.optimizer
    if args.loss:                  C.LOSS_FN    = args.loss
    if args.aug:                   C.AUG_LEVEL  = args.aug
    if args.lr:                    C.LR         = args.lr

    train(experiment=args.experiment, tag=args.tag, resume_ckpt=args.resume)
