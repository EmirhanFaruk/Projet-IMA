"""
visualize_loss_landscape.py — 2D loss landscape per (optimizer, architecture).

Method: filter-normalised perturbation (Li et al., 2018, "Visualizing the Loss
Landscape of Neural Nets"). Two random directions d1, d2 in parameter space are
generated and normalised filter-wise to match the scale of the learned weights.
The loss is then evaluated on a 2D grid:
    L(α, β) = loss(θ* + α·d1 + β·d2)
where θ* is the trained model.

This produces a 2D contour plot showing the local loss surface around the
trained minimum, characterising the geometry each optimizer settles into.

Output: outputs/visualizations/loss_landscape/<exp>_<optimizer>.png
        outputs/visualizations/loss_landscape/comparison_grid.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from dataset import DRIVEPatchDataset
from model import build_model
from losses import build_loss
from utils import get_logger, load_checkpoint, ensure_dirs

logger = get_logger("loss_landscape")


def get_random_direction(model, rng):
    """Returns list of tensors with same shape as each parameter."""
    return [torch.randn_like(p) for p in model.parameters()]


def filter_normalise(direction, weights):
    """Filter-normalise: scale each filter of the direction to match the
    norm of the corresponding filter in the weights."""
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            d.zero_()
            continue
        for i in range(d.size(0)):
            w_norm = w[i].norm()
            d_norm = d[i].norm() + 1e-10
            d[i].mul_(w_norm / d_norm)


def perturb_model(model, base_weights, d1, d2, alpha, beta):
    """Set model parameters to base + alpha·d1 + beta·d2."""
    for p, w, a, b in zip(model.parameters(), base_weights, d1, d2):
        p.data.copy_(w + alpha * a + beta * b)


@torch.no_grad()
def eval_loss(model, loader, criterion, device, max_batches=10):
    """Average loss over up to max_batches batches (for speed)."""
    model.eval()
    total = 0.0
    n     = 0
    for i, (imgs, masks, _) in enumerate(loader):
        if i >= max_batches:
            break
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        total += criterion(logits, masks).item()
        n += 1
    return total / max(1, n)


def compute_landscape(ckpt_path, experiment, grid_size=11, alpha_range=0.5,
                      max_batches=8, seed=42):
    """Build a grid_size × grid_size loss surface around the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + checkpoint
    model = build_model(experiment).to(device)
    load_checkpoint(str(ckpt_path), model)
    model.eval()

    # Save trained weights (theta*)
    base_weights = [p.data.clone() for p in model.parameters()]

    # Generate two filter-normalised directions
    rng = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    d1 = get_random_direction(model, rng)
    d2 = get_random_direction(model, rng)
    filter_normalise(d1, base_weights)
    filter_normalise(d2, base_weights)

    # Loader with a few batches for loss evaluation
    val_ds = DRIVEPatchDataset("val")
    loader = torch.utils.data.DataLoader(val_ds, batch_size=C.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=2, pin_memory=True)
    criterion = build_loss(C.LOSS_FN)

    # Compute loss on grid
    alphas = np.linspace(-alpha_range, alpha_range, grid_size)
    betas  = np.linspace(-alpha_range, alpha_range, grid_size)
    L = np.zeros((grid_size, grid_size))

    logger.info(f"Computing {grid_size}x{grid_size} landscape "
                f"(α±{alpha_range}, max_batches={max_batches})...")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            perturb_model(model, base_weights, d1, d2, a, b)
            L[i, j] = eval_loss(model, loader, criterion, device, max_batches)
        logger.info(f"  Row {i+1}/{grid_size} done (a={a:.2f})")

    # Restore base weights
    for p, w in zip(model.parameters(), base_weights):
        p.data.copy_(w)

    return alphas, betas, L


def plot_landscape(alphas, betas, L, exp, optimizer, out_path):
    fig = plt.figure(figsize=(14, 6))

    # 2D contour
    ax1 = fig.add_subplot(1, 2, 1)
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    cs   = ax1.contourf(A, B, L, levels=20, cmap="viridis")
    plt.colorbar(cs, ax=ax1, label="Loss")
    ax1.contour(A, B, L, levels=20, colors="white", linewidths=0.5, alpha=0.4)
    ax1.scatter([0], [0], color="red", marker="*", s=200,
                edgecolor="white", lw=1.5, zorder=10, label="Trained θ*")
    ax1.set_xlabel("α (direction 1)")
    ax1.set_ylabel("β (direction 2)")
    ax1.set_title(f"2D loss landscape\n{exp} — optimizer: {optimizer}")
    ax1.legend()

    # 3D surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(A, B, L, cmap="viridis", alpha=0.85,
                      linewidth=0, antialiased=True)
    ax2.set_xlabel("α")
    ax2.set_ylabel("β")
    ax2.set_zlabel("Loss")
    ax2.set_title("3D view")
    ax2.view_init(elev=30, azim=45)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="original",
                   choices=["original", "msca", "deva"])
    p.add_argument("--optimizer", default=None,
                   help="Just a label for the output filename")
    p.add_argument("--ckpt",      default=None,
                   help="Path to .pth (default: best of fold0_seed42)")
    p.add_argument("--fold",      type=int, default=0)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--grid_size", type=int, default=11)
    p.add_argument("--alpha",     type=float, default=0.5)
    p.add_argument("--max_batches", type=int, default=8)
    args = p.parse_args()

    C.FOLD = args.fold
    C.SEED = args.seed

    out_dir = C.VIZ_DIR / "loss_landscape"
    ensure_dirs(out_dir)

    if args.ckpt is None:
        sub = f"fold{args.fold}_seed{args.seed}"
        ckpt_path = C.CKPT_DIR / args.experiment / sub / "best_model.pth"
    else:
        ckpt_path = Path(args.ckpt)

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    optimizer_label = args.optimizer or C.OPTIMIZER
    out_path = out_dir / f"{args.experiment}_{optimizer_label}_landscape.png"

    alphas, betas, L = compute_landscape(
        ckpt_path, args.experiment,
        grid_size=args.grid_size,
        alpha_range=args.alpha,
        max_batches=args.max_batches,
    )
    plot_landscape(alphas, betas, L, args.experiment, optimizer_label, out_path)

    # Save raw data so we can replot later or build the comparison grid
    np.savez(out_dir / f"{args.experiment}_{optimizer_label}_data.npz",
             alphas=alphas, betas=betas, loss=L)


if __name__ == "__main__":
    main()
