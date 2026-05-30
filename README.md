# DRIVE Retinal Vessel Segmentation — Final Research Code

Complete error-free implementation of the U-Net family on DRIVE with three
architectures, comprehensive visualisations, and a rigorous research protocol.

## Quick start

```bash
# 1) Install deps (one time)
pip install -r requirements.txt

# 2) Place DRIVE data (Kaggle) at:
#    data/raw/DRIVE/training/{images,mask,1st_manual}/
#    data/raw/DRIVE/test/{images,mask}/                 (no GT, not used for eval)

# 3) Preprocess data (≈1 min) — produces .npy files + visualisations
python prepare_data.py

# 4) Visualise augmentation pipelines on a sample patch
python visualize_augmentation.py

# 5) Sanity-check single run (≈20 min on RTX 4060)
python src/train.py    --experiment original --fold 0 --seed 42 --epochs 30
python src/evaluate.py --experiment original --fold 0 --seed 42 --tta
```

## Full research run (≈83 GPU-hours)

```bash
# Main 3 architectures × 5 folds × 3 seeds = 45 runs
python run_research.py --stage main --epochs 200

# Ablations × 5 folds × 3 seeds = 30 runs
python run_research.py --stage ablation --epochs 200

# Hyperparameter search across 7 axes (full 200 epochs each)
python hparam_search.py

# Aggregate everything into a paper-ready table + t-tests
python aggregate_results.py

# Cross-experiment comparison plot
python compare_experiments.py

# Loss landscape per (arch, optimizer)
python visualize_loss_landscape.py --experiment original --optimizer adam
python visualize_loss_landscape.py --experiment msca     --optimizer adam
python visualize_loss_landscape.py --experiment deva     --optimizer adam
```

To launch the long run safely overnight:

```bash
nohup python run_research.py --stage all --epochs 200 > research_run.log 2>&1 &
```

## Output structure after a full run

```
outputs/
├── visualizations/
│   ├── preprocessing/       # step-by-step preprocessing figures
│   ├── augmentation/        # aug level comparison
│   ├── loss_landscape/      # 2D + 3D landscapes per (arch, optimizer)
│   ├── comparison/          # cross-architecture bar charts
│   └── <experiment>/fold<F>_seed<S>/   # per-run figures
│       ├── per_image/       # 5-panel predictions
│       ├── *_training_curves.png
│       ├── *_roc_curve.png
│       ├── *_pr_curve.png
│       ├── *_confusion_matrix.png
│       ├── *_threshold_search.png
│       └── *_per_image_dice.png
├── metrics/
│   ├── <experiment>/fold<F>_seed<S>/
│   │   ├── summary.json     # final metrics
│   │   ├── summary.txt
│   │   ├── per_image.csv
│   │   └── original_history.json
│   ├── aggregated_results.json
│   ├── aggregated_results.md   # paper-ready table
│   └── ttests.json
├── hparam_search/
│   ├── <axis>_results.json
│   ├── <axis>_metrics_bar.png
│   ├── <axis>_training_curves.png
│   ├── hparam_summary_heatmap.png
│   └── hparam_radar.png
├── checkpoints/
│   └── <experiment>/fold<F>_seed<S>/
│       ├── best_model.pth
│       ├── last_model.pth
│       └── epoch_XXX.pth (every 10 epochs)
└── predictions/
    └── <experiment>/fold<F>_seed<S>/
        ├── *_prob.npy       # raw probability maps
        └── *_binary.png     # thresholded predictions
```

## Idempotency

Every run is identified by `<experiment>/fold<F>_seed<S>[_<tag>]`. Re-running
the orchestrator skips any combination that has already produced a
`summary.json`, so you can stop and resume freely.

## File map

| File | Purpose |
|------|---------|
| `src/config.py` | All hyperparameters and paths |
| `src/cv_splits.py` | Reproducible K-fold CV splits |
| `src/dataset.py` | Patch & full-image datasets, augmentation pipelines |
| `src/model.py` | UNet, MSCA, DEVA, ablations |
| `src/losses.py` | BCE+Dice, Focal+Dice, Dice-only |
| `src/metrics.py` | Dice/IoU/Sens/Spec/Prec/AUC + threshold search |
| `src/optimizers.py` | Adam/AdamW/SGD/RMSprop + ReduceLROnPlateau |
| `src/infer.py` | Sliding-window full-image inference |
| `src/tta.py` | 4-flip Test-Time Augmentation |
| `src/train.py` | Training loop with AMP + namespaced checkpoints |
| `src/evaluate.py` | Test-set evaluation with all visualisations |
| `src/utils.py` | Logging, checkpointing, history tracking |
| `prepare_data.py` | Preprocessing with step-by-step visualisations |
| `visualize_augmentation.py` | Sample patches through each aug level |
| `visualize_loss_landscape.py` | 2D/3D loss landscape per (arch, optimizer) |
| `hparam_search.py` | 7-axis hparam study with full visualisations |
| `run_research.py` | Master orchestrator (CV × seeds × ablations) |
| `aggregate_results.py` | Mean ± std + paired t-tests across runs |
| `compare_experiments.py` | Cross-architecture comparison plots |
