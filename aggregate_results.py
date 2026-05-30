"""
aggregate_results.py — Aggregate results across folds and seeds.

Walks outputs/metrics/<experiment>/fold<F>_seed<S>/summary.json,
computes mean ± std per (experiment, metric) across all (fold, seed) pairs,
and produces:
  - outputs/metrics/aggregated_results.json
  - outputs/metrics/aggregated_results.md  (paper-ready table)
  - outputs/metrics/ttests.json (paired t-tests Original vs MSCA vs DEVA)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import json
from collections import defaultdict
import numpy as np

import config as C


METRICS = [
    ("dice_mean",        "Dice"),
    ("iou_mean",         "IoU"),
    ("sensitivity_mean", "Sensitivity"),
    ("specificity_mean", "Specificity"),
    ("precision_mean",   "Precision"),
    ("accuracy_mean",    "Accuracy"),
    ("auc_roc_global",   "AUC-ROC"),
    ("auc_pr_global",    "AUC-PR"),
]


def collect_runs(experiment):
    runs = []
    base = C.METRIC_DIR / experiment
    if not base.exists():
        return runs
    for run_dir in base.glob("fold*_seed*"):
        # Skip hparam runs (they have _hparam suffix)
        if "hparam" in run_dir.name:
            continue
        summary = run_dir / "summary.json"
        if summary.exists():
            with open(summary) as f:
                data = json.load(f)
            data["_run_dir"] = str(run_dir)
            runs.append(data)
    return runs


def aggregate_one(experiment):
    runs = collect_runs(experiment)
    if not runs:
        return {"experiment": experiment, "n_runs": 0}
    agg = {"experiment": experiment, "n_runs": len(runs)}
    for key, label in METRICS:
        vals = [r.get(key) for r in runs if r.get(key) is not None]
        if vals:
            v = np.array(vals, dtype=float)
            agg[key] = {
                "mean":   float(v.mean()),
                "std":    float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min":    float(v.min()),
                "max":    float(v.max()),
                "median": float(np.median(v)),
                "n":      len(v),
                "values": v.tolist(),
            }
    return agg


def paired_ttest(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return {"t": None, "p": None, "n": len(a),
                "note": "insufficient or mismatched data"}
    d = a - b
    mean_d = float(d.mean())
    std_d  = float(d.std(ddof=1))
    n      = len(d)
    if std_d == 0:
        return {"t": float("inf"), "p": 0.0, "n": n,
                "mean_diff": mean_d, "std_diff": std_d}
    t = mean_d / (std_d / np.sqrt(n))
    try:
        from scipy.stats import t as tdist
        p = 2 * (1 - tdist.cdf(abs(t), df=n - 1))
    except ImportError:
        p = None
    return {"t": float(t), "p": float(p) if p is not None else None,
            "mean_diff": mean_d, "std_diff": std_d, "n": n}


def write_markdown(aggregations, out_path):
    lines = ["# Aggregated DRIVE Results", ""]
    n_total = sum(a.get("n_runs", 0) for a in aggregations.values())
    lines.append(f"Total runs: **{n_total}** "
                 f"(across {C.N_FOLDS}-fold CV × {len(C.SEEDS)} seeds)")
    lines.append("")

    experiments = list(aggregations.keys())
    lines.append("| Metric | " + " | ".join(experiments) + " |")
    lines.append("|" + "---|" * (len(experiments) + 1))

    for key, label in METRICS:
        row = [label]
        for exp in experiments:
            a = aggregations[exp]
            if key in a:
                m, s = a[key]["mean"], a[key]["std"]
                row.append(f"{m:.4f} ± {s:.4f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*",
                    default=["original", "msca", "deva",
                             "ablation_spatial", "ablation_channel"])
    args = ap.parse_args()

    aggregations = {}
    for exp in args.experiments:
        a = aggregate_one(exp)
        if a.get("n_runs", 0) > 0:
            aggregations[exp] = a

    out_dir = C.METRIC_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregations, f, indent=2)
    write_markdown(aggregations, out_dir / "aggregated_results.md")

    # Pairwise t-tests on Dice
    if {"original", "msca", "deva"}.issubset(aggregations.keys()):
        comparisons = [("msca", "original"), ("deva", "original"), ("deva", "msca")]
        ttests = {}
        for a, b in comparisons:
            v_a = aggregations[a].get("dice_mean", {}).get("values", [])
            v_b = aggregations[b].get("dice_mean", {}).get("values", [])
            ttests[f"{a}_vs_{b}"] = paired_ttest(v_a, v_b)
        with open(out_dir / "ttests.json", "w") as f:
            json.dump(ttests, f, indent=2)

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    for exp, a in aggregations.items():
        print(f"\n{exp}  ({a['n_runs']} runs)")
        for key, label in METRICS:
            if key in a:
                m, s = a[key]["mean"], a[key]["std"]
                print(f"  {label:15s}: {m:.4f} ± {s:.4f}")
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
