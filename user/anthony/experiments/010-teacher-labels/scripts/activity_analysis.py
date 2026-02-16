#!/usr/bin/env python3
"""Analyze how many clips show bird activity based on max confidence.

Applies sigmoid to raw logits and reports activity counts at various
probability thresholds. Uses polars lazy evaluation to avoid OOM.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

LABEL_DIR = Path("~/scratch/birdclef-2026/outputs/exp010/labels").expanduser()
META_COLS = {"file", "start_time", "end_time", "index"}
THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def analyze_activity(model: str, source: str) -> dict:
    predict_dir = LABEL_DIR / model / source / "parts" / "predict"
    if not predict_dir.exists():
        return {}

    lf = pl.scan_parquet(predict_dir / "*.parquet")
    species_cols = [c for c in lf.collect_schema().names() if c not in META_COLS]

    # Max logit per row via polars (memory-efficient)
    max_logits = (
        lf.select(pl.max_horizontal(species_cols).alias("max_logit"))
        .collect()["max_logit"]
        .to_numpy()
    )

    max_probs = sigmoid(max_logits)
    total = len(max_probs)

    # Activity counts at each threshold
    threshold_counts = {}
    for t in THRESHOLDS:
        active = int((max_probs >= t).sum())
        threshold_counts[str(t)] = {
            "active": active,
            "active_pct": round(100.0 * active / total, 2),
            "inactive": total - active,
        }

    # Percentile distribution
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {f"p{p}": round(float(np.percentile(max_probs, p)), 4) for p in pcts}

    return {
        "total_clips": total,
        "thresholds": threshold_counts,
        "max_prob_percentiles": percentiles,
        "max_prob_mean": round(float(np.mean(max_probs)), 4),
        "max_prob_std": round(float(np.std(max_probs)), 4),
    }


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = {}

    for model in ["birdnet", "perch"]:
        results[model] = {}
        for source in ["train_audio", "train_soundscapes"]:
            source_label = "focal" if "audio" in source else "soundscape"
            print(f"Analyzing {model}/{source_label}...")
            results[model][source_label] = analyze_activity(model, source)

    # Print summary table
    for model in ["birdnet", "perch"]:
        print(f"\n{'='*65}")
        print(f"  {model.upper()}")
        print(f"{'='*65}")

        for source_label in ["focal", "soundscape"]:
            r = results[model][source_label]
            if not r:
                continue
            total = r["total_clips"]
            print(f"\n  {source_label} ({total:,} clips):")
            print(f"  {'Threshold':>10}  {'Active':>10}  {'%':>7}  {'Inactive':>10}  {'%':>7}")
            print(f"  {'-'*52}")
            for t in THRESHOLDS:
                tc = r["thresholds"][str(t)]
                print(
                    f"  {t:>10.2f}  {tc['active']:>10,}  "
                    f"{tc['active_pct']:>6.1f}%  {tc['inactive']:>10,}  "
                    f"{100 - tc['active_pct']:>6.1f}%"
                )

            pct = r["max_prob_percentiles"]
            print(f"\n  Max-prob percentiles:")
            parts = [f"P{k[1:]}={v:.3f}" for k, v in pct.items()]
            print(f"    {' | '.join(parts)}")

    if output_path:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
