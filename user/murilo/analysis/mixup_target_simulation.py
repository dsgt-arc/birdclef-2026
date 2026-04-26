#!/usr/bin/env python3
"""Simulate v5j's training-time MixUp targets and record the value distribution.

Motivated by the 2026-04-26 mixup audit
(`notes/2026-04-26-mixup-implementation-audit.md`) which found that v5j uses
**interpolated** multi-label targets:

    labels = lam * labels1 + (1 - lam) * labels2
    # then theta-cap at 0.8: labels >= 0.8 → 1.0

So during training, most multi-hot positives end up as sub-1 values rather
than the binary 0/1 they came in as. This script quantifies how often the
model is being trained on "fully present" (1.0) vs "partially present"
(sub-1) target signals — which feeds directly into the D1 interpretation:
when v5j produces "honest" diffuse attention on hard val crops, part of
that honesty might be a learned response to its training signal having
been uncertain in the first place.

No model, no audio loading. Just builds multi-hot vectors from train.csv
and train_soundscapes_labels.csv, simulates v5j's per-sample mixup over
many random pairs, and produces a histogram + summary.

Usage:
    cd user/murilo/analysis
    python mixup_target_simulation.py \\
        --data-dir /storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026 \\
        --output-dir data/mixup_sim/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_secondary_labels(s) -> list[str]:
    """train.csv's secondary_labels column is a stringified list, e.g. "['a','b']"."""
    if not isinstance(s, str) or not s.strip() or s.strip() in ("[]", "nan"):
        return []
    try:
        # Safe eval for list-of-strings literal
        s_clean = s.replace("'", '"').strip()
        if not s_clean.startswith("["):
            return []
        return [str(x) for x in pd.io.json.loads(s_clean)] if False else eval(s)  # noqa
    except Exception:
        return []


def build_multi_hot_vectors(
    data_dir: Path,
    species_list: list[str],
) -> np.ndarray:
    """Build a (N, num_classes) array of multi-hot label vectors.

    Includes focal recordings (primary + secondary labels) and labeled
    soundscape segments (all species annotated for that segment).
    """
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    K = len(species_list)
    vectors = []

    # Focal
    train_csv = data_dir / "train.csv"
    if train_csv.exists():
        focal = pd.read_csv(train_csv)
        n_focal = 0
        for _, row in focal.iterrows():
            v = np.zeros(K, dtype=np.float32)
            primary = row["primary_label"]
            if primary in species_to_idx:
                v[species_to_idx[primary]] = 1.0
            secs = _parse_secondary_labels(row.get("secondary_labels", ""))
            for sec in secs:
                if sec in species_to_idx:
                    v[species_to_idx[sec]] = 1.0
            vectors.append(v)
            n_focal += 1
        logger.info(f"Focal: {n_focal} multi-hot vectors")

    # Soundscape segments (one row per segment in train_soundscapes_labels.csv,
    # with `birds` column containing comma- or space-separated species).
    sc_csv = data_dir / "train_soundscapes_labels.csv"
    if sc_csv.exists():
        sc = pd.read_csv(sc_csv)
        # Detect the labels column — different schemas in different competition
        # data dumps. Look for the most likely one.
        label_cols = [c for c in sc.columns if c.lower() in ("birds", "labels", "primary_label")]
        n_sc = 0
        for _, row in sc.iterrows():
            v = np.zeros(K, dtype=np.float32)
            for col in label_cols:
                val = row.get(col, "")
                if isinstance(val, str) and val.strip():
                    # Split on either space, comma, or semicolon
                    for sp in val.replace(",", " ").replace(";", " ").split():
                        if sp in species_to_idx:
                            v[species_to_idx[sp]] = 1.0
            if v.sum() > 0:
                vectors.append(v)
                n_sc += 1
        logger.info(f"Soundscape: {n_sc} multi-hot vectors")

    return np.stack(vectors, axis=0) if vectors else np.zeros((0, K), dtype=np.float32)


def simulate_mixup(
    vectors: np.ndarray,
    n_simulations: int = 100_000,
    mixup_alpha: float = 1.0,
    mixup_theta: float = 0.8,
    seed: int = 42,
) -> dict:
    """Simulate v5j's per-sample mixup: lam~Beta(α,α) max-flipped to [0.5,1],
    interpolated targets, theta cap.

    Records the distribution of non-zero target values across simulated samples.
    Each simulation = (sample i, sample j) pair, mixed.
    """
    rng = np.random.default_rng(seed)
    N, K = vectors.shape
    if N < 2:
        raise ValueError("Need at least 2 vectors for mixup simulation.")

    # Sample pairs and lambdas
    i_idx = rng.integers(0, N, size=n_simulations)
    j_idx = rng.integers(0, N, size=n_simulations)
    lam = rng.beta(mixup_alpha, mixup_alpha, size=n_simulations).astype(np.float32)
    lam = np.maximum(lam, 1.0 - lam)  # max-flip to [0.5, 1]

    # Mix targets: target = lam * y1 + (1 - lam) * y2
    y1 = vectors[i_idx]  # (n_simulations, K)
    y2 = vectors[j_idx]
    mixed = lam[:, None] * y1 + (1.0 - lam[:, None]) * y2

    # Apply theta cap
    mixed_post = np.where(mixed >= mixup_theta, 1.0, mixed)

    # Distribution stats over non-zero values
    nz_pre = mixed[mixed > 0]
    nz_post = mixed_post[mixed_post > 0]

    # How many positives are exactly 1.0 vs sub-1 after the cap?
    n_pre_ones = int((mixed_post == 1.0).sum())
    n_pre_sub1 = int(((mixed_post > 0) & (mixed_post < 1.0)).sum())
    total_pos = n_pre_ones + n_pre_sub1

    # Bins for histogram
    bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    hist_pre, _ = np.histogram(nz_pre, bins=bins)
    hist_post, _ = np.histogram(nz_post, bins=bins)

    # Per-sample summary: how often does a mixed sample have ANY 1.0 positive?
    has_full_pos_post = (mixed_post == 1.0).any(axis=1)
    pct_with_full_pos = float(has_full_pos_post.mean() * 100)

    # Per-sample summary: average number of positives per sample
    n_pos_per_sample = (mixed_post > 0).sum(axis=1).astype(np.float32)
    n_full_pos_per_sample = (mixed_post == 1.0).sum(axis=1).astype(np.float32)

    return {
        "n_simulations": n_simulations,
        "n_classes": K,
        "n_positive_target_values_total": int(total_pos),
        "n_full_positives_post_cap": n_pre_ones,
        "n_sub1_positives_post_cap": n_pre_sub1,
        "frac_full_positives_post_cap": (n_pre_ones / total_pos if total_pos > 0 else float("nan")),
        "pct_samples_with_at_least_one_full_pos_post_cap": pct_with_full_pos,
        "mean_positives_per_sample_post_cap": float(n_pos_per_sample.mean()),
        "mean_full_positives_per_sample_post_cap": float(n_full_pos_per_sample.mean()),
        "hist_bins": bins.tolist(),
        "hist_pre_cap": hist_pre.tolist(),
        "hist_post_cap": hist_post.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Simulate v5j training-target distribution")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026",
    )
    parser.add_argument("--n-simulations", type=int, default=100_000)
    parser.add_argument("--mixup-alpha", type=float, default=1.0, help="v5j uses 1.0")
    parser.add_argument(
        "--mixup-theta",
        type=float,
        default=0.8,
        help="v5j theta cap: labels >= theta snap to 1.0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mixup_sim/",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build species list from taxonomy.csv
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    species_list = sorted(taxonomy["primary_label"].astype(str).unique().tolist())
    logger.info(f"Species list: {len(species_list)} species")

    logger.info("Building multi-hot label vectors from train.csv + soundscape labels...")
    vectors = build_multi_hot_vectors(data_dir, species_list)
    logger.info(f"Vector shape: {vectors.shape}")

    if len(vectors) < 2:
        raise SystemExit("Not enough multi-hot vectors built. Check data paths.")

    # Diagnostic — base distribution before any mixup
    n_positives_per_sample = (vectors > 0).sum(axis=1)
    print(
        f"\nBase (pre-mixup) label distribution:\n"
        f"  Total samples: {len(vectors)}\n"
        f"  Mean positives per sample: {n_positives_per_sample.mean():.3f}\n"
        f"  Median positives per sample: {int(np.median(n_positives_per_sample))}\n"
        f"  Max positives per sample: {int(n_positives_per_sample.max())}\n"
    )

    print(
        f"Simulating {args.n_simulations} mixup events with "
        f"alpha={args.mixup_alpha}, theta={args.mixup_theta}..."
    )
    result = simulate_mixup(
        vectors,
        n_simulations=args.n_simulations,
        mixup_alpha=args.mixup_alpha,
        mixup_theta=args.mixup_theta,
        seed=args.seed,
    )

    # Pretty print
    print("\n=== Post-mixup target value distribution ===")
    print(
        f"  Total positive target values across simulations: "
        f"{result['n_positive_target_values_total']:,}"
    )
    print(
        f"  Full positives (= 1.0) after theta cap: "
        f"{result['n_full_positives_post_cap']:,} "
        f"({100 * result['frac_full_positives_post_cap']:.1f}% of positives)"
    )
    print(
        f"  Sub-1 positives (in (0, 1.0)) after theta cap: "
        f"{result['n_sub1_positives_post_cap']:,}"
    )
    print(
        f"  % of mixed samples with ≥ 1 full-positive (=1.0) after cap: "
        f"{result['pct_samples_with_at_least_one_full_pos_post_cap']:.1f}%"
    )
    print(
        f"  Mean positives per mixed sample (post-cap): "
        f"{result['mean_positives_per_sample_post_cap']:.3f}"
    )
    print(
        f"  Mean full-positives per mixed sample (post-cap): "
        f"{result['mean_full_positives_per_sample_post_cap']:.3f}"
    )

    # Histogram
    print("\n=== Histogram of non-zero target values ===")
    print(f"{'Bin':<12} {'Pre-cap':>12} {'Post-cap':>12}")
    bins = result["hist_bins"]
    for i in range(len(bins) - 1):
        bin_label = f"[{bins[i]:.2f}, {bins[i+1]:.2f})"
        print(f"{bin_label:<12} {result['hist_pre_cap'][i]:>12,} {result['hist_post_cap'][i]:>12,}")

    # Save
    import json

    out_json = output_dir / "mixup_target_simulation.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved result to {out_json}")


if __name__ == "__main__":
    main()
