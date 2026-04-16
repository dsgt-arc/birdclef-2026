#!/usr/bin/env python3
"""Smoke test the staged cache pipeline on a tiny synthetic dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from birdclef_107.cache import BASE_DIRNAME, load_training_data_cached
from birdclef_107.perch_features import TOPK_RESIDUAL


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test staged cache sidecar pipeline")
    p.add_argument("--root", type=str, required=True)
    args = p.parse_args()
    root = Path(args.root).expanduser()
    features_dir = root / "features"
    artifacts_dir = root / "artifacts"
    cache_dir = root / "cache"

    run([sys.executable, "scripts/build_cache_bootstrap.py",
         "--features-dir", str(features_dir),
         "--cache-dir", str(cache_dir),
         "--artifacts-dir", str(artifacts_dir)])

    n_units = 1 + len(list((features_dir / "focal").glob("shard-*.parquet"))) + len(list((features_dir / "soundscape_unlabeled").glob("shard-*.parquet")))
    for task_id in range(n_units):
        run([sys.executable, "scripts/build_cache_shard.py",
             "--features-dir", str(features_dir),
             "--cache-dir", str(cache_dir),
             "--task-id", str(task_id),
             "--include-perch-features"])

    run([sys.executable, "scripts/build_cache_finalize.py",
         "--features-dir", str(features_dir),
         "--cache-dir", str(cache_dir),
         "--include-perch-features",
         "--focal-cache-size", "10",
         "--unsup-cache-size", "8"])

    labeled, focal, unsup = load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_dir,
        focal_size=10,
        unsup_size=8,
        include_perch_logits=True,
    )

    assert labeled.perch_mapped is not None and labeled.perch_mapped.shape == (15, 3)
    assert labeled.perch_topk_idx is not None and labeled.perch_topk_idx.shape == (15, TOPK_RESIDUAL)
    assert labeled.perch_topk_val is not None and labeled.perch_topk_val.shape == (15, TOPK_RESIDUAL)
    assert focal.perch_mapped is not None and focal.perch_mapped.shape == (10, 3)
    assert focal.perch_topk_idx is not None and focal.perch_topk_idx.shape == (10, TOPK_RESIDUAL)
    assert focal.perch_topk_val is not None and focal.perch_topk_val.shape == (10, TOPK_RESIDUAL)
    assert unsup.perch_mapped is not None and unsup.perch_mapped.shape == (8, 3)
    assert unsup.perch_topk_idx is not None and unsup.perch_topk_idx.shape == (8, TOPK_RESIDUAL)
    assert unsup.perch_topk_val is not None and unsup.perch_topk_val.shape == (8, TOPK_RESIDUAL)

    base = cache_dir / BASE_DIRNAME
    assert (base / "labeled_perch.npz").exists()
    assert len(list((base / "focal_perch").glob("*.npz"))) == 3
    assert len(list((base / "unsup_perch").glob("*.npz"))) == 3
    assert np.all(labeled.perch_topk_idx >= -1)
    assert np.all(labeled.perch_topk_val >= 0.0)

    print("smoke sidecar pipeline: OK", flush=True)


if __name__ == "__main__":
    main()
