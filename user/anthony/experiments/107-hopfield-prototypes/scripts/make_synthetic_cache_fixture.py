#!/usr/bin/env python3
"""Create a tiny exp106-shaped synthetic feature/artifact tree for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

SPECIES = ["sp_a", "sp_b", "sp_c"]
SITES = ["S01", "S02", "S03"]
EMB_DIM = 8
PERCH_DIM = 8


def _fname(site: str, i: int) -> str:
    return f"BC2026_Train_{i:04d}_{site}_20240101_120000.ogg"


def _make_df(n: int, *, sites: list[str], with_labels: bool, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    site_picks = rng.choice(sites, size=n)
    filenames = [_fname(s, seed * 10_000 + i) for i, s in enumerate(site_picks)]
    embeddings = rng.standard_normal((n, EMB_DIM)).astype(np.float32)
    perch_logits = rng.standard_normal((n, PERCH_DIM)).astype(np.float32)
    cols = {
        "filename": filenames,
        "embedding": [row.tolist() for row in embeddings],
        "perch_logits": [row.tolist() for row in perch_logits],
    }
    if with_labels:
        gt = []
        for _ in range(n):
            k = int(rng.integers(1, 3))
            picks = rng.choice(SPECIES, size=k, replace=False)
            gt.append(";".join(picks.tolist()))
        cols["ground_truth"] = gt
    return pl.DataFrame(cols)


def write_fixture(root: Path) -> tuple[Path, Path]:
    features_dir = root / "features"
    artifacts_dir = root / "artifacts"
    features_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _make_df(15, sites=SITES, with_labels=True, seed=1).write_parquet(
        features_dir / "soundscape_labeled.parquet"
    )

    (features_dir / "focal").mkdir(exist_ok=True)
    for i in range(3):
        _make_df(12, sites=SITES, with_labels=True, seed=100 + i).write_parquet(
            features_dir / "focal" / f"shard-{i:02d}.parquet"
        )

    (features_dir / "soundscape_unlabeled").mkdir(exist_ok=True)
    for i in range(3):
        _make_df(10, sites=SITES, with_labels=False, seed=200 + i).write_parquet(
            features_dir / "soundscape_unlabeled" / f"shard-{i:02d}.parquet"
        )

    by_site = features_dir / "soundscape_unlabeled" / "by_site"
    by_site.mkdir(exist_ok=True)
    for site in SITES:
        pl.DataFrame({"filename": []}).write_parquet(by_site / f"{site}.parquet")

    mapping = {
        "direct_map": {"sp_a": 1},
        "genus_proxy_map": {"sp_b": [2, 4]},
        "unmapped": ["sp_c"],
        "birdclef_species": SPECIES,
        "taxa_classes": {s: "Aves" for s in SPECIES},
    }
    (artifacts_dir / "perch_species_mapping.json").write_text(json.dumps(mapping, indent=2))
    return features_dir, artifacts_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Create synthetic cache smoke-test fixture")
    p.add_argument("--root", type=str, required=True)
    args = p.parse_args()
    root = Path(args.root).expanduser()
    features_dir, artifacts_dir = write_fixture(root)
    print(f"features_dir={features_dir}")
    print(f"artifacts_dir={artifacts_dir}")


if __name__ == "__main__":
    main()
