"""Tests for the persistent sharded training-data cache.

These tests build a tiny synthetic "features_dir" with a handful of parquet
shards — same schema the real exp106 cache uses, just smaller — then exercise
``load_training_data_cached`` against it. No real corpus required.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from birdclef_107 import cache as cache_mod


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


SPECIES = ["sp_a", "sp_b", "sp_c"]
EMB_DIM = 8
PERCH_DIM = 5
SITES = ["S01", "S02", "S03"]


def _fname(site: str, i: int) -> str:
    return f"BC2026_Train_{i:04d}_{site}_20240101_120000.ogg"


def _make_df(
    n: int,
    *,
    sites: list[str],
    with_labels: bool,
    seed: int,
) -> pl.DataFrame:
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
        # One or two species per row
        gt = []
        for _ in range(n):
            k = int(rng.integers(1, 3))
            picks = rng.choice(SPECIES, size=k, replace=False)
            gt.append(";".join(picks.tolist()))
        cols["ground_truth"] = gt
    return pl.DataFrame(cols)


@pytest.fixture
def features_dir(tmp_path: Path) -> Path:
    """Build a minimal exp106-shaped features dir under tmp_path.

    Layout::
        features/
        ├── soundscape_labeled.parquet              # 15 rows
        ├── focal/shard-00..02.parquet              # 3 shards × 12 rows = 36
        ├── soundscape_unlabeled/shard-00..02.parquet   # 3 shards × 10 rows = 30
        └── soundscape_unlabeled/by_site/S01..S03.parquet  # empty stubs
    """
    d = tmp_path / "features"
    d.mkdir()

    _make_df(15, sites=SITES, with_labels=True, seed=1).write_parquet(
        d / "soundscape_labeled.parquet"
    )

    (d / "focal").mkdir()
    for i in range(3):
        _make_df(12, sites=SITES, with_labels=True, seed=100 + i).write_parquet(
            d / "focal" / f"shard-{i:02d}.parquet"
        )

    (d / "soundscape_unlabeled").mkdir()
    for i in range(3):
        _make_df(10, sites=SITES, with_labels=False, seed=200 + i).write_parquet(
            d / "soundscape_unlabeled" / f"shard-{i:02d}.parquet"
        )

    by_site = d / "soundscape_unlabeled" / "by_site"
    by_site.mkdir()
    # list_sites() only reads stem names, so empty stubs are fine.
    for site in SITES:
        pl.DataFrame({"filename": []}).write_parquet(by_site / f"{site}.parquet")

    return d


@pytest.fixture(autouse=True)
def _patch_species_list(monkeypatch):
    """Replace the 234-species canonical list with our 3-species synthetic one."""
    monkeypatch.setattr(cache_mod, "get_species_list", lambda: list(SPECIES))


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_roundtrip_matches_direct_load(features_dir: Path, cache_root: Path):
    """Cold build → load → every field bit-exact vs a direct parquet read."""
    labeled, focal, unsup = cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=0,
        unsup_size=0,
    )

    assert labeled.embeddings.shape == (15, EMB_DIM)
    assert labeled.labels.shape == (15, len(SPECIES))
    assert len(labeled.filenames) == 15
    assert labeled.species_list == SPECIES
    assert set(labeled.site_to_idx.keys()) >= set(SITES)
    assert labeled.perch_mapped is None  # default off
    assert labeled.perch_topk_idx is None
    assert labeled.perch_topk_val is None

    assert focal.embeddings.shape == (36, EMB_DIM)
    assert focal.labels.shape == (36, len(SPECIES))
    assert focal.species_list == SPECIES
    assert focal.perch_mapped is None
    assert focal.perch_topk_idx is None
    assert focal.perch_topk_val is None

    assert unsup.embeddings.shape == (30, EMB_DIM)
    assert unsup.site_indices.shape == (30,)
    assert unsup.recording_indices.shape == (30,)
    assert unsup.perch_mapped is None
    assert unsup.perch_topk_idx is None
    assert unsup.perch_topk_val is None

    # Spot-check: labeled embeddings match direct read.
    df = pl.read_parquet(features_dir / "soundscape_labeled.parquet")
    expected_emb = np.stack(df["embedding"].to_list()).astype(np.float32)
    np.testing.assert_array_equal(labeled.embeddings, expected_emb)


def test_size_zero_bypasses_subsampling(features_dir: Path, cache_root: Path):
    _, focal, unsup = cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=0,
        unsup_size=0,
    )
    assert focal.embeddings.shape[0] == 36
    assert unsup.embeddings.shape[0] == 30


def test_size_subsample(features_dir: Path, cache_root: Path):
    _, focal, unsup = cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=10,
        unsup_size=8,
    )
    assert focal.embeddings.shape[0] == 10
    assert unsup.embeddings.shape[0] == 8

    idx_files = sorted((cache_root / cache_mod.IDX_DIRNAME).glob("*.npy"))
    assert {p.name for p in idx_files} == {
        "focal-n10-seed42.npy",
        "unsup-n8-seed42.npy",
    }


def test_touching_one_shard_rebuilds_only_that_shard(
    features_dir: Path, cache_root: Path
):
    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=0,
        unsup_size=0,
    )

    base = cache_root / cache_mod.BASE_DIRNAME
    focal_npzs = sorted((base / "focal").glob("shard-*.npz"))
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in focal_npzs}
    inodes_before = {p.name: p.stat().st_ino for p in focal_npzs}

    # Bump mtime on just one source parquet.
    os.utime(features_dir / "focal" / "shard-01.parquet", None)
    # Ensure mtime advances on fast filesystems.
    new_time_ns = max(mtimes_before.values()) + 1_000_000
    os.utime(
        features_dir / "focal" / "shard-01.parquet",
        ns=(new_time_ns, new_time_ns),
    )

    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=0,
        unsup_size=0,
    )

    mtimes_after = {p.name: p.stat().st_mtime_ns for p in focal_npzs}
    inodes_after = {p.name: p.stat().st_ino for p in focal_npzs}

    assert mtimes_after["shard-01.npz"] != mtimes_before["shard-01.npz"]
    assert inodes_after["shard-01.npz"] != inodes_before["shard-01.npz"]
    for name in ("shard-00.npz", "shard-02.npz"):
        assert inodes_after[name] == inodes_before[name], (
            f"{name} was rebuilt but its source parquet was untouched"
        )


def test_seed_change_only_rebuilds_index(features_dir: Path, cache_root: Path):
    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=10,
        unsup_size=8,
        seed=42,
    )

    base = cache_root / cache_mod.BASE_DIRNAME
    all_npz = list(base.rglob("*.npz"))
    inodes_before = {p: p.stat().st_ino for p in all_npz}

    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=10,
        unsup_size=8,
        seed=7,
    )

    for p, ino in inodes_before.items():
        assert p.stat().st_ino == ino, f"{p.name} was rebuilt on seed change"

    idx_names = {p.name for p in (cache_root / cache_mod.IDX_DIRNAME).glob("*.npy")}
    assert "focal-n10-seed42.npy" in idx_names
    assert "focal-n10-seed7.npy" in idx_names
    assert "unsup-n8-seed42.npy" in idx_names
    assert "unsup-n8-seed7.npy" in idx_names


def test_second_identical_call_is_noop(features_dir: Path, cache_root: Path):
    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=10,
        unsup_size=8,
    )

    all_files = list((cache_root / cache_mod.BASE_DIRNAME).rglob("*.npz"))
    all_files += list((cache_root / cache_mod.IDX_DIRNAME).rglob("*.npy"))
    inodes_before = {p: p.stat().st_ino for p in all_files}

    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=10,
        unsup_size=8,
    )

    for p, ino in inodes_before.items():
        assert p.stat().st_ino == ino, f"{p.name} was rewritten on a no-op call"


def test_perch_sidecars_opt_in(features_dir: Path, cache_root: Path):
    # First call: no perch. Sidecars must not exist.
    labeled, focal, unsup = cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=0,
        unsup_size=0,
    )
    assert labeled.perch_mapped is None
    assert labeled.perch_topk_idx is None
    assert labeled.perch_topk_val is None
    assert focal.perch_mapped is None
    assert focal.perch_topk_idx is None
    assert focal.perch_topk_val is None
    assert unsup.perch_mapped is None
    assert unsup.perch_topk_idx is None
    assert unsup.perch_topk_val is None

    base = cache_root / cache_mod.BASE_DIRNAME
    assert not (base / "labeled_perch.npz").exists()
    assert not (base / "focal_perch").exists() or not list(
        (base / "focal_perch").glob("*.npz")
    )

    # Second call with include_perch_logits: compact sidecars built, fields populated.
    labeled2, focal2, unsup2 = cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=0,
        unsup_size=0,
        include_perch_logits=True,
    )
    assert labeled2.perch_mapped is not None
    assert labeled2.perch_mapped.shape == (15, len(SPECIES))
    assert labeled2.perch_topk_idx is not None
    assert labeled2.perch_topk_idx.shape == (15, 64)
    assert labeled2.perch_topk_val is not None
    assert labeled2.perch_topk_val.shape == (15, 64)
    assert focal2.perch_mapped is not None
    assert focal2.perch_mapped.shape == (36, len(SPECIES))
    assert focal2.perch_topk_idx is not None
    assert focal2.perch_topk_idx.shape == (36, 64)
    assert focal2.perch_topk_val is not None
    assert focal2.perch_topk_val.shape == (36, 64)
    assert unsup2.perch_mapped is not None
    assert unsup2.perch_mapped.shape == (30, len(SPECIES))
    assert unsup2.perch_topk_idx is not None
    assert unsup2.perch_topk_idx.shape == (30, 64)
    assert unsup2.perch_topk_val is not None
    assert unsup2.perch_topk_val.shape == (30, 64)

    assert (base / "labeled_perch.npz").exists()
    focal_perch_shards = sorted((base / "focal_perch").glob("shard-*.npz"))
    unsup_perch_shards = sorted((base / "unsup_perch").glob("shard-*.npz"))
    assert len(focal_perch_shards) == 3
    assert len(unsup_perch_shards) == 3

    # Third call with perch: no-op (inodes stable).
    inodes_before = {
        p: p.stat().st_ino
        for p in [base / "labeled_perch.npz", *focal_perch_shards, *unsup_perch_shards]
    }
    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=0,
        unsup_size=0,
        include_perch_logits=True,
    )
    for p, ino in inodes_before.items():
        assert p.stat().st_ino == ino, f"{p.name} was rewritten on perch no-op call"


def test_force_rebuild_index(features_dir: Path, cache_root: Path):
    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=10,
        unsup_size=8,
    )
    idx_dir = cache_root / cache_mod.IDX_DIRNAME
    before = {p: p.stat().st_mtime_ns for p in idx_dir.glob("*.npy")}

    # Ensure some wall-clock delta so mtimes differ.
    new_time_ns = max(before.values()) + 1_000_000
    for p in before:
        os.utime(p, ns=(new_time_ns - 10_000_000, new_time_ns - 10_000_000))
    floor = max(p.stat().st_mtime_ns for p in before)

    cache_mod.load_training_data_cached(
        features_dir=str(features_dir),
        cache_root=cache_root,
        focal_size=10,
        unsup_size=8,
        force_rebuild_index=True,
    )
    after = {p: p.stat().st_mtime_ns for p in idx_dir.glob("*.npy")}
    assert all(mt > floor for mt in after.values())
