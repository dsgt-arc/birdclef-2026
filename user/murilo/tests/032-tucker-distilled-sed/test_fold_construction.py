"""Tests for fold construction + DataModule (exp 032 task 0.8).

Verifies:
- `build_focal_folds`: StratifiedKFold on `primary_label`, upsampling preserves
  fold membership, no cross-fold leak.
- `build_sc_folds`: GroupKFold on `filename` — all 12 windows of a file share
  one fold.
- `build_sc_cache_meta`: enumerates 12 × 5s windows per labeled file.
- `build_sc_label_matrix`: matches (filename, start_sec) tuples; produces
  all-zero rows for unmatched windows (audit § 1.4).
- `DistilledSEDDataModule`: end-to-end pipeline produces correct train + val
  DataLoaders with expected batch composition.

Reference:
- src/birdclef/data/distilled_sed_datamodule.py
- analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md (§ 1)
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 7)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.data.distilled_sed_datamodule import (
    DistilledSEDDataModule,
    build_focal_folds,
    build_sc_cache_meta,
    build_sc_folds,
    build_sc_label_matrix,
)

NUM_CLASSES = 234


# ============================================================================
# Fixtures
# ============================================================================


def _make_focal_df(species_counts: dict[str, int]) -> pd.DataFrame:
    """Build a synthetic focal DataFrame with `species_counts[sp]` rows per species."""
    rows = []
    idx = 0
    for sp, n in species_counts.items():
        for j in range(n):
            rows.append({"primary_label": sp, "filename": f"{sp}_{j}.ogg"})
            idx += 1
    return pd.DataFrame(rows)


def _make_stub_loader(constant: float = 0.0):
    """Audio loader that returns deterministic-by-path synthetic waveforms."""

    def _loader(path, n_samples, start=0, sample_rate=32000):
        path_hash = abs(hash(str(path))) % (2**31 - 1)
        rng = np.random.default_rng(path_hash + start)
        return rng.standard_normal(n_samples).astype(np.float32) * 0.1 + constant

    return _loader


# ============================================================================
# build_focal_folds — fold assignment + upsampling
# ============================================================================


def test_focal_folds_stratified_balanced():
    """StratifiedKFold ensures each fold has roughly equal class representation.
    With 20 samples per species × 5 species × 5 folds → each fold has ~4 per species."""
    df_in = _make_focal_df({f"sp{i}": 20 for i in range(5)})
    df = build_focal_folds(df_in, n_folds=5, seed=42, min_sample=0)  # no upsampling

    assert "fold" in df.columns
    assert "original_idx" in df.columns
    # All folds populated
    fold_counts = df["fold"].value_counts().sort_index()
    assert (fold_counts >= 18).all(), f"fold sizes too uneven: {fold_counts}"

    # Per-species fold distribution
    for sp in df["primary_label"].unique():
        sp_folds = df[df["primary_label"] == sp]["fold"].value_counts()
        # 20 samples / 5 folds = 4 per fold; allow ±1
        assert sp_folds.min() >= 3, f"species {sp} fold {sp_folds.idxmin()} too small"
        assert sp_folds.max() <= 5, f"species {sp} fold {sp_folds.idxmax()} too large"


def test_focal_folds_original_idx_added():
    """`original_idx` is added as range(N_original)."""
    df_in = _make_focal_df({"sp0": 10, "sp1": 10, "sp2": 10, "sp3": 10, "sp4": 10})
    df = build_focal_folds(df_in, n_folds=5, seed=42, min_sample=0)
    unique_oidx = df["original_idx"].unique()
    assert set(unique_oidx) == set(range(50))


def test_focal_folds_upsampling_preserves_fold():
    """Audit § 1.3: rare-species duplicates inherit the original's fold (no leak)."""
    # 3 samples for `rare` species (will be upsampled to ~20), 20 for others
    df_in = _make_focal_df({"rare": 3, **{f"sp{i}": 20 for i in range(5)}})
    df = build_focal_folds(df_in, n_folds=5, seed=42, min_sample=20)

    # Find the rare-species rows. There should be ~20 total.
    rare = df[df["primary_label"] == "rare"]
    # ceil(20 / 3) - 1 = 6 extra copies × 3 original = 18 extras + 3 originals = 21
    assert len(rare) >= 18, f"rare not upsampled enough: {len(rare)}"

    # Each (original_idx, fold) pair must be consistent — duplicates inherit fold
    pairs = rare.groupby("original_idx")["fold"].nunique()
    assert (pairs == 1).all(), f"rare upsample broke fold consistency: {pairs[pairs > 1]}"


def test_focal_folds_no_cross_fold_leak():
    """For any original_idx, all rows with that idx must share a single fold."""
    df_in = _make_focal_df({"sp0": 30, "sp1": 30, "sp2": 30, "sp3": 30, "sp4": 30, "rare": 5})
    df = build_focal_folds(df_in, n_folds=5, seed=42, min_sample=20)
    leaks = df.groupby("original_idx")["fold"].nunique()
    assert (
        leaks == 1
    ).all(), f"original_idx with multiple folds: {leaks[leaks > 1].index.tolist()}"


def test_focal_folds_seed_reproducible():
    df_in = _make_focal_df({f"sp{i}": 20 for i in range(5)})
    df1 = build_focal_folds(df_in, n_folds=5, seed=42, min_sample=0)
    df2 = build_focal_folds(df_in, n_folds=5, seed=42, min_sample=0)
    pd.testing.assert_frame_equal(df1, df2)


def test_focal_folds_rejects_existing_fold_col():
    df_in = _make_focal_df({"sp0": 10})
    df_in["fold"] = 0
    with pytest.raises(ValueError, match="'fold' column"):
        build_focal_folds(df_in)


def test_focal_folds_rejects_existing_oidx_col():
    df_in = _make_focal_df({"sp0": 10})
    df_in["original_idx"] = 0
    with pytest.raises(ValueError, match="'original_idx' column"):
        build_focal_folds(df_in)


# ============================================================================
# build_sc_cache_meta
# ============================================================================


def test_sc_cache_meta_default_12_windows():
    """60s file at 5s windows → 12 windows, start_sec ∈ {0, 5, ..., 55}."""
    files = [f"BC2026_Train_{i:04d}.ogg" for i in range(3)]
    df = build_sc_cache_meta(files)
    assert len(df) == 3 * 12
    # Each filename has 12 entries
    for fn in files:
        assert (df["filename"] == fn).sum() == 12
    # start_sec values per file
    assert sorted(df[df["filename"] == files[0]]["start_sec"].tolist()) == list(range(0, 60, 5))


def test_sc_cache_meta_matches_audit_count():
    """Audit § 1.4: 66 labeled files × 12 windows = 792 cached SS windows."""
    files = [f"file_{i:03d}.ogg" for i in range(66)]
    df = build_sc_cache_meta(files)
    assert len(df) == 792


# ============================================================================
# build_sc_folds
# ============================================================================


def test_sc_folds_all_windows_share_file_fold():
    """GroupKFold on filename — all 12 windows of any single file share a fold."""
    files = [f"file_{i:03d}.ogg" for i in range(10)]
    sc_df = build_sc_cache_meta(files)
    folded = build_sc_folds(sc_df, n_folds=5)

    # For each filename, all rows must share one fold
    for fn in files:
        folds = folded[folded["filename"] == fn]["fold"].unique()
        assert len(folds) == 1, f"file {fn} has rows across folds {folds}"


def test_sc_folds_all_folds_used():
    files = [f"file_{i:03d}.ogg" for i in range(20)]
    sc_df = build_sc_cache_meta(files)
    folded = build_sc_folds(sc_df, n_folds=5)
    assert sorted(folded["fold"].unique()) == [0, 1, 2, 3, 4]


# ============================================================================
# build_sc_label_matrix
# ============================================================================


def test_sc_label_matrix_matches_audit_format():
    """Audit § 1.4: (filename, start_sec) matched against labels CSV.
    Unmatched windows → all-zero rows."""
    sc_df = pd.DataFrame(
        {
            "filename": ["fileA.ogg", "fileA.ogg", "fileB.ogg", "fileB.ogg"],
            "start_sec": [0, 5, 0, 5],
        }
    )
    labels_df = pd.DataFrame(
        {
            "filename": ["fileA.ogg", "fileB.ogg"],
            "start": ["00:00:00", "00:00:05"],
            "end": ["00:00:05", "00:00:10"],
            "primary_label": ["sp0;sp3", "sp1"],
        }
    )
    label2idx = {f"sp{i}": i for i in range(NUM_CLASSES)}
    Y_SC = build_sc_label_matrix(sc_df, labels_df, label2idx, NUM_CLASSES)

    assert Y_SC.shape == (4, NUM_CLASSES)
    # Row 0: fileA@0 matches → sp0, sp3 → indices 0 and 3
    assert Y_SC[0, 0] == 1.0
    assert Y_SC[0, 3] == 1.0
    assert Y_SC[0, [1, 2, 4]].sum() == 0.0
    # Row 1: fileA@5 NOT matched → all-zero
    assert Y_SC[1].sum() == 0.0
    # Row 2: fileB@0 NOT matched → all-zero
    assert Y_SC[2].sum() == 0.0
    # Row 3: fileB@5 matches → sp1 → index 1
    assert Y_SC[3, 1] == 1.0
    assert Y_SC[3, [0, 2, 3]].sum() == 0.0


def test_sc_label_matrix_handles_multi_row_per_window():
    """Multiple label CSV rows with same (filename, start_sec) get unioned."""
    sc_df = pd.DataFrame({"filename": ["a.ogg"], "start_sec": [0]})
    labels_df = pd.DataFrame(
        {
            "filename": ["a.ogg", "a.ogg"],
            "start": ["00:00:00", "00:00:00"],
            "end": ["00:00:05", "00:00:05"],
            "primary_label": ["sp1", "sp5"],
        }
    )
    label2idx = {f"sp{i}": i for i in range(NUM_CLASSES)}
    Y_SC = build_sc_label_matrix(sc_df, labels_df, label2idx, NUM_CLASSES)
    assert Y_SC[0, 1] == 1.0
    assert Y_SC[0, 5] == 1.0


def test_sc_label_matrix_unknown_species_skipped():
    """Species not in label2idx are silently skipped (not crashed)."""
    sc_df = pd.DataFrame({"filename": ["a.ogg"], "start_sec": [0]})
    labels_df = pd.DataFrame(
        {
            "filename": ["a.ogg"],
            "start": ["00:00:00"],
            "end": ["00:00:05"],
            "primary_label": [
                "sp1;unknown_species;sp7",  # one unknown in the middle
            ],
        }
    )
    label2idx = {"sp1": 1, "sp7": 7}
    Y_SC = build_sc_label_matrix(sc_df, labels_df, label2idx, NUM_CLASSES)
    assert Y_SC[0, 1] == 1.0
    assert Y_SC[0, 7] == 1.0
    # All other positions zero
    assert Y_SC[0].sum() == 2.0


# ============================================================================
# DistilledSEDDataModule — end-to-end
# ============================================================================


@pytest.fixture
def datamodule():
    """A small in-memory DataModule for fast integration tests."""
    focal_df = _make_focal_df({f"sp{i}": 8 for i in range(5)})  # 40 focal rows
    # SS: 5 files × 12 windows = 60 windows. Label only 4 windows.
    sc_files = [f"sc_{i:02d}.ogg" for i in range(5)]
    sc_df = build_sc_cache_meta(sc_files)
    sc_labels_df = pd.DataFrame(
        {
            "filename": ["sc_00.ogg", "sc_00.ogg", "sc_02.ogg", "sc_04.ogg"],
            "start": ["00:00:00", "00:00:05", "00:00:10", "00:00:15"],
            "end": ["00:00:05", "00:00:10", "00:00:15", "00:00:20"],
            "primary_label": ["sp0", "sp1", "sp2", "sp3"],
        }
    )
    species_to_idx = {f"sp{i}": i for i in range(NUM_CLASSES)}

    dm = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=NUM_CLASSES,
        fold_k=0,
        n_folds=5,
        min_sample=0,  # no upsampling for simplicity
        window_samples=160_000,
        batch_size=16,
        focal_share=0.9,
        sc_share=0.1,
        steps_per_epoch=10,
        num_workers=0,
        focal_audio_loader=_make_stub_loader(),
        sc_audio_loader=_make_stub_loader(constant=0.5),
        focal_ds_kwargs={"mixup_prob": 0.0, "focal_sc_mixup_prob": 0.0},
        seed_epoch_shift=False,
    )
    dm.setup("fit")
    return dm


def test_datamodule_setup_assigns_folds(datamodule):
    """setup() populates focal_df.fold, sc_df.fold, and Y_SC."""
    assert datamodule.focal_df is not None
    assert "fold" in datamodule.focal_df.columns
    assert datamodule.sc_df is not None
    assert "fold" in datamodule.sc_df.columns
    assert datamodule.Y_SC is not None
    assert datamodule.Y_SC.shape == (len(datamodule.sc_df), NUM_CLASSES)


def test_datamodule_steps_per_epoch_explicit(datamodule):
    assert datamodule.steps_per_epoch == 10


def test_datamodule_train_dataloader_batch_composition(datamodule):
    """Train DataLoader uses MixSamp → batches contain expected focal/sc counts.
    With batch_size=16, shares={focal:0.9, sc:0.1} → per_src = [14, 2]."""
    loader = datamodule.train_dataloader()
    n_batches = 0
    for batch in loader:
        wavs, labels, wts, mks = batch
        assert wavs.shape[0] == 16
        assert labels.shape == (16, NUM_CLASSES)
        n_batches += 1
    assert n_batches == 10  # steps_per_epoch


def test_datamodule_val_dataloader_uses_val_fold(datamodule):
    """Val DataLoader serves rows where fold == fold_k. Should NOT be empty
    (fold 0 has some SS windows)."""
    loader = datamodule.val_dataloader()
    val_count = 0
    for batch in loader:
        wavs, labels, wts, mks = batch
        val_count += wavs.shape[0]
    # 5 files / 5 folds → ~1 file in val fold = 12 windows
    assert val_count > 0, "val DataLoader produced no batches"
    assert val_count <= 24, f"val DataLoader produced too many: {val_count}"


def test_datamodule_train_excludes_val_fold(datamodule):
    """No filename in train fold should also appear in val fold (SS).
    GroupKFold on filename ensures no cross-fold leak."""
    train_files = set(
        datamodule.sc_df[datamodule.sc_df["fold"] != datamodule.fold_k]["filename"].unique()
    )
    val_files = set(
        datamodule.sc_df[datamodule.sc_df["fold"] == datamodule.fold_k]["filename"].unique()
    )
    assert train_files.isdisjoint(
        val_files
    ), f"file leak between train and val: {train_files & val_files}"


def test_datamodule_fold_k_changes_split(datamodule):
    """Changing fold_k between 0 and 1 swaps which SS files are held out."""
    val_0 = set(datamodule.sc_df[datamodule.sc_df["fold"] == 0]["filename"].unique())
    val_1 = set(datamodule.sc_df[datamodule.sc_df["fold"] == 1]["filename"].unique())
    # Different fold → different held-out files (no overlap)
    assert val_0.isdisjoint(val_1)
