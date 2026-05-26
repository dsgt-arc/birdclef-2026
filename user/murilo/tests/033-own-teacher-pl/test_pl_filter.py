"""Tests for PL parquet filtering in DistilledSEDDataModule (exp 033).

Verifies the leakage-prevention + GT/PL conflict-avoidance filters in
`DistilledSEDDataModule.setup()`:
- val-fold SS files are excluded from the PL pool (no val leak into train)
- labeled-SS train files are excluded (Antoine: GT used for labeled, PL only
  for unlabeled) when `pseudo_label_exclude_labeled_sc=True`
- Resulting pool only contains unlabeled SS segments

Uses synthetic data (no real audio I/O).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.data.distilled_sed_datamodule import DistilledSEDDataModule

NUM_CLASSES = 234


def _make_synthetic_focal_df(n_species: int = 5, per_species: int = 20):
    rows = []
    for sp_i in range(n_species):
        for j in range(per_species):
            rows.append({"primary_label": f"sp{sp_i}", "filename": f"focal_{sp_i}_{j}.ogg"})
    return pd.DataFrame(rows)


def _make_synthetic_sc_setup(n_labeled_files: int = 10):
    """Build sc_df with 12 windows × n_labeled_files SS files, plus labels_df."""
    sc_rows = []
    for fi in range(n_labeled_files):
        for w in range(12):
            sc_rows.append({"filename": f"sc_labeled_{fi}.ogg", "start_sec": w * 5})
    sc_df = pd.DataFrame(sc_rows)

    # Labels: every row gets sp0 set to 1.0 (schema matches build_sc_label_matrix:
    # filename, start, end, primary_label)
    sc_labels_df = pd.DataFrame(
        [
            {
                "filename": row["filename"],
                "start": f"00:00:{int(row['start_sec']):02d}",
                "end": "00:01:00",
                "primary_label": "sp0",
            }
            for _, row in sc_df.iterrows()
        ]
    )
    return sc_df, sc_labels_df


def _make_pl_df(labeled_files: list[str], unlabeled_files: list[str], species_list: list[str]):
    """Mock PL parquet: includes BOTH labeled + unlabeled SS files (mirrors v1/sed_pseudo_labels.parquet)."""
    rows = []
    rng = np.random.default_rng(0)
    for fn in labeled_files + unlabeled_files:
        for chunk_idx in range(12):
            row = {"filename": fn, "chunk_idx": chunk_idx}
            for sp in species_list:
                row[sp] = float(rng.random())
            rows.append(row)
    return pd.DataFrame(rows)


def test_pl_filter_excludes_val_fold_files(tmp_path):
    """The 14-fold-0-style val SS files must be removed from PL pool."""
    species_list = [f"sp{i}" for i in range(NUM_CLASSES)]
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}

    sc_df_raw, sc_labels_df = _make_synthetic_sc_setup(n_labeled_files=10)
    focal_df = _make_synthetic_focal_df()

    # Unlabeled SS files (not in sc_df)
    unlabeled_files = [f"sc_unlabeled_{i}.ogg" for i in range(5)]
    pl_df_full = _make_pl_df(
        labeled_files=sc_df_raw["filename"].unique().tolist(),
        unlabeled_files=unlabeled_files,
        species_list=species_list,
    )

    # Write PL parquet to tmp_path
    pl_path = tmp_path / "pl.parquet"
    pl_df_full.to_parquet(pl_path)

    dm = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df_raw,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=NUM_CLASSES,
        fold_k=0,
        n_folds=5,
        pseudo_label_path=pl_path,
        pseudo_label_exclude_labeled_sc=True,
        soundscape_dir=tmp_path,  # dummy; won't load audio
    )
    dm.setup("fit")

    # After filtering: only unlabeled SS files should remain in pl_df
    assert dm.pl_df is not None
    pl_files_remaining = set(dm.pl_df["filename"].unique())
    val_fold_files = set(dm.sc_df[dm.sc_df["fold"] == 0]["filename"].unique())
    labeled_files = set(dm.sc_df["filename"].unique())

    # Val-fold leak check
    assert pl_files_remaining.isdisjoint(val_fold_files), (
        f"PL pool contains val-fold files (leak): " f"{pl_files_remaining & val_fold_files}"
    )
    # Labeled-SS exclusion check
    assert pl_files_remaining.isdisjoint(labeled_files), (
        f"PL pool contains labeled SS files (Antoine: should use GT instead): "
        f"{pl_files_remaining & labeled_files}"
    )
    # All unlabeled files should remain
    assert pl_files_remaining == set(unlabeled_files), (
        f"Expected only unlabeled files: missing={set(unlabeled_files) - pl_files_remaining}, "
        f"extra={pl_files_remaining - set(unlabeled_files)}"
    )


def test_pl_filter_keeps_labeled_when_disabled(tmp_path):
    """With `exclude_labeled_sc=False`, labeled SS files SHOULD appear in PL pool
    (still filtered for val-fold leak though)."""
    species_list = [f"sp{i}" for i in range(NUM_CLASSES)]
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}

    sc_df_raw, sc_labels_df = _make_synthetic_sc_setup(n_labeled_files=10)
    focal_df = _make_synthetic_focal_df()
    unlabeled_files = [f"sc_unlabeled_{i}.ogg" for i in range(3)]
    pl_df_full = _make_pl_df(
        labeled_files=sc_df_raw["filename"].unique().tolist(),
        unlabeled_files=unlabeled_files,
        species_list=species_list,
    )
    pl_path = tmp_path / "pl.parquet"
    pl_df_full.to_parquet(pl_path)

    dm = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df_raw,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=NUM_CLASSES,
        fold_k=0,
        n_folds=5,
        pseudo_label_path=pl_path,
        pseudo_label_exclude_labeled_sc=False,  # ← KEEP labeled
        soundscape_dir=tmp_path,
    )
    dm.setup("fit")

    pl_files_remaining = set(dm.pl_df["filename"].unique())
    val_fold_files = set(dm.sc_df[dm.sc_df["fold"] == 0]["filename"].unique())
    train_fold_labeled = set(dm.sc_df[dm.sc_df["fold"] != 0]["filename"].unique())

    # Val-fold STILL filtered (always)
    assert pl_files_remaining.isdisjoint(val_fold_files)
    # Train-fold labeled SS files should be PRESENT
    assert train_fold_labeled.issubset(pl_files_remaining)
    # Unlabeled files also present
    assert set(unlabeled_files).issubset(pl_files_remaining)


def test_pl_disabled_when_path_is_none(tmp_path):
    """When `pseudo_label_path=None`, dm.pl_df must remain None (backward-compat
    with exp 032 — no PL integration)."""
    species_list = [f"sp{i}" for i in range(NUM_CLASSES)]
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    sc_df_raw, sc_labels_df = _make_synthetic_sc_setup(n_labeled_files=10)
    focal_df = _make_synthetic_focal_df()

    dm = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df_raw,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=NUM_CLASSES,
        fold_k=0,
        n_folds=5,
        pseudo_label_path=None,  # NO PL
    )
    dm.setup("fit")
    assert dm.pl_df is None


def test_soundscape_dir_required_when_pl_enabled():
    """Constructor must raise if pseudo_label_path is set but soundscape_dir is missing."""
    species_to_idx = {"sp0": 0}
    sc_df_raw, sc_labels_df = _make_synthetic_sc_setup(n_labeled_files=5)
    focal_df = _make_synthetic_focal_df()

    import pytest

    with pytest.raises(ValueError, match="soundscape_dir is required"):
        DistilledSEDDataModule(
            focal_df=focal_df,
            sc_df=sc_df_raw,
            sc_labels_df=sc_labels_df,
            species_to_idx=species_to_idx,
            num_classes=NUM_CLASSES,
            pseudo_label_path="/tmp/dummy.parquet",  # PL path SET
            soundscape_dir=None,  # but soundscape_dir is MISSING
        )
