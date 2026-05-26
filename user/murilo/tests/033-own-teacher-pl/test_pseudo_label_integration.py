"""Tests for PLDS + GTDatasetWrapper + collate integration (exp 033).

Verifies:
- `PLDS` returns the correct 6-tuple shape with soft labels in [0,1]
  and loss_weight as a scalar.
- `GTDatasetWrapper` correctly wraps a 5-tuple Dataset into a 6-tuple.
- `ConcatDataset([gt_focal, gt_sc, pl])` indexing returns the right
  tuple from each underlying source.
- `_collate_drop_extras` produces a 4-tuple when given 5-tuple input
  (Tucker default) and a 5-tuple when given 6-tuple input (PL).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.data.distilled_sed_datamodule import _collate_drop_extras
from birdclef.data.distilled_sed_dataset import (
    PLDS,
    GTDatasetWrapper,
    ScDS,
)

NUM_CLASSES = 234
WINDOW_SAMPLES = 160_000


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_stub_loader():
    """Audio loader that returns deterministic-by-path synthetic waveforms."""

    def _loader(path, n_samples, start=0, sample_rate=32000):
        path_hash = abs(hash(str(path))) % (2**31 - 1)
        rng = np.random.default_rng(path_hash + start)
        return rng.standard_normal(n_samples).astype(np.float32) * 0.1

    return _loader


def _make_pl_df(n_files: int = 3, species: list[str] = None):
    """Synthetic PL parquet DataFrame: n_files × 12 windows × len(species) cols."""
    if species is None:
        species = [f"sp{i}" for i in range(NUM_CLASSES)]
    rows = []
    rng = np.random.default_rng(0)
    for fi in range(n_files):
        for chunk_idx in range(12):
            row = {"filename": f"sound{fi}.ogg", "chunk_idx": chunk_idx}
            for sp in species:
                row[sp] = float(rng.random())  # soft prob in [0,1)
            rows.append(row)
    return pd.DataFrame(rows), species


def _make_sc_ds(n_rows: int = 5, species_list: list[str] = None) -> ScDS:
    if species_list is None:
        species_list = [f"sp{i}" for i in range(NUM_CLASSES)]
    sc_df = pd.DataFrame([{"filename": f"sc_{i}.ogg", "start_sec": 0} for i in range(n_rows)])
    Y_SC = np.zeros((n_rows, NUM_CLASSES), dtype=np.float32)
    Y_SC[:, 0] = 1.0
    return ScDS(
        sc_df=sc_df,
        Y_SC=Y_SC,
        window_samples=WINDOW_SAMPLES,
        num_classes=NUM_CLASSES,
        aug=False,
        audio_loader=_make_stub_loader(),
        sample_rate=32000,
    )


# ----------------------------------------------------------------------
# PLDS — basic shape + value contract
# ----------------------------------------------------------------------


def test_plds_returns_6tuple(tmp_path):
    """PLDS.__getitem__ must return 6-tuple `(wav, lb, wt, mk, tag, loss_weight)`."""
    pl_df, species = _make_pl_df(n_files=2)
    ds = PLDS(
        pl_df=pl_df,
        species_list=species,
        soundscape_dir=tmp_path,
        window_samples=WINDOW_SAMPLES,
        loss_weight=1.0,
        audio_loader=_make_stub_loader(),
        sample_rate=32000,
    )
    item = ds[0]
    assert len(item) == 6
    wav, lb, wt, mk, tag, loss_weight = item
    assert wav.shape == (1, WINDOW_SAMPLES), f"wav shape: {wav.shape}"
    assert lb.shape == (NUM_CLASSES,), f"lb shape: {lb.shape}"
    assert wt.shape == (NUM_CLASSES,) and mk.shape == (NUM_CLASSES,)
    assert tag == "pl"
    assert isinstance(loss_weight, float) and loss_weight == 1.0


def test_plds_soft_labels_in_range(tmp_path):
    """PLDS soft labels must be ∈ [0, 1] (from the PL parquet)."""
    pl_df, species = _make_pl_df(n_files=2)
    ds = PLDS(
        pl_df=pl_df,
        species_list=species,
        soundscape_dir=tmp_path,
        audio_loader=_make_stub_loader(),
    )
    for i in range(len(ds)):
        _, lb, *_ = ds[i]
        assert (lb >= 0).all() and (
            lb <= 1
        ).all(), f"row {i}: labels out of [0,1] range: min={lb.min()}, max={lb.max()}"


def test_plds_len_equals_rows(tmp_path):
    pl_df, species = _make_pl_df(n_files=3)
    ds = PLDS(pl_df=pl_df, species_list=species, soundscape_dir=tmp_path)
    assert len(ds) == len(pl_df) == 3 * 12


def test_plds_custom_loss_weight(tmp_path):
    """loss_weight should be configurable per-instance."""
    pl_df, species = _make_pl_df(n_files=1)
    ds = PLDS(
        pl_df=pl_df,
        species_list=species,
        soundscape_dir=tmp_path,
        loss_weight=0.5,
        audio_loader=_make_stub_loader(),
    )
    _, _, _, _, _, lw = ds[0]
    assert lw == 0.5


def test_plds_rejects_missing_species(tmp_path):
    """PLDS must raise if pl_df is missing required species columns."""
    pl_df, species = _make_pl_df(n_files=1)
    # Drop one species col
    pl_df_bad = pl_df.drop(columns=[species[0]])
    with pytest.raises(ValueError, match="missing"):
        PLDS(pl_df=pl_df_bad, species_list=species, soundscape_dir=tmp_path)


# ----------------------------------------------------------------------
# GTDatasetWrapper — converts 5-tuple → 6-tuple
# ----------------------------------------------------------------------


def test_gt_wrapper_converts_5tuple_to_6tuple():
    sc_ds = _make_sc_ds(n_rows=3)
    wrapped = GTDatasetWrapper(sc_ds, loss_weight=1.0)
    assert len(wrapped) == 3
    item = wrapped[0]
    assert len(item) == 6
    *_, tag, lw = item
    assert tag == "sc"  # preserves the underlying tag
    assert lw == 1.0


def test_gt_wrapper_idempotent_on_6tuple():
    """If the wrapped dataset already returns 6-tuple (e.g., already wrapped),
    GTDatasetWrapper must NOT add a 7th element."""

    class Stub:
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return (
                torch.zeros(1, 100),
                torch.zeros(NUM_CLASSES),
                torch.ones(NUM_CLASSES),
                torch.ones(NUM_CLASSES),
                "stub",
                0.7,  # already has loss_weight
            )

    wrapped = GTDatasetWrapper(Stub(), loss_weight=999.0)  # 999 must be IGNORED
    item = wrapped[0]
    assert len(item) == 6
    assert item[5] == 0.7  # preserves existing loss_weight


# ----------------------------------------------------------------------
# ConcatDataset + collate — end-to-end
# ----------------------------------------------------------------------


def test_concat_dataset_gt_sc_pl(tmp_path):
    """ConcatDataset([gt_sc, pl_ds]) must yield correctly-shaped items from both."""
    pl_df, species = _make_pl_df(n_files=2)
    pl_ds = PLDS(
        pl_df=pl_df,
        species_list=species,
        soundscape_dir=tmp_path,
        audio_loader=_make_stub_loader(),
    )
    gt_sc = GTDatasetWrapper(_make_sc_ds(n_rows=4))
    concat = ConcatDataset([gt_sc, pl_ds])
    assert len(concat) == 4 + 2 * 12

    # First 4 items are from gt_sc → tag="sc"
    for i in range(4):
        item = concat[i]
        assert len(item) == 6
        assert item[4] == "sc"
        assert item[5] == 1.0
    # Next items are from pl_ds → tag="pl"
    for i in range(4, len(concat)):
        item = concat[i]
        assert len(item) == 6
        assert item[4] == "pl"


def test_collate_5tuple_input_produces_4tuple_output(tmp_path):
    """Tucker default: ScDS (5-tuple) → _collate_drop_extras → 4-tuple."""
    sc_ds = _make_sc_ds(n_rows=4)
    loader = DataLoader(sc_ds, batch_size=4, collate_fn=_collate_drop_extras)
    batch = next(iter(loader))
    assert len(batch) == 4, f"Tucker collate must produce 4-tuple, got {len(batch)}"
    wavs, labels, wts, mks = batch
    assert wavs.shape == (4, 1, WINDOW_SAMPLES)
    assert labels.shape == (4, NUM_CLASSES)
    assert wts.shape == (4, NUM_CLASSES)
    assert mks.shape == (4, NUM_CLASSES)


def test_collate_6tuple_input_produces_5tuple_output(tmp_path):
    """PL path: PLDS (6-tuple) → _collate_drop_extras → 5-tuple with loss_weights (B,)."""
    pl_df, species = _make_pl_df(n_files=1)
    pl_ds = PLDS(
        pl_df=pl_df,
        species_list=species,
        soundscape_dir=tmp_path,
        loss_weight=0.5,
        audio_loader=_make_stub_loader(),
    )
    loader = DataLoader(pl_ds, batch_size=4, collate_fn=_collate_drop_extras)
    batch = next(iter(loader))
    assert len(batch) == 5, f"PL collate must produce 5-tuple, got {len(batch)}"
    wavs, labels, wts, mks, loss_weights = batch
    assert wavs.shape == (4, 1, WINDOW_SAMPLES)
    assert loss_weights.shape == (
        4,
    ), f"loss_weights must be (B,) per-sample, got {loss_weights.shape}"
    assert torch.all(loss_weights == 0.5)


def test_collate_6tuple_via_concat_focal_sc_pl(tmp_path):
    """End-to-end: ConcatDataset of GT-wrapped + PL → collate → 5-tuple
    with mixed loss_weights (1.0 for GT, custom for PL)."""
    pl_df, species = _make_pl_df(n_files=1)
    pl_ds = PLDS(
        pl_df=pl_df,
        species_list=species,
        soundscape_dir=tmp_path,
        loss_weight=0.3,
        audio_loader=_make_stub_loader(),
    )
    gt_sc = GTDatasetWrapper(_make_sc_ds(n_rows=4), loss_weight=1.0)
    concat = ConcatDataset([gt_sc, pl_ds])

    loader = DataLoader(concat, batch_size=4, shuffle=False, collate_fn=_collate_drop_extras)
    # First batch: all from gt_sc (loss_weight=1.0)
    batch = next(iter(loader))
    assert len(batch) == 5
    assert torch.all(batch[4] == 1.0)
