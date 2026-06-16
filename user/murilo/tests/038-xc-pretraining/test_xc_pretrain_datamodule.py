"""Exp 038 — XCPretrainDataModule tests (stratified per-species split, setup contract)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.data.xc_pretrain_dataset import XCPretrainDataModule  # noqa: E402


def _manifest(n_a=10, n_b=4, n_c=1):
    codes = ["a"] * n_a + ["b"] * n_b + ["c"] * n_c
    return pd.DataFrame(
        {
            "filepath": [f"XC{i}.ogg" for i in range(len(codes))],
            "ebird_code": codes,
            "ebird_code_secondary": [None] * len(codes),
            "length": [10] * len(codes),
        }
    )


def test_stratified_split_proportions_and_no_overlap():
    df = _manifest()  # a:10, b:4, c:1 -> val_fraction 0.2 -> a:2, b:1, c:0
    train, val = XCPretrainDataModule.stratified_split(df, val_fraction=0.2, seed=42)
    assert len(val) == 3
    assert len(train) == 12
    assert len(train) + len(val) == len(df)
    # no recording appears in both splits
    assert set(train["filepath"]) & set(val["filepath"]) == set()
    # singleton species 'c' stays entirely in train (round(0.2)=0)
    assert "c" not in set(val["ebird_code"])
    # val species are a subset of all species
    assert set(val["ebird_code"]) <= set(df["ebird_code"])


def test_stratified_split_seed_deterministic():
    df = _manifest()
    a = XCPretrainDataModule.stratified_split(df, 0.2, seed=7)[1]["filepath"].tolist()
    b = XCPretrainDataModule.stratified_split(df, 0.2, seed=7)[1]["filepath"].tolist()
    assert a == b


def test_setup_loads_vocab_and_splits(tmp_path):
    manifest = _manifest()
    manifest.to_parquet(tmp_path / "manifest.parquet")
    (tmp_path / "species_list.json").write_text(json.dumps(["a", "b", "c"]))

    dm = XCPretrainDataModule(
        manifest_path=tmp_path / "manifest.parquet",
        species_list_path=tmp_path / "species_list.json",
        audio_dir=tmp_path / "audio",
        require_audio=False,  # no audio on disk in this unit test
        val_fraction=0.2,
        num_workers=0,
        batch_size=4,
    )
    dm.setup()
    assert dm.num_classes == 3
    assert len(dm.train_df) + len(dm.val_df) == len(manifest)
    assert len(dm.val_df) == 3
    assert dm.species_to_idx == {"a": 0, "b": 1, "c": 2}
