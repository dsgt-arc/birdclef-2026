"""Exp 037 — round-1 config inheritance + the corrected-recipe knobs.

Replicates train.py's `_deep_merge` (load _common.yaml, deep-merge the cell on top) and pins the
knobs that distinguish this experiment from exp 033's failures:
  - pseudo_label.path is set + points at the cell's parquet (raw vs t2)
  - pseudo_label.sampler matches the cell (mixsamp 50/50 vs uniform natural)
  - pseudo_label.weight == 1.0 (NOT exp 025's 0.5), exclude_labeled_sc == True
  - KD on (use_perch_distill, alpha_distill=1.0); 50/50 shares sum to ~1.0
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CFG = Path(__file__).parent.parent.parent / "experiments/037-noisy-student-pl/configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load(cell: str) -> dict:
    base = yaml.safe_load((CFG / "_common.yaml").read_text())
    override = yaml.safe_load((CFG / cell).read_text())
    return _deep_merge(base, override)


CELLS = {
    "round1_raw_5050.yaml": ("v037-ens-raw", "mixsamp"),
    "round1_raw_natural.yaml": ("v037-ens-raw", "uniform"),
    "round1_t2_5050.yaml": ("v037-ens-t2", "mixsamp"),
    "round1_t2_natural.yaml": ("v037-ens-t2", "uniform"),
}


@pytest.mark.parametrize("cell,expected", CELLS.items())
def test_cell_pl_config(cell, expected):
    parquet_tag, sampler = expected
    cfg = _load(cell)
    pl = cfg["pseudo_label"]
    assert pl["path"] is not None and parquet_tag in pl["path"]
    assert pl["path"].endswith("ensemble_pseudo_labels.parquet")
    assert pl["sampler"] == sampler
    assert pl["weight"] == 1.0  # Antoine 1:1, NOT exp 025's 0.5 confound
    assert pl["exclude_labeled_sc"] is True
    assert cfg["run_name"].startswith("2026-exp037-r1-")


def test_kd_on_and_shares_sum():
    cfg = _load("round1_raw_5050.yaml")
    assert cfg["training"]["use_perch_distill"] is True
    assert cfg["training"]["alpha_distill"] == 1.0
    pl = cfg["pseudo_label"]
    total = pl["focal_share"] + pl["sc_share"] + pl["pl_share"]
    assert abs(total - 1.0) < 1e-9
    assert abs((pl["focal_share"] + pl["sc_share"]) - pl["pl_share"]) < 1e-9  # ~50/50 GT:PL


def test_common_is_not_runnable_sentinel():
    # _common has path=null → must be overridden per cell (train.py would no-op PL otherwise)
    common = yaml.safe_load((CFG / "_common.yaml").read_text())
    assert common["pseudo_label"]["path"] is None
    assert common["pseudo_label"]["sampler"] == "mixsamp"
