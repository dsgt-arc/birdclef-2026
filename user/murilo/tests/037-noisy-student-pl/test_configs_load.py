"""Exp 037 — round-1 factorial config validation.

Replicates train.py's `_deep_merge` (load _common.yaml, deep-merge each cell on top) and pins,
for every `r1_*.yaml` cell, that the merged config encodes the intended factorial point:
KD regime (A/B/C → use_perch_distill / kd_detach_head), PL transform (raw/t2 parquet),
PL aug (augF/augT), and loss_type (bce/softauc). Also pins the corrected-recipe invariants
(mixsamp 50/50, loss_weight=1.0, exclude_labeled_sc, GT-only steps_per_epoch, 25 ep).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CFG = Path(__file__).parent.parent.parent / "experiments/037-noisy-student-pl/configs"
CELLS = sorted(CFG.glob("r1_*.yaml"))


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load(cell: Path) -> dict:
    base = yaml.safe_load((CFG / "_common.yaml").read_text())
    return _deep_merge(base, yaml.safe_load(cell.read_text()))


def _parse(name: str) -> dict:
    # r1_{KD}_{transform}_{augTag}[_softauc]
    parts = name.split("_")
    return {
        "kd": parts[1],
        "transform": parts[2],
        "aug": parts[3] == "augT",
        "softauc": parts[-1] == "softauc",
    }


def test_factorial_complete():
    # 3 KD × 2 transform × 2 aug = 12 soft-BCE + 3 SoftAUC (raw, augT) = 15
    assert len(CELLS) == 15, [c.name for c in CELLS]
    softauc = [c for c in CELLS if c.stem.endswith("softauc")]
    assert len(softauc) == 3


@pytest.mark.parametrize("cell", CELLS, ids=[c.stem for c in CELLS])
def test_cell_encodes_its_factorial_point(cell):
    spec = _parse(cell.stem)
    cfg = _load(cell)
    pl, tr = cfg["pseudo_label"], cfg["training"]

    # PL transform → parquet + invariants
    assert f"v037-ens-{spec['transform']}" in pl["path"]
    assert pl["path"].endswith("ensemble_pseudo_labels.parquet")
    assert pl["aug"] is spec["aug"]
    assert pl["sampler"] == "mixsamp" and pl["weight"] == 1.0 and pl["exclude_labeled_sc"] is True
    assert abs(pl["focal_share"] + pl["sc_share"] - pl["pl_share"]) < 1e-9  # ~50/50 GT:PL

    # KD regime
    expect_kd = spec["kd"] != "B"
    assert tr["use_perch_distill"] is expect_kd
    if spec["kd"] == "A":
        assert tr["kd_detach_head"] is True
    elif spec["kd"] == "C":
        assert tr["kd_detach_head"] is False

    # loss + training budget
    assert tr["loss_type"] == ("softauc" if spec["softauc"] else "bce")
    assert cfg["training"]["max_epochs"] == 25
    assert cfg["data"]["steps_per_epoch"] == 500  # GT-only-equivalent (not the 5× PL balloon)


def test_common_defaults():
    common = yaml.safe_load((CFG / "_common.yaml").read_text())
    assert common["pseudo_label"]["path"] is None  # must be overridden per cell
    assert common["pseudo_label"]["sampler"] == "mixsamp"
    assert common["pseudo_label"]["aug"] is False  # augF is the default/control
    assert common["training"]["kd_detach_head"] is True  # cell-A default
    assert common["training"]["loss_type"] == "bce"
