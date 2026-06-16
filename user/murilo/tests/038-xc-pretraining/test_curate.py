"""Exp 038 — BirdSet Neotropical curation logic tests (the methodical data-selection core).

Pins: Neotropical-occurring (>=min_recs in bbox) species are kept; non-Neotropical non-targets are
dropped; bird targets are ALWAYS kept (even if recorded outside the bbox); scientific-name fallback
matches targets whose ebird_code differs; per-species cap + min_recs + quality A/B filters apply;
missing targets are reported.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).parent.parent.parent / "experiments/038-xc-pretraining/scripts"
spec = importlib.util.spec_from_file_location(
    "curate_birdset", SCRIPTS / "curate_birdset_neotropical.py"
)
cb = importlib.util.module_from_spec(spec)
sys.modules["curate_birdset"] = cb
spec.loader.exec_module(cb)

INSIDE = {"lat": -16.0, "long": -56.0}  # Pantanal — inside the Neotropics bbox
OUTSIDE = {"lat": 50.0, "long": 10.0}  # Europe — outside


def _rows(code, n, *, where, quality="A", length=30.0, sciname="Sci sp"):
    return pd.DataFrame(
        [
            {
                "filepath": f"{code}/{i}.ogg",
                "ebird_code": code,
                "ebird_code_secondary": None,
                "lat": where["lat"],
                "long": where["long"],
                "length": length,
                "quality": quality,
                "sciname": sciname,
            }
            for i in range(n)
        ]
    )


def _meta():
    return pd.concat(
        [
            _rows("neo1", 15, where=INSIDE, quality="A"),  # Neotropical (>=10) → keep
            _rows("neo1", 5, where=INSIDE, quality="C"),  # quality C → filtered out
            _rows("neo2", 5, where=INSIDE, quality="A"),  # <10 in bbox, not target → drop
            _rows("outside1", 20, where=OUTSIDE, quality="A"),  # not Neotropical, not target → drop
            _rows("target1", 12, where=OUTSIDE, quality="A"),  # target by code, outside → keep
            _rows(
                "zzz_code", 11, where=OUTSIDE, quality="B", sciname="Target Sci"
            ),  # target by sciname → keep
            _rows("neo3", 600, where=INSIDE, quality="A"),  # Neotropical, cap → 500
        ],
        ignore_index=True,
    )


def _curate():
    return cb.curate(
        _meta(),
        target_codes={"target1", "targmissing"},  # targmissing absent from corpus
        target_scinames={"Target Sci"},
        cap=500,
        min_recs=10,
    )


def test_pool_membership():
    manifest, _ = _curate()
    assert set(manifest["ebird_code"].unique()) == {"neo1", "neo3", "target1", "zzz_code"}


def test_quality_filter():
    manifest, _ = _curate()
    assert manifest["quality"].isin(["A", "B"]).all()
    assert (manifest["ebird_code"] == "neo1").sum() == 15  # the 5 quality-C rows dropped


def test_cap_and_targets_kept():
    manifest, _ = _curate()
    counts = manifest.groupby("ebird_code").size()
    assert counts["neo3"] == 500  # capped from 600
    assert counts["target1"] == 12  # target kept despite outside bbox
    assert counts["zzz_code"] == 11  # sciname-fallback target kept


def test_report():
    _, report = _curate()
    assert report["n_species"] == 4
    assert report["missing_bird_targets"] == ["targmissing"]
    assert report["n_bird_targets_in_pool"] == 2  # target1 (code) + zzz_code (sciname)


def test_in_bbox():
    lat = pd.Series([-16.0, 50.0, 25.0])
    lon = pd.Series([-56.0, 10.0, -40.0])
    out = cb.in_bbox(lat, lon, cb.DEFAULT_BBOX)
    assert out.tolist() == [True, False, True]
