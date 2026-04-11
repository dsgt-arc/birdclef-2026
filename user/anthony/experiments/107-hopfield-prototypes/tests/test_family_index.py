"""Phase A test 5 — docs/families.csv integrates cleanly with the model.

Ensures build_family_index returns a valid tensor on real species data and
that the resulting tensor is accepted by PrototypeRetrievalModel.family_shrinkage_loss
without shape or encoding errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from birdclef_107.family_utils import build_family_index
from birdclef_107.model import PrototypeRetrievalModel

REPO_EXP_DIR = Path(__file__).resolve().parents[1]
FAMILIES_CSV = REPO_EXP_DIR / "docs" / "families.csv"


def _all_species_from_csv() -> list[str]:
    import csv

    with open(FAMILIES_CSV, newline="") as f:
        return [row["primary_label"] for row in csv.DictReader(f)]


def test_families_csv_exists() -> None:
    assert FAMILIES_CSV.exists(), f"expected {FAMILIES_CSV} to exist"


def test_build_family_index_shape_and_range() -> None:
    species_list = _all_species_from_csv()
    assert len(species_list) == 234, (
        f"expected 234 species rows in families.csv, got {len(species_list)}"
    )

    idx = build_family_index(species_list, FAMILIES_CSV)
    assert idx.dtype == torch.long
    assert idx.shape == (234,)

    n_families = int(idx.max().item()) + 1
    # design.md specifies 67 families (including Insecta_unresolved pseudo-family).
    assert 60 <= n_families <= 80, (
        f"expected ~67 families, got {n_families}"
    )

    # Contiguous encoding: every integer in [0, n_families) is used.
    unique = torch.unique(idx).tolist()
    assert unique == list(range(n_families)), (
        f"family encoding is not contiguous: missing "
        f"{set(range(n_families)) - set(unique)}"
    )


def test_build_family_index_raises_on_missing_species(tmp_path) -> None:
    tiny_csv = tmp_path / "tiny.csv"
    tiny_csv.write_text("primary_label,family\nsp1,FamA\n")
    with pytest.raises(ValueError, match="missing from"):
        build_family_index(["sp1", "sp2"], tiny_csv)


def test_family_index_works_with_model_shrinkage_loss() -> None:
    species_list = _all_species_from_csv()
    idx = build_family_index(species_list, FAMILIES_CSV)

    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    loss = model.family_shrinkage_loss(idx)
    assert loss.shape == ()
    assert torch.isfinite(loss).all()
    assert float(loss.item()) >= 0.0
