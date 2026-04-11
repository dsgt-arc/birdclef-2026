"""Build the species → family index tensor consumed by ``L_family``.

The raw mapping lives in ``docs/families.csv`` (built by
``scripts/fetch_families.py`` via the iNat API). This module reads that CSV,
aligns to a canonical 234-species ordering, and returns a contiguous
integer encoding suitable for ``PrototypeRetrievalModel.family_shrinkage_loss``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch


#: Pseudo-family bucket for Insecta morphotypes whose taxonomy doesn't
#: resolve to a real family via iNat. Design doc §Label preparation calls
#: this out explicitly: the 25 ``47158son01..25`` sonotypes all collapse to
#: a single shared ``Insecta_unresolved`` centroid.
INSECTA_UNRESOLVED = "Insecta_unresolved"


def _load_species_to_family(families_csv: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(families_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"primary_label", "family"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{families_csv} is missing required columns: {sorted(missing)}"
            )
        has_class = "class_name" in (reader.fieldnames or [])
        for row in reader:
            species = row["primary_label"].strip()
            family = row["family"].strip()
            class_name = row["class_name"].strip() if has_class else ""

            if not species:
                raise ValueError(
                    f"Empty primary_label in {families_csv}: {row}"
                )

            # Insecta morphotypes with no iNat family resolution collapse
            # to the shared pseudo-family centroid (see design.md).
            if not family:
                if class_name == "Insecta":
                    family = INSECTA_UNRESOLVED
                else:
                    raise ValueError(
                        f"Empty family for non-Insecta species in {families_csv}: {row}"
                    )

            if species in mapping and mapping[species] != family:
                raise ValueError(
                    f"Species {species} has conflicting families "
                    f"{mapping[species]!r} vs {family!r} in {families_csv}"
                )
            mapping[species] = family
    return mapping


def build_family_index(
    species_list: list[str],
    families_csv: str | Path,
) -> torch.LongTensor:
    """Return a ``(n_species,)`` long tensor of contiguous family indices.

    Args:
        species_list: the canonical ordering of species used as the
            ``n_species`` dimension of ``M_species``. Typically
            ``birdclef_106.data_loading.build_species_list()``.
        families_csv: path to the ``primary_label,family,...`` CSV.

    Returns:
        A ``torch.LongTensor`` of shape ``(len(species_list),)`` with values
        in ``[0, n_families)``. The encoding is stable: families are
        assigned integer ids in the order they are first seen in
        ``species_list``.

    Raises:
        ValueError: if any species in ``species_list`` is missing from the
            CSV, or if the CSV is malformed.
    """
    families_csv = Path(families_csv)
    if not families_csv.exists():
        raise FileNotFoundError(f"families CSV not found: {families_csv}")

    species_to_family = _load_species_to_family(families_csv)

    missing = [s for s in species_list if s not in species_to_family]
    if missing:
        raise ValueError(
            f"{len(missing)} species missing from {families_csv}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    family_to_id: dict[str, int] = {}
    ids: list[int] = []
    for species in species_list:
        family = species_to_family[species]
        if family not in family_to_id:
            family_to_id[family] = len(family_to_id)
        ids.append(family_to_id[family])

    return torch.tensor(ids, dtype=torch.long)
