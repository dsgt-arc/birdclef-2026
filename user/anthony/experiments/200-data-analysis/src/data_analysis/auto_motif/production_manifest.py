"""Consolidate the materialized auto-motif batch artifacts for the full
soundscape∩train subset into ONE production manifest.

Pure table-join over on-disk batch manifests + per-species metadata — **no
recompute**. Routing is the conservative, documented 3-way scheme (see
``docs/concepts/decouple-frequency-localization-from-focal-selection.md`` §v4):

* ``consensus`` status ``ok``  -> production = ``consensus``/``global`` (the
  validated narrowband default).
* ``consensus`` failed (the broadband/FM-blind species) -> production =
  ``free``/``global``: the robust floor that never fails. ``gabor_stack`` is
  carried as the *opportunistic alternative* (richer multi-motif where it
  succeeds) but is **not** auto-selected: the per-species free+global↔
  gabor_stack selector is not independently validated (Perch-embedding
  corroboration 7/13 ≈ chance; 74113 is a concrete false positive).

``member_bg_dprime`` is carried as an informational quality proxy only — it is
a *weak* predictor of independent separation (Spearman ~0.43); do not rank on
it. The ``indep_perch_emb_auc_*`` columns are best-effort independent
corroboration from ``_gabor_stack/_perch_emb_validation.csv`` (left null if
that analysis artifact is absent). The one production row not independently
corroborated is 74113 (free+global d′ ok, Perch-emb AUC ≈ chance).

CLI::

    python -m data_analysis.auto_motif.production_manifest            # build + write
    python -m data_analysis.auto_motif.production_manifest --print    # also echo table
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from data_analysis.auto_motif.batch import (
    AUTO_MOTIF_DIR,
    _batch_dir,
    _mode_tag,
    _selected_labels,
    _species_dir,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = AUTO_MOTIF_DIR / "_production_manifest.csv"
# Independent (Perch-embedding) corroboration of the 17 blind species; an
# analysis artifact, enriched best-effort.
PERCH_EMB_CSV = AUTO_MOTIF_DIR / "_gabor_stack" / "_perch_emb_validation.csv"

_BLIND_NOTE = (
    "free+global default (robust, never fails); gabor_stack opportunistic — "
    "per-species selector NOT independently validated (concept doc §v4 "
    "validation)"
)


def _manifest_path(band_mode: str, cluster_mode: str) -> Path:
    return _batch_dir(_mode_tag(band_mode, cluster_mode)) / "manifest.csv"


def _max_dprime(band_mode: str, cluster_mode: str, label: str) -> float | None:
    """Max ``member_bg_dprime`` from a species' metadata, or ``None``.

    ``None`` is correct, not a gap, for 22967 (Marbled Frog): it predates the
    d′ screen and instead carries a true hand-curated ``curated_recovery``.
    """
    tag = _mode_tag(band_mode, cluster_mode)
    meta = _species_dir(label, tag) / "tables" / "auto_templates_metadata.csv"
    if not meta.exists():
        return None
    m = pl.read_csv(meta)
    if m.height == 0 or "member_bg_dprime" not in m.columns:
        return None
    v = m["member_bg_dprime"].max()
    return round(float(v), 3) if v is not None else None


def build_production_manifest(min_focal: int = 8) -> pl.DataFrame:
    """Join the consensus / free+global / gabor_stack batch artifacts into one
    conservative-routing production manifest (one row per subset species)."""
    cons_path = _manifest_path("consensus", "global")
    if not cons_path.exists():
        raise FileNotFoundError(
            f"consensus batch manifest missing: {cons_path}. Run "
            f"`python -m data_analysis.auto_motif.batch` first."
        )
    cons = pl.read_csv(cons_path)

    bf_by = {
        r["label"]: r
        for r in pl.read_csv(_manifest_path("free", "global")).iter_rows(named=True)
    }
    gs_by = {
        r["label"]: r
        for r in pl.read_csv(
            _manifest_path("free", "gabor_stack")
        ).iter_rows(named=True)
    }

    pe_auc: dict[tuple[str, str], float] = {}
    if PERCH_EMB_CSV.exists():
        for r in pl.read_csv(PERCH_EMB_CSV).iter_rows(named=True):
            if r["status"] == "ok":
                pe_auc[(r["species"], r["mode"])] = r["perch_emb_auc"]
    else:
        logger.warning(
            "Perch-emb validation absent (%s); indep_* columns left null",
            PERCH_EMB_CSV,
        )

    expected = set(_selected_labels(min_focal, None))
    got = set(cons["label"].to_list())
    if expected - got:
        raise ValueError(
            f"consensus manifest is missing selected species: "
            f"{sorted(expected - got)} — re-run the consensus batch"
        )

    rows = []
    for c in cons.iter_rows(named=True):
        label = c["label"]
        if c["status"] == "ok":
            rows.append(
                dict(
                    label=label,
                    common_name=c["common_name"],
                    class_name=c["class_name"],
                    n_focal=c["n_focal"],
                    routing_leg="consensus",
                    production_band_mode="consensus",
                    production_cluster_mode="global",
                    production_status="ok",
                    production_n_bands=c["n_bands"],
                    production_n_families=c["n_families"],
                    production_member_bg_dprime=_max_dprime(
                        "consensus", "global", label
                    ),
                    curated_recovery=c.get("curated_recovery"),
                    alt_gabor_stack_status=None,
                    alt_gabor_stack_n_families=None,
                    alt_gabor_stack_member_bg_dprime=None,
                    indep_perch_emb_auc_production=None,
                    indep_perch_emb_auc_gabor_stack=None,
                    artifact_path=str(
                        _species_dir(label, "").relative_to(
                            AUTO_MOTIF_DIR.parents[1]
                        )
                    ),
                    note="validated narrowband default",
                )
            )
        else:
            b = bf_by.get(label, {})
            g = gs_by.get(label, {})
            rows.append(
                dict(
                    label=label,
                    common_name=c["common_name"],
                    class_name=c["class_name"],
                    n_focal=c["n_focal"],
                    routing_leg="consensus_blind",
                    production_band_mode="free",
                    production_cluster_mode="global",
                    production_status=b.get("status"),
                    production_n_bands=b.get("n_bands"),
                    production_n_families=b.get("n_families"),
                    production_member_bg_dprime=_max_dprime(
                        "free", "global", label
                    ),
                    curated_recovery=None,
                    alt_gabor_stack_status=g.get("status"),
                    alt_gabor_stack_n_families=g.get("n_families"),
                    alt_gabor_stack_member_bg_dprime=_max_dprime(
                        "free", "gabor_stack", label
                    ),
                    indep_perch_emb_auc_production=pe_auc.get(
                        (label, "free+global")
                    ),
                    indep_perch_emb_auc_gabor_stack=pe_auc.get(
                        (label, "gabor_stack")
                    ),
                    artifact_path=str(
                        _species_dir(label, _mode_tag("free", "global"))
                        .relative_to(AUTO_MOTIF_DIR.parents[1])
                    ),
                    note=_BLIND_NOTE,
                )
            )

    return pl.DataFrame(rows).sort(["routing_leg", "n_focal"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-focal", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_PATH, help="manifest CSV path"
    )
    parser.add_argument(
        "--print", action="store_true", help="echo the full table to stdout"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tab = build_production_manifest(args.min_focal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tab.write_csv(args.out)

    by_leg = tab.group_by("routing_leg").len().sort("routing_leg")
    logger.info("%d species: %s", tab.height, by_leg.to_dicts())
    if args.print:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=40):
            print(tab)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
