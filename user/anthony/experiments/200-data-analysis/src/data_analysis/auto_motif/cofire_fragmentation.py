"""Label-free co-firing fragmentation guardrail for auto-motif families.

When a detector over-segments one recurring call into several "families",
those families re-detect the *same call instants*: within a recording their
detections are temporally coincident. Genuinely distinct call types fire at
different times. So the fraction of a family's detections that coincide (within
one event duration) with another family's is a label-free fragmentation
signal — no curation, no Perch, no template introspection.

Method (one established-technique mapping, no tuned constants):

* coincidence window ``eps`` = the corpus **median detection duration** for
  that (species, mode) — the natural single-event timescale, derived from the
  data, not picked.
* pairwise co-fire ``C[A,B]`` = fraction of A's detections within ``eps`` of
  some B detection in the *same recording*; symmetrised ``S = max(C, Cᵀ)``.
* collapse families by average-linkage agglomeration on distance ``1 − S``,
  cut at ``1 − cofire_threshold``. ``cofire_threshold = 0.5`` is the natural
  *majority-coincidence* boundary (two families merge iff most of one's
  detections land on the other's calls), not a fitted knob.
* ``frac_distinct = effective_groups / n_families`` ∈ (0, 1]: ``1.0`` = no
  fragmentation (every family is its own co-firing group); low = the family
  count is mostly spurious over-segmentation.

This is the per-species safety signal the breadth-gated mode selector lacked:
it flags exactly the species where a mode's family richness is fragmentation
(see concept doc §v4 — fusfly1 0.27 is the worst case; 6/12 gabor_stack-rich
species show *zero* collapse, i.e. genuine richness).

CLI::

    python -m data_analysis.auto_motif.cofire_fragmentation                  # gabor_stack-ok set
    python -m data_analysis.auto_motif.cofire_fragmentation --band-mode free --cluster-mode global
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from data_analysis.auto_motif.batch import (
    AUTO_MOTIF_DIR,
    _batch_dir,
    _mode_tag,
    _species_dir,
)

logger = logging.getLogger(__name__)

COFIRE_THRESHOLD = 0.5  # majority-coincidence merge boundary (not tuned)


def cofire_collapse(
    events: pl.DataFrame, *, cofire_threshold: float = COFIRE_THRESHOLD
) -> dict:
    """Collapse motif families by temporal co-firing; return the guardrail.

    Expects columns ``motif_cluster, filename, peak_time_sec, start_sec,
    end_sec``. ``eps`` is the median detection duration (data-derived).
    """
    families = sorted(events["motif_cluster"].unique().to_list())
    n = len(families)
    eps = float((events["end_sec"] - events["start_sec"]).median() or 0.0)
    if n < 2:
        return {
            "n_families": n,
            "effective_groups": n,
            "frac_distinct": 1.0,
            "mean_cofire": float("nan"),
            "eps_sec": round(eps, 4),
            "groups": [[families[0]]] if n else [],
        }

    times: dict[int, dict[str, np.ndarray]] = {}
    for c in families:
        rec: dict[str, list[float]] = {}
        for r in (
            events.filter(pl.col("motif_cluster") == c)
            .select(["filename", "peak_time_sec"])
            .iter_rows(named=True)
        ):
            rec.setdefault(r["filename"], []).append(r["peak_time_sec"])
        times[c] = {k: np.sort(np.asarray(v)) for k, v in rec.items()}

    co = np.zeros((n, n))
    for i, a in enumerate(families):
        for j, b in enumerate(families):
            if i == j:
                continue
            hit = tot = 0
            for fn, arr in times[a].items():
                tot += len(arr)
                bt = times[b].get(fn)
                if bt is None:
                    continue
                for x in arr:
                    k = np.searchsorted(bt, x)
                    if any(
                        0 <= q < len(bt) and abs(bt[q] - x) <= eps
                        for q in (k - 1, k)
                    ):
                        hit += 1
            co[i, j] = hit / max(tot, 1)

    sym = np.maximum(co, co.T)
    dist = 1.0 - sym
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(
        link, t=1.0 - cofire_threshold, criterion="distance"
    )
    groups: dict[int, list[int]] = {}
    for fam, g in zip(families, labels):
        groups.setdefault(int(g), []).append(fam)
    eye = np.eye(n, dtype=bool)
    return {
        "n_families": n,
        "effective_groups": len(groups),
        "frac_distinct": round(len(groups) / n, 3),
        "mean_cofire": round(float(sym[~eye].mean()), 3),
        "eps_sec": round(eps, 4),
        "groups": sorted(groups.values(), key=len, reverse=True),
    }


def _gabor_ok_labels() -> list[str]:
    """Labels whose gabor_stack batch succeeded (the rich-family candidates)."""
    mpath = _batch_dir(_mode_tag("free", "gabor_stack")) / "manifest.csv"
    m = pl.read_csv(mpath, infer_schema_length=0)
    return m.filter(pl.col("status") == "ok")["label"].to_list()


def build_table(
    band_mode: str, cluster_mode: str, labels: list[str]
) -> pl.DataFrame:
    tag = _mode_tag(band_mode, cluster_mode)
    rows = []
    for label in labels:
        nms = (
            _species_dir(label, tag)
            / "tables"
            / "template_match_events_nms.csv"
        )
        if not nms.exists():
            continue
        r = cofire_collapse(pl.read_csv(nms))
        rows.append(
            dict(
                species=label,
                band_mode=band_mode,
                cluster_mode=cluster_mode,
                n_families=r["n_families"],
                effective_groups=r["effective_groups"],
                frac_distinct=r["frac_distinct"],
                mean_cofire=r["mean_cofire"],
                eps_sec=r["eps_sec"],
                # readable, csv-safe: groups as "1094|1119,1933,..."
                groups=" | ".join(
                    ",".join(str(c) for c in g) for g in r["groups"]
                ),
            )
        )
    return pl.DataFrame(rows).sort("frac_distinct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band-mode", default="free")
    parser.add_argument("--cluster-mode", default="gabor_stack")
    parser.add_argument(
        "--labels",
        default=None,
        help="comma-separated; default = the band/cluster batch's ok species",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    elif (args.band_mode, args.cluster_mode) == ("free", "gabor_stack"):
        labels = _gabor_ok_labels()
    else:
        raise SystemExit("--labels required for non-gabor_stack modes")

    tab = build_table(args.band_mode, args.cluster_mode, labels)
    tag = _mode_tag(args.band_mode, args.cluster_mode)
    out = AUTO_MOTIF_DIR / tag / "_cofire_fragmentation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.write_csv(out)

    with pl.Config(tbl_rows=-1, fmt_str_lengths=60):
        print(tab)
    multi = tab.filter(pl.col("n_families") > 1)
    if multi.height:
        logger.info(
            "%d multi-family species: median frac_distinct=%.2f; "
            "severe collapse (<=0.5): %s; zero collapse (==1.0): %s",
            multi.height,
            float(multi["frac_distinct"].median()),
            multi.filter(pl.col("frac_distinct") <= 0.5)["species"].to_list(),
            multi.filter(pl.col("frac_distinct") == 1.0)["species"].to_list(),
        )
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
