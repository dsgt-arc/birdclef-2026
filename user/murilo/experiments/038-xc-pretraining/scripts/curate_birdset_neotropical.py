"""Exp 038 Stage 0 — curate a broad Neotropical bird subset of BirdSet-XCL (NO GPU).

Pretrain pool = {species occurring in the Neotropics (>= MIN_RECS quality-A/B recordings inside a
South/Central-America lat/long bbox)} ∪ {the BirdCLEF-2026 bird targets (162 Aves), mapped via
ebird_code, scientific-name fallback}. Per-species cap ~500 (quality A>B, then longer first); drop
species < MIN_RECS. We KEEP breadth (Neotropical, ~1.5-3k sp) — NOT narrow-to-234. Emits a file
manifest + species_list + size report so a later step selectively downloads ONLY this audio (skip
the full 480 GB + the slow XC API crawl).

Two parts:
  - `curate(meta_df, target_codes, target_scinames, bbox, cap, min_recs, qualities)` — PURE PANDAS,
    unit-tested. The methodical data-selection logic lives here.
  - `load_birdset_metadata(...)` — HF metadata loader (Murilo runs + verifies field names on PACE
    with internet). See its docstring for the metadata-only caveat.

Usage (Murilo, on PACE w/ internet; smoke first):
    python scripts/curate_birdset_neotropical.py --smoke 2000      # verify schema + counts, no full pull
    python scripts/curate_birdset_neotropical.py                   # full metadata pass → manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Neotropics bbox (Central + South America). Pantanal (~ -16..-20 lat, -55..-58 long) is well inside.
DEFAULT_BBOX = {"lat_min": -56.0, "lat_max": 30.0, "long_min": -120.0, "long_max": -33.0}
DEFAULT_CAP = 500
DEFAULT_MIN_RECS = 10
DEFAULT_QUALITIES = ("A", "B")
META_COLS = [
    "filepath",
    "ebird_code",
    "ebird_code_secondary",
    "lat",
    "long",
    "length",
    "quality",
    "sciname",
]


def in_bbox(lat: pd.Series, lon: pd.Series, bbox: dict) -> pd.Series:
    return lat.between(bbox["lat_min"], bbox["lat_max"]) & lon.between(
        bbox["long_min"], bbox["long_max"]
    )


def curate(
    meta_df: pd.DataFrame,
    target_codes: set[str],
    target_scinames: set[str] | None = None,
    bbox: dict | None = None,
    cap: int = DEFAULT_CAP,
    min_recs: int = DEFAULT_MIN_RECS,
    qualities: tuple[str, ...] = DEFAULT_QUALITIES,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Return (manifest_df, report). meta_df needs: ebird_code, lat, long, quality, length, filepath
    (+ optional sciname for fallback target matching). PURE — no I/O."""
    bbox = bbox or DEFAULT_BBOX
    df = meta_df.copy()

    # 1) quality filter (A/B)
    df = df[df["quality"].isin(qualities)].reset_index(drop=True)

    # 2) Neotropical-OCCURRING species: >= min_recs geo-located (A/B) recordings inside the bbox.
    geo = df.dropna(subset=["lat", "long"])
    inside = geo[in_bbox(geo["lat"], geo["long"], bbox)]
    neo_counts = inside.groupby("ebird_code").size()
    neo_species = set(neo_counts[neo_counts >= min_recs].index)

    # 3) targets present in the corpus (by ebird_code, then scientific-name fallback)
    present_codes = set(df["ebird_code"].unique())
    target_in_corpus = set(target_codes) & present_codes
    if target_scinames and "sciname" in df.columns:
        sci_hit = set(df[df["sciname"].isin(target_scinames)]["ebird_code"].unique())
        target_in_corpus |= sci_hit
    missing_targets = set(target_codes) - target_in_corpus  # bird targets BirdSet lacks

    # 4) pool = Neotropical-occurring ∪ targets; keep ALL recordings of pool species (max data)
    pool_species = neo_species | target_in_corpus
    pool = df[df["ebird_code"].isin(pool_species)].copy()

    # 5) per-species cap: quality A before B, then longer recordings first; drop species < min_recs
    pool["_q"] = pool["quality"].map({q: i for i, q in enumerate(qualities)}).fillna(len(qualities))
    pool = pool.sort_values(["ebird_code", "_q", "length"], ascending=[True, True, False])
    manifest = pool.groupby("ebird_code", group_keys=False).head(cap).drop(columns="_q")
    kept_counts = manifest.groupby("ebird_code").size()
    keep_species = set(kept_counts[kept_counts >= min_recs].index)
    manifest = manifest[manifest["ebird_code"].isin(keep_species)].reset_index(drop=True)

    total_sec = float(manifest["length"].fillna(0).sum())
    report = {
        "n_species": int(manifest["ebird_code"].nunique()),
        "n_recordings": int(len(manifest)),
        "total_hours": round(total_sec / 3600.0, 1),
        "n_neotropical_species": len(neo_species),
        "n_bird_targets_in_pool": len(target_in_corpus & keep_species),
        "n_bird_targets_total": len(target_codes),
        "missing_bird_targets": sorted(missing_targets)[:50],
        "bbox": bbox,
        "cap": cap,
        "min_recs": min_recs,
        "qualities": list(qualities),
    }
    return manifest, report


def load_birdset_metadata(smoke: int | None = None) -> pd.DataFrame:
    """Load BirdSet-XCL METADATA into a DataFrame with the META_COLS schema.

    *** Murilo: verify on PACE (internet) with --smoke first. ***
    BirdSet (`DBD-research-group/BirdSet`, config "XCL") rows carry: ebird_code, ebird_code_secondary,
    lat, long, length, quality, filepath, + audio. We need metadata ONLY (no 480 GB audio):
      - PREFERRED: if the HF repo ships a metadata-only parquet/csv, read that directly (fast).
      - FALLBACK (below): stream the dataset and collect metadata fields WITHOUT touching audio['array']
        (the Audio feature decodes lazily). If parquet inlines audio bytes this still pulls shards —
        in that case switch to the metadata-only file. Confirm field names match META_COLS on smoke.
    Genus/Species (2021 eBird taxonomy) → `sciname` for the target scientific-name fallback.
    """
    from datasets import load_dataset

    ds = load_dataset("DBD-research-group/BirdSet", "XCL", split="train", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        if smoke is not None and i >= smoke:
            break
        sci = None
        if r.get("Genus") and r.get("Species"):
            sci = f"{r['Genus']} {r['Species']}".strip()
        rows.append(
            {
                "filepath": r.get("filepath"),
                "ebird_code": r.get("ebird_code"),
                "ebird_code_secondary": r.get("ebird_code_secondary"),
                "lat": r.get("lat"),
                "long": r.get("long"),
                "length": r.get("length"),
                "quality": r.get("quality"),
                "sciname": sci,
            }
        )
    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    return df


def load_birdclef_bird_targets(taxonomy_csv: Path) -> tuple[set[str], set[str]]:
    """Return (target_ebird_codes, target_scinames) for the 162 Aves targets."""
    tax = pd.read_csv(taxonomy_csv)
    aves = tax[tax["class_name"] == "Aves"]
    codes = set(aves["primary_label"].astype(str))  # BirdCLEF bird primary_label ≈ eBird code
    scinames = set(aves["scientific_name"].astype(str)) if "scientific_name" in aves else set()
    return codes, scinames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--smoke",
        type=int,
        default=None,
        help="Load only first N BirdSet rows (verify schema/counts).",
    )
    ap.add_argument(
        "--taxonomy",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/storage/project/ps-clef2026_birdclef-0/shared/pretrain/birdset-neotropical"),
    )
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--min-recs", type=int, default=DEFAULT_MIN_RECS)
    args = ap.parse_args()

    codes, scinames = load_birdclef_bird_targets(args.taxonomy)
    print(f"BirdCLEF-2026 bird targets: {len(codes)} ebird codes, {len(scinames)} scientific names")
    print(f"Loading BirdSet-XCL metadata{f' (smoke {args.smoke})' if args.smoke else ''}...")
    meta = load_birdset_metadata(smoke=args.smoke)
    print(f"  loaded {len(meta):,} rows; columns: {list(meta.columns)}")

    manifest, report = curate(meta, codes, scinames, cap=args.cap, min_recs=args.min_recs)
    print("\n=== curation report ===")
    print(json.dumps({k: v for k, v in report.items() if k != "missing_bird_targets"}, indent=2))
    print(
        f"missing bird targets ({len(report['missing_bird_targets'])} shown): {report['missing_bird_targets']}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out_dir / "manifest.parquet", index=False)
    (args.out_dir / "species_list.json").write_text(
        json.dumps(sorted(manifest["ebird_code"].unique().tolist()), indent=2)
    )
    (args.out_dir / "curation_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"\nWrote manifest ({len(manifest):,} rows, {report['n_species']} sp, {report['total_hours']}h) → {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
