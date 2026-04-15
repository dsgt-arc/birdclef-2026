"""Fetch species→family from iNaturalist API for BC2026 taxonomy.

Uses inat_taxon_id from taxonomy.csv. iNat /v1/taxa/{id} returns an
`ancestors` list with each ancestor's rank; we pull the one with
rank == 'family'. Polite throttling: ~1 req/sec.

Output: families.csv with columns
  primary_label, class_name, family, family_id, genus

Run on a login node — light HTTP, no compute. ~4 min.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TAXONOMY = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv"
)
OUT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "birdclef_107"
    / "assets"
    / "families.csv"
)
API = "https://api.inaturalist.org/v1/taxa/{id}"
DELAY = 0.6  # seconds between calls — iNat asks for <1 req/sec


def fetch(taxon_id: str) -> dict:
    req = urllib.request.Request(
        API.format(id=taxon_id),
        headers={"User-Agent": "birdclef-2026-research/0.1"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def extract(payload: dict) -> tuple[str, str, str]:
    """Return (family_name, family_id, genus_name) or empty strings if missing."""
    results = payload.get("results") or []
    if not results:
        return "", "", ""
    ancestors = results[0].get("ancestors") or []
    fam_name = fam_id = genus = ""
    for a in ancestors:
        if a.get("rank") == "family":
            fam_name = a.get("name", "")
            fam_id = str(a.get("id", ""))
        elif a.get("rank") == "genus":
            genus = a.get("name", "")
    # If the species itself is the genus row (rare), still keep it.
    return fam_name, fam_id, genus


def main() -> int:
    rows = list(csv.DictReader(TAXONOMY.open()))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    failed: list[str] = []
    for i, row in enumerate(rows, 1):
        taxon_id = row["inat_taxon_id"].strip()
        primary = row["primary_label"]
        cls = row.get("class_name", "")
        try:
            payload = fetch(taxon_id)
            fam, fam_id, genus = extract(payload)
            if not fam:
                failed.append(f"{primary} (taxon {taxon_id}): no family in ancestors")
        except urllib.error.HTTPError as e:
            failed.append(f"{primary} (taxon {taxon_id}): HTTP {e.code}")
            fam = fam_id = genus = ""
        except Exception as e:
            failed.append(f"{primary} (taxon {taxon_id}): {type(e).__name__} {e}")
            fam = fam_id = genus = ""

        out_rows.append(
            {
                "primary_label": primary,
                "class_name": cls,
                "family": fam,
                "family_id": fam_id,
                "genus": genus,
            }
        )
        if i % 25 == 0 or i == len(rows):
            print(f"  [{i:3d}/{len(rows)}] {primary} → {fam or '???'}", flush=True)
        time.sleep(DELAY)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["primary_label", "class_name", "family", "family_id", "genus"]
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows → {OUT}")
    if failed:
        print(f"Failed lookups ({len(failed)}):")
        for line in failed:
            print(f"  - {line}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
