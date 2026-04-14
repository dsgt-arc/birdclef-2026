"""Family coverage + long-tail analysis for exp 107 prereqs.

Inputs:
  - families.csv  (built by fetch_families.py)
  - taxonomy.csv
  - train.csv  (focal recordings)
  - train_soundscapes_labels.csv

Outputs (printed):
  1. Per-class species counts
  2. Family size distribution (with the 25 47158son## insects rolled
     into pseudo-family "Insecta_unresolved")
  3. Families with ≥3 species (where hierarchical regularization is
     actually meaningful)
  4. Orphaned species (no focal data) and whether a family sibling
     has focal data — this is the regime where the L_family partial-
     pooling shrinkage in the model provides any orphan rescue
     (family-level detection, not species-level discrimination)
  5. Labeled SC site counts (LOSO fold inventory)
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
EXP = Path(__file__).resolve().parent.parent
FAM_CSV = EXP / "docs" / "families.csv"


def load_families() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with FAM_CSV.open() as f:
        for row in csv.DictReader(f):
            fam = row["family"].strip()
            # Roll unresolved insect morphotypes into a class-level pseudo-family
            if not fam and row["primary_label"].startswith("47158son"):
                fam = "Insecta_unresolved"
            elif not fam:
                fam = f"_unknown_{row['primary_label']}"
            row["family_resolved"] = fam
            out[row["primary_label"]] = row
    return out


def load_focal_counts() -> Counter[str]:
    """Number of focal recordings per primary_label."""
    c: Counter[str] = Counter()
    with (DATA / "train.csv").open() as f:
        for row in csv.DictReader(f):
            c[row["primary_label"]] += 1
    return c


def load_sc_site_windows() -> Counter[str]:
    c: Counter[str] = Counter()
    with (DATA / "train_soundscapes_labels.csv").open() as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            # BC2026_Train_NNNN_S##_...
            try:
                site = fn.split("_S")[1].split("_")[0]
                c[site] += 1
            except IndexError:
                pass
    return c


def main() -> None:
    fams = load_families()
    focal = load_focal_counts()
    sc_sites = load_sc_site_windows()

    print("=" * 70)
    print("EXP 107 PREFLIGHT — Family hierarchy & long-tail coverage")
    print("=" * 70)

    # 1. Per-class
    by_class = Counter(r["class_name"] for r in fams.values())
    print("\n[1] Species per class (n=234):")
    for cls, n in by_class.most_common():
        print(f"  {cls:12s} {n:4d}")

    # 2. Family size distribution
    fam_to_species: dict[str, list[str]] = defaultdict(list)
    for sp, r in fams.items():
        fam_to_species[r["family_resolved"]].append(sp)

    fam_sizes = [len(v) for v in fam_to_species.values()]
    print(f"\n[2] Families: {len(fam_to_species)} total")
    print(f"  size distribution: min={min(fam_sizes)} max={max(fam_sizes)} "
          f"mean={sum(fam_sizes)/len(fam_sizes):.2f}")
    size_dist = Counter(fam_sizes)
    print("  size : #families")
    for sz in sorted(size_dist):
        print(f"  {sz:4d} : {size_dist[sz]}")

    # 3. Families with ≥3 species — where hierarchy is meaningful
    big_fams = {f: v for f, v in fam_to_species.items() if len(v) >= 3}
    species_in_big_fams = sum(len(v) for v in big_fams.values())
    print(f"\n[3] Families with ≥3 species: {len(big_fams)} "
          f"(covering {species_in_big_fams}/234 species = "
          f"{100*species_in_big_fams/234:.0f}%)")
    print("  Top 15 by size:")
    for f, v in sorted(big_fams.items(), key=lambda kv: -len(kv[1]))[:15]:
        cls = fams[v[0]]["class_name"]
        print(f"    {f:25s} {cls:10s} n={len(v)}")

    # 4. Orphaned species + family-sibling rescue
    orphan_species = [sp for sp in fams if focal.get(sp, 0) == 0]
    print(f"\n[4] Orphaned species (zero focal recordings): {len(orphan_species)}")
    rescue_buckets = {"rescuable_via_family_sibling": [], "truly_orphan": []}
    for sp in orphan_species:
        fam = fams[sp]["family_resolved"]
        siblings = [s for s in fam_to_species[fam] if s != sp]
        sib_with_focal = [s for s in siblings if focal.get(s, 0) > 0]
        if sib_with_focal:
            rescue_buckets["rescuable_via_family_sibling"].append((sp, fam, len(sib_with_focal)))
        else:
            rescue_buckets["truly_orphan"].append((sp, fam))
    print(f"  rescuable via family sibling: "
          f"{len(rescue_buckets['rescuable_via_family_sibling'])}")
    print(f"  truly orphan (no sibling with focal): "
          f"{len(rescue_buckets['truly_orphan'])}")
    print("  Truly orphan species:")
    for sp, fam in rescue_buckets["truly_orphan"]:
        cls = fams[sp]["class_name"]
        print(f"    {sp:15s} {cls:10s} family={fam}")
    print("  Rescuable orphans (sample):")
    for sp, fam, n in rescue_buckets["rescuable_via_family_sibling"][:10]:
        cls = fams[sp]["class_name"]
        print(f"    {sp:15s} {cls:10s} family={fam:25s} siblings_w_focal={n}")

    # 5. Long-tail focal counts by class
    print("\n[5] Focal data distribution by class (median, p10, p90):")
    by_class_focal: dict[str, list[int]] = defaultdict(list)
    for sp, r in fams.items():
        by_class_focal[r["class_name"]].append(focal.get(sp, 0))
    for cls in sorted(by_class_focal):
        vals = sorted(by_class_focal[cls])
        n = len(vals)
        p10 = vals[max(0, n // 10)]
        p50 = vals[n // 2]
        p90 = vals[min(n - 1, (9 * n) // 10)]
        zeros = sum(1 for v in vals if v == 0)
        print(f"  {cls:10s} n={n:3d}  p10={p10:5d}  p50={p50:5d}  p90={p90:5d}  "
              f"zero_focal={zeros}")

    # 6. SC site coverage (LOSO fold inventory)
    print(f"\n[6] Labeled SC windows per site (LOSO fold inventory):")
    print(f"  total labeled rows: {sum(sc_sites.values())}")
    print(f"  sites: {len(sc_sites)}")
    for site, n in sorted(sc_sites.items(), key=lambda kv: -kv[1]):
        marker = "*" if n >= 20 else " "
        print(f"  {marker} S{site}: {n}")
    qualifying = sum(1 for n in sc_sites.values() if n >= 20)
    print(f"  sites with ≥20 windows: {qualifying}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
