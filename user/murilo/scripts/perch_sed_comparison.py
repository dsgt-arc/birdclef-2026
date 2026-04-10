#!/usr/bin/env python3
"""Phase D: Perch vs SED Comparative EDA.

Compares Perch 234-class predictions (from Phase C) against SED v5j 4-fold
predictions on the same 127K unlabeled soundscape segments. Answers the 5
critical questions from inference EDA deep dive section 10.8.

Produces findings for sections 12-13 of the EDA deep dive doc.

Usage:
    python user/murilo/scripts/perch_sed_comparison.py

Requires: pandas, numpy, scipy, scikit-learn
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# ============================================================================
# Paths
# ============================================================================

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA = SHARED / "data" / "birdclef-2026"

SED_4FOLD = SHARED / "inference" / "birdclef-2026" / "hgnet-b0-sed-v5j-4fold" / "train_soundscapes"
PERCH_DIR = SHARED / "inference" / "birdclef-2026" / "perch-v2-tf" / "train_soundscapes"

META_COLS = ["filename", "source", "chunk_idx", "start_sec", "end_sec", "site", "hour_utc"]
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# ============================================================================
# Data loading
# ============================================================================


def load_data():
    """Load SED 4-fold + Perch predictions, aligned on the same segments."""
    print("Loading competition metadata...")
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    sc_labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()

    # Identify seen vs unseen species
    def parse_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    seen_species = set()
    for x in sc_labels["primary_label"]:
        seen_species.update(parse_labels(x))
    seen_species = sorted(c for c in seen_species if c in label_to_idx)
    unseen_species = sorted(c for c in PRIMARY_LABELS if c not in seen_species)

    seen_idx = np.array([label_to_idx[c] for c in seen_species])
    unseen_idx = np.array([label_to_idx[c] for c in unseen_species])

    # Labeled soundscape filenames
    labeled_files = set(sc_labels["filename"].unique())

    # --- Load SED 4-fold max-across-folds ---
    print("Loading SED 4-fold data (max-across-folds)...")
    fold_probs = []
    for fold in range(4):
        logits = pd.read_parquet(SED_4FOLD / f"fold{fold}" / "logits.parquet")
        logits = logits.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)
        sp_cols = [c for c in logits.columns if c not in META_COLS]
        fold_probs.append(sigmoid(logits[sp_cols].values.astype(np.float32)))

    # Max across folds
    sed_max = np.maximum.reduce(fold_probs)
    sed_meta = logits[["filename", "chunk_idx"]].copy()
    del fold_probs
    print(f"  SED max-folds: {sed_max.shape}")

    # --- Load Perch raw probs (fair comparison — no prior suppression) ---
    print("Loading Perch raw probs...")
    perch_raw = pd.read_parquet(PERCH_DIR / "perch_234_raw_probs.parquet")
    perch_raw = perch_raw.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)
    perch_raw_sp = [c for c in perch_raw.columns if c not in META_COLS]
    perch_raw_vals = perch_raw[perch_raw_sp].values.astype(np.float32)

    # Also load full pipeline probs
    print("Loading Perch full pipeline probs...")
    perch_full = pd.read_parquet(PERCH_DIR / "perch_234_probs.parquet")
    perch_full = perch_full.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)
    perch_full_sp = [c for c in perch_full.columns if c not in META_COLS]
    perch_full_vals = perch_full[perch_full_sp].values.astype(np.float32)

    # Verify alignment
    assert len(sed_meta) == len(
        perch_raw
    ), f"Row mismatch: SED={len(sed_meta)}, Perch={len(perch_raw)}"
    assert (sed_meta["filename"].values == perch_raw["filename"].values).all(), "Filename mismatch"

    # Parse site info from filenames
    sites = []
    for f in sed_meta["filename"].values:
        m = FNAME_RE.match(f)
        sites.append(m.group(2) if m else "UNK")
    sites = np.array(sites)

    # Unlabeled mask (exclude the 66 labeled files)
    unlabeled_mask = ~sed_meta["filename"].isin(labeled_files).values

    # Build ground truth matrix for labeled soundscapes (for D.10 validation)
    labeled_mask = sed_meta["filename"].isin(labeled_files).values
    N_CLASSES = len(PRIMARY_LABELS)

    # Parse multi-label soundscape labels into per-window multi-hot matrix
    sc_labels["primary_label"] = sc_labels["primary_label"].astype(str)
    sc_clean = (
        sc_labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(
            lambda s: sorted(
                {
                    lbl.strip()
                    for x in s
                    for lbl in (str(x).split(";") if pd.notna(x) else [])
                    if lbl.strip()
                }
            )
        )
        .reset_index(name="label_list")
    )
    sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    # Match to chunk_idx: chunk_idx=0 → end_sec=5, chunk_idx=1 → end_sec=10, etc.
    sc_clean["chunk_idx"] = (sc_clean["end_sec"] // 5) - 1

    Y_labeled = np.zeros((labeled_mask.sum(), N_CLASSES), dtype=np.uint8)
    labeled_meta = sed_meta[labeled_mask].reset_index(drop=True)
    for _, row in sc_clean.iterrows():
        match = labeled_meta[
            (labeled_meta["filename"] == row["filename"])
            & (labeled_meta["chunk_idx"] == row["chunk_idx"])
        ]
        if len(match) == 1:
            ridx = match.index[0]
            for lbl in row["label_list"]:
                if lbl in label_to_idx:
                    Y_labeled[ridx, label_to_idx[lbl]] = 1

    return {
        "sed_max": sed_max,
        "perch_raw": perch_raw_vals,
        "perch_full": perch_full_vals,
        "meta": sed_meta,
        "sites": sites,
        "unlabeled_mask": unlabeled_mask,
        "PRIMARY_LABELS": PRIMARY_LABELS,
        "label_to_idx": label_to_idx,
        "CLASS_NAME_MAP": CLASS_NAME_MAP,
        "seen_species": seen_species,
        "unseen_species": unseen_species,
        "seen_idx": seen_idx,
        "unseen_idx": unseen_idx,
        "labeled_files": labeled_files,
        "labeled_mask": labeled_mask,
        "Y_labeled": Y_labeled,
        "sp_cols": perch_raw_sp,
    }


# ============================================================================
# D.1: Coverage — SED-zero species that Perch covers
# ============================================================================


def d1_coverage(data):
    print("\n" + "=" * 70)
    print("D.1: Coverage — How many SED-zero species does Perch cover?")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]
    labels = data["PRIMARY_LABELS"]

    # SED max-folds coverage on unlabeled soundscapes (unseen species)
    sed_unlabeled = sed[mask][:, unseen_idx]
    perch_unlabeled = perch[mask][:, unseen_idx]

    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        sed_covered = (sed_unlabeled.max(axis=0) > thresh).sum()
        perch_covered = (perch_unlabeled.max(axis=0) > thresh).sum()

        sed_zero = sed_unlabeled.max(axis=0) <= thresh
        perch_rescues = ((perch_unlabeled.max(axis=0) > thresh) & sed_zero).sum()

        print(f"\n  @>{thresh}:")
        print(f"    SED max-folds:  {sed_covered}/{len(unseen_idx)} unseen species")
        print(f"    Perch raw:      {perch_covered}/{len(unseen_idx)} unseen species")
        print(f"    Perch rescues:  {perch_rescues} species SED missed")

    # Detailed rescue list at >0.5
    thresh = 0.5
    sed_zero_mask = sed_unlabeled.max(axis=0) <= thresh
    perch_above = perch_unlabeled.max(axis=0) > thresh
    rescued_species_idx = np.where(sed_zero_mask & perch_above)[0]

    unseen_labels = [labels[i] for i in unseen_idx]
    cn = data["CLASS_NAME_MAP"]

    print(f"\n  Species rescued by Perch at >{thresh} (SED=0, Perch>0):")
    rescued_data = []
    for ri in rescued_species_idx:
        sp = unseen_labels[ri]
        perch_max = perch_unlabeled[:, ri].max()
        perch_count = (perch_unlabeled[:, ri] > thresh).sum()
        rescued_data.append((sp, cn.get(sp, "?"), perch_max, perch_count))

    rescued_data.sort(key=lambda x: -x[2])
    for sp, taxon, pmax, pcnt in rescued_data[:20]:
        print(f"    {sp:15s} ({taxon:10s}): Perch max={pmax:.3f}, segments={pcnt}")
    if len(rescued_data) > 20:
        print(f"    ... and {len(rescued_data) - 20} more")

    return rescued_data


# ============================================================================
# D.2: Truly hard species — 34 species zero at >0.3 even with SED max-folds
# ============================================================================


def d2_truly_hard(data):
    print("\n" + "=" * 70)
    print("D.2: Truly hard species (SED zero at >0.3 even with max-folds)")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]
    labels = data["PRIMARY_LABELS"]
    cn = data["CLASS_NAME_MAP"]

    sed_unlabeled = sed[mask][:, unseen_idx]
    perch_unlabeled = perch[mask][:, unseen_idx]

    # Truly hard: SED max-folds max prob <= 0.3
    truly_hard_mask = sed_unlabeled.max(axis=0) <= 0.3
    n_truly_hard = truly_hard_mask.sum()
    print(f"\n  {n_truly_hard} truly hard species (SED max ≤0.3)")

    unseen_labels = [labels[i] for i in unseen_idx]

    # How many does Perch cover?
    for thresh in [0.1, 0.3, 0.5]:
        perch_covers = (perch_unlabeled[:, truly_hard_mask].max(axis=0) > thresh).sum()
        print(f"  Perch covers at >{thresh}: {perch_covers}/{n_truly_hard}")

    # Per-species detail
    print(f"\n  Per-species detail (all {n_truly_hard} truly hard):")
    hard_idx = np.where(truly_hard_mask)[0]
    rows = []
    for hi in hard_idx:
        sp = unseen_labels[hi]
        sed_m = sed_unlabeled[:, hi].max()
        perch_m = perch_unlabeled[:, hi].max()
        perch_cnt = (perch_unlabeled[:, hi] > 0.3).sum()
        rows.append((sp, cn.get(sp, "?"), sed_m, perch_m, perch_cnt))

    rows.sort(key=lambda x: -x[3])
    for sp, taxon, smax, pmax, pcnt in rows:
        marker = "✓" if pmax > 0.3 else " "
        print(
            f"    {marker} {sp:15s} ({taxon:10s}): SED={smax:.4f}, Perch={pmax:.4f}, Perch@0.3={pcnt}"
        )

    return rows


# ============================================================================
# D.3: Union coverage
# ============================================================================


def d3_union_coverage(data):
    print("\n" + "=" * 70)
    print("D.3: Union coverage — SED max-folds + Perch raw")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]

    sed_u = sed[mask][:, unseen_idx]
    perch_u = perch[mask][:, unseen_idx]

    # Union: per-species max(SED, Perch)
    union = np.maximum(sed_u, perch_u)

    n_unseen = len(unseen_idx)
    print(f"\n  {'Threshold':>10s} | {'SED':>5s} | {'Perch':>5s} | {'Union':>5s} | {'Gain':>5s}")
    print(f"  {'-'*10} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5}")
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = (sed_u.max(axis=0) > thresh).sum()
        p = (perch_u.max(axis=0) > thresh).sum()
        u = (union.max(axis=0) > thresh).sum()
        g = u - max(s, p)
        print(
            f"  >{thresh:>8.1f} | {s:>3d}/{n_unseen} | {p:>3d}/{n_unseen} | {u:>3d}/{n_unseen} | +{g:>3d}"
        )

    # Also report all 234 species
    print("\n  All 234 species:")
    sed_all = sed[mask]
    perch_all = perch[mask]
    union_all = np.maximum(sed_all, perch_all)

    print(f"  {'Threshold':>10s} | {'SED':>5s} | {'Perch':>5s} | {'Union':>5s}")
    print(f"  {'-'*10} | {'-'*5} | {'-'*5} | {'-'*5}")
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = (sed_all.max(axis=0) > thresh).sum()
        p = (perch_all.max(axis=0) > thresh).sum()
        u = (union_all.max(axis=0) > thresh).sum()
        print(f"  >{thresh:>8.1f} | {s:>3d}/234 | {p:>3d}/234 | {u:>3d}/234")


# ============================================================================
# D.4: Complementarity — Venn diagram
# ============================================================================


def d4_complementarity(data):
    print("\n" + "=" * 70)
    print("D.4: Complementarity — Perch-only, SED-only, both, neither")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]
    labels = data["PRIMARY_LABELS"]
    cn = data["CLASS_NAME_MAP"]

    sed_u = sed[mask][:, unseen_idx]
    perch_u = perch[mask][:, unseen_idx]
    unseen_labels = [labels[i] for i in unseen_idx]

    for thresh in [0.3, 0.5]:
        sed_has = sed_u.max(axis=0) > thresh
        perch_has = perch_u.max(axis=0) > thresh

        both = (sed_has & perch_has).sum()
        sed_only = (sed_has & ~perch_has).sum()
        perch_only = (~sed_has & perch_has).sum()
        neither = (~sed_has & ~perch_has).sum()

        print(f"\n  @>{thresh} (unseen species):")
        print(f"    Both:       {both}")
        print(f"    SED only:   {sed_only}")
        print(f"    Perch only: {perch_only}")
        print(f"    Neither:    {neither}")
        print(f"    Total:      {both + sed_only + perch_only + neither}")

        # List SED-only species (Perch can't detect)
        if sed_only > 0:
            sed_only_idx = np.where(sed_has & ~perch_has)[0]
            print(f"\n    SED-only species ({sed_only}):")
            for si in sed_only_idx[:10]:
                sp = unseen_labels[si]
                print(
                    f"      {sp} ({cn.get(sp, '?')}): SED={sed_u[:, si].max():.3f}, Perch={perch_u[:, si].max():.3f}"
                )

        # List Perch-only species (SED can't detect)
        if perch_only > 0:
            perch_only_idx = np.where(~sed_has & perch_has)[0]
            print(f"\n    Perch-only species ({perch_only}):")
            for pi in sorted(perch_only_idx, key=lambda i: -perch_u[:, i].max())[:15]:
                sp = unseen_labels[pi]
                print(
                    f"      {sp} ({cn.get(sp, '?')}): SED={sed_u[:, pi].max():.3f}, Perch={perch_u[:, pi].max():.3f}"
                )

        # List neither species
        if neither > 0 and thresh == 0.3:
            neither_idx = np.where(~sed_has & ~perch_has)[0]
            print(f"\n    Neither species ({neither}) — truly hard:")
            for ni in neither_idx:
                sp = unseen_labels[ni]
                print(
                    f"      {sp} ({cn.get(sp, '?')}): SED={sed_u[:, ni].max():.4f}, Perch={perch_u[:, ni].max():.4f}"
                )


# ============================================================================
# D.5: Per-species agreement
# ============================================================================


def d5_agreement(data):
    print("\n" + "=" * 70)
    print("D.5: Per-species agreement (both detect at >0.3)")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]
    labels = data["PRIMARY_LABELS"]
    cn = data["CLASS_NAME_MAP"]

    sed_u = sed[mask][:, unseen_idx]
    perch_u = perch[mask][:, unseen_idx]
    unseen_labels = [labels[i] for i in unseen_idx]

    # Species both detect at >0.3
    both_mask = (sed_u.max(axis=0) > 0.3) & (perch_u.max(axis=0) > 0.3)
    both_idx = np.where(both_mask)[0]
    print(f"\n  {len(both_idx)} species detected by both at >0.3")

    # Per-species Spearman correlation
    correlations = []
    for bi in both_idx:
        sp = unseen_labels[bi]
        rho, pval = stats.spearmanr(sed_u[:, bi], perch_u[:, bi])
        correlations.append(
            (sp, cn.get(sp, "?"), rho, pval, sed_u[:, bi].max(), perch_u[:, bi].max())
        )

    correlations.sort(key=lambda x: -x[2])

    rhos = [c[2] for c in correlations]
    print(
        f"  Spearman ρ: mean={np.mean(rhos):.3f}, median={np.median(rhos):.3f}, "
        f"min={min(rhos):.3f}, max={max(rhos):.3f}"
    )

    print("\n  Top 10 highest agreement:")
    for sp, taxon, rho, _, smax, pmax in correlations[:10]:
        print(f"    {sp:15s} ({taxon:10s}): ρ={rho:.3f}, SED={smax:.3f}, Perch={pmax:.3f}")

    print("\n  Bottom 10 lowest agreement:")
    for sp, taxon, rho, _, smax, pmax in correlations[-10:]:
        print(f"    {sp:15s} ({taxon:10s}): ρ={rho:.3f}, SED={smax:.3f}, Perch={pmax:.3f}")

    return correlations


# ============================================================================
# D.6: Per-site analysis
# ============================================================================


def d6_per_site(data):
    print("\n" + "=" * 70)
    print("D.6: Per-site analysis — labeled vs unlabeled-only sites")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    sites = data["sites"]
    unseen_idx = data["unseen_idx"]
    labeled_files = data["labeled_files"]

    # Which sites have labeled data?
    labeled_sites = set()
    for f in labeled_files:
        m = FNAME_RE.match(f)
        if m:
            labeled_sites.add(m.group(2))

    # Per-site coverage analysis on unlabeled segments
    unique_sites = sorted(set(sites[mask]))

    print(
        f"\n  {'Site':>5s} | {'Labeled':>7s} | {'Files':>6s} | {'SED >0.5':>9s} | {'Perch >0.5':>10s} | {'Union >0.5':>10s}"
    )
    print(f"  {'-'*5} | {'-'*7} | {'-'*6} | {'-'*9} | {'-'*10} | {'-'*10}")

    site_rows = []
    for site in unique_sites:
        site_mask = mask & (sites == site)
        n_files = len(set(data["meta"]["filename"][site_mask]))
        is_labeled = site in labeled_sites

        sed_site = sed[site_mask][:, unseen_idx]
        perch_site = perch[site_mask][:, unseen_idx]
        union_site = np.maximum(sed_site, perch_site)

        sed_sp = (sed_site.max(axis=0) > 0.5).sum()
        perch_sp = (perch_site.max(axis=0) > 0.5).sum()
        union_sp = (union_site.max(axis=0) > 0.5).sum()

        label_str = "YES" if is_labeled else "no"
        print(
            f"  {site:>5s} | {label_str:>7s} | {n_files:>6d} | {sed_sp:>4d}/159 | {perch_sp:>5d}/159 | {union_sp:>5d}/159"
        )
        site_rows.append((site, is_labeled, n_files, sed_sp, perch_sp, union_sp))

    # Aggregate: labeled vs unlabeled sites
    labeled_rows = [r for r in site_rows if r[1]]
    unlabeled_rows = [r for r in site_rows if not r[1]]

    print("\n  Aggregate:")
    l_sed = np.mean([r[3] for r in labeled_rows]) if labeled_rows else 0
    l_perch = np.mean([r[4] for r in labeled_rows]) if labeled_rows else 0
    u_sed = np.mean([r[3] for r in unlabeled_rows]) if unlabeled_rows else 0
    u_perch = np.mean([r[4] for r in unlabeled_rows]) if unlabeled_rows else 0
    print(
        f"    Labeled sites   ({len(labeled_rows):>2d}): SED mean={l_sed:.1f}, Perch mean={l_perch:.1f}"
    )
    print(
        f"    Unlabeled sites ({len(unlabeled_rows):>2d}): SED mean={u_sed:.1f}, Perch mean={u_perch:.1f}"
    )
    if u_perch > u_sed:
        print(f"    Perch advantage on new sites: +{u_perch - u_sed:.1f} species")

    return site_rows


# ============================================================================
# D.7: Ensemble teacher simulation
# ============================================================================


def d7_ensemble_teacher(data):
    print("\n" + "=" * 70)
    print("D.7: Ensemble teacher simulation")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]
    n_unseen = len(unseen_idx)

    sed_u = sed[mask][:, unseen_idx]
    perch_u = perch[mask][:, unseen_idx]

    # Strategy 1: max(SED, Perch) — equivalent to union
    union = np.maximum(sed_u, perch_u)

    # Strategy 2: weighted average at different weights
    print("\n  Strategy: w * Perch + (1-w) * SED_max")
    print(
        f"  {'Weight':>7s} | {'Coverage@0.3':>12s} | {'Coverage@0.5':>12s} | {'Pseudo-labels@0.5':>18s}"
    )
    print(f"  {'-'*7} | {'-'*12} | {'-'*12} | {'-'*18}")

    best_coverage = 0
    best_w = 0
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blended = w * perch_u + (1 - w) * sed_u
        cov3 = (blended.max(axis=0) > 0.3).sum()
        cov5 = (blended.max(axis=0) > 0.5).sum()
        pl5 = (blended > 0.5).sum()
        wlabel = f"{'SED only' if w == 0 else 'Perch only' if w == 1.0 else f'w={w:.1f}'}"
        print(f"  {wlabel:>7s} | {cov3:>5d}/{n_unseen} | {cov5:>5d}/{n_unseen} | {pl5:>10d}")
        if cov5 > best_coverage:
            best_coverage = cov5
            best_w = w

    # Strategy 3: max(SED, Perch) then power transform
    print("\n  Strategy: max(SED, Perch) + power transform")
    for power in [1.0, 1.5, 2.0]:
        transformed = union**power
        cov3 = (transformed.max(axis=0) > 0.3).sum()
        cov5 = (transformed.max(axis=0) > 0.5).sum()
        pl5 = (transformed > 0.5).sum()
        print(
            f"    P^{power}: coverage@0.3={cov3}/{n_unseen}, @0.5={cov5}/{n_unseen}, pseudo-labels@0.5={pl5}"
        )

    print(f"\n  Best weighted average: w={best_w:.1f} → {best_coverage}/{n_unseen} @>0.5")
    print(f"  Max(SED, Perch) union: {(union.max(axis=0) > 0.5).sum()}/{n_unseen} @>0.5")


# ============================================================================
# D.8: Taxonomy breakdown
# ============================================================================


def d8_taxonomy_breakdown(data):
    print("\n" + "=" * 70)
    print("D.8: Coverage by taxonomy (unseen species)")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]
    labels = data["PRIMARY_LABELS"]
    cn = data["CLASS_NAME_MAP"]

    sed_u = sed[mask][:, unseen_idx]
    perch_u = perch[mask][:, unseen_idx]
    unseen_labels = [labels[i] for i in unseen_idx]

    # Group unseen species by taxonomy
    taxa = {}
    for i, sp in enumerate(unseen_labels):
        t = cn.get(sp, "Unknown")
        taxa.setdefault(t, []).append(i)

    print(
        f"\n  {'Taxon':>12s} | {'N':>4s} | {'SED@0.3':>8s} | {'SED@0.5':>8s} | {'Perch@0.3':>9s} | {'Perch@0.5':>9s} | {'Neither@0.3':>11s}"
    )
    print(f"  {'-'*12} | {'-'*4} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*9} | {'-'*11}")

    for taxon in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
        idx = taxa.get(taxon, [])
        if not idx:
            continue
        n = len(idx)
        s3 = (sed_u[:, idx].max(axis=0) > 0.3).sum()
        s5 = (sed_u[:, idx].max(axis=0) > 0.5).sum()
        p3 = (perch_u[:, idx].max(axis=0) > 0.3).sum()
        p5 = (perch_u[:, idx].max(axis=0) > 0.5).sum()
        neither = ((sed_u[:, idx].max(axis=0) <= 0.3) & (perch_u[:, idx].max(axis=0) <= 0.3)).sum()
        print(
            f"  {taxon:>12s} | {n:>4d} | {s3:>4d}/{n:<3d} | {s5:>4d}/{n:<3d} | {p3:>5d}/{n:<3d} | {p5:>5d}/{n:<3d} | {neither:>5d}/{n:<3d}"
        )

    # Perch's Aves problem: does it distinguish species or predict all equally?
    aves_idx = taxa.get("Aves", [])
    if aves_idx:
        aves_perch = perch_u[:, aves_idx]
        print("\n  Aves detail (Perch calibration check):")
        print(f"    Mean prob across all Aves×segments: {aves_perch.mean():.4f}")
        print(
            f"    Std prob across species (per segment mean): {aves_perch.mean(axis=0).std():.4f}"
        )

        # Per-species mean prob — how flat is the distribution?
        sp_means = aves_perch.mean(axis=0)
        print(
            f"    Species mean prob: min={sp_means.min():.4f}, median={np.median(sp_means):.4f}, max={sp_means.max():.4f}"
        )
        print(f"    Species with mean prob >0.5: {(sp_means > 0.5).sum()}/{len(aves_idx)}")
        print(f"    Species with mean prob >0.9: {(sp_means > 0.9).sum()}/{len(aves_idx)}")


# ============================================================================
# D.9: Calibration quantification
# ============================================================================


def d9_calibration(data):
    print("\n" + "=" * 70)
    print("D.9: Calibration — species per segment and pseudo-label density")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]

    sed_u = sed[mask][:, unseen_idx]
    perch_u = perch[mask][:, unseen_idx]
    n_seg = sed_u.shape[0]
    n_unseen = len(unseen_idx)

    # Species per segment
    print("\n  Species predicted per segment (unseen only):")
    print(
        f"  {'Threshold':>10s} | {'SED mean':>9s} | {'SED max':>8s} | {'Perch mean':>11s} | {'Perch max':>10s}"
    )
    print(f"  {'-'*10} | {'-'*9} | {'-'*8} | {'-'*11} | {'-'*10}")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        s_per = (sed_u > thresh).sum(axis=1)
        p_per = (perch_u > thresh).sum(axis=1)
        print(
            f"  >{thresh:>8.1f} | {s_per.mean():>9.1f} | {s_per.max():>8d} | {p_per.mean():>11.1f} | {p_per.max():>10d}"
        )

    # Pseudo-label density per species
    print("\n  Pseudo-labels per species at >0.5 (unseen, unlabeled):")
    sed_segs = (sed_u > 0.5).sum(axis=0)
    perch_segs = (perch_u > 0.5).sum(axis=0)

    sed_covered = sed_segs[sed_segs > 0]
    perch_covered = perch_segs[perch_segs > 0]

    print(f"    SED ({len(sed_covered)} species with >0 segments):")
    if len(sed_covered) > 0:
        print(
            f"      min={sed_covered.min()}, median={int(np.median(sed_covered))}, "
            f"mean={sed_covered.mean():.0f}, max={sed_covered.max()}"
        )
        print(f"      Total pseudo-labels: {sed_segs.sum()}")

    print(f"    Perch ({len(perch_covered)} species with >0 segments):")
    if len(perch_covered) > 0:
        print(
            f"      min={perch_covered.min()}, median={int(np.median(perch_covered))}, "
            f"mean={perch_covered.mean():.0f}, max={perch_covered.max()}"
        )
        print(f"      Total pseudo-labels: {perch_segs.sum()}")
        # How many species are in >50% of segments?
        n_majority = (perch_segs > n_seg * 0.5).sum()
        n_near_all = (perch_segs > n_seg * 0.9).sum()
        print(f"      Species in >50% of segments: {n_majority}/{n_unseen}")
        print(f"      Species in >90% of segments: {n_near_all}/{n_unseen}")

    # Ratio comparison
    print("\n  Calibration ratio (Perch / SED):")
    print(
        f"    Total pseudo-labels: {perch_segs.sum()} / {sed_segs.sum()} = {perch_segs.sum() / max(sed_segs.sum(), 1):.0f}x"
    )
    print(
        f"    Mean species per segment: {(perch_u > 0.5).sum(axis=1).mean():.1f} / {(sed_u > 0.5).sum(axis=1).mean():.2f}"
    )


# ============================================================================
# D.10: Labeled soundscape validation
# ============================================================================


def d10_labeled_validation(data):
    print("\n" + "=" * 70)
    print("D.10: Labeled soundscape validation (75 known species)")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    lmask = data["labeled_mask"]
    Y = data["Y_labeled"]
    labels = data["PRIMARY_LABELS"]
    cn = data["CLASS_NAME_MAP"]
    seen_idx = data["seen_idx"]

    sed_labeled = sed[lmask][:, seen_idx]
    perch_labeled = perch[lmask][:, seen_idx]
    Y_seen = Y[:, seen_idx]

    n_seg = sed_labeled.shape[0]
    n_seen = len(seen_idx)
    seen_labels = [labels[i] for i in seen_idx]

    print(f"\n  Labeled segments: {n_seg}")
    print(f"  Seen species: {n_seen}")
    print(f"  Positive labels: {Y_seen.sum()} (across {n_seg} segments × {n_seen} species)")

    # Per-species AUC (skip species with <2 positives)
    sed_aucs = []
    perch_aucs = []
    species_detail = []
    for i in range(n_seen):
        n_pos = Y_seen[:, i].sum()
        if n_pos < 2 or n_pos >= n_seg - 1:
            continue
        sp = seen_labels[i]
        s_auc = roc_auc_score(Y_seen[:, i], sed_labeled[:, i])
        p_auc = roc_auc_score(Y_seen[:, i], perch_labeled[:, i])
        sed_aucs.append(s_auc)
        perch_aucs.append(p_auc)
        species_detail.append((sp, cn.get(sp, "?"), s_auc, p_auc, n_pos))

    print(f"\n  Species with enough data for AUC: {len(sed_aucs)}/{n_seen}")
    print("\n  Macro-AUC on labeled soundscapes:")
    print(f"    SED max-folds: {np.mean(sed_aucs):.4f}")
    print(f"    Perch raw:     {np.mean(perch_aucs):.4f}")

    # Precision at different thresholds
    print("\n  Precision (positive predictions that are correct):")
    print(
        f"  {'Threshold':>10s} | {'SED prec':>9s} | {'SED preds':>10s} | {'Perch prec':>10s} | {'Perch preds':>12s}"
    )
    print(f"  {'-'*10} | {'-'*9} | {'-'*10} | {'-'*10} | {'-'*12}")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        s_pred = sed_labeled > thresh
        p_pred = perch_labeled > thresh
        s_tp = (s_pred & (Y_seen == 1)).sum()
        p_tp = (p_pred & (Y_seen == 1)).sum()
        s_total = s_pred.sum()
        p_total = p_pred.sum()
        s_prec = s_tp / max(s_total, 1)
        p_prec = p_tp / max(p_total, 1)
        print(
            f"  >{thresh:>8.1f} | {s_prec:>8.4f} | {s_total:>10d} | {p_prec:>9.4f} | {p_total:>12d}"
        )

    # Species where Perch is better vs SED is better
    species_detail.sort(key=lambda x: x[3] - x[2])
    perch_wins = [(sp, t, sa, pa, n) for sp, t, sa, pa, n in species_detail if pa > sa + 0.01]
    sed_wins = [(sp, t, sa, pa, n) for sp, t, sa, pa, n in species_detail if sa > pa + 0.01]

    print(f"\n  SED better (AUC gap >0.01): {len(sed_wins)} species")
    print(f"  Perch better (AUC gap >0.01): {len(perch_wins)} species")

    if sed_wins:
        print("\n  Top 5 SED advantages:")
        for sp, taxon, sa, pa, n in sorted(sed_wins, key=lambda x: x[2] - x[3])[-5:][::-1]:
            print(
                f"    {sp:15s} ({taxon:10s}): SED={sa:.3f}, Perch={pa:.3f}, Δ={sa-pa:+.3f} (n={n})"
            )

    if perch_wins:
        print("\n  Top 5 Perch advantages:")
        for sp, taxon, sa, pa, n in sorted(perch_wins, key=lambda x: x[3] - x[2])[-5:][::-1]:
            print(
                f"    {sp:15s} ({taxon:10s}): SED={sa:.3f}, Perch={pa:.3f}, Δ={pa-sa:+.3f} (n={n})"
            )


# ============================================================================
# Summary
# ============================================================================


def summary(data):
    print("\n" + "=" * 70)
    print("SUMMARY: Key Phase D findings")
    print("=" * 70)

    sed = data["sed_max"]
    perch = data["perch_raw"]
    mask = data["unlabeled_mask"]
    unseen_idx = data["unseen_idx"]
    n_unseen = len(unseen_idx)

    sed_u = sed[mask][:, unseen_idx]
    perch_u = perch[mask][:, unseen_idx]
    union_u = np.maximum(sed_u, perch_u)

    # Key metrics at >0.5
    sed_cov = (sed_u.max(axis=0) > 0.5).sum()
    perch_cov = (perch_u.max(axis=0) > 0.5).sum()
    union_cov = (union_u.max(axis=0) > 0.5).sum()

    # SED zeros rescued by Perch
    sed_zero = sed_u.max(axis=0) <= 0.5
    perch_rescues = (sed_zero & (perch_u.max(axis=0) > 0.5)).sum()

    # Truly hard (SED ≤0.3)
    truly_hard = (sed_u.max(axis=0) <= 0.3).sum()
    perch_covers_hard = ((sed_u.max(axis=0) <= 0.3) & (perch_u.max(axis=0) > 0.3)).sum()

    # Neither at >0.3
    neither = ((sed_u.max(axis=0) <= 0.3) & (perch_u.max(axis=0) <= 0.3)).sum()

    print(f"""
  Coverage at >0.5 (unseen species):
    SED max-folds:     {sed_cov}/{n_unseen}
    Perch raw:         {perch_cov}/{n_unseen}
    Union:             {union_cov}/{n_unseen}
    Perch rescues:     {perch_rescues} (SED=0, Perch>0.5)

  Truly hard species (SED ≤0.3):
    Total:             {truly_hard}
    Perch covers @0.3: {perch_covers_hard}/{truly_hard}
    Neither @0.3:      {neither}

  Success criteria:
    ≥120/159 unseen at >0.3: {(union_u.max(axis=0) > 0.3).sum()}/159 {'✓ PASS' if (union_u.max(axis=0) > 0.3).sum() >= 120 else '✗ FAIL'}
    Perch covers 30+ of 50 SED-zero: {perch_rescues} {'✓ PASS' if perch_rescues >= 30 else '✗ FAIL'}
""")


# ============================================================================
# Main
# ============================================================================


def main():
    data = load_data()
    d1_coverage(data)
    d2_truly_hard(data)
    d3_union_coverage(data)
    d4_complementarity(data)
    d5_agreement(data)
    d6_per_site(data)
    d7_ensemble_teacher(data)
    d8_taxonomy_breakdown(data)
    d9_calibration(data)
    d10_labeled_validation(data)
    summary(data)

    print("\nDone — update notes/2026-04-09-inference-eda-deep-dive.md sections 12-13")


if __name__ == "__main__":
    main()
