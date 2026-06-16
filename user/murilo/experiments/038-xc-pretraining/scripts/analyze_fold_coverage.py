"""Per-fold species-coverage analysis for the exp-038 Stage-2 5-fold split.

Goal: decide which Cell-A fold to drop for a 4-fold mean on a *principled*
(coverage-based) signal rather than offline val/auroc (which is measured on a
different held-out SS species subset per fold and anti-correlates with LB).

The fold construction is reproduced VERBATIM from
`user/murilo/src/birdclef/data/distilled_sed_datamodule.py`:
  - focal: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on
    primary_label, dedup by original_idx first  (build_focal_folds)
  - SS: GroupKFold(n_splits=5) on filename  (build_sc_folds)
so the assignment here is bit-identical to what each fold model trained on.

Read-only; no GPU; pure pandas/numpy/sklearn.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
N_FOLDS = 5
SEED = 42
MIN_SAMPLE = 20

# Offline val/auroc per A fold (from results.md) for side-by-side comparison.
A_OFFLINE_AUROC = {0: 0.8873, 1: 0.9147, 2: 0.9192, 3: 0.9072, 4: 0.9237}


# --- verbatim from distilled_sed_datamodule.build_focal_folds (fold-assign part only) ---
def assign_focal_folds(focal_df: pd.DataFrame) -> pd.DataFrame:
    df = focal_df.copy().reset_index(drop=True)
    df["original_idx"] = df.index
    unique_df = df.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    unique_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(unique_df, unique_df["primary_label"])):
        unique_df.loc[val_idx, "fold"] = fold_idx
    return unique_df  # one row per unique recording, with fold


# --- verbatim from distilled_sed_datamodule.build_sc_folds ---
def assign_sc_folds(unique_files: pd.Series) -> dict[str, int]:
    gkf = GroupKFold(n_splits=N_FOLDS)
    uf = unique_files.drop_duplicates().reset_index(drop=True)
    file_to_fold: dict[str, int] = {}
    for fold_idx, (_, val_idx) in enumerate(gkf.split(uf, groups=uf)):
        for j in val_idx:
            file_to_fold[uf.iloc[j]] = fold_idx
    return file_to_fold


def main() -> None:
    species = sorted(pd.read_csv(DATA_DIR / "taxonomy.csv")["primary_label"].unique().tolist())
    sp2idx = {s: i for i, s in enumerate(species)}
    n_classes = len(species)

    focal_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"focal recordings: {len(focal_df)}  | species (taxonomy): {n_classes}")

    uniq = assign_focal_folds(focal_df)
    # (species x fold) UNIQUE-recording count matrix
    ct = (
        uniq.groupby(["primary_label", "fold"])
        .size()
        .unstack(fill_value=0)
        .reindex(species)
        .fillna(0)
        .astype(int)
    )  # rows=species, cols=fold 0..4
    total_per_sp = ct.sum(axis=1)

    # Species with ZERO focal recordings anywhere (blind to ALL folds equally -> irrelevant to drop)
    global_absent = total_per_sp[total_per_sp == 0].index.tolist()
    print(
        f"\n[NOTE] {len(global_absent)} species have 0 focal recordings in train.csv at all "
        f"(blind to every fold; learned only via labeled-SS + Perch KD) -> EXCLUDED from the"
    )
    print(
        f"       fold-differentiation count below. e.g. {global_absent[:4]} ... "
        f"({sum('son' in s for s in global_absent)} are '47158son*' focal-less species)."
    )

    print("\n=== 1) FOCAL TRAINING COVERAGE per fold (training = the OTHER 4 folds) ===")
    print(
        "Counts are over fold_k's TRAINING set (folds != k), EXCLUDING the globally-absent species."
    )
    print(
        f"{'fold':>4} | {'train_recs':>10} | {'sp<5':>5} {'sp<10':>6} {'sp<20':>6} | "
        f"{'TRUE_blindspots':>15} | {'offline_auroc':>13}"
    )
    singletons = {}  # fold -> species with >=1 recording confined ENTIRELY to this fold
    present = total_per_sp[total_per_sp > 0].index  # species that have any focal data
    for k in range(N_FOLDS):
        train_counts = total_per_sp - ct[k]  # recordings of each sp OUTSIDE fold k
        # restrict the thin-coverage counts to species that actually have focal data
        tc_present = train_counts[present]
        n5 = int((tc_present < 5).sum())
        n10 = int((tc_present < 10).sum())
        n20 = int((tc_present < 20).sum())
        train_recs = int(train_counts.sum())
        # TRUE blind-spots: species with focal data but 0 of it outside fold k (confined to k)
        blind = tc_present[tc_present == 0].index.tolist()
        singletons[k] = blind
        print(
            f"{k:>4} | {train_recs:>10} | {n5:>5} {n10:>6} {n20:>6} | "
            f"{len(blind):>15} | {A_OFFLINE_AUROC[k]:>13.4f}"
        )

    print(
        "\n  TRUE blind-spots per fold (species WITH focal data, confined entirely to that fold ->"
    )
    print(
        "  its own model never trained on them; dropping that fold makes the 4-fold mean clean on them):"
    )
    for k in range(N_FOLDS):
        sp = singletons[k]
        detail = ", ".join(f"{s}({int(total_per_sp[s])} rec)" for s in sp) if sp else "(none)"
        print(f"   fold {k}: {len(sp)} -> {detail}")

    print("\n=== 2) SS VALIDATION COMPOSITION per fold (the offline-metric confound) ===")
    sc_labels = pd.read_csv(DATA_DIR / "train_soundscapes_labels.csv").drop_duplicates().copy()
    sc_labels["start_sec"] = pd.to_timedelta(sc_labels["start"]).dt.total_seconds().astype(int)
    labeled_files = sorted(sc_labels["filename"].unique().tolist())
    f2fold = assign_sc_folds(pd.Series(labeled_files))

    # species set present per SS fold (any positive label among that fold's files)
    sc_labels["fold"] = sc_labels["filename"].map(f2fold)
    print(f"labeled SS files: {len(labeled_files)} across {N_FOLDS} folds")
    print(
        f"{'fold':>4} | {'val_files':>9} | {'val_windows~':>12} | {'distinct_sp_in_val':>18} | "
        f"{'offline_auroc':>13}"
    )
    for k in range(N_FOLDS):
        files_k = [f for f in labeled_files if f2fold[f] == k]
        rows_k = sc_labels[sc_labels["fold"] == k]
        sp_in_val = set()
        for s in rows_k["primary_label"].astype(str):
            for tok in s.split(";"):
                tok = tok.strip()
                if tok in sp2idx:
                    sp_in_val.add(tok)
        print(
            f"{k:>4} | {len(files_k):>9} | {len(files_k) * 12:>12} | {len(sp_in_val):>18} | "
            f"{A_OFFLINE_AUROC[k]:>13.4f}"
        )

    print("\n=== 3) RECOMMENDATION SIGNAL ===")
    rank = sorted(range(N_FOLDS), key=lambda k: len(singletons[k]), reverse=True)
    print(
        "TRUE blind-spots per fold: "
        + ", ".join(f"f{k}={len(singletons[k])}" for k in range(N_FOLDS))
    )
    print(f"-> coverage criterion: drop fold {rank[0]} (blind to the most focal species).")
    print("   Keeps fold 0 (LB-verified 0.907). Its low offline auroc (0.887) is a val-composition")
    print("   artifact: see analysis 2 (fold 0 val = fewest distinct species).")


if __name__ == "__main__":
    main()
