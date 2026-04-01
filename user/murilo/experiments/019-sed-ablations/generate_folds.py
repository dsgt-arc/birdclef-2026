#!/usr/bin/env python3
"""Generate 4-fold cross-validation assignments using ttahara's MultiLabelStratifiedGroupKFold.

Produces fold_assignments.json with fold IDs (0-3) for every focal recording and
soundscape file. All segments from the same soundscape file share a fold (grouped
by filename). Focal recordings are grouped by audio_id (filename stem).

Usage:
    python generate_folds.py
    python generate_folds.py --data-dir /path/to/birdclef-2026
    python generate_folds.py --n-folds 4 --seed 1086

Output:
    fold_assignments.json in the same directory as this script.

Reference: user/murilo/notes/ttahara-hgnetv2-b0-training.py (lines 310-400)
"""

import argparse
import ast
import json
import random
from collections import Counter
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

DEFAULT_DATA_DIR = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026"


# ============================================================================
# ttahara's MultiLabelStratifiedGroupKFold (copied verbatim)
# Source: kaggle kernels pull ttahara/birdclef-2026-hgnetv2-b0-baseline-training
# ============================================================================


class MultiLabelStratifiedGroupKFold:
    def __init__(self, n_splits: int, random_state: int):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, label_arr: np.ndarray, gid_arr: np.ndarray):
        """Create multi-label stratified group kfold indices.

        Reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation

        Args:
            label_arr: (n_train, n_class) multi-hot label vectors.
            gid_arr: (n_train,) group ID for each sample.

        Yields:
            (train_indices, val_indices) for each fold.
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        start_time = time()
        n_train, n_class = label_arr.shape
        gid_unique = sorted(set(gid_arr))
        n_group = len(gid_unique)

        gid2aid = dict(zip(gid_unique, range(n_group), strict=False))
        aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

        cnts_by_class = label_arr.sum(axis=0)
        # Mask species with 0 samples to avoid division by zero (NaN breaks comparisons)
        class_mask = cnts_by_class > 0

        col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
        cnts_by_group = (
            coo_matrix((np.ones(len(label_arr)), (row, col)))
            .dot(coo_matrix(label_arr))
            .toarray()
            .astype(int)
        )
        del col, row
        cnts_by_fold = np.zeros((self.n_splits, n_class), int)

        groups_by_fold = [[] for _ in range(self.n_splits)]
        group_and_cnts = list(enumerate(cnts_by_group))
        np.random.shuffle(group_and_cnts)
        print(f"  Preparation: {time() - start_time:.1f}s")

        for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for fid in range(self.n_splits):
                cnts_by_fold[fid] += cnt_by_g
                fold_eval = (
                    (cnts_by_fold[:, class_mask] / cnts_by_class[class_mask]).std(axis=0).mean()
                )
                cnts_by_fold[fid] -= cnt_by_g
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = fid
            cnts_by_fold[best_fold] += cnt_by_g
            groups_by_fold[best_fold].append(aid)

        print(f"  Assignment: {time() - start_time:.1f}s")

        idx_arr = np.arange(n_train)
        for fid in range(self.n_splits):
            val_groups = groups_by_fold[fid]
            val_indexs_bool = np.isin(aid_arr, val_groups)
            train_indexs = idx_arr[~val_indexs_bool]
            val_indexs = idx_arr[val_indexs_bool]
            print(
                f"  [fold {fid}] groups: train={n_group - len(val_groups)}, "
                f"val={len(val_groups)} | samples: train={len(train_indexs)}, "
                f"val={len(val_indexs)}"
            )
            yield train_indexs, val_indexs


def main():
    parser = argparse.ArgumentParser(description="Generate 4-fold CV assignments")
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR, help="BirdCLEF 2026 data directory"
    )
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1086, help="Random state (ttahara uses 1086)")
    parser.add_argument(
        "--output", type=str, default=None, help="Output path (default: fold_assignments.json)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = (
        Path(args.output) if args.output else Path(__file__).parent / "fold_assignments.json"
    )

    # ========================================================================
    # Load data
    # ========================================================================

    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    classes = taxonomy["primary_label"].values.tolist()
    label2idx = {label: idx for idx, label in enumerate(classes)}
    n_classes = len(classes)

    # Focal recordings
    focal_df = pd.read_csv(data_dir / "train.csv")
    print(f"Focal recordings: {len(focal_df)}")

    # Soundscape labels (deduplicated)
    sc_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    sc_df = sc_df.drop_duplicates().reset_index(drop=True)
    print(f"Soundscape segments: {len(sc_df)} (from {sc_df['filename'].nunique()} files)")

    # ========================================================================
    # Build combined DataFrame matching ttahara's format
    # ========================================================================

    # Focal: audio_id from filename, labels from primary + secondary
    focal_records = []
    for _, row in focal_df.iterrows():
        audio_id = Path(row["filename"]).stem
        secondary = (
            ast.literal_eval(row["secondary_labels"]) if pd.notna(row["secondary_labels"]) else []
        )
        all_labels = [row["primary_label"]] + secondary
        focal_records.append(
            {
                "audio_id": audio_id,
                "filename": row["filename"],
                "source": "focal",
                "labels": all_labels,
            }
        )
    focal_combined = pd.DataFrame(focal_records)

    # Soundscape: per-segment rows (matching ttahara), grouped by filename.
    # Each segment is a separate row for better stratification granularity,
    # but audio_id = filename stem so all segments from the same file share
    # a group and always end up in the same fold.
    sc_records = []
    for _, row in sc_df.iterrows():
        audio_id = Path(row["filename"]).stem
        sc_records.append(
            {
                "audio_id": audio_id,
                "filename": row["filename"],
                "source": "soundscape",
                "labels": [row["primary_label"]],
            }
        )
    sc_combined = pd.DataFrame(sc_records)

    combined = pd.concat([focal_combined, sc_combined], ignore_index=True)
    n_sc_files = sc_combined["audio_id"].nunique()
    print(
        f"Combined: {len(combined)} entries "
        f"({len(focal_combined)} focal + {len(sc_combined)} soundscape segments from {n_sc_files} files)"
    )
    print(f"Unique audio_ids (groups): {combined['audio_id'].nunique()}")

    # ========================================================================
    # Build multi-hot label array
    # ========================================================================

    labels_arr = np.zeros((len(combined), n_classes), dtype=np.float32)
    for idx, label_list in enumerate(combined["labels"]):
        for label in label_list:
            if label in label2idx:
                labels_arr[idx, label2idx[label]] = 1

    species_coverage = (labels_arr.sum(axis=0) > 0).sum()
    print(f"Species with ≥1 sample: {species_coverage}/{n_classes}")

    # ========================================================================
    # Run MultiLabelStratifiedGroupKFold
    # ========================================================================

    print(
        f"\nRunning MultiLabelStratifiedGroupKFold (n_splits={args.n_folds}, seed={args.seed})..."
    )
    splitter = MultiLabelStratifiedGroupKFold(n_splits=args.n_folds, random_state=args.seed)
    splits = list(splitter.split(labels_arr, combined["audio_id"].values))

    # Assign fold IDs
    fold_ids = np.full(len(combined), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(splits):
        fold_ids[val_idx] = fold_id

    assert (fold_ids >= 0).all(), "Some samples have no fold assignment"

    # ========================================================================
    # Build output JSON
    # ========================================================================

    focal_folds = {}
    soundscape_folds = {}

    for idx, row in combined.iterrows():
        fold = int(fold_ids[idx])
        if row["source"] == "focal":
            focal_folds[row["filename"]] = fold
        else:
            soundscape_folds[row["filename"]] = fold

    output = {
        "n_folds": args.n_folds,
        "random_state": args.seed,
        "n_focal": len(focal_folds),
        "n_soundscape_files": len(soundscape_folds),
        "focal": focal_folds,
        "soundscape": soundscape_folds,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # ========================================================================
    # Print summary
    # ========================================================================

    print("\nFold distribution:")
    focal_counter = Counter(focal_folds.values())
    sc_counter = Counter(soundscape_folds.values())
    for fold in range(args.n_folds):
        print(f"  Fold {fold}: {focal_counter[fold]} focal, {sc_counter[fold]} soundscape files")


if __name__ == "__main__":
    main()
