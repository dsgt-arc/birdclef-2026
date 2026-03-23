#!/usr/bin/env python3
"""Sweep temperature scaling on exp 014 pseudo-label logits.

Finds the optimal temperature T that maximizes macro AUROC on the 66 labeled
soundscape files. Optionally regenerates pseudo-label parquets with the best T.

Temperature scaling divides logits by T before sigmoid:
    probs = sigmoid(logits / T)
T < 1.0 sharpens predictions (higher confidence), T > 1.0 softens them.

Usage:
    # Sweep only (prints AUROC table)
    python sweep_temperature.py

    # Sweep + regenerate parquets with best T
    python sweep_temperature.py --regenerate

    # Custom temperatures
    python sweep_temperature.py --temperatures 0.4 0.5 0.6 0.7 0.8 0.9 1.0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026"
DEFAULT_PSEUDO_DIR = (
    "/storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels"
    "/birdclef-2026/pcr-cnx-soundscape-v1"
)
DEFAULT_TEMPERATURES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_teacher_logits(logits_dir: Path) -> tuple[np.ndarray, list[tuple], list[str]]:
    """Load and average logits from both teacher models.

    Returns:
        mean_logits: (n_segments, 234) averaged logits
        seg_keys: list of (filename, end_sec) tuples
        species: list of 234 species names
    """
    npz_files = sorted(logits_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {logits_dir}")

    all_logits = []
    seg_keys = None
    species = None

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        logits = data["logits"]  # (n_segments, 234)
        keys = data["seg_keys"]  # (n_segments,) tuples
        sp = data["species"]  # (234,) species names

        all_logits.append(logits)

        if seg_keys is None:
            seg_keys = [tuple(k) for k in keys]
            species = list(sp)
        else:
            # Verify alignment
            current_keys = [tuple(k) for k in keys]
            assert current_keys == seg_keys, "Segment keys mismatch between teacher models"

    mean_logits = np.mean(all_logits, axis=0)
    logger.info(
        f"Loaded {len(npz_files)} teacher models, "
        f"{mean_logits.shape[0]} segments, {mean_logits.shape[1]} species"
    )
    return mean_logits, seg_keys, species


def load_ground_truth(data_dir: Path, species_list: list[str]) -> pd.DataFrame:
    """Load ground truth labels as binary matrix aligned with pseudo-label segments.

    Returns:
        DataFrame with columns [filename, end_sec, *species_list] (binary 0/1)
    """
    labels_csv = data_dir / "train_soundscapes_labels.csv"
    gt = pd.read_csv(labels_csv)

    def time_to_sec(t):
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    gt["end_sec"] = gt["end"].apply(time_to_sec)

    species_set = set(species_list)
    rows = []
    for (fn, end_sec), group in gt.groupby(["filename", "end_sec"]):
        labels = set()
        for label_str in group["primary_label"]:
            for sp in str(label_str).split(";"):
                sp = sp.strip()
                if sp in species_set:
                    labels.add(sp)
        row = {"filename": fn, "end_sec": end_sec}
        for sp in species_list:
            row[sp] = 1 if sp in labels else 0
        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute calibration-sensitive metrics for temperature sweep.

    AUROC is rank-based and invariant to temperature scaling. Instead we use:
    - Log loss (BCE): measures how close probabilities are to true labels
    - Brier score: mean squared error between probabilities and labels
    - Macro AUROC: included for reference (won't change with T)

    All metrics computed only over species with both positives and negatives.
    Lower is better for log_loss and brier_score.
    """
    valid_aurocs = []
    valid_log_losses = []
    valid_brier_scores = []
    n_evaluated = 0

    eps = 1e-7  # clamp probabilities for log loss stability

    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        col_prob = y_prob[:, i]
        n_pos = col_true.sum()
        n_neg = len(col_true) - n_pos

        if n_pos > 0 and n_neg > 0:
            n_evaluated += 1

            # AUROC (rank-based, for reference)
            valid_aurocs.append(roc_auc_score(col_true, col_prob))

            # Log loss / BCE (calibration-sensitive)
            p = np.clip(col_prob, eps, 1.0 - eps)
            ll = -np.mean(col_true * np.log(p) + (1 - col_true) * np.log(1 - p))
            valid_log_losses.append(ll)

            # Brier score (calibration-sensitive)
            bs = np.mean((col_prob - col_true) ** 2)
            valid_brier_scores.append(bs)

    return {
        "auroc": np.mean(valid_aurocs) if valid_aurocs else 0.0,
        "log_loss": np.mean(valid_log_losses) if valid_log_losses else float("inf"),
        "brier_score": np.mean(valid_brier_scores) if valid_brier_scores else float("inf"),
        "n_species": n_evaluated,
    }


def sweep(
    mean_logits: np.ndarray,
    seg_keys: list[tuple],
    species_list: list[str],
    gt_df: pd.DataFrame,
    temperatures: list[float],
) -> dict[float, dict[str, float]]:
    """Sweep temperature values and return {T: metrics_dict}."""

    # Align pseudo-label segments with ground truth
    gt_df = gt_df.set_index(["filename", "end_sec"])
    aligned_indices = []
    for i, (fn, end_sec) in enumerate(seg_keys):
        key = (fn, int(end_sec))
        if key in gt_df.index:
            aligned_indices.append(i)

    if not aligned_indices:
        raise ValueError("No segments aligned between logits and ground truth")

    aligned_logits = mean_logits[aligned_indices]

    # Build aligned ground truth matrix
    aligned_gt_rows = []
    for i in aligned_indices:
        fn, end_sec = seg_keys[i]
        key = (fn, int(end_sec))
        aligned_gt_rows.append(gt_df.loc[key][species_list].values)

    y_true = np.array(aligned_gt_rows, dtype=np.float32)
    logger.info(f"Aligned {len(aligned_indices)} segments for evaluation")

    results = {}
    for T in temperatures:
        probs = sigmoid(aligned_logits / T)
        metrics = compute_metrics(y_true, probs)
        results[T] = metrics
        logger.info(
            f"  T={T:.2f}  →  log_loss={metrics['log_loss']:.4f}  "
            f"brier={metrics['brier_score']:.4f}  auroc={metrics['auroc']:.4f}  "
            f"({metrics['n_species']} species)"
        )

    return results


def regenerate_parquets(
    logits_dir: Path,
    output_dir: Path,
    best_T: float,
    species_list: list[str],
):
    """Regenerate pseudo-label parquets from raw logits with temperature scaling."""
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(logits_dir.glob("*.npz"))
    # Group by shard (files may be like model_shard0.npz, model_shard1.npz)
    # We need to average across models for each shard

    # Detect shard pattern
    shard_files = {}  # shard_id -> [npz_paths]
    for f in npz_files:
        name = f.stem
        # Extract shard suffix if present (e.g., "protoclr-2026-soundscape_shard3")
        if "_shard" in name:
            shard_id = name.split("_shard")[1]
        else:
            shard_id = "none"

        if shard_id not in shard_files:
            shard_files[shard_id] = []
        shard_files[shard_id].append(f)

    total_segments = 0
    for shard_id, files in sorted(shard_files.items()):
        # Average logits across models for this shard
        all_logits = []
        seg_keys = None
        for f in files:
            data = np.load(f, allow_pickle=True)
            all_logits.append(data["logits"])
            if seg_keys is None:
                seg_keys = data["seg_keys"]

        mean_logits = np.mean(all_logits, axis=0)
        probs = sigmoid(mean_logits / best_T)

        # Build parquet
        rows = []
        for i, key in enumerate(seg_keys):
            fn, end_sec = tuple(key)
            row = {"filename": fn, "end_sec": int(end_sec)}
            for j, sp in enumerate(species_list):
                row[sp] = float(probs[i, j])
            rows.append(row)

        df = pd.DataFrame(rows)

        if shard_id == "none":
            out_path = output_dir / "pseudo_labels.parquet"
        else:
            out_path = output_dir / f"pseudo_labels_shard{shard_id}.parquet"

        df.to_parquet(out_path, index=False)
        total_segments += len(df)
        logger.info(f"  Wrote {out_path.name}: {len(df)} segments")

    logger.info(f"Regenerated {total_segments} total segments with T={best_T}")


def main():
    parser = argparse.ArgumentParser(description="Sweep temperature scaling for pseudo-labels")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to BirdCLEF 2026 data directory",
    )
    parser.add_argument(
        "--pseudo-dir",
        type=str,
        default=DEFAULT_PSEUDO_DIR,
        help="Path to pseudo-label run directory (contains labeled/ and unlabeled/ subdirs)",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=DEFAULT_TEMPERATURES,
        help="Temperature values to sweep",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate parquets for both splits with the best temperature",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for regenerated parquets (default: auto-named with T suffix)",
    )
    args = parser.parse_args()

    pseudo_dir = Path(args.pseudo_dir)
    data_dir = Path(args.data_dir)
    labeled_logits_dir = pseudo_dir / "labeled" / "logits"
    taxonomy_csv = data_dir / "taxonomy.csv"

    # Build species list from taxonomy
    taxonomy = pd.read_csv(taxonomy_csv)
    species_list = sorted(taxonomy["primary_label"].unique().tolist())
    logger.info(f"Species list: {len(species_list)} species")

    # Load teacher logits (labeled split)
    logger.info(f"Loading labeled logits from {labeled_logits_dir}")
    mean_logits, seg_keys, logit_species = load_teacher_logits(labeled_logits_dir)

    # Load ground truth
    logger.info(f"Loading ground truth from {data_dir}")
    gt_df = load_ground_truth(data_dir, species_list)
    logger.info(f"Ground truth: {len(gt_df)} segments")

    # Sweep temperatures
    logger.info(f"Sweeping temperatures: {args.temperatures}")
    results = sweep(mean_logits, seg_keys, species_list, gt_df, args.temperatures)

    # Print results table
    # Best T = lowest log loss (best calibration for training targets)
    best_T = min(results, key=lambda t: results[t]["log_loss"])
    baseline = results.get(1.0, {})

    print("\n" + "=" * 72)
    print("Temperature Scaling Sweep Results")
    print("=" * 72)
    print("Note: AUROC is rank-based and invariant to T. Log loss and Brier score")
    print("measure calibration — how close probabilities are to true labels.")
    print("Lower log_loss/brier = better calibrated pseudo-labels for training.")
    print("-" * 72)
    print(
        f"{'T':>6}  {'Log Loss':>10}  {'Δ LogLoss':>10}  {'Brier':>8}  {'AUROC':>8}  {'Species':>7}"
    )
    print("-" * 72)
    for T in sorted(results):
        m = results[T]
        delta = m["log_loss"] - baseline.get("log_loss", m["log_loss"])
        marker = " <-- best" if T == best_T else ""
        print(
            f"{T:>6.2f}  {m['log_loss']:>10.4f}  {delta:>+10.4f}  "
            f"{m['brier_score']:>8.4f}  {m['auroc']:>8.4f}  {m['n_species']:>7}{marker}"
        )
    print(
        f"\nBest temperature: T={best_T:.2f} "
        f"(log_loss={results[best_T]['log_loss']:.4f}, "
        f"brier={results[best_T]['brier_score']:.4f})"
    )

    # Regenerate parquets if requested
    if args.regenerate:
        if args.output_dir:
            output_base = Path(args.output_dir)
        else:
            output_base = pseudo_dir.parent / f"{pseudo_dir.name}-T{best_T}"

        logger.info(f"\nRegenerating parquets with T={best_T}")

        # Labeled split
        labeled_out = output_base / "labeled"
        logger.info(f"Regenerating labeled split → {labeled_out}")
        regenerate_parquets(labeled_logits_dir, labeled_out, best_T, species_list)

        # Unlabeled split
        unlabeled_logits_dir = pseudo_dir / "unlabeled" / "logits"
        if unlabeled_logits_dir.exists():
            unlabeled_out = output_base / "unlabeled"
            logger.info(f"Regenerating unlabeled split → {unlabeled_out}")
            regenerate_parquets(unlabeled_logits_dir, unlabeled_out, best_T, species_list)
        else:
            logger.warning(f"Unlabeled logits dir not found: {unlabeled_logits_dir}")

        print(f"\nRegenerated parquets saved to: {output_base}")
        print(f"Update config pseudo_labels.parquet_dir to: {output_base / 'unlabeled'}/")


if __name__ == "__main__":
    main()
