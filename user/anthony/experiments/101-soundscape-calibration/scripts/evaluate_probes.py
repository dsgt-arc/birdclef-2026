"""Evaluate BirdNET/Perch linear probes on labeled soundscape embeddings.

Loads probe weights from exp 101 training, applies them to soundscape embeddings
from exp 010-2026, and evaluates against ground truth.

Usage:
    python evaluate_probes.py --output artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# --- Paths ---

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data/birdclef-2026"

EMBED_ROOT = Path("~/scratch/birdclef-2026/outputs/exp010-2026/labels").expanduser()
PROBE_DIR = Path("~/scratch/birdclef-2026/outputs/exp101/probes").expanduser()

EDA_ARTIFACTS = Path(__file__).resolve().parents[2] / "100-eda/artifacts"

# Also load Murilo's npz for comparison
INFERENCE_DIR = SHARED / "inference/birdclef-2026"
NPZ_PATHS = {
    "protoclr": INFERENCE_DIR / "protoclr-2026-soundscape/labeled.npz",
    "convnext": INFERENCE_DIR / "convnext-2026-soundscape/labeled.npz",
    "eat": INFERENCE_DIR / "eat-2026-soundscape/labeled.npz",
}

NO_FOCAL_AMPHIBIA = {"1491113", "25073", "517063"}
SONOTYPE_PREFIX = "47158son"


def load_ground_truth(eda_artifacts: Path) -> pd.DataFrame:
    """Load and build ground truth binary matrix."""
    labels = pd.read_parquet(eda_artifacts / "soundscape_labels.parquet")
    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    species_list = sorted(taxonomy["primary_label"].astype(str).unique())

    windows = labels[["filename", "start_sec", "end_sec"]].drop_duplicates().copy()
    windows["site"] = windows["filename"].str.extract(r"_(S\d+)_")

    # Build binary columns
    label_sets = (
        labels.groupby(["filename", "start_sec"])["primary_label"]
        .apply(set)
        .to_dict()
    )
    for sp in species_list:
        windows[sp] = [
            1 if sp in label_sets.get((f, s), set()) else 0
            for f, s in zip(windows["filename"], windows["start_sec"])
        ]

    return windows, species_list


def load_probe(probe_path: Path) -> dict:
    """Load probe weights from npz."""
    data = np.load(probe_path, allow_pickle=True)
    return {
        "coef": data["coef"],          # (n_species, embed_dim)
        "intercept": data["intercept"], # (n_species,)
        "species_list": [str(s) for s in data["species_list"]],
    }


def load_soundscape_embeddings(model: str) -> tuple[np.ndarray, list[str], list[float]]:
    """Load soundscape embeddings from parquet parts.

    Returns: (embeddings array, filenames, start_times)
    """
    embed_dir = EMBED_ROOT / model / "train_soundscapes" / "parts" / "embed"
    parts = sorted(embed_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files in {embed_dir}")

    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)

    filenames = df["file"].tolist()
    start_times = df["start_time"].tolist()

    # Embedding columns are the numeric ones (exclude file, start_time, end_time)
    meta_cols = {"file", "start_time", "end_time"}
    embed_cols = [c for c in df.columns if c not in meta_cols]
    embeddings = df[embed_cols].values.astype(np.float32)

    return embeddings, filenames, start_times


def apply_probe(
    embeddings: np.ndarray,
    probe: dict,
) -> np.ndarray:
    """Apply linear probe to embeddings -> sigmoid probabilities.

    Returns: (N, n_species) probabilities
    """
    # logits = X @ W^T + b
    logits = embeddings @ probe["coef"].T + probe["intercept"]
    # sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs


def aggregate_to_5s_windows(
    probs: np.ndarray,
    filenames: list[str],
    start_times: list[float],
    window_sec: float = 5.0,
    method: str = "max",
) -> tuple[np.ndarray, list[str], list[float]]:
    """Aggregate sub-window predictions to 5s windows.

    BirdNET has 1s steps, Perch has 5s steps. Aggregate clips that fall
    within each 5s window.
    """
    # Build window keys
    window_starts = [float(int(t / window_sec) * window_sec) for t in start_times]

    df = pd.DataFrame({
        "filename": filenames,
        "window_start": window_starts,
    })
    # Add prediction columns
    for i in range(probs.shape[1]):
        df[f"p_{i}"] = probs[:, i]

    pred_cols = [f"p_{i}" for i in range(probs.shape[1])]

    if method == "max":
        agg = df.groupby(["filename", "window_start"])[pred_cols].max()
    else:
        agg = df.groupby(["filename", "window_start"])[pred_cols].mean()

    agg = agg.reset_index()
    out_filenames = agg["filename"].tolist()
    out_starts = agg["window_start"].tolist()
    out_probs = agg[pred_cols].values.astype(np.float32)

    return out_probs, out_filenames, out_starts


def align_to_ground_truth(
    gt: pd.DataFrame,
    probs: np.ndarray,
    filenames: list[str],
    start_times: list[float],
    probe_species: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Align probe predictions to ground truth windows.

    Returns: (N_aligned, n_species) array aligned to gt rows, and species list.
    """
    pred_df = pd.DataFrame({
        "filename": [Path(f).name for f in filenames],
        "start_sec": start_times,
    })
    for i, sp in enumerate(probe_species):
        pred_df[f"pred_{sp}"] = probs[:, i]

    merged = gt.merge(pred_df, on=["filename", "start_sec"], how="inner")

    pred_cols = [f"pred_{sp}" for sp in probe_species]
    return merged[pred_cols].values, probe_species


def identify_evaluable(species_list: list[str], gt: pd.DataFrame, min_pos: int = 2) -> list[str]:
    """Get evaluable species (have focal data + enough positive windows)."""
    evaluable = []
    for sp in species_list:
        if sp.startswith(SONOTYPE_PREFIX) or sp in NO_FOCAL_AMPHIBIA:
            continue
        if sp not in gt.columns:
            continue
        if gt[sp].sum() < min_pos:
            continue
        evaluable.append(sp)
    return evaluable


def compute_metrics(
    gt: pd.DataFrame,
    pred: np.ndarray,
    pred_species: list[str],
    evaluable: list[str],
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """Per-species AUROC and AP."""
    sp_to_idx = {sp: i for i, sp in enumerate(pred_species)}
    rows = []
    for sp in evaluable:
        if sp not in sp_to_idx or sp not in gt.columns:
            continue
        y_true = gt[sp].values
        y_score = pred[:, sp_to_idx[sp]]
        n_pos = int(y_true.sum())
        if n_pos == 0 or n_pos == len(y_true):
            continue
        auroc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        taxa_row = taxonomy.loc[taxonomy["primary_label"].astype(str) == sp]
        taxa = taxa_row.iloc[0].get("class_name", "Unknown") if len(taxa_row) > 0 else "Unknown"
        rows.append({
            "species": sp, "auroc": auroc, "ap": ap,
            "n_positive": n_pos, "taxa": taxa,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--eda-artifacts", type=Path, default=EDA_ARTIFACTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)

    print("=== Loading ground truth ===")
    gt, species_list = load_ground_truth(args.eda_artifacts)
    evaluable = identify_evaluable(species_list, gt)
    print(f"  {len(gt)} windows, {len(evaluable)} evaluable species")

    # --- Evaluate probes ---
    probe_results = {}
    for model in ["birdnet", "perch"]:
        probe_path = PROBE_DIR / model / "species_probe.npz"
        if not probe_path.exists():
            print(f"\n--- {model} probe: NOT FOUND, skipping ---")
            continue

        print(f"\n=== {model} probe ===")
        probe = load_probe(probe_path)
        print(f"  Probe: {probe['coef'].shape[1]}d -> {probe['coef'].shape[0]} species")

        print(f"  Loading soundscape embeddings...")
        embeddings, filenames, start_times = load_soundscape_embeddings(model)
        print(f"  {len(embeddings)} embedding rows")

        print(f"  Applying probe...")
        probs = apply_probe(embeddings, probe)

        print(f"  Aggregating to 5s windows (max)...")
        probs_5s, fnames_5s, starts_5s = aggregate_to_5s_windows(
            probs, filenames, start_times
        )
        print(f"  {len(probs_5s)} 5s windows")

        print(f"  Aligning to ground truth...")
        aligned, probe_species = align_to_ground_truth(
            gt, probs_5s, fnames_5s, starts_5s, probe["species_list"]
        )
        print(f"  {len(aligned)} aligned windows")

        print(f"  Computing metrics...")
        metrics = compute_metrics(gt, aligned, probe_species, evaluable, taxonomy)
        macro_auroc = metrics["auroc"].mean()
        macro_ap = metrics["ap"].mean()
        print(f"  Macro AUROC: {macro_auroc:.4f}")
        print(f"  Macro AP:    {macro_ap:.4f}")

        for taxa, group in metrics.groupby("taxa"):
            print(f"    {taxa}: AUROC={group['auroc'].mean():.4f} ({len(group)} species)")

        metrics.to_csv(args.output / f"metrics_probe_{model}.csv", index=False)
        probe_results[f"probe_{model}"] = {
            "macro_auroc": float(macro_auroc),
            "macro_ap": float(macro_ap),
            "n_species": len(metrics),
            "n_windows": len(aligned),
        }

    # --- Load Murilo's model results for comparison ---
    print("\n=== Comparison with fine-tuned models ===")
    prev_results = {}
    for name in ["protoclr", "convnext", "eat", "ensemble"]:
        csv_path = args.output / f"metrics_{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            prev_results[name] = {
                "macro_auroc": float(df["auroc"].dropna().mean()),
                "macro_ap": float(df["ap"].dropna().mean()),
            }

    # Summary table
    print(f"\n{'Model':<25} {'AUROC':>10} {'AP':>10}")
    print("-" * 47)
    for name, res in prev_results.items():
        print(f"{name:<25} {res['macro_auroc']:>10.4f} {res['macro_ap']:>10.4f}")
    for name, res in probe_results.items():
        print(f"{name:<25} {res['macro_auroc']:>10.4f} {res['macro_ap']:>10.4f}")

    # Save combined results
    all_results = {**prev_results, **probe_results}
    with open(args.output / "all_model_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nArtifacts saved to {args.output}/")


if __name__ == "__main__":
    main()
