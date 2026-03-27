"""Local validation of the Perch ensemble on labeled soundscapes.

Runs the full ensemble pipeline (SED + Perch + probes + priors) on the
66 labeled soundscapes and evaluates against ground truth. Sweeps blend
weights to find optimal mix.

Requires:
  - Perch embeddings extracted (outputs/perch_embeddings.parquet)
  - Perch mapped scores (outputs/perch_mapped_scores.parquet)
  - Probe artifacts trained (artifacts/perch_probes/)
  - Species mapping (artifacts/perch_species_mapping.json)
  - SED predictions exported (or --skip-sed to run Perch-only)

Usage:
  python scripts/local_validate.py
  python scripts/local_validate.py --skip-sed
  python scripts/local_validate.py --sed-predictions outputs/sed_predictions.parquet
  python scripts/local_validate.py --sweep-weights
"""

import argparse
import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local validation of Perch ensemble on labeled soundscapes"
    )
    parser.add_argument(
        "--embeddings-parquet",
        type=str,
        default=None,
        help="Path to perch_embeddings.parquet (default: outputs/perch_embeddings.parquet)",
    )
    parser.add_argument(
        "--mapped-scores-parquet",
        type=str,
        default=None,
        help="Path to perch_mapped_scores.parquet (default: outputs/perch_mapped_scores.parquet)",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes_labels.csv",
        help="Path to train_soundscapes_labels.csv",
    )
    parser.add_argument(
        "--species-mapping",
        type=str,
        default=None,
        help="Path to perch_species_mapping.json (default: artifacts/perch_species_mapping.json)",
    )
    parser.add_argument(
        "--probes-dir",
        type=str,
        default=None,
        help="Path to probe artifacts directory (default: artifacts/perch_probes/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to perch_ensemble config YAML (default: config/perch_ensemble_v1.yaml)",
    )
    parser.add_argument(
        "--sed-predictions",
        type=str,
        default=None,
        help="Path to SED predictions parquet (filename, window_idx, species cols). "
        "If not provided and --skip-sed is not set, uses mapped Perch scores as SED proxy.",
    )
    parser.add_argument(
        "--skip-sed",
        action="store_true",
        help="Run Perch-only evaluation (no SED component)",
    )
    parser.add_argument(
        "--sweep-weights",
        action="store_true",
        help="Sweep blend weights: w_perch in {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}",
    )
    return parser.parse_args()


def parse_filename_metadata(filename: str) -> dict:
    """Parse site and hour from soundscape filename."""
    stem = Path(filename).stem
    match = re.match(
        r"BC2026_Train_(\d+)_([A-Z]\d+)_(\d{8})_(\d{6})",
        stem,
    )
    if match:
        return {
            "site": match.group(2),
            "hour": int(match.group(4)[:2]),
        }
    return {"site": "unknown", "hour": -1}


def build_labels_matrix(
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    birdclef_species: list[str],
) -> np.ndarray:
    """Build binary label matrix (n_windows, n_species) from soundscape labels."""
    n_windows = len(metadata_df)
    n_species = len(birdclef_species)
    species_to_idx = {s: i for i, s in enumerate(birdclef_species)}

    labels_matrix = np.zeros((n_windows, n_species), dtype=np.float32)

    for window_idx, row in metadata_df.iterrows():
        filename = row["filename"]
        window_start = row["window_start"]
        window_end = row["window_end"]

        start_str = f"{int(window_start // 3600):02d}:{int((window_start % 3600) // 60):02d}:{int(window_start % 60):02d}"
        end_str = f"{int(window_end // 3600):02d}:{int((window_end % 3600) // 60):02d}:{int(window_end % 60):02d}"

        mask = (
            (labels_df["filename"] == filename)
            & (labels_df["start"] == start_str)
            & (labels_df["end"] == end_str)
        )
        matching = labels_df[mask]

        for _, label_row in matching.iterrows():
            label_str = str(label_row["primary_label"])
            if not label_str or label_str == "nan":
                continue
            for label in label_str.split(";"):
                label = label.strip()
                if label in species_to_idx:
                    labels_matrix[window_idx, species_to_idx[label]] = 1.0

    return labels_matrix


def build_features(
    pca_embeddings: np.ndarray,
    mapped_scores: np.ndarray,
    species_idx: int,
    metadata_df: pd.DataFrame,
) -> np.ndarray:
    """Build feature matrix for a single species probe (matches train_probes.py)."""
    n_samples = pca_embeddings.shape[0]
    species_scores = mapped_scores[:, species_idx]

    prev_scores = np.zeros(n_samples, dtype=np.float32)
    next_scores = np.zeros(n_samples, dtype=np.float32)
    file_mean_scores = np.zeros(n_samples, dtype=np.float32)
    file_max_scores = np.zeros(n_samples, dtype=np.float32)

    filenames = metadata_df["filename"].values
    unique_files = metadata_df["filename"].unique()

    for fname in unique_files:
        file_mask = filenames == fname
        file_indices = np.where(file_mask)[0]
        file_scores = species_scores[file_indices]

        file_mean_scores[file_indices] = file_scores.mean()
        file_max_scores[file_indices] = file_scores.max()

        for j, idx in enumerate(file_indices):
            if j > 0:
                prev_scores[idx] = species_scores[file_indices[j - 1]]
            if j < len(file_indices) - 1:
                next_scores[idx] = species_scores[file_indices[j + 1]]

    features = np.column_stack(
        [
            pca_embeddings,
            species_scores,
            prev_scores,
            next_scores,
            file_mean_scores,
            file_max_scores,
        ]
    )

    return features


def apply_probes(
    pca_embeddings: np.ndarray,
    mapped_scores: np.ndarray,
    metadata_df: pd.DataFrame,
    birdclef_species: list[str],
    probe_models: dict,
    probe_alphas: dict,
) -> np.ndarray:
    """Apply trained probes to get blended Perch scores.

    For species with probes: (1-alpha)*base_score + alpha*probe_score
    For species without probes: base_score
    """
    perch_probs = mapped_scores.copy()

    for sp_idx, species in enumerate(birdclef_species):
        if species not in probe_models:
            continue

        model = probe_models[species]
        alpha = probe_alphas.get(species, 0.0)

        features = build_features(pca_embeddings, mapped_scores, sp_idx, metadata_df)
        probe_score = model.predict_proba(features)[:, 1]

        perch_probs[:, sp_idx] = (1 - alpha) * mapped_scores[:, sp_idx] + alpha * probe_score

    return perch_probs


def apply_metadata_priors(
    scores: np.ndarray,
    metadata_df: pd.DataFrame,
    birdclef_species: list[str],
    taxa_mapping: dict,
    priors: dict,
    config: dict,
) -> np.ndarray:
    """Apply site/hour metadata priors with taxa-aware lambda."""
    lambda_event = config["priors"]["lambda_event"]
    lambda_texture = config["priors"]["lambda_texture"]

    texture_classes = {"Amphibia", "Insecta"}
    out = scores.copy()

    for i, row in metadata_df.iterrows():
        meta = parse_filename_metadata(row["filename"])
        site = meta["site"]
        hour = str(meta["hour"])

        # Get prior vector for this (site, hour)
        interaction_key = f"{site}_{hour}"
        if interaction_key in priors["interaction_priors"]:
            prior = np.array(priors["interaction_priors"][interaction_key], dtype=np.float32)
        elif site in priors["site_priors"]:
            prior = np.array(priors["site_priors"][site], dtype=np.float32)
        elif hour in priors["hour_priors"]:
            prior = np.array(priors["hour_priors"][hour], dtype=np.float32)
        else:
            prior = np.array(priors["global_prior"], dtype=np.float32)

        # Taxa-aware lambda per species
        for sp_idx, species in enumerate(birdclef_species):
            taxa = taxa_mapping.get(species, "Aves")
            lam = lambda_texture if taxa in texture_classes else lambda_event
            out[i, sp_idx] = scores[i, sp_idx] * (prior[sp_idx] ** lam)

    return out


def apply_temporal_smoothing(
    scores: np.ndarray,
    metadata_df: pd.DataFrame,
    birdclef_species: list[str],
    taxa_mapping: dict,
    config: dict,
) -> np.ndarray:
    """Taxa-aware temporal smoothing within each file.

    Events (Aves): local-max propagation with alpha_event
    Textures (Amphibia/Insecta): neighbor average with alpha_texture
    """
    alpha_event = config["smoothing"]["alpha_event"]
    alpha_texture = config["smoothing"]["alpha_texture"]
    texture_classes = {"Amphibia", "Insecta"}

    out = scores.copy()
    filenames = metadata_df["filename"].values
    unique_files = metadata_df["filename"].unique()

    for fname in unique_files:
        file_mask = filenames == fname
        file_indices = np.where(file_mask)[0]

        if len(file_indices) < 2:
            continue

        for sp_idx, species in enumerate(birdclef_species):
            taxa = taxa_mapping.get(species, "Aves")
            file_scores = scores[file_indices, sp_idx]

            if taxa in texture_classes:
                # Average-neighbor smoothing
                smoothed = file_scores.copy()
                for j in range(len(file_scores)):
                    neighbors = [file_scores[j]]
                    if j > 0:
                        neighbors.append(file_scores[j - 1])
                    if j < len(file_scores) - 1:
                        neighbors.append(file_scores[j + 1])
                    avg = np.mean(neighbors)
                    smoothed[j] = (1 - alpha_texture) * file_scores[j] + alpha_texture * avg
                out[file_indices, sp_idx] = smoothed
            else:
                # Local-max propagation
                smoothed = file_scores.copy()
                for j in range(len(file_scores)):
                    local_max = file_scores[j]
                    if j > 0:
                        local_max = max(local_max, file_scores[j - 1])
                    if j < len(file_scores) - 1:
                        local_max = max(local_max, file_scores[j + 1])
                    smoothed[j] = (1 - alpha_event) * file_scores[j] + alpha_event * local_max
                out[file_indices, sp_idx] = smoothed

    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    birdclef_species: list[str],
    taxa_mapping: dict,
) -> dict:
    """Compute macro AUROC overall and per taxa class."""
    # Per-species AUROC (only for species with both classes present)
    species_aucs = {}
    for sp_idx, species in enumerate(birdclef_species):
        y_sp = y_true[:, sp_idx]
        if y_sp.sum() == 0 or y_sp.sum() == len(y_sp):
            continue
        try:
            auc = roc_auc_score(y_sp, y_pred[:, sp_idx])
            species_aucs[species] = auc
        except ValueError:
            pass

    if not species_aucs:
        return {"macro_auroc": 0.0, "n_evaluated": 0}

    macro_auroc = np.mean(list(species_aucs.values()))

    # Per-taxa breakdown
    taxa_aucs = {}
    for species, auc in species_aucs.items():
        taxa = taxa_mapping.get(species, "Aves")
        if taxa not in taxa_aucs:
            taxa_aucs[taxa] = []
        taxa_aucs[taxa].append(auc)

    taxa_summary = {}
    for taxa, aucs in sorted(taxa_aucs.items()):
        taxa_summary[taxa] = {
            "mean_auroc": round(np.mean(aucs), 4),
            "n_species": len(aucs),
        }

    return {
        "macro_auroc": round(macro_auroc, 4),
        "n_evaluated": len(species_aucs),
        "per_taxa": taxa_summary,
        "species_aucs": {sp: round(auc, 4) for sp, auc in species_aucs.items()},
    }


def print_metrics(label: str, metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n  {label}")
    print(f"    Macro AUROC: {metrics['macro_auroc']:.4f} ({metrics['n_evaluated']} species)")
    if "per_taxa" in metrics:
        for taxa, info in sorted(metrics["per_taxa"].items()):
            print(f"      {taxa}: {info['mean_auroc']:.4f} ({info['n_species']} spp)")


def main() -> None:
    args = parse_args()

    # Resolve paths
    experiment_dir = Path(__file__).resolve().parent.parent

    embeddings_path = (
        Path(args.embeddings_parquet)
        if args.embeddings_parquet
        else experiment_dir / "outputs" / "perch_embeddings.parquet"
    )
    mapped_scores_path = (
        Path(args.mapped_scores_parquet)
        if args.mapped_scores_parquet
        else experiment_dir / "outputs" / "perch_mapped_scores.parquet"
    )
    species_mapping_path = (
        Path(args.species_mapping)
        if args.species_mapping
        else experiment_dir / "artifacts" / "perch_species_mapping.json"
    )
    probes_dir = (
        Path(args.probes_dir) if args.probes_dir else experiment_dir / "artifacts" / "perch_probes"
    )
    config_path = (
        Path(args.config) if args.config else experiment_dir / "config" / "perch_ensemble_v1.yaml"
    )

    # Load config
    print("=" * 60)
    print("Experiment 018: Perch Ensemble Local Validation")
    print("=" * 60)

    print(f"\nConfig: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load species mapping
    print(f"Species mapping: {species_mapping_path}")
    with open(species_mapping_path) as f:
        species_mapping = json.load(f)

    birdclef_species = species_mapping["birdclef_species"]
    n_species = len(birdclef_species)
    print(f"  {n_species} BirdCLEF species")

    # Load taxa mapping
    taxa_mapping_path = probes_dir / "taxa_mapping.json"
    if taxa_mapping_path.exists():
        with open(taxa_mapping_path) as f:
            taxa_mapping = json.load(f)
    else:
        # Fall back to species_mapping taxa_classes
        taxa_mapping = species_mapping.get("taxa_classes", {})
    print(f"  Taxa mapping loaded ({len(taxa_mapping)} entries)")

    # Load embeddings and mapped scores
    print(f"\nLoading embeddings: {embeddings_path}")
    embeddings_df = pd.read_parquet(embeddings_path)
    metadata_cols = ["filename", "window_idx", "window_start", "window_end"]
    metadata_df = embeddings_df[metadata_cols].copy()
    embedding_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    raw_embeddings = embeddings_df[embedding_cols].values.astype(np.float32)
    print(f"  {len(metadata_df)} windows, {raw_embeddings.shape[1]}-dim embeddings")

    print(f"Loading mapped scores: {mapped_scores_path}")
    mapped_df = pd.read_parquet(mapped_scores_path)
    mapped_scores = mapped_df[birdclef_species].values.astype(np.float32)
    print(f"  Shape: {mapped_scores.shape}")

    # Load ground truth labels
    print(f"\nLoading labels: {args.labels_csv}")
    labels_df = pd.read_csv(args.labels_csv)
    labels_matrix = build_labels_matrix(metadata_df, labels_df, birdclef_species)
    n_positive = labels_matrix.sum()
    n_species_present = (labels_matrix.sum(axis=0) > 0).sum()
    print(f"  {n_positive:.0f} positive labels across {n_species_present} species")

    # Load probe artifacts
    print(f"\nLoading probe artifacts: {probes_dir}")
    pca_path = probes_dir / "pca_transformer.pkl"
    probes_path = probes_dir / "probe_models.pkl"
    alphas_path = probes_dir / "probe_alphas.json"
    priors_path = probes_dir / "metadata_priors.json"

    has_probes = pca_path.exists() and probes_path.exists()

    if has_probes:
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        with open(probes_path, "rb") as f:
            probe_models = pickle.load(f)
        with open(alphas_path) as f:
            probe_alphas = json.load(f)
        with open(priors_path) as f:
            priors = json.load(f)

        pca_embeddings = pca.transform(raw_embeddings)
        print(f"  PCA: {raw_embeddings.shape[1]} -> {pca_embeddings.shape[1]} dims")
        print(f"  Probes: {len(probe_models)} species")
        print(
            f"  Priors: {len(priors.get('unique_sites', []))} sites, {len(priors.get('unique_hours', []))} hours"
        )
    else:
        print("  WARNING: Probe artifacts not found — running raw Perch scores only")
        pca_embeddings = None
        probe_models = {}
        probe_alphas = {}
        priors = None

    # ---- Evaluate components ----
    print(f"\n{'=' * 60}")
    print("Evaluation")
    print("=" * 60)

    # 1. Raw Perch mapped scores (baseline)
    metrics_raw = compute_metrics(labels_matrix, mapped_scores, birdclef_species, taxa_mapping)
    print_metrics("Perch raw mapped scores (sigmoid only):", metrics_raw)

    # 2. Perch + probes
    if has_probes:
        perch_probed = apply_probes(
            pca_embeddings,
            mapped_scores,
            metadata_df,
            birdclef_species,
            probe_models,
            probe_alphas,
        )
        metrics_probed = compute_metrics(
            labels_matrix, perch_probed, birdclef_species, taxa_mapping
        )
        print_metrics("Perch + LogReg probes:", metrics_probed)

        # 3. Perch + probes + metadata priors
        perch_priors = apply_metadata_priors(
            perch_probed,
            metadata_df,
            birdclef_species,
            taxa_mapping,
            priors,
            config,
        )
        metrics_priors = compute_metrics(
            labels_matrix, perch_priors, birdclef_species, taxa_mapping
        )
        print_metrics("Perch + probes + metadata priors:", metrics_priors)

        # 4. Perch + probes + priors + temporal smoothing
        perch_smoothed = apply_temporal_smoothing(
            perch_priors,
            metadata_df,
            birdclef_species,
            taxa_mapping,
            config,
        )
        metrics_smoothed = compute_metrics(
            labels_matrix, perch_smoothed, birdclef_species, taxa_mapping
        )
        print_metrics("Perch + probes + priors + smoothing:", metrics_smoothed)

        perch_final = perch_smoothed
    else:
        perch_final = mapped_scores

    # 5. SED predictions (if available)
    sed_probs = None
    if not args.skip_sed and args.sed_predictions:
        sed_path = Path(args.sed_predictions)
        if sed_path.exists():
            print(f"\n  Loading SED predictions: {sed_path}")
            sed_df = pd.read_parquet(sed_path)
            sed_probs = sed_df[birdclef_species].values.astype(np.float32)
            metrics_sed = compute_metrics(labels_matrix, sed_probs, birdclef_species, taxa_mapping)
            print_metrics("SED only:", metrics_sed)

    # 6. Ensemble (if SED available)
    if sed_probs is not None:
        sed_weight = config["ensemble"]["sed_weight"]
        perch_weight = config["ensemble"]["perch_weight"]
        ensemble_probs = sed_weight * sed_probs + perch_weight * perch_final
        metrics_ensemble = compute_metrics(
            labels_matrix,
            ensemble_probs,
            birdclef_species,
            taxa_mapping,
        )
        print_metrics(
            f"Ensemble (SED={sed_weight:.1f} + Perch={perch_weight:.1f}):",
            metrics_ensemble,
        )

    # ---- Weight sweep ----
    if args.sweep_weights and sed_probs is not None:
        print(f"\n{'=' * 60}")
        print("Blend Weight Sweep")
        print("=" * 60)
        print(f"\n  {'w_perch':>8} {'w_sed':>8} {'macro_auroc':>12}")
        print(f"  {'-'*8} {'-'*8} {'-'*12}")

        best_auroc = 0.0
        best_w = 0.0

        for w_perch in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            w_sed = 1.0 - w_perch
            blended = w_sed * sed_probs + w_perch * perch_final
            m = compute_metrics(labels_matrix, blended, birdclef_species, taxa_mapping)
            marker = (
                " <-- config" if abs(w_perch - config["ensemble"]["perch_weight"]) < 0.01 else ""
            )
            best_marker = ""
            if m["macro_auroc"] > best_auroc:
                best_auroc = m["macro_auroc"]
                best_w = w_perch
                best_marker = " ***"
            print(f"  {w_perch:8.1f} {w_sed:8.1f} {m['macro_auroc']:12.4f}{marker}{best_marker}")

        print(f"\n  Best: w_perch={best_w:.1f}, macro_auroc={best_auroc:.4f}")

    elif args.sweep_weights and sed_probs is None:
        print("\n  Skipping weight sweep — no SED predictions provided.")
        print("  Use --sed-predictions <path> to enable.")

    # ---- Top/bottom species ----
    print(f"\n{'=' * 60}")
    print("Per-Species Analysis (Perch final)")
    print("=" * 60)

    final_metrics = compute_metrics(labels_matrix, perch_final, birdclef_species, taxa_mapping)
    if final_metrics["species_aucs"]:
        sorted_species = sorted(
            final_metrics["species_aucs"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\n  Top 10 species:")
        for sp, auc in sorted_species[:10]:
            taxa = taxa_mapping.get(sp, "?")
            print(f"    {sp:40s} {auc:.4f}  ({taxa})")

        print("\n  Bottom 10 species:")
        for sp, auc in sorted_species[-10:]:
            taxa = taxa_mapping.get(sp, "?")
            print(f"    {sp:40s} {auc:.4f}  ({taxa})")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"  Perch raw:          {metrics_raw['macro_auroc']:.4f}")
    if has_probes:
        print(f"  Perch + probes:     {metrics_probed['macro_auroc']:.4f}")
        print(f"  + metadata priors:  {metrics_priors['macro_auroc']:.4f}")
        print(f"  + smoothing:        {metrics_smoothed['macro_auroc']:.4f}")
    if sed_probs is not None:
        print(f"  SED only:           {metrics_sed['macro_auroc']:.4f}")
        print(f"  Ensemble:           {metrics_ensemble['macro_auroc']:.4f}")
    print()


if __name__ == "__main__":
    main()
