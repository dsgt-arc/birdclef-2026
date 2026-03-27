"""Train per-class LogReg probes and build metadata priors.

Following chiranjithdharma's approach (0.910 LB):
- PCA compress embeddings: 1536 -> 32 dims
- Per-species LogisticRegression on PCA embed + context features
- 5-fold GroupKFold by site for honest evaluation
- Site/hour Bayesian prior tables
- Dynamic alpha blending per species

Reads:
  - outputs/perch_embeddings.parquet
  - outputs/perch_mapped_scores.parquet
  - train_soundscapes_labels.csv
  - artifacts/perch_species_mapping.json

Outputs (to artifacts/perch_probes/):
  - pca_transformer.pkl
  - probe_models.pkl ({primary_label: LogisticRegression})
  - probe_alphas.json ({primary_label: float})
  - metadata_priors.json (site/hour/interaction tables)
  - taxa_mapping.json ({primary_label: class_name})
  - config.json (ensemble hyperparameters)
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-class LogReg probes and build metadata priors"
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
        "--config",
        type=str,
        default=None,
        help="Path to perch_ensemble config YAML (default: config/perch_ensemble_v1.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/perch_probes/)",
    )
    return parser.parse_args()


def parse_filename_metadata(filename: str) -> dict:
    """Parse site and datetime from soundscape filename.

    Pattern: BC2026_Train_XXXX_SNN_YYYYMMDD_HHMMSS.ogg
    where XXXX=file_id, SNN=site, YYYYMMDD_HHMMSS=datetime
    """
    stem = Path(filename).stem  # Remove .ogg
    # Pattern: BC2026_Train_0039_S22_20211231_201500
    match = re.match(
        r"BC2026_Train_(\d+)_([A-Z]\d+)_(\d{8})_(\d{6})",
        stem,
    )
    if match:
        file_id = match.group(1)
        site = match.group(2)
        date_str = match.group(3)
        time_str = match.group(4)
        hour = int(time_str[:2])
        return {
            "file_id": file_id,
            "site": site,
            "date": date_str,
            "time": time_str,
            "hour": hour,
        }
    else:
        # Fallback
        return {
            "file_id": "unknown",
            "site": "unknown",
            "date": "unknown",
            "time": "unknown",
            "hour": -1,
        }


def build_labels_matrix(
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    birdclef_species: list[str],
) -> np.ndarray:
    """Build binary label matrix (n_windows, n_species) from soundscape labels.

    Labels CSV has inat_taxon_ids and eBird codes separated by semicolons.
    We need to match these to primary_labels in the species list.
    """
    n_windows = len(metadata_df)
    n_species = len(birdclef_species)
    species_to_idx = {s: i for i, s in enumerate(birdclef_species)}

    labels_matrix = np.zeros((n_windows, n_species), dtype=np.float32)

    for window_idx, row in metadata_df.iterrows():
        filename = row["filename"]
        window_start = row["window_start"]
        window_end = row["window_end"]

        # Find matching label rows
        # Labels CSV uses HH:MM:SS format for start/end
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

            individual_labels = label_str.split(";")
            for label in individual_labels:
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
    """Build feature matrix for a single species probe.

    Features per (window, species):
    - PCA embedding (32 dims)
    - Mapped Perch score for this species (1)
    - Previous window score (1)
    - Next window score (1)
    - File mean score (1)
    - File max score (1)
    Total: ~37 features
    """
    n_samples = pca_embeddings.shape[0]
    species_scores = mapped_scores[:, species_idx]

    # Temporal context: previous and next window scores
    prev_scores = np.zeros(n_samples, dtype=np.float32)
    next_scores = np.zeros(n_samples, dtype=np.float32)
    file_mean_scores = np.zeros(n_samples, dtype=np.float32)
    file_max_scores = np.zeros(n_samples, dtype=np.float32)

    # Group by filename for temporal context
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
            pca_embeddings,  # 32 dims
            species_scores,  # 1
            prev_scores,  # 1
            next_scores,  # 1
            file_mean_scores,  # 1
            file_max_scores,  # 1
        ]
    )

    return features


def build_metadata_priors(
    labels_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    birdclef_species: list[str],
    config: dict,
) -> dict:
    """Build Bayesian site/hour/interaction prior tables.

    P(species | site) with Bayesian smoothing
    P(species | hour) with Bayesian smoothing
    P(species | site, hour) with Bayesian smoothing

    Smoothing: n/(n+k) * local_prior + (1-n/(n+k)) * global_prior
    Laplace smoothing alpha for zero counts.
    """
    k_site = config["priors"]["bayesian_k_site"]
    k_hour = config["priors"]["bayesian_k_hour"]
    k_interaction = config["priors"]["bayesian_k_interaction"]
    laplace_alpha = config["priors"]["laplace_alpha"]

    # Parse metadata
    sites = []
    hours = []
    for _, row in metadata_df.iterrows():
        meta = parse_filename_metadata(row["filename"])
        sites.append(meta["site"])
        hours.append(meta["hour"])

    sites = np.array(sites)
    hours = np.array(hours)

    unique_sites = sorted(set(sites))
    unique_hours = sorted(set(hours))

    # Global prior: P(species) across all windows
    global_prior = labels_matrix.mean(axis=0)  # (n_species,)

    # Site priors: P(species | site)
    site_priors = {}
    for site in unique_sites:
        site_mask = sites == site
        n = site_mask.sum()
        if n == 0:
            site_priors[site] = global_prior.tolist()
            continue
        local_prior = labels_matrix[site_mask].mean(axis=0)
        # Bayesian smoothing
        weight = n / (n + k_site)
        smoothed = weight * local_prior + (1 - weight) * global_prior
        # Laplace smoothing for zeros
        smoothed = np.maximum(smoothed, laplace_alpha)
        site_priors[site] = smoothed.tolist()

    # Hour priors: P(species | hour)
    hour_priors = {}
    for hour in unique_hours:
        hour_mask = hours == hour
        n = hour_mask.sum()
        if n == 0:
            hour_priors[str(hour)] = global_prior.tolist()
            continue
        local_prior = labels_matrix[hour_mask].mean(axis=0)
        weight = n / (n + k_hour)
        smoothed = weight * local_prior + (1 - weight) * global_prior
        smoothed = np.maximum(smoothed, laplace_alpha)
        hour_priors[str(hour)] = smoothed.tolist()

    # Interaction priors: P(species | site, hour)
    interaction_priors = {}
    for site in unique_sites:
        for hour in unique_hours:
            mask = (sites == site) & (hours == hour)
            n = mask.sum()
            key = f"{site}_{hour}"
            if n == 0:
                # Fall back to geometric mean of site and hour priors
                site_p = np.array(site_priors[site])
                hour_p = np.array(hour_priors[str(hour)])
                interaction_priors[key] = np.sqrt(site_p * hour_p).tolist()
                continue
            local_prior = labels_matrix[mask].mean(axis=0)
            weight = n / (n + k_interaction)
            # Use site prior as base for interaction smoothing
            site_p = np.array(site_priors[site])
            smoothed = weight * local_prior + (1 - weight) * site_p
            smoothed = np.maximum(smoothed, laplace_alpha)
            interaction_priors[key] = smoothed.tolist()

    priors = {
        "global_prior": global_prior.tolist(),
        "site_priors": site_priors,
        "hour_priors": hour_priors,
        "interaction_priors": interaction_priors,
        "unique_sites": unique_sites,
        "unique_hours": [int(h) for h in unique_hours],
        "species_order": birdclef_species,
    }

    return priors


def main() -> None:
    args = parse_args()

    # Resolve paths
    experiment_dir = Path(__file__).resolve().parent.parent

    if args.embeddings_parquet is None:
        embeddings_path = experiment_dir / "outputs" / "perch_embeddings.parquet"
    else:
        embeddings_path = Path(args.embeddings_parquet)

    if args.mapped_scores_parquet is None:
        mapped_scores_path = experiment_dir / "outputs" / "perch_mapped_scores.parquet"
    else:
        mapped_scores_path = Path(args.mapped_scores_parquet)

    if args.species_mapping is None:
        species_mapping_path = experiment_dir / "artifacts" / "perch_species_mapping.json"
    else:
        species_mapping_path = Path(args.species_mapping)

    if args.config is None:
        config_path = experiment_dir / "config" / "perch_ensemble_v1.yaml"
    else:
        config_path = Path(args.config)

    if args.output_dir is None:
        output_dir = experiment_dir / "artifacts" / "perch_probes"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    probe_config = config["probes"]
    prior_config = config["priors"]

    # Load species mapping
    print(f"Loading species mapping from: {species_mapping_path}")
    with open(species_mapping_path) as f:
        species_mapping = json.load(f)

    birdclef_species = species_mapping["birdclef_species"]
    taxa_classes = species_mapping["taxa_classes"]
    n_species = len(birdclef_species)
    print(f"  {n_species} BirdCLEF species")

    # Load embeddings
    print(f"\nLoading embeddings from: {embeddings_path}")
    embeddings_df = pd.read_parquet(embeddings_path)
    metadata_cols = ["filename", "window_idx", "window_start", "window_end"]
    metadata_df = embeddings_df[metadata_cols].copy()
    embedding_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    raw_embeddings = embeddings_df[embedding_cols].values.astype(np.float32)
    print(f"  Shape: {raw_embeddings.shape}")

    # Load mapped scores
    print(f"Loading mapped scores from: {mapped_scores_path}")
    mapped_df = pd.read_parquet(mapped_scores_path)
    mapped_scores = mapped_df[birdclef_species].values.astype(np.float32)
    print(f"  Shape: {mapped_scores.shape}")

    # Load labels
    print(f"\nLoading labels from: {args.labels_csv}")
    labels_df = pd.read_csv(args.labels_csv)
    print(f"  {len(labels_df)} label rows")

    # Build binary label matrix
    print("Building binary label matrix...")
    labels_matrix = build_labels_matrix(metadata_df, labels_df, birdclef_species)
    n_positive_per_species = labels_matrix.sum(axis=0)
    print(f"  Species with any positive: {(n_positive_per_species > 0).sum()}")
    print(f"  Total positive labels: {labels_matrix.sum():.0f}")

    # PCA compression
    pca_components = probe_config["pca_components"]
    print(f"\nFitting PCA: {raw_embeddings.shape[1]} -> {pca_components} dims")
    pca = PCA(n_components=pca_components, random_state=42)
    pca_embeddings = pca.fit_transform(raw_embeddings)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained_var:.4f}")
    print(f"  PCA embeddings shape: {pca_embeddings.shape}")

    # Parse sites for GroupKFold
    site_labels = []
    for _, row in metadata_df.iterrows():
        meta = parse_filename_metadata(row["filename"])
        site_labels.append(meta["site"])
    site_labels = np.array(site_labels)
    unique_sites = sorted(set(site_labels))
    print(f"\nSites for GroupKFold: {len(unique_sites)} unique sites")
    for site in unique_sites:
        n = (site_labels == site).sum()
        print(f"  {site}: {n} windows")

    # Train per-species probes
    min_positive = probe_config["min_positive_samples"]
    logreg_C = probe_config["logreg_C"]
    logreg_solver = probe_config["logreg_solver"]
    class_weight = probe_config["class_weight"]
    alpha_base = probe_config["alpha_base"]
    alpha_scale = probe_config["alpha_scale"]
    alpha_threshold = probe_config["alpha_threshold"]
    n_folds = probe_config["n_folds"]

    print(f"\nTraining per-species probes (min_positive={min_positive}, C={logreg_C})...")
    probe_models = {}
    probe_alphas = {}
    cv_scores = {}

    n_trained = 0
    n_skipped = 0

    for sp_idx, species in enumerate(birdclef_species):
        n_pos = int(n_positive_per_species[sp_idx])

        if n_pos < min_positive:
            n_skipped += 1
            continue

        # Build features for this species
        features = build_features(pca_embeddings, mapped_scores, sp_idx, metadata_df)
        y = labels_matrix[:, sp_idx]

        # Dynamic alpha: higher for species with more positive samples
        alpha = alpha_base + alpha_scale * min(1.0, (n_pos - min_positive) / alpha_threshold)
        probe_alphas[species] = round(alpha, 4)

        # Train on all data (final model)
        model = LogisticRegression(
            C=logreg_C,
            solver=logreg_solver,
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )
        model.fit(features, y)
        probe_models[species] = model
        n_trained += 1

        # Cross-validation for evaluation
        if n_folds > 1 and len(unique_sites) >= n_folds:
            gkf = GroupKFold(n_splits=min(n_folds, len(unique_sites)))
            fold_aucs = []

            for train_idx, val_idx in gkf.split(features, y, groups=site_labels):
                # Check if both classes present in train and val
                if y[train_idx].sum() == 0 or y[val_idx].sum() == 0:
                    continue
                if y[train_idx].sum() == len(train_idx) or y[val_idx].sum() == len(val_idx):
                    continue

                fold_model = LogisticRegression(
                    C=logreg_C,
                    solver=logreg_solver,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=42,
                )
                fold_model.fit(features[train_idx], y[train_idx])
                pred = fold_model.predict_proba(features[val_idx])[:, 1]

                try:
                    auc = roc_auc_score(y[val_idx], pred)
                    fold_aucs.append(auc)
                except ValueError:
                    pass

            if fold_aucs:
                mean_auc = np.mean(fold_aucs)
                cv_scores[species] = round(mean_auc, 4)

    print(f"\n  Trained: {n_trained} probes")
    print(f"  Skipped (n_pos < {min_positive}): {n_skipped}")

    if cv_scores:
        aucs = list(cv_scores.values())
        print(f"\n  CV AUC summary ({len(aucs)} species with valid CV):")
        print(f"    Mean: {np.mean(aucs):.4f}")
        print(f"    Median: {np.median(aucs):.4f}")
        print(f"    Min: {np.min(aucs):.4f}")
        print(f"    Max: {np.max(aucs):.4f}")

        # Top/bottom 5
        sorted_aucs = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
        print("\n    Top 5:")
        for sp, auc in sorted_aucs[:5]:
            print(f"      {sp}: {auc:.4f}")
        print("    Bottom 5:")
        for sp, auc in sorted_aucs[-5:]:
            print(f"      {sp}: {auc:.4f}")

    # Build metadata priors
    print("\nBuilding metadata priors...")
    priors = build_metadata_priors(labels_matrix, metadata_df, birdclef_species, config)
    print(f"  Sites: {len(priors['unique_sites'])}")
    print(f"  Hours: {len(priors['unique_hours'])}")
    print(f"  Interaction keys: {len(priors['interaction_priors'])}")

    # Build taxa mapping
    taxa_mapping = {sp: taxa_classes[sp] for sp in birdclef_species}

    # Build config summary
    config_summary = {
        "pca_components": pca_components,
        "pca_explained_variance": round(explained_var, 6),
        "logreg_C": logreg_C,
        "logreg_solver": logreg_solver,
        "class_weight": class_weight,
        "min_positive_samples": min_positive,
        "n_folds": n_folds,
        "n_trained_probes": n_trained,
        "n_skipped_probes": n_skipped,
        "n_total_species": n_species,
        "n_windows": len(metadata_df),
        "feature_dim": pca_components + 5,  # PCA + 5 context features
        "alpha_base": alpha_base,
        "alpha_scale": alpha_scale,
        "alpha_threshold": alpha_threshold,
        "priors": {
            "lambda_event": prior_config["lambda_event"],
            "lambda_texture": prior_config["lambda_texture"],
            "lambda_proxy_texture": prior_config["lambda_proxy_texture"],
            "bayesian_k_site": prior_config["bayesian_k_site"],
            "bayesian_k_hour": prior_config["bayesian_k_hour"],
            "bayesian_k_interaction": prior_config["bayesian_k_interaction"],
            "laplace_alpha": prior_config["laplace_alpha"],
        },
        "smoothing": config.get("smoothing", {}),
        "ensemble": config.get("ensemble", {}),
        "cv_scores": cv_scores,
    }

    # Save all artifacts
    print(f"\nSaving artifacts to: {output_dir}")

    # PCA transformer
    pca_path = output_dir / "pca_transformer.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"  pca_transformer.pkl ({pca_path.stat().st_size / 1024:.1f} KB)")

    # Probe models
    probes_path = output_dir / "probe_models.pkl"
    with open(probes_path, "wb") as f:
        pickle.dump(probe_models, f)
    print(f"  probe_models.pkl ({probes_path.stat().st_size / 1024:.1f} KB)")

    # Probe alphas
    alphas_path = output_dir / "probe_alphas.json"
    with open(alphas_path, "w") as f:
        json.dump(probe_alphas, f, indent=2)
    print(f"  probe_alphas.json ({alphas_path.stat().st_size / 1024:.1f} KB)")

    # Metadata priors
    priors_path = output_dir / "metadata_priors.json"
    with open(priors_path, "w") as f:
        json.dump(priors, f)
    print(f"  metadata_priors.json ({priors_path.stat().st_size / 1024:.1f} KB)")

    # Taxa mapping
    taxa_path = output_dir / "taxa_mapping.json"
    with open(taxa_path, "w") as f:
        json.dump(taxa_mapping, f, indent=2)
    print(f"  taxa_mapping.json ({taxa_path.stat().st_size / 1024:.1f} KB)")

    # Config
    config_path_out = output_dir / "config.json"
    with open(config_path_out, "w") as f:
        json.dump(config_summary, f, indent=2)
    print(f"  config.json ({config_path_out.stat().st_size / 1024:.1f} KB)")

    print("\nDone! All probe artifacts saved.")


if __name__ == "__main__":
    main()
