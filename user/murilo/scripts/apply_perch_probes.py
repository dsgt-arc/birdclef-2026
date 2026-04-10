#!/usr/bin/env python3
"""Phase C: Apply MLP probes to Perch TF embeddings → 234-class predictions.

Extracts the Perch→BirdCLEF pipeline from the exp 023 submission kernel
(submission_ensemble.py) and applies it to Phase B embeddings + logits
for train_soundscapes.

Pipeline per segment:
  1. Map Perch 14,795 logits → 234 BirdCLEF species (direct + genus proxy)
  2. Compute prior-fused base scores (site/hour tables)
  3. Temporal smoothing (texture + event species)
  4. StandardScaler → PCA(128) on embeddings
  5. Per-species MLP probes (58 species): 142 features → binary prob
  6. Alpha-blend probe predictions with base scores
  7. Sigmoid → probabilities
  8. Save perch_234_scores.parquet (logit) + perch_234_probs.parquet (prob)

Input:
  - shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/
      embeddings.parquet  (127,896 × 1,541)
      logits.parquet      (127,896 × 14,800)
  - shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/
  - shared/models/perch-v2-cpu/assets/labels.csv
  - shared/data/birdclef-2026/{taxonomy.csv, sample_submission.csv,
    train_soundscapes_labels.csv}

Output:
  - shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/
      perch_234_scores.parquet  (logit-space, 127,896 × 239)
      perch_234_probs.parquet   (probability-space, 127,896 × 239)

Usage:
  python apply_perch_probes.py [--source train_soundscapes]
"""

import argparse
import gc
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ============================================================================
# Paths
# ============================================================================

PROJECT = Path("/storage/project/ps-clef2026_birdclef-0")
SHARED = PROJECT / "shared"
DATA = SHARED / "data" / "birdclef-2026"
PERCH_MODEL = SHARED / "models" / "perch-v2-cpu"
ARTIFACTS = (
    SHARED / "artifacts" / "020-perch-onnx" / "pretrained-protossm-v18" / "pretrained_artifacts"
)
INFERENCE = SHARED / "inference" / "birdclef-2026" / "perch-v2-tf"

N_WINDOWS = 12  # 5s windows per 60s soundscape file

# ============================================================================
# Utility functions (from submission_ensemble.py)
# ============================================================================

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "UNK", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def seq_features_1d(v):
    """Temporal context features within 12-window file blocks."""
    assert len(v) % N_WINDOWS == 0, f"Expected multiple of {N_WINDOWS}, got {len(v)}"
    x = v.reshape(-1, N_WINDOWS)
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v = np.repeat(x.max(axis=1), N_WINDOWS)
    std_v = np.repeat(x.std(axis=1), N_WINDOWS)
    return prev_v, next_v, mean_v, max_v, std_v


def build_class_features(emb_proj, raw_col, prior_col, base_col):
    """Build 142-dim feature vector for MLP probe (128 PCA + 14 contextual)."""
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)
    diff_mean = base_col - mean_base
    diff_prev = base_col - prev_base
    diff_next = base_col - next_base
    feats = np.concatenate(
        [
            emb_proj,
            raw_col[:, None],
            prior_col[:, None],
            base_col[:, None],
            prev_base[:, None],
            next_base[:, None],
            mean_base[:, None],
            max_base[:, None],
            std_base[:, None],
            diff_mean[:, None],
            diff_prev[:, None],
            diff_next[:, None],
            (raw_col * prior_col)[:, None],
            (raw_col * base_col)[:, None],
            (prior_col * base_col)[:, None],
        ],
        axis=1,
    )
    return feats.astype(np.float32, copy=False)


def smooth_cols_fixed12(scores, cols, alpha=0.35):
    """Temporal smoothing for texture species (Amphibia, Insecta)."""
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s


def smooth_events_fixed12(scores, cols, alpha=0.15):
    """Temporal smoothing for event species (Aves, Mammalia, Reptilia)."""
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
    return s


def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    """Compute prior logits from site/hour tables."""
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    site_idx = np.fromiter(
        (tables["site_to_i"].get(str(s), -1) for s in sites), dtype=np.int32, count=n
    )
    hour_idx = np.fromiter(
        (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        dtype=np.int32,
        count=n,
    )
    sh_idx = np.fromiter(
        (
            tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1
            for s, h in zip(sites, hours, strict=False)
        ),
        dtype=np.int32,
        count=n,
    )

    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        wh = nh / (nh + 8.0)
        p[valid] = wh * tables["hour_p"][hour_idx[valid]] + (1.0 - wh) * p[valid]

    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        ws = ns / (ns + 8.0)
        p[valid] = ws * tables["site_p"][site_idx[valid]] + (1.0 - ws) * p[valid]

    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        wsh = nsh / (nsh + 4.0)
        p[valid] = wsh * tables["sh_p"][sh_idx[valid]] + (1.0 - wsh) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)


# ============================================================================
# Main pipeline
# ============================================================================


def main(source="train_soundscapes"):
    t0 = time.time()

    # --- Load competition data ---
    print("Loading competition metadata...")
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    soundscape_labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")

    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES = len(PRIMARY_LABELS)
    label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)

    CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
    TEXTURE_TAXA = {"Amphibia", "Insecta"}

    # Identify active classes (species present in labeled soundscapes)
    def parse_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    all_sc_labels = set()
    for x in soundscape_labels["primary_label"]:
        all_sc_labels.update(parse_labels(x))
    ACTIVE_CLASSES = sorted(c for c in all_sc_labels if c in label_to_idx)

    print(f"  {N_CLASSES} species, {len(ACTIVE_CLASSES)} active in labeled soundscapes")

    # --- Build Perch → BirdCLEF species mapping ---
    print("Building Perch → BirdCLEF species mapping...")
    bc_labels = (
        pd.read_csv(PERCH_MODEL / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    NO_LABEL_INDEX = len(bc_labels)

    tax = taxonomy.copy()
    tax["scientific_name_lookup"] = tax["scientific_name"]
    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})
    mapping = tax.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]],
        on="scientific_name_lookup",
        how="left",
    )
    mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)
    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]

    BC_INDICES = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
    MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
    MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
    UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

    print(f"  Direct mapping: {MAPPED_MASK.sum()}/{N_CLASSES} species")

    # Build genus proxies for unmapped species
    unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()

    proxy_map = {}
    for _, row in unmapped_non_sonotype.iterrows():
        target = row["primary_label"]
        sci = row["scientific_name"]
        genus = str(sci).split()[0]
        hits = bc_labels[
            bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
        ]
        if len(hits) > 0:
            proxy_map[target] = np.array(hits["bc_index"].astype(int).tolist(), dtype=np.int32)

    PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
    selected_proxy_targets = sorted(
        t for t in proxy_map.keys() if CLASS_NAME_MAP.get(t) in PROXY_TAXA
    )
    selected_proxy_pos_to_bc = {label_to_idx[t]: proxy_map[t] for t in selected_proxy_targets}
    selected_proxy_pos = np.array([label_to_idx[c] for c in selected_proxy_targets], dtype=np.int32)

    print(f"  Genus proxies: {len(selected_proxy_targets)} unmapped species")

    # Build taxonomy index arrays for prior fusion & smoothing
    idx_active_texture = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
        dtype=np.int32,
    )
    idx_active_event = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
        dtype=np.int32,
    )
    idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
    idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]
    idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
    idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]
    idx_unmapped_inactive = np.array(
        [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32
    )
    idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
    idx_selected_prioronly_active_texture = np.setdiff1d(
        idx_unmapped_active_texture, selected_proxy_pos
    )
    idx_selected_prioronly_active_event = np.setdiff1d(
        idx_unmapped_active_event, selected_proxy_pos
    )

    # --- Load pretrained artifacts ---
    print("Loading pretrained artifacts...")
    with open(ARTIFACTS / "config.json") as f:
        config = json.load(f)

    emb_scaler = joblib.load(ARTIFACTS / "emb_scaler.joblib")
    emb_pca = joblib.load(ARTIFACTS / "emb_pca.joblib")
    probe_models = joblib.load(ARTIFACTS / "probe_models.joblib")
    prior_tables = joblib.load(ARTIFACTS / "prior_tables.joblib")

    ALPHA = float(config["frozen_best_probe"]["alpha"])
    print(
        f"  PCA dim: {emb_pca.n_components_}, probes: {len(probe_models)} species, alpha: {ALPHA}"
    )

    # Fusion hyperparameters (from exp 023 kernel)
    LAMBDA_EVENT = 0.45
    LAMBDA_TEXTURE = 1.1
    LAMBDA_PROXY_TEXTURE = 0.9
    SMOOTH_TEXTURE = 0.35
    SMOOTH_EVENT = 0.15

    # --- Load Phase B embeddings + logits ---
    source_dir = INFERENCE / source
    print(f"Loading Phase B data from {source_dir}...")

    # Load embeddings (1.2 GB — fits in memory)
    emb_df = pd.read_parquet(source_dir / "embeddings.parquet")
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]

    # Ensure ordered by filename then chunk_idx (critical for seq_features_1d)
    emb_df = emb_df.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)

    N = len(emb_df)
    n_files = emb_df["filename"].nunique()
    chunks_per_file = emb_df.groupby("filename").size()

    if source == "train_soundscapes":
        assert (
            chunks_per_file == N_WINDOWS
        ).all(), (
            f"Not all files have {N_WINDOWS} chunks: {chunks_per_file.value_counts().to_dict()}"
        )
    print(f"  Embeddings: {emb_df.shape} ({N} segments, {n_files} files)")

    # Parse site/hour from filenames
    filenames = emb_df["filename"].values
    parsed = [parse_soundscape_filename(f) for f in filenames]
    sites = np.array([p["site"] for p in parsed])
    hours = np.array([p["hour_utc"] for p in parsed], dtype=np.int32)

    # Extract embeddings array
    embeddings = emb_df[emb_cols].values.astype(np.float32)
    # Keep sort order for logits
    sort_order_filenames = emb_df["filename"].values.copy()
    sort_order_chunks = emb_df["chunk_idx"].values.copy()
    del emb_df
    gc.collect()

    # --- Step 1: Map Perch logits → 234 BirdCLEF species ---
    # MEMORY OPTIMIZATION: only read the ~200 logit columns we need (not all 14,795)
    print("Step 1: Mapping Perch logits → 234 BirdCLEF species...")

    # Collect all needed Perch taxonomy indices
    needed_bc_indices = set(MAPPED_BC_INDICES.tolist())
    for bc_idx_arr in selected_proxy_pos_to_bc.values():
        needed_bc_indices.update(bc_idx_arr.tolist())
    needed_bc_indices = sorted(needed_bc_indices)
    needed_logit_cols = [f"logit_{i}" for i in needed_bc_indices]
    bc_idx_to_col_pos = {idx: i for i, idx in enumerate(needed_bc_indices)}

    print(f"  Reading {len(needed_logit_cols)}/{14795} logit columns (memory optimization)...")
    meta_logit_cols = ["filename", "chunk_idx"]
    logits_df = pd.read_parquet(
        source_dir / "logits.parquet",
        columns=meta_logit_cols + needed_logit_cols,
    )
    logits_df = logits_df.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)

    # Verify alignment with embeddings
    assert len(logits_df) == N, f"Row count mismatch: emb={N}, logits={len(logits_df)}"
    assert (logits_df["filename"].values == sort_order_filenames).all(), "Filename order mismatch"
    assert (logits_df["chunk_idx"].values == sort_order_chunks).all(), "Chunk order mismatch"

    logits_subset = logits_df[needed_logit_cols].values.astype(np.float32)
    del logits_df
    gc.collect()

    # Map to 234-class scores
    scores_raw = np.full((N, N_CLASSES), -10.0, dtype=np.float32)  # default: strong negative

    # Direct mapping
    mapped_col_positions = np.array([bc_idx_to_col_pos[idx] for idx in MAPPED_BC_INDICES])
    scores_raw[:, MAPPED_POS] = logits_subset[:, mapped_col_positions]

    # Genus proxy mapping (max across genus members)
    for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
        col_positions = np.array([bc_idx_to_col_pos[idx] for idx in bc_idx_arr])
        sub = logits_subset[:, col_positions]
        scores_raw[:, pos] = sub.max(axis=1).astype(np.float32)

    n_mapped = MAPPED_MASK.sum()
    n_proxy = len(selected_proxy_pos_to_bc)
    n_unmapped = N_CLASSES - n_mapped - n_proxy
    print(f"  Mapped: {n_mapped}, Proxy: {n_proxy}, Unmapped (default -10): {n_unmapped}")

    del logits_subset
    gc.collect()

    # Save raw mapped scores (before priors/probes) for Phase D analysis
    print("  Saving raw mapped scores...")
    raw_meta = pd.DataFrame(
        {
            "filename": filenames,
            "chunk_idx": sort_order_chunks,
            "site": sites,
            "hour_utc": hours,
        }
    )
    raw_probs_df = pd.DataFrame(sigmoid(scores_raw), columns=PRIMARY_LABELS)
    raw_out = pd.concat(
        [raw_meta[["filename", "chunk_idx", "site", "hour_utc"]], raw_probs_df], axis=1
    )
    raw_out.to_parquet(source_dir / "perch_234_raw_probs.parquet", index=False)
    print(f"  Saved perch_234_raw_probs.parquet ({raw_out.shape})")
    del raw_probs_df, raw_out
    gc.collect()

    # --- Step 2: Prior fusion ---
    print("Step 2: Computing prior-fused base scores...")
    prior_scores = prior_logits_from_tables(sites, hours, prior_tables)

    base_scores = scores_raw.copy()

    # Add prior logits for mapped active species
    if len(idx_mapped_active_event):
        base_scores[:, idx_mapped_active_event] += (
            LAMBDA_EVENT * prior_scores[:, idx_mapped_active_event]
        )
    if len(idx_mapped_active_texture):
        base_scores[:, idx_mapped_active_texture] += (
            LAMBDA_TEXTURE * prior_scores[:, idx_mapped_active_texture]
        )

    # Add proxy texture priors
    if len(idx_selected_proxy_active_texture):
        base_scores[:, idx_selected_proxy_active_texture] += (
            LAMBDA_PROXY_TEXTURE * prior_scores[:, idx_selected_proxy_active_texture]
        )

    # Prior-only for unmapped active species (no Perch logit, use prior only)
    if len(idx_selected_prioronly_active_event):
        base_scores[:, idx_selected_prioronly_active_event] = (
            LAMBDA_EVENT * prior_scores[:, idx_selected_prioronly_active_event]
        )
    if len(idx_selected_prioronly_active_texture):
        base_scores[:, idx_selected_prioronly_active_texture] = (
            LAMBDA_TEXTURE * prior_scores[:, idx_selected_prioronly_active_texture]
        )

    # Set unmapped inactive species to strong negative
    if len(idx_unmapped_inactive):
        base_scores[:, idx_unmapped_inactive] = -8.0

    # --- Step 3: Temporal smoothing ---
    print("Step 3: Temporal smoothing...")
    base_scores = smooth_cols_fixed12(base_scores, idx_active_texture, alpha=SMOOTH_TEXTURE)
    base_scores = smooth_events_fixed12(base_scores, idx_active_event, alpha=SMOOTH_EVENT)

    # --- Step 4: PCA on embeddings ---
    print("Step 4: StandardScaler → PCA on embeddings...")
    emb_scaled = emb_scaler.transform(embeddings)
    Z = emb_pca.transform(emb_scaled).astype(np.float32)
    print(f"  PCA output: {Z.shape}")

    del embeddings, emb_scaled
    gc.collect()

    # --- Step 5: MLP probes ---
    print(f"Step 5: Applying {len(probe_models)} MLP probes (alpha={ALPHA})...")
    mlp_scores = base_scores.copy()

    for i, (cls_idx, clf) in enumerate(probe_models.items()):
        X_cls = build_class_features(
            Z,
            raw_col=scores_raw[:, cls_idx],
            prior_col=prior_scores[:, cls_idx],
            base_col=base_scores[:, cls_idx],
        )
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_cls)[:, 1].astype(np.float32)
            pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)  # logit
        else:
            pred = clf.decision_function(X_cls).astype(np.float32)

        mlp_scores[:, cls_idx] = (1.0 - ALPHA) * base_scores[:, cls_idx] + ALPHA * pred

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(probe_models)} probes applied")

    print(f"  All {len(probe_models)} probes applied")

    # --- Step 6: Convert to probabilities and save ---
    print("Step 6: Saving outputs...")
    final_scores = mlp_scores

    meta_df = pd.DataFrame(
        {
            "filename": filenames,
            "chunk_idx": sort_order_chunks,
            "site": sites,
            "hour_utc": hours,
            "source": source,
        }
    )

    # Logit-space scores
    scores_df = pd.concat(
        [
            meta_df,
            pd.DataFrame(final_scores, columns=PRIMARY_LABELS),
        ],
        axis=1,
    )
    scores_df.to_parquet(source_dir / "perch_234_scores.parquet", index=False)
    print(f"  Saved perch_234_scores.parquet ({scores_df.shape})")

    # Probability-space
    probs = sigmoid(final_scores).astype(np.float32)
    probs_df = pd.concat(
        [
            meta_df,
            pd.DataFrame(probs, columns=PRIMARY_LABELS),
        ],
        axis=1,
    )
    probs_df.to_parquet(source_dir / "perch_234_probs.parquet", index=False)
    print(f"  Saved perch_234_probs.parquet ({probs_df.shape})")

    # --- Summary statistics ---
    print("\n=== Summary ===")
    print(f"  Segments: {N}")
    print(f"  Species: {N_CLASSES}")
    print(f"  Score range: [{final_scores.min():.3f}, {final_scores.max():.3f}]")
    print(f"  Score mean: {final_scores.mean():.3f}")
    print(f"  Prob range: [{probs.min():.6f}, {probs.max():.6f}]")

    # Coverage at thresholds
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        n_species = (probs.max(axis=0) > thresh).sum()
        n_segments = (probs.max(axis=1) > thresh).sum()
        print(f"  @>{thresh}: {n_species}/{N_CLASSES} species, {n_segments}/{N} segments")

    # NaN/inf check
    n_nan = np.isnan(final_scores).sum()
    n_inf = np.isinf(final_scores).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {n_nan} NaN, {n_inf} inf in scores!")
    else:
        print("  No NaN/inf ✓")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="train_soundscapes", choices=["train_soundscapes", "train_audio"]
    )
    args = parser.parse_args()
    main(source=args.source)
