# Pantanal-Distill Implementation Guide

Complete step-by-step breakdown of `dingjiarun/pantanal-distill-birdclef2026-improvement-a4dc68`.
Fresh pull: 2026-04-02. Source: `/tmp/pantanal-fresh/pantanal-latest.py` (3001 lines).

## Critical Finding: V17 vs V18

The notebook has **two config layers**:
- **V17 CFG** (Cell 5, lines 91-204): `d_model=256, n_ssm_layers=3, epochs=40 (submit), patience=8`
- **V18 CFG override** (Cell 6, lines 220-255): `d_model=320, n_ssm_layers=4, epochs=80, patience=20`

Cell 6 **always executes**, overriding V17 with V18. The V18 config is significantly
heavier (bigger model, more epochs, 5 TTA shifts, 5-fold OOF). This is almost
certainly why our kernels timeout — V18 training + 5-shift TTA + bigger model
eats too much of the 90-minute budget.

**The 0.930 LB score was likely achieved with V17 config** (before Cell 6 was added).
Our tasks.md already notes: "Config: V17 (d=256, 3 layers) — the actual code,
despite the file being labeled V18."

**Action**: Start with V17 config. Skip Cell 6 entirely. Only add V18 after V17 works.

---

## Timeout Root Cause Analysis

Our v2 (no cache) ran in 1110s (~18.5 min) on dry-run with 20 test files.
Real submission processes all hidden test files (~unknown count, but Perch TF
inference is ~609ms/sample = ~73s per file of 12 windows).

Key timing differences from original:
1. **TF install**: Original uses `!pip install -q --no-deps <specific.whl>` (2 wheels).
   Ours uses `subprocess + pip install --quiet` (no `--no-deps`, installs ALL wheels in dir).
2. **V18 training**: 80 epochs vs V17's 40. Bigger model (320 vs 256 dim).
3. **TTA**: V18 uses 5 shifts `[0,1,-1,2,-2]` vs V17's 3 `[0,1,-1]`.
4. **ResidualSSM**: V18 has d_model=128, 2 layers vs V17's 64, 1 layer.

---

## Execution Flow (Submit Mode)

### Phase 1: Setup (Cells 0-9)

#### Step 1: TF 2.20 Installation (Cell 1)
```
Source: ashok205/tf-wheels (notebook source, NOT a dataset)
Path:   /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/
Action: pip install -q --no-deps tensorboard-2.20.0.whl
        pip install -q --no-deps tensorflow-2.20.0.whl
```
**Critical**: Uses `--no-deps` to avoid pulling in extra packages.
**Our bug**: We don't use `--no-deps` and install ALL wheels in the directory.

#### Step 2: Mode Switch (Cell 3)
```python
MODE = "submit"
```

#### Step 3: Imports & CFG (Cell 5)
Imports: os, gc, json, re, time, warnings, collections, pathlib,
         numpy, pandas, soundfile, tensorflow, torch, sklearn, tqdm

Key constants:
```python
BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
SR = 32000; WINDOW_SEC = 5; WINDOW_SAMPLES = 160000; FILE_SAMPLES = 1920000; N_WINDOWS = 12
DEVICE = torch.device("cpu")
```

V17 CFG (the working config):
```python
CFG = {
    "mode": "submit",
    "verbose": False,
    "run_oof_baseline": False,  # skipped in submit
    "run_probe_check": False,
    "run_probe_grid": False,
    "batch_files": 16,
    "proxy_reduce": "max",
    "dryrun_n_files": 20,       # submit mode: 20 files for dry-run
    "require_full_cache_in_submit": False,
    "full_cache_input_dir": Path("/kaggle/input/perch-meta"),
    "full_cache_work_dir": Path("/kaggle/working/perch_cache"),
    "best_fusion": {
        "lambda_event": 0.4, "lambda_texture": 1.0,
        "lambda_proxy_texture": 0.8, "smooth_texture": 0.35, "smooth_event": 0.15,
    },
    "proto_ssm": {
        "d_model": 256, "d_state": 16, "n_ssm_layers": 3,
        "dropout": 0.15, "n_prototypes": 1, "n_sites": 20,
        "meta_dim": 16, "use_cross_attn": True, "cross_attn_heads": 4,
    },
    "proto_ssm_train": {
        "n_epochs": 40,          # V17 submit mode
        "lr": 1e-3, "weight_decay": 2e-3, "val_ratio": 0.15,
        "patience": 8,           # V17 submit mode
        "pos_weight_cap": 30.0, "distill_weight": 0.1, "proto_margin": 0.1,
        "label_smoothing": 0.02, "oof_n_splits": 3,
        "mixup_alpha": 0.3, "focal_gamma": 2.0,
        "swa_start_frac": 0.7, "swa_lr": 5e-4,
    },
    "frozen_best_probe": {"pca_dim": 64, "min_pos": 8, "C": 0.50, "alpha": 0.40},
    "residual_ssm": {
        "d_model": 64, "d_state": 8, "n_ssm_layers": 1,
        "dropout": 0.1, "correction_weight": 0.3,
        "n_epochs": 30, "lr": 1e-3, "patience": 8,
    },
    "temperature": {"aves": 1.10, "texture": 0.95},
    "file_level_top_k": 2,
    "tta_shifts": [0, 1, -1],
    "rank_aware_scale": True, "rank_aware_power": 0.5,
    "delta_shift_alpha": 0.15,
    "threshold_grid": [0.3, 0.4, 0.5, 0.6, 0.7],
    "probe_backend": "mlp",
    "mlp_params": {
        "hidden_layer_sizes": (128,), "activation": "relu",
        "max_iter": 300, "early_stopping": True, "validation_fraction": 0.15,
        "n_iter_no_change": 15, "random_state": 42,
        "learning_rate_init": 0.001, "alpha": 0.01,
    },
}
```

#### Step 4: V18 CFG Override (Cell 6) — SKIP FOR V17
**DO NOT include Cell 6.** It overrides to V18 config which causes timeouts.

#### Step 5: Helper Functions (Cells 7-9)
- `get_cosine_restart_scheduler()` — only used with V18's `use_cosine_restart`
- `mixup_cutmix(emb, logits, labels, alpha, cutmix_prob)` — data augmentation
- `build_class_freq_weights(Y_FULL, cap)` — inverse-sqrt frequency weighting
- `species_focal_loss(logits, targets, class_weights, gamma, label_smoothing)`

### Phase 2: Data Loading (Cells 11-14)

#### Step 6: Load Competition Data (Cell 11)
```python
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
PRIMARY_LABELS = sample_sub.columns[1:].tolist()  # 234 classes
N_CLASSES = len(PRIMARY_LABELS)
```

Functions defined:
- `parse_soundscape_labels(x)` — split ";" separated labels
- `parse_soundscape_filename(name)` — regex extract file_id, site, date, hour
- `union_labels(series)` — deduplicate labels per window

Processing:
- Deduplicate `soundscape_labels` by groupby(filename, start, end)
- Compute `start_sec`, `end_sec`, `row_id`
- Parse metadata (site, hour, month) from filenames
- Identify `full_files` (files with all 12 windows labeled) → 59 files
- Build multi-hot label matrix `Y_SC` shape (n_windows, 234)
- Build `full_truth` and `Y_FULL_TRUTH` for fully-labeled files only

#### Step 7: Class Weights (Cell 12)
```python
CLASS_WEIGHTS = build_class_freq_weights(Y_FULL_TRUTH)
```

#### Step 8: Calibration & Threshold Functions (Cell 13)
- `calibrate_and_optimize_thresholds(oof_probs, Y_FULL, threshold_grid, n_windows)`
  Uses isotonic regression + F1 grid search per class.
  **Note**: Only used in train mode or with hardcoded thresholds in V18.

#### Step 9: Ensemble Weight Sweep (Cell 14)
- `sweep_ensemble_weight(oof_proto, oof_mlp, Y_FULL, n_windows, candidates)`
  Grid search over ProtoSSM vs MLP blend weights.

### Phase 3: Model & Mapping (Cell 16)

#### Step 10: Load Perch Model & Build Taxonomy Mapping
```python
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures["serving_default"]
bc_labels = pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
```

Mapping pipeline:
1. Join taxonomy → bc_labels on scientific_name
2. `BC_INDICES[i]` = Perch's bc_index for competition class i (or NO_LABEL_INDEX if unmapped)
3. `MAPPED_MASK`, `MAPPED_POS`, `UNMAPPED_POS`, `MAPPED_BC_INDICES`
4. Classify by `CLASS_NAME_MAP` → `TEXTURE_TAXA = {"Amphibia", "Insecta"}`
5. Build index arrays: `idx_active_texture`, `idx_active_event`, etc.

Genus proxy system:
- For unmapped species (not in Perch's vocabulary), find genus-level proxies
- `proxy_map[target]` = list of bc_indices for same-genus species
- `PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}`
- `selected_proxy_pos_to_bc` = mapping from class position to bc_index array

### Phase 4: Utilities (Cells 18-19)

#### Step 11: Metrics & Smoothing (Cell 18)
- `macro_auc_skip_empty(y_true, y_score)` — macro AUROC skipping zero-count classes
- `smooth_cols_fixed12(scores, cols, alpha)` — temporal smoothing for texture taxa
- `smooth_events_fixed12(scores, cols, alpha)` — soft max-pool for event birds (Aves)
- `seq_features_1d(v)` — returns (prev, next, mean, max, std) features per window

#### Step 12: V17 Post-Processing Functions (Cell 19)
- `focal_bce_with_logits(logits, targets, gamma, pos_weight, reduction)`
- `file_level_confidence_scale(preds, n_windows, top_k)` — scale by file top-K mean
- `temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts)` — circular-shift TTA
- `rank_aware_scaling(scores, n_windows, power)` — scale by file_max^power
- `delta_shift_smooth(scores, n_windows, alpha)` — neighbor-average smoothing
- `optimize_per_class_thresholds(oof_scores, y_true, n_windows, thresholds)` — F1 grid
- `apply_per_class_thresholds(scores, thresholds, n_windows)` — sharpen around thresholds

### Phase 5: Perch Inference (Cells 21-23)

#### Step 13: Perch Inference Function (Cell 21)
- `read_soundscape_60s(path)` — read 60s OGG, mono, pad/trim to FILE_SAMPLES
- `infer_perch_with_embeddings(paths, batch_files, verbose, proxy_reduce)`:
  - Process files in batches of `batch_files` (16)
  - Each file → 12 windows of 160000 samples
  - TF inference: `infer_fn(inputs=tf.convert_to_tensor(x))`
  - Extract `label` logits and `embedding` (1536-dim)
  - Map Perch logits to competition classes via `MAPPED_BC_INDICES`
  - Apply genus proxies for unmapped species
  - Returns: `meta_df`, `scores` (n_rows, 234), `embeddings` (n_rows, 1536)

#### Step 14: Cache Management (Cell 23)
- `resolve_full_cache_paths()` — checks working dir, legacy paths, and attached dataset
- If cache found: load `full_perch_meta.parquet` + `full_perch_arrays.npz`
- If no cache: run Perch inference on 59 fully-labeled files, save cache
- Align `full_truth` to cached row order → `Y_FULL`

**With perch-meta cache**: Skips labeled file inference (~10 min saved)
**Without cache**: Must run Perch on 59 files (~10 min)

### Phase 6: Prior Tables (Cell 25)

#### Step 15: Fit Prior Tables
- `fit_prior_tables(prior_df, Y_prior)`:
  - Computes `global_p`, `site_p`, `hour_p`, `site_hour_p` priors
  - Bayesian-smoothed: weight = n / (n + shrinkage)
- `prior_logits_from_tables(sites, hours, tables)`:
  - Combines global → hour → site → site-hour priors
  - Returns log-odds
- `fuse_scores_with_tables(base_scores, sites, hours, tables, ...)`:
  - Adds prior logits to base scores with per-category lambdas
  - Applies temporal smoothing (texture: average, events: soft max-pool)
  - Prior-only classes get pure prior scores
  - Inactive unmapped classes → -8.0

### Phase 7: OOF Stacking (Cell 27)

#### Step 16: Build OOF Base/Prior Features
- `build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5)`:
  - 5-fold GroupKFold (grouped by filename)
  - Each fold: fit fold-safe priors (exclude val files), fuse val scores
  - Returns `oof_base`, `oof_prior`, `fold_id`
- Caches to `full_oof_meta_features.npz`

### Phase 8: Classwise Probes (Cell 29)

#### Step 17: Embedding Probe System
- `build_class_features(emb_proj, raw_col, prior_col, base_col)`:
  - Concatenates: PCA embedding + raw + prior + base + 5 sequential features + 3 diffs + 3 interactions = d+15 features
- `run_oof_embedding_probe(scores_raw, emb, meta_df, y_true, pca_dim, min_pos, C, alpha)`:
  - 5-fold GroupKFold
  - Per-fold: StandardScaler → PCA → per-class MLP probe
  - MLP uses oversampling for class imbalance
  - Probe prediction blended: `(1-alpha) * base + alpha * probe_logit`
  - **Only in train mode** (for tuning)

### Phase 9: ProtoSSM Architecture (Cell 31)

#### Step 18: Model Classes
- `SelectiveSSM(d_model, d_state, d_conv)` — Mamba-style SSM with sequential scan
- `TemporalCrossAttention(d_model, n_heads, dropout)` — MHA + FFN with residual
- `ProtoSSMv2(d_input, d_model, d_state, n_ssm_layers, n_classes, n_windows, ...)`:
  - input_proj: Linear(1536→d_model) + LayerNorm + GELU + Dropout
  - pos_enc: learnable (1, 12, d_model)
  - site_emb + hour_emb → meta_proj
  - Bidirectional SSM layers (fwd + bwd → merge → norm)
  - Cross-attention (if enabled)
  - Prototypical cosine similarity + class_bias + gated fusion with Perch logits
  - Optional family head

V17 params: d_model=256, d_state=16, 3 SSM layers, dropout=0.15, cross_attn_heads=4

### Phase 10: ProtoSSM Training (Cell 33)

#### Step 19: Training Infrastructure
- `build_taxonomy_groups(taxonomy_df, primary_labels)` — family/order grouping
- `build_site_mapping(meta_df)` — site string → integer index (1-based, 0=unknown)
- `reshape_to_files(flat_array, meta_df, n_windows)` — (N*12, D) → (N, 12, D)
- `get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max)` — per-file site/hour
- `mixup_files(emb, logits, labels, site_ids, hours, families, alpha)` — file-level mixup

- `train_proto_ssm_single(model, emb_train, logits_train, labels_train, ...)`:
  - Label smoothing, mixup (skip first 5 epochs), focal BCE loss
  - Knowledge distillation: MSE(output, perch_logits)
  - Taxonomic auxiliary: BCE on family head
  - Total = main + 0.1*distill + 0.1*family
  - AdamW + OneCycleLR
  - SWA: accumulate checkpoints after swa_start_frac of epochs
  - Early stopping on val_loss with patience
  - Apply SWA average if ≥3 checkpoints, else restore best

- `run_proto_ssm_oof(...)` — GroupKFold OOF for ProtoSSM (train mode only)
- `optimize_ensemble_weight(oof_proto_flat, oof_mlp_flat, y_true_flat)` — grid search

### Phase 11: Probe Tuning & Final Fit (Cells 35-39)

#### Step 20: Probe Tuning (Cell 35) — **Submit: skip grid, use frozen params**
```python
BEST_PROBE = CFG["frozen_best_probe"]  # {"pca_dim": 64, "min_pos": 8, "C": 0.50, "alpha": 0.40}
```

#### Step 21: Final Prior Tables (Cell 38)
```python
final_prior_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)
```
Uses ALL labeled data (no fold-safe exclusion — this is for inference).

#### Step 22: Final Scaler + PCA (Cell 39)
```python
emb_scaler = StandardScaler().fit(emb_full)
emb_pca = PCA(n_components=min(pca_dim, ...)).fit(emb_full_scaled)
Z_FULL = emb_pca.transform(emb_full_scaled)
```

### Phase 12: ProtoSSM Training on All Data (Cell 41)

#### Step 23: Reshape & Setup
```python
emb_files, file_list = reshape_to_files(emb_full, meta_full)      # (59, 12, 1536)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)     # (59, 12, 234)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)              # (59, 12, 234)
```

Build site mapping, file metadata, file family labels.

#### Step 24: OOF Cross-Validation — **Submit: SKIP**
```python
ENSEMBLE_WEIGHT_PROTO = 0.5  # default, not optimized in submit mode
```
**Note**: In train mode, the OOF optimization sets `ENSEMBLE_WEIGHT_PROTO = 0.0`
(MLP-only wins), but in submit mode, it defaults to 0.5. This is a potential
discrepancy if the 0.930 was achieved with `ENSEMBLE_WEIGHT_PROTO = 0.0`.

#### Step 25: Train Final ProtoSSM
```python
model = ProtoSSMv2(d_input=1536, d_model=256, ...)
model.init_prototypes_from_data(emb_flat, labels_flat)
model.init_family_head(n_families, class_to_family)
model, history = train_proto_ssm_single(model, emb_files, logits_files, labels_files, ...)
```
V17 submit: 40 epochs, patience=8, no validation (trains on all data).

#### Step 26: Train MLP Probes
For each class with ≥ min_pos positive samples:
```python
X_cls = build_class_features(Z_FULL, raw_col, prior_col, base_col)
# Oversample positives, fit MLPClassifier
probe_models[cls_idx] = clf
```

#### Step 27: Ensemble Weight — **Submit: use default 0.5**

### Phase 13: Residual SSM (Cell 43)

#### Step 28: Wall-Time Safety Check
```python
_wall_min = (time.time() - _WALL_START) / 60.0
if _wall_min < 4.0:  # Only runs if everything above took < 4 minutes
    # Train ResidualSSM...
```
**With perch-meta cache**: wall_min ≈ 2-3 min → ResidualSSM runs
**Without cache**: wall_min ≈ 12+ min → ResidualSSM SKIPPED

#### Step 29: ResidualSSM Architecture & Training
- `ResidualSSM`: concat(emb, first_pass_scores) → project → 1-layer BiSSM → linear head
- Trains on residuals = labels - sigmoid(first_pass_ensemble)
- MSE loss, 30 epochs (V17), early stopping

### Phase 14: Test Inference (Cells 46-48)

#### Step 30: Perch Inference on Test Files (Cell 46)
```python
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
# If no test files → dry-run on first 20 train soundscapes
meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(test_paths, ...)
```
**This is the slowest step** — ~609ms per 5s window with TF.

#### Step 31: Score Fusion (Cell 48)
1. **ProtoSSM inference with TTA**:
   ```python
   proto_scores = temporal_shift_tta(emb_test_files, logits_test_files, model, ...)
   ```
   V17: 3 shifts → 3 forward passes
2. **Prior-fused base scores**:
   ```python
   test_base_scores, test_prior_scores = fuse_scores_with_tables(scores_test_raw, ...)
   ```
3. **MLP probe scores**: Per-class probe prediction blended with base
4. **Ensemble fusion**:
   ```python
   final = ENSEMBLE_WEIGHT_PROTO * proto + (1 - ENSEMBLE_WEIGHT_PROTO) * mlp
   ```
5. **Residual SSM correction** (if trained):
   ```python
   final += CORRECTION_WEIGHT * res_model(emb, first_pass)
   ```

### Phase 15: Post-Processing & Submission (Cells 50-51)

#### Step 32: Post-Processing Pipeline (Cell 50)
1. **Per-taxon temperature scaling**: Aves=1.10, Texture=0.95
2. **Sigmoid**: `probs = sigmoid(scores / temperature)`
3. **File-level confidence scaling**: top-K=2
4. **Rank-aware scaling**: power=0.5 (V17) — scale by file_max^power
5. **Adaptive delta shift smoothing**: alpha=0.15 (V17), confidence-adaptive
6. **Per-class threshold sharpening**: V18 uses hardcoded thresholds, V17 uses default 0.5

**V17 note**: The `PER_CLASS_THRESHOLDS` in Cell 50 are V18-specific hardcoded values.
For V17, we should either use default 0.5 for all classes or compute from OOF.

#### Step 33: Build Submission CSV
```python
submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission.to_csv("submission.csv", index=False)
```

---

## Key Differences Between Original & Our Kernel

| Aspect | Original | Ours | Impact |
|--------|----------|------|--------|
| TF install | `!pip install -q --no-deps <2 wheels>` | `subprocess + pip install --quiet <all wheels>` | Slower install, possible dep conflicts |
| TF wheel source | `ashok205/tf-wheels` (notebook) | `birdclef-tf220-wheel` (our dataset) | Different path resolution |
| V18 override | Cell 6 always runs | Cell 6 always runs | **Both have V18** — both timeout? |
| Wall-time var | `_WALL_START` set after TF import | Same | Same behavior |
| Safety cutoff | `TIMEOUT_SECONDS` unused | `TIMEOUT_SECONDS` unused | Neither uses it |

## Recommended Implementation Plan

### V17 First (target: match 0.930)

1. **Remove V18 override entirely** (no Cell 6)
2. **Fix TF install**: use `--no-deps`, only install the 2 specific wheels
3. **Use perch-meta cache** to skip labeled inference
4. **Use V17 post-processing**: default thresholds (0.5), V17 smoothing params
5. **Verify timing**: dry-run should be well under budget

### V18 Later (if V17 works)

Only after V17 scores ≥0.920 on LB, selectively add V18 improvements.
