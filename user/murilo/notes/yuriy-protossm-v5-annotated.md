# Yuriy ProtoSSM v5 Time-Optimised 0.925 — Annotated Pipeline Reference

Source: `kaggle kernels pull yuriygreben/birdclef-2026-protossm-v5-time-optimised`
Full extracted code: `user/murilo/experiments/018-perch-probes/kaggle_kernel/ref_yuriy_protossm_v5.py`

---

## Settings (frozen for submit mode)

```python
CFG = {
    "batch_files": 16,
    "proxy_reduce": "max",         # grid-searchable: max vs mean

    # Prior fusion (same as chiranjithdharma)
    "best_fusion": {
        "lambda_event": 0.4,
        "lambda_texture": 1.0,
        "lambda_proxy_texture": 0.8,
        "smooth_texture": 0.35,
        "smooth_event": 0.15,      # NOTE: event smoothing IS included (unlike chiranjithdharma)
    },

    # ProtoSSM v4 architecture
    "proto_ssm": {
        "d_model": 128,            # NOTE: 128 in CFG, but doc says 256 — see actual model init
        "d_state": 16,
        "n_ssm_layers": 2,
        "dropout": 0.15,
        "n_sites": 20,
        "meta_dim": 16,
        "use_cross_attn": True,
        "cross_attn_heads": 4,
    },

    # ProtoSSM training
    "proto_ssm_train": {
        "n_epochs": 60,
        "lr": 1e-3,
        "weight_decay": 2e-3,
        "patience": 15,
        "pos_weight_cap": 30.0,
        "distill_weight": 0.1,
        "label_smoothing": 0.02,
        "mixup_alpha": 0.3,
        "focal_gamma": 2.0,
        "swa_start_frac": 0.7,
        "oof_n_splits": 3,
    },

    # Frozen probe params
    "frozen_best_probe": {
        "pca_dim": 64,             # ← 64 not 32
        "min_pos": 8,
        "C": 0.50,                 # ← 0.50 not 0.25
        "alpha": 0.40,
    },

    # Probe backend: MLP (not LogReg)
    "probe_backend": "mlp",
    "mlp_params": {
        "hidden_layer_sizes": (128,),
        "activation": "relu",
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "learning_rate_init": 0.001,
        "alpha": 0.01,
    },

    # ResidualSSM
    "residual_ssm": {
        "d_model": 64,
        "d_state": 8,
        "n_ssm_layers": 1,
        "dropout": 0.1,
        "correction_weight": 0.3,
        "n_epochs": 30,
        "lr": 1e-3,
        "patience": 8,
    },

    # Post-processing
    "temperature": {"aves": 1.10, "texture": 0.95},
    "file_level_top_k": 2,
    "tta_shifts": [0, 1, -1],
    "rank_aware_scale": True,
    "rank_aware_power": 0.5,
    "delta_shift_alpha": 0.15,
    "threshold_grid": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
}
```

## Data Loading

Uses `jaejohn/perch-meta` cache for labeled soundscapes (same as chiranjithdharma):
- `full_perch_meta.parquet` — row_id, filename, site, hour_utc
- `full_perch_arrays.npz` — `scores_full_raw` (N, 234), `emb_full` (N, 1536)

Only **fully-labeled files** (all 12 windows annotated).

Also reads `train_soundscapes_labels.csv` directly to build `sc_clean` and `Y_SC`
for prior tables (same pattern as chiranjithdharma — richer priors from all annotations).

## Species Mapping

Same 6-category split as chiranjithdharma, with one key difference:

```python
# Proxy targets include Amphibia, Insecta, AND Aves (not just Amphibia!)
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
SELECTED_PROXY_TARGETS = sorted([
    t for t in proxy_map.keys()
    if CLASS_NAME_MAP.get(t) in PROXY_TAXA
])
```

Also has **proxy_reduce grid search** (train mode): tests `max` vs `mean` for genus proxy
logit aggregation. Default frozen at `max`.

## fuse_scores_with_tables

Same 6-category structure as chiranjithdharma, plus **event smoothing** which
chiranjithdharma does NOT include:

```python
def fuse_scores_with_tables(base_scores, sites, hours, tables, ...):
    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, tables)

    scores[:, idx_mapped_active_event]   += lambda_event * prior[...]
    scores[:, idx_mapped_active_texture] += lambda_texture * prior[...]
    scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[...]
    scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[...]
    scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[...]
    scores[:, idx_unmapped_inactive] = -8.0

    # BOTH smoothing types applied (unlike chiranjithdharma which skips events)
    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores, prior
```

Event smoothing uses **local-max propagation** (not neighbor averaging):
```python
def smooth_events_fixed12(scores, cols, alpha=0.15):
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
```

## OOF Pipeline

### Step 1: OOF fold-safe priors (grouped by FILENAME, not site)

```python
def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5):
    groups_full = meta_full["filename"].to_numpy()  # ← by filename!
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(..., groups=groups_full)):
        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())

        # Fold-safe: exclude val FILES from sc_clean
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        tables = fit_prior_tables(sc_clean[prior_mask], Y_SC[prior_mask])

        oof_base[va_idx], oof_prior[va_idx] = fuse_scores_with_tables(
            scores_full_raw[va_idx], val_sites, val_hours, tables
        )
```

### Step 2: Global PCA (64 dims, fit once on all data)

```python
emb_scaler = StandardScaler()
emb_pca = PCA(n_components=64)
Z_FULL = emb_pca.fit_transform(emb_scaler.fit_transform(emb_full))
```

### Step 3: build_class_features (77-dim)

```python
def build_class_features(emb_proj, raw_col, prior_col, base_col):
    prev, next, mean, max, std = seq_features_1d(base_col)

    diff_mean = base_col - mean   # deviation from file mean
    diff_prev = base_col - prev   # onset detection
    diff_next = base_col - next   # offset detection

    return concat([
        emb_proj,                 # PCA embeddings (64)
        raw_col,                  # raw Perch logit (1)
        prior_col,                # prior logit (1)
        base_col,                 # fused base logit (1)
        prev,                     # prev window base (1)
        next,                     # next window base (1)
        mean,                     # file mean base (1)
        max,                      # file max base (1)
        std,                      # file std base (1)     ← NEW
        diff_mean,                # deviation from mean (1) ← NEW
        diff_prev,                # onset (1)              ← NEW
        diff_next,                # offset (1)             ← NEW
        raw * prior,              # interaction (1)        ← NEW
        raw * base,               # interaction (1)        ← NEW
        prior * base,             # interaction (1)        ← NEW
    ])  # total: 64 + 13 = 77 dims
```

Differences from chiranjithdharma's 39-dim (PCA32 + 7):
- PCA: 32 → 64 (+32)
- Added: std, diff_mean, diff_prev, diff_next (+4)
- Added: raw*prior, raw*base, prior*base (+3)
- Removed: nothing
- Net: 39 → 77 (+38 dims)

### Step 4: MLP Probes (not LogReg)

```python
# Positive oversampling (MLPClassifier doesn't support class_weight)
if n_pos > 0 and n_neg > n_pos:
    repeat = max(1, n_neg // n_pos)
    pos_idx = np.where(y_tr == 1)[0]
    X_bal = np.vstack([X_tr, np.tile(X_tr[pos_idx], (repeat, 1))])
    y_bal = np.concatenate([y_tr, np.ones(len(pos_idx) * repeat)])

clf = MLPClassifier(
    hidden_layer_sizes=(128,), activation="relu",
    max_iter=300, early_stopping=True,
    validation_fraction=0.15, learning_rate_init=0.001, alpha=0.01,
)
clf.fit(X_bal, y_bal)

# Output → logit space via manual log-odds
pred = clf.predict_proba(X)[:, 1]
pred_logit = log(pred + 1e-7) - log(1 - pred + 1e-7)
```

Key: MLP uses `predict_proba` → manual log-odds conversion, not `decision_function`
like LogReg. The oversampling balances classes since MLPClassifier doesn't have
`class_weight='balanced'`.

Probe blending: `(1 - 0.40) * base + 0.40 * probe_logit` (fixed alpha=0.40)

### Step 5: Final probes on all data with OOF features

Same pattern as chiranjithdharma: global PCA + probes trained on all data
using `oof_base`/`oof_prior` as features.

## ProtoSSM v4 Architecture

```
Input: emb (B, 12, 1536) + perch_logits (B, 12, 234)

1. input_proj: Linear(1536 → d_model) + LayerNorm + GELU + Dropout(0.15)
2. pos_enc: Learnable (1, 12, d_model)
3. meta_emb: site_emb(20, 16) + hour_emb(24, 16) → Linear(32 → d_model)
4. BiSSM layers × n_ssm_layers:
   - forward SSM + backward SSM (flipped) → Linear(2*d_model → d_model) + norm + residual
5. Cross-attention: 4-head MHA + FFN(d_model→2*d_model→d_model) + pre-norm residual
6. Prototypical head: cosine_sim(h, prototypes) × learned_temp + class_bias
7. Gated fusion: alpha * proto_logits + (1-alpha) * perch_logits
   (alpha = per-class learned sigmoid gate)
8. Taxonomic aux head: Linear(d_model → n_families) on mean-pooled h
```

**SelectiveSSM** (Mamba-style):
- in_proj(2D), conv1d(d_model, groups=d_model, kernel=4), SiLU
- dt_proj, A_log (learnable), B_proj, C_proj
- Sequential scan (efficient for T=12)
- out_proj + skip connection

## ProtoSSM Training

**Multi-task loss**:
- Primary: Focal BCE (gamma=2.0) with class pos_weight (capped at 30)
- Distillation: 0.1 * MSE(output, perch_logits) — knowledge distillation from Perch
- Taxonomic: 0.1 * BCE(family_head, family_labels)

**Training enhancements**:
- AdamW (weight_decay=2e-3), OneCycleLR (pct_start=0.1, cos anneal)
- Label smoothing: 0.02
- File-level MixUp: alpha=0.3, skip first 5 warmup epochs, lambda>=0.5
- SWA: start at 70% of epochs, average if >=3 checkpoints
- Early stopping: patience=15, gradient clip=1.0
- OOF: 3-fold GroupKFold by site

**Prototype initialization**: Class-mean embeddings from training data (normalized).

## ResidualSSM (Second-Pass Boosting)

```
Input: concat(emb 1536 + first_pass_scores 234) = 1770-dim

1. Linear(1770 → 64) + LayerNorm + GELU + Dropout(0.1)
2. Metadata: site_emb(20, 8) + hour_emb(24, 8) → Linear(16 → 64)
3. Learnable pos_enc (1, 12, 64)
4. 1× BiSSM layer (d_model=64, d_state=8)
5. Linear(64 → 234), zero-initialized (corrections start small)
```

**Training**: 30 epochs, MSE loss on residuals (target - sigmoid(first_pass)),
patience=8, 15% val split.

**Application**: `final = first_pass + 0.3 * correction`

**Wall-time safety**: Skips if >4 min elapsed (preserves time for test inference).

## Score Fusion Pipeline

```
1. ProtoSSM v4 inference with TTA (circular shifts [0, +1, -1])
2. Prior-fused base scores via metadata tables
3. MLP probe scores (77-dim features)
4. Ensemble: W * ProtoSSM + (1-W) * MLP
   (W=0.5 default, OOF-optimized in train mode)
5. ResidualSSM correction: + 0.3 * residual_output
```

## Post-Processing Pipeline (6 stages)

All applied in PROBABILITY SPACE (after sigmoid), not logit space:

```python
# 1. Per-taxon temperature scaling (applied before sigmoid)
scaled_scores = final_logits / T   # T_aves=1.10, T_texture=0.95
probs = sigmoid(scaled_scores)

# 2. File-level confidence scaling
top_k_mean = sort(probs, axis=1)[:, -2:, :].mean(axis=1)
probs = probs * top_k_mean

# 3. Rank-aware scaling (2025 Rank 3 technique)
file_max = probs.reshape(F, 12, C).max(axis=1)
probs *= file_max ** 0.5

# 4. Delta shift smoothing (2025 Rank 1 technique)
probs = (1-0.15)*probs + 0.075*(prev + next)

# 5. Per-class threshold sharpening (OOF-optimized in train, default 0.5 in submit)
above_t: 0.5 + 0.5 * (score - t) / (1 - t)
below_t: 0.5 * score / t
```

NOTE: In submit mode, per-class thresholds are fixed at 0.5 (no OOF optimization).
This means the threshold sharpening in submit mode is a no-op (identity transform).

## Key Differences from chiranjithdharma (0.910)

| Aspect | chiranjithdharma | Yuriy ProtoSSM v5 |
|--------|-----------------|-------------------|
| **Probe features** | 39-dim (PCA32 + 7) | 77-dim (PCA64 + 13) |
| **Probe type** | LogReg (C=0.25) | MLPClassifier (128,) |
| **PCA dim** | 32 | 64 |
| **OOF grouping** | by site | by filename |
| **Event smoothing** | None in fuse_scores | local-max (alpha=0.15) |
| **Proxy taxa** | Amphibia only | Amphibia + Insecta + Aves |
| **proxy_reduce** | fixed max | grid-searchable (max/mean) |
| **ProtoSSM** | None | d_model=128, 2 BiSSM + cross-attn |
| **ResidualSSM** | None | d=64, 1 BiSSM, zero-init |
| **TTA** | None | circular shifts [0, +1, -1] |
| **Post-processing** | Gaussian smoothing only | 6-stage pipeline |
| **Ensemble** | Probes only | W*ProtoSSM + (1-W)*MLP |
| **Test alpha** | dynamic per species | fixed 0.40 |
| **Final smoothing** | Gaussian [0.1,0.2,0.4,0.2,0.1] | Delta shift (alpha=0.15) |

## Key Differences from jaejohn (perch-meta producer)

| Aspect | jaejohn | Yuriy ProtoSSM v5 |
|--------|---------|-------------------|
| **PCA dim** | 64 | 64 (same) |
| **Probe C** | 0.50 | 0.50 (same) |
| **Probe type** | LogReg | MLP (128,) |
| **Features** | 39-dim | 77-dim (+std, diffs, interactions) |
| **OOF grouping** | by filename | by filename (same) |
| **Test alpha** | fixed 0.40 | fixed 0.40 (same) |
| **Final smoothing** | None | 6-stage post-processing |
| **ProtoSSM** | None | Full ensemble |

## Incremental Path from Our v2-fix → This Notebook

### Step 1: Richer features (39 → 77 dims) — "v3"
- PCA 32 → 64
- Add: std, diff_mean, diff_prev, diff_next
- Add: raw*prior, raw*base, prior*base
- Estimated impact: ~0.003-0.005 LB

### Step 2: MLP probes (LogReg → MLPClassifier) — "v3"
- MLPClassifier(128,) with positive oversampling
- predict_proba → manual log-odds
- Estimated impact: ~0.001-0.003 LB

### Step 3: ProtoSSM temporal model — "v4"
- BiSSM + cross-attention on file-level sequences
- Gated fusion with Perch logits
- OOF-optimized ensemble weight (W * ProtoSSM + (1-W) * MLP)
- Estimated impact: ~0.005-0.010 LB

### Step 4: ResidualSSM boosting — "v4"
- Second-pass error correction
- Zero-initialized corrections (conservative)
- Estimated impact: ~0.002-0.004 LB

### Step 5: Post-processing pipeline — "v5"
- Per-taxon temperature, file-level scaling, rank-aware, delta-shift, thresholds
- Estimated impact: ~0.005-0.008 LB

### Step 6: Event smoothing in fuse_scores
- local-max propagation for event species (alpha=0.15)
- We currently skip this — chiranjithdharma also skips it
- Estimated impact: ~0.001-0.002 LB

### Step 7: Proxy taxa expansion (Amphibia → Amphibia+Insecta+Aves)
- More species get genus proxy scores instead of -10.0 / prior-only
- Estimated impact: ~0.001 LB

## Runtime Budget (90 min Kaggle kernel)

- TF install + data loading: ~1 min
- Prior tables + OOF: ~2s
- ProtoSSM training (60 epochs): ~10s
- MLP probes: ~5s
- ResidualSSM training (30 epochs): ~3s
- **Perch on ~600 test files: ~80 min** (dominant cost)
- ProtoSSM inference + TTA: ~1s
- Post-processing: ~1s
- Total: ~82 min (fits within 90 min)

## Key Takeaways

1. **All training happens in-notebook** — no pre-trained artifacts, fully reproducible
2. **77-dim features are the biggest probe-level improvement** — std, diff, and interaction
   features capture temporal dynamics that 39-dim misses
3. **ProtoSSM is lightweight** — ~10s training on 59 files, minimal parameter count
4. **ResidualSSM is conservative** — zero-initialized, small correction weight (0.3)
5. **Post-processing is borrowed from 2025 winners** — rank-aware and delta-shift
   are proven techniques from prior competitions
6. **jaejohn's params (PCA=64, C=0.50) used** — not chiranjithdharma's (PCA=32, C=0.25)
7. **OOF by filename** (like jaejohn), not by site (like chiranjithdharma)
8. **Submit mode uses default thresholds (0.5)** — per-class optimization only in train mode
