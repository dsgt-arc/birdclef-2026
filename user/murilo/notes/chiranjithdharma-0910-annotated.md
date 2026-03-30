# chiranjithdharma 0.910 — Annotated Pipeline Reference

Source: `kaggle kernels pull chiranjithdharma/0-910-score`
Full extracted code: `user/murilo/experiments/018-perch-probes/kaggle_kernel/ref_chiranjithdharma_0910.py`

---

## Settings (frozen for submit mode)

```python
PROBE_PCA_DIM = 32
PROBE_MIN_POS = 8
PROBE_C       = 0.25
PROBE_ALPHA   = 0.40          # fixed alpha in Settings, but dynamic at test time (see below)

LAMBDA_EVENT         = 0.4    # birds
LAMBDA_TEXTURE       = 1.0    # frogs/insects
LAMBDA_PROXY_TEXTURE = 0.8    # genus proxy amphibians
SMOOTH_TEXTURE_ALPHA = 0.35   # texture smoothing inside fuse_scores

SR = 32000
WINDOW_SEC = 5
N_WINDOWS = 12
BATCH_FILES = 16              # batch Perch inference (16 files at once)
```

## Data Loading

Uses `jaejohn/perch-meta` cache for labeled soundscapes:
- `full_perch_meta.parquet` — row_id, filename, site, hour_utc
- `full_perch_arrays.npz` — `scores_full_raw` (N, 234), `emb_full` (N, 1536)

Only **fully-labeled files** (all 12 windows annotated) are used. This gives
59 files / 708 windows (not 66 files / 792 — 7 files are partially labeled).

## Species Mapping

```python
# Direct mapping via taxonomy → Perch labels join on scientific_name
BC_INDICES = [label_to_bc[c] for c in PRIMARY_LABELS]
MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX

# Genus proxy: unmapped non-sonotype Amphibia only
for row in unmapped_non_sonotype:
    genus = scientific_name.split()[0]
    hits = bc_labels matching genus
    proxy_map[primary_label] = hits['bc_index'].tolist()
# At inference: scores[:, pos] = logits[:, proxy_bc_indices].max(axis=1)
```

Six species categories in `fuse_scores`:
1. `idx_mapped_active_event` — direct-mapped birds → additive prior (λ=0.4)
2. `idx_mapped_active_texture` — direct-mapped frogs/insects → additive prior (λ=1.0)
3. `idx_selected_proxy_active_texture` — genus proxy frogs → additive prior (λ=0.8)
4. `idx_selected_prioronly_active_event` — unmapped active birds → score REPLACED by prior
5. `idx_selected_prioronly_active_texture` — unmapped active frogs → score REPLACED by prior
6. `idx_unmapped_inactive` — unmapped inactive → score = -8.0

## Prior Tables

```python
def fit_prior_tables(prior_df, Y_prior):
    """Bayesian site/hour/site-hour tables. Input is a DataFrame with
    'site' and 'hour_utc' columns, and Y_prior is the label matrix."""
    global_p = Y_prior.mean(axis=0)

    # Site priors: P(species | site)
    for s in unique_sites:
        mask = prior_df['site'] == s
        site_n[s] = mask.sum()
        site_p[s] = Y_prior[mask].mean(axis=0)

    # Hour priors: P(species | hour)
    for h in unique_hours:
        mask = prior_df['hour_utc'] == h
        hour_n[h] = mask.sum()
        hour_p[h] = Y_prior[mask].mean(axis=0)

    # Site-hour interaction
    for (s, h) in groupby(['site', 'hour_utc']):
        sh_n[(s,h)] = count
        sh_p[(s,h)] = Y_prior[idx].mean(axis=0)
```

```python
def prior_logits(sites, hours, tables, eps=1e-4):
    """Convert probability tables to logits. Bayesian smoothing k values:
    k_hour=8, k_site=8, k_interaction=4."""
    p = global_p.copy()

    # Hour blending: p = w*hour_p + (1-w)*p  where w = n/(n+8)
    # Site blending: p = w*site_p + (1-w)*p  where w = n/(n+8)
    # Interaction:   p = w*sh_p   + (1-w)*p  where w = n/(n+4)

    clip(p, eps, 1-eps)
    return log(p / (1-p))  # logits
```

## Smoothing (inside fuse_scores)

Only texture species get smoothed inside `fuse_scores`:

```python
def smooth_cols(scores, cols, alpha=0.35):
    """Neighbor averaging for texture species. Applied in logit space."""
    view = scores.reshape(-1, N_WINDOWS, n_species)
    x = view[:, :, cols]
    prev = concat([x[:, :1], x[:, :-1]], axis=1)
    nxt  = concat([x[:, 1:], x[:, -1:]], axis=1)
    view[:, :, cols] = (1-alpha)*x + 0.5*alpha*(prev + nxt)
```

No event smoothing inside `fuse_scores` — only texture. Event species get the
Gaussian final smoothing applied later (see below).

## OOF Pipeline (the critical part)

### Step 1: OOF fold-safe priors

```python
gkf    = GroupKFold(n_splits=5)
groups = meta_full['site'].to_numpy()      # group by SITE, not file

oof_base  = np.zeros_like(scores_full_raw)  # fold-safe base scores
oof_prior = np.zeros_like(scores_full_raw)  # fold-safe prior logits

for _, va_idx in gkf.split(scores_full_raw, groups=groups):
    va_idx    = np.sort(va_idx)
    val_sites = set(meta_full.iloc[va_idx]['site'])

    # Fold-safe: exclude val sites from prior table training data
    prior_mask = ~sc_clean['site'].isin(val_sites).values
    tables = fit_prior_tables(sc_clean[prior_mask], Y_SC[prior_mask])

    # Fuse val scores with fold-safe priors
    oof_base[va_idx], oof_prior[va_idx] = fuse_scores(
        scores_full_raw[va_idx], val_sites, val_hours, tables
    )
```

**Key**: `oof_base` and `oof_prior` contain "honest" prior-fused scores where
each sample's priors were computed WITHOUT seeing that sample's site. These
arrays are used as FEATURES for probe training (next step).

### Step 2: Global PCA (fit once on all data)

```python
emb_scaler = StandardScaler()
emb_scaled = emb_scaler.fit_transform(emb_full)
emb_pca = PCA(n_components=32)
Z_FULL = emb_pca.fit_transform(emb_scaled)
```

PCA is NOT per-fold. Single global fit on all labeled data.

### Step 3: Train final probes on ALL data with OOF features

```python
for cls_idx in probe_idx:
    y = Y_FULL[:, cls_idx]
    X = build_class_features(
        Z_FULL,                              # global PCA
        raw_col=scores_full_raw[:, cls_idx], # raw Perch logits
        prior_col=oof_prior[:, cls_idx],     # ← OOF fold-safe prior!
        base_col=oof_base[:, cls_idx],       # ← OOF fold-safe base!
    )
    clf = LogisticRegression(C=0.25, max_iter=400, solver='liblinear',
                             class_weight='balanced')
    clf.fit(X, y)
    probe_models[cls_idx] = clf
```

**CRITICAL**: The probes see `oof_base` and `oof_prior` as features, NOT
all-data base/prior. This prevents probes from exploiting leaked prior signal.

### Step 4: Features (39-dim)

```python
def build_class_features(Z, raw_col, prior_col, base_col):
    prev, nxt, mean, max = seq_features_1d(base_col)  # temporal context
    return concat([
        Z,                    # PCA embeddings (32)
        raw_col[:, None],     # raw Perch logit (1)
        prior_col[:, None],   # prior logit (1)
        base_col[:, None],    # fused base logit (1)
        prev[:, None],        # prev window base (1)
        nxt[:, None],         # next window base (1)
        mean[:, None],        # file mean base (1)
        max[:, None],         # file max base (1)
    ])  # total: 39 dims
```

## Test-Time Inference

```python
# All-data priors for test (no leakage — test is unseen)
final_tables = fit_prior_tables(sc_clean, Y_SC)

# Perch inference on ~600 test files (~80 min)
meta_test, scores_test_raw, emb_test = infer_perch_batch(test_paths)

# Prior fusion with all-data tables
test_base, test_prior = fuse_scores(scores_test_raw, test_sites, test_hours, final_tables)

# PCA transform
Z_TEST = emb_pca.transform(emb_scaler.transform(emb_test))

# Apply probes with DYNAMIC alpha (not fixed 0.40)
final_scores = test_base.copy()
for cls_idx, clf in probe_models.items():
    X = build_class_features(Z_TEST, scores_test_raw[:, cls_idx],
                             test_prior[:, cls_idx], test_base[:, cls_idx])
    pred = clf.decision_function(X)     # logit output
    n_pos = int(Y_FULL[:, cls_idx].sum())
    alpha = 0.25 + 0.25 * min(1.0, (n_pos - 8) / 40.0)   # dynamic!
    final_scores[:, cls_idx] = (1-alpha) * test_base[:, cls_idx] + alpha * pred
```

Note: `PROBE_ALPHA = 0.40` in Settings is for train-mode grid search only.
The actual test-time code uses the dynamic formula `0.25 + 0.25 * scale`.

## Gaussian Final Smoothing

Applied to ALL species AFTER probes, before sigmoid:

```python
from scipy.ndimage import convolve1d

def gauss_smooth_final(scores, weights=np.array([0.1, 0.2, 0.4, 0.2, 0.1])):
    smoothed = scores.reshape(-1, N_WINDOWS, n_species).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
    return smoothed.reshape(-1, n_species)

final_scores_smoothed = gauss_smooth_final(final_scores)
submission = sigmoid(final_scores_smoothed)
```

This is a 5-point Gaussian kernel: `[0.1, 0.2, 0.4, 0.2, 0.1]`.
Different from the texture-only neighbor smoothing inside `fuse_scores`.

## Train-Mode Grid Search (not used in submit)

When `MODE='train'`, runs a param grid with OOF probe evaluation:

```python
param_grid = [
    {'pca_dim': 32, 'min_pos': 8,  'C': 0.25, 'alpha': 0.40},
    {'pca_dim': 64, 'min_pos': 8,  'C': 0.25, 'alpha': 0.40},
    {'pca_dim': 64, 'min_pos': 8,  'C': 0.25, 'alpha': 0.50},
    {'pca_dim': 64, 'min_pos': 8,  'C': 0.50, 'alpha': 0.40},
    {'pca_dim': 64, 'min_pos': 12, 'C': 0.25, 'alpha': 0.40},
    {'pca_dim': 96, 'min_pos': 8,  'C': 0.25, 'alpha': 0.40},
    {'pca_dim': 96, 'min_pos': 8,  'C': 0.50, 'alpha': 0.40},
]
```

Each config does full OOF: per-fold PCA + per-fold probes → OOF AUC.
Best frozen params (pca=32, C=0.25) were found this way.

## Key Differences from Our v1 (0.893)

| Aspect | Our v1 | chiranjithdharma |
|--------|--------|------------------|
| **Probe features** | All-data base/prior | OOF base/prior (fold-safe) |
| **PCA fitting** | Global (all data) | Global (all data) — same |
| **Event smoothing** | Local-max in fuse_scores | None in fuse_scores |
| **Final smoothing** | None | Gaussian [0.1,0.2,0.4,0.2,0.1] on all species |
| **Prior-only species** | Not handled | Score REPLACED by prior for unmapped active species |
| **Unmapped inactive** | Score = -10.0 | Score = -8.0 |
| **Labeled files** | 66 files (792 windows) | 59 fully-labeled files (708 windows) |
| **Perch batching** | ThreadPoolExecutor(4) | batch_files=16 (single TF call) |
