# jaejohn perch-v2-starter-train-infer — Annotated Pipeline Reference

Source: `kaggle kernels pull jaejohn/perch-v2-starter-train-infer`
Full extracted code: `user/murilo/experiments/018-perch-probes/kaggle_kernel/ref_jaejohn_perch_meta.py`

This notebook produces the `jaejohn/perch-meta` dataset used by chiranjithdharma (0.910),
zirach (0.913), pantanal-distill (0.917), maryna (0.921), and yuriy (0.925).

---

## Settings (frozen for submit mode)

```python
# frozen probe params from OOF tuning
"frozen_best_probe": {
    "pca_dim": 64,       # ← 64 not 32! (chiranjithdharma froze at 32)
    "min_pos": 8,
    "C": 0.50,           # ← 0.50 not 0.25! (chiranjithdharma froze at 0.25)
    "alpha": 0.40,
}

# prior fusion params (same as chiranjithdharma)
"best_fusion": {
    "lambda_event": 0.4,
    "lambda_texture": 1.0,
    "lambda_proxy_texture": 0.8,
    "smooth_texture": 0.35,
}

batch_files = 16
```

## Dataset Output (jaejohn/perch-meta)

The notebook produces three files saved as kernel output:

```python
# Saved to /kaggle/working/perch_cache/
full_perch_meta.parquet     # row_id, filename, site, hour_utc
full_perch_arrays.npz       # scores_full_raw (N, 234), emb_full (N, 1536)
full_oof_meta_features.npz  # oof_base, oof_prior, fold_id (OOF features)
```

Only **fully-labeled files** (all 12 windows annotated) are included.
These become the `jaejohn/perch-meta` Kaggle dataset that downstream notebooks reference.

## Pipeline Overview

```
1. Load/compute Perch embeddings for labeled soundscapes → cache
2. Build OOF fold-safe priors → oof_base, oof_prior arrays
3. Global StandardScaler + PCA (fit once on all data)
4. Train final probes on all data with OOF features
5. Perch inference on test soundscapes (live, ~80 min)
6. Apply probes + sigmoid → submission.csv
```

## Key Difference: OOF Grouped by FILENAME (not site)

```python
def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5):
    groups_full = meta_full["filename"].to_numpy()   # ← by filename, not site!
    gkf = GroupKFold(n_splits=n_splits)

    oof_base  = np.zeros_like(scores_full_raw)
    oof_prior = np.zeros_like(scores_full_raw)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(..., groups=groups_full)):
        val_files = set(meta_full.iloc[va_idx]["filename"])

        # Fold-safe: exclude val FILES from prior tables
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        tables = fit_prior_tables(sc_clean[prior_mask], Y_SC[prior_mask])

        oof_base[va_idx], oof_prior[va_idx] = fuse_scores_with_tables(
            scores_full_raw[va_idx], val_sites, val_hours, tables
        )
```

Note: chiranjithdharma groups by site. jaejohn groups by filename. Both produce
fold-safe priors but with different fold compositions.

## OOF Probe Grid (train mode only, per-fold PCA)

The `run_oof_embedding_probe` function uses **per-fold PCA**:

```python
def run_oof_embedding_probe(scores_raw, emb, meta_df, y_true, pca_dim=64, ...):
    gkf = GroupKFold(n_splits=5)
    groups = meta_df["filename"].to_numpy()

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(..., groups=groups)):
        # Fold-safe priors
        val_files = set(meta_df.iloc[va_idx]["filename"])
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        tables = fit_prior_tables(sc_clean[prior_mask], Y_SC[prior_mask])

        base_tr, prior_tr = fuse_scores_with_tables(scores_raw[tr_idx], ..., tables)
        base_va, prior_va = fuse_scores_with_tables(scores_raw[va_idx], ..., tables)

        # Per-fold StandardScaler + PCA
        scaler = StandardScaler()
        Z_tr = PCA(pca_dim).fit_transform(scaler.fit_transform(emb[tr_idx]))
        Z_va = PCA(pca_dim).transform(scaler.transform(emb[va_idx]))

        for cls_idx in classes_with_enough_positives:
            X_tr = build_class_features(Z_tr, raw, prior_tr, base_tr)
            X_va = build_class_features(Z_va, raw, prior_va, base_va)
            clf = LogisticRegression(C=C, ...).fit(X_tr, y_tr)
            pred = clf.decision_function(X_va)
            oof_final[va_idx, cls_idx] = (1-alpha)*base_va[:, cls_idx] + alpha*pred
```

Per-fold PCA is more honest for OOF evaluation but the **final model uses global PCA**.

## Final Probe Training (global PCA + OOF features)

```python
# Cell 13: Global PCA on all data
emb_scaler = StandardScaler()
emb_pca = PCA(n_components=BEST_PROBE["pca_dim"])  # 64!
Z_FULL = emb_pca.fit_transform(emb_scaler.fit_transform(emb_full))

# Cell 14: Final probes with OOF features
for cls_idx in PROBE_CLASS_IDX:
    X_cls = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=oof_prior[:, cls_idx],     # ← OOF fold-safe!
        base_col=oof_base[:, cls_idx],        # ← OOF fold-safe!
    )
    clf = LogisticRegression(C=BEST_PROBE["C"], ...).fit(X_cls, y)
    probe_models[cls_idx] = clf
```

Same pattern as chiranjithdharma: global PCA, OOF base/prior as features.

## Test-Time Inference

```python
# All-data priors for test
final_prior_tables = fit_prior_tables(sc_clean, Y_SC)
test_base, test_prior = fuse_scores_with_tables(scores_test_raw, ..., final_prior_tables)
Z_TEST = emb_pca.transform(emb_scaler.transform(emb_test))

# Apply probes with FIXED alpha (not dynamic!)
for cls_idx, clf in probe_models.items():
    X = build_class_features(Z_TEST, raw, test_prior, test_base)
    pred = clf.decision_function(X)
    final_test_scores[:, cls_idx] = (
        (1.0 - BEST_PROBE["alpha"]) * test_base[:, cls_idx] +
        BEST_PROBE["alpha"] * pred                 # ← fixed 0.40!
    )

# No Gaussian smoothing — just sigmoid
submission = sigmoid(final_test_scores)
```

## Probe Grid Results (train mode)

```python
param_grid = [
    {"pca_dim": 32, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
    {"pca_dim": 64, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
    {"pca_dim": 64, "min_pos": 8,  "C": 0.25, "alpha": 0.5},
    {"pca_dim": 64, "min_pos": 12, "C": 0.25, "alpha": 0.4},
    {"pca_dim": 96, "min_pos": 8,  "C": 0.25, "alpha": 0.4},
    {"pca_dim": 64, "min_pos": 8,  "C": 0.50, "alpha": 0.4},
]
# Winner: pca_dim=64, C=0.50, alpha=0.40
```

## Key Differences: jaejohn vs chiranjithdharma

| Aspect | jaejohn (this notebook) | chiranjithdharma (0.910) |
|--------|------------------------|--------------------------|
| **PCA dim** | 64 | 32 |
| **Probe C** | 0.50 | 0.25 |
| **OOF grouping** | by filename | by site |
| **OOF probe PCA** | per-fold | global (not used for OOF probes in submit) |
| **Test alpha** | fixed 0.40 | dynamic 0.25 + 0.25*scale |
| **Final smoothing** | None (just sigmoid) | Gaussian [0.1,0.2,0.4,0.2,0.1] |
| **Output** | Dataset (perch-meta) + submission | Submission only |
| **Role** | Cache producer + reference | Consumer of cache |

## Key Takeaways

1. **jaejohn is the "source of truth" notebook** — it produces the labeled embedding
   cache that all downstream notebooks consume. The OOF meta-features
   (`full_oof_meta_features.npz`) are also cached and reused.

2. **pca_dim=64 and C=0.50 won the grid search** in this notebook. chiranjithdharma
   independently froze at pca_dim=32, C=0.25. Both work but are different optima.

3. **GroupKFold by filename vs by site** — jaejohn groups by filename (more folds,
   finer-grained), chiranjithdharma by site (fewer but more meaningful groups).

4. **No Gaussian smoothing** — jaejohn doesn't apply final temporal smoothing.
   chiranjithdharma adds this as a post-processing step (+contribution to 0.910).

5. **Fixed alpha at test time** — jaejohn uses the grid-search winner alpha=0.40
   for all species. chiranjithdharma uses dynamic alpha scaled by n_positive.
