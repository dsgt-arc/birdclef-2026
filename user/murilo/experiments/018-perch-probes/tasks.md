# Experiment 018: Perch Probes — Tasks

## Goal

Climb the Perch-only leaderboard ladder by progressively improving the probe
pipeline. Every top notebook (0.910–0.925) is Perch-only — no SED/EfficientNet
needed. The progression is: better OOF training → richer features → MLP probes →
ProtoSSM temporal modeling → post-processing.

**Perch Raw Baseline**: 0.836 LB (with genus proxy mapping)
**Current Best**: 0.911 LB (MLP probes, OOF v2-fix, beat chiranjithdharma)

### Leaderboard Reference Chain

| Score | Notebook | Key technique over previous |
|-------|----------|---------------------------|
| 0.893 | Ours (perch-probes-v1) | Perch + LogReg probes, no OOF |
| 0.910 | chiranjithdharma | + OOF 5-fold GroupKFold, fold-safe priors |
| 0.911 | **Ours (perch-probes-v2-fix)** | + 6-cat fuse, sc_clean priors, K_HOUR fix |
| 0.913 | zirach | + ProtoSSM temporal modeling (BiSSM) |
| 0.917 | pantanal-distill | + ResidualSSM boosting, knowledge distillation |
| 0.921 | maryna | + focal loss, mixup, cross-attn, SWA, TTA |
| 0.925 | yuriy/yaroslav | + wider SSM (d=256, 3 layers), 6-stage post-proc |

### Status: v4.1 = **0.918 LB** (same as v4, bug fixes had no effect). Gap to Yuriy (0.925) = 0.007.

- **Branch**: `mg/018-perch-ensemble` (PR #20)

---

## Phase A: Perch Baseline — DONE

- [x] 1.1 Species mapping — 203 direct, 6 genus proxy, 25 unmapped
- [x] 1.2 Perch extraction (perch_v2_cpu) — embeddings (792, 1536), mapped scores (792, 234)
- [x] 1.3 Probe training — 53 probes, CV AUC mean 0.7285
- [x] 1.4 Artifacts packaged in shared storage
- [x] 2.1 Local validation: raw 0.7521 → +probes 0.9453 → +priors 0.9604
- [x] 3.1 Raw Perch kernel — **0.825 LB** (matches starter notebook)
- [x] 3.2 Perch cache kernel — **0.836 LB** (genus proxy mapping adds +0.011)

## Phase B: Perch + Probes — DONE (0.893 LB)

Validated that probes work when the pipeline is correct. Three critical fixes:
logit space throughout, additive priors, StandardScaler before PCA.

- [x] 5b.3 Prob-space pipeline — **0.757 LB** (probes destroying signal)
- [x] 5c.2 Logit-space pipeline rewrite (matching chiranjithdharma)
- [x] 5c.5 **0.893 LB** — probes now help (+0.057 over raw 0.836)

Gap analysis vs chiranjithdharma (0.910): 0.017 gap from lack of OOF
cross-validation. All hyperparameters, feature counts, and prior configs match.
PACE vs Kaggle embedding extraction is a non-issue (same audio + same model =
same embeddings).

---

## Phase C: OOF Probes + Richer Features (0.893 → ~0.910)

Target: match chiranjithdharma (0.910) by adding OOF cross-validation and
upgrading probe features. Single kernel rewrite, same runtime (~85 min).

### Step 6: OOF cross-validation for probes

Verified against chiranjithdharma source code (pulled via Kaggle API). Key insight:
PCA is global, OOF loop produces fold-safe priors only, and final probes are trained
with OOF base/prior as features (not all-data priors). Also added Gaussian final
smoothing and removed event smoothing from fuse_scores.

Reference: `user/murilo/notes/chiranjithdharma-0910-annotated.md`
Raw code: `kaggle_kernel/ref_chiranjithdharma_0910.py`

Also pulled jaejohn/perch-v2-starter-train-infer (the notebook that produces the
`jaejohn/perch-meta` dataset everyone uses). Key findings from comparing the two:

| Aspect | chiranjithdharma (0.910) | jaejohn (perch-meta producer) |
|--------|--------------------------|-------------------------------|
| PCA dim | 32 | **64** (grid search winner) |
| Probe C | 0.25 | **0.50** (grid search winner) |
| OOF grouping | by site | by filename |
| Test alpha | dynamic (0.25 + 0.25*scale) | fixed 0.40 |
| Final smoothing | Gaussian [0.1,0.2,0.4,0.2,0.1] | None |
| OOF probe PCA | global | per-fold |

Both use the same core pattern: OOF fold-safe priors → oof_base/oof_prior as
features for final probes. The param differences suggest pca=64 + C=0.50 may be
a better optimum — worth testing in v3.

Reference: `user/murilo/notes/jaejohn-perch-meta-annotated.md`
Raw code: `kaggle_kernel/ref_jaejohn_perch_meta.py`

- [x] 6.1 Pull chiranjithdharma source via Kaggle API, create annotated reference
- [x] 6.2 Implement OOF in `submission_perch_probes.py` matching chiranjithdharma:
      - GroupKFold by site (not file) — 5 folds across ~20 sites
      - Global StandardScaler + PCA (fit once on all data, NOT per-fold)
      - OOF loop fills oof_base/oof_prior with fold-safe prior-fused scores
      - Final probes trained on all data using oof_base/oof_prior as features
- [x] 6.3 Fold-safe priors: exclude val sites when building prior tables per fold
- [x] 6.4 Remove event smoothing from fuse_scores (chiranjithdharma only smooths texture)
- [x] 6.5 Add Gaussian final smoothing [0.1, 0.2, 0.4, 0.2, 0.1] on all species
- [x] 6.6 Create kernel-metadata-perch-probes-v2.json (separate kernel from v1)
- [x] 6.7 Push perch-probes-v2 — dry-run passed (3 train files, 1.8min)
      - 9 sites, 5-fold GroupKFold: S22 alone = 480/792 windows (60%!)
      - Fold 0 val = S22 (480 windows), folds 1-4 val = 72-84 windows each
      - OOF baseline AUROC (prior fusion only): 0.6421
      - 53 probes trained, Phase 1 total: 1.5s
      - **Note**: S22 imbalance means fold 0's prior tables miss the dominant site
- [x] 6.8 Submit perch-probes-v2 to competition (2026-03-29 evening) — running inference
- [x] 6.9 Check LB result: **0.898 LB** (only +0.005 over v1, expected ~0.910)
      - Gap analysis revealed 4 bugs in our v2 vs chiranjithdharma:
        1. K_HOUR=4 (should be 8) — never actually fixed despite notes saying so
        2. Prior tables built from labeled subset only (792 rows), not ALL soundscape
           annotations (~800+ rows) — chiranjithdharma uses sc_clean/Y_SC
        3. Missing 6-category fuse_scores: no proxy λ=0.8, no prior-only REPLACE
           for unmapped active species (stuck at -10.0), no -8.0 for inactive
        4. Included 7 partially-labeled files (noisy labels as false negatives)
- [x] 6.10 Implement v2-fix with all 4 fixes:
      - K_HOUR = 8
      - Load train_soundscapes_labels.csv → sc_clean/Y_SC for all prior tables
      - 6-category fuse_scores matching chiranjithdharma exactly
      - Filter to 59 fully-labeled files (708 windows) for probe training
- [x] 6.11 Push v2-fix as perch-probes-v2 version 2 (dry-run passed, 153s)
      - 739 soundscape rows, 59 fully-labeled files, 708 windows
      - Categories: mapped_event=31, mapped_texture=15, proxy_texture=2,
        prioronly_event=2, prioronly_texture=25, unmapped_inactive=2
      - 8 sites, 5-fold GroupKFold (S22 still 468/708 = 66%)
      - 52 probes trained, Phase 1: 2.0s
- [x] 6.12 Submit v2-fix to competition (2026-03-29) — running inference
- [x] 6.13 Check v2-fix LB result: **0.911 LB** (+0.013 over buggy v2, +0.018 over v1, beat chiranjithdharma 0.910)

### Step 7: Richer features (39-dim → 77-dim) — DONE (in v3)

- [x] 7.1 Increase PCA from 32 → 64 dimensions
- [x] 7.2 Add features: std_base, diff_mean, diff_prev, diff_next (4 new)
- [x] 7.3 Add interaction features: raw*prior, raw*base, prior*base (3 new)
- [x] 7.4 Total: 64 PCA + 13 context = 77-dim (matching Yuriy)
- [x] 7.5 Using C=0.50, alpha=0.40 (jaejohn/Yuriy optimal)

### Step 8: MLP probes (LogReg → MLPClassifier) — DONE (in v3)

- [x] 8.1 Implement MLPClassifier probes (hidden=(128,), ReLU, early_stopping)
- [x] 8.2 Add positive oversampling for class balance
- [x] 8.3 v3 kernel ready (submission_perch_probes.py), not yet submitted separately

---

## Phase D: Full ProtoSSM v4 Pipeline (0.911 → 0.918 → pending v4.1) — NEAR COMPLETE

Ported Yuriy's ProtoSSM v5 (0.925 LB) into modular codebase under
`src/birdclef/perch/` (8 modules, 140 pytest tests). Assembled into
self-contained kernel `submission_perch_protossm.py` (2756 lines).

Reference: `user/murilo/notes/yuriy-protossm-v5-annotated.md`
Source: `user/murilo/src/birdclef/perch/` (models, training, losses,
postprocessing, ensemble, data_utils, config)

### Step 9: ProtoSSM v4 implementation

- [x] 9.1 Build `src/birdclef/perch/` module package:
      - `models.py`: SelectiveSSM (Mamba-style), TemporalCrossAttention,
        ProtoSSMv2 (d=128, 2 BiSSM + 4-head cross-attn + prototypical head
        + gated Perch fusion + taxonomy aux head), ResidualSSM (d=64, zero-init)
      - `training.py`: train_proto_ssm_single (focal BCE + distillation +
        mixup after epoch 5 + SWA from 70% + gradient clip + early stopping),
        run_proto_ssm_oof (3-fold GroupKFold by site),
        train_residual_ssm (MSE on residuals, wall-time gated)
      - `losses.py`: focal_bce_with_logits, mixup_files
      - `postprocessing.py`: 6-stage pipeline (temperature, file-level
        confidence, rank-aware, delta-shift, threshold sharpening)
      - `ensemble.py`: temporal_shift_tta, optimize_ensemble_weight
      - `data_utils.py`: reshape_to_files, site mapping, taxonomy groups
      - `config.py`: all frozen hyperparameters matching Yuriy
- [x] 9.2 Shape tests: all models pass (ProtoSSM 722K params, ResidualSSM 176K)
- [x] 9.3 140 pytest tests passing (models, losses, training, postprocessing,
      ensemble, data_utils, config, batched inference, kernel-module parity)
- [x] 9.4 Assemble self-contained kernel `submission_perch_protossm.py`
- [x] 9.5 Changes vs v2-fix:
      - Proxy taxa: {Amphibia} → {Amphibia, Insecta, Aves}
      - OOF grouping: site → filename (matching Yuriy)
      - Event smoothing added in fuse_scores (α=0.15)
      - Gaussian smoothing → 6-stage post-processing pipeline
- [x] 9.6 Batched Perch inference (16 files/batch, single TF call) — v2 kernel
- [x] 9.7 Push v4 kernel — **0.918 LB** (nb version 1, ThreadPoolExecutor)
      - Ensemble weight: 0.50 proto / 0.50 MLP, OOF AUC=0.9133
- [x] 9.8 Audit v4 vs Yuriy (0.925) — found 3 bugs:
      - Bug 1: OOF prior exclusion by site instead of filename (fixed)
      - Bug 2: Post-processing missing per-stage clipping (fixed)
      - Bug 3: Final ProtoSSM trained with 15% val split instead of all data (fixed)
      - Also fixed: Perch labels.csv parsing (inat2024_fsd50k → scientific_name)
      - Confirmed NOT bugs: dynamic proxy expansion (all unmapped are sonotypes),
        model architectures, post-processing formulas, config values
- [x] 9.9 Push v4.1 kernel (nb version 4) with all fixes — submitted 2026-03-30,
      running inference (~90 min)
      - OOF baseline AUROC: 0.6656 → 0.8061 (filename exclusion fix)
      - OOF ensemble AUC: 0.9133 → 0.9447
      - SWA applied: 6 checkpoints (vs 0 before, all-data training)
- [x] 9.10 Check v4.1 LB result: **0.918 LB** (same as v4 — bug fixes had no effect)
      - OOF improvements (0.6656→0.8061 baseline, 0.9133→0.9447 ensemble) did NOT translate to LB
      - Gap to Yuriy (0.925) remains at 0.007 — root cause is elsewhere
- [x] 9.11 Push v4.2 kernel with 4 new fixes from line-by-line diff vs Yuriy:
      1. **Taxonomy aux head**: genus (161 groups) → class_name (5 groups) — biggest fix.
         Our 161-way genus classifier was noisy on 59 files. Yuriy uses 5-way class_name
         (Aves/Amphibia/Insecta/Mammalia/Reptilia) for clean regularization.
      2. **Ensemble weight**: Always OOF-optimized → fixed 0.5 in submit mode (matching Yuriy).
         OOF optimization is not fold-safe since MLP probes train on all data.
      3. **MLP probes**: Added n_iter_no_change=15, random_state=42 (matching Yuriy).
      4. **Site mapping**: 0-based → 1-based (index 0 = unknown sites, matching Yuriy).
      - All 142 tests passing after fixes
      - Estimated impact: +0.004 to +0.008 LB
      - Pushed as nb version 5, dry-run passed (145s, families=5, sites=9)
      - Submitted to competition (2026-03-30), running inference (~90 min)
- [ ] 9.12 Check v4.2 LB result — target: 0.920-0.925

---

## Future Work (if needed after v4.2 LB result)

### Scale ProtoSSM
- [ ] Scale d_model: 128 → 256, layers: 2 → 3
- [ ] OOF-optimized per-class thresholds (currently fixed at 0.5 in submit mode)
- [ ] Proxy taxa expansion: build genus proxies dynamically at inference
      (Yuriy does this; we use static genus_proxy_map from species_mapping.json)

### Additional techniques from other top notebooks
- [ ] Cross-attention refinements from maryna (0.921)
- [ ] Knowledge distillation from stronger teacher models
- [ ] Ensemble with SED models (exp 017 v4b = 0.860 LB)

---

## Results Tracking

| Submission | LB | Notes |
|------------|-----|-------|
| SED v4b (exp 017) | 0.859 | EfficientNet-B0, AmplitudeToDB, single model |
| Perch raw (no probes) | 0.825 | Matches starter notebook — inference validated |
| Perch cache v1 (no probes) | **0.836** | +0.011 from genus proxy mapping (6 species) |
| Perch + probes (wrong model) | 0.699 | Bug: probes from perch_v2, inference on perch_v2_cpu |
| Perch + probes (scoring v1) | 0.500 | FAILED: sklearn version mismatch (pickled PCA) |
| Perch + probes (scoring v2) | 0.500 | FAILED: kernel_sources uses push-run cache |
| Perch + probes (prob-space) | 0.757 | FAILED: sigmoid scores, multiplicative priors |
| Perch + probes v1 (logit) | **0.893** | Logit pipeline, 39-dim LogReg, no OOF |
| Perch + probes v2 (OOF) | **0.898** | + OOF fold-safe priors, Gaussian smoothing — **buggy** (K_HOUR=4, subset priors, missing fuse categories, partial files) |
| Perch + probes v2-fix | **0.911** | Fixes: K_HOUR=8, sc_clean priors, 6-cat fuse, 59 fully-labeled files (perch-probes-v2 ver.2) |
| Perch + probes v3 (MLP) | not submitted | + 77-dim features, MLP probes (included in v4) |
| **ProtoSSM v4** | **0.918** | + BiSSM×2 + cross-attn + TTA + ResidualSSM + 6-stage PP, ensemble 0.45/0.55 |
| **ProtoSSM v4.1** | **0.918** | Bug fixes: OOF prior by filename, per-stage clipping, train on all data — **no improvement** |

---

## Key Config Comparison

| Parameter | v1 (0.893) | v2-fix (0.911) | v4 (ProtoSSM) | Yuriy (0.925) |
|-----------|-----------|----------------|---------------|---------------|
| PCA dims | 32 | 32 | **64** | 64 |
| Features | 39-dim | 39-dim | **77-dim** | 77 |
| Probe type | LogReg | LogReg | **MLP(128)** | MLP(128) |
| Probe alpha | dynamic | dynamic | **0.40 fixed** | 0.40 |
| OOF grouping | — | site (5-fold) | **filename (5-fold)** | filename (5-fold) |
| ProtoSSM | — | — | **BiSSM×2 + cross-attn** | same |
| ProtoSSM OOF | — | — | **3-fold by site** | 3-fold by site |
| Ensemble | — | — | **0.50 proto / 0.50 MLP** | OOF-optimized |
| ResidualSSM | — | — | **d=64, w=0.3** | d=64, w=0.3 |
| TTA | — | — | **shifts [0, ±1]** | [0, ±1] |
| Post-processing | — | Gaussian | **6-stage pipeline** | 6-stage |
| Event smoothing | — | no | **yes (α=0.15)** | yes |
| Proxy taxa | Amphibia | Amphibia | **Amphibia+Insecta+Aves** | same |
| Inference | ThreadPool | ThreadPool | **batched TF (16 files)** | batched (16) |
