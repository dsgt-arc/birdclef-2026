# Experiment 018: Perch Probes — Tasks

## Goal

Climb the Perch-only leaderboard ladder by progressively improving the probe
pipeline. Every top notebook (0.910–0.925) is Perch-only — no SED/EfficientNet
needed. The progression is: better OOF training → richer features → MLP probes →
ProtoSSM temporal modeling → post-processing.

**Perch Raw Baseline**: 0.836 LB (with genus proxy mapping)
**Current Best**: 0.898 LB (LogReg probes, buggy OOF — v2-fix submitted, awaiting LB)

### Leaderboard Reference Chain

| Score | Notebook | Key technique over previous |
|-------|----------|---------------------------|
| 0.893 | **Ours (perch-probes-v1)** | Perch + LogReg probes, no OOF |
| 0.910 | chiranjithdharma | + OOF 5-fold GroupKFold, fold-safe priors |
| 0.913 | zirach | + ProtoSSM temporal modeling (BiSSM) |
| 0.917 | pantanal-distill | + ResidualSSM boosting, knowledge distillation |
| 0.921 | maryna | + focal loss, mixup, cross-attn, SWA, TTA |
| 0.925 | yuriy/yaroslav | + wider SSM (d=256, 3 layers), 6-stage post-proc |

### Status: Phase C — v2 buggy=0.898, v2-fix submitted (perch-probes-v2 ver.2), awaiting LB

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
- [ ] 6.13 Check v2-fix LB result, target: ~0.910

### Step 7: Richer features (39-dim → 77-dim)

Upgrade from our 39-dim features to Yuriy's 77-dim feature set. Requires PCA 64
(up from 32) plus additional context and interaction features.

Note: jaejohn's grid search found pca_dim=64 and C=0.50 as optimal (vs our
current pca_dim=32, C=0.25 from chiranjithdharma). Worth testing both.

- [ ] 7.1 Increase PCA from 32 → 64 dimensions
- [ ] 7.2 Add features: std_base, diff_mean, diff_prev, diff_next (4 new)
- [ ] 7.3 Add interaction features: raw*prior, raw*base, prior*base (3 new)
- [ ] 7.4 Total: 64 PCA + 13 context = 77-dim (matching Yuriy)
- [ ] 7.5 Try C=0.50 (jaejohn optimal) vs C=0.25 (chiranjithdharma)

### Step 8: MLP probes (LogReg → MLPClassifier)

Replace LogisticRegression with sklearn MLPClassifier, matching reference notebooks.

- [ ] 8.1 Implement MLPClassifier probes (hidden=(128,), ReLU, early_stopping)
- [ ] 8.2 Add positive oversampling for class balance
- [ ] 8.3 Submit perch-probes-v3 (OOF + 77-dim + MLP) — target: ~0.913

---

## Phase D: ProtoSSM Temporal Modeling (~0.913 → ~0.917)

Add a lightweight bidirectional SSM that models temporal patterns across the 12
windows per soundscape. Trains in ~10s on 66 labeled files. Fits in kernel budget.

- [ ] 9.1 Implement ProtoSSM v2 (d_model=128, 2 BiSSM layers, prototypical head)
      - Input projection: 1536 → 128 + LayerNorm + GELU
      - Learnable positional encoding (12 positions)
      - Site/hour metadata embeddings
      - Gated fusion with Perch logits
- [ ] 9.2 Multi-task training: focal BCE + 0.1*MSE distillation + 0.1*taxonomy aux
- [ ] 9.3 3-fold OOF GroupKFold training
- [ ] 9.4 Ensemble: W * ProtoSSM + (1-W) * MLP probes (OOF-optimized W)
- [ ] 9.5 Add ResidualSSM second-pass boosting (d_model=64, correction_weight=0.3)
- [ ] 9.6 Submit — target: ~0.917

## Phase E: Post-Processing Pipeline (~0.917 → ~0.925)

Cherry-pick techniques from maryna (0.921) and yuriy (0.925):

| Stage | Technique | Source | Config |
|-------|-----------|--------|--------|
| 1 | TTA (temporal shifts) | yuriy | shifts=[0, +1, -1], circular roll |
| 2 | Per-taxon temperature | maryna | Aves=1.10, Texture=0.95 |
| 3 | File-level confidence scaling | 2025 Rank 1 | top_k=2 |
| 4 | Rank-aware scaling | 2025 Rank 3 | power=0.5 |
| 5 | Delta shift smoothing | 2025 Rank 1 | alpha=0.15 |
| 6 | Per-class threshold sharpening | yuriy | OOF-optimized F1 thresholds |

- [ ] 10.1 Add TTA (3 temporal shifts through ProtoSSM, no extra Perch cost)
- [ ] 10.2 Add per-taxon temperature scaling
- [ ] 10.3 Add file-level confidence scaling + rank-aware scaling
- [ ] 10.4 Add delta shift smoothing
- [ ] 10.5 Submit — target: ~0.921-0.925

## Phase F: Scale ProtoSSM (if needed)

- [ ] 11.1 Scale ProtoSSM: d_model=128 → 256, 2 → 3 SSM layers
- [ ] 11.2 Add cross-attention after SSM (4-head)
- [ ] 11.3 Add focal loss (gamma=2.0) + file-level mixup (alpha=0.3) + SWA
- [ ] 11.4 Submit — target: match yuriy (0.925)

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
| Perch + probes v2-fix | pending | Fixes: K_HOUR=8, sc_clean priors, 6-cat fuse, 59 fully-labeled files (perch-probes-v2 ver.2) |
| Perch + probes v3 (MLP) | pending | + 77-dim features, MLP probes |
| ProtoSSM v1 | pending | + temporal modeling, ResidualSSM |

---

## Key Config Comparison

| Parameter | v1 (0.893) | v2 target | v3 target | Reference |
|-----------|-----------|-----------|-----------|-----------|
| PCA dims | 32 | 32 | **64** | Yuriy: 64 |
| Features | 39-dim | 39-dim | **77-dim** | Yuriy: 77 |
| Probe type | LogReg | LogReg | **MLP(128)** | Yuriy: MLP(128) |
| Probe C | 0.25 | 0.25 | — | — |
| OOF folds | None | **5-fold** | 5-fold | chiranjithdharma: 5 |
| Fold-safe priors | No | **Yes** | Yes | chiranjithdharma: Yes |
| Alpha blend | 0.25+scale | OOF-calibrated | OOF-calibrated | — |
| Min positive | 8 | 8 | 8 | Same |
| Lambda event | 0.4 | 0.4 | 0.4 | Same |
| Lambda texture | 1.0 | 1.0 | 1.0 | Same |
| Temperature | — | 1.10 | 1.10 | Yuriy: Aves=1.10 |
