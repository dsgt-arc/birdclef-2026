# Experiment 019: Perch-Guided SED Training — Tasks

## Goal

Push SED from 0.860 → 0.900-0.915 LB by combining HGNet-inspired architecture
improvements with Perch pseudo-labels on 10K+ unlabeled soundscapes.

**Current SED Best**: 0.860 LB (EfficientNet-B0 SED v4b, exp 017)
**Teacher**: Perch ProtoSSM v4.1 ~0.920 LB (exp 018)
**Target**: 0.900-0.915 LB (SED alone, enabling future Perch+SED ensemble)

### Reference Chain

| Score | Model | Key technique |
|-------|-------|---------------|
| 0.860 | Our SED v4b | EfficientNet-B0, AmplitudeToDB, AttHead, soft CE |
| 0.884 | ttahara HGNetV2-B0 | LSEHead, BCE, MixUp α=1.0, 4-fold ensemble |
| 0.908 | Salman Ahmed SED | Single EfficientNet-B0 (unknown techniques) |
| ~0.920 | Our Perch ProtoSSM | Teacher for pseudo-labels |
| 0.939 | Competition #1 | Likely ensemble + pseudo-labels |

### Status: Proposed — ready to start Phase A

- **Branch**: `mg/019-perch-guided-sed`

---

## Phase A: Improved SED Baseline (target: 0.870-0.885 LB)

Establish a stronger SED model before pseudo-labeling. Apply quick wins
from HGNet gap analysis, then 4-fold ensemble.

Reference: `user/murilo/notes/gap-analysis-hgnet-vs-effnet.md`

### Step 1: Quick wins ablation (parallel runs on PACE)

All ablations start from v4b (0.860 LB) with a single change.
90/10 split, 50 epochs, same data pipeline.

- [ ] 1.1 **v5a — BCEWithLogitsLoss**: Replace soft CE with standard BCE.
      Different from v4i (which used single-label CE, not multi-label BCE).
      Config: `loss=bce`, everything else from v4b.
- [ ] 1.2 **v5b — Stronger MixUp**: α=1.0 (uniform), θ=0.8 label clipping,
      5-epoch warmup (no mixup during warmup). Config: `mixup_alpha=1.0,
      mixup_theta=0.8, mixup_warmup=5`.
- [ ] 1.3 **v5c — Lower weight_decay**: 1e-4 (from 0.01). Config:
      `weight_decay=1e-4`.
- [ ] 1.4 **v5d — LSEHead**: Replace AttHead with Log-Sum-Exp pooling head.
      mean(freq) → Linear+ReLU+Dropout+Linear → LSEPool(time, T=1.0).
      Requires implementing LSEHead in `src/birdclef/heads/`.
- [ ] 1.5 **v5e — HGNetV2-B0 backbone**: Drop-in replacement via timm
      `hgnetv2_b0.ssld_stage2_ft_in1k`. Keep our SED head + training pipeline.

### Step 2: Combine winners

- [ ] 2.1 Analyze ablation results: which improvements stack?
- [ ] 2.2 Train combined model with all positive ablations
- [ ] 2.3 Validate on LB — target: ≥0.870 single fold

### Step 3: 4-fold ensemble

- [ ] 3.1 Implement MultiLabel Stratified GroupKFold (from ttahara, group by audio_id)
- [ ] 3.2 Train 4 folds of the best variant on PACE (~16h total)
- [ ] 3.3 Export all 4 models to ONNX (or OpenVINO if faster)
- [ ] 3.4 Implement fold-averaged inference kernel
- [ ] 3.5 Validate on LB — target: ≥0.885 ensemble

---

## Phase B: Perch Pseudo-Label Extraction (target: 127K labeled windows)

Run Perch ProtoSSM on ALL unlabeled soundscapes offline on PACE to
generate soft labels for Phase C training.

### Step 4: Perch GPU extraction on unlabeled soundscapes

- [ ] 4.1 Write SBATCH script for Perch v2 GPU extraction on PACE
      (reference: exp 014 pseudo-labeling, `scripts/extract_perch_embeddings.py`)
- [ ] 4.2 Extract embeddings + logits for all ~10,600 unlabeled soundscapes
      Output: embeddings (127K, 1536) + mapped logits (127K, 234)
      Estimate: ~2-4 hours on GPU
- [ ] 4.3 Verify extraction: spot-check shapes, value ranges, no NaN

### Step 5: Apply ProtoSSM pipeline to extracted embeddings

- [ ] 5.1 Load ProtoSSM v4.1 trained model artifacts (probes, ProtoSSM weights,
      ResidualSSM weights, prior tables, PCA scaler)
      Note: these are trained in-kernel on 59 labeled files. Need to save
      them from a training run or re-train on PACE.
- [ ] 5.2 Apply full pipeline: prior fusion → MLP probes → ProtoSSM → ensemble
      → ResidualSSM → post-processing → soft probabilities
- [ ] 5.3 Output: pseudo_labels.npz with shape (n_files, 12, 234) float32
- [ ] 5.4 Quality check: compute pseudo-label statistics
      - Mean confidence per species
      - Number of species detected per file
      - Distribution of max probabilities
      - Cross-check against labeled soundscapes (should match ~0.920 quality)

### Step 6: Confidence filtering

- [ ] 6.1 Apply power transform: `p_sharp = p^γ / (p^γ + (1-p)^γ)`, γ=1.5
- [ ] 6.2 Filter low-confidence windows (max species prob < 0.1)
- [ ] 6.3 Save filtered pseudo-labels as Kaggle dataset for kernel access
      (or store on PACE shared storage for local training)

---

## Phase C: Pseudo-Label Training (target: 0.900-0.915 LB)

Train improved SED on labeled + pseudo-labeled data with multi-round
refinement. Each round uses a stronger teacher.

### Step 7: Round 1 — Conservative pseudo-labels

- [ ] 7.1 Prepare training data:
      - Focal recordings: 35K files, weight=1.0
      - Labeled soundscapes: 708 windows, weight=1.0
      - Pseudo-labeled soundscapes: 127K windows, weight=0.1, γ=1.5
- [ ] 7.2 Implement per-sample loss weighting in SED training pipeline
      (BCE with sample weights, similar to exp 015 but with lower weight)
- [ ] 7.3 Train Round 1 SED (best variant from Phase A)
      Config: same as Phase A winner, + pseudo-label data
- [ ] 7.4 Evaluate: offline val AUROC + cmAP, compare vs Phase A baseline
- [ ] 7.5 Validate on LB — confirm pseudo-labels help (not regress)

### Step 8: Round 2 — Stronger teacher

- [ ] 8.1 Generate new pseudo-labels using Round 1 SED + Perch ensemble
      as teacher (average predictions, should be stronger than Perch alone)
- [ ] 8.2 Increase pseudo-label weight to 0.15, increase γ to 2.0
- [ ] 8.3 Train Round 2 SED
- [ ] 8.4 Validate on LB — target: improvement over Round 1

### Step 9: Round 3 — Final refinement

- [ ] 9.1 Generate pseudo-labels with Round 2 teacher
- [ ] 9.2 Weight 0.2, γ=2.5
- [ ] 9.3 Train Round 3 SED
- [ ] 9.4 4-fold ensemble of Round 3 models
- [ ] 9.5 Validate on LB — target: 0.900-0.915

---

## Decision Gates

After each phase, decide whether to continue:

**Phase A → B gate**: Any SED improvement reaches ≥0.870 LB?
- Yes → proceed to Phase B (pseudo-labels will help a good model more)
- No → debug SED improvements, try more ablations before pseudo-labeling

**Phase B → C gate**: Pseudo-label extraction succeeds + quality check passes?
- Yes → proceed to Phase C
- No → investigate Perch extraction issues, retry

**Phase C Round 1 gate**: Round 1 SED ≥ Phase A baseline?
- Yes → continue to Round 2 (pseudo-labels are helping)
- No → STOP. Same failure as exp 015. Investigate weight/confidence settings.

---

## Results Tracking

| Submission | LB | Phase | Notes |
|------------|-----|-------|-------|
| SED v4b (baseline) | 0.860 | — | EfficientNet-B0, AmplitudeToDB |
| v5a (BCE) | pending | A | |
| v5b (MixUp) | pending | A | |
| v5c (weight_decay) | pending | A | |
| v5d (LSEHead) | pending | A | |
| v5e (HGNetV2) | pending | A | |
| Combined winner | pending | A | |
| 4-fold ensemble | pending | A | |
| Round 1 pseudo | pending | C | |
| Round 2 pseudo | pending | C | |
| Round 3 pseudo + 4-fold | pending | C | |

---

## Runtime Budget (Kaggle kernel, 90 min)

| Component | Single-fold | 4-fold ensemble |
|-----------|-------------|-----------------|
| Model loading | ~1s | ~4s |
| Audio → mel spectrogram | ~5 min | ~5 min |
| SED inference (ONNX) | ~10 min | ~40 min |
| Post-processing | ~1s | ~1s |
| **Total** | **~15 min** | **~45 min** |

Both single-fold and 4-fold fit comfortably in 90 min. This leaves
room for potential future Perch+SED ensemble in a single kernel if
Perch inference can be optimized (or via two-kernel approach).
