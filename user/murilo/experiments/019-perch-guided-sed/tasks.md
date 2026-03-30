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

### Status: Phase A Step 2 complete. v5f submitted to LB, v5g blocked by daily quota (resets 2026-03-31 AM). Decision gate passed: v5g 0.872 ≥ 0.870 → Step 3 (4-fold) next.

- **Branch**: `mg/018-perch-ensemble` (shared with exp 018 for now)
- **Parallel**: Exp 018 ProtoSSM v4.1 running inference on Kaggle (~90 min)

---

## Phase A: Improved SED Baseline (target: 0.870-0.885 LB)

Establish a stronger SED model before pseudo-labeling. Apply quick wins
from HGNet gap analysis, then 4-fold ensemble.

Reference: `user/murilo/notes/gap-analysis-hgnet-vs-effnet.md`

### Step 1: Quick wins ablation (parallel runs on PACE)

All ablations start from v4b (0.860 LB) with a single change.
90/10 split, 50 epochs, same data pipeline.

Code changes implemented (all backward compatible with exp 017):
- `src/birdclef/encoders/efficientnet.py`: generic timm support via `in_chans=1`
- `src/birdclef/heads/sed_head.py`: added `LSEPooling` + `LSEHead`
- `src/birdclef/lightning/sed_finetune.py`: added `mixup_warmup_epochs`,
  `mixup_theta`, weighted label mixing, `lam >= 0.5` enforcement
- `experiments/017-mel-overhaul/train.py`: registered LSEHead + new mixup params

Fast-dev-run: all 5 configs passed. GPU check: >95% utilization on all.

- [x] 1.1 **v5a — BCEWithLogitsLoss**: Replace soft CE with standard BCE.
      Config: `loss_type: "bce"`. Submitted to PACE (2026-03-30).
- [x] 1.2 **v5b — Stronger MixUp**: α=1.0 (uniform), θ=0.8 label clipping,
      5-epoch warmup. Config: `mixup_alpha=1.0, mixup_theta=0.8,
      mixup_warmup_epochs=5`. Submitted to PACE (2026-03-30).
- [x] 1.3 **v5c — Lower weight_decay**: 1e-4 (from 0.01). Config:
      `weight_decay: 1.0e-4`. Submitted to PACE (2026-03-30).
- [x] 1.4 **v5d — LSEHead**: Log-Sum-Exp pooling head (GeM freq + LSE time).
      Config: `sed_head.type: "lse_head"`. Submitted to PACE (2026-03-30).
- [x] 1.5 **v5e — HGNetV2-B0 backbone**: `hgnetv2_b0.ssld_stage2_ft_in1k`
      (embed_dim=1024). Submitted to PACE (2026-03-30).
- [x] 1.6 Collect results: val/auroc, val/cmap for all 5 runs
      - v5a: 0.875 / 0.620 (LB 0.851, -0.009)
      - v5b: 0.869 / 0.638 (LB 0.840, -0.020)
      - v5c: 0.867 / 0.624 (LB pending)
      - v5d: 0.861 / — (LB pending)
      - v5e: 0.861 / 0.612 (LB pending)
      - **Conclusion**: All ablations regressed on LB despite some offline gains.
        val/auroc is a poor LB predictor for single-variable changes on v4b.

### Step 2: ttahara Full Recipe Reproduction

Pivoted from "combine winners" to faithful reproduction of ttahara's complete
training pipeline (0.884 LB). The Step 1 results showed that individual changes
don't help — the full recipe must be adopted together.

Code changes: LSEHead(freq_pool=mean), TtaharaMelTransform, per-sample MixUp,
OneCycleLR. See `src/birdclef/heads/sed_head.py`, `src/birdclef/preprocessing/ttahara_mel.py`,
`src/birdclef/lightning/sed_finetune.py`. 22 pytest tests passing.

- [x] 2.1 Implement LSEHead with `freq_pool="mean"` option (matching ttahara)
- [x] 2.2 Implement TtaharaMelTransform (torchaudio, resize 256x256, min-max)
- [x] 2.3 Implement per-sample MixUp lambda in SEDFinetuneModule
- [x] 2.4 Implement OneCycleLR scheduler option (single param group, per-step)
- [x] 2.5 Update train.py: mel type routing, freq_pool passthrough, new params
- [x] 2.6 Create v5f config (EfficientNet-B0 + ttahara recipe)
- [x] 2.7 Create v5g config (HGNetV2-B0 + ttahara recipe)
- [x] 2.8 Audit: cross-referenced all 12 components against ttahara ground truth
- [x] 2.9 Submit v5f and v5g to PACE (2026-03-30)
- [x] 2.10 Collect v5g results: **0.872 val/auroc**, epoch 18, 21 min training
      - +0.011 over v5e (same backbone, old recipe: 0.861)
      - +0.007 over v4b (EfficientNet, old recipe: 0.865)
      - Training 8-10× faster (21 min vs ~3-4h)
- [x] 2.10b Collect v5f results: **0.866 val/auroc**, epoch 14
      - +0.001 over v4b (same backbone, old recipe: 0.865) — recipe barely helps EfficientNet
      - -0.006 vs v5g (HGNet with same recipe: 0.872) — backbone is the main factor
- [x] 2.11 Compare results:
      - Recipe impact on EfficientNet: v5f (0.866) vs v4b (0.865) = +0.001 (negligible)
      - Backbone impact under ttahara recipe: v5g (0.872) vs v5f (0.866) = +0.006
      - Recipe unlocking HGNet: v5e (0.861) → v5g (0.872) = +0.011
- [x] 2.12 Decision gate: v5g 0.872 ≥ 0.870 target → proceed to Step 3 (4-fold ensemble)
- [x] 2.13 Add v5f/v5g to exp 010 submission pipeline:
      - export_efficientnet_sed.py: pass freq_pool to LSEHead, handle ttahara mel_config
      - export.sbatch: add export functions + dispatch for v5f/v5g
      - upload_kaggle.py: add MODELS entries
      - submission.py: add compute_mel_ttahara + MODEL_SLUGS (per-segment, not full-signal)
      - kernel-metadata JSON files for both models
- [x] 2.14 Export v5f/v5g to ONNX + upload weights + push v5f kernel to Kaggle
- [ ] 2.15 Push v5g kernel to Kaggle (blocked: daily submission quota, resets ~2026-03-31 morning)
- [x] 2.16 Check v5f LB result: **0.855** (−0.005 vs v4b 0.860)
      - ttahara recipe on EfficientNet regressed on LB despite matching offline (0.866 vs 0.865)
      - Consistent with prior pattern: offline gains don't translate for EfficientNet
- [ ] 2.17 Check v5g LB result

### Step 3: 4-fold ensemble

- [ ] 3.1 Implement MultiLabel Stratified GroupKFold (from ttahara, group by audio_id)
- [ ] 3.2 Train 4 folds of the **winning backbone** on PACE (~16h total)
- [ ] 3.3 Optionally train 4 folds of the **other backbone** too — mixed-backbone
      ensemble (EfficientNet + HGNet) may outperform single-backbone ensemble
- [ ] 3.4 Export all models to ONNX (or OpenVINO if faster)
- [ ] 3.5 Implement fold-averaged inference kernel
- [ ] 3.6 Validate on LB — target: ≥0.885 ensemble

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

| Submission | val/auroc | LB | Phase | Notes |
|------------|----------|-----|-------|-------|
| SED v4b (baseline) | 0.865 | 0.860 | — | EfficientNet-B0, AmplitudeToDB, AttHead, soft CE |
| v5a (BCE) | **0.875** | **0.851** | A | `loss_type: "bce"`, epoch 29, **+0.010 val but −0.009 LB vs baseline** |
| v5b (MixUp) | **0.869** | **0.840** | A | `mixup_alpha=1.0, theta=0.8, warmup=5`, epoch 31, **+0.004 val but −0.020 LB vs baseline** |
| v5c (weight_decay) | **0.867** | pending | A | `weight_decay=1e-4`, epoch 35, **+0.002 val vs baseline** |
| v5d (LSEHead) | **0.861** | pending | A | `sed_head.type: "lse_head"`, epoch 21, **-0.004 vs baseline** |
| v5e (HGNetV2) | 0.861 | pending | A | `hgnetv2_b0.ssld_stage2_ft_in1k`, embed=1024, epoch 23, **-0.004 vs baseline** |
| v5f (EffNet+ttahara) | **0.866** | **0.855** | A | Full ttahara recipe on EfficientNet-B0, epoch 14, **-0.005 LB vs v4b** |
| v5g (HGNet+ttahara) | **0.872** | quota blocked | A | Full ttahara recipe on HGNetV2-B0, epoch 18, **21 min** |
| 4-fold ensemble | — | — | A | |
| Round 1 pseudo | — | — | C | |
| Round 2 pseudo | — | — | C | |
| Round 3 pseudo + 4-fold | — | — | C | |

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
