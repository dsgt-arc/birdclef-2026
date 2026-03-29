# Experiment 018: Perch Ensemble — Tasks

## Goal

Build a Perch v2 pipeline that matches or exceeds the public notebook baselines
(~0.910+), then ensemble with SED v4b (0.859 LB) to combine fundamentally different
feature sources.

**SED Baseline**: v4b = 0.859 LB
**Perch Raw Baseline**: 0.825 LB (validated — matches starter notebook)
**Perch + Probes Target**: ~0.910 LB (chiranjithdharma reference)
**Ensemble Target**: 0.92+ LB

### Status: IN PROGRESS — Phase B: two-notebook chain created, cache kernel pushed

- **Branch**: `mg/018-perch-ensemble` (PR #20)

### Architecture: Two-Notebook Chain

The 0.910+ notebooks use a **two-notebook chain** to solve the 90-min budget problem:

1. **Notebook A (Perch cache)**: Runs Perch v2 on all ~600 test soundscapes, saves
   embeddings + logits as kernel output (~80 min, runs separately)
2. **Notebook B (scoring)**: Loads cached embeddings → applies probes + priors + smoothing
   → optionally blends with SED → submits (~5-10 min)

This is how chiranjithdharma (0.910) and all higher-scoring notebooks work — they reference
`jaejohn/perch-meta` (a cache notebook output) as an input dataset.

We need to implement the same pattern. Running Perch + probes + SED in a single kernel
exceeds the 90-min budget.

---

## Phase A: Perch Baseline (DONE — raw inference validated)

### Step 1: Offline Pipeline (PACE) — DONE

```bash
cd user/murilo/experiments/018-perch-ensemble
sbatch sbatch/extract_embeddings.sh
```

- [x] 1.1 Species mapping — 203 direct, 6 genus proxy, 25 unmapped
- [x] 1.2 Perch extraction (perch_v2_cpu) — embeddings (792, 1536), mapped scores (792, 234)
- [x] 1.3 Probe training — 53 probes, CV AUC mean 0.7285
- [x] 1.4 Artifacts packaged in shared storage

### Step 2: Local Validation — DONE

- [x] 2.1 Raw 0.7521 → +probes 0.9453 → +priors 0.9604 → +smoothing 0.9604
- [x] 2.2 Probes are the biggest lift (+0.193), priors add +0.015

### Step 3: Raw Perch LB Validation — DONE

- [x] 3.1 Raw Perch kernel (no probes) — **0.825 LB** (matches starter notebook exactly)
- [x] 3.2 Confirms base Perch inference is correct

### Step 3b: Perch + Probes LB — FAILED (0.699)

- [x] 3.3 Perch + probes kernel — **0.699 LB** (probes actively hurting)
- [x] 3.4 Root cause: probes trained on perch_v2 (CUDA), inference on perch_v2_cpu → **FIXED**
      (re-extracted embeddings with perch_v2_cpu, but haven't re-tested on LB yet)

**Note**: The 0.699 submission also took ~85 min (single-kernel Perch inference on 600 files).
Even if probes work correctly, we need the two-notebook architecture to fit in 90 min.

---

## Phase B: Two-Notebook Architecture (NEXT)

### Step 4: Create Perch Cache Notebook (Notebook A)

Creates a kernel that runs Perch v2 on all test soundscapes and saves
embeddings + logits as kernel output. This runs separately from the scoring kernel.

```bash
cd user/murilo/experiments/018-perch-ensemble
kaggle_kernel/push.sh cache
kaggle kernels status murilogustineli/birdclef-2026-perch-cache-v1
```

- [x] 4.1 Create `submission_perch_cache.py` — runs Perch, saves embeddings/logits as output
- [x] 4.2 Create `kernel-metadata-perch-cache.json` (needs TF wheel + Perch model, no probes)
- [x] 4.3 Push cache notebook and verify it completes within 90 min — **completed**
- [x] 4.4 Verify output files: embeddings + mapped scores for all test soundscapes
- [x] 4.5 Submit cache notebook to competition — **0.836 LB** (raw Perch, no probes)

**Finding**: Cache notebook scores +0.011 over perch-raw (0.825 → 0.836). The gain
comes from **genus proxy mapping**: 6 species (e.g., Caiman yacare) mapped via
max(sigmoid(logits[same_genus])) instead of zeros in the raw notebook. Confirms
the species mapping work adds real value.

### Step 5: Create Scoring Notebook (Notebook B)

Runs Perch on 66 labeled soundscapes (~11 min), trains PCA + LogReg probes +
metadata priors from scratch inside the notebook, then applies to cached test
embeddings from Notebook A.

**Why in-notebook training**: v1 loaded pre-trained pickles from PACE (sklearn 1.8.0)
but Kaggle has sklearn 1.6.1. The version mismatch caused PCA/probes to produce
garbage → 0.500 LB. Reference notebooks (chiranjithdharma, zirach) all train
probes inside the notebook, avoiding cross-environment issues.

```bash
cd user/murilo/experiments/018-perch-ensemble
kaggle_kernel/push.sh scoring
kaggle kernels status murilogustineli/birdclef-2026-scoring-v1
kaggle competitions submissions birdclef-2026 | head -20
```

- [x] 5.1 Create `submission_scoring.py` — loads cache, applies probes, submits
- [x] 5.2 Create `kernel-metadata-scoring.json` (references cache notebook output + probe artifacts)
- [x] 5.3 Push scoring v1 — **0.500 LB** (FAILED: sklearn version mismatch on pickled PCA/probes)
- [x] 5.4 Rewrite scoring notebook: train probes in-notebook on labeled soundscapes (~11 min)
      - Adds TF wheel + Perch model to Notebook B dependencies
      - Runs Perch on 66 labeled files → PCA + probes + priors from scratch
      - Eliminates all cross-environment pickle issues
- [x] 5.5 Push scoring v2 — submitted, running inference (~12 min Notebook B budget)
      - Perch on 66 labeled files: 619s (~10 min), probe training: 1.5s, 53 probes
      - Total Notebook B: ~12 min (well within 90 min)
- [x] 5.6 Record scoring v2 LB result — **0.500 LB** (FAILED again)
      - Root cause: kernel_sources uses cached push-run output (3 train files),
        not competition test data. Row_ids mismatch → all zeros → 0.500.

### Step 5b: Single-Kernel Architecture (fix for kernel_sources failure)

Abandon two-notebook chain. Match reference notebook pattern (Yuriy, chiranjithdharma):
single kernel loads pre-extracted labeled embeddings from dataset, runs Perch on test
data live, trains probes in-notebook, applies + submits.

```bash
cd user/murilo/experiments/018-perch-ensemble

# 1. Build + upload labeled embeddings dataset
python scripts/build_labeled_embeddings_dataset.py
cd /storage/project/.../shared/datasets/birdclef-2026/perch-labeled-embeddings
kaggle datasets create -p . -r zip

# 2. Push single-kernel notebook
kaggle_kernel/push.sh perch-probes
kaggle competitions submissions birdclef-2026 | head -20
```

- [x] 5b.1 Build labeled embeddings dataset (NPZ + parquet, ~5 MB)
- [x] 5b.2 Upload `birdclef-2026-perch-labeled-embeddings` to Kaggle
- [x] 5b.3 Push `birdclef-2026-perch-probes-v1` notebook (probability-space pipeline)
- [x] 5b.4 Record LB: **0.757** (WORSE than raw 0.836 — probes destroying signal)

**Root cause**: Analysis of chiranjithdharma (0.910) and jaejohn source notebooks
revealed 3 critical bugs in our pipeline:
1. **Probability space**: We applied sigmoid to Perch scores immediately.
   Reference keeps everything as raw logits until final submission.
2. **Multiplicative priors**: We did `probs *= prior^lambda` (can zero out scores).
   Reference does `logits += lambda * prior_logits` (additive, can't zero out).
3. **No StandardScaler**: Reference applies StandardScaler before PCA to normalize
   embedding dimensions. We skipped this.

### Step 5c: Logit-Space Pipeline (fix all 3 bugs)

Rewrite scoring notebook to match chiranjithdharma exactly:

```bash
cd user/murilo/experiments/018-perch-ensemble

# 1. Re-build dataset with raw logits (not sigmoid)
python scripts/build_labeled_embeddings_dataset.py
# Upload: kaggle datasets version -p <dir> -m 'Raw logits' -r zip

# 2. Push fixed notebook
kaggle_kernel/push.sh perch-probes
```

- [x] 5c.1 Fix A: Re-build labeled dataset with raw logits (scores_raw, not sigmoid)
- [x] 5c.2 Fix B: Rewrite `submission_perch_probes.py` in logit space
      - StandardScaler → PCA on embeddings
      - Additive priors: `logits += lambda * prior_logits`
      - 39-dim features: PCA + raw_logit + prior_logit + base_logit + temporal
      - `decision_function()` for probes (logit output)
      - Logit-space blending: `(1-alpha)*base_logit + alpha*probe_logit`
      - Sigmoid only at final submission
- [x] 5c.3 Upload updated dataset to Kaggle (raw logits, scores_raw key)
- [x] 5c.4 Push perch-probes-v1 (logit pipeline) — submitted, awaiting LB score
      - Phase 1: labeled logits loaded (range [-10, 12.8]), 53 probes trained in 2.5s
      - Phase 2: Perch on test files (~80 min with ThreadPoolExecutor)
      - Phase 3: logit-space probes + additive priors + smoothing + sigmoid at end
- [x] 5c.5 Record LB: **0.893** (+0.057 over raw Perch 0.836, probes now helping)
- [ ] 5c.6 Compare to chiranjithdharma (0.910) — gap of 0.017 remains

### Phase B Result

**0.893 LB** — probes work (+0.057 over raw 0.836). Three critical fixes validated:
logit space, additive priors, StandardScaler. Gap to chiranjithdharma (0.910) = 0.017.

Remaining gap likely from: PACE vs Kaggle embedding mismatch (PCA/probes trained on
PACE embeddings but applied to Kaggle test embeddings), simpler features (39-dim vs
77-dim in reference notebooks), LogReg vs MLP probes.

---

## Phase C: Close the Gap to 0.910+ (Kaggle-native embeddings + probe improvements)

Focus: eliminate cross-environment embedding mismatch and optimize inference time
by caching embeddings. The perch-cache-v1 notebook already runs Perch on Kaggle —
extend it to also cache labeled soundscape embeddings as a reusable dataset.

### Step 6: Create Kaggle-native labeled embeddings dataset

Use `perch-cache-v1` notebook (already runs on Kaggle) to extract embeddings for
BOTH labeled AND test soundscapes. Save labeled output as a Kaggle dataset
(like jaejohn/perch-meta). This eliminates the PACE/Kaggle embedding mismatch.

- [ ] 6.1 Modify perch-cache-v1 to also process 66 labeled soundscapes
      - Save labeled embeddings + raw logits as `labeled_perch_arrays.npz`
      - Save labeled metadata as `labeled_perch_meta.parquet`
      - Still produces submission.csv (raw Perch scores) for competition
- [ ] 6.2 Submit modified cache notebook to competition
- [ ] 6.3 Download output, upload as `birdclef-2026-perch-labeled-embeddings` v3
- [ ] 6.4 Re-run perch-probes-v1 with Kaggle-native labeled embeddings
- [ ] 6.5 Record LB result — target: close to 0.910

### Step 7: Improve probe pipeline (close remaining gap)

Cherry-pick techniques from 0.910-0.925 reference notebooks:

| Technique | Source | Expected Gain | Effort |
|-----------|--------|---------------|--------|
| **Richer features (77-dim)** | Yuriy 0.925 | +0.005-0.010 | Low |
| **MLP probes** (sklearn MLPClassifier) | Yuriy 0.925 | +0.003-0.005 | Low |
| **Fold-safe priors** (OOF) | chiranjithdharma 0.910 | +0.002-0.004 | Medium |
| File-level confidence scaling | maryna 0.921 | +0.002-0.003 | Low |
| Per-taxon temperature | maryna 0.921 | +0.001-0.002 | Low |
| Temporal-shift TTA | maryna 0.921 | +0.001-0.002 | Low |

- [ ] 7.1 Add richer features: std, diff_mean, diff_prev, diff_next, interactions
- [ ] 7.2 Test MLP probes vs LogReg
- [ ] 7.3 Add fold-safe priors (exclude val files when building prior tables)
- [ ] 7.4 Add post-processing: file-level scaling, rank-aware scaling, delta smoothing
- [ ] 7.5 Record LB results for each improvement

---

## Phase D: SED + Perch Ensemble (after Phase C closes gap)

Combine best Perch pipeline (target 0.910+) with best SED model (v4b 0.859).

- [ ] 8.1 Add SED inference to scoring notebook
- [ ] 8.2 Sweep blend weights
- [ ] 8.3 Submit ensemble — target 0.92+

---

## Phase E: Advanced Improvements (if needed)

| Technique | Source | Expected Gain | Effort |
|-----------|--------|---------------|--------|
| ProtoSSM temporal model | zirach 0.913 | +0.003-0.010 | High |
| ResidualSSM boosting | pantanal-distill 0.917 | +0.003-0.005 | High |
| Focal loss (gamma=2.0) | maryna 0.921 | +0.002-0.004 | Low |

---

## Results Tracking

| Submission | LB | Notes |
|------------|-----|-------|
| SED v4b (exp 017) | 0.859 | EfficientNet-B0, AmplitudeToDB, single model |
| Perch raw (no probes) | 0.825 | Matches starter notebook — inference validated |
| Perch cache v1 (no probes) | **0.836** | +0.011 from genus proxy mapping (6 species) |
| Perch + probes (wrong model) | 0.699 | Bug: probes from perch_v2, inference on perch_v2_cpu |
| Perch + probes (scoring v1) | 0.500 | FAILED: sklearn version mismatch (pickled PCA from 1.8.0, Kaggle 1.6.1) |
| Perch + probes (scoring v2) | 0.500 | FAILED: kernel_sources uses push-run cache (train row_ids, not test) |
| Perch + probes (prob-space) | 0.757 | FAILED: sigmoid scores, multiplicative priors, no StandardScaler |
| Perch + probes (logit-space) | **0.893** | Logit pipeline: +0.057 over raw, 0.017 gap to chiranjithdharma (0.910) |
| Ensemble v1 | pending | After Perch probes validated |

---

## Key Config: v1

| Parameter | Value |
|-----------|-------|
| Perch model (Kaggle) | `google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1` |
| Perch model (PACE) | `/storage/.../shared/models/perch-v2-cpu` |
| PCA components | 32 |
| LogReg C | 0.25 |
| Min positive samples | 8 |
| Probe alpha base | 0.25 |
| Lambda event (Aves) | 0.4 |
| Lambda texture (Amphibia/Insecta) | 1.0 |
| Smoothing alpha event | 0.15 |
| Smoothing alpha texture | 0.35 |
| SED weight | 0.3 |
| Perch weight | 0.7 |
| Temperature | 1.10 |
