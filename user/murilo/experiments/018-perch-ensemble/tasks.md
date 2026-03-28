# Experiment 018: Perch Ensemble — Tasks

## Goal

Build a Perch v2 pipeline that matches or exceeds the public notebook baselines
(~0.910+), then ensemble with SED v4b (0.859 LB) to combine fundamentally different
feature sources.

**SED Baseline**: v4b = 0.859 LB
**Perch Raw Baseline**: 0.825 LB (validated — matches starter notebook)
**Perch + Probes Target**: ~0.910 LB (chiranjithdharma reference)
**Ensemble Target**: 0.92+ LB

### Status: IN PROGRESS — Phase A validated, debugging probes

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
# Create + push cache notebook
cd user/murilo/experiments/018-perch-ensemble/kaggle_kernel
cp kernel-metadata-perch-cache.json kernel-metadata.json
kaggle kernels push -p .
rm kernel-metadata.json
```

- [ ] 4.1 Create `submission_perch_cache.py` — runs Perch, saves embeddings/logits as output
- [ ] 4.2 Create `kernel-metadata-perch-cache.json` (needs TF wheel + Perch model, no probes)
- [ ] 4.3 Push cache notebook and verify it completes within 90 min
- [ ] 4.4 Verify output files: embeddings + logits for all test soundscapes

### Step 5: Create Scoring Notebook (Notebook B)

Loads cached embeddings from Notebook A, applies probes + priors + smoothing.

- [ ] 5.1 Create `submission_scoring.py` — loads cache, applies probes, submits
- [ ] 5.2 Create `kernel-metadata-scoring.json` (references cache notebook output + probe artifacts)
- [ ] 5.3 Push scoring notebook and record Perch-only LB result
- [ ] 5.4 Compare to chiranjithdharma (0.910) — this is the real Phase A validation

### Phase B Decision Gate

- If Perch + probes LB >= 0.900: probes work → continue to ensemble
- If Perch + probes LB 0.850-0.900: probes partially work → debug
- If Perch + probes LB < 0.850: probe pipeline has bugs → fix before ensemble

---

## Phase C: SED + Perch Ensemble

Only proceed after Phase B validates Perch + probes.

### Step 6: Add SED to Scoring Notebook

- [ ] 6.1 Add SED inference to scoring notebook (ONNX, ~30 min)
- [ ] 6.2 Sweep blend weights locally
- [ ] 6.3 Push ensemble scoring kernel
- [ ] 6.4 Record ensemble LB vs Perch-only and SED baselines

---

## Phase D: Perch Pipeline Improvements (from reference notebooks)

Cherry-pick techniques from 0.921-0.926 notebooks if ensemble alone isn't enough.

| Technique | Source | Expected Gain | Effort |
|-----------|--------|---------------|--------|
| Focal loss (gamma=2.0) | maryna 0.921 | +0.002-0.004 | Low |
| File-level confidence scaling | maryna 0.921 | +0.002-0.003 | Low |
| Per-taxon temperature | maryna 0.921 | +0.001-0.002 | Low |
| Temporal-shift TTA | maryna 0.921 | +0.001-0.002 | Low |
| ProtoSSM temporal model | zirach 0.913 | +0.003-0.010 | High |
| ResidualSSM boosting | pantanal-distill 0.917 | +0.003-0.005 | High |

---

## Results Tracking

| Submission | LB | Notes |
|------------|-----|-------|
| SED v4b (exp 017) | 0.859 | EfficientNet-B0, AmplitudeToDB, single model |
| Perch raw (no probes) | **0.825** | Matches starter notebook — inference validated |
| Perch + probes (wrong model) | 0.699 | Bug: probes from perch_v2, inference on perch_v2_cpu |
| Perch + probes (fixed) | pending | Re-extracted with perch_v2_cpu, needs two-notebook arch |
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
