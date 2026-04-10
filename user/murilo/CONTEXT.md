# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-10

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` | **THE most important doc** — what the model sees and misses | Read fully (11 sections) |
| `user/murilo/experiments/024-data-centric-exploration/tasks.md` | Current experiment task tracking | Read fully |
| `user/murilo/experiments/024-data-centric-exploration/results.md` | Results so far (Phase A complete) | Read fully |
| `user/murilo/experiments/024-data-centric-exploration/proposal.md` | Experiment proposal and methodology | Read fully |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006-023) | Read fully |
| `user/murilo/notes/2026-04-09-data-centric-pivot.md` | Data-centric pivot plan with revised priorities | Read fully |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor intel + CV/LB correlation data | Read fully |
| `user/murilo/EXPERIMENTS.md` | Per-experiment methodology summaries (001-024) | Skim Section 1 table, read exp 024 details |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim latest sections (exp 020-023) |
| `user/murilo/MODEL-TRAINING.md` | Training techniques reference (backbones, heads, loss, PP) | Skim relevant sections as needed |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/024-data-centric-exploration` — PR #26

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold, proto_weight=0.0** |
| Perch+SED w=0.5 (exp 023) | 0.927 | ProtoSSM+MLP blend |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED |
| SED v5j single (exp 019) | 0.906 | Best single-fold SED |

## What happened in this session (2026-04-09 → 2026-04-10)

### 1. Data-Centric Pivot

Analyzed Gemini Deep Research brainstorm against 23-experiment evidence base. Adopted
Karpathy's "Recipe for Training Neural Networks" and MIT "Hacker's Guide to Deep Learning"
philosophy: focus on data quality and quantity, not training recipes.

Revised priority ordering (see `notes/2026-04-09-data-centric-pivot.md`):
1. Data exploration & diagnostics
2. VAD cleaning
3. **Pseudo-labeling v2** (highest expected gain)
4. Backbone diversity
5. Hard negative mining
6. Soundscape CV (demoted — no one has solved CV-LB correlation)
7. Overlap reconstruction
8. Waveform MixUp fix (deprioritized — 3 failed attempts)

### 2. Inference EDA Deep Dive (CRITICAL FINDINGS)

Ran comprehensive analysis on v5j predictions across all 127K soundscape segments
and 265K focal segments. Key findings (`notes/2026-04-09-inference-eda-deep-dive.md`):

- **0.9958 macro-AUC** on 75 labeled-soundscape species — model is saturated
- **159/234 species (68%)** have zero soundscape representation — THIS is the LB gap
- v5j covers only 80/159 unseen species at >0.5 threshold; 79 get zero pseudo-labels
- **Max-across-folds recovers +29 species** (80 → 109/159) — use for pseudo-labeling
- 12.6% of focal "confusions" are correct background species detections (secondary labels)
- 54 species with 200+ focal recordings get zero soundscape predictions (domain shift)
- 14/21 unlabeled soundscape sites have no labeled counterpart
- **34 truly hard species** remain invisible even with max-folds at >0.3 — need Perch

**The fundamental insight**: The LB gap is a DATA COVERAGE problem, not a model quality
problem. The model is near-perfect on species it has soundscape data for.

### 3. Exp 024: Data-Centric Exploration (in progress)

- **Phase A: Max-across-folds** ✅ — Completed, findings above
- **Phase B: Perch v2 TF inference** 🔄 — train_soundscapes DONE, train_audio running (~47%)
  - `perch_inference.py` written and tested (10 files, 200 files)
  - Throughput: 10.7 chunks/s on 24 CPUs (cpu-medium, TF 2.21)
  - ✅ train_soundscapes: 127,896 rows (10,658 files × 12 chunks), verified
    - embeddings.parquet: 1.19 GB (1,536 dims), logits.parquet: 11.3 GB (14,795 Perch taxonomy)
    - No NaN/inf, distributions look correct
  - 🔄 train_audio: ~47% complete, ETA 2026-04-10
  - Output: `shared/inference/birdclef-2026/perch-v2-tf/{train_soundscapes,train_audio}/`
- **Phase C**: UNBLOCKED for train_soundscapes — can apply MLP probes now
- **Phases D-F**: Pending (Phase D needs both train_soundscapes + train_audio for full comparison)

### 4. Phase B Verification (2026-04-10)

Perch TF train_soundscapes job completed. Verified outputs:
- embeddings.parquet: 127,896 rows × 1,541 cols (5 meta + 1,536 emb dims), 1.19 GB
- logits.parquet: 127,896 rows × 14,800 cols (5 meta + 14,795 Perch taxonomy), 11.3 GB
- Row count exact: 10,658 files × 12 chunks = 127,896 ✅
- No NaN/inf, embedding mean ~0, logit mean ~-1.3
- train_audio job still running (~47%), ETA 2026-04-10

Phase C is now unblocked for train_soundscapes.

### 5. Updated Kaggle Discussion Notes

New competitor data on CV/LB correlation:
- Gaurav Rawat: GroupKFold on site shows poor CV-LB correlation
- Anil Ozturk: soundscape-only GKF OOF 0.608 → LB 0.879 (useless)
- Tucker Arrants: GKF "overly harsh" because train/test sites overlap
- hengck23: detailed diagnostics post (logit std, per-group AUC, overfit indicators)
- Boredom is now #1, ceiling ~0.94-0.945 without unlabeled data

## What needs to happen next

### Immediate (train_soundscapes ready NOW)

1. ~~**Verify Perch outputs**~~: ✅ Done — 127,896 rows, correct shapes, no NaN/inf
2. **Phase C**: Apply saved MLP probes to train_soundscapes Perch embeddings → 234-class predictions
   - Artifacts at: `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/`
   - Pipeline: raw 1536-dim → StandardScaler → PCA(128) → 5-fold MLP probes → 234-class probs
   - Can proceed NOW — only needs train_soundscapes embeddings (ready)
3. **Phase D**: Perch vs SED comparative EDA — THE key analysis:
   - How many of the 50 remaining zero-coverage species does Perch catch?
   - Are Perch and SED complementary or redundant?
   - What does the ensemble teacher coverage look like?
   - Can start on train_soundscapes as soon as Phase C completes

### When train_audio job finishes (~2026-04-10)

4. **Verify train_audio outputs** (same checks as train_soundscapes)
5. Extend Phase C/D analysis to focal recordings if needed

### After Phase D

6. **Phase E**: Ecological sanity check on 54 suspect species
7. **Phase F**: Design pseudo-labeling teacher strategy → exp 025 proposal
8. **Exp 025**: Pseudo-labeling pipeline (highest expected LB gain)

## Key patterns (updated)

- **Only data and inference changes improve LB** — 15+ training recipe changes, 0 successes
- **The LB gap is data coverage** — 0.9958 AUC on 75 known species, 159 species unseen in soundscapes
- **Max-across-folds >> averaging for pseudo-labeling** — +29 species coverage at >0.5
- **12.6% of focal "errors" are correct** — model is more multi-label capable than metrics show
- **Offline metrics are anti-correlated with LB** — don't trust val/auroc
- **Simpler models beat complex ones in ensemble** — MLP-only > ProtoSSM+MLP
- **34 truly hard species need Perch** — SED cannot detect them under any strategy

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` | Inference EDA findings (11 sections) |
| `user/murilo/notes/2026-04-09-data-centric-pivot.md` | Data-centric pivot plan |
| `user/murilo/scripts/inference_eda.py` | Reproducible EDA script (9 sections) |
| `user/murilo/inference/perch_inference.py` | Perch v2 TF inference script |
| `user/murilo/inference/sbatch/run_perch.sh` | Perch SLURM job runner |
| `user/murilo/experiments/024-data-centric-exploration/` | Exp 024 proposal, tasks, results |
| `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/` | v5j inference outputs (embeddings, logits) |
| `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/` | v5j 4-fold inference outputs |
| `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/` | Perch TF soundscape outputs (VERIFIED: 127,896 rows) |
| `shared/inference/birdclef-2026/perch-v2-tf/train_audio/` | Perch TF focal outputs (running ~47%) |
| `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/` | MLP probes, PCA, ProtoSSM weights |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor intel + CV/LB data |
