# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-10

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` | **THE most important doc** — 14 sections of model analysis + Perch vs SED comparison | Read fully (14 sections) |
| `user/murilo/experiments/024-data-centric-exploration/tasks.md` | Current experiment task tracking (Phases A-E ✅, F pending) | Read fully |
| `user/murilo/experiments/024-data-centric-exploration/results.md` | Results so far (Phases A-E complete) | Read fully |
| `user/murilo/experiments/024-data-centric-exploration/proposal.md` | Experiment proposal and methodology | Read fully |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006-024) | Read fully |
| `user/murilo/notes/2026-04-09-data-centric-pivot.md` | Data-centric pivot plan with revised priorities | Read fully |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor intel + CV/LB correlation data | Read fully |
| `user/murilo/EXPERIMENTS.md` | Per-experiment methodology summaries (001-024) | Skim Section 1 table, read exp 024 details |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim latest sections (exp 020-023) |
| `user/murilo/MODEL-TRAINING.md` | Training techniques reference (backbones, heads, loss, PP) | Skim relevant sections as needed |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/025-iter-pseudo-labeling` — PR #27 (exp 024 merged to main via PR #26)

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold, proto_weight=0.0** |
| Perch+SED w=0.5 (exp 023) | 0.927 | ProtoSSM+MLP blend |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED |
| SED v5j single (exp 019) | 0.906 | Best single-fold SED |

## What happened in this session (2026-04-10)

### 1. Phase C: Applied MLP Probes to Perch Embeddings ✅

Extracted the full Perch→234 pipeline from the exp 023 Kaggle submission kernel and
applied it to Phase B embeddings for all 127,896 soundscape segments.

Key discoveries:
- MLP probes need **142 features** (128 PCA + 14 contextual), not just PCA embeddings
- Pipeline: logit mapping (203 direct, 3 genus proxy, 28 unmapped) → prior fusion → temporal smoothing → 58 MLP probes → alpha blend
- Memory-optimized to read only 220/14,795 logit columns (login node OOM fix)
- Verified Phase B embeddings identical to exp 020 training data (cosine sim = 1.0)
- Script: `user/murilo/scripts/apply_perch_probes.py`

Output (at `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/`):
- `perch_234_raw_probs.parquet` — pure Perch signal (158 MB)
- `perch_234_scores.parquet` — full pipeline logit-space (172 MB)
- `perch_234_probs.parquet` — full pipeline probability-space (171 MB)

### 2. Phase D: Perch vs SED Comparative EDA ✅ (CRITICAL FINDINGS)

Comprehensive comparison on 127K unlabeled soundscape segments.
Script: `user/murilo/scripts/perch_sed_comparison.py`
Findings in EDA doc sections 13.1-13.12.

**Success criteria — both pass:**
- ✅ Union ≥120/159 unseen at >0.3: **153/159**
- ✅ Perch covers 30+ of 50 SED-zero: **40 rescued at >0.5**

**Key findings (D.1-D.7):**
- Perch rescues **40 SED-zero species** at >0.5 (109→149/159 coverage)
- 28/34 truly hard species covered by Perch at >0.3; only **6 truly invisible**
- **Zero SED-only species** at any threshold — Perch is strict superset for coverage
- Per-species Spearman ρ = 0.223 — models are highly complementary
- Perch ~145/159 species per site vs SED ~13/159

**CRITICAL — Perch calibration problem (D.8-D.10):**
- Perch predicts **119 unseen species per segment** at >0.5 (vs SED: 0.04)
- **15M pseudo-labels** vs SED's **5K** — a 3,000x difference
- 126/159 species appear in >50% of all segments (not segment-specific detection)
- On labeled soundscapes: SED macro-AUC **0.9987** vs Perch **0.7311**
- SED precision **94%** vs Perch **6%** at >0.5
- SED better on **62/70 species**, Perch better on **0/70**
- Root cause: Perch's 14,795-class taxonomy gives permanently positive logits for common taxa
- **Perch raw probs NOT directly usable as pseudo-labels**

**Taxonomy breakdown:**
- Perch covers 134/134 unseen Aves (saturated — not real detection)
- Only 10/18 Amphibia, 3/4 Mammalia covered
- 5 Amphibia + 1 Mammalia truly invisible (hard floor)

### 3. Phase E: Ecological Sanity Check ✅

- All **54 original suspects resolved**: 32 by SED max-folds, 22 by Perch
- **6 truly invisible species** all have 1-5 focal recordings:
  - 4 ecologically expected in Pantanal (nocturnal frogs) — data scarcity
  - 1 out of range (P. albifrons — NE Brazil only)
  - 1 present but diurnal (M. melanurus — won't appear in nocturnal recordings)
- Root cause: data scarcity, not domain shift or model failure

### 4. Teacher Strategy (from Phase D analysis)

**Recommended tiered teacher for exp 025:**

| Tier | Species | Teacher | Rationale |
|------|---------|---------|-----------|
| 1 | 109 SED-covered unseen | SED max-folds + P^1.5 | High precision (94%), 5K labels |
| 2 | 75 seen species | SED max-folds | Already saturated (0.9987 AUC) |
| 3 | 40 Perch-only species | Perch with aggressive filtering | Only signal available, but 6% precision |
| 4 | 6 truly invisible | No teacher — focal only | 1-5 recordings, can't help |

## What needs to happen next

### Immediate — Phase F: Synthesis + Exp 025 Design

1. **F.1**: Write final results.md consolidating all Phase A-E findings
2. **F.2**: Design pseudo-labeling teacher strategy:
   - Tiered teacher: SED-primary (tiers 1-2) + Perch-rescue (tier 3)
   - SED: max-across-folds, P^1.5 power transform, threshold >0.5
   - Perch (tier 3 only): Need calibration strategy — options:
     - Very high threshold (>0.9 or >0.95)
     - Per-file consistency (species in ≥N/12 windows)
     - Site-level frequency caps
     - Calibrate against SED on shared species
   - Multi-label: threshold approach (keep all species above thresh after power transform)
   - Loss weighting: ground truth w=1.0, pseudo-labels w=0.3-0.5
3. **F.3**: Draft exp 025 proposal (pseudo-labeling pipeline)

### After Exp 024

4. **Exp 025**: Pseudo-labeling pipeline (highest expected LB gain)
5. **Backbone diversity**: Train v5j on different backbone for ensemble diversity
6. **VAD cleaning**: Remove speech contamination from focal data

## Key patterns (updated)

- **Only data and inference changes improve LB** — 15+ training recipe changes, 0 successes
- **The LB gap is data coverage** — 0.9958 AUC on 75 known species, 159 species unseen in soundscapes
- **Max-across-folds >> averaging for pseudo-labeling** — +29 species coverage at >0.5
- **SED is precise but narrow, Perch is broad but noisy** — 94% vs 6% precision
- **Perch raw probs are NOT directly usable** — 15M FPs, 119 species/segment
- **Perch is strict superset of SED for coverage** — zero SED-only species at any threshold
- **6 species are a hard floor** — 1-5 focal recordings, neither model can detect them
- **12.6% of focal "confusions" are correct** — model is multi-label capable
- **Offline metrics are anti-correlated with LB** — don't trust val/auroc
- **Simpler models beat complex ones in ensemble** — MLP-only > ProtoSSM+MLP

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` | Inference EDA findings (14 sections) |
| `user/murilo/notes/2026-04-09-data-centric-pivot.md` | Data-centric pivot plan |
| `user/murilo/scripts/inference_eda.py` | SED inference EDA script (9 sections) |
| `user/murilo/scripts/perch_sed_comparison.py` | Perch vs SED comparison script (D.1-D.10) |
| `user/murilo/scripts/apply_perch_probes.py` | Phase C: Perch→234 probe application |
| `user/murilo/inference/perch_inference.py` | Perch v2 TF inference script |
| `user/murilo/experiments/024-data-centric-exploration/` | Exp 024 proposal, tasks, results |
| `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/` | v5j 4-fold inference outputs |
| `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/` | Perch TF outputs + 234-class probs |
| `shared/inference/birdclef-2026/perch-v2-tf/train_audio/` | Perch TF focal outputs |
| `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/` | MLP probes, PCA, ProtoSSM weights |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor intel + CV/LB data |
| `user/murilo/experiments/023-perch-sed-ensemble/kaggle_kernel/submission_ensemble.py` | Perch+SED submission kernel (reference for pipelines) |
