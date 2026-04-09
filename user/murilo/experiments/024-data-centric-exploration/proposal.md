# Proposal: Data-Centric Exploration

**Status**: in-progress
**Created**: 2026-04-09
**Author**: Murilo Gustineli
**Branch**: `main` (no separate branch — analysis-only experiment)

## Hypothesis

The LB gap (0.929 → 0.948) is a **data coverage problem**, not a model quality problem.
Our model achieves 0.9958 macro-AUC on the 75 species with labeled soundscape data, but
159 species (68%) have zero soundscape representation. Running Perch v2 TF inference on
all soundscapes and comparing with SED predictions will reveal the complementarity between
models and inform the pseudo-labeling teacher strategy for exp 025.

## Background

### What we know (from inference EDA — `notes/2026-04-09-inference-eda-deep-dive.md`)

- v5j achieves **0.9958 AUC** on 75 labeled-soundscape species (saturated)
- **159 species** have zero soundscape data — the core LB gap
- v5j covers 80/159 unseen species at >0.5 threshold; **79 get zero** pseudo-labels
- 4-fold averaging HURTS coverage (-12 species vs single-fold); max-across-folds is better
- 12.6% of focal "confusions" are correct secondary species detections
- 54 species with 200+ focal recordings get zero soundscape predictions (domain shift)
- 14/21 unlabeled soundscape sites have no labeled counterpart

### What we don't know (and need to answer)

1. How many of the 79 zero-coverage species does **Perch** detect?
2. Are Perch and SED complementary or redundant for unseen species?
3. Does max-across-folds recover any of the 79 zero-coverage species?
4. Which of the 54 suspect species are ecologically absent vs domain-shift failures?

### Data-centric pivot rationale (`notes/2026-04-09-data-centric-pivot.md`)

Inspired by Karpathy ("spend copious time scanning data") and MIT OCW ("become friends
with your data"). 23 experiments, 40+ variants: only data/inference changes improve LB.
Training recipe changes are 0-for-15. The path forward is data quality and data quantity.

## Method

### Approach

This is an **exploration experiment** — no model training, no LB submissions. All outputs
are analysis artifacts that inform the design of exp 025 (pseudo-labeling pipeline).

Code lives in its existing locations. This experiment directory holds only the proposal,
tasks, and results docs that tie everything together.

### Setup

| Component | Details |
|-----------|---------|
| Data | 10,658 soundscapes (127K segments), 35,549 focal recordings |
| Compute | PACE CPU nodes for Perch TF inference (~21h); analysis on login node |
| Dependencies | TensorFlow 2.x (Perch TF), onnxruntime (Perch ONNX), sklearn, pandas |
| Code | `inference/infer.py` (extend for Perch), `scripts/inference_eda.py` |

### Procedure

#### Phase A: Max-Across-Folds Analysis (immediate, no compute needed)

Compute `max(fold0, fold1, fold2, fold3)` from existing 4-fold inference data at
`shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/`.

**Key question**: How many of the 79 zero-coverage species are recovered?

**Code**: Add section to `scripts/inference_eda.py`
**Output**: Updated EDA findings in `notes/2026-04-09-inference-eda-deep-dive.md`

#### Phase B: Perch v2 TF Inference on All Soundscapes

Run Perch v2 TF SavedModel on all 10,658 soundscape files (127K 5s segments).
Use TF (not ONNX) for best fidelity — MLP probes were trained on TF embeddings.

**Model**: `shared/models/perch-v2-cpu/` (TF SavedModel)
**Output**: `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/`
  - `embeddings.parquet` — 1536-dim per segment
  - `logits.parquet` — 14,795-class raw logits per segment (Perch taxonomy)

**Runtime**: ~609ms/segment × 127K ≈ 21.5 hours (single CPU). Can parallelize with SLURM.

**Code**: Extend `inference/infer.py` or write `scripts/perch_inference.py`

#### Phase C: Apply Saved MLP Probes to Perch Embeddings

Load pretrained artifacts from `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/`:
- `emb_scaler.joblib` → StandardScaler
- `pca_transformer.joblib` → PCA(128)
- `probe_models.joblib` → 5-fold MLP probes (234 classes)

Apply to Phase B embeddings → Perch 234-class predictions for every segment.

**Output**: `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet`

#### Phase D: Perch vs SED Comparative EDA

Compare Perch 234-class predictions against v5j predictions on the same 127K segments:

1. **Species coverage**: How many of the 79 v5j-zero species does Perch cover at >0.5?
2. **Complementarity**: Union of species covered by Perch-only, SED-only, and both
3. **Per-species agreement**: For species both detect, Spearman correlation per species
4. **Ensemble coverage**: What does `max(perch, sed)` or `w*perch + (1-w)*sed` cover?
5. **Per-site analysis**: Does Perch perform better on the 14 unlabeled-only sites?

**Code**: Extend `scripts/inference_eda.py`
**Output**: Updated `notes/2026-04-09-inference-eda-deep-dive.md` + new sections

#### Phase E: Ecological Sanity Check on 54 Suspect Species

Quick manual/web check: are the 54 species with high focal count but zero soundscape
predictions ecologically expected in Pantanal wetland soundscapes?

Categorize into:
- **Genuinely absent**: urban/habitat-specialist species not in Pantanal (e.g., House Sparrow)
- **Domain-shift failures**: species present but model fails to detect
- **Uncertain**: need more investigation

**Output**: Species list with categories in results.md

### Variables

- **Independent**: Teacher model (v5j single, v5j max-folds, Perch alone, Perch+SED)
- **Dependent**: Species coverage, prediction confidence, complementarity
- **Controlled**: Same unlabeled soundscape data, same thresholds

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Species coverage @threshold | N species with ≥1 segment above threshold |
| Complementarity | % of unseen species covered by Perch but NOT SED (and vice versa) |
| Per-species Spearman r | Agreement between Perch and SED on shared species |
| Pseudo-label pool size | N segments × N species at each threshold |

### Baseline

- v5j single-fold: 80/159 unseen species at >0.5
- v5j 4-fold averaged: 68/159 (worse than single)
- 79 species with zero coverage from any v5j variant

### Success Criteria

- **Phase A succeeds if**: max-across-folds recovers ≥10 of the 79 zero-coverage species
- **Phase D succeeds if**: Perch+SED union covers ≥120/159 unseen species at >0.3
- **Overall succeeds if**: we have enough data to design a concrete pseudo-labeling pipeline
  with clear teacher choice, threshold, and expected species coverage

## Limitations

- Perch TF inference is CPU-only and slow (~21h). Cannot iterate quickly.
- MLP probes were trained on 708 labeled soundscape segments — they may not generalize
  perfectly to the 127K unlabeled segments (different sites, different species mix).
- Ecological sanity check is manual and approximate — no authoritative species range maps.
- This experiment produces no LB submissions — the value is entirely in informing exp 025.

## Key References

| Document | Location |
|----------|----------|
| Inference EDA deep dive | `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` |
| Data-centric pivot plan | `user/murilo/notes/2026-04-09-data-centric-pivot.md` |
| Kaggle discussion (CV/LB) | `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` |
| Original EDA | `docs/references/birdclef-2026-eda-analysis.md` |
| Inference EDA script | `user/murilo/scripts/inference_eda.py` |
| Perch ONNX model | `shared/models/perch-v2-onnx/perch_v2.onnx` |
| Perch TF model | `shared/models/perch-v2-cpu/` |
| ProtoSSM v18 artifacts | `shared/artifacts/020-perch-onnx/pretrained-protossm-v18/pretrained_artifacts/` |
| v5j inference outputs | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/` |
| v5j-4fold inference outputs | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/` |
