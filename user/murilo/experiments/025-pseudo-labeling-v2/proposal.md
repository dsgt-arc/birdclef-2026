# Proposal: Pseudo-Labeling v2

**Status**: proposed
**Created**: 2026-04-10
**Author**: Murilo Gustineli
**Branch**: TBD (create from `mg/024-data-centric-exploration` or `main`)
**Depends on**: Exp 024 (all findings), Exp 021 (v5j training recipe), Exp 023 (ensemble inference)

## Hypothesis

Adding SED-generated pseudo-labels for 127K unlabeled soundscape segments will improve
LB by +0.005-0.010 by providing same-domain training data for the 159 species currently
unseen in soundscapes. This directly addresses the data coverage gap identified in exp 024.

## Background

### What exp 024 established

- The LB gap (0.929 → 0.948) is a **data coverage problem**: 0.9958 AUC on 75 seen species,
  159 species have zero soundscape data
- SED max-folds covers **109/159** unseen species at >0.5 with **94% precision**
- Perch covers 149/159 but with **6% precision** — permanently positive logits, not usable raw
- Only **6 species** are truly invisible (1-5 focal recordings)
- All 54 original "data quality suspects" are resolved
- Max-across-folds dramatically outperforms averaging (+29 species at >0.5)

### Why exp 015 failed and why this should work now

| Factor | Exp 015 (failed) | Exp 025 (this) |
|--------|------------------|----------------|
| Teacher LB | 0.767 | **0.929** (5× stronger) |
| Architecture | Clip-level (ProtoCLR/EAT) | **SED** (frame-level predictions) |
| Pseudo-label precision | Unknown, likely low | **94%** (SED on labeled soundscapes) |
| Domain match | Focal → soundscape | **Soundscape → soundscape** |
| Power transform | None | **P^1.5** (Nikita's technique) |
| Multi-label | Single-label | **Multi-label BCE** with soft targets |
| Species coverage | Unknown | **109/159** unseen at >0.5 |

### Reference: Nikita 1st place BirdCLEF 2025

- 3-4 pseudo-labeling iterations
- Temperature 0.55-0.6, power transform
- 7 diverse models × 5 backbones
- Used unlabeled soundscape data as same-domain training signal

## Method

### Approach

Train v5j on focal + labeled soundscapes + pseudo-labeled soundscapes. Use SED max-folds
as teacher. Keep the v5j recipe exactly as-is (FINDINGS.md: every recipe change regresses).
The ONLY change is adding pseudo-labeled data.

### Phase A: Generate Pseudo-Labels

**Already done** — SED 4-fold inference on all 10,658 soundscape files exists at
`shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/`.

Processing pipeline:
1. Load 4-fold logits → sigmoid → `max(fold0, fold1, fold2, fold3)` per species per segment
2. Apply power transform P^1.5 (suppress uncertain predictions)
3. Threshold at >0.5: keep species with probability >0.5 after power transform
4. Save multi-hot soft labels: 127,896 segments × 234 species probability matrix

**Expected yield**: ~5K pseudo-labels (species × segment pairs) across ~109 unseen species,
plus expanded coverage for 75 seen species on new sites/hours.

### Phase B: Train Student (Round 1)

Train v5j recipe on expanded dataset:

| Data Source | Segments | Label Type | Loss Weight |
|-------------|----------|------------|-------------|
| Focal (train_audio) | ~265K | Hard ground truth | 1.0 |
| Labeled soundscapes | ~792 | Hard ground truth | 1.0 |
| Pseudo-labeled soundscapes | ~127K | Soft (P^1.5 probs) | 0.5 |

Training details:
- Recipe: v5j (HGNetV2-B0, LSEHead, BCE, 5s windows, 20 epochs)
- Batch: 50% focal/labeled, 50% pseudo-labeled (stratified sampling)
- Loss: BCE on all sources, weighted by confidence
- Scheduler: OneCycleLR (same as v5j)
- No recipe changes — only data changes

### Phase C: Evaluate and Iterate

1. Submit round 1 student to LB
2. **If LB improves** (>0.929):
   - Student becomes teacher for round 2
   - Consider adding Perch Tier C (40 species) with top-K filtering
   - Consider backbone diversity (EfficientNet-B3 NS as second backbone)
3. **If LB regresses**:
   - Diagnose: over-pseudo-labeling? Wrong loss weight? Recipe sensitivity?
   - Try: higher threshold (>0.7), lower PL weight (0.3), fewer PL segments
   - Try: conservative variant — only segments with max species prob >0.8

### Variables

| Variable | Type | Values |
|----------|------|--------|
| Pseudo-label threshold | Independent | 0.5, 0.7 |
| Power transform exponent | Independent | 1.5, 2.0 |
| Pseudo-label loss weight | Independent | 0.3, 0.5 |
| Batch PL fraction | Independent | 0.3, 0.5 |
| Training recipe | Controlled | v5j (unchanged) |
| Focal + labeled data | Controlled | Same as v5j baseline |

### Setup

| Component | Details |
|-----------|---------|
| Data | Focal (35,549 files) + labeled soundscapes (66 files) + pseudo-labeled soundscapes (10,658 files) |
| Compute | GPU (gpu-rtx6000), ~4-6h per training run |
| Code | Extend `experiments/021-sed-training/` data loader to include pseudo-labels |
| Evaluation | LB submission (offline metrics unreliable) |

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Public LB** | Primary metric — only reliable evaluation |
| Macro-AUC on labeled soundscapes | Sanity check (should remain >0.99) |
| Pseudo-label species coverage | How many species get pseudo-labels at each threshold |
| Training loss convergence | Should converge similarly to v5j baseline |

### Baseline

- v5j single-fold: **0.906** LB
- v5j 4-fold: **0.913** LB
- Perch+SED ensemble: **0.929** LB (current best)

### Success Criteria

- **Round 1**: Student model (single fold) exceeds **0.910** LB (v5j baseline + 0.004)
- **Round 1 ensemble**: 4-fold student exceeds **0.920** LB
- **Best case**: Student+Perch ensemble exceeds **0.935** (current best + 0.006)
- **Round 2** (if applicable): Any further gain from iteration

## Procedure

### Round 1 (minimum viable experiment)

1. **Generate pseudo-label parquet** (Phase A)
   - Script: new `scripts/generate_pseudo_labels.py`
   - Input: SED 4-fold logits
   - Output: `shared/pseudo-labels/v1/pseudo_labels.parquet` (segment × species soft probs)
   - Parameters: max-folds, P^1.5, threshold >0.5

2. **Extend data loader** (Phase B)
   - Modify SED training pipeline to load pseudo-labeled soundscapes
   - Add stratified batch sampling (50% focal/labeled, 50% pseudo)
   - Add loss weighting (w=0.5 for pseudo-labels)

3. **Train v5j + pseudo-labels** (Phase B)
   - 4 folds × 20 epochs
   - Submit each fold individually + 4-fold ensemble

4. **Evaluate** (Phase C)
   - Compare single-fold and 4-fold LB against baselines
   - If gain: proceed to round 2

### Round 2 (conditional on round 1 success)

5. Student-as-teacher: run round 1 student inference on soundscapes
6. Generate new pseudo-labels with student predictions
7. Train round 2 student on improved pseudo-labels
8. Optionally add Perch Tier C (40 species, top-K segments)

### Variants to test (if round 1 succeeds)

| Variant | Change | Rationale |
|---------|--------|-----------|
| Higher threshold | >0.7 instead of >0.5 | More precision, fewer labels |
| Lower PL weight | 0.3 instead of 0.5 | Less influence from noisy labels |
| Power 2.0 | P^2.0 instead of P^1.5 | More aggressive noise suppression |
| Perch Tier C | Add 40 Perch-only species | Coverage expansion (risky) |
| Backbone diversity | EfficientNet-B3 NS | Diverse ensemble teacher for round 2 |

## Limitations

1. **No reliable offline evaluation** — must use LB submissions (5/day limit)
2. **Pseudo-label precision unknown for unseen species** — 94% is measured on seen species only
3. **Single backbone** — v5j recipe, no backbone diversity in round 1
4. **Perch Tier C deferred** — 40 species with 6% precision too risky for round 1
5. **6 invisible species** cannot improve regardless

## Key References

| Document | Location |
|----------|----------|
| Exp 024 results (all findings) | `user/murilo/experiments/024-data-centric-exploration/results.md` |
| Inference EDA deep dive (14 sections) | `user/murilo/notes/2026-04-09-inference-eda-deep-dive.md` |
| Data-centric pivot plan | `user/murilo/notes/2026-04-09-data-centric-pivot.md` |
| Exp 015 failure analysis | `user/murilo/FINDINGS.md` (insight #10) |
| Nikita 1st place approach | `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` |
| v5j training recipe | `user/murilo/experiments/021-sed-training/` |
| SED 4-fold inference outputs | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/` |
| Perch 234-class probs | `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/` |
