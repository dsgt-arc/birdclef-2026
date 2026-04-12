# Experiment 027: Model Training Fixes — Tasks

## Goal

Fix two fundamental implementation errors identified by deep research (Claude +
Gemini, 2026-04-11). Both LLMs independently converged on the same diagnosis:
our MixUp and loss implementations were wrong compared to BirdCLEF winners.

**Baseline**: v5j (exp 019) = 0.903 LB single model
**Target**: > 0.910 LB (meaningful improvement from corrected implementations)
**Stretch**: > 0.920 LB (match Salman's single-model level)

**Branch**: `mg/027-model-training-fixes`

---

## Phase 1: Combined Fix (v10a — both fixes together)

Hypothesis: Additive waveform MixUp + CE on primary label are independently
validated by BirdCLEF 2024-2025 top solutions. Testing both together first
since both have high-confidence evidence.

- [x] Add `mixup_mode` parameter to `SEDFinetuneModule` (shared source)
- [x] Create exp 027 scaffold (train.py, save_metrics.py, sbatch, pyproject.toml)
- [x] Create `v10a_addmix_ce.yaml` config (additive MixUp + CE + 5s LSEHead)
- [ ] Fast-dev-run sanity check (verify additive MixUp + CE loss fires correctly)
- [ ] Full training run (90/10 split, 20 epochs)
- [ ] Record offline metrics (val/auroc, val_sc/auroc, cmap, top1)
- [ ] ONNX export + Kaggle submission
- [ ] LB score → compare against v5j baseline (0.903)

## Phase 2: Ablation (conditional on Phase 1 result)

If v10a improves or regresses, isolate which fix contributed:

- [ ] v10b: Additive MixUp only (keep BCE loss) — isolate MixUp contribution
- [ ] v10c: CE loss only (keep interpolative spectrogram MixUp) — isolate CE contribution

## Phase 3: Refinements (conditional on Phase 1/2)

- [ ] Label smoothing ablation: test label_smoothing=0.05, 0.1 with CE
- [ ] SUMix variant: RMS-normalized additive mixing instead of raw addition
- [ ] 4-fold ensemble with best config
- [ ] Submit ensemble to LB

## Success Criteria

- [ ] v10a single model LB > 0.906 (improve on v5j baseline)
- [ ] v10a single model LB > 0.910 (meaningful improvement)
- [ ] Identify which fix (MixUp vs CE) contributed more via ablation
- [ ] Stretch: single model LB > 0.920
