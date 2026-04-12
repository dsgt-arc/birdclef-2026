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
- [x] Fast-dev-run sanity check ✅ — job 6439254, V100, 5m08s. CE loss=5.454
  (matches log(234)≈5.46 for random softmax), both val dataloaders fire,
  waveform mixup before mel confirmed. All working correctly.
- [x] Full training run (90/10 split, 20 epochs) ✅ — job 6439641, RTX 6000, 38m51s.
  Best val_sc/auroc at epoch 11 (0.863), best val/auroc at epoch 18 (0.839).
- [x] Record offline metrics ✅ — val/auroc=0.835, val/cmap=0.571, val/top1=0.733,
  val_sc/auroc=0.863. Focal metrics lower than v5j (0.874 auroc) — expected with
  CE loss producing different logit calibration. See results.md for analysis.
- [x] Fixed checkpoint monitor: switched from val_sc/auroc → val/auroc in config.
  val_sc/auroc picked epoch 11, but per-epoch analysis shows val/auroc peaks at
  epoch 18 (0.839). Using `last.ckpt` (epoch 19, val/auroc=0.835) for export —
  matches v5j protocol and is within noise of the peak.
- [ ] ONNX export of `last.ckpt` (epoch 19) + Kaggle submission
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
