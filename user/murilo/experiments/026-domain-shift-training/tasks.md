# Experiment 026: Domain-Shift Training — Tasks

## Goal

Close the single-model LB gap: 0.903 (v5j) → 0.922+ (Salman/Boredom). The gap
likely comes from window length (5s vs 20s), head architecture (LSEHead vs AttHead),
and possibly loss function / augmentation strategy.

Status: **Phase 1 (dual validation) in progress.**

**Branch**: `mg/025-iter-pseudo-labeling` (shared branch for now)

---

## Phase 1: Dual Validation Infrastructure ✅

- [x] Add `dual_validation` param to `SEDDataModule` (`src/birdclef/data/sed.py`)
- [x] Store focal_val and soundscape_val datasets separately
- [x] Return list of 2 dataloaders from `val_dataloader()` when dual_validation=True
- [x] Add `dual_validation` param to `SEDFinetuneModule` (`src/birdclef/lightning/sed_finetune.py`)
- [x] Add `val_sc_*` metric sets for soundscape-only validation
- [x] Handle `dataloader_idx` in `validation_step()`
- [x] Log `val_sc/auroc`, `val_sc/cmap`, `val_sc/f1_macro` in `on_validation_epoch_end()`
- [x] Create exp 026 scaffold (train.py, configs, sbatch, pyproject.toml)
- [x] Create `v5j_dualval.yaml` config (v5j + dual_validation + monitor val_sc/auroc)
- [ ] Fast-dev-run to verify dual validation works

## Phase 2: Re-establish Baseline

- [ ] Train v5j-dualval fold 0 (90/10 split)
- [ ] Record focal AUC vs soundscape AUC (first time seeing both)
- [ ] Submit to LB → should match v5j (~0.903-0.906)
- [ ] Establish soundscape AUC baseline for comparison

## Phase 3: Window Length + Head Ablation

Already tested (from exp 021):
- v5j: 5s, LSEHead, 1 chunk → **0.906 LB**
- v5i: 10s, LSEHead, 1 (naive) → 0.880 LB
- v8a: 15s, LSEHead, 3 chunks → 0.882 LB
- v8b: 20s, LSEHead, 4 chunks → 0.896 LB

Untested: **AttHead + longer windows** (single prediction, attention selects frames)

- [ ] Train 10s + AttHead (v9a) → compare dual val metrics
- [ ] Train 15s + AttHead (v9b) → compare
- [ ] Train 20s + AttHead (v9c) → compare
- [ ] Submit most promising to LB

## Phase 4: Further Ablations (conditional)

- [ ] Waveform MixUp (with best window/head from Phase 3)
- [ ] CE loss + sigmoid inference (with best window/head from Phase 3)

## Success Criteria

- [ ] Soundscape AUC metric correlates with LB (Phase 2 validation)
- [ ] Single model LB > 0.910 (Phase 3 — meaningful improvement over 0.906)
- [ ] Single model LB > 0.920 (stretch — match Salman's level)
