# Experiment 026: Domain-Shift Training — Tasks

## Goal

Close the single-model LB gap: 0.903 (v5j) → 0.922+ (Salman/Boredom). The gap
likely comes from window length (5s vs 20s), head architecture (LSEHead vs AttHead),
and possibly loss function / augmentation strategy.

Status: **Phase 2 complete. Phase 3 training in progress (v9a/v9b/v9c).**

**Branch**: `mg/026-domain-shift-training`

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
- [x] Fast-dev-run ✅ — both val dataloaders fire, both metric sets logged (job 6423218)

## Phase 2: v5j-dualval Baseline ✅

- [x] Train v5j-dualval (90/10 split) ✅ — 20 epochs, job 6425604, RTX 6000, 39m55s
- [x] Record focal AUC vs soundscape AUC:
  - val/auroc (focal): **0.849**
  - val_sc/auroc (soundscape): **0.094** ⚠️ BROKEN (see below)
- [ ] Submit to LB → compare vs v5j (0.903–0.906)
- [x] Best checkpoint: epoch 17 (by val_sc/auroc)

### ⚠️ val_sc/auroc metric is broken

The soundscape macro AUROC of 0.094 is meaningless. With only 86 soundscape val
segments, ~35 of 234 species have any positives. The macro average includes ~199
species with zero positives (AUROC undefined → ~0), dragging the average to ~0.09.
The metric barely moved across training (0.046 → 0.094) — it's noise.

**Fix needed**: compute AUROC only for species with ≥1 positive in soundscape val.
This requires a custom metric or post-hoc filtering. Not blocking Phase 3
(AttHead configs don't depend on this), but must be fixed before soundscape AUC
can guide checkpoint selection.

## Phase 3: Window Length + Head Ablation (training in progress)

Already tested (from exp 021):
- v5j: 5s, LSEHead, 1 chunk → **0.906 LB**
- v5i: 10s, LSEHead, 1 (naive) → 0.880 LB
- v8a: 15s, LSEHead, 3 chunks → 0.882 LB
- v8b: 20s, LSEHead, 4 chunks → 0.896 LB

Untested: **AttHead + longer windows** (single prediction, attention selects frames)

- [x] Train 10s + AttHead (v9a) ✅ — job 6426103, 1h07m, val/auroc=0.859,
  val/cmap=0.645, val/top1=0.827. Best checkpoint epoch 9 (broken val_sc metric).
  Strong offline improvement vs v5j-dualval (+0.010 auroc, +0.049 cmap, +0.075 top1).
- [ ] Train 15s + AttHead (v9b) — submitted job 6426104 (training)
- [ ] Train 20s + AttHead (v9c) — submitted job 6426105 (training)
- [ ] Record offline metrics for v9b/v9c when complete
- [ ] Submit most promising to LB
- [ ] Compare against v5j baseline (0.903–0.906)

## Phase 4: Further Ablations (conditional on Phase 3)

- [ ] Waveform MixUp (with best window/head from Phase 3)
- [ ] CE loss + sigmoid inference (with best window/head from Phase 3)
- [ ] Energy-based crop selection (RMS sampling instead of random)

## Success Criteria

- [x] Dual validation infrastructure works — ✅ (but metric needs fixing)
- [ ] Fix val_sc/auroc to only include species with positives
- [ ] Single model LB > 0.910 (Phase 3 — meaningful improvement over 0.906)
- [ ] Single model LB > 0.920 (stretch — match Salman's level)
