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
- [x] ONNX export of `last.ckpt` (epoch 19) + Kaggle submission ✅
- [x] LB score → compare against v5j baseline (0.903) ✅ — **0.494 LB (catastrophic)**

**v10a result**: 0.494 LB. CE loss produces logits with ~1 unit spread centered
at zero. Sigmoid maps everything to ~0.50 → random predictions on soundscapes.
Root cause: CE optimizes relative ranking (softmax), not absolute magnitudes.
See results.md for full analysis.

## Phase 2: Ablation — isolate MixUp contribution

v10a combined two changes; CE was catastrophic and masks MixUp. CE is now
confirmed dead (7 tests, 0 successes). Only MixUp ablation is needed.

- [x] v10b: Additive waveform MixUp + BCE loss (isolate MixUp fix) ✅
  - Config: `v10b_addmix_bce.yaml` — same as v5j except `raw_waveform_mixup=true`,
    `mixup_mode="additive"`, `loss_type="bce"` (v5j loss)
  - Training: job 6465883, V100, 44m32s. Best val/auroc=0.847 at epoch 17.
  - Offline metrics: val/auroc=0.844, val/cmap=0.592, val/top1=0.747,
    val_sc/auroc=0.966 (+0.010 vs v5j-dualval). Similar to v5j-dualval.
- [x] ONNX export of v10b `best-epochepoch=17.ckpt` + Kaggle submission ✅
- [x] LB score → compare against v5j baseline (0.906) ✅ — **0.875 LB (-0.031)**
- [x] ~~v10c: CE loss only~~ — CANCELLED. CE confirmed dead, no value in isolating.

**v10b result**: 0.875 LB (-0.031 from v5j). Additive waveform MixUp regresses
even with BCE loss. The "corrected" formulation is worse than our interpolative
spectrogram MixUp (-0.031 vs -0.019 for v6b interpolative waveform).

## Phase 3: Inference-side tests

- [x] I-1: Overlapping window inference ✅ — **0.906 LB (flat vs v5j)**
  - Tested 10s window on v5j (2.5s context each side). No improvement.
  - Existing TTA (±2.5s shifts) already provides similar overlap — redundant.

## Phases 1-3 Outcome: All deep research hypotheses rejected

| Change | LB | Delta |
|--------|-----|-------|
| v10a: additive MixUp + CE loss | 0.494 | -0.412 |
| v10b: additive MixUp + BCE loss | 0.875 | -0.031 |
| I-1: overlapping window inference | 0.906 | +0.000 |

Generic "winner technique" recommendations all failed. The path forward requires
understanding the **specific implementation differences** between our pipeline
and Salman's 0.922 single model.

---

## Phase 4: Salman-Derived Implementation Changes

Source: Salman Ahmed's BirdCLEF 2025 training notebook + writeup (26th place,
0.908 LB) + his BirdCLEF 2026 claim of 0.922 single / 0.932 ensemble.
Reference: `notes/2026-04-12-salman-ahmed-birdclef2025.md`

These are NOT generic recipe tweaks — they are specific, concrete implementation
differences from a competitor who achieves 0.922 on this competition. Each change
addresses a structural pipeline difference, not a hyperparameter variation.

### Control baseline

- [x] v5j-control: Exact v5j reproduction in exp 027 W&B project ✅
  - Config: `configs/v5j_control.yaml` (identical to v5j except W&B project + dual_validation)
  - Job 6468944, RTX 6000, 39m24s. Best val/auroc=0.852 at epoch 14.
  - Metrics: val/auroc=0.849, val/cmap=0.596, val_sc/auroc=0.955, val_sc/cmap=0.853.
  - Matches v5j-dualval (exp 026) closely. W&B baseline verified.

### Experiment 4a: Max trick + RMS sampling (HIGHEST PRIORITY)

Salman's most distinctive technique. Fundamentally changes how the model handles
weak labels — this is NOT a recipe tweak.

**What**: Pick loudest 15s via RMS sampling → split into 3×5s segments →
model predicts all 3 → take max logits → backpropagate only on the max.
At inference: standard 5s segments (no change).

```python
BS, K, C, H, W = spec.size()     # K=3 segments
spec = spec.view(BS * K, C, H, W)
logits = model(spec)
logits = logits.view(BS, K, num_classes)
logits, _ = torch.max(logits, dim=1)  # max across segments
```

**Why this is different from all 16+ failed changes**: Every previous change
modified the loss, augmentation, or head architecture. This changes HOW the
model selects and learns from training data. The gradient flows only through
the segment with strongest evidence — silent segments get zero gradient.

Implementation needed:
- [ ] Add RMS-based crop selection to `SEDFocalDataset.__getitem__()`
- [ ] Add `max_trick` and `max_trick_factor` params to training config
- [ ] Modify `training_step()` to reshape (B, K, C, H, W) → (B*K, C, H, W),
      run forward, reshape back, take max before loss computation
- [ ] Create `v11a_maxtrick.yaml` config (15s window, max_trick_factor=3)
- [ ] Train + ONNX export + LB submission

### Experiment 4b: Mel normalization change

**What**: Replace our global mean/std standardization with Salman's simple
`(X + 80) / 80` normalization that maps [-80, 0] dB range to [0, 1].

**Why test**: Completely different normalization chain. Our pipeline:
`AmplitudeToDB → subtract mean (-4.268) → divide std (4.569) → min-max [0,1]`.
Salman's: `AmplitudeToDB → add 80 → divide 80`. Simpler, no dataset statistics.

Implementation needed:
- [x] Add `norm_type: "simple_range"` option to NNAudioMelSpectrogram ✅
- [x] Create `v11b_melnorm.yaml` config (same as v5j except normalization) ✅
- [x] Training complete ✅ — job 6469719, RTX 6000. Best val/auroc=0.852 at epoch 18.
  Offline: val/auroc=0.847, val/cmap=0.599, val_sc/auroc=0.960, val_sc/cmap=0.873.
  Flat vs v5j-control on focal (-0.002), slight soundscape improvement (+0.005/+0.020).
- [ ] ONNX export + LB submission

### Experiment 4c: Model soup (checkpoint averaging)

**What**: Instead of picking the single best checkpoint, average weights from
all checkpoints that improve any validation metric (epochs ~12-19).

**Why test**: Both Salman AND Mario (BirdCLEF 2024 2nd place) use this. Zero
extra training cost — just a post-processing step on existing checkpoints.
We need `save_top_k=-1` (save all checkpoints) to enable this.

Implementation needed:
- [ ] Add `save_top_k: -1` option to checkpoint callback in config
- [ ] Write model soup script (average checkpoints that improve val metrics)
- [ ] Test on v5j-control checkpoints first (zero training cost)
- [ ] If effective, apply to best Phase 4 variant

### Experiment 4d: BCE + Focal combined loss

**What**: `loss = BCE(logits, targets) + sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2)`.
NOT standalone focal (which failed as ASL), but BCE + Focal combined.

**Why test**: Both Salman and Mario use this exact combination. Different from
our ASL test (ASL uses asymmetric gamma, probability clipping — different mechanism).

Implementation needed:
- [ ] Add `loss_type: "bce_focal"` to `_compute_loss()` in sed_finetune.py
- [ ] Create `v11d_bcefocal.yaml` config
- [ ] Train + ONNX export + LB submission

### Experiment 4e: Upsampling rare species

**What**: Upsample species with fewer than 15-20 recordings to that threshold
(with replacement). Salman uses `upsample_data(df, thr=15)`.

**Why test**: Our 82 species with <100 recordings are systematically undertrained.
Upsampling ensures every species gets minimum representation per epoch.

Implementation needed:
- [ ] Add upsampling logic to `SEDDataModule.setup()`
- [ ] Create `v11e_upsample.yaml` config
- [ ] Train + ONNX export + LB submission

### Execution order

```
1. v5j-control (baseline for W&B comparison)
2. v11a: Max trick + RMS sampling (highest structural difference)
3. v11b: Mel normalization (easy to implement, independent variable)
4. v11c: Model soup on v5j-control checkpoints (zero training cost)
5. v11d: BCE + Focal loss (if 4a or 4b show training changes CAN work)
6. v11e: Upsampling rare species (if 4a or 4b show training changes CAN work)
```

Test one at a time. If 4a (max trick) succeeds, compound with other changes.

## Success Criteria (Phase 4)

- [ ] v5j-control reproduces ~0.903-0.906 LB (pipeline verification)
- [ ] Any single Phase 4 variant LB > 0.906 (first training change to improve LB)
- [ ] Compound best changes → LB > 0.920 (match Salman's single-model level)
- [ ] Stretch: single model LB > 0.930 (exceed Salman's ensemble)
