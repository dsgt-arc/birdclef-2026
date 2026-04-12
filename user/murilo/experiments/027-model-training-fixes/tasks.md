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

## Experiment Outcome: CLOSED — ALL HYPOTHESES REJECTED

**Every deep research recommendation tested and failed:**

| Change | LB | Delta |
|--------|-----|-------|
| v10a: additive MixUp + CE loss | 0.494 | -0.412 |
| v10b: additive MixUp + BCE loss | 0.875 | -0.031 |
| I-1: overlapping window inference | 0.906 | +0.000 |

**Additive MixUp comparison** (all waveform MixUp variants regress):

| MixUp variant | LB | Delta |
|--------------|-----|-------|
| v5j (interpolative spectrogram) | 0.906 | baseline |
| v6b (interpolative waveform, exp 021) | 0.887 | -0.019 |
| v10b (additive waveform, exp 027) | 0.875 | -0.031 |

The deep research diagnosis was wrong: our MixUp and CE implementations were
not "errors" — they were correct choices for our pipeline. The competition
techniques that work for other teams do not transfer to our specific setup
(HGNetV2-B0 + LSEHead + focal→soundscape domain shift).

**Training recipe changes: 16+ tests, 0 successes.** The evidence is now
overwhelming. The path to 0.92+ requires either (a) fundamentally different
approach (not recipe tweaks), or (b) understanding exactly what makes our
pipeline different from Salman's 0.922 single model.

## Next: Deep research with full experimental context

Stop guessing. Conduct targeted research to identify the specific pipeline
differences between our v5j (0.906) and Salman's 0.922 single model. Focus
on data-centric and architectural differences, not training recipe variations.
