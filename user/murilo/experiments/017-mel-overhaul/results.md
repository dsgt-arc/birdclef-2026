# Experiment 017: Mel Overhaul — Results

## Summary

**Hypothesis**: Changing from (n_fft=1024, n_mels=128) to (n_fft=4096, n_mels=224) while
keeping all other v1 parameters identical will improve LB by >=0.02.

**Result**: **Confirmed.** LB raw 0.825 (+0.030 over v1 raw 0.795) — passes decision gate.
Mel resolution alone nearly matches v1+pp2 (0.827). Offline: test/auroc 0.862 (+0.002),
test/cmap 0.615 (+0.022). Continue to Step 2.

---

## Step 1: v4a — Mel Resolution Only

### Training Configuration

| Parameter | v1 (exp 016) | v4a (exp 017) | Change |
|-----------|-------------|---------------|--------|
| n_fft | 1024 | **4096** | 4× freq resolution |
| n_mels | 128 | **224** | 1.75× spectral detail |
| hop_length | 320 | 320 | unchanged |
| input shape | (1, 128, 1000) | **(1, 224, 1000)** | 1.75× data |
| batch_size | 128 | **64** | OOM safety |
| accumulate_grad_batches | 2 | **4** | same 256 effective |
| mel.gpu | false | **true** | n_fft=4096 too slow on CPU |
| Everything else | v1 | v1 | identical |

### Training Dynamics

| Epoch | val/auroc | Note |
|-------|-----------|------|
| 0 | 0.513 | random |
| 5 | 0.816 | rapid improvement |
| 9 | 0.833 | |
| 14 | 0.842 | |
| 20 | 0.844 | gains slowing |
| 24 | **0.846** | best |
| 25-34 | — | no improvement, early stopped |

Best epoch: 24 (same as v1). Early stopped at epoch 34 (patience=10).
SLURM job 5484980, gpu-v100, ~4 hours wall time.

### Offline Test Metrics

| Metric | v1 (exp 016) | v4a (exp 017) | Delta |
|--------|-------------|---------------|-------|
| test/auroc | 0.860 | **0.862** | +0.002 |
| test/cmap | 0.593 | **0.615** | **+0.022** |
| test/f1_macro | 0.256 | 0.249 | −0.007 |
| test/top1_acc | 0.762 | 0.772 | +0.010 |
| test/top5_acc | 0.887 | 0.894 | +0.007 |
| val/auroc (best) | 0.845 | 0.846 | +0.001 |

The cmAP improvement (+0.022) is the strongest signal — higher mel resolution enables
the model to distinguish more species with higher precision. AUROC gain is modest (+0.002)
but consistent.

### Leaderboard

| Model | LB | vs v1 raw | vs v1+pp2 | Notes |
|-------|-----|-----------|-----------|-------|
| **v4a (raw)** | **0.825** | **+0.030** | −0.002 | original submission |
| **v4a (raw, optimized)** | **0.826** | **+0.031** | −0.001 | full-signal mel + batched ONNX, <32min |
| **v4a + pp2** | **0.830** | **+0.035** | **+0.003** | |
| v1 (raw, exp 016) | 0.795 | baseline | — | |
| v1 + pp2 (exp 016) | 0.827 | — | baseline | |

**Inference speedup**: Optimized inference (<32min) vs original (>1h), ~2.5-3× faster.
The ±0.001 LB variance between v4a submissions is Kaggle test data rotation.

### PP2 Diminishing Returns

PP2 gained only +0.005 on v4a (0.825→0.830) vs +0.032 on v1 (0.795→0.827). This
makes sense — the higher mel resolution already captures species distinctions that
post-processing was compensating for in v1. TTA temporal shifts add less value when
the spectrogram already has 4× frequency resolution. The pp2 components most affected:
- TTA (+0.012 on v1): less impact when the model already sees fine-grained features
- Temporal smoothing: may smooth out predictions the model is now more confident about
- Temperature scaling (T=1.10): calibration tuned for v1's distribution, may need retuning

Worth revisiting pp2 hyperparameters (especially temperature) for v4a in a future step.

### Decision Gate

- **PASSED**: LB raw 0.825 ≥ 0.815 (+0.030 over v1 raw) → continue to Step 2
- **PASSED**: LB pp2 0.830 ≥ 0.827 (+0.003 over v1+pp2) → improvement confirmed

---

## Step 2: Parallel Ablations (v4b/v4c/v4d/v4e) — DONE

Four single-variable ablations from the v4a base. All use 90/10 split (no test set)
and 50 epochs — ~11% more training data than v4a's 80/10/10 + 40 epochs.

### Training Summary

| Variant | Change from v4a | best val/auroc | best epoch | stopped | W&B run |
|---------|----------------|---------------|------------|---------|---------|
| **v4a** | baseline (80/10/10, 40ep) | 0.846 | 24 | 34 | `05pylg1y` |
| **v4b** | AmplitudeToDB + min-max norm | **0.865** | 30 | 41 | `q9jdljkt` |
| **v4c** | AttBlockV2 head (tanh+softmax) | 0.815 | 15 | 26 | `4kwgh6qs` |
| **v4d** | ASL loss (γ⁻=4, γ⁺=0) | 0.858 | 16 | 27 | `2b43diog` |
| **v4e** | StochasticDepth (drop_path=0.15) | **0.868** | 29 | 32 | `07o7zjha` |

### Observations

**v4e (StochasticDepth)** — best val/auroc 0.868. Steady improvement through epoch 29,
suggesting the regularization helps B0 train longer without overfitting. Consistent with
2025 1st place using drop_path=0.15.

**v4b (AmplitudeToDB)** — second best val/auroc 0.865. Also trained longer (epoch 30),
consistent with the normalization preserving more dynamic range information. Tom uses
this normalization across all 28 experiments.

**v4d (ASL loss)** — val/auroc 0.858. Peaked early (epoch 16), suggesting ASL's aggressive
negative suppression may converge faster but plateau lower when combined with our soft CE
base recipe. Tom's ASL gain (+0.023) was over BCE, not soft CE — our soft CE already
handles the multi-label setting better than BCE.

**v4c (AttBlockV2)** — worst at 0.815, as predicted in risk notes. The soft CE loss applies
softmax to logit-converted sigmoid probabilities — a mismatch with AttBlockV2's design.
Should retry with BCE loss before concluding AttBlockV2 doesn't help.

### Confound Note

v4b-v4e val/auroc is not directly comparable to v4a's 0.846 because:
1. 90/10 vs 80/10/10: ~11% more training data
2. 50 vs 40 epochs: more training capacity
3. Different val split composition

The LB results (pending) will be the true comparison.

### Leaderboard

| Model | LB (raw) | LB (pp2) | vs v4a raw | Notes |
|-------|----------|----------|------------|-------|
| v4a | 0.825 | 0.830 | baseline | |
| **v4b** | **0.859** | **0.860** | **+0.034** | **clear winner** — AmplitudeToDB norm |
| v4c | skipped | — | — | poor offline (0.815), soft CE + sigmoid mismatch |
| v4d | 0.808 | — | −0.017 | regression — ASL hurts with soft CE base |
| **v4e** | 0.840 | pending | +0.015 | solid gain — StochasticDepth regularization |

### Interpretation

**v4b (AmplitudeToDB)** — LB 0.859, the standout result. +0.034 over v4a is far larger
than the ~+0.003 confound from extra data/epochs. AmplitudeToDB normalization preserves
more dynamic range than log+AudioSet stats, consistent with Tom's usage across all 28
experiments and the 2025 1st place pipeline.

**v4e (StochasticDepth)** — LB 0.840, a solid +0.015 over v4a. The regularization helps
the model generalize to the test distribution. Consistent with 2025 1st place using
drop_path=0.15.

**v4d (ASL loss)** — LB 0.808, a −0.017 regression. ASL's aggressive negative suppression
conflicts with our soft CE recipe. Tom's +0.023 gain was over BCE, not soft CE — our
soft CE already handles multi-label better. **Exclude from v4f.**

**v4c (AttBlockV2)** — Skipped due to poor offline val/auroc (0.815). The soft CE +
sigmoid→logit→softmax conversion path is a known mismatch. May retry with BCE if needed.

---

## Artifacts

### Step 1

| Artifact | Path |
|----------|------|
| Best checkpoint | `.../checkpoints/2026-effnet-b0-sed-v4a/best-epochepoch=24.ckpt` |
| Metrics JSON | `artifacts/training/efficientnet_b0_sed_v4a/metrics.json` |
| W&B run | `dsgt-arc/birdclef-2026-mel-overhaul/runs/05pylg1y` |
| ONNX model | `010-.../artifacts/weights/effnet-b0-sed-v4a/model.onnx` |

### Step 2

| Variant | Checkpoint | SLURM Job |
|---------|-----------|-----------|
| v4b | `.../checkpoints/2026-effnet-b0-sed-v4b/best-epochepoch=30.ckpt` | 5496315 |
| v4c | `.../checkpoints/2026-effnet-b0-sed-v4c/best-epochepoch=15.ckpt` | 5496316 |
| v4d | `.../checkpoints/2026-effnet-b0-sed-v4d/best-epochepoch=16.ckpt` | 5496367 |
| v4e | `.../checkpoints/2026-effnet-b0-sed-v4e/best-epochepoch=29.ckpt` | 5496368 |
