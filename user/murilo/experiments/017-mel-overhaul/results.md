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

## Artifacts

| Artifact | Path |
|----------|------|
| Best checkpoint | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-effnet-b0-sed-v4a/best-epochepoch=24.ckpt` |
| Metrics JSON | `artifacts/training/efficientnet_b0_sed_v4a/metrics.json` |
| W&B run | `dsgt-arc/birdclef-2026-mel-overhaul/runs/05pylg1y` |
| ONNX model | `010-.../artifacts/weights/effnet-b0-sed-v4a/model.onnx` (pending export) |
