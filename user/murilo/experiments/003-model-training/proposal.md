# Proposal: Model Training (ConvNeXT-Tiny & ViT-Small)

**Status**: completed
**Created**: 2026-02-08
**Author**: Murilo

## Hypothesis

Training ConvNeXT-Tiny and ViT-Small from scratch on HSN will produce substantially lower cmAP than ImageNet-pretrained variants, quantifying the pretrain advantage. ConvNeXT-Tiny pretrained should approach exp001's cmAP=0.480, and both scratch models will underperform exp002's ViT-Base pretrained (cmAP=0.164).

## Background

Experiment 002 trained ViT-Base (85.5M params) with ImageNet pretrained weights on HSN, achieving cmAP=0.164 (34% of BirdSet baseline). Experiment 001 ran inference with a pre-trained ConvNeXT from `bioacoustics_model_zoo` (cmAP=0.480). Comparing a scratch-trained ViT against a pre-trained ConvNeXT is unfair. This experiment trains both architectures from scratch and with pretrained weights to establish fair, size-comparable baselines.

## Method

### Approach

Train 4 models in a 2x2 matrix: {ConvNeXT-Tiny, ViT-Small} x {scratch, ImageNet-pretrained}. Use a generic `AudioClassifier` class that handles 1-channel input adaptation for both model families.

### Design Rationale: Model Size Parity

ConvNeXT-Tiny was chosen over ConvNeXT-Small to keep parameter counts comparable with ViT-Small:

| Model | Params | Family |
|-------|--------|--------|
| ViT-Small | ~22M | Transformer |
| ConvNeXT-Tiny | ~28.6M | CNN |
| ConvNeXT-Small | ~50M | CNN |

Exp 002 used ViT-Base (85.5M params), making any comparison with ConvNeXT unfair due to the ~3x size gap. By pairing ConvNeXT-Tiny (~28.6M) with ViT-Small (~22M), both models sit in the ~20-30M range, isolating architecture differences (CNN vs Transformer) from capacity differences. ConvNeXT-Small at ~50M would reintroduce this confound.

### Setup

| Component | Details |
|-----------|---------|
| Data | HSN dataset from BirdSet, mel-spectrograms (128x500) |
| Compute | 1x RTX 6000 GPU, 64GB RAM, 8 CPUs, up to 12h per run |
| Dependencies | timm>=0.9.0, PyTorch Lightning, W&B |
| Code | `user/murilo/experiments/003-model-training/` |

### Procedure

1. Create `AudioClassifier` supporting both ViT and ConvNeXT families
2. Train all 4 configurations on HSN
3. Compare cmAP across all variants

### Variables

- **Independent**: Architecture (ConvNeXT-Tiny vs ViT-Small), initialization (scratch vs ImageNet-pretrained)
- **Dependent**: cmAP, training curves, convergence speed
- **Controlled**: Dataset (HSN), input size (128x500), training infrastructure

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| cmAP | Class-mean average precision (primary) |
| val/loss | Validation loss (convergence indicator) |

### Baseline

| Run | cmAP |
|-----|------|
| Exp 001 - ConvNeXT (bioacoustics_model_zoo) | 0.480 |
| Exp 002 - ViT-Base pretrained | 0.164 |

### Success Criteria

- **Confirm if**: Pretrained variants achieve higher cmAP than scratch variants for both architectures
- **Reject if**: Scratch variants match or exceed pretrained variants (would suggest pretraining is not helpful)

## GPU Utilization Investigation

### Problem

ConvNeXT-Tiny training showed spiky GPU utilization (bursts to 100%, drops to 0%) while ViT-Small in exp 002 maintained steady utilization with the same data pipeline.

### Root Cause

ConvNeXT uses depthwise separable convolutions that are heavily CUDA-optimized and processes each batch significantly faster than ViT. ViT-Small with `patch16` on a 128x500 input creates ~248 patches, and self-attention is O(n^2) on sequence length, so each forward/backward pass takes long enough for the data pipeline to prepare the next batch. ConvNeXT finishes each batch before the next one arrives from network storage, causing the GPU to idle waiting for I/O.

In short: ViT was never fast enough to expose the data-loading bottleneck. ConvNeXT is.

### Data Pipeline Bottleneck

Each `__getitem__` call performed these steps every batch:
1. `soundfile.sf.read(path)` -- network I/O from `/storage/home/...` (primary bottleneck)
2. Stereo-to-mono conversion
3. Resampling check (no-op for HSN since source=target=32kHz)
4. Pad/crop to 5s
5. Mel spectrogram computation (CPU FFT)

### Optimizations Applied

| Commit | Change | Impact |
|--------|--------|--------|
| `019fd6f` | Vectorize mel-spectrogram transform, increase batch to 128 | Reduced per-batch mel overhead |
| `c2b65cd` | Move mel-spectrogram from main-process collate into worker `__getitem__` | 8x parallelism for mel computation |
| `b46bd94` | Reduce `num_workers` from 8 to 6 | Match PACE GPU CPU:GPU ratio limit |
| `60fa19c` | Add opt-in `cache_mel_specs` to pre-compute all mels in `setup()` | Eliminates all disk I/O during training |

### Mel Caching

The final optimization pre-computes all mel spectrograms during `setup()` and caches them in RAM. HSN totals ~4.4 GB (17,460 samples x (1, 128, 500) x 4 bytes), fitting easily in the 64 GB allocation. This is controlled by `cache_mel_specs: true` in config (default `false` for backward compatibility with larger datasets).

Trade-off: setup takes ~34 minutes (sequential reads from network storage) but training loop becomes memory-only with near-constant GPU utilization.

### Results: Before vs After Caching (ConvNeXT-Tiny Scratch)

| | Before (on-the-fly, SLURM) | After (cached, interactive) |
|---|---|---|
| **Batch size** | 32 | 128 |
| **GPU utilization** | Spiky (0-100%) | Steady ~75-85% |
| **Epochs** | 100 (no early stop) | 26 (early stopped) |
| **Time per epoch** | ~34s | ~12s |
| **Caching overhead** | N/A | ~34 min (one-time) |
| **Total walltime** | ~57 min | <37 min |
| **Test cmAP** | 0.0557 | 0.0452 |
| **Test AUROC** | 0.6214 | 0.5726 |
| **Test top-5 acc** | 0.0348 | 0.3495 |
| **Test batches/s** | 15.6 it/s (bs=32) | 8.0 it/s (bs=128) |

Notes:
- The cmAP difference (0.056 vs 0.045) is likely random seed / early stopping variance — both are low as expected for scratch training.
- The top-5 accuracy jump (3.5% → 35%) is likely due to the batch size increase (32 → 128) improving gradient quality, not the caching itself.
- The "before" run used on-the-fly mel computation with 6 workers and no caching. The "after" run pre-computed all mels in `setup()`.
- GPU utilization drops between training and validation epochs are expected and acceptable.

## Limitations

- Only testing on HSN dataset; generalization to XCM not tested
- Scratch hyperparameters (lr, warmup, epochs) are educated guesses; may need tuning
- ConvNeXT-Tiny (~28.6M) and ViT-Small (~22M) are similar but not identical in parameter count
