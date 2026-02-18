# Experiment 012: EfficientNet-B1 Linear Probe

- **Date**: 2026-02-18
- **Status**: proposed
- **Depends on**: exp 009 (data pipeline, training loop)

## Hypothesis

A frozen EfficientNet-B1 pretrained on Xeno-Canto Large (BirdSet-XCL) provides
stronger bird audio features than our SSAMBA-Tiny backbone, establishing a feature
quality ceiling for linear probe evaluation on BirdCLEF 2025.

## Background

BirdSet benchmarks show EfficientNet at 35–38 cmAP on HSN, while our SSAMBA-Tiny
achieves 0.165–0.179 cmAP on BirdCLEF 2025 val (exp 007). The gap may be due to
backbone feature quality, training recipe, or both. A linear probe isolates the
backbone contribution.

## Method

- **Backbone**: `DBD-research-group/EfficientNet-B1-BirdSet-XCL` (HuggingFace)
- **Probe**: Frozen backbone → adaptive avg pool → Dropout(0.3) → Linear(1280, 206)
- **Data**: Same as exp 009 — BirdCLEF 2025 focal recordings, 10s chunks, 90/10 split
- **Mel**: nnAudio, n_mels=128, sr=32000 (note: BirdSet used n_mels=256; using 128 for
  consistency with SSAMBA pipeline — conv backbone should handle the resolution difference)
- **Input**: Single mel channel repeated 3x for RGB-expected EfficientNet
- **Training**: AdamW, lr=1e-3, cosine schedule, no warmup, focal BCE (gamma=2), bf16
- **Augmentation**: SpecAugment (2×freq 20 bins, 2×time 50 frames)
- **Epochs**: 30 max, early stop patience 10 on val/cmap

## Expected Outcome

- Linear probe should converge fast (only ~265K trainable params vs 7M total)
- If cmAP >> 0.179, the bottleneck is SSAMBA backbone quality
- If cmAP ≈ 0.179, the training recipe is the bottleneck
- If cmAP << 0.179, n_mels=128 is hurting EfficientNet (follow-up with n_mels=256)

## Metrics

- Primary: val/cmap (class-mean average precision)
- Secondary: val/auroc, train/cmap (overfitting gap)

## Resources

- 1× RTX 6000, ~12h max (should converge much faster)
- Memory: ~16 GB (large batch + frozen backbone = low VRAM)
