---
experiment: "019-sed-recipe-polish"
status: proposed
date: 2026-03-27
author: Murilo Gustineli
---

# Proposal: SED Recipe Polish

**Status**: proposed
**Created**: 2026-03-27
**Author**: Murilo Gustineli

## Hypothesis

Small single-variable ablations on v4b (0.859 LB) can squeeze out incremental gains (+0.005-0.01 per technique) by adopting proven techniques from top BirdCLEF solutions.

## Background

Our v4b SED model achieves 0.859 LB. Tawara's HGNetV2-B0 baseline achieves the same 0.859 with techniques we haven't yet adopted: LogSumExp aggregation, SpecAugment, and tuned MixUp. The 2023 and 2024 1st place solutions also use SpecAugment. These are low-risk, high-signal changes that compound with the ensemble from exp 018.

## Method

### Approach

Three independent single-variable ablations on the v4b recipe, each targeting a different aspect of the training pipeline.

### v5a: LogSumExp Frame Aggregation (T=0.1)

- Replace our `framewise_logits.max(dim=-1)` with `T * logsumexp(timewise_logits / T, axis=1)` where T=0.1
- Tawara (0.859 LB with 4-fold HGNetV2-B0) uses this exact approach
- Smooth differentiable approximation to max — distributes gradient across near-max frames
- Implementation: Change ~5 lines in SEDFinetuneModule
- Reference: `docs/references/kaggle/birdclef-2026/discussions/hgnetv2-b0-baseline-tawara.md`

### v5b: SpecAugment (Freq + Time Masking)

- Add frequency masking (len=10, masks=3) + time masking (len=20, masks=3)
- 2023 1st place and 2024 1st place both use SpecAugment
- Applied to mel spectrogram during training only
- Implementation: torchaudio.transforms.FrequencyMasking + TimeMasking
- Reference: `docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md`

### v5c: MixUp alpha=1.0 with Warmup Disable

- Change from alpha=0.5 (Beta(0.5, 0.5)) to alpha=1.0 (Beta(1.0, 1.0) = Uniform)
- Disable MixUp during first 5 warmup epochs (Tawara's approach)
- Also test label binarization at theta=0.8 (Tawara: labels >= 0.8 set to 1.0)
- Reference: `docs/references/kaggle/birdclef-2026/discussions/hgnetv2-b0-baseline-tawara.md`

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 training set, same splits as v4b |
| Compute | 1x RTX 6000, ~4h per ablation (~12h total) |
| Dependencies | Exp 017 v4b training infrastructure |
| Code | `user/murilo/experiments/017-mel-overhaul/` |

### Procedure

1. Train v5a: LogSumExp aggregation (T=0.1), all else identical to v4b
2. Train v5b: SpecAugment (freq + time masking), all else identical to v4b
3. Train v5c: MixUp alpha=1.0 + warmup disable, all else identical to v4b
4. Submit each to Kaggle LB
5. If multiple improve, combine best techniques into v5d

### Variables

- **Independent**: Frame aggregation method (v5a), augmentation (v5b), MixUp config (v5c)
- **Dependent**: Kaggle LB score (AUROC), validation loss
- **Controlled**: Architecture (EfficientNet-B0), mel config, dataset, all other hyperparameters

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Kaggle LB (AUROC) | Primary competition metric |
| Validation loss | Training stability indicator |

### Baseline

v4b = 0.859 LB.

### Success Criteria

- **Confirm if**: Any ablation achieves >= 0.864 LB (+0.005)
- **Reject if**: All ablations are <= 0.859 LB (no improvement)

## Limitations

- Single-variable ablations may not capture interaction effects
- LB variance (~0.003) may obscure small gains
- Gate condition: Only proceed if exp 018 Perch ensemble is working — these small SED gains compound with the ensemble
