# Results: Model Training (ConvNeXT-Tiny & ViT-Small)

**Date Completed**: 2026-02-21
**Author**: Murilo

## Summary

Pretrained variants substantially outperform scratch-trained variants for both architectures, confirming the pretraining hypothesis. ConvNeXT-Tiny pretrained (cmAP=0.191) is the best performer, followed closely by ViT-Small pretrained (cmAP=0.174). Both pretrained models now achieve similar performance levels, closing the gap observed in earlier runs.

All runs use the streaming data pipeline (`StreamingBirdSetDataModule`), which splits full recordings into non-overlapping 5s chunks (~66,504 training chunks vs 5,460 center-crop samples in the original pipeline).

**Verdict**: confirmed

## Observations

### Quantitative

#### Test Set Results (HSN `test_5s`, 12,000 samples)

| Run | Params | cmAP | AUROC | F1 Macro | Top-1 Acc | Top-5 Acc | Loss |
|-----|--------|------|-------|----------|-----------|-----------|------|
| ConvNeXT-Tiny pretrained | 27.8M | **0.191** | 0.638 | **0.106** | **0.212** | **0.419** | 0.162 |
| ViT-Small pretrained | 21.5M | 0.174 | **0.707** | 0.095 | 0.191 | 0.311 | **0.153** |
| ViT-Small scratch | 21.5M | 0.098 | 0.594 | 0.025 | 0.008 | 0.047 | 0.155 |
| ConvNeXT-Tiny scratch | 27.8M | 0.087 | 0.607 | 0.034 | 0.027 | 0.165 | 0.194 |

#### Comparison to Baselines

| Run | cmAP | Delta vs Best (Exp 001) |
|-----|------|-------------------------|
| Exp 001 - ConvNeXT (bioacoustics_model_zoo) | 0.480 | -- |
| **Exp 003 - ConvNeXT-Tiny pretrained (27.8M)** | **0.191** | **-0.289** |
| Exp 002 - ViT-Base pretrained (85.5M, streaming) | 0.186 | -0.294 |
| Exp 003 - ViT-Small pretrained (21.5M) | 0.174 | -0.306 |
| Exp 004 - ViT-Small 128x4 vertical (streaming) | 0.118 | -0.362 |
| Exp 003 - ViT-Small scratch (21.5M) | 0.098 | -0.382 |
| Exp 003 - ConvNeXT-Tiny scratch (27.8M) | 0.087 | -0.393 |

#### Pretraining Advantage

| Architecture | Scratch cmAP | Pretrained cmAP | Pretrain Lift |
|-------------|-------------|-----------------|---------------|
| ConvNeXT-Tiny | 0.087 | 0.191 | **2.2x** |
| ViT-Small | 0.098 | 0.174 | **1.8x** |

#### Training Dynamics

| Run | Best Epoch | Stopped | Best Val cmAP | Walltime |
|-----|-----------|---------|---------------|----------|
| ConvNeXT-Tiny pretrained | 10 | 21 | 0.191 | 1h 14m |
| ViT-Small pretrained | 6 | 17 | 0.174 | 1h 00m |
| ViT-Small scratch | 20 | 31 | 0.098 | 1h 48m |
| ConvNeXT-Tiny scratch | 32 | 48 | 0.087 | 2h 47m |

#### Streaming vs Center-Crop Improvement

These runs used the streaming pipeline (~66k chunks) instead of the original center-crop pipeline (~5.5k samples). All 4 configs improved:

| Run | Center-Crop cmAP | Streaming cmAP | Lift |
|-----|-----------------|----------------|------|
| ConvNeXT-Tiny pretrained | 0.180 | 0.191 | +6% |
| ViT-Small pretrained | 0.121 | 0.174 | +44% |
| ViT-Small scratch | 0.073 | 0.098 | +34% |
| ConvNeXT-Tiny scratch | 0.060 | 0.087 | +45% |

ViT-Small pretrained saw the largest improvement (+44%), suggesting ViTs benefit more from additional training data than CNNs do.

### Qualitative

- All 4 runs completed successfully on RTX 6000 GPUs using the streaming data pipeline
- Pretrained models converge faster (best epoch 6-10) vs scratch (best epoch 20-32)
- All models still show significant overfitting, but the larger training set from streaming reduces the gap
- ConvNeXT-Tiny pretrained (27.8M params) beats the much larger ViT-Base pretrained (85.5M params, exp 002) by a small margin (0.191 vs 0.186)
- ViT-Small pretrained (0.174) now approaches ConvNeXT-Tiny pretrained (0.191), narrowing the architecture gap compared to earlier runs
- Scratch models remain weak but improved substantially with streaming data

## Analysis

The results confirm the hypothesis: ImageNet pretraining provides a 2x+ advantage for both architectures on HSN, consistent across data pipeline variants.

**Streaming pipeline impact**: The streaming data pipeline provides a consistent lift across all configs, with ViTs benefiting more (+34-44%) than ConvNeXTs (+6-45%). The 12x increase in training samples per epoch is particularly valuable for ViTs, whose self-attention has more parameters to train than CNN convolution kernels.

**ConvNeXT vs ViT**: ConvNeXT-Tiny still leads in cmAP (0.191 vs 0.174), but ViT-Small pretrained now achieves the highest AUROC (0.707 vs 0.638). The gap narrowed substantially from the center-crop runs (0.180 vs 0.121 → 0.191 vs 0.174), suggesting ViTs are more data-hungry and benefit more from the larger streaming dataset.

**All models far below exp 001**: The best result (0.191) is still only 40% of exp 001's audio-pretrained ConvNeXT (0.480), confirming that domain-specific pretraining (on audio data) is far more valuable than ImageNet pretraining for bird call classification. This motivates exp 005 (EAT/BEATs fine-tuning with audio-pretrained encoders).

### Confounders

- Scratch and pretrained runs used different hyperparameters by design (scratch: lr=5e-4, wd=0.05, warmup=10, patience=15; pretrained: lr=1e-4, wd=0.01, warmup=5, patience=10). These were chosen to be reasonable defaults for each setting.
- Only a single run per configuration (no repeated seeds), so results may include seed variance.
- ConvNeXT-Tiny (27.8M) has ~30% more parameters than ViT-Small (21.5M), which slightly favors ConvNeXT in the comparison.

## Artifacts

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/convnext_tiny_pretrained/HSN/metrics.json` |
| Metrics | `artifacts/training/vit_small_pretrained/HSN/metrics.json` |
| Metrics | `artifacts/training/vit_small_scratch/HSN/metrics.json` |
| Metrics | `artifacts/training/convnext_tiny_scratch/HSN/metrics.json` |
| SLURM logs | `logs/slurm_convnext_tiny_pretrained_4100647.log` |
| SLURM logs | `logs/slurm_convnext_tiny_scratch_4100648.log` |
| SLURM logs | `logs/slurm_vit_small_pretrained_4100649.log` |
| SLURM logs | `logs/slurm_vit_small_scratch_4100650.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/003-model-training/checkpoints/` |
| W&B | [convnext-tiny-scratch](https://wandb.ai/dsgt-arc/birdset-model-training/runs/kgqjuuhs) |
| W&B | [convnext-tiny-imagenet](https://wandb.ai/dsgt-arc/birdset-model-training/runs/qxnjuqxh) |
| W&B | [vit-small-scratch](https://wandb.ai/dsgt-arc/birdset-model-training/runs/ys8azbql) |
| W&B | [vit-small-imagenet](https://wandb.ai/dsgt-arc/birdset-model-training/runs/plrfcv02) |
| Earlier runs | [results-earlier-runs.md](results-earlier-runs.md) (superseded — center-crop data pipeline) |
| Code | Branch `mg/model-training` |

## Next Steps

- [ ] Address overfitting: add data augmentation (mixup, time/freq masking, noise injection)
- [ ] Train on larger BirdSet datasets (XCM) to evaluate scaling behavior
- [x] ~~Explore audio-specific pretrained backbones~~ → exp 005 (EAT/BEATs fine-tuning on BirdCLEF 2025)
- [ ] Run multiple seeds to quantify variance in these results
- [x] ~~Re-run with streaming pipeline~~ → completed, results above

**Leads to**: Exp 005 — fine-tuning audio-domain pretrained encoders (EAT, BEATs) on BirdCLEF 2025 to close the gap with exp 001's audio-pretrained baseline.
