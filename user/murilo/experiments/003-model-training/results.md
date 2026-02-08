# Results: Model Training (ConvNeXT-Tiny & ViT-Small)

**Date Completed**: 2026-02-08
**Author**: Murilo

## Summary

Pretrained variants substantially outperform scratch-trained variants for both architectures, confirming the pretraining hypothesis. ConvNeXT-Tiny pretrained (cmAP=0.180) is the best performer, surpassing ViT-Small pretrained (cmAP=0.121) and even exp 002's larger ViT-Base pretrained (cmAP=0.164) with ~3x fewer parameters.

**Verdict**: confirmed

## Observations

### Quantitative

#### Test Set Results (HSN `test_5s`, 12,000 samples)

| Run | Params | cmAP | AUROC | F1 Macro | Top-1 Acc | Top-5 Acc | Loss |
|-----|--------|------|-------|----------|-----------|-----------|------|
| ConvNeXT-Tiny pretrained | 27.8M | **0.1800** | **0.6473** | **0.0992** | **0.2137** | **0.5505** | **0.1212** |
| ViT-Small pretrained | 21.5M | 0.1208 | 0.6448 | 0.0471 | 0.0688 | 0.3777 | 0.1677 |
| ViT-Small scratch | 21.5M | 0.0732 | 0.5087 | 0.0114 | 0.0019 | 0.0485 | 0.1467 |
| ConvNeXT-Tiny scratch | 27.8M | 0.0599 | 0.5422 | 0.0247 | 0.0015 | 0.0365 | 0.1619 |

#### Comparison to Baselines

| Run | cmAP | Delta vs Best (Exp 001) |
|-----|------|-------------------------|
| Exp 001 - ConvNeXT (bioacoustics_model_zoo) | 0.480 | -- |
| Exp 002 - ViT-Base pretrained (85.5M) | 0.164 | -0.316 |
| **Exp 003 - ConvNeXT-Tiny pretrained (27.8M)** | **0.180** | **-0.300** |
| Exp 003 - ViT-Small pretrained (21.5M) | 0.121 | -0.359 |
| Exp 003 - ViT-Small scratch (21.5M) | 0.073 | -0.407 |
| Exp 003 - ConvNeXT-Tiny scratch (27.8M) | 0.060 | -0.420 |

#### Pretraining Advantage

| Architecture | Scratch cmAP | Pretrained cmAP | Pretrain Lift |
|-------------|-------------|-----------------|---------------|
| ConvNeXT-Tiny | 0.060 | 0.180 | **3.0x** |
| ViT-Small | 0.073 | 0.121 | **1.7x** |

#### Training Dynamics

| Run | Best Epoch | Early Stop Epoch | Best Val cmAP | Train cmAP (final) | Walltime |
|-----|-----------|-----------------|---------------|---------------------|----------|
| ConvNeXT-Tiny pretrained | 7 | 17 (patience=10) | 0.180 | 1.000 | 39m 43s |
| ViT-Small pretrained | 5 | 15 (patience=10) | 0.121 | 0.999 | 36m 06s |
| ViT-Small scratch | 8 | 23 (patience=15) | 0.073 | 0.644 | 37m 47s |
| ConvNeXT-Tiny scratch | 15 | 30 (patience=15) | 0.060 | 0.922 | 45m 26s |

### Qualitative

- All 4 runs completed successfully on a single RTX 6000 GPU with mel caching enabled
- Pretrained models converge much faster (best epoch 5-7) vs scratch (best epoch 8-15)
- All models show significant overfitting: train cmAP approaches 1.0 while test cmAP remains low, suggesting the HSN training set (5,460 samples) is too small relative to model capacity
- ConvNeXT-Tiny pretrained beats the much larger ViT-Base pretrained (exp 002, 85.5M params) by +0.016 cmAP, showing architecture matters more than raw parameter count
- Scratch models have near-zero top-1 accuracy (<0.2%), indicating they barely learn meaningful class discrimination from random initialization on this small dataset

## Analysis

The results clearly confirm the hypothesis: ImageNet pretraining provides a substantial advantage for both architectures on HSN.

**ConvNeXT vs ViT**: ConvNeXT-Tiny outperforms ViT-Small in the pretrained setting (0.180 vs 0.121) but slightly underperforms in the scratch setting (0.060 vs 0.073). This suggests ConvNeXT's inductive biases (locality, translation equivariance) are better at leveraging pretrained features for audio, while ViT's flexibility allows marginally better learning from scratch — though both scratch results are poor.

**Pretraining lift is architecture-dependent**: ConvNeXT sees a 3.0x lift from pretraining while ViT sees only 1.7x. ConvNeXT may transfer ImageNet features more effectively to mel spectrograms because its hierarchical convolutional structure naturally maps to frequency-time patterns.

**All models far below exp 001**: The best result (0.180) is still only 37.5% of exp 001's ConvNeXT from `bioacoustics_model_zoo` (0.480). That model was pretrained on audio-specific data, suggesting domain-specific pretraining is far more valuable than ImageNet pretraining for bird call classification.

**Severe overfitting across the board**: With only 5,460 training samples and 21.5-27.8M parameters, all models overfit heavily. The gap between train cmAP (~1.0) and test cmAP (0.06-0.18) is extreme. Data augmentation, regularization, or larger training sets would likely help.

### Confounders

- Scratch and pretrained runs used different hyperparameters by design (scratch: lr=5e-4, wd=0.05, warmup=10, patience=15; pretrained: lr=1e-4, wd=0.01, warmup=5, patience=10). These were chosen to be reasonable defaults for each setting, but the comparison is not perfectly controlled for hyperparameters alone.
- Only a single run per configuration (no repeated seeds), so results may include seed variance.
- ConvNeXT-Tiny (27.8M) has ~30% more parameters than ViT-Small (21.5M), which slightly favors ConvNeXT in the comparison.

## Artifacts

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/convnext_tiny_scratch/HSN/metrics.json` |
| Metrics | `artifacts/training/convnext_tiny_pretrained/HSN/metrics.json` |
| Metrics | `artifacts/training/vit_small_scratch/HSN/metrics.json` |
| Metrics | `artifacts/training/vit_small_pretrained/HSN/metrics.json` |
| SLURM logs | `logs/slurm_convnext_tiny_scratch_3811769.log` |
| SLURM logs | `logs/slurm_convnext_tiny_pretrained_3811771.log` |
| SLURM logs | `logs/slurm_vit_small_scratch_3811772.log` |
| SLURM logs | `logs/slurm_vit_small_pretrained_3811773.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/003-model-training/checkpoints/` |
| W&B | [convnext-tiny-scratch](https://wandb.ai/dsgt-arc/birdset-model-training/runs/q4ds47n6) |
| W&B | [convnext-tiny-imagenet](https://wandb.ai/dsgt-arc/birdset-model-training/runs/42iwob2s) |
| W&B | [vit-small-scratch](https://wandb.ai/dsgt-arc/birdset-model-training/runs/qulhf926) |
| W&B | [vit-small-imagenet](https://wandb.ai/dsgt-arc/birdset-model-training/runs/jovw63uq) |
| Code | Branch `mg/model-training` |

## Next Steps

- [ ] Address overfitting: add data augmentation (mixup, time/freq masking, noise injection)
- [ ] Train on larger BirdSet datasets (XCM) to evaluate scaling behavior
- [ ] Explore audio-specific pretrained backbones (BEATs, AST, PaSST) to close the gap with exp 001
- [ ] Run multiple seeds to quantify variance in these results
- [ ] Consider unfreezing pretrained weights progressively (gradual unfreezing) to improve fine-tuning

**Leads to**: Investigating audio-domain pretraining (BEATs/AST) as the likely key to reaching exp 001 performance levels; data augmentation strategies for small-data bird classification.
