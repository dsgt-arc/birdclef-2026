# Earlier Runs (superseded — center-crop data pipeline)

> **WARNING**: The runs below used the original `BirdSetDataModule` which loads only a single center-crop 5s segment per recording (5,460 training samples). These were superseded by runs using the `StreamingBirdSetDataModule` which splits full recordings into non-overlapping 5s chunks (~66,504 training chunks, ~12x more data per epoch). See [results.md](results.md) for the canonical streaming pipeline results.

## Test Set Results (HSN `test_5s`, 12,000 samples)

| Run | Params | cmAP | AUROC | F1 Macro | Top-1 Acc | Top-5 Acc | Loss |
|-----|--------|------|-------|----------|-----------|-----------|------|
| ConvNeXT-Tiny pretrained | 27.8M | 0.1800 | 0.6473 | 0.0992 | 0.2137 | 0.5505 | 0.1212 |
| ViT-Small pretrained | 21.5M | 0.1208 | 0.6448 | 0.0471 | 0.0688 | 0.3777 | 0.1677 |
| ViT-Small scratch | 21.5M | 0.0732 | 0.5087 | 0.0114 | 0.0019 | 0.0485 | 0.1467 |
| ConvNeXT-Tiny scratch | 27.8M | 0.0599 | 0.5422 | 0.0247 | 0.0015 | 0.0365 | 0.1619 |

## Training Dynamics

| Run | Best Epoch | Stopped | Best Val cmAP | Walltime |
|-----|-----------|---------|---------------|----------|
| ConvNeXT-Tiny pretrained | 7 | 17 | 0.180 | 39m 43s |
| ViT-Small pretrained | 5 | 15 | 0.121 | 36m 06s |
| ViT-Small scratch | 8 | 23 | 0.073 | 37m 47s |
| ConvNeXT-Tiny scratch | 15 | 30 | 0.060 | 45m 26s |

## Artifacts

| Type | Location |
|------|----------|
| SLURM logs | `logs/slurm_convnext_tiny_scratch_3811769.log` |
| SLURM logs | `logs/slurm_convnext_tiny_pretrained_3811771.log` |
| SLURM logs | `logs/slurm_vit_small_scratch_3811772.log` |
| SLURM logs | `logs/slurm_vit_small_pretrained_3811773.log` |
| W&B | [convnext-tiny-scratch](https://wandb.ai/dsgt-arc/birdset-model-training/runs/q4ds47n6) |
| W&B | [convnext-tiny-imagenet](https://wandb.ai/dsgt-arc/birdset-model-training/runs/42iwob2s) |
| W&B | [vit-small-scratch](https://wandb.ai/dsgt-arc/birdset-model-training/runs/qulhf926) |
| W&B | [vit-small-imagenet](https://wandb.ai/dsgt-arc/birdset-model-training/runs/jovw63uq) |
