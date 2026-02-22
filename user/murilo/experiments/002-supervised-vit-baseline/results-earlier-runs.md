# Earlier Run (superseded — center-crop data pipeline)

> **WARNING**: The run below used the original `BirdSetDataModule` which loads only a single center-crop 5s segment per recording (5,460 training samples). This was superseded by a run using the `StreamingBirdSetDataModule` which splits full recordings into non-overlapping 5s chunks (~66,504 training chunks, ~12x more data per epoch). See [results.md](results.md) for the canonical streaming pipeline results.

## Test Set Results (HSN `test_5s`, 12,000 samples)

| Metric | ViT-Base (Ours) | ConvNeXT (Baseline) | % of Baseline |
|--------|-----------------|---------------------|---------------|
| **cmAP** | **0.164** | 0.480 | 34% |
| AUROC | 0.578 | 0.883 | 65% |
| F1 Macro | 0.086 | 0.139 | 62% |
| F1 Micro | 0.095 | 0.121 | 79% |
| Top-1 Acc | 0.112 | 0.251 | 45% |
| Top-5 Acc | 0.547 | 0.330 | 166% |

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | ViT-Base-Patch16 (pretrained ImageNet-21k) |
| Parameters | 85.5M |
| Epochs trained | 14 (early stopped at epoch 3) |
| Best epoch | 3 |
| Training time | 37 min 16 sec |
| GPU | Quadro RTX 6000 (24GB) |

## Artifacts

| Type | Location |
|------|----------|
| Best checkpoint | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/002-supervised-vit-baseline/checkpoints/best-epochepoch=03.ckpt` |
| W&B run | [exp002-HSN](https://wandb.ai/dsgt-arc/birdset-vit-baseline/runs/dvoxy50z) |
| SLURM log | `logs/slurm_3692914.log` |
