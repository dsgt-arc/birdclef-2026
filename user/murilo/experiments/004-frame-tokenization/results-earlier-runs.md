# Earlier Runs (superseded — flawed data pipeline)

> **WARNING**: The nnAudio and torchaudio runs below used a flawed data pipeline that shuffled random segments from different audio files rather than properly chunking full recordings into non-overlapping 5s windows. These results should NOT be used for the tokenization comparison. See [results.md](results.md) for the canonical streaming pipeline results.

## Torchaudio Rerun — Test Set Results (HSN `test_5s`, 12,000 samples)

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.066 | 0.533 | 0.005 | 0.028 | 0.161 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.086 | 0.648 | 0.014 | 0.137 | 0.153 | 21.54M |
| B: 128×4 | 128×4 | 125 | 0.087 | 0.579 | 0.051 | 0.438 | 0.134 | 21.55M |
| A: 128×2 | 128×2 | 250 | 0.098 | 0.621 | 0.012 | 0.304 | 0.145 | 21.50M |

### Torchaudio Rerun — Training Dynamics

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 8 | 24 | 0.066 | 5.13 it/s | 29m 16s |
| A: 128×2 | 6 | 22 | 0.098 | 5.12 it/s | 28m 43s |
| B: 128×4 | 8 | 24 | 0.087 | 8.01 it/s | 27m 19s |
| C: 128×1 | 6 | 22 | 0.086 | 2.59 it/s | 34m 38s |

## nnAudio Run — Test Set Results

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.029 | 0.458 | 0.001 | 0.544 | 0.179 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.051 | 0.536 | 0.483 | 0.566 | 0.122 | 21.54M |
| A: 128×2 | 128×2 | 250 | 0.054 | 0.536 | 0.093 | 0.573 | 0.123 | 21.50M |
| B: 128×4 | 128×4 | 125 | 0.063 | 0.601 | 0.440 | 0.547 | 0.118 | 21.55M |

### nnAudio Run — Training Dynamics

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 7 | 17 | 0.029 | 4.16 it/s | 38m 27s |
| A: 128×2 | 2 | 18 | 0.054 | 4.15 it/s | 38m 28s |
| B: 128×4 | 2 | 17 | 0.063 | 5.95 it/s | 37m 23s |
| C: 128×1 | 2 | 17 | 0.051 | 2.34 it/s | 43m 01s |

## nnAudio vs Torchaudio Comparison

| Config | nnAudio cmAP | Torchaudio cmAP | Lift |
|--------|-------------|-----------------|------|
| Baseline (16×16) | 0.029 | 0.066 | +128% (2.28×) |
| A: 128×2 | 0.054 | 0.098 | +81% (1.81×) |
| B: 128×4 | 0.063 | 0.087 | +39% (1.39×) |
| C: 128×1 | 0.051 | 0.086 | +69% (1.69×) |

The nnAudio mel backend substantially underperformed torchaudio due to a mel filterbank configuration mismatch (`htk=False, norm=1` defaults vs torchaudio's `htk=True, norm=None`). The streaming pipeline resolved this with an AMP precision fix.

## Cross-Experiment Baseline Comparison

| Run | Mel Backend | GPU | cmAP |
|-----|------------|-----|------|
| Exp 003 — ViT-Small scratch | CPU (torchaudio) | RTX 6000 | 0.073 |
| Exp 004 — Baseline 16×16 (nnAudio) | GPU (nnAudio) | V100 | 0.029 |
| Exp 004 — Baseline 16×16 (torchaudio rerun) | CPU (torchaudio) | V100 | 0.066 |

## Artifacts

### Torchaudio Rerun

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/baseline_16x16_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x2_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x4_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x1_torchaudio/HSN/metrics.json` |
| SLURM logs | `logs/slurm_baseline_16x16_torchaudio_4017565.log` |
| SLURM logs | `logs/slurm_vertical_128x2_torchaudio_4017567.log` |
| SLURM logs | `logs/slurm_vertical_128x4_torchaudio_4017569.log` |
| SLURM logs | `logs/slurm_vertical_128x1_torchaudio_4017568.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/004-frame-tokenization/checkpoints/*-torchaudio/` |

### nnAudio Run

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/baseline_16x16/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x2/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x4/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x1/HSN/metrics.json` |
| GPU Utilization | `artifacts/gpu-utilization.png` |
| SLURM logs | `logs/slurm_baseline_16x16_4011400.log` |
| SLURM logs | `logs/slurm_vertical_128x2_4011401.log` |
| SLURM logs | `logs/slurm_vertical_128x4_4011402.log` |
| SLURM logs | `logs/slurm_vertical_128x1_4011403.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/004-frame-tokenization/checkpoints/` |
| W&B | [baseline-16x16](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/pwal) |
| W&B | [vertical-128x2](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/kqek) |
| W&B | [vertical-128x4](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/1ybfne3c) |
| W&B | [vertical-128x1](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/jan07qgv) |
