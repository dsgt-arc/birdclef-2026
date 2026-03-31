# Experiment 103: SSAMBA SSL Pretraining

## What this is

Self-supervised pretraining of SSAMBA-small (26M params) on BirdCLEF 2026 audio at 32kHz
using masked patch prediction. Goal: domain-specific representations that beat AudioSet
pretrained weights (which cap at 8kHz, missing upper bird call frequencies).

## Architecture

- **Backbone**: SSAMBA-small (384d, 24 BiMamba layers, ~26M params)
- **Patches**: 224×1 full-band mel frames (10ms/token, 3000 tokens per 30s window)
- **Mel**: n_fft=4096, hop=320, n_mels=224, fmin=50, fmax=16kHz, AmplitudeToDB normalization
- **Loss**: dual NCE + MSE (`L = L_nce + 10 × L_mse`)
- **NCE**: cross-sample shared negative pool (K=100, cosine sim, temperature=0.1)
- **Masking**: 40% span masking (5-15 token spans = 50-150ms)

## Key files

| File | Purpose |
|------|---------|
| `src/birdclef_103/models/pretrain.py` | Core model: mel, masking, backbone, NCE+MSE heads |
| `src/birdclef_103/training/pretrain.py` | Lightning module: loss weighting, LR schedule, logging |
| `src/birdclef_103/cli/pretrain.py` | CLI: config loading, trainer setup, checkpoint resume |
| `src/birdclef_103/data/pretrain.py` | Dataset: soundscape+focal scanning, 30s chunks, val split |
| `src/birdclef_103/data/audio.py` | Audio I/O: load_audio_chunk, collate_pretrain |
| `configs/pretrain-103a.yaml` | Training config for run 103-A |
| `design.md` | Full architecture doc, decisions, ablation plan |

## Running

```bash
# Dev sanity check (fast_dev_run + overfit 4 batches)
sbatch sbatch/pretrain-dev.sbatch

# Full training (15 epochs, 20h limit, checkpoints every epoch)
sbatch sbatch/pretrain.sbatch

# Resume from checkpoint
sbatch sbatch/pretrain.sbatch configs/pretrain-103a.yaml --ckpt-path last
```

## Known issues and fixes

- **Short focal recordings** (<2s, ~370 files): filtered out in dataset — too few tokens for masking
- **Double norm_f**: backbone's RMSNorm was applied twice, causing NaN in fp16. Fixed to single application.
- **NCE within-sample**: fails with 224×1 patches (adjacent frames identical). Fixed: cross-sample negatives.
- **Per-prediction sampling**: 860MB gather, 3x slower. Fixed: shared negative pool (90KB, single matmul).
- **PCEN**: unbounded output breaks NCE. Deferred to ablation 103-C; using AmplitudeToDB for 103-A.

## Data

- BirdCLEF 2026 soundscapes: 10,129 train files × 2 halves = 20,258 chunks
- BirdCLEF 2026 focal: ~35,179 chunks (after filtering <2s)
- Val: 5% soundscapes by file hash = 1,058 chunks
- Total train: ~55,437 chunks

## Monitoring

- TensorBoard: `logs/pretrain/` — watch `train/loss_mpg_epoch` (MSE primary signal)
- NCE accuracy should climb from ~1% (chance) toward higher values
- MSE should decrease steadily; NCE loss may plateau (expected)
- Checkpoints: `logs/pretrain/checkpoints/` (top-3 by val/loss + last)

## Ablation plan (gated sequentially on 103-A results)

| Run | Change |
|-----|--------|
| 103-B | MSE-only (drop NCE) |
| 103-C | PCEN normalization |
| 103-D | MixIT 4-ch input |
| 103-E | Multi-stream K=4 |
| 103-F | SSAMBA-base (99M) |
| 103-G | BirdCLEF 2025 data |
