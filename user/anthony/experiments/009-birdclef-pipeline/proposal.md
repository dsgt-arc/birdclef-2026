# Proposal: BirdCLEF End-to-End Pipeline

**Status**: confirmed (pipeline operational, performance improvements ongoing)
**Created**: 2026-02-13
**Author**: Anthony

## Hypothesis

An end-to-end BirdCLEF pipeline — self-supervised SSAMBA pretraining on competition soundscapes with 128x1 tokenization, followed by classifier fine-tuning and Kaggle submission — will produce a working baseline submission and establish wall-clock benchmarks for pretraining and inference on real competition data. This is infrastructure, not a performance experiment: the goal is a functioning pipeline, not a competitive score.

## Background

Experiments 004-008 validated SSAMBA on BirdSet HSN (21 classes, academic benchmark). Key learnings:
- Pre-training is the dominant factor (0.086 → 0.17 cmAP)
- SSAMBA plateaus at ~0.17 cmAP on HSN regardless of pooling, model size, or loss function
- The 128x1 tokenizer produces 1000 tokens per 10s clip, suitable for frame-level tasks
- On-the-fly nnAudio mel computation works with no throughput penalty

None of this has been tested on actual BirdCLEF data (hundreds of species, soundscape evaluation, CPU inference). This experiment bridges that gap by building the minimum viable pipeline from data download to Kaggle submission.

The SSAMBA paper uses joint discriminative + generative masked patch modeling (MSPM) for self-supervised pretraining:
- **MPC** (Masked Patch Classification): NCE contrastive loss — predict which patch belongs at each masked position
- **MPG** (Masked Patch Generation): MSE loss — reconstruct masked patch pixel values
- **Joint loss**: L = L_MPC + 10 × L_MPG, masking ~78% of patches

Pretraining on BirdCLEF soundscapes (instead of AudioSet) serves two purposes:
1. Adapts representations to bird audio domain
2. Benchmarks how long pretraining takes on competition-scale data (~thousands of 10-min soundscape recordings)

See: [SSAMBA concept](../../../docs/concepts/ssamba-audio-mamba.md), [BirdCLEF history](../../../docs/references/deep-research/birdclef-history.md)

## Method

### Approach

Three stages executed sequentially:

**Stage 1: Data** — Download BirdCLEF 2025 competition data from Kaggle. Catalog train recordings (focal, labeled by species) and test soundscapes (unlabeled, 10-min field recordings).

**Stage 2: Pretrain** — Self-supervised MSPM on the unlabeled soundscapes using SSAMBA with 128x1 patches over 10s chunks. This uses the existing vendor pretraining code adapted for the HF model port and 128x1 tokenization. No labels needed — the soundscapes are the pretraining corpus.

**Stage 3: Fine-tune + Submit** — Fine-tune a SED classifier head on the labeled training recordings (hundreds of species). The model produces per-token (10ms) species predictions via a SED head, which are then algorithmically aggregated into 5-second windows matching the Kaggle submission format. Inference processes 10s chunks (matching the model's native input) and aggregates per-token predictions within each 5s half (tokens 0-499, 500-999) via max-pool or thresholding. This reuses the SED architecture from exp 008 but for a structural purpose (temporal output format) rather than expecting it to improve clip-level accuracy. Package for Kaggle submission using Adrian's offline submission infrastructure.

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2025: ~200 species, train recordings (focal), test soundscapes (10-min each) |
| Compute | 1x RTX 6000 (24GB) for pretraining + fine-tuning. CPU for inference benchmarking. |
| Dependencies | ssamba_hf, mamba-ssm, nnAudio, pytorch-lightning, kaggle API |
| Code | `user/anthony/experiments/009-birdclef-pipeline/` |
| Pre-existing | Vendor SSAMBA pretraining code (`vendor/ssamba/src/`), HF port (`004-ssamba-hf-port/`), Kaggle submission template (`user/adrian/experiments/002-kaggle-submission/`) |

### Procedure

1. Download BirdCLEF 2025 data via Kaggle API to scratch storage
2. Explore data: count recordings, species, soundscape durations, file formats
3. Build data loader for soundscapes: 10s non-overlapping chunks, nnAudio mel → 128×1000 spectrograms
4. Adapt SSAMBA MSPM pretraining for 128x1 patches (mask ~780/1000 tokens, joint MPC+MPG loss)
5. Pretrain SSAMBA Tiny on soundscapes, logging wall-clock time per epoch
6. Build classifier: fine-tune pretrained SSAMBA on labeled train recordings (multi-class, hundreds of species)
7. Implement inference: process 10s chunks, aggregate per-token SED predictions within each 5s half (tokens 0-499 and 500-999) to produce 5s submission slots
8. Format predictions as submission CSV (`row_id`, species columns)
9. Package for Kaggle using Adrian's offline wheel + submission template
10. Submit to Kaggle (even if score is low — the pipeline is the deliverable)

### Variables

- **Independent**: Pipeline exists vs doesn't exist (binary — this is infrastructure)
- **Dependent**: Wall-clock time for pretraining (per epoch, total), fine-tuning time, inference throughput (soundscapes/minute), Kaggle submission score
- **Controlled**: SSAMBA Tiny architecture, 128x1 patches, 10s chunks, nnAudio mel frontend

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Submission success | Binary: does the Kaggle submission complete without error? |
| Pretrain wall-clock | Hours to pretrain on soundscapes (total and per epoch) |
| Fine-tune wall-clock | Hours to fine-tune classifier |
| Inference throughput | Soundscapes processed per minute (GPU and CPU) |
| Kaggle score | Competition metric (row-wise micro-averaged AUROC or similar) |

### Baseline

No existing BirdCLEF submission from this project. The baseline is "no pipeline exists."

For context, BirdCLEF 2025 winning solutions scored ~0.75+ on the leaderboard. We expect a much lower score from this minimal SSAMBA baseline — the goal is pipeline completion, not competitive performance.

### Success Criteria

- **Confirm if**: A valid submission.csv is produced and accepted by Kaggle, AND wall-clock benchmarks are documented for pretraining, fine-tuning, and inference
- **Reject if**: Unable to produce a valid Kaggle submission due to pipeline failures

## Limitations

- **BirdCLEF 2025, not 2026**: Using 2025 data since 2026 may not be released yet. Pipeline should transfer with minimal changes when 2026 opens.
- **SSAMBA is not competitive**: We know SSAMBA peaks at ~0.17 cmAP on HSN. The Kaggle score will be low. This is intentional — future experiments (ConvNeXt backbone, pseudo-labeling, ensembling) build on this pipeline.
- **CPU inference constraint**: Kaggle evaluates on CPU. SSAMBA with mamba-ssm CUDA kernels won't work — need pure PyTorch or ONNX export. This is a known challenge to solve in this experiment.
- **No pseudo-labeling or SSL refinement**: Explicitly out of scope. Future experiments handle this.
- **Pretraining scale**: BirdCLEF soundscapes are much smaller than AudioSet (thousands vs millions of clips). Pretraining may underfit. The benchmark is still valuable regardless.
