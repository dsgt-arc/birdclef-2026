# Proposal: SSAMBA Token Context Shaping

**Status**: completed
**Created**: 2026-02-06
**Author**: Anthony

## Hypothesis

Rectangular patches (128x2, 128x4, 128x1, 64x2) at 50-100 tokens/second enable practical 30-60s SSAMBA inference on a single GPU (24GB RTX 6000), with Mamba's O(n) complexity confirmed by linear throughput scaling from 30s to 60s inputs.

## Background

Experiment 004 ported SSAMBA to pure PyTorch with optional CUDA kernel support. The `docs/concepts/temporal-resolution-sequence-length.md` document establishes that tall-and-skinny patches (128x2, 128x1) provide 8-16x finer temporal resolution than square 16x16 patches at the same token budget. For bioacoustics, this is desirable because bird calls have narrow frequency extent but rapid temporal modulation (FM sweeps, trills). Mamba's O(n) complexity means we can process 30-60s of audio as a single sequence (1,500-6,000 tokens) without chunking.

This experiment answers: can we actually instantiate and run SSAMBA with these rectangular patch shapes at long durations, and what are the parameter counts, memory costs, and throughput?

## Method

### Approach

Extend the ssamba_hf library to support rectangular (non-square) patch sizes, then benchmark multiple patch configurations at 30s and 60s durations across two model sizes (tiny=192d, small=384d). No training -- synthetic random data only.

### Setup

| Component | Details |
|-----------|---------|
| Data | Synthetic `torch.randn` spectrograms (128 mel bins, variable time) |
| Compute | 1x RTX 6000 (24GB), 60 min |
| Dependencies | ssamba_hf (exp 004), mamba-ssm + causal-conv1d (CUDA kernels) |
| Code | `user/anthony/experiments/004-ssamba-hf-port/ssamba_hf/` (modified), `005-ssamba-token-context/scripts/` |

### Procedure

1. Extend `ssamba_hf` config and layers to support `patch_size` as `Union[int, Tuple[int,int]]`
2. Write benchmarking script with 5 patch configurations x 2 model sizes x 2 durations
3. Run on GPU node with CUDA Mamba kernels (pure PyTorch SSM loop is too slow for 6,000 tokens)
4. Collect parameter counts, peak GPU memory, and forward pass timing

### Variables

- **Independent**: Patch shape (5 configs), model size (tiny/small), audio duration (30s/60s)
- **Dependent**: Parameter count, peak GPU memory (bytes), forward pass time (ms), tokens/sec throughput
- **Controlled**: Mel bins (128), frame rate (100 fps), batch size (1), depth (24 layers)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Parameters | Total model parameter count per configuration |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` during forward pass |
| Forward time (ms) | Mean of 10 forward passes after 3 warmup |
| Tokens/sec | `num_tokens / forward_time` |
| 60s/30s time ratio | Should be ~2x if O(n) holds (not 4x as O(n^2) would give) |

### Baseline

SSAMBA small with 16x16 patches at 30s (~1,500 tokens). This is the standard configuration from experiment 004.

### Success Criteria

- **Confirm if**: All 5 patch configs instantiate and run forward passes at 30s without OOM on 24GB; 60s/30s time ratio is <2.5x (confirming near-linear scaling)
- **Reject if**: Rectangular patches cause >2x memory overhead vs equivalent token count with square patches, or 60s configs OOM on 24GB

## Limitations

- No training or accuracy evaluation -- this is purely a feasibility and resource profiling study
- CUDA kernels required for practical throughput; pure PyTorch SSM has O(L) Python loop that makes 6,000-token sequences impractical
- Batch size fixed at 1; real training would use larger batches with potentially different memory characteristics
- `adapt_for_finetuning()` not yet updated for rectangular patches (not needed for this experiment)
