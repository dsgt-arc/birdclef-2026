# Results: SSAMBA Token Context Shaping

**Date Completed**: 2026-02-06
**Author**: Anthony

## Summary

Rectangular patch shapes work and scale to 192k tokens on a single 24GB GPU without OOM. Mamba's O(n) scaling holds from ~3k-24k tokens but becomes superlinear above ~48k due to memory bandwidth limits. The practical sweet spot is **6k-12k tokens** (~142k tok/s throughput for tiny, ~65k tok/s for small). The baseline 16x16 square patch is anomalously 2.3x slower than rectangular patches at equivalent token counts.

**Verdict**: confirmed (with caveats on superlinear regime above 48k tokens)

## Observations

### Quantitative

All 11 patch configurations ran successfully at both 30s and 60s durations for both tiny (192d) and small (384d) models. Zero OOMs on the 24GB RTX 6000. 44/44 benchmarks completed.

#### Tiny model (192d, 24 layers) at 30s

| Config | Patch (f x t) | Tokens | Params | GPU Mem | ms/fwd | tok/s |
|--------|--------------|--------|--------|---------|--------|-------|
| coarse-128x24 | 128 x 24 | 125 | 7.4M | 41 MB | 28.3 | 4,412 |
| medium-128x6 | 128 x 6 | 500 | 7.0M | 48 MB | 28.2 | 17,741 |
| wideframe-128x4 | 128 x 4 | 750 | 7.0M | 54 MB | 28.9 | 25,984 |
| baseline-16x16 | 16 x 16 | 1,496 | 7.1M | 72 MB | 65.4 | 22,881 |
| fullband-128x2 | 128 x 2 | 1,500 | 7.1M | 72 MB | 28.3 | 52,964 |
| quarterband-32x4 | 32 x 4 | 3,000 | 7.4M | 109 MB | 28.6 | 104,839 |
| framelevel-128x1 | 128 x 1 | 3,000 | 7.4M | 109 MB | 29.2 | 102,607 |
| halfframe-64x1 | 64 x 1 | 6,000 | 7.9M | 179 MB | 43.1 | 139,150 |
| quarterframe-32x1 | 32 x 1 | 12,000 | 9.1M | 327 MB | 84.4 | 142,227 |
| eighthframe-8x1 | 8 x 1 | 48,000 | 16.0M | 1,178 MB | 490.5 | 97,858 |
| fine-4x1 | 4 x 1 | 96,000 | 25.2M | 2,317 MB | 2,010 | 47,751 |

#### Small model (384d, 24 layers) at 30s

| Config | Patch (f x t) | Tokens | Params | GPU Mem | ms/fwd | tok/s |
|--------|--------------|--------|--------|---------|--------|-------|
| coarse-128x24 | 128 x 24 | 125 | 26.3M | 119 MB | 28.4 | 4,397 |
| medium-128x6 | 128 x 6 | 500 | 25.5M | 134 MB | 28.3 | 17,701 |
| wideframe-128x4 | 128 x 4 | 750 | 25.5M | 146 MB | 28.9 | 25,930 |
| baseline-16x16 | 16 x 16 | 1,496 | 25.7M | 180 MB | 65.8 | 22,749 |
| fullband-128x2 | 128 x 2 | 1,500 | 25.7M | 180 MB | 28.3 | 52,938 |
| quarterband-32x4 | 32 x 4 | 3,000 | 26.3M | 251 MB | 51.6 | 58,139 |
| framelevel-128x1 | 128 x 1 | 3,000 | 26.3M | 251 MB | 49.3 | 60,867 |
| halfframe-64x1 | 64 x 1 | 6,000 | 27.4M | 398 MB | 90.3 | 66,474 |
| quarterframe-32x1 | 32 x 1 | 12,000 | 29.7M | 688 MB | 182.3 | 65,829 |
| eighthframe-8x1 | 8 x 1 | 48,000 | 43.5M | 2,378 MB | 1,078 | 44,509 |
| fine-4x1 | 4 x 1 | 96,000 | 61.9M | 4,654 MB | 4,245 | 22,614 |

#### O(n) scaling check (60s / 30s time ratio, expect ~2.0x)

| Config | Model | 30s ms | 60s ms | Ratio | Verdict |
|--------|-------|--------|--------|-------|---------|
| fullband-128x2 | tiny | 28.3 | 28.3 | 1.00 | linear (below floor) |
| framelevel-128x1 | tiny | 29.2 | 42.7 | 1.46 | linear |
| halfframe-64x1 | tiny | 43.1 | 84.3 | 1.96 | linear |
| quarterframe-32x1 | tiny | 84.4 | 174.4 | 2.07 | linear |
| quarterframe-32x1 | small | 182.3 | 379.8 | 2.08 | linear |
| fine-4x1 | tiny | 2,010 | 4,114 | 2.05 | linear |
| fine-4x1 | small | 4,245 | 8,607 | 2.03 | linear |
| eighthframe-8x1 | tiny | 490.5 | 2,021 | **4.12** | superlinear |
| eighthframe-8x1 | small | 1,078 | 4,255 | **3.95** | superlinear |

### Qualitative

- **Fixed overhead floor (~28ms)**: Everything up to ~3,000 tokens is dominated by CUDA kernel launch and model setup overhead, not actual SSM compute. The SSM doesn't become the bottleneck until ~6k tokens.
- **baseline-16x16 is anomalously slow**: 65ms for 1,496 tokens vs 28ms for fullband-128x2 at 1,500 tokens. Same token count, 2.3x slower. This persists across both model sizes and both durations (ratio stays ~1.0 from 30s to 60s, suggesting a fixed cost unrelated to sequence length). This may be due to the 2D patch grid creating a less efficient memory access pattern for the bidirectional scan.
- **Parameter count grows with tokens**: The learnable position embedding scales with token count. At 96k tokens, the tiny model has 25.2M params (vs 7.1M at 1,500 tokens). This is ~3.5x more parameters, mostly from the position embedding.
- **Memory scales linearly**: GPU memory is roughly proportional to token count across the board, as expected.

## Analysis

### Three scaling regimes

1. **Floor regime (< ~3k tokens)**: Fixed ~28ms overhead dominates. Throughput increases linearly with token count because the numerator grows while the denominator stays constant. Good for real-time inference.

2. **Linear regime (~6k-24k tokens)**: SSM compute dominates. Doubling tokens roughly doubles time (ratios 1.9-2.1x). Peak throughput: ~142k tok/s (tiny) and ~66k tok/s (small). This is the practical sweet spot.

3. **Superlinear regime (>48k tokens)**: `eighthframe-8x1` shows 4x slowdown for 2x tokens. Likely caused by activations exceeding GPU L2 cache, forcing every layer to stream from VRAM. `fine-4x1` at the same token counts (96k) shows 2.0x scaling, suggesting the superlinearity in eighthframe is an artifact of crossing a memory threshold between 48k and 96k tokens, not a fundamental O(n^2) issue.

### Practical implications for BirdCLEF

For 30s audio (BirdCLEF standard clip length) on a 24GB GPU:
- **fullband-128x2** (1,500 tokens, 28ms): Fastest inference, full frequency resolution, 20ms temporal resolution. Good default.
- **framelevel-128x1** (3,000 tokens, 29ms): 10ms temporal resolution at negligible extra cost (still in the floor regime). Best temporal precision with full frequency resolution.
- **halfframe-64x1** (6,000 tokens, 43ms): Splits frequency into 2 bands for finer spectral modeling. Still very fast.
- **quarterframe-32x1** (12,000 tokens, 84ms): 4 frequency bands, 10ms resolution. Peak throughput. Good balance.

For training with batch sizes > 1, memory is the binding constraint. At 3,000 tokens (small model), 251MB per sample allows ~90 samples per batch on 24GB. At 12,000 tokens, 688MB allows ~33 per batch.

### Confounders

- **Batch size 1 only**: Real training uses larger batches. Memory scaling may differ due to gradient accumulation.
- **Synthetic data**: Real audio spectrograms have different value distributions that could affect numerical stability.
- **Single GPU type**: RTX 6000 (Turing). A100/H100 would have different memory bandwidth characteristics and different superlinear thresholds.
- **No backward pass**: Forward-only benchmarking. Training requires backward pass which roughly triples memory.

## Artifacts

| Type | Location |
|------|----------|
| Per-config results | `./results/config-*.json` |
| Combined results | `./results.json` |
| Benchmark script | `./scripts/benchmark_token_shapes.py` |
| SLURM logs | `./logs/` |
| sbatch scripts | `./sbatch/` |

## Next Steps

- [ ] Test backward pass (training) memory to determine practical batch sizes
- [ ] Benchmark the base model (768d) at the promising configs (128x2, 128x1, 64x1, 32x1)
- [ ] Investigate the baseline-16x16 anomaly — profile to understand why square patches are 2.3x slower
- [ ] Design MAE-style self-supervised pretraining with token masking (mask 75%, encoder sees 25% — making high token counts efficient for SSL)
- [ ] Explore MIL (multiple instance learning) head for weak Xeno-Canto labels — attention pooling over token-level representations

**Leads to**: Self-supervised pretraining experiment with rectangular patches; MIL-based weak supervision pipeline for BirdCLEF training data
