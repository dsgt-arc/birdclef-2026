# Proposal: SSAMBA HuggingFace Port

**Status**: confirmed
**Created**: 2026-01-25
**Author**: anthony

## Hypothesis

The SSAMBA pretrained weights can be loaded into a pure PyTorch implementation using HuggingFace's `transformers.MambaMixer` layers, producing numerically equivalent outputs (within floating-point tolerance) to the original CUDA-dependent implementation.

## Background

SSAMBA (Self-Supervised Audio Mamba) provides efficient audio representations using bidirectional Mamba state space models. However, the current implementation requires:
- `mamba_ssm` CUDA kernels
- `causal_conv1d` CUDA kernels
- `VisionMamba` from the Vim repository

This limits deployment to CUDA-enabled environments. For CPU inference (e.g., Kaggle CPU-only submissions, local development), we need a pure PyTorch implementation.

Investigation shows that HuggingFace's `MambaMixer` uses identical parameter names (`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `A_log`, `D`, `out_proj`) to `mamba_ssm`, making weight transfer feasible.

**Related**: [docs/vendor/ssamba.md](../../../docs/vendor/ssamba.md)

## Method

### Approach

Create a pure PyTorch SSAMBA model that:
1. Uses HuggingFace's `MambaMixer` for the core Mamba operations
2. Implements bidirectional processing (forward + backward passes)
3. Preserves the audio patch embedding and positional encoding from original SSAMBA
4. Loads pretrained weights via a conversion function

### Setup

| Component | Details |
|-----------|---------|
| Data | Single test audio file for verification |
| Compute | CPU only (the whole point) |
| Dependencies | `transformers>=4.40.0`, `torch`, `timm` |
| Code | `birdclef/models/ssamba_hf.py` (new) |
| Pretrained | `amba_small_400.pth` from SSAMBA release |

### Procedure

1. Download SSAMBA pretrained checkpoint and inspect structure
2. Implement `SSAMBAConfig` dataclass for model configuration
3. Implement `BidirectionalMambaBlock` wrapping two `MambaMixer` instances
4. Implement `SSAMBAModel` with audio patch embedding + bidirectional Mamba stack
5. Write weight conversion function `convert_ssamba_weights()`
6. Verify numerical equivalence against original implementation (requires GPU for comparison)
7. Benchmark CPU inference speed

### Variables

- **Independent**: Implementation (original vs HuggingFace-based)
- **Dependent**: Output embeddings, inference time
- **Controlled**: Input audio, model size (small), pretrained weights

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Max absolute difference | Largest element-wise difference in output embeddings |
| Mean absolute difference | Average element-wise difference |
| Cosine similarity | Similarity between original and ported embeddings |
| CPU inference time | Time to process 5-second audio on CPU |

### Baseline

Original SSAMBA implementation with `mamba_ssm` CUDA kernels on GPU.

### Success Criteria

- **Confirm if**:
  - Cosine similarity > 0.999 between original and ported outputs
  - Max absolute difference < 1e-4
  - Model loads and runs on CPU without errors
- **Reject if**:
  - Cosine similarity < 0.99
  - Weight loading fails
  - Outputs are qualitatively different

## Limitations

- Only targeting inference, not training
- Will be slower than CUDA implementation (acceptable for CPU deployment)
- HuggingFace Mamba's slow path is sequential, not optimized
- Bidirectional implementation may have subtle differences from VisionMamba's `bimamba_type="v2"`
