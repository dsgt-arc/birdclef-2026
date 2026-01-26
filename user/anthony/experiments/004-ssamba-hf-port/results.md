# Results: SSAMBA HuggingFace Port

**Status**: confirmed
**Created**: 2026-01-25
**Author**: anthony

## Summary

Successfully ported SSAMBA to pure PyTorch using HuggingFace's Mamba implementation. The HF port achieves perfect cosine similarity (1.0) with the original CUDA implementation, confirming numerical equivalence.

## Verification Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Cosine Similarity | 1.000000 | > 0.999 | PASS |
| Max Absolute Diff | 5.25e-05 | < 1e-04 | PASS |
| Mean Absolute Diff | 4.72e-06 | - | - |
| Embedding Norm (HF) | 20.7406 | - | - |
| Embedding Norm (Original) | 20.7406 | - | - |

Test conditions:
- Input: Deterministic random spectrogram (seed=42), shape (1, 1024, 128)
- Checkpoint: `ssamba_small_400.pth` (24 layers, 384 embed_dim)
- Original: VisionMamba with mamba_ssm CUDA kernels on RTX 6000
- HF Port: Pure PyTorch on CPU

## Key Implementation Details

### Bidirectional Mamba Architecture

The original SSAMBA uses `bimamba_type="v2"` which processes sequences bidirectionally within each layer:

1. **Forward branch**: Process sequence left-to-right
2. **Backward branch**: Process reversed sequence
3. **Combine**: `output = forward_out * D + backward_out * D_b`

Each layer has separate weights for forward and backward branches:
- Forward: `A_log`, `D`, `conv1d`, `in_proj`, `x_proj`, `dt_proj`, `out_proj`
- Backward: `A_b_log`, `D_b`, `conv1d_b` (shares `in_proj`, `x_proj`, `dt_proj`, `out_proj`)

### Weight Mapping

The conversion maps 413 weights from the original checkpoint:

```
module.v.patch_embed.proj.{weight,bias} -> patch_embed.proj.{weight,bias}
module.v.cls_token -> cls_token
module.v.pos_embed -> pos_embed
module.v.layers.{i}.mixer.* -> layers.layers.{i}.mamba.{forward,backward}.*
module.v.layers.{i}.norm.weight -> layers.layers.{i}.norm.weight
module.v.norm_f.weight -> layers.norm_f.weight
```

### CLS Token Handling

SSAMBA inserts the CLS token in the middle of the sequence (`use_middle_cls_token=True`):
- Position = num_patches // 2
- Final embedding = mean of all patch tokens (excluding CLS)

## Limitations

- **Inference only**: Training not implemented
- **Speed**: HF slow path is ~10x slower than CUDA kernels (acceptable for CPU deployment)
- **Memory**: No chunked processing for very long sequences

## Files

| File | Description |
|------|-------------|
| `ssamba_hf/` | Pure PyTorch implementation package |
| `ssamba_hf/ssamba_hf/model.py` | Main SSAMBAModel class |
| `ssamba_hf/ssamba_hf/mamba.py` | BidirectionalMamba implementation |
| `ssamba_hf/ssamba_hf/convert.py` | Weight conversion utilities |
| `scripts/extract_original_embeddings.py` | Reference embedding extraction |
| `sbatch/run-verification.sbatch` | Verification job script |

## Conclusion

The HuggingFace port successfully replicates the original SSAMBA implementation with numerical equivalence. The pure PyTorch implementation enables CPU inference for deployment scenarios where CUDA is unavailable (e.g., Kaggle CPU-only submissions).

## Next Steps

1. Integrate into main `birdclef/` package for distillation experiments
2. Benchmark CPU inference speed on various audio lengths
3. Consider adding ONNX export for further deployment optimization
