# Concept: SSAMBA (Self-Supervised Audio Mamba)

**Status**: developing
**Created**: 2026-01-15
**Updated**: 2026-01-15

## Idea

SSAMBA is the first self-supervised, attention-free audio representation model based on bidirectional Mamba state space models. It replaces the quadratic-complexity transformer in SSAST with linear-complexity Mamba blocks, achieving **92.7% faster inference** and **95.4% better memory efficiency** while maintaining competitive accuracy.

This could be the key to getting under BirdCLEF's inference budget without sacrificing accuracy through aggressive distillation.

## Motivation

Our Experiment 002 showed that static embeddings achieve 32x speedup but suffer 48% accuracy loss due to context collapse. SSAMBA offers an alternative path: keep the full sequence model but use an inherently efficient architecture.

**Key advantages for BirdCLEF**:
- Linear complexity in sequence length (vs quadratic for transformers)
- 92.7% faster than SSAST at 22k tokens (typical for 10s audio)
- Pretrained on AudioSet - same domain as BEATs
- Multiple model sizes available (7M, 26M, 99M params)

## Background

### Architecture

| Component | SSAST | SSAMBA |
|-----------|-------|--------|
| Backbone | Transformer (self-attention) | Bidirectional Mamba (SSM) |
| Complexity | O(n^2) | O(n) |
| Memory | Quadratic in sequence length | Linear in sequence length |

### Model Sizes

| Model | Params | ESC-50 | AudioSet-20K mAP |
|-------|--------|--------|------------------|
| SSAMBA-tiny | 7M | 80.1% | 23.3 |
| SSAMBA-small | 26M | 85.5% | 25.6 |
| SSAMBA-base | 99M | 89.3% | 28.3 |

### Efficiency Gains (vs SSAST-tiny, batch=4, 22k tokens)

- **Inference speed**: 92.7% faster
- **Memory usage**: 95.4% reduction
- **Scaling**: Advantage grows with sequence length due to linear vs quadratic complexity

### Pretraining

- **Data**: AudioSet-2M (~2M 10s clips) + LibriSpeech (960h)
- **Objective**: Joint discriminative and generative masked patch modeling (same as SSAST)
- **Patch masking**: 400 masked patches performs best for audio tasks

### References

- [SSAMBA Paper (SLT 2024)](https://arxiv.org/abs/2405.11831)
- [Official Code](https://github.com/SiavashShams/ssamba)
- [Pretrained Weights](https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX)
- [Mamba Paper](https://arxiv.org/abs/2312.00752)

### Vendor Documentation

- [SSAMBA Vendor Doc](../vendor/ssamba.md) - Local code at `vendor/ssamba/`
- [SSAST Vendor Doc](../vendor/ssast.md) - Transformer baseline at `vendor/ssast/`
- [AST Vendor Doc](../vendor/ast.md) - Supervised baseline at `vendor/ast/`

## Open Questions

- How does SSAMBA-small (85.5% ESC-50) compare to BEATs (95.8%) on bird audio?
- Can we distill BEATs into SSAMBA architecture (best of both worlds)?
- What is the actual CPU inference time for SSAMBA vs our current BEATs pipeline?
- Does bidirectional Mamba work well for variable-length audio (BirdCLEF has varying clip lengths)?

## Testable Hypotheses

1. **SSAMBA-small achieves >80% ESC-50 accuracy at <5ms CPU inference per sample** - This would satisfy BirdCLEF inference budget while maintaining acceptable accuracy.

2. **SSAMBA-base matches BEATs on bird-specific classification** - Despite lower general audio scores (89.3% vs 95.8% on ESC-50), SSAMBA-base might perform comparably on bird vocalizations.

3. **Distilling BEATs into SSAMBA architecture recovers >90% of BEATs accuracy** - Use BEATs as teacher, SSAMBA-small as student architecture. The student learns BEATs' representations but runs at SSAMBA speed.

4. **SSAMBA inference scales better than BEATs for long recordings** - For full soundscape analysis (minutes of audio), SSAMBA's linear complexity should provide even larger speedups.

## Connections

- **Based on**: Mamba state space models, SSAST pretraining framework
- **Related to**: [ssast-audio-transformer](ssast-audio-transformer.md), [distilled-sequence-model](distilled-sequence-model.md)
- **Vendor docs**: [SSAMBA](../vendor/ssamba.md), [SSAST](../vendor/ssast.md), [AST](../vendor/ast.md), [BEATs](../vendor/beats.md)
- **Experiment**: [001-encoder-benchmark](../../user/anthony/experiments/001-encoder-benchmark/proposal.md) Part C
- **Could lead to**:
  - New experiment: Benchmark SSAMBA on ESC-50/BirdSet
  - Hybrid approach: BEATs teacher + SSAMBA student distillation
  - Direct use of pretrained SSAMBA for BirdCLEF submission

## Notes

### Why This Might Be Better Than Our Current Approach

Current Exp 003 approach: Distill BEATs into CNN/MLP-Mixer/TinyTransformer
- Risk: Student might not have capacity to capture BEATs' temporal patterns
- Result: Still struggling to close the accuracy gap

SSAMBA approach: Use pretrained SSAMBA directly or as student architecture
- Advantage: Already pretrained on AudioSet with proven results
- Advantage: Bidirectional context (unlike CNN which is local)
- Advantage: Linear complexity means we can keep full sequence processing

### Integration Path

1. **Quick win**: Benchmark pretrained SSAMBA-small on ESC-50 and BirdSet
2. **If good**: Use SSAMBA directly for BirdCLEF submission
3. **If gap exists**: Distill BEATs into SSAMBA architecture (student has proven capacity)

### Caveats

- SSAMBA-base (89.3% ESC-50) is still 6.5% below BEATs (95.8%)
- Mamba requires custom CUDA kernels for full speed (pure PyTorch is slower)
- Need to verify pretrained weights work with our preprocessing pipeline
