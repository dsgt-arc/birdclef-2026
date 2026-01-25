# Results: 003 Model Distillation

**Date Completed**: In progress
**Author**: Anthony

## Summary

Static embeddings baseline from 500k AudioSet shows modest improvement over exp002, with PCA whitening providing significant gains. Context recovery remains critical for closing the gap to teacher performance.

**Verdict**: In progress

## Observations

### Static Embeddings Baseline (ESC-50 5-fold CV)

| Embedding | Accuracy | Std | Notes |
|-----------|----------|-----|-------|
| exp002 static (old) | ~36% | - | Previous baseline |
| exp003 full (768d) | 38.25% | ±0.42% | 500k AudioSet training |
| exp003 PCA (256d) | 44.55% | ±2.02% | PCA with whitening |
| Full BEATs teacher | 84% | - | Upper bound |

### Key Findings

1. **PCA whitening helps significantly**: 256d PCA embeddings outperform 768d full embeddings by ~6%. Whitening decorrelates features and normalizes variance.

2. **500k AudioSet improves over exp002**: ~2% gain for full embeddings, suggesting more diverse training data helps token representations.

3. **Context gap confirmed**: 40% accuracy gap between bag-of-words (44%) and full teacher (84%) validates the context-recovery hypothesis.

### Qualitative

- Static embeddings pipeline now optimized: 5:48 to process 138 shards (9x speedup from original)
- v2 cache format working well: `teacher_embeds` (fp16) + `token_ids` (int16)
- ShardAffinityDataset with worker affinity eliminates redundant I/O

## Analysis

The PCA result suggests that raw BEATs embeddings have correlated dimensions that hurt linear probe performance. Model2Vec's standard pipeline (PCA + whitening + SIF weights) is validated here.

The 40% context gap confirms that temporal modeling is essential. A lightweight ConvNeXt1D or similar should recover most of this gap while maintaining fast inference.

### Confounders

- Missing 5 shards (00006, 00109, 140-144) from AudioSet download
- ESC-50 is environmental sounds, not bioacoustics - may not fully represent BirdCLEF performance

## Artifacts

| Type | Location |
|------|----------|
| Static embeddings | `scratch/outputs/exp003-distillation/static_embeddings_500k_v2/` |
| ESC-50 results | `scratch/outputs/exp003-distillation/esc50_static_baseline_*.json` |
| Teacher cache | `scratch/outputs/exp003-distillation/teacher_cache_500k_v2/` |
| Logs | `sbatch/logs/` |

## Next Steps

- [ ] Train ConvNeXt1D student on cached teacher embeddings
- [ ] Evaluate student model on ESC-50
- [ ] Test sparse distillation targets (25%, 50% of positions)
- [ ] Cache remaining shards (140-144) when GPU available
- [ ] Evaluate on BirdCLEF validation set

**Leads to**: Context recovery validation, sparse distillation experiments
