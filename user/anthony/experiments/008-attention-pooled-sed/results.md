# Results: Attention-Pooled Sound Event Detection

**Date Completed**: 2026-02-13
**Author**: Anthony

## Summary

Attention-pooled SED with dual focal loss did **not** improve over the simpler mean-pooled clip-level baseline from exp 007. SSAMBA Tiny with SED peaked at 0.164 cmAP vs 0.173 for mean pooling (both 128x1 patches). Scaling to SSAMBA Small (3.6x params) yielded 0.165 cmAP — no gain from increased capacity. Overfitting severity was unchanged.

**Verdict**: rejected

## Observations

### Quantitative

| Metric | Baseline (exp 007 128x1) | SED Tiny | SED Small | Delta (Tiny) |
|--------|--------------------------|----------|-----------|--------------|
| val/cmAP | **0.173** (epoch 11) | 0.164 (epoch 20) | 0.165 (epoch 22) | -0.009 |
| val/AUROC | 0.730 | 0.732 | 0.728 | +0.002 |
| val/F1 macro | 0.086 | 0.069 | 0.100 | -0.017 |
| Train cmAP (final) | 0.952 | 0.910 | 0.950 | — |
| Train-val gap | 0.779 | 0.746 | 0.785 | -0.033 |
| Epochs trained | 27 (early stop) | 36 (early stop) | 30+ (running) | — |
| Params | 7.0M | 7.0M | 25.5M | — |
| Pre-trained loaded | 96.9% | 96.9% | 98.3% | — |

Exp 007 reference (all Tiny, pre-trained):
| Condition | Patch | Pooling | cmAP | AUROC |
|-----------|-------|---------|------|-------|
| A (16x16) | 16x16 | Mean | 0.165–0.179 | 0.647–0.684 |
| B (128x1) | 128x1 | Mean | 0.173 | 0.730 |

### Qualitative

- **SED did not reduce overfitting**: The hypothesis that dense per-token supervision would regularize the backbone was not supported. Train-val gap remained ~0.75–0.78 across all conditions.
- **Scaling model size did not help**: SSAMBA Small (25.5M) performed identically to Tiny (7.0M), confirming the bottleneck is not model capacity.
- **Attention pooling ≈ mean pooling on this task**: AUROC was nearly identical across methods, suggesting the pooling strategy has minimal impact when the backbone representations are the same quality.
- **On-the-fly audio pipeline worked well**: nnAudio mel computation on GPU added negligible overhead. Training throughput was ~0.43 it/s (Small) and ~0.41 it/s (Tiny, batch 64) — comparable to exp 007's precomputed cache.
- **Focal loss trained slower**: Convergence took ~20 epochs to peak vs ~11 for exp 007's plain BCE, likely because focal loss down-weights easy examples and slows early learning.
- **Tiny run showed variance**: Exp 007 condition B hit 0.173 on the clean re-run but 0.172 on the first run. Exp 007 condition A hit 0.179 first, 0.165 on re-run. The ~0.01 gap between SED Tiny (0.164) and baseline (0.173) is within this noise band.

## Analysis

The SED attention-pooling architecture failed to improve over mean pooling for three likely reasons:

1. **Noisy SED labels**: The `detected_events` from bambird are unsupervised detections with unknown precision/recall. If many true vocalizations are unlabeled (false negatives) or noise is labeled as events (false positives), the SED loss provides a misleading training signal. This would explain why adding SED loss didn't help and may have slightly hurt.

2. **The ~0.17 ceiling is a representation problem, not a pooling problem**: Both exp 007 and 008 plateau at the same level regardless of pooling strategy, model size, or loss function. The SSAMBA backbone (pre-trained on AudioSet with 16x16 patches, applied here with 128x1) produces representations that simply can't distinguish HSN bird species beyond ~0.17 cmAP. The patch_embed and pos_embed are re-initialized for 128x1, which discards spatial structure the pre-training learned.

3. **Focal loss may be counterproductive at this scale**: With only 21 classes and moderate class imbalance, focal loss's aggressive down-weighting of easy examples may slow useful gradient flow without meaningfully addressing the real imbalance.

The near-identical performance of Tiny (7M) and Small (25.5M) with SED is strong evidence that the bottleneck is upstream of the classifier — likely in the patch embedding and the mismatch between pre-training tokenization (16x16) and fine-tuning tokenization (128x1).

### Confounders

- **SED Small still running**: The Small model (job 3924395) was at epoch 30 of 100 with early stopping patience 15. Best was 0.165 at epoch ~22, and validation has been flat since. The final number is unlikely to change materially.
- **Different data pipeline**: Exp 008 uses sequential 10s chunks from raw audio (34,642 chunks) while exp 007 used event-centered precomputed mels (38,960 samples). The different chunking strategy may affect results independently of the SED architecture.
- **Batch size differences**: SED Tiny used batch_size=64, SED Small used 32. Exp 007 128x1 used 16. Larger batches may affect optimization dynamics.
- **Run-to-run variance**: Exp 007 condition A showed 0.014 cmAP variance across two runs (0.179 → 0.165). The 0.009 deficit of SED Tiny vs baseline is within this noise.

## Artifacts

| Type | Location |
|------|----------|
| SED Tiny log | `logs/train-sed-3908977.out` |
| SED Small log | `logs/train-sed-small-3924395.out` |
| SED Tiny best ckpt | `logs/sed-128x1/version_3/checkpoints/sed-hsn-epoch=020-val/cmap=0.1639.ckpt` |
| SED Small best ckpt | `logs/sed-128x1-small/` (job 3924395) |
| Tiny config | `config-sed-128x1.yaml` |
| Small config | `config-sed-128x1-small.yaml` |
| Pre-trained ckpts | `~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_{tiny,small}_400.pth` |
| Code | `user/anthony/experiments/008-attention-pooled-sed/` |

## Next Steps

- [ ] Abandon SED attention pooling for SSAMBA — no evidence it helps over mean pooling
- [ ] Investigate the 16x16 → 128x1 patch mismatch as the primary bottleneck: try fine-tuning with 16x16 patches and the SED approach (requires redesigning token-to-time alignment)
- [ ] Consider stronger regularization (mixup, cutmix, larger dropout) to close the train-val gap — this is the dominant problem across all SSAMBA experiments
- [ ] Explore CNN-Mamba hybrid frontends where a pre-trained CNN (ConvNeXt, EfficientNet) provides patch features, avoiding the patch_embed re-initialization problem entirely
- [ ] Revisit whether SSAMBA is the right backbone — ConvNeXt achieves 0.47 cmAP on the same task

**Leads to**: concepts/cnn-mamba-hybrid-frontend.md, potential pivot to CNN-based architectures
