# Results: Pre-trained SSAMBA Transfer

**Date Completed**: 2026-02-11
**Author**: Anthony

## Summary

Pre-trained SSAMBA Tiny transfer achieves ~0.18 cmAP on BirdSet HSN, doubling the from-scratch baseline (0.086). Both the default 16x16 tokenizer (condition A) and the frame-level 128x1 tokenizer (condition B) exceeded the >0.15 cmAP success threshold. Surprisingly, condition B (Mamba-only transfer) nearly matches condition A (full transfer) despite re-initializing 7% of parameters.

**Verdict**: confirmed

## Observations

### Quantitative

| Metric | Scratch (exp 006) | Condition A (16x16) | Condition B (128x1) |
|--------|-------------------|---------------------|---------------------|
| cmAP | 0.086 | **0.179** (epoch 31) | **0.172** (epoch 23) |
| AUROC | — | 0.684 (epoch 27) | 0.665 (epoch 18) |
| F1 macro | — | 0.137 | 0.102 |
| Epochs trained | — | 46 (early stop) | 33 (timed out) |
| Train cmAP | — | 0.974 | 0.952 |

Reference benchmarks (HSN):
| Model | cmAP | AUROC | Source |
|-------|------|-------|--------|
| ConvNeXt (BirdSet LT) | 0.47 | 0.89 | BirdSet paper |
| EfficientNet (BirdSet LT) | 0.35 | 0.86 | BirdSet paper |
| Bird-MAE ViT-L | 0.55 | — | Bird-MAE paper |
| ViT-Base (fine-tuned) | 0.164 | 0.578 | Murilo exp 002 |

### Qualitative

- Both conditions show severe overfitting: train cmAP >0.95 vs val cmAP ~0.17
- AUROC is noisy and plateaus around epoch 15 for both conditions, no clear upward trend after that
- Condition B (128x1, 1000 tokens) trains 2x slower per epoch than A (496 tokens) due to halved batch size (16 vs 32), but per-step throughput is identical (~1.79 it/s)
- Condition B was still climbing when it timed out at epoch 33; may have slightly higher ceiling with more time

## Analysis

Pre-training provides a substantial boost (+108% cmAP over scratch), confirming that AudioSet-learned representations transfer to bird audio. However, the absolute performance (0.18 cmAP) is still far from competitive models (ConvNeXt 0.47, Bird-MAE 0.55).

The near-parity of conditions A and B suggests the Mamba layers carry most of the transferable knowledge, while patch_embed and pos_embed contribute minimally. This is consistent with the idea that the SSM's learned dynamics (how to process sequential audio features) matter more than the specific patch tokenization.

The large train-val gap indicates the model memorizes the training set quickly but struggles to generalize. Possible causes:
- SSAMBA Tiny (7M params) may lack capacity compared to ConvNeXt/ViT-Base (85M+)
- AudioSet pre-training provides generic audio features, not bird-specific ones
- The 38,960 event-centered samples, while 7x more than the raw HSN train split (5,460), may still be insufficient for the 21-class multilabel problem

### Confounders

- **Shared log directory**: Both conditions accidentally wrote to the same TensorBoard version and checkpoint directory in the first run (jobs 3869164/3869165). Metrics were extracted from per-job SLURM output logs, which are reliable. Resubmitted with separate `--run-name` to get clean logs (jobs 3897690/3897692).
- **Condition B timeout**: Hit the 12h wall limit at epoch 33. Early stopping (patience 15) hadn't triggered yet, so the true converged performance may be slightly higher.
- **Training data difference**: Our runs used 38,960 event-centered samples vs Murilo's ViT-Base which used only 5,460 native HSN samples — not a direct apples-to-apples comparison with that baseline.

## Artifacts

| Type | Location |
|------|----------|
| Condition A log | `logs/train-default-3869164.out` |
| Condition B log | `logs/train-framelevel-3869165.out` |
| Best checkpoint A | `logs/ssamba-hsn-pretrained/version_3/checkpoints/ssamba-hsn-epoch=031-val/cmap=0.1787.ckpt` |
| Clean re-run A | SLURM job 3897690 (pending) |
| Clean re-run B | SLURM job 3897692 (pending) |
| Mel cache | `~/scratch/birdclef-2026/cache/007-hsn-events/` (38,960 train, 12,000 val) |
| Pre-trained ckpt | `~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_tiny_400.pth` |
| Code | `user/anthony/experiments/007-pretrained-ssamba-transfer/` |

## Next Steps

- [ ] Collect clean results from re-run (jobs 3897690/3897692) with separated logs
- [ ] Consider stronger regularization (more augmentation, mixup, larger dropout) to close train-val gap
- [ ] Investigate whether SSAMBA Tiny's capacity (7M params) is a fundamental bottleneck — compare with SSAMBA Small/Base if checkpoints available
- [ ] Evaluate whether a CNN frontend (hybrid architecture) could provide better features than raw mel patches

**Leads to**: concepts/cnn-mamba-hybrid-frontend.md, future experiments on model scaling or hybrid architectures
