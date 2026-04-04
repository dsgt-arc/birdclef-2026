# Results: SSAMBA SSL Pretraining (Experiment 103)

**Date Paused**: 2026-04-04
**Author**: Anthony
**Status**: PAUSED — linear probe evaluation inconclusive per literature findings

## Summary

Trained SSAMBA-tiny (7.6M params) from scratch on 55K BirdCLEF 2026 audio chunks
with MSE-only masked patch reconstruction (103c config). Linear probe on the
epoch 10 checkpoint gave cmAP 0.009 (~random). **Subsequent literature search
revealed that linear probing is a known-bad evaluation for MAE features** —
Bird-MAE paper explicitly shows 37 mAP gap between linear probe and prototypical
probing at 1.6M data scale. Result is inconclusive: backbone may be learning
useful features that linear probe can't surface. Paused before implementing
prototypical probing or full fine-tuning.

**Verdict**: inconclusive

## Observations

### Quantitative

#### SSL Pretraining (103c config, SSAMBA-tiny, MSE-only)

| Metric | Epoch 0 | Epoch 10 | Epoch 21 |
|--------|---------|----------|----------|
| val/loss (10×MSE) | 7.117 | 5.731 | 5.501 |
| MSE (R² equiv) | 0.712 (0.29) | 0.573 (0.43) | 0.550 (0.45) |
| Δ per epoch | — | -0.061→-0.003 | -0.003 (plateau) |

Learning plateaued after the 5-epoch warmup; each subsequent epoch yields
diminishing returns (~0.5-1% MSE reduction).

#### Linear Probe on epoch 10 checkpoint

| Head | cmAP | Notes |
|------|------|-------|
| Mean-pool linear (job 6270293) | 0.009 | Random ≈ 0.004 |
| Attention-pool (exp 102 style, job 6273638) | 0.009 | Per-token + framewise max |
| **Target (exp 007 AudioSet-pretrained SSAMBA-tiny)** | **0.179** | Our success gate was 0.20 |

### Qualitative

- MSE reconstruction loss decreased smoothly during warmup, plateaued after LR reached peak
- Backbone reaches ~43-45% of variance explained on normalized patch targets
- GPU throughput: 1.9h/epoch on RTX 6000, ~97-100% SM utilization once mel moved to GPU
- Mean-pooling 3072 tokens → 192d discarded spatial structure; attention pooling (per-token logits + framewise max) gave identical results, suggesting information loss is not at the pooling stage
- Auto-chain sbatch with SIGUSR1 trap working: job submits follow-up 120s before wall timeout

## Analysis

### Why the linear probe gave 0.009

The strongest explanation comes from the **Bird-MAE paper** (Can Masked Autoencoders
Also Listen to Birds?, arxiv 2504.12880, 2025), which directly addresses this:

> "Linear probing performs poorly for Audio-MAE and Bird-MAE" — the learned
> features are "reconstruction-oriented patterns rather than discriminative features."

Their results, on the HSN task with Bird-MAE-L trained on 1.6M bird clips:

| Head | mAP |
|------|-----|
| Linear probe | 13.29 |
| Prototypical probing | **49.03** |
| Full fine-tune | 55.28 |

**Even with 29× more data and a Large model, linear probe gets 13 mAP.**
Prototypical probing (learns J class prototypes, uses max-pool patch similarity)
closes 37 mAP of the gap and gets within 6.25% of full fine-tuning.

This makes our 0.009 result consistent with:
1. Linear probe being a fundamentally bad evaluation for MAE features
2. Being 14× below the smallest data scale tested in the Bird-MAE scaling ablation
   (their minimum was 0.8M random samples, which gets 43 mAP after fine-tuning)
3. Early training — the backbone may continue to improve with more epochs

### Why CNN backbones reach 0.92 cmAP but SSL struggles

CNNs (EfficientNet, ConvNeXt) used in exps 012/102 are not "trained on bird audio"
from scratch — they are ImageNet pretrained (1.3M labeled images) then supervised
fine-tuned on BirdSet-XCL (~400K labeled bird clips), giving ~1.7M labeled examples
worth of supervision. ImageNet features (edge, texture, repetition detectors)
transfer surprisingly well to spectrograms.

Our SSL starts from scratch with 0 labels and 55K clips. MSE reconstruction
doesn't require semantic understanding, only local pixel prediction. This is a
fundamentally harder setup for the same downstream task.

### Training progression

Key debugging finds during 103-A/B/C:
- Grid jitter, weight decay, LR schedule bugs in overfit test → fixed
- Depth=12 vs 24: identical results, ruling out gradient flow as the cause
- Freq-first raster order beats time-first for NCE (103-B finding)
- NCE never exceeded ~10% accuracy → dropped for 103-C
- 75% random masking matches SSAMBA paper, implemented cleanly

### Confounders

- **Only 10 epochs evaluated** for linear probe — backbone may be undertrained
  (50 epoch target, training continues in background)
- **Linear probe hyperparameters** (LR, dropout, epochs) not tuned; probe may be
  underperforming its potential
- **55K vs 800K+**: we're at 1/14th the smallest tested data scale for audio SSL
- **Pretrain model uses single norm_f**, classifier replicates this; any mismatch
  between pretrain and classifier forward paths would hurt representations
- **Mel normalization pipeline** extracted as separate function — verified to match
  pretrain numerically, but a regression here would invalidate the evaluation

## Artifacts

| Type | Location |
|------|----------|
| Pretrain checkpoints | `logs/pretrain-103c/checkpoints/` (epochs 0-21+, ~160MB each) |
| Best checkpoint | `logs/pretrain-103c/checkpoints/last.ckpt` |
| Pretrain logs | `logs/pretrain-103c-6242736.out` (ep 0-10), `6265507`, `6273731` |
| Linprobe logs | `logs/linprobe-103c-6270293.out`, `6273638` |
| TensorBoard | `logs/pretrain-103c/lightning_logs/run/` |
| Config | `configs/pretrain-103c-tiny-fp32.yaml`, `configs/linprobe-103c.yaml` |
| Code | Git commit `a105f18` (linear probe), `160f176` (chain fix), `403b8f8` (103c) |

## Next Steps

- [ ] **Implement prototypical probing** (per Bird-MAE paper) — learns J class
      prototypes, max-pools patch similarity. Parameter-efficient, expected to
      close most of the gap to fine-tuning. If this is still ~random, we have
      real evidence backbone is undertrained.
- [ ] **Alternative: full fine-tuning** with labels — skips the probe entirely,
      directly optimizes features for discrimination.
- [ ] **Consider starting from AudioSet-pretrained SSAMBA** (exp 007 got 0.179
      linear probe cmAP) instead of from scratch on 55K.
- [ ] **Data scale reality check**: even with perfect probing, 55K is 14× below
      smallest tested SSL scale. Adding xeno-canto (~1M clips) would bring us
      closer to the published regime.
- [ ] **Stop or let finish**: chain job 6273731 is still running toward 50 epochs.
      Continuing gives us more training data for whatever evaluation we do next.

**Leads to**:
- Concept doc on "MAE features need prototypical/fine-tune eval, not linear probe"
- Potential new exp: AudioSet-pretrained SSAMBA-small + BirdCLEF fine-tune (shortcut
  to strong features without the SSL-from-scratch data problem)
- Reference capture of Bird-MAE findings

## Literature references

- **Bird-MAE** (Rauch et al., 2025): [Can Masked Autoencoders Also Listen to Birds?](https://arxiv.org/html/2504.12880v1)
  — Direct precedent, shows linear probe is broken for MAE
- **SSAMBA** (Shams et al., 2024): [arxiv 2405.11831](https://arxiv.org/html/2405.11831v2)
  — Original architecture, pretrained on AudioSet (2M clips)
- **SSAST** (Gong et al., 2022) — SSAMBA's training recipe predecessor
- **Audio-MAE** (Huang et al., 2022) — Canonical audio MAE, 80% masking, AudioSet scale
