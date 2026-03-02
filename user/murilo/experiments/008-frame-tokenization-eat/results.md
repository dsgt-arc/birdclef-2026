# Results: Frame Tokenization for EAT on BirdCLEF 2025

**Date Completed**: 2026-03-02
**Author**: Murilo Gustineli

## Summary

Frame tokenization does not compensate for losing pretrained patch embeddings in EAT. The
pretrained 16x16 baseline (cmAP=0.494) dominates all configs with randomly initialized patch
embeddings, and frame-shaped patches (128x2, 128x4) perform *worse* than random square 16x16
patches (cmAP=0.260-0.284 vs 0.360). The +34-37% advantage of frame tokenization observed in
Exp 004 (training from scratch) disappears entirely when using pretrained transformers — EAT's
attention layers are tuned to 16x16-shaped token distributions and cannot adapt via LoRA alone.

**Verdict**: rejected — frame tokenization actively hurts when paired with pretrained EAT transformers

## Observations

### Quantitative

Phase 1 test results (4 runs, BirdCLEF 2025, 206 species):

| Config | Patch Embed | Patch Size | Tokens | test cmAP | AUROC | F1 macro | F1 micro | Top-1 | Top-5 | Best Epoch | Stopped | Walltime |
|--------|-------------|------------|--------|-----------|-------|----------|----------|-------|-------|------------|---------|----------|
| **A: baseline** | Pretrained (frozen) | 16x16 | 256 | **0.494** | **0.933** | **0.420** | **0.539** | **0.602** | **0.777** | 25 | 41 | 11h 11m |
| B: control | Random (trained) | 16x16 | 256 | 0.360 | 0.899 | 0.309 | 0.430 | 0.481 | 0.681 | 29 | 45 | 12h 07m |
| C1b: frame 4x128 | Random (trained) | 4x128 | 128 | 0.284 | 0.873 | 0.230 | 0.332 | 0.379 | 0.594 | 28 | 44 | 6h 56m |
| C1a: frame 2x128 | Random (trained) | 2x128 | 256 | 0.260 | 0.864 | 0.231 | 0.315 | 0.357 | 0.575 | 38 | 50 | 13h 45m |

All configs: EAT-base (`worstchan/EAT-base_epoch30_pretrain`), LoRA r=8 (qkv+proj), linear
head, batch_size=32, accum=8 (effective=256), lr=3e-4, early_stopping patience=15, 1x RTX 6000.

**Relative performance (cmAP deltas vs baseline A):**

| Config | test cmAP | Delta vs A | Delta vs B (control) |
|--------|-----------|------------|----------------------|
| A: baseline (pretrained 16x16) | 0.494 | — | +0.134 (+37%) |
| B: control (random 16x16) | 0.360 | -0.134 (-27%) | — |
| C1b: frame 4x128 | 0.284 | -0.210 (-43%) | -0.076 (-21%) |
| C1a: frame 2x128 | 0.260 | -0.234 (-47%) | -0.100 (-28%) |

**Cross-experiment comparison (frame tokenization):**

| Method | Backbone | Pretraining | Patch | cmAP | Dataset |
|--------|----------|-------------|-------|------|---------|
| Exp 008 A: EAT LoRA (pretrained 16x16) | EAT-base | AudioSet | 16x16 | **0.494** | BirdCLEF 2025 |
| Exp 006: EAT LoRA attn+MLP (adapters) | EAT-base | AudioSet | 16x16 | 0.536 | BirdCLEF 2025 |
| Exp 006: EAT LoRA attn (PEFT) | EAT-base | AudioSet | 16x16 | 0.469 | BirdCLEF 2025 |
| Exp 007: ConvNeXt LoRA MLP (PEFT) | ConvNeXt-Base | ImageNet | — | 0.527 | BirdCLEF 2025 |
| Exp 004: ViT 128x4 (from scratch) | ViT-Small | None | 128x4 | 0.118 | HSN |
| Exp 004: ViT 16x16 (from scratch) | ViT-Small | None | 16x16 | 0.086 | HSN |

### Qualitative

- **Pretrained patches are essential**: Config A (pretrained, frozen 16x16) scores 37% higher
  cmAP than Config B (random, trained 16x16), despite Config A's patch embedding being completely
  frozen. The pretrained Conv2d(1, 768, 16, 16) learned discriminative spectral features during
  AudioSet self-supervised training that cannot be recovered by training a random patch embed
  alongside LoRA-adapted transformer blocks.

- **Frame tokenization hurts with pretrained transformers**: Both frame configs score worse than
  the random square control (C1a: -28%, C1b: -21% vs B). This is the opposite of Exp 004, where
  frames beat squares by +34-37%. The pretrained transformer's position embeddings, attention
  patterns, and layer norms were all calibrated for a 32x8 grid of 16x16 patches. Frame tokens
  produce a 1x128 or 1x256 grid that is fundamentally incompatible with these learned patterns.

- **4x128 > 2x128**: Config C1b (4x128, 128 tokens) outperforms C1a (2x128, 256 tokens) by
  +0.024 cmAP. This mirrors Exp 004's finding that shorter sequences are better, but the advantage
  is smaller here (+9% vs +3% in Exp 004). With only 128 tokens, C1b's self-attention is more
  efficient and trains 2x faster (6h 56m vs 13h 45m).

- **Convergence is slower without pretrained patches**: Configs B, C1a, and C1b all train longer
  (38-50 best epoch, 44-50 stopped) compared to Config A (best epoch 25, stopped 41). The
  randomly initialized patch embedding needs more epochs to learn useful features, but the
  final performance is still far behind.

- **GPU utilization is low**: Only 2.4% of RTX 6000's 24 GB VRAM is used (avg 0.61 GB allocated).
  Batch size could theoretically be ~41x larger, and data loading consumes 34.6% of step time.
  Future experiments should increase batch_size and num_workers.

## Analysis

The proposal defined four possible outcomes:

| Outcome | Meaning | Result |
|---------|---------|--------|
| B > C > A | Frame tokenization beats even pretrained patches | No |
| A > B > C | Frame helps vs random squares, but pretrained patches still win | No |
| A > C ~ B | Frame shape doesn't matter; pretraining is what counts | No |
| **A > B > C** | **Frame tokenization actively hurts with pretrained transformers** | **Yes** |

The actual ranking is **A >> B > C1b > C1a**, which maps to the worst-case outcome: frame
tokenization actively hurts. This is worse than "shape doesn't matter" (A > C ~ B) because
frames score significantly below even the random square control.

**Why does frame tokenization fail here when it succeeded in Exp 004?**

1. **Positional encoding mismatch**: EAT's pretrained position embeddings encode a 32x8 spatial
   grid. Frame tokens create a 1xN grid (1x256 or 1x128). Even with regenerated sincos
   embeddings for the new grid, the transformer's attention patterns were trained to exploit
   2D spatial locality across the 32x8 grid. A 1D sequence of frame tokens eliminates the
   frequency-axis positional structure that EAT learned.

2. **Patch embedding is more than a projection**: The pretrained Conv2d(1, 768, 16, 16) doesn't
   just downsample — it extracts learned spectro-temporal features at the 160ms x 16-mel-bin
   scale. A randomly initialized Conv2d(1, 768, 2, 128) or Conv2d(1, 768, 4, 128) starts from
   noise and must learn these features from scratch with only the LoRA-adapted transformer as
   signal. The gradient path through frozen position embeddings and 10 frozen transformer
   blocks (only LoRA adapters trainable) doesn't provide sufficient learning signal.

3. **LoRA alone can't bridge the gap**: LoRA adapts ~0.7% of EAT's 90M parameters. This is
   enough to fine-tune an already-compatible representation (Config A: +0.494 cmAP) but not
   enough to reprogram the transformer for a fundamentally different tokenization scheme.
   Phase 2 (higher LoRA rank, partial unfreeze) was planned to address this, but the magnitude
   of the gap (A is 74% ahead of C1a) makes it unlikely that more capacity would close it.

4. **Exp 004 trained everything from scratch**: In Exp 004, the entire ViT-Small (21.5M params)
   was trained from scratch — patch embedding, position encoding, transformer blocks, and head.
   The model could co-adapt all components to the frame tokenization scheme. In Exp 008, only
   the patch embedding and LoRA adapters are trainable, creating a bottleneck.

### Confounders

- **Asymmetric optimization**: Config A freezes the patch embedding entirely, while B/C configs
  fully train theirs. Different optimization regimes for the same component make the comparison
  imperfect. However, this asymmetry *favors* B/C (more trainable params), making the result
  even more convincing.

- **No data augmentation**: SpecAugment, mixup, and time masking are not used. These could
  interact differently with frame vs square tokenization, but are unlikely to close a 47% gap.

- **Single seed**: All runs use seed=42. Small cmAP differences (<0.01) may not be significant,
  but the observed gaps (0.076-0.234) are well beyond seed variance.

- **Phase 2 not run**: Higher LoRA rank (r=16) and partial unfreeze (blocks 0-1 fully trained)
  were planned but not executed. These could narrow the gap but are unlikely to reverse the
  ranking given the magnitude of the deficit.

- **Different dataset from Exp 004**: Exp 004 used HSN (BirdSet) with ~5K samples; Exp 008 uses
  BirdCLEF 2025 with ~24K recordings. The datasets have different class counts (21 vs 206),
  data volume, and difficulty, making direct comparison imprecise.

## Artifacts

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/{config_a_baseline,config_b_control,config_c1a_frame_2x128,config_c1b_frame_4x128}/metrics.json` |
| Test results | `outputs/test_results/{config-a-baseline,config-b-control,config-c1a-frame-2x128,config-c1b-frame-4x128}.json` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/008-frame-tokenization-eat/checkpoints/` |
| TensorBoard | `outputs/tensorboard/exp008-config-*/` |
| W&B | [config-a-baseline](https://wandb.ai/dsgt-arc/birdclef-frame-tokenization-eat/runs/127350qe), [config-b-control](https://wandb.ai/dsgt-arc/birdclef-frame-tokenization-eat/runs/btqz8g3j), [config-c1a](https://wandb.ai/dsgt-arc/birdclef-frame-tokenization-eat/runs/n8xxglht), [config-c1b](https://wandb.ai/dsgt-arc/birdclef-frame-tokenization-eat/runs/4e946xwj) |
| GPU profiling | `outputs/gpu_check/config-a-baseline_4469090.{txt,png}` |
| SLURM logs | `logs/slurm_config_{a_baseline,b_control,c1a_frame_2x128,c1b_frame_4x128}_*.log` |
| Code | Branch `mg/008-frame-tokenization-eat` |

## Next Steps

- [x] ~~Phase 1: 4 core runs~~ — completed
- [ ] ~~Phase 2: capacity exploration~~ — cancelled (frame configs too far behind to justify)
- [ ] Consider frame tokenization only for models trained from scratch (where Exp 004 showed it helps)
- [ ] For pretrained models (EAT, ConvNeXt), use the native patch size from pretraining
- [ ] Increase batch_size and num_workers in future experiments (GPU utilization was only 2.4%)

**Leads to**: Pretrained patch embeddings are a key component of EAT's performance, not just the
transformer weights. Future work should preserve the pretrained tokenization scheme and focus on
representation quality via SSL pretraining (Exp 009) or teacher-guided training (Exp 011) rather
than modifying the tokenization. Config A (cmAP=0.494) is slightly below Exp 006's best
(cmAP=0.536) due to using PEFT instead of adapters and a linear head instead of MLP, confirming
those earlier findings still hold.
