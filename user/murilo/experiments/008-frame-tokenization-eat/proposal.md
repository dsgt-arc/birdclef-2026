# Experiment 008: Frame Tokenization for EAT on BirdCLEF 2025

**Status**: Proposed
**Created**: 2026-02-28
**Author**: Murilo Gustineli

## Hypothesis

Replacing EAT's standard 16×16 square patch embedding with full-frequency vertical frame
tokenization (128×W, where W ∈ {2, 4}) will produce comparable or higher cmAP than the
pretrained 16×16 tokenization, despite requiring a randomly initialized patch embedding,
because full-band spectral tokens better capture the harmonic structure of bird vocalizations.

## Background

Exp 004 demonstrated that vertical frame tokenization (128×2, 128×4) consistently outperforms
square 16×16 patches when training ViT from scratch on spectrograms (+34-37% cmAP). The
intuition is that spectrograms are anisotropic: the frequency axis encodes harmonic structure
while the time axis encodes temporal dynamics. Square patches fragment both axes, while vertical
frames preserve the full frequency band per token.

However, exp 004 trained from scratch — no pretrained weights to preserve. EAT's value comes
from its AudioSet self-supervised pretraining, and the pretrained patch embedding
(`PatchEmbed_new`: `Conv2d(1, 768, 16, 16)`) was trained with 16×16 patches. Changing the
patch shape to (128, W) requires a randomly initialized Conv2d, breaking compatibility with
pretrained patch embeddings. The transformer blocks, however, can be preserved via LoRA.

Exp 006 established that LoRA fine-tuning EAT achieves cmAP=0.536 (best config, adapters
backend) and cmAP=0.482 (best PEFT config with all-layer LoRA) on BirdCLEF 2025. Exp 007
showed ConvNeXt-Base with PEFT LoRA reaches cmAP=0.527. These serve as baselines.

The core research question: **does full-frequency frame tokenization compensate for losing
pretrained patch embeddings in EAT?**

Downstream goal: select the best teacher model for pseudo-label generation in a future
weak supervision experiment.

## Method

### Approach

Replace EAT's `PatchEmbed_new` with a new Conv2d using vertical frame kernel/stride
(128×2 or 128×4). Fully train the new patch embedding from scratch while LoRA fine-tuning
the pretrained transformer blocks. Use a phased experiment design with a proper control
to isolate the tokenization effect from the pretraining loss.

### Data Flow

```
BirdCLEF2025DataModule (16kHz waveforms, 5s chunks)
  → fbank transform (128 mel bins, 512 time frames)
  → (B, 1, 512, 128) spectrograms
  → PatchEmbed: Conv2d(1, 768, (128, W), (128, W))   — W ∈ {2, 4}
  → (B, 512/W, 768) frame tokens                      — 256 or 128 tokens
  → EAT transformer (12 blocks, pretrained + LoRA)
  → mean pool → classification head
  → (B, 206) logits
```

### Run Matrix

**Phase 1** — Core question + frame size selection (4 runs):

| Config | Patch Embed | Transformer | Tokens | Purpose |
|--------|-------------|-------------|--------|---------|
| **A: baseline** | Pretrained 16×16 (frozen) | LoRA r=8 | 256 | Reproduces exp 006 PEFT baseline |
| **C: control** | Random 16×16 (full train) | LoRA r=8 | 256 | Isolates pretraining loss |
| **B1a: frame-128×2** | Random 128×2 (full train) | LoRA r=8 | 256 | Frame tokenization, 20ms resolution |
| **B1b: frame-128×4** | Random 128×4 (full train) | LoRA r=8 | 128 | Frame tokenization, 40ms resolution |

**Phase 2** — Capacity exploration (2 runs, uses best frame size from Phase 1):

| Config | Patch Embed | Transformer | Tokens | Purpose |
|--------|-------------|-------------|--------|---------|
| **B2: frame + higher rank** | Random 128×? (full train) | LoRA r=16 | ? | More adapter capacity |
| **B3: frame + partial unfreeze** | Random 128×? (full train) | LoRA r=8 + blocks 0-1 fully trained | ? | Bridge layers for new tokenization |

### Key Implementation Details

- **Patch embedding**: Fully trained (not LoRA, not frozen) in all B and C configs. Only ~196K
  params for 128×2, ~98K for 128×4. Config A keeps pretrained patch embedding frozen.
- **Positional encoding**: New sincos 2D encoding computed for the frame token grid
  (1×256 for 128×2, 1×128 for 128×4). Config A/C use original 32×8 grid encoding.
- **LoRA targets**: attention layers `["qkv", "proj"]` via PEFT (same as exp 006).
- **B3 partial unfreeze**: First 2 transformer blocks fully trained (no LoRA), remaining
  10 blocks use LoRA. Provides a "bridge" between new patch embedding and deeper layers.

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2025 (206 species, 5s chunks, 80/20 train/test split) |
| Compute | 1× RTX 6000 (24GB), 8-12h per run, PACE cluster |
| Dependencies | PEFT, transformers, pytorch-lightning, nnAudio, timm |
| Code | Branch `mg/008-frame-tokenization-eat` |
| Base model | `worstchan/EAT-base_epoch30_pretrain` (HuggingFace) |

### Procedure

1. Set up experiment directory, configs, and SLURM scripts
2. Implement frame patch embedding (modify `PatchEmbed_new` to accept variable kernel/stride)
3. Implement positional encoding regeneration for non-square grids
4. Implement partial unfreeze logic for B3 config
5. Validate with fast-dev-run (all configs)
6. **Phase 1**: Submit configs A, C, B1a, B1b
7. Analyze Phase 1 results, select best frame size
8. **Phase 2**: Submit configs B2, B3 with winning frame size
9. Analyze all results, document findings

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 32 | Same as exp 006 (EAT activation maps) |
| lr | 3e-4 | Standard LoRA LR |
| max_epochs | 100 | Extended for configs with random patch embed |
| early_stopping_patience | 15 | Longer patience for slower convergence |
| accumulate_grad_batches | 8 | effective batch_size = 256 |
| hidden_dim | 0 | Linear head (consistent with exp 006) |
| warmup_epochs | 5 | Same as exp 006 |
| precision | 16-mixed | AMP for memory efficiency |

### LoRA Configuration

| Parameter | Value (default) | Value (B2) |
|-----------|-----------------|------------|
| rank (r) | 8 | 16 |
| alpha | 16 | 32 |
| dropout | 0.1 | 0.1 |
| target_modules | `["qkv", "proj"]` | `["qkv", "proj"]` |
| bias | "none" | "none" |

### Variables

- **Independent**: patch embedding shape (16×16 vs 128×2 vs 128×4), patch embedding
  initialization (pretrained vs random), LoRA rank (8 vs 16), transformer layer freezing
  (all LoRA vs partial unfreeze)
- **Dependent**: test cmAP, AUROC, F1, training time, convergence speed (best epoch)
- **Controlled**: dataset (BirdCLEF 2025), EAT-base backbone, training hyperparameters
  (lr, effective batch size, optimizer), PEFT backend, classification head, seed (42)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| test cmAP | Class-mean average precision (primary) |
| AUROC | Area under ROC curve |
| F1 macro/micro | Classification F1 scores |
| Top-1/Top-5 accuracy | Standard accuracy metrics |
| Best epoch | Convergence speed indicator |
| Training time | Wall clock time per run |

### Baseline

| Method | Source | test cmAP |
|--------|--------|-----------|
| EAT LoRA attn+MLP head (adapters) | Exp 006 | 0.536 |
| EAT LoRA all (PEFT) | Exp 006 | 0.482 |
| EAT LoRA attn (PEFT) | Exp 006 | 0.469 |
| ConvNeXt LoRA MLP (PEFT) | Exp 007 | 0.527 |
| Frozen linear probe | Exp 005 | 0.301 |

### Success Criteria

- **Confirm if**: Best frame config (B1a or B1b) achieves cmAP > Config C (random 16×16 control),
  demonstrating that frame tokenization provides a benefit beyond just retraining the patch embedding
- **Reject if**: Best frame config cmAP ≤ Config C, meaning the frame shape provides no advantage
  over standard square patches when both start from random initialization
- **Bonus finding**: Any B config achieves cmAP > Config A (pretrained baseline), meaning frame
  tokenization overcomes the pretraining loss entirely

### Interpretation Guide

| Outcome | Meaning |
|---------|---------|
| B > C > A | Frame tokenization is powerful enough to beat even pretrained patches |
| A > B > C | Frame helps vs random squares, but pretrained patches still win |
| A > C ≈ B | Frame shape doesn't matter; pretraining is what counts |
| A > C > B | Frame tokenization actively hurts with pretrained transformers |

## Limitations

- **Asymmetric comparison**: Config A uses pretrained patch embedding (frozen), while B/C
  configs fully train theirs. Different optimization regimes for the same component.
- **Single seed**: All runs use seed=42. Small cmAP differences (<0.01) may not be significant.
- **PEFT only**: Exp 006 showed adapters outperform PEFT by +0.01-0.025 cmAP. We use PEFT
  for simplicity; the winning config could be re-run with adapters later.
- **Token count varies**: 128×2 produces 256 tokens (same as 16×16), but 128×4 produces
  128 tokens. Sequence length affects transformer compute and attention patterns.
- **No data augmentation**: SpecAugment, mixup, etc. are not used. These could interact
  differently with frame vs square tokenization.
