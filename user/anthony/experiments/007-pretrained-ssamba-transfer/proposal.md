# Proposal: Pre-trained SSAMBA Transfer

**Status**: in-progress
**Created**: 2026-02-10
**Author**: Anthony

## Hypothesis

Loading AudioSet pre-trained SSAMBA weights and fine-tuning on BirdSet HSN with 10s event-centered windows will significantly outperform training from scratch (exp 006: 0.086 cmAP), achieving >0.15 cmAP. The default 16x16 tokenizer (condition A) should transfer best since patch_embed and pos_embed match the pre-training configuration.

## Background

Exp 006 trained SSAMBA Tiny from scratch on BirdSet HSN and achieved only 0.086 cmAP, compared to 0.48 for pre-trained ConvNeXT. The dominant cause is lack of pre-training — the Mamba backbone needs AudioSet pre-training to learn general audio representations before fine-tuning on the smaller BirdSet dataset.

Additionally, exp 006 used 30s sequential chunks, which waste computation on silence. Switching to 10s event-centered windows using BirdSet's `detected_events` field:
- Matches AudioSet's pre-training length (10s), so pos_embed transfers directly for 16x16
- Provides natural negatives (non-event windows)
- Reduces wasted computation on silence

## Method

### Approach

Two conditions testing different tokenizer configurations:

| Condition | Patch Size | Tokens @10s | Pre-trained Transfer |
|-----------|-----------|-------------|---------------------|
| A (default) | 16x16 | 496 | ~99% params (patch_embed + pos_embed transfer) |
| B (framelevel) | 128x1 | 1000 | ~93% params (Mamba layers only; patch_embed + pos_embed re-initialized) |

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdSet HSN train (event-centered 10s) + test_5s (padded to 10s) |
| Compute | 1x RTX 6000, ~12h per condition |
| Dependencies | ssamba_hf, mamba-ssm, pytorch-lightning |
| Code | `user/anthony/experiments/007-pretrained-ssamba-transfer/` |
| Pre-trained | `~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_tiny_400.pth` |

### Procedure

1. Pre-compute event-centered 10s mel spectrograms from BirdSet HSN
2. Load AudioSet pre-trained SSAMBA Tiny weights
3. Freeze backbone for 5 epochs, train patch_embed + head
4. Unfreeze and fine-tune all parameters
5. Evaluate on test_5s validation set

### Variables

- **Independent**: Tokenizer configuration (16x16 vs 128x1)
- **Dependent**: val/cmAP, val/auroc, val/f1_macro
- **Controlled**: Pre-trained checkpoint, learning rate, freeze schedule, data pipeline

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| cmAP | Class-mean average precision (primary) |
| AUROC | Area under ROC curve |
| F1 macro | Macro-averaged F1 score |

### Baseline

- Exp 006 (SSAMBA Tiny from scratch, 30s chunks): 0.086 cmAP
- ConvNeXT (pre-trained): 0.48 cmAP

### Success Criteria

- **Confirm if**: val/cmAP > 0.15 for at least one condition
- **Reject if**: val/cmAP < 0.10 for both conditions after full training

## Limitations

- Pre-trained checkpoint is AudioSet-only (no bird-specific pre-training)
- 10s window extraction depends on quality of BirdSet `detected_events` annotations
- Condition B re-initializes ~7% of parameters (patch_embed + pos_embed), which may need more careful learning rate scheduling
