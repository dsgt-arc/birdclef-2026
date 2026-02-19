# Proposal: Frame-Based Tokenization for ViT

**Status**: proposed
**Created**: 2026-02-16
**Author**: Murilo

## Related Documentation

- [Concept: Temporal Resolution and Sequence Length](../../../../docs/concepts/temporal-resolution-sequence-length.md) — detailed analysis of patch shapes, token budgets, and bioacoustic implications
- [Exp 003 Results](../003-model-training/results.md) — ConvNeXT-Tiny outperforms ViT-Small; analysis attributes CNN advantage to inductive bias for spectrograms
- [AST paper (Gong et al., Interspeech 2021)](https://github.com/YuanGongND/ast) — rectangular 128×2 patches outperform 16×16 when training from scratch

## Hypothesis

Replacing ViT's standard 16×16 square patch tokenization with vertical frame tokenization (128×W patches that span the full frequency axis) will improve cmAP on HSN compared to 16×16 patches trained from scratch, because full-band vertical frames preserve the complete frequency spectrum per token and provide finer temporal resolution — matching how bird calls are naturally structured (narrow in frequency, rapid in time).

## Background

Experiment 003 showed that ConvNeXT-Tiny outperforms ViT-Small in both pretrained (0.180 vs 0.121 cmAP) and scratch (0.060 vs 0.073 cmAP) settings. The analysis identified a core architectural reason: "CNNs have a stronger inductive bias for spectrograms: convolutional layers learn hierarchical features (edges → textures → patterns) that naturally map to frequency-time structure in mel spectrograms. ViTs, by contrast, tokenize the input into fixed patches (16×16) and must learn spatial relationships from scratch via self-attention."

The 16×16 patch is designed for natural images where spatial structure is roughly isotropic — objects span both axes similarly. But spectrograms are fundamentally **anisotropic**: the frequency axis (vertical) represents *what* sounds are present, while the time axis (horizontal) represents *when* they occur. A 16×16 patch on a 128×500 spectrogram captures only 1/8 of the frequency range over 160ms — fragmenting bird calls across multiple frequency patches and smearing temporal detail.

Vertical frame tokenization addresses this mismatch by treating each token as a full-band snapshot at a narrow time window. Each token sees the *entire* frequency spectrum at a given moment, and the sequence of tokens captures how the spectrum evolves over time. This is how birders read sonograms: scanning left-to-right, seeing the full frequency band at each time point.

The AST paper already showed that 128×2 rectangular patches outperform 16×16 when training from scratch (same total patch area of 256 pixels). The reason everyone uses 16×16 is that pretrained ImageNet ViT checkpoints only have 16×16 patch embeddings. Since we're training from scratch on HSN, we are not constrained by pretrained patch sizes.

## Method

### Approach

Modify the ViT-Small patch embedding to use vertical frame patches instead of 16×16 square patches. Test multiple frame widths to find the best temporal resolution. Compare all configurations against the 16×16 scratch baseline from exp 003.

### Configurations

| Config | Patch Shape (freq × time) | Temporal Resolution | Freq Resolution | Tokens (5s, 500 frames) | Embedding Input Dim |
|--------|--------------------------|--------------------|-----------------|--------------------------|--------------------|
| **Baseline** (exp 003) | 16×16 | 160ms | 1/8 band (16 mel bins) | 8 × 31 = 248 | 1 × 16 × 16 = 256 |
| **A: Full-band narrow** | 128×2 | 20ms | full band (128 mel bins) | 1 × 250 = 250 | 1 × 128 × 2 = 256 |
| **B: Full-band wide** | 128×4 | 40ms | full band (128 mel bins) | 1 × 125 = 125 | 1 × 128 × 4 = 512 |
| **C: Full-band single** | 128×1 | 10ms | full band (128 mel bins) | 1 × 500 = 500 | 1 × 128 × 1 = 128 |

Key observations:
- Config A (128×2) produces **250 tokens** — nearly identical to the baseline's 248 tokens, enabling a fair comparison with the same computational cost for self-attention
- Config A has the same embedding input dimensionality (256) as the 16×16 baseline — same number of pixels per patch, just reshaped
- Config B halves the token count (125), reducing attention cost by ~4× but with coarser 40ms temporal resolution
- Config C doubles the token count (500), increasing attention cost by ~4× but with maximum 10ms frame-level temporal resolution

### Setup

| Component | Details |
|-----------|---------|
| Data | HSN dataset from BirdSet, mel-spectrograms (128×500), same as exp 003 |
| Compute | 1× RTX 6000 GPU, 64GB RAM, 8 CPUs |
| Dependencies | timm>=0.9.0, PyTorch Lightning, W&B |
| Code | `user/murilo/experiments/004-frame-tokenization/` |
| Base code | `AudioClassifier` from `user/murilo/src/birdset_benchmark/models/` |

### Procedure

1. Extend `AudioClassifier` (or create a subclass) to accept a configurable `patch_size` tuple — replacing the hardcoded 16×16 patch embedding with `(freq_patches, time_patches)`. timm's `create_model` supports `patch_size` as a keyword argument.
2. Verify positional embedding handling: with non-square patches, the grid of positions changes (e.g., from 8×31 to 1×250 for config A). timm's ViT interpolates positional embeddings when `img_size` changes, but we need to verify this works correctly for the extreme aspect ratios. For scratch training, positional embeddings are randomly initialized so this is less critical.
3. Create config files for each configuration (A, B, C) plus re-run the 16×16 baseline with the same random seed for a controlled comparison.
4. Train all configurations on HSN with mel caching enabled.
5. Compare cmAP, training dynamics, and convergence across configurations.

### Variables

- **Independent**: Patch shape — (16,16) vs (128,2) vs (128,4) vs (128,1)
- **Dependent**: cmAP, AUROC, top-5 accuracy, training convergence speed, walltime per epoch
- **Controlled**: Dataset (HSN), model family (ViT-Small), initialization (scratch), mel-spectrogram parameters (128 bins, 500 frames, 32kHz), training hyperparameters (lr, weight decay, warmup, patience), random seed

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| cmAP | Class-mean average precision (primary) — macro-averaged across 21 HSN species |
| AUROC | Area under ROC curve — discrimination ability |
| Top-5 Accuracy | Practical accuracy for species ID |
| Best Epoch | Convergence speed indicator |
| Time/Epoch | Computational efficiency |

### Baseline

| Run | Patch | cmAP | Source |
|-----|-------|------|--------|
| Exp 003 — ViT-Small scratch | 16×16 | 0.073 | [exp 003 results](../003-model-training/results.md) |
| Exp 003 — ViT-Small pretrained | 16×16 | 0.121 | [exp 003 results](../003-model-training/results.md) |
| Exp 003 — ConvNeXT-Tiny scratch | 16×16 | 0.060 | [exp 003 results](../003-model-training/results.md) |

The primary comparison is against ViT-Small scratch (0.073 cmAP). If vertical frame tokenization closes the gap toward the pretrained ViT-Small (0.121) or even ConvNeXT-Tiny pretrained (0.180), it would demonstrate that the tokenization strategy is a significant factor in ViT's underperformance on spectrograms.

### Success Criteria

- **Confirm if**: Any vertical frame configuration (A, B, or C) achieves higher cmAP than the 16×16 scratch baseline (0.073) on HSN, indicating that full-band vertical tokenization provides a better inductive bias for spectrograms.
- **Reject if**: All vertical frame configurations perform equal to or worse than 16×16 scratch (≤0.073 cmAP), suggesting patch shape is not a significant factor and the ViT's disadvantage is due to other architectural properties.

### Stretch Goals

- Config A (128×2) narrows the gap between ViT-Small scratch and ViT-Small pretrained by ≥50% (i.e., cmAP ≥ 0.097), suggesting that better tokenization partially compensates for the lack of pretraining.
- Best vertical frame config approaches or exceeds ConvNeXT-Tiny scratch (0.060 cmAP) — would be notable since ConvNeXT currently outperforms ViT in the scratch setting.

## Future Scope: Horizontal Frame Tokenization

This experiment focuses on **vertical frames** (full frequency band, narrow time). A natural follow-up is **horizontal frame tokenization** — patches that span the full time axis but capture narrow frequency bands (e.g., W×500 patches). This would produce tokens that represent how a specific frequency band evolves over the entire 5-second clip.

Horizontal frames could be useful for:
- Species with sustained tonal calls at a fixed frequency
- Detecting harmonic structure (each harmonic band becomes its own token)
- Combining with vertical frames in a multi-axis tokenization scheme

These are out of scope for this experiment but the infrastructure (configurable patch sizes) built here will directly support testing horizontal frames.

## Limitations

- Training from scratch only — no pretrained weights exist for non-square ViT patch embeddings, so we cannot test whether vertical frames also help in the pretrained setting. (A follow-up experiment could explore reshaping/interpolating pretrained 16×16 weights into 128×2.)
- Single dataset (HSN, 21 species, ~5.4k training samples) — results may not generalize to larger datasets or different species compositions.
- Single seed per configuration — no variance estimate. If results are close, a multi-seed follow-up would be needed.
- Hyperparameters are carried over from exp 003 scratch settings (lr=5e-4, wd=0.05, warmup=10). These were tuned for 16×16 patches and may not be optimal for vertical frames. A hyperparameter sweep is out of scope but noted as a potential confound.
- The embedding projection dimension changes with patch size (128 for config C, 256 for A, 512 for B) while the ViT hidden dimension stays at 384. This means the linear projection from patch to hidden dim has different shapes across configs, which could affect learning dynamics independently of the tokenization itself.
