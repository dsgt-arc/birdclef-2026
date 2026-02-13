# Proposal: Attention-Pooled Sound Event Detection

**Status**: rejected
**Created**: 2026-02-11
**Author**: Anthony

## Hypothesis

Replacing mean pooling with attention-pooled SED — where per-token class predictions serve as both temporal detections and attention weights for clip-level aggregation — will improve val/cmAP beyond exp 007's 0.172 (128x1 condition), because:

1. The SED head provides dense supervision (one label per token vs one per clip), giving the backbone stronger gradients and reducing overfitting (exp 007: train cmAP 0.95 vs val 0.17).
2. Attention pooling focuses clip predictions on tokens where the model detects activity, ignoring silence and noise that dilute mean pooling.
3. Per-token `detected_events` labels introduce natural negatives at the token level, teaching the model to discriminate between vocalization and background.

Additionally, switching from precomputed mel caches to on-the-fly nnAudio mel computation makes the pipeline scalable beyond HSN.

## Background

Experiment 007 confirmed that AudioSet pre-trained SSAMBA Tiny achieves 0.172 cmAP on BirdSet HSN with 128x1 patches (frame-level tokenization), and 0.179 with 16x16. Both conditions severely overfit (train cmAP >0.95 vs val ~0.17). The current architecture uses masked mean pooling: all patch tokens are averaged equally, then a linear head predicts class logits.

PANNs (Kong et al.) and similar audio tagging models use per-token predictions as attention weights for clip-level aggregation. The SED head outputs `(B, T, C)` per-token logits, which are used both as temporal detections (trained with `detected_events` labels) and as attention weights (softmax over time) for clip-level pooling. This is standard practice in audio tagging.

See: [Unsupervised Event Detection](../../docs/concepts/unsupervised-event-detection-labeling.md)

## Method

### Approach

This experiment uses only **128x1 patches** (1 token = 1 frame = 10ms) for native SED alignment. No 16x16 condition — exp 007 showed both patch sizes perform similarly on clip-level metrics, and 128x1 is the natural fit for per-token SED.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Exp 007 (current)                                              │
│                                                                 │
│  raw audio ──[precomputed]──► mel (B, 1, 128, T)               │
│       │                                                         │
│       ▼                                                         │
│  Conv2d(1, D, (128,1))  ──► patch tokens (B, T, D)            │
│       │                                                         │
│       ▼                                                         │
│  Mamba backbone (24 layers, pre-trained)                        │
│       │                                                         │
│       ▼                                                         │
│  Mean pool over T  ──► (B, D)                                  │
│       │                                                         │
│       ▼                                                         │
│  Linear(D, C)  ──► clip logits (B, C)                          │
│       │                                                         │
│       ▼                                                         │
│  L = BCE(clip_logits, recording_labels)                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Exp 008 (proposed)                                             │
│                                                                 │
│  raw audio ──[nnAudio on-the-fly]──► mel (B, 1, 128, T)       │
│       │                                                         │
│       ▼                                                         │
│  Conv2d(1, D, (128,1))  ──► patch tokens (B, T, D)            │
│       │                          (same linear tokenizer)        │
│       ▼                                                         │
│  Mamba backbone (24 layers, pre-trained)                        │
│       │                                                         │
│       ▼                                                         │
│  token embeddings (B, T, D)                                     │
│       │                                                         │
│       ▼                                                         │
│  SED head: Linear(D, C)  ──► sed_logits (B, T, C)             │
│       │                                                         │
│       ├──► sigmoid ──► token_probs (B, T, C)                   │
│       │       │                                                 │
│       │       ▼                                                 │
│       │    L_sed = BCE(token_probs, detected_events_labels)     │
│       │                                                         │
│       ├──► softmax(dim=T) ──► att_weights (B, T, C)            │
│       │                                                         │
│       ▼                                                         │
│  clip_pred = Σ_t (att_weights * token_probs)  ──► (B, C)      │
│       │                                                         │
│       ▼                                                         │
│  L_clip = BCE(clip_pred, recording_labels)                      │
│                                                                 │
│  L_total = L_clip + α · L_sed                                  │
└─────────────────────────────────────────────────────────────────┘
```

The SED head is a single `Linear(D, C)` applied per token. Its logits serve two purposes:
1. **sigmoid** → per-token species predictions (trained with `detected_events` labels via L_sed)
2. **softmax over time** → per-class attention weights for clip-level pooling

The clip prediction is the attention-weighted sum of token predictions — tokens where the model detects activity contribute more to the clip score.

#### Dual Loss

```
L_total = L_clip + α · L_sed

L_clip: FocalBCE(clip_pred, recording_labels, γ=γ_clip)       # clip-level
L_sed:  FocalBCE(token_probs, token_labels, γ=γ_sed) * mask   # token-level
```

Token labels for L_sed: for each token at time `t`, label = 1 for the recording's species if any `detected_event` overlaps that frame, else 0. Masked to exclude padded tokens.

Both losses use **focal loss** (Lin et al., 2017) with configurable γ per loss. Focal loss down-weights easy examples and focuses gradients on hard cases:

```
FL(p, y) = -(1 - p_t)^γ · log(p_t)     # γ=0 reduces to plain BCE
```

Default: `α=1.0, γ_sed=2, γ_clip=2`. The token-level imbalance is severe (80%+ background tokens), so focal SED loss is well-motivated. Clip-level imbalance is milder (~90% negative per class in multi-label), so γ_clip may not be needed — configurable to test both `γ_clip=0` (plain BCE) and `γ_clip=2` (focal) empirically.

#### On-the-fly Data Pipeline

Replace precomputed mel cache with sequential streaming:
- Dataset loads raw audio waveforms from BirdSet HF dataset
- **Sequential 10s chunks** — iterate through each recording in non-overlapping 10s windows (no event-centered windowing). This is simpler, sees all audio, and matches inference on real soundscapes.
- All-silence chunks become natural negatives (all token labels = 0, clip labels = 0)
- Per-token label construction from `detected_events`: for each chunk, overlay event boundaries to get per-token labels. `detected_events` is used only for labeling, not windowing.
- nnAudio `MelSpectrogram` as nn.Module for GPU-accelerated mel computation
- Last chunk per recording padded to 10s if shorter (with padding mask)

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdSet HSN: train split (5,460 recordings, ~33K sequential 10s chunks) + test_5s (12,000 clips) |
| Compute | 1x RTX 6000 (24GB), ~14h estimated (128x1 = 1000 tokens, similar to exp 007 condition B) |
| Dependencies | ssamba_hf, mamba-ssm, pytorch-lightning, nnAudio, datasets (HF) |
| Code | `user/anthony/experiments/008-attention-pooled-sed/` |
| Pre-trained | `~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_tiny_400.pth` (Mamba layers only, patch_embed re-initialized) |

### Procedure

1. Add nnAudio to project dependencies, verify mel output matches torchaudio kaldi fbank
2. Implement on-the-fly data pipeline: HF BirdSet → sequential 10s chunks → nnAudio mel → model
3. Implement per-token label construction by overlaying `detected_events` on each chunk
4. Implement `SSAMBASEDClassifier` with attention-pooled SED head (Linear(D, C) per token)
5. Implement dual focal loss (clip + SED, configurable γ)
6. Fast-dev-run and overfit test to verify correctness
7. Full training run with pre-trained Mamba weights
8. Compare against exp 007 condition B (128x1, mean pooling, 0.172 cmAP)

### Variables

- **Independent**: Pooling strategy (attention-pooled SED vs mean pool), training objective (dual loss vs clip-only)
- **Dependent**: val/cmAP, val/AUROC, val/F1_macro, train-val gap
- **Controlled**: Backbone (SSAMBA Tiny 192-dim 24-layer, pre-trained), tokenizer (Conv2d 128x1), patch size (128x1), data (BirdSet HSN), learning rate schedule, augmentation

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| val/cmAP | Class-mean average precision (primary, clip-level) |
| val/AUROC | Area under ROC curve (clip-level) |
| train-val gap | train/cmAP - val/cmAP (overfitting severity) |
| throughput | Samples/sec with nnAudio vs exp 007 precomputed cache |

### Baseline

Experiment 007 condition B (pre-trained SSAMBA, 128x1, mean pooling, precomputed mels):
- val/cmAP: **0.172** (epoch 23, timed out at epoch 33)
- val/AUROC: 0.665
- train-val gap: ~0.78

Reference:
- ConvNeXt (BirdSet LT): 0.47 cmAP, 0.89 AUROC
- Exp 007 condition A (16x16): 0.179 cmAP, 0.684 AUROC

### Success Criteria

- **Confirm if**: val/cmAP > 0.20 AND train-val gap reduced (< 0.70), indicating SED attention pooling and frequency attention improve both performance and generalization
- **Reject if**: val/cmAP < 0.17 or no reduction in train-val gap, indicating these modifications don't help over mean pooling

## Limitations

- **bambird event quality**: `detected_events` are unsupervised detections, not ground truth. Noisy per-token labels may limit the SED loss signal.
- **nnAudio overhead**: On-the-fly mel computation adds GPU compute. If this bottlenecks training, may need to pipeline CPU audio loading with GPU mel + model.
- **Loss hyperparameters**: α (SED loss weight), γ_sed, and γ_clip are all configurable. Starting at α=1.0, γ_sed=2.0, γ_clip=2.0. If double focal gating causes issues (correlated gradient suppression), γ_clip can be set to 0.
- **No contrastive training**: Episode-based contrastive training (discussed as a potential regularizer) is deferred. If overfitting persists after this experiment, contrastive training becomes the natural next step.
