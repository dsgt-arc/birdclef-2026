# Proposal: Multi-Signal Sound Event Detection

**Status**: proposed
**Created**: 2026-02-08
**Author**: Anthony

## Hypothesis

Combining three complementary supervision signals — knowledge distillation from a pre-trained ConvNeXT teacher, bambird temporal event labels, and a self-supervised masked token prediction auxiliary loss — will produce a stronger SSAMBA SED model than any single signal alone, while converging faster due to denser supervision per training sample.

Specifically, the multi-signal model will exceed the ConvNeXT teacher's clip-level cmAP (0.48 on HSN) despite being a smaller model, because the SED formulation provides finer-grained temporal learning that clip-level models cannot achieve.

## Background

Experiment 007 establishes pure token-level SED using bambird's `detected_events`. However, bambird only tells us *when* something vocalizes, not *what* species — it relies on recording-level labels for species identity. This means:
- Chunks where a secondary species vocalizes (not the labeled one) get incorrect per-token labels
- bambird has no confidence scores — it's hard clustering, so borderline events are either fully in or fully out
- No supervision signal from silent/background tokens beyond "this is not a vocalization"

Meanwhile, Murilo's BirdSet ConvNeXT achieves 0.48 cmAP on HSN — it knows species well but only at clip level (5s windows). And SSAMBA was originally designed for masked spectrogram prediction (SSL) but we're training it purely supervised.

This experiment combines all three signals to maximize what each contributes:

| Signal | Provides | Lacks |
|--------|----------|-------|
| bambird | Temporal precision (10ms) | Species identity, confidence |
| ConvNeXT | Species identity + confidence | Temporal precision (5s windows) |
| SSL | Representation quality from all audio | Task-specific supervision |
| Recording labels | Ground-truth species | Temporal info, negatives |

See: [Unsupervised Event Detection](../../docs/concepts/unsupervised-event-detection-labeling.md), [bambird vendor doc](../../docs/vendor/bambird-scikit-maad.md)

## Method

### Approach

Train SSAMBA with three heads sharing a backbone, each providing a different supervision signal. The SED head gets per-token pseudo-labels constructed by intersecting bambird temporal events with ConvNeXT species predictions. The SSL head learns general audio representations via masked token prediction. The clip head provides weak supervision from recording-level labels.

### Architecture

```
Input: mel spectrogram (1, 128, 3000)
              ↓
      SSAMBA backbone (24 layers, 192-dim)
              ↓
  token representations (B, 3000, D)
        ↓           ↓            ↓
    SED head     SSL head     Clip head
   Linear(D,C)  Linear(D,D)  Pool+Linear(D,C)
   (B, T, C)    (B, T, D)    (B, C)
        ↓           ↓            ↓
    L_sed        L_ssl        L_clip

L_total = α·L_sed + β·L_ssl + γ·L_clip
```

### Pseudo-Label Construction

Per-token SED labels are constructed by combining bambird and ConvNeXT:

1. **bambird pass** (CPU, one-time): extract `detected_events` for each recording → `[(t_start, t_end)]` pairs with cluster IDs
2. **ConvNeXT pass** (GPU, one-time): run BirdSet ConvNeXT over each recording in sliding 5s windows → per-window species predictions with confidence scores
3. **Intersection**: for each 10ms token at time `t`:
   - If bambird says a vocalization is happening at `t` (dominant cluster), AND
   - ConvNeXT predicts species X in the surrounding 5s window with confidence > threshold
   - → token label for species X = ConvNeXT confidence (soft label)
   - Otherwise → token label = 0

This gives us:
- **Temporal precision** from bambird (10ms resolution vs ConvNeXT's 5s)
- **Species identity + confidence** from ConvNeXT (disambiguates multi-species recordings)
- **Natural negatives** at the token level (most frames are silent)
- **Soft labels** from ConvNeXT confidence (handles uncertain detections gracefully)

### Loss Functions

**SED loss** — per-token BCE with soft pseudo-labels, masked for padded tokens:
```python
# logits: (B, T, C), pseudo_labels: (B, T, C), mask: (B, T)
L_sed = BCE(logits, pseudo_labels) * mask.unsqueeze(-1)
L_sed = L_sed.sum() / (mask.sum() * C)
```

**SSL loss** — masked token prediction (following SSAMBA/EAT pre-training):
```python
# Mask ~40% of input tokens, predict original from context
# Only compute loss on masked positions
L_ssl = MSE(predicted_tokens, original_tokens) at masked positions
```

**Clip loss** — weak supervision from recording-level labels:
```python
# Pool real tokens → clip prediction
pooled = masked_mean_pool(tokens, mask)  # (B, D)
clip_logits = clip_head(pooled)          # (B, C)
L_clip = BCE(clip_logits, recording_labels)
```

**Total**: `L = α·L_sed + β·L_ssl + γ·L_clip`

Loss weights `α, β, γ` are hyperparameters. Starting point: `α=1.0, β=0.1, γ=0.5` (SED primary, SSL regularizer, clip secondary).

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdSet HSN train (with `detected_events`) + ConvNeXT pseudo-labels (pre-computed). Val on test_5s |
| Compute | 1x RTX 6000 (24GB). Three heads add minimal VRAM over single-head. SSL masking reduces compute (skip masked tokens in backbone) |
| Dependencies | ssamba_hf, pytorch_lightning, torchaudio, soundfile. ConvNeXT checkpoint from Murilo's exp 001 |
| Code | Fork from exp 007 (SED baseline), add SSL and clip heads |
| Pre-compute | bambird events (CPU job), ConvNeXT inference (GPU job, ~8 min on HSN per Murilo's results) |

### Procedure

1. Pre-compute ConvNeXT pseudo-labels: run Murilo's ConvNeXT over all HSN train recordings in 5s sliding windows, save per-window species predictions as parquet
2. Pre-compute bambird events (if not using BirdSet's `detected_events` directly): run bambird on all train recordings, save event timestamps
3. Build intersection: combine bambird temporal events + ConvNeXT predictions into per-token soft labels, cache alongside mel spectrograms
4. Implement multi-head architecture: SED head (from exp 007) + SSL head (masked token prediction) + clip head (pooled weak supervision)
5. Implement multi-task loss with configurable weights
6. Fast-dev-run: verify all three losses are finite, shapes correct, no OOM
7. Ablation: train with each signal individually, then combined, compare convergence and final metrics
8. Full training run with best configuration

### Variables

- **Independent**: supervision signals (SED only, SSL only, clip only, SED+SSL, SED+clip, all three)
- **Dependent**: val/cmAP, val/AUROC, convergence speed (epochs to reach target cmAP)
- **Controlled**: backbone (SSAMBA 192-dim, 24-layer), data (BirdSet HSN), mel parameters, training schedule

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| val/cmAP (clip) | Class-mean AP on val clips — primary metric, comparable across all experiments |
| val/AUROC (clip) | Area under ROC on val clips |
| convergence_epoch | Epoch at which val/cmAP first exceeds 0.10 — measures learning efficiency |
| token/event_f1 | Token-level event detection F1 on a held-out subset with known events — measures temporal precision |
| ssl/reconstruction_loss | SSL head loss — sanity check that backbone learns good representations |

### Baselines

| Experiment | cmAP | Description |
|------------|------|-------------|
| 006 (clip-level, 30s) | ~0.08 | Recording-level labels, no negatives |
| 007 (SED, bambird) | TBD | Per-token labels from bambird only |
| ConvNeXT teacher | 0.48 | Upper bound from pre-trained model |
| ViT supervised | 0.164 | From-scratch supervised baseline |

### Success Criteria

- **Confirm if**: val/cmAP > 0.25 (exceeding ViT-from-scratch, closing gap to ConvNeXT) AND converges faster than exp 007 (reaches 0.10 cmAP in fewer epochs)
- **Reject if**: val/cmAP < 0.15 or no improvement over exp 007 alone, indicating that multi-signal complexity doesn't justify the additional engineering

## Limitations

- **Teacher quality ceiling**: ConvNeXT pseudo-labels are noisy — the student can't exceed the teacher's knowledge unless the SED formulation provides complementary signal. On HSN specifically, ConvNeXT is strong (0.48), but on harder datasets it may produce poor pseudo-labels.
- **Loss weight sensitivity**: Three-way multi-task learning is notoriously sensitive to loss weights. May require significant tuning of `α, β, γ`.
- **SSL masking + SED interaction**: Masking tokens for SSL removes them from SED supervision. Need to ensure masked tokens don't corrupt SED gradients (detach or separate forward passes).
- **Pre-compute cost**: Requires running both bambird and ConvNeXT over all training audio. For BirdCLEF competition (potentially 10k+ recordings), this adds pipeline complexity.
- **Depends on exp 007**: This experiment builds on exp 007's SED implementation. If 007 reveals fundamental issues with token-level SED on SSAMBA, this experiment may need redesign.
- **Ablation cost**: Testing all signal combinations (7 configurations) multiplies GPU time. May need to prioritize key comparisons.
