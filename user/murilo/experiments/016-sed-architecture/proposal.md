---
experiment: "016-sed-architecture"
status: proposed
date: 2026-03-23
author: Murilo Gustineli
---

# Experiment 016: SED Architecture with CE Loss and 10s Context

## Hypothesis

Replacing clip-level classification (global avg pool → linear head) with Sound Event
Detection architecture (SED: GeM frequency pooling → attention temporal aggregation →
framewise max inference), combined with CE loss and 10-second input windows, will
improve LB score by ≥0.10 over the exp 012 baseline (ProtoCLR 0.767).

## Motivation

### Competition Evidence

A BirdCLEF 2026 competitor reported **0.908 LB with a single EfficientNet-B0** using
only SED + CE loss + 10s windows + mixup. No pseudo-labels, no unlabeled data, no
complex augmentation. This is +0.141 over our best single model (ProtoCLR 0.767).

Their recipe (BirdCLEF 2026 competition discussion, 2026-03-23):
- SED EfficientNet B0, trained on 10 seconds
- Mixup on raw waveforms
- CrossEntropy loss on clipwise and framewise max
- Trained for 20+ epochs
- Single model: 0.908 LB
- 2-model ensemble: 0.914 LB
- No other augmentation
- No unlabeled data

### Winning Solutions Consensus

| Solution | SED? | CE Loss? | Context | Single Model LB |
|----------|------|----------|---------|-----------------|
| 2025 1st (0.937) | GeM + AttHead | CE (dual) | 20s, 5s step | ~0.90+ |
| 2025 9th | AttBlockV2 | BCE | 10s | ~0.88+ |
| 2024 1st (0.691) | No (pure backbone) | CE + sigmoid infer | 10s | ~0.67 |
| 2023 1st | GeMFreq + AttHead | Focal | 5s | ~0.85+ |
| **Us (exp 012)** | **No (global avg pool)** | **BCE** | **5s** | **0.767** |

3 of 4 winning solutions use SED. 2 of 4 use CE loss. 3 of 4 use overlapping context.
We use none of these.

### Exp 015 Confirmed the Bottleneck

Pseudo-label student training (exp 015, +127K soundscape segments) produced no LB
improvement (0.762 vs 0.767 baseline). Offline metrics were flat. The model early-stopped
within 5-9 epochs. Conclusion: the clip-level architecture is the bottleneck — more
data through the same global average pooling cannot fix it.

### Anthony's SED Experiments Provide Guardrails

Anthony's exp A-015 tested SED with soft labels on a **frozen backbone** and got zero
gain. The key finding: SED requires a **trainable backbone** so temporal features can
co-adapt with the attention head. Our LoRA setup provides exactly this — the backbone
adapts its feature representation to produce temporally-informative features that the
SED head can leverage.

## Approach

### Architecture: SED Head

Replace the current linear/MLP classification head with a Sound Event Detection head
following the 2025 1st place pattern:

```
Backbone features: (B, C, F, T)      # e.g., (B, 384, 8, 250) for ProtoCLR
                                      # C=channels, F=freq bins, T=time frames
    │
    ├─ GeM Pooling (frequency axis only)
    │   Collapse F → 1 while preserving T
    │   Output: (B, C, T)
    │
    ├─ Dense projection + dropout
    │   Linear(C, hidden) + ReLU + Dropout(0.5)
    │   Output: (B, hidden, T)
    │
    ├─ Framewise logits (1x1 Conv)
    │   Conv1d(hidden, num_classes, 1)
    │   Output: (B, num_classes, T) — per-frame predictions
    │
    ├─ Attention weights (1x1 Conv)
    │   Conv1d(hidden, num_classes, 1) + softmax(dim=T)
    │   Output: (B, num_classes, T) — per-frame attention
    │
    ├─ Training output:
    │   clipwise = sum(attention * logits, dim=T)   # (B, num_classes)
    │   framewise_max = max(logits, dim=T)          # (B, num_classes)
    │
    └─ Inference output:
        blended = 0.75 * clipwise + 0.25 * framewise_max
```

**Key design decisions:**
- **GeM (not avg) on frequency**: Generalized mean pooling with learnable parameter p.
  At p=1 it's average pooling; at p→∞ it's max pooling. Learns the optimal trade-off.
- **Frequency-only pooling**: Preserves ALL time steps. A 10s segment at hop=320 has
  500 frames (ProtoCLR) or 1000 frames (ConvNeXt) — each frame is a ~20ms window.
- **Dual output**: Clipwise (attention-weighted) captures sustained calls. Framewise max
  captures brief vocalizations (<0.5s) that attention might miss.
- **Blending at inference**: 75/25 clipwise/framewise follows the 2025 1st place ratio.

### Backbone Feature Extraction

Our encoders currently output a single embedding vector via global pooling. For SED,
we need the **intermediate feature map** before the final pooling:

| Encoder | Current Output | SED Feature Map | Channels | Freq | Time (10s) |
|---------|---------------|-----------------|----------|------|------------|
| ProtoCLR (CvT-13) | (B, 384) | (B, 384, F, T) | 384 | TBD | ~500 |
| ConvNeXt-Base | (B, 1024) | (B, 1024, F, T) | 1024 | TBD | ~312 |
| EAT (ViT) | (B, 768) | (B, 768, T) | 768 | 1* | ~500 |

*EAT is a ViT — it outputs (B, seq_len, 768) tokens. Frequency information is already
mixed by self-attention. We can reshape tokens back to a 2D grid or treat it as 1D
temporal sequence (skip GeM, apply attention directly on the token sequence).

**Critical change**: Each encoder needs a `forward_features()` method that returns the
feature map instead of the pooled embedding. This requires modifying the encoder classes.

### Loss Function: Cross-Entropy with Dual Supervision

Replace BCE with CE loss, following the 2024 and 2025 winners:

```python
# Training
clipwise_logits = attention_weighted_sum(frame_logits, attention_weights)  # (B, C)
framewise_max_logits = frame_logits.max(dim=-1).values                     # (B, C)

loss = 0.5 * CE(clipwise_logits, labels) + 0.5 * CE(framewise_max_logits, labels)

# Inference
clipwise_probs = sigmoid(clipwise_logits)
framewise_probs = sigmoid(framewise_max_logits)
final_probs = 0.75 * clipwise_probs + 0.25 * framewise_probs
```

**Why CE over BCE:**
- With 234 classes and typically 1-4 active per segment, the problem is closer to
  multiclass than true multi-label
- CE's competitive softmax produces better-calibrated predictions than independent
  BCE sigmoids
- 2024 winner: CE was "significantly better" than BCE/focal loss
- At inference: sigmoid allows multi-label output

**Why dual supervision:**
- Clipwise loss teaches the attention mechanism to localize vocalizations
- Framewise max loss ensures the model produces high-confidence frame-level predictions
- The 2026 competitor explicitly uses "CrossEntropyLoss on clipwise and framewise max"

### Input: 10-Second Windows

Train on 10-second segments instead of 5-second:

**Training:**
- Load 10s of audio centered on the target 5s segment (2.5s context before + after)
- For focal recordings: random 10s crop (most are >10s)
- For soundscape segments: load 10s centered on the labeled 5s window
  - Edge handling: zero-pad for first/last segments of a soundscape file

**Inference:**
- Process 10s windows with 5s step (same as 2025 1st place)
- For each 5s prediction, the model sees 2.5s of past and 2.5s of future context
- Overlap-average reconstruction for overlapping predictions

**Spectrogram dimensions (10s):**

| Encoder | Sample Rate | Mel Shape (5s) | Mel Shape (10s) |
|---------|------------|----------------|-----------------|
| ProtoCLR | 16kHz | (128, 250) | (128, 500) |
| ConvNeXt | 32kHz | (128, 500) | (128, 1000) |
| EAT | 16kHz | (128, 512) | (128, 1024) |

### Augmentation: Raw Waveform Mixup

Following the competitor's approach:

```python
# During training, 50% probability:
if random() < 0.5:
    # Sample another training example
    waveform2, labels2 = random_sample()
    lambda_ = beta(0.5, 0.5)  # or uniform(0, 1)
    waveform = lambda_ * waveform + (1 - lambda_) * waveform2
    labels = union(labels, labels2)  # multi-hot OR
```

Mixup on raw waveforms (before mel conversion) is more natural for audio than
spectrogram-domain mixup because it preserves phase relationships.

**Round 1 implementation note**: The initial implementation uses **spectrogram-domain
mixup** instead of raw waveform mixup. This is because the mel transform runs in the
dataset's `__getitem__` (CPU workers), so the model receives pre-computed spectrograms.
Moving to true waveform mixup requires either: (a) moving mel computation to the model's
forward pass (GPU-side), or (b) applying mixup in a custom collate function before the
transform. This is deferred to future work if the architecture shows promise.

### Training Strategy

1. **Start from exp 012 checkpoints** (ProtoCLR, ConvNeXt, EAT — soundscape-adapted)
2. **Replace head only** — the backbone LoRA weights transfer directly
3. **New SED head initialized from scratch** (random init)
4. **Higher LR for head** (1e-3) vs backbone (1e-4) — head needs to learn from scratch
5. **Longer training** (30-50 epochs) — new head needs more epochs than fine-tuning
6. **LoRA fine-tuning** — same rank=8, alpha=16 as before, trainable backbone

### Variables

- **Independent**: Architecture (clip vs SED), loss function (BCE vs CE),
  input duration (5s vs 10s), mixup (on/off), blending ratio at inference
- **Dependent**: LB score, offline test AUROC, per-species and per-taxon breakdown
- **Controlled**: Same encoders, same LoRA config, same data splits as exp 012

## Evaluation

### Metrics
- **Primary**: Kaggle LB macro-AUROC
- **Secondary**: Offline test AUROC, cmAP, per-taxon breakdown, focal test AUROC
  (catastrophic forgetting check)

### Baseline
- Exp 012 clip-level: ProtoCLR 0.767, ConvNeXt 0.755, EAT 0.720
- Exp 015 pseudo-label: ProtoCLR 0.762 (no improvement)
- Competition reference: EfficientNet-B0 SED single model 0.908

### Success Criteria
- **Confirm if**: LB improves by ≥0.05 over exp 012 for at least one model
- **Strong confirm if**: LB exceeds 0.85 (approaching the competition reference)
- **Reject if**: LB shows no improvement or focal test AUROC drops >0.03

### Ablation Plan

To isolate the contribution of each change, train ProtoCLR with incremental additions:

| Variant | SED Head | CE Loss | 10s Input | Mixup | Expected |
|---------|----------|---------|-----------|-------|----------|
| A (baseline) | No | BCE | 5s | No | 0.767 (exp 012) |
| B | **Yes** | BCE | 5s | No | ~0.80-0.82 |
| C | Yes | **CE** | 5s | No | ~0.82-0.84 |
| D | Yes | CE | **10s** | No | ~0.84-0.86 |
| E (full) | Yes | CE | 10s | **Yes** | ~0.86-0.90 |

If time is limited, skip ablation and go directly to variant E for all 3 encoders.

## Implementation Scope

### New Code (src/birdclef/)
- `heads/sed_head.py`: AttHead with GeM freq pooling, attention, framewise logits
- `heads/gem_pool.py`: Generalized Mean Pooling module
- `lightning/sed_finetune.py`: SEDFinetuneModule with dual CE loss, blending inference
- `data/sed_dataset.py`: 10s window loading with context padding, waveform mixup

### Modified Code
- Encoder classes: add `forward_features()` returning feature maps (not pooled embeddings)
- ONNX export: update for SED head (different output shape and inference logic)

### Experiment Files (016-sed-architecture/)
- `train.py`, configs, SLURM scripts (following exp 012/015 pattern)

## Dependencies

- Exp 012 checkpoints (complete) — backbone LoRA weights transfer
- Exp 012 training infrastructure (complete) — adapt for SED
- Anthony's SED findings (exp A-015) — trainable backbone required (LoRA provides this)

## Phase 2: Extend SED to Existing Encoders

If EfficientNet-B0 SED shows significant LB improvement (≥0.05 over baseline),
extend the SED architecture to our 3 existing encoders:

### Per-Encoder Approach

| Encoder | Feature Map | GeM Applicable? | Starting Checkpoint |
|---------|-------------|-----------------|---------------------|
| ProtoCLR (CvT-13) | (B, 384, H, W) from last stage | Yes | Exp 012 LoRA checkpoint |
| ConvNeXt-Base | (B, 1024, H, W) via `forward_features()` | Yes | Exp 012 LoRA checkpoint |
| EAT (ViT) | (B, seq_len, 768) token sequence | No — use 1D temporal attention | Exp 012 LoRA checkpoint |

**Strategy**: Add `forward_features()` to each encoder, replace clip-level head with
AttHead, resume from exp 012 LoRA checkpoints (backbone weights transfer, head is new).
Same dual CE loss, 10s windows, and mixup as EfficientNet-B0.

**EAT special handling**: ViT outputs a 1D token sequence, not a 2D feature map.
Need an `AttHead1D` variant that skips GeM and applies temporal attention directly
on the (B, T, C) token sequence.

**Expected outcome**: Our pretrained encoders (19-90M params, bird-audio specialized)
should match or exceed EfficientNet-B0 (5.3M, ImageNet only) when paired with the
same SED head. The combination of strong pretrained features + temporal attention +
CE loss should approach the 0.908 reference.

## Leads to

- **4-model SED ensemble**: EfficientNet-B0 + ProtoCLR + ConvNeXt + EAT with temporal smoothing
- **Pseudo-labeling with SED teachers**: SED models as teachers will produce better
  pseudo-labels because they detect brief calls that clip-level models miss
- **Iterative self-training**: SED teacher → pseudo-label → SED student → repeat
- **OpenVINO optimization**: Enable multi-model ensemble within Kaggle time budget

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Encoder feature maps have unexpected shapes | Test `forward_features()` on each encoder first |
| EAT (ViT) doesn't have 2D feature maps | Use 1D temporal attention (skip GeM freq pooling) |
| 10s windows increase memory/compute | May need smaller batch sizes; LoRA keeps params low |
| CE loss with multi-label data | Sigmoid at inference handles multi-label; CE is softmax only during training |
| Mixup changes label distribution | Start without mixup (variant D), add later if needed |
| LoRA backbone may not produce good temporal features | The competitor's from-scratch EfficientNet works; LoRA should too since backbone is trainable |

## Competitive Intelligence (2026-03-23)

Detailed analysis of active competitor experiments and 2025 winner techniques:
see [competition discussion analysis](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md).

### Key Findings That Affect This Experiment

1. **NS-JFT pretrained B0 > standard ImageNet**: `tf_efficientnet_b0.ns_jft_in1k`
   consistently outperforms our `efficientnet_b0`. Should switch for next run.

2. **ASL may outperform our soft CE**: Competitor ablation showed ASL (γ⁻=4, γ⁺=0,
   clip=0.05) gained +0.023 over BCE. Handles multi-label natively. However, the
   0.908 competitor used CE, so both CE and ASL are valid — NOT using plain BCE is key.

3. **Audio MixUp confirmed +0.036**: 2025 2nd place ablation. We use spectrogram mixup
   which is weaker. This is the single biggest augmentation gain.

4. **Perch baseline = 0.825 LB**: Google's Perch model out-of-the-box beats our best
   clip-level model (0.767). Also serves as a strong teacher for knowledge distillation
   (+0.030 val when used for soft pseudo-labels on soundscapes).

5. **Free post-processing gains (+0.033 combined)**: TopN (+0.011), temporal smoothing
   (+0.010), TTA shifts (+0.012). No retraining needed.

6. **Model soup is free**: Average last 3-5 epoch checkpoints for stability.

7. **2025 1st place core innovation**: PowerTransform pseudo-labels (p^γ, γ=2.0) across
   4 rounds, combined with SoftAUCLoss in later rounds. Baseline 0.872 → final 0.933.

8. **Perch + ProtoSSM = 0.913 LB**: A public notebook uses Perch v2 as frozen feature
   extractor (1536-dim embeddings) + ProtoSSM temporal model + per-class MLP probes +
   site/hour metadata priors. Key insight: Perch alone = 0.825, and building lightweight
   heads on top of a strong foundation model outperforms training from scratch. Suggests
   a hybrid approach (ensemble our SED with Perch-based predictions) or Perch KD as a
   faster path. See [analysis](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach-analysis.md).

## References

- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md): 28 SED experiments, technique rankings, code snippets
- [Perch + ProtoSSM analysis](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach-analysis.md): Foundation model approach (0.913 LB)
- [2025 1st place](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md): GeM + AttHead, CE loss, 20s windows
- [2025 9th place](../../docs/references/kaggle/birdclef-2025/bird2025-9th-place-solution.md): AttBlockV2, framewise max, 10s windows
- [2024 1st place](../../docs/references/kaggle/birdclef-2024/birdclef2024-1st-place-solution.md): CE >> BCE, 10s context, +0.015 from context overlap
- [Winning solutions analysis](../../docs/references/kaggle/winning-solutions-analysis.md): Phase A roadmap
- Anthony's exp A-015: SED needs trainable backbone (our LoRA provides this)
