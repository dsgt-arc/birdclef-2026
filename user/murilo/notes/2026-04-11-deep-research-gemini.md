# Deep Research: Diagnosing the BirdCLEF 2026 Performance Ceiling

**Source**: Gemini Pro Deep Research
**Date**: 2026-04-11
**Prompt**: `user/murilo/notes/deep-research-prompt.md` + full `sed-architecture-overview.md`

---

## Executive Summary

The 0.903 → 0.922+ gap is rooted in four systemic pipeline failures:
1. Temporal impedance mismatch (train 20s, infer 5s)
2. BCE loss "reward hacking" under extreme class imbalance
3. Spectrogram MixUp violating physics of additive sound waves
4. Random crop selection starving the network of positive acoustic energy

---

## 1. Resolving the Temporal Impedance Mismatch

### The Problem

Training on 20s but evaluating on 5s segments disrupts batch normalization
statistics, misaligns temporal pooling mechanisms, and alters the geometry of
intermediate feature maps. This is not just an attention calibration issue —
it fundamentally breaks the model's learned representations.

### Sliding Window Inference with Overlap (THE key technique)

Nikita Babych (2025 1st place) heavily emphasized overlapping sliding window
inference as a consistent LB booster.

**How it works**: Instead of processing 12 non-overlapping 5s segments from a
60s test file, the pipeline:
1. Extracts a 20s window
2. Passes through the SED model → framewise predictions
3. Strides forward by 1–2.5 seconds
4. Repeats, generating heavily overlapping predictions
5. For each required 5s evaluation segment, **averages all framewise predictions
   from all overlapping windows** that intersect with it

**Three advantages**:
- **Test-time augmentation**: same 5s event viewed in different 20s contexts
- **Boundary smoothing**: calls spanning 5s boundaries are captured intact in
  at least one overlapping window
- **Variance reduction**: aggregation of multiple inferences stabilizes logits,
  reducing false positives from transient noise

### Estimated Impact

| Strategy | Approximate LB |
|----------|---------------|
| Naive 5s cropping | 0.890–0.900 |
| Center crop from 15s | 0.900–0.910 |
| **Overlapping SED TTA (1–2.5s stride)** | **0.920+** |

### Implication for Us

We never changed our inference pipeline. Every longer-window experiment was
evaluated with naive 5s cropping — handicapping the model. This alone could
explain the full regression we saw with v9a/v9b/v9c.

---

## 2. Cross-Entropy Loss: Why BCE Fails at Scale

### The BCE Failure Mode

With 234 classes and typically 1–2 positive labels per sample, BCE computes
independent losses across all classes. The network quickly learns that pushing
ALL logits toward negative infinity minimizes the cumulative penalty from the
~232 zero-labels. This is "reward hacking" — the model becomes conservative
and refuses to make high-confidence predictions for rare species.

### Why CE Works

Categorical CE with softmax forces the model to allocate exactly 1.0 probability
mass across classes. The model **cannot** push everything to zero — it is forced
to guess. This creates a strict zero-sum competition between species
representations.

**Gradient dynamics**: If the network confuses a rare bird with a dominant bird,
CE sends a massive negative gradient specifically to the weights that caused the
confusion. BCE, operating independently per class, allows the network to lazily
lower both probabilities instead of learning the boundary.

### The CE-Train / Sigmoid-Infer Paradigm

- **Training**: CE on primary label (single-class softmax). Forces discriminative,
  mutually exclusive feature learning. The model learns "which bird is loudest?"
- **Inference**: Strip softmax, apply sigmoid. Evaluates learned feature magnitudes
  independently. Because features are already highly discriminative from CE
  training, the raw logits naturally reflect species presence. Sigmoid allows
  multi-label predictions for soundscapes.

### Label Smoothing with Secondary Labels

When secondary labels are present, pipelines apply smoothing:
- Primary label weight: 0.5
- Remaining 0.5 distributed among secondary labels
- This preserves the CE competition while acknowledging multi-species presence

---

## 3. Waveform MixUp: Physics Matters

### Why Spectrogram MixUp Is Wrong

Mel spectrograms are **logarithmic** representations of acoustic power. Linear
averaging of two log-power matrices does NOT correspond to physical sound mixing.
It creates ghostly translucent artifacts where two calls overlap — the network
learns to identify these artificial patterns rather than true biological polyphony.

### Correct Implementation: Additive Waveform Mixing

Sound waves mix by **linear superposition** — literal addition of pressure waves.
Waveform MixUp adds raw audio arrays together:

```
mixed_waveform = waveform_A + waveform_B
```

Then compute mel spectrogram from the mixed waveform. The nonlinear mel transform
applies naturally and correctly to the physically mixed signal, producing realistic
spectrograms of overlapping species.

### Labels: Element-wise Maximum (Union)

Labels use `max(label_A, label_B)` — union of species present. NOT the standard
weighted blend `λ*y1 + (1-λ)*y2`. For soft pseudo-labels, element-wise sum
clipped to [0,1].

### SUMix: Energy-Scaled Mixing

Naive additive mixing has an amplitude masking problem: a loud bird obliterates
a quiet bird's signal. SUMix normalizes by RMS energy before mixing:

```
mixed = normalize(waveform_A) + normalize(waveform_B)
```

This ensures both species' acoustic signatures remain visible in the spectrogram.

---

## 4. Crop Selection: RMS Energy Thresholding

### The Problem with Random Crops

A 3-minute focal recording labeled "Social Flycatcher" where the bird calls for
2 seconds. A random 5s crop from seconds 40–45 contains zero target bioacoustics.
The network is trained to map rustling leaves → "Social Flycatcher." This
false-negative label noise cements the performance ceiling.

### RMS-Based Voice Activity Detection

Gemini advocates for **RMS energy-based crop selection** (contrasting with
Claude's finding that energy-based selection was rejected by competitors):

1. Divide audio into 1-second frames with overlap
2. Compute RMS energy for every frame
3. Set threshold based on distribution (e.g., +3dB above median)
4. Extract crops **centered around high-RMS peaks**
5. Every tensor passed to the network contains dense, relevant acoustic information

### High-RMS Rejection

Extremely high RMS values correspond to anthrophony (speech, vehicles),
microphone handling noise, or digital clipping. Dropping the top 20% loudest
data resulted in cleaner training. This is complementary to low-RMS filtering.

**Note**: This contradicts Claude's finding that BirdCLEF 2025 38th place
explicitly rejected RMS-based selection. The 2025 2nd place found random crops
from first/last 7 seconds more effective. This is an unresolved disagreement.

---

## 5. Salman's Pipeline Deconstructed

Salman's 0.922 single model / 0.932 ensemble achieved without external data,
pseudo-labels, or knowledge distillation:

| Component | Choice |
|-----------|--------|
| Architecture | Standard convolutional SED |
| Window | 20s (also tested 10s, 15s — similar scores) |
| Loss | CE (categorical) during training |
| Inference | Sigmoid activation (strip softmax) |
| Augmentation | Waveform MixUp (1D, before mel) |
| Epochs | 25 |
| Data | train_audio + labeled soundscapes only |
| Priors | Site/hour priors pushed 0.917 → 0.923 |
| Head | Attention-based (framewise max) |
| Ensemble | 4 models → 0.932 |

Key insight: **architectural simplicity with perfected data handling and loss
function**. The 0.903 barrier is a symptom of poor data handling, inappropriate
augmentations, and mathematically starved loss gradients.

---

## 6. Boredom's Pipeline

Achieved 0.929 LB using "Claude Code" agents with 1099 tested methods over 53
rounds. Confirmed that "a single EfficientNet model can easily achieve LB 925+."
This eliminates the hypothesis that novel architectures (ViTs, ASTs, SSMs) are
needed. The limiting factor is the pipeline, not the model.

Boredom's jump from 0.926 → 0.929 likely involves pseudo-labeling via
self-distillation on unlabeled soundscape data.

---

## 7. Post-Processing (Modest but Consistent Gains)

| Technique | Impact | Source |
|-----------|--------|--------|
| Min-ensemble reduction | +0.003 | BirdCLEF 2024 1st |
| Temporal prediction smoothing | +0.01 | BirdCLEF 2024 2nd/3rd/4th |
| Overlapping window inference | +0.01–0.015 | BirdCLEF 2024 1st/4th |
| Soundscape-level enhancement | +0.02 public | BirdCLEF 2024 3rd |
| Cut-off filtering | — | BirdCLEF 2024 4th |

---

## 8. The 0.90 → 0.93 Recipe (Gemini's Synthesis)

The progression from 0.90 to 0.93+ requires fixing all four systemic failures
simultaneously. Each individually provides a modest gain; together they compound:

1. **CE loss + sigmoid inference** — forces discriminative feature learning,
   addresses BCE gradient starvation under class imbalance
2. **Waveform MixUp (additive + max labels)** — physically correct augmentation,
   simulates real multi-species soundscapes
3. **Overlapping sliding window inference** — resolves train/inference window
   mismatch, provides boundary smoothing and variance reduction
4. **RMS energy-based crop selection** — eliminates false-negative label noise
   from silent crops (though this point is disputed by Claude's research)
5. **Iterative pseudo-labeling** — bridges focal→soundscape domain shift
6. **Metadata priors** — reduces false positives using ecological constraints

---

## 9. Key Disagreement with Claude's Research

| Topic | Claude | Gemini |
|-------|--------|--------|
| RMS energy cropping | Rejected by competitors; first/last 7s better | Advocated as essential VAD technique |
| AttHead length invariance | Should work but doesn't in practice | Should work due to no positional encoding |
| Window mismatch solution | Henkel reshape trick (backbone sees 5s) | Sliding window inference (backbone sees 20s) |
| Primary bottleneck | Waveform MixUp implementation + CE formulation | All four failures equally important |
