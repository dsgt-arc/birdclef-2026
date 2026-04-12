---
title: "Salman Ahmed — BirdCLEF 2025 26th Place (0.908)"
source: "Kaggle writeup + training/inference notebooks"
date: 2026-04-12
type: reference
tags: [birdclef-2025, rms-sampling, max-trick, model-soup, focal-bce, pseudo-labeling]
---

# Salman Ahmed — BirdCLEF 2025 26th Place Solution

**Source**: Kaggle writeup + `salmanahmedtamu/20th-place-training` + `salmanahmedtamu/fork-of-fork-of-less-smoothing-ensemble-eff-and-nf`
**Result**: Public 20th / Private 26th, LB 0.908 (BirdCLEF 2025)
**Also**: BirdCLEF 2024 baseline author that Mario Lasseck (2nd place) built upon. In BirdCLEF 2026, Salman claims single model 0.922 LB and ensemble 0.932 LB.

## Architecture Overview

### Stage 1 Models (single-model 0.865-0.872, ensemble 0.886)

**Two model types**: CNN (clip-level classifier) and SED (attention-based)

**CNN Model** (`TimmCNN`):
- Standard timm backbone → sigmoid → per-class probabilities
- Backbones used: `tf_efficientnet_b0_ns`, `tf_efficientnetv2_b0.in1k`, `tf_efficientnetv2_b1.in1k`, `eca_nfnet_l0`
- Input: **3-channel** mel spectrogram (expanded via `spec.expand(3, -1, -1)`)
- Resized to fixed image dimensions (e.g., 384x384)

**SED Model** (`SEDModel`):
- timm backbone with classification head removed
- **BN0**: BatchNorm2d on mel frequencies BEFORE the backbone (normalizes per-frequency-band)
- Mean pool over frequency → temporal smoothing (max_pool1d + avg_pool1d sum) → FC(in_features→in_features, ReLU) → dropout(0.5) → AttBlockV2
- **AttBlockV2**: `softmax(tanh(att_conv1d(x)))` attention × `sigmoid(cla_conv1d(x))` → sum over time
- Output: clipwise probabilities in [eps, 1-eps] (clamped sigmoid)
- Note: SED model was commented out in inference — not used in final submission

### The "Max Trick" (KEY TECHNIQUE)

This is Salman's most distinctive technique. During training:
1. Pick 15 seconds of audio from a recording using **RMS sampling** (not random)
2. Divide into **3 x 5-second segments**
3. Each segment → mel spectrogram → model prediction → 3 x (B, num_classes) logits
4. Take the **max** across the 3 segments: `logits, _ = torch.max(logits, dim=1)` → (B, num_classes)
5. Backpropagate on the max logits

```python
BS, K, C, H, W = spec.size()    # K=3 segments
spec = spec.view(BS * K, C, H, W)
logits = model(spec)
logits = logits.view(BS, K, 206)
logits, _ = torch.max(logits, dim=1)  # max across segments
```

**Why this works**: The bird is probably calling in at least ONE of the 3 segments. Max selects the segment with strongest evidence and ignores silent segments — effectively handling the weak-label problem without energy-based crop selection. The gradient flows only through the max-selected segment.

At inference: predictions are on individual 5-second segments (no max trick needed).

### RMS Sampling (Data-Centric)

Instead of random crops, Salman selects the 15-second window with the highest RMS energy:
```python
def combined_pick_rms_sample(waves, duration_samples):
    stride = 32000  # 1-second stride
    for start in range(0, len(waves) - duration_samples + 1, stride):
        window = waves[start:start + duration_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms > max_rms:
            max_rms = rms
            max_rms_start = start
    return waves[max_rms_start:max_rms_start + duration_samples]
```

This ensures the model always sees audio with actual acoustic content — not silence.

### Mel Spectrogram Parameters

Two configurations used:
```python
# Config 1 (primary, 384 mels)
{'sample_rate': 32000, 'n_mels': 384, 'f_min': 0, 'f_max': 16000,
 'n_fft': 3072, 'normalized': True, 'hop_length': 420}

# Config 2 (alternative, 448 mels)
{'sample_rate': 32000, 'n_mels': 448, 'f_min': 50, 'f_max': 16000,
 'n_fft': 4096, 'normalized': True, 'hop_length': 334}
```

**Critical**: `normalized=True` in torchaudio MelSpectrogram applies frame-level normalization to the STFT window. This is different from our NNAudio pipeline.

**Normalization chain**: `MelSpectrogram(normalized=True)` → `AmplitudeToDB(top_db=80)` → manual normalize: `(X + 80) / 80` → expand to 3 channels → resize to square image

The `(X + 80) / 80` normalization maps the [-80, 0] dB range to [0, 1]. This is DIFFERENT from our pipeline which uses global mean/std standardization.

### Loss Function: FocalBCE

Average of BCE and Focal loss:
```python
class FocalLossBCE:
    def forward(self, logits, targets):
        focal_loss = sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2)
        bce_loss = BCEWithLogitsLoss(logits, targets)
        return bce_weight * bce_loss + focal_weight * bce_loss  # default weights both 1.0
```

Note: default weights are both 1.0, so the loss is `BCE + Focal` (sum, not average). This differs from Mario's avg(BCE + Focal).

### Data Handling

**Human voice filtering**: Uses Silero VAD to detect speech in CSA (citizen science) recordings. For recordings with detected speech, the human audio segments are identified and:
- Non-human portions are extracted
- With 30% probability, human audio is randomly added back as augmentation with 50% probability

**Upsampling**: Species with fewer than 15 recordings are upsampled to 15 (with replacement)

**Duplicate removal**: `df.drop_duplicates(subset=['filename_base'])` — removes duplicates based on filename

**Secondary labels**: included during both training and validation (target value 1.0, treated equally to primary)

### Augmentations (on spectrograms, NOT waveforms)

- **MixUp** (p=1.0, alpha=1.0): standard interpolative lambda-weighted blend — NOT additive
- **LocalGlobalStretch**: resize random time/frequency regions of the spectrogram
- **SpecAugment**: FrequencyMasking(15, 2 masks) + TimeMasking(35, 2 masks)
- **GaussianNoise** on spectrogram
- **TimeFrequencyShift**: shift the spectrogram in time/freq dimensions

### Model Soup (Checkpoint Averaging)

Average weights from epochs 12-15 that improve any validation metric. Implementation:
- After each epoch, evaluate validation metrics (AUC, cMAP, mAP)
- If ANY metric improves, add that checkpoint to the soup
- Final model = average of all accepted checkpoint weights
- If adding a checkpoint hurts, revert to previous soup

### Training Configuration

```python
{
    "batch_size": 32,
    "n_epochs": 15,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "warmp_epoch": 3,
    "fold_n": 5,
    "model_name": "tf_efficientnet_b0_ns",
    "drop_rate": 0.5,
    "drop_path_rate": 0.2,
    "max_trick": True,
    "max_trick_factor": 3,  # 3 segments of 5s = 15s
    "base_duration": 5,
    "combined_sampling": True,  # RMS sampling
    "random_shuffle": False,
}
```

### Stage 2 (0.908 → 0.909 public, 0.908 private)

1. Generate predictions for all 5s chunks of ALL training audio
2. Filter: keep only segments where max prediction matches primary label
3. Generate pseudo-labels on these filtered segments
4. Train 4 CNNs with batch: 96 train audio segments + 5x12=60 soundscape segments
5. Fine-tune on just pseudo-labels for 7 more epochs

### Inference Pipeline

- **OpenVINO** for inference (not ONNX) — faster on CPU
- 4 CNN models (no SED in final submission): b0, v2_b0, v2_b1, eca_nfnet_l0
- Each model processes 5s chunks independently (no max trick at inference)
- **Temporal smoothing**: `smooth_array_general()`:
  - If max prediction <= 0.1: square all predictions (suppress low-confidence)
  - If max prediction >= 0.8: square all predictions (sharpen high-confidence)
  - Neighbor blending: `new[i] = 0.15 * pred[i-1] + 0.7 * pred[i] + 0.15 * pred[i+1]`
- Simple mean ensemble of 4 models

## Critical Differences from Our v5j Pipeline

| Feature | Our v5j | Salman 2025 |
|---------|---------|-------------|
| **Crop selection** | Fully random | **RMS sampling** (highest energy window) |
| **Window handling** | Single 5s random crop | **Max trick**: 15s RMS → 3x5s → max logits |
| **Input channels** | 1-channel mel | **3-channel** expanded mel |
| **Mel normalization** | Global mean/std standardization | **(X + 80) / 80** simple range mapping |
| **STFT normalization** | Not normalized | **normalized=True** (frame-level) |
| **Image resize** | No resize (256x511) | **Resize to square** (384x384) |
| **Loss** | Pure BCE | **BCE + Focal** |
| **Head** | LSEHead (no attention) | **AttBlockV2** (softmax-tanh attention x sigmoid) |
| **BN0** | None | **BatchNorm2d on mel freqs** before backbone |
| **Backbone** | HGNetV2-B0 | tf_efficientnet_b0_ns |
| **Checkpoint** | Single best | **Model soup** (epoch 12-15 weight averaging) |
| **Temporal post-proc** | Gaussian smoothing + taxa-aware | **Square + neighbor blend** |
| **Epochs** | 20 | 15 (with model soup) |
| **Dropout** | 1 layer (0.5) | **Drop path 0.2 + drop rate 0.5** |
| **Human filtering** | None | **Silero VAD on CSA recordings** |
| **Upsampling** | None | **Min 15 samples per species** |

## Most Impactful Differences (Hypothesis)

1. **Max trick + RMS sampling** — fundamentally different approach to weak labels. Instead of training on random (often silent) crops, selects loudest 15s and uses max-across-3-segments to find the bird.

2. **Mel normalization** — `(X+80)/80` is a simple [0,1] range map, while our global mean/std centers at 0 with variable range. The simple normalization may be more robust across domains.

3. **3-channel input + resize to square** — leverages ImageNet pretrained features better (3-channel RGB expected) and fixed square size matches ImageNet training resolution.

4. **BN0 on mel frequencies** — normalizes each mel band independently before the backbone. This could handle domain-specific amplitude differences between focal and soundscape recordings.

5. **BCE + Focal combined loss** — different from our pure BCE. The focal term downweights easy negatives, while BCE provides stable gradients. This is not the same as our failed ASL test.

6. **Model soup** — averaging multiple checkpoints may provide the smoothing effect we're missing.
