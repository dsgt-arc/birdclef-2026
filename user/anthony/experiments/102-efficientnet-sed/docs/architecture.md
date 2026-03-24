# Model Architecture

## Input

A 5-second audio clip at 32kHz (160,000 samples). The data module loads a
focal recording (which can be 5–130s long), splits it into 5s chunks, and
feeds each chunk independently. Each chunk has a clip-level species label
inherited from the recording.

## Pipeline

### 1. Mel spectrogram (nnAudio, on GPU)

```
raw waveform (B, 160000)
    → nnAudio MelSpectrogram(sr=32000, n_fft=2048, hop=256, n_mels=256)
    → (B, 256, 625)          ← 256 mel frequency bins, 625 time frames
    → log + global normalize  ← hardcoded mean/std from BirdSet pretrained model
    → (B, 1, 256, 625)       ← add channel dim for CNN input
```

nnAudio runs as a PyTorch module on GPU — no pre-computed spectrograms.
The normalization uses **global constants** (mean=-4.268, std=4.569), not
per-clip statistics. This means the model sees absolute loudness differences
between clips.

### 2. SpecAugment (training only)

Random frequency and time masking applied to the mel spectrogram:
- 2 frequency masks of up to 40 mel bins each
- 2 time masks of up to 50 frames each

Standard regularization for audio models. Disabled during validation/inference.

### 3. EfficientNet-B1 backbone

```
(B, 1, 256, 625)
    → EfficientNet-B1 (pretrained on BirdSet/XenoCanto bird audio)
    → (B, 1280, 8, 20)       ← 1280 feature channels, 8×20 spatial grid
```

The backbone is a CNN that progressively downsamples the spectrogram through
MBConv blocks. The 256×625 input becomes an 8×20 feature map — roughly
8 frequency bands × 20 time steps (each time step ≈ 250ms).

The model is `DBD-research-group/EfficientNet-B1-BirdSet-XCL` from
HuggingFace — pretrained on XenoCanto bird recordings, not ImageNet. So
it already knows spectral patterns of bird vocalizations.

With LoRA: backbone weights are frozen, small trainable adapter matrices
(rank 8) are injected into projection layers. Only the adapters + head train.

### 4. Pooling

```
(B, 1280, 8, 20)
    → adaptive_avg_pool2d(output_size=1)
    → (B, 1280)
```

**Why pooling?** The backbone outputs a 2D feature map (8 freq × 20 time),
but we need a single vector per clip for classification. Adaptive average
pooling takes the mean over all spatial positions, collapsing the 8×20 grid
into a single 1280-dim vector.

**The tradeoff**: averaging over the 20 time steps means a brief 0.5s call
in an otherwise silent 5s clip gets diluted by 19 silent time steps. This is
the "GAP bottleneck" that Murilo's exp 015 identified — with global average
pooling, pseudo-labeled soundscape data didn't improve the model because the
pooling smears the signal.

**Alternatives** (not implemented yet):
- Attention pooling: learn which time steps to weight more heavily
- Framewise max: take the max across time (catches brief events)
- Activity-weighted pooling: use pre-computed spectral contrast to weight frames

These are potential improvements for later runs. The current architecture uses
simple average pooling to match the exp 012 baseline.

### 5. Classification head

```
(B, 1280)
    → Dropout(0.3)
    → Linear(1280, num_classes)
    → (B, num_classes)         ← raw logits, not probabilities
```

The output is raw logits. The training loss (focal BCE) applies sigmoid
internally. At inference, apply sigmoid to get per-species probabilities.

## Loss

Focal BCE with gamma=2.0. Each of the 206 species is an independent binary
classification. Focal loss downweights easy negatives (the vast majority of
species are absent in any given clip) and focuses learning on hard cases.

## Diagram

```
5s audio @ 32kHz
    │
    ▼
┌──────────────┐
│  nnAudio Mel │ (on GPU, not preprocessing)
│  256 bins    │
│  625 frames  │
└──────┬───────┘
       │ (B, 1, 256, 625)
       ▼
┌──────────────┐
│  SpecAugment │ (train only)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ EfficientNet │ pretrained BirdSet/XenoCanto
│    -B1       │ frozen weights + LoRA adapters
│              │
└──────┬───────┘
       │ (B, 1280, 8, 20)
       ▼
┌──────────────┐
│  Avg Pool    │ collapse 8×20 → single vector
└──────┬───────┘
       │ (B, 1280)
       ▼
┌──────────────┐
│  Dropout     │
│  + Linear    │ 1280 → 206 species
└──────┬───────┘
       │ (B, 206)
       ▼
  focal BCE loss (training)
  sigmoid → probabilities (inference)
```
