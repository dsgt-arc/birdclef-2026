---
name: "Bird MixIT (Unsupervised Bird Sound Separation)"
source: "https://github.com/google-research/sound-separation/tree/master/models/bird_mixit"
license: "Apache 2.0"
category: "pretrained-model"
pipeline_stage:
  - "data-processing"
  - "inference"
relevance: "Unsupervised source separation of overlapping bird calls; potential preprocessing for classifiers and pseudo-label denoising"
status: "reference"
---

# Bird MixIT (Unsupervised Bird Sound Separation)

**Paper analysis**: See [docs/references/bird-mixit-paper.md](../references/bird-mixit-paper.md) for the full method deep-dive, equations, experimental results, and relevance to our BirdCLEF pipeline.

**GitHub**: [google-research/sound-separation](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit)

## Overview

Bird MixIT is a pretrained unsupervised sound separation model for birdsong, trained with Mixture Invariant Training (MixIT) on Xeno-Canto + Macaulay Library data (704 species). Given an input audio waveform, it outputs M separated source waveforms (M=4 or M=8) that sum to the original input. The model does not require clean source audio for training — it learns purely from noisy field recordings.

The released implementation is **TensorFlow 1.x only** (inference-only, no training code).

The bird-specific model achieves 10.5 dB SI-SNR improvement on bird audio -- 6.1 dB higher than the 4.4 dB achieved by a generic AudioSet-trained model. (Note: SI-SNR and SI-SDR are mathematically equivalent metrics.)

**Key papers:**
- Wisdom et al., "Unsupervised Sound Separation Using Mixture Invariant Training," NeurIPS 2020. [arXiv:2006.12701](https://arxiv.org/abs/2006.12701)
- Denton, Wisdom & Hershey, "Improving Bird Classification with Unsupervised Sound Separation," ICASSP 2022. [arXiv:2110.03209](https://arxiv.org/abs/2110.03209)

## How MixIT Works

### Core Concept: Mixture-of-Mixtures Training

Traditional supervised source separation requires clean isolated source recordings as training targets -- impractical for wildlife audio. MixIT solves this by training on **mixtures of mixtures (MoMs)**:

1. **Sample two field recordings** x1 and x2 from the training set (each already a mixture of bird calls + noise)
2. **Add them together** to form a MoM: x_bar = x1 + x2
3. **Feed x_bar through the separation model** to predict M source estimates: s_hat = f(x_bar)
4. **Assign each estimated source to one of the two original mixtures** via a binary mixing matrix A (each column sums to 1)
5. **Minimize reconstruction loss**: the sum of assigned sources should reconstruct each original mixture

The assignment matrix A is optimized via exhaustive O(2^M) search over all possible binary assignments (feasible for M <= 8). The loss function uses **negative thresholded SNR**:

```
L(y, y_hat) = -10 * log10( ||y||^2 / (||y - y_hat||^2 + tau * ||y||^2) )
```

where tau = 10^(-SNR_max/10) with SNR_max = 30 dB. This soft threshold prevents already well-separated examples from dominating gradients.

**Key insight**: If sources are independent across the two mixtures, the model cannot determine which sources came from which mixture. The only way to minimize the loss is to **actually separate all sources**, since any under-separation would leave residual cross-mixture leakage in the reconstructions.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `process_wav.py` | CLI tool for separating a single WAV file | `models/tools/process_wav.py` |
| `inference.py` | `SeparationModel` class with `.separate()` method | `models/tools/inference.py` |
| `inference.meta` + checkpoint | TF1 MetaGraph + weights (frozen architecture) | GCS bucket (see below) |

## Architecture Summary

```
Input waveform (22.05 kHz mono)
    ↓
Learnable analysis basis (Conv1D, 1ms window, 0.5ms hop, 256 coefficients)
    ↓
TDCN++ (Improved Time-Domain Convolutional Network)
    ↓
M sigmoid masks
    ↓
Mask × analysis coefficients → Learnable synthesis basis (overlap-add)
    ↓
Mixture consistency projection (sources constrained to sum to input)
    ↓
M source waveforms
```

The model is masking-based with a learned filterbank (not STFT). Architecture is baked into the frozen graph — TDCN++ source code is not released.

The TDCN++ network processes the basis coefficients and predicts M sigmoid masks (one per output source), based on Conv-TasNet with architectural improvements from Kavalerov et al. (WASPAA 2019). A final mixture consistency projection constrains the separated sources to sum to the original input, eliminating artifacts.

The model can produce **fewer than M active sources** by generating near-zero signals for unused output channels. This handles the variable number of sources in real recordings.

### Training Details

- **Hardware**: 4 Google Cloud TPUs (16 chips)
- **Optimizer**: Adam, learning rate 1e-3
- **Batch size**: 256
- **Loss**: Negative thresholded SNR with SNR_max = 30 dB
- **Training data**: Up to 250 recordings per species from Xeno-Canto and Macaulay Library, 6-second windows at 22.05 kHz
- **Activity detection**: wavelet peak detector (scipy `find_peaks_cwt`) selects up to 5 high-energy windows per recording

## Separation Performance

Evaluated on 3,856 MoMs from held-out Xeno-Canto data using the MoMi score (mean SI-SNR improvement over estimated mixtures):

| Train Data  | Input SR   | Eval SR   | Window | MoMi     |
|-------------|-----------|-----------|--------|----------|
| AudioSet    | 16 kHz    | 16 kHz    | 2.5 ms | **4.4 dB**  |
| Xeno-Canto  | 22.05 kHz | 16 kHz    | 2.5 ms | 9.6 dB   |
| Xeno-Canto  | 22.05 kHz | 16 kHz    | 1.0 ms | 10.4 dB  |
| Xeno-Canto  | 22.05 kHz | 22.05 kHz | 1.0 ms | **10.5 dB** |

Key findings:
- Domain-matched training provides **>5 dB boost** over generic AudioSet model
- Smaller analysis window (1.0 ms vs 2.5 ms) adds ~1 dB
- Higher sample rate (22.05 kHz vs 16 kHz) provides marginal additional gain

### Downstream Classification Improvement

The Separate+Classify pipeline applies separation, then classifies each separated channel plus the original audio, taking the **max probability per species** across all channels:

**Sapsucker Woods (SSW)** -- 70 species, 40K segments:

| Method          | CMAP  | lwlrap | d'    | Top-1 |
|-----------------|-------|--------|-------|-------|
| Mix Only        | 0.304 | 0.431  | 1.117 | 0.398 |
| Separation Only | 0.268 | 0.413  | 1.116 | 0.360 |
| Mix+Separation  | **0.306** | **0.441** | **1.123** | 0.397 |

**High Sierras (HSN)** -- 18 species, 4,928 segments:

| Method          | CMAP  | lwlrap | d'    | Top-1 |
|-----------------|-------|--------|-------|-------|
| Mix Only        | 0.527 | 0.531  | 1.149 | 0.432 |
| Separation Only | 0.548 | 0.548  | 1.149 | 0.448 |
| Mix+Separation  | **0.560** | **0.560** | **1.153** | **0.451** |

**Caples (CAP)** -- 42 species, 2,944 segments:

| Method          | CMAP  | lwlrap | d'    | Top-1 |
|-----------------|-------|--------|-------|-------|
| Mix Only        | 0.334 | 0.569  | 1.144 | 0.496 |
| Separation Only | 0.327 | 0.581  | 1.154 | 0.506 |
| Mix+Separation  | **0.341** | **0.590** | **1.155** | **0.517** |

**Critical finding**: Including the original mixture as an additional channel always outperforms separation alone. Separation-only can actually **hurt** CMAP (SSW: 0.268 vs 0.304).

## Pre-trained Checkpoints

Download via Google Cloud Storage:

```bash
gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .
```

| Model | Sources | Checkpoint | Notes |
|-------|---------|------------|-------|
| 4-source | 4 | `output_sources4/model.ckpt-3223090` | Used in ICASSP paper |
| 8-source | 8 | `output_sources8/model.ckpt-2178900` | Higher capacity, more channels |

Both require **22.05 kHz mono** input. Unused channels output near-zero signals (M is an upper bound).

## Usage

### CLI

```bash
# 4-source separation
python3 process_wav.py \
  --model_dir bird_mixit_model_checkpoints/output_sources4 \
  --checkpoint bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090 \
  --num_sources 4 \
  --input recording.wav \
  --output separated.wav
# Produces: separated_source0.wav through separated_source3.wav

# 8-source separation
python3 process_wav.py \
  --model_dir bird_mixit_model_checkpoints/output_sources8 \
  --checkpoint bird_mixit_model_checkpoints/output_sources8/model.ckpt-2178900 \
  --num_sources 8 \
  --input recording.wav \
  --output separated.wav
# Produces: separated_source0.wav through separated_source7.wav
```

### Python API

```python
from inference import SeparationModel

model = SeparationModel(
    checkpoint_path="bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090",
    metagraph_path="bird_mixit_model_checkpoints/output_sources4/inference.meta",
    input_tensor_name="input_audio/receiver_audio:0",
    output_tensor_name="denoised_waveforms:0",
)

# Input shapes: (num_samples,) or (num_mics, num_samples) or (1, num_mics, num_samples)
# Output shape: (num_sources, num_samples)
sources = model.separate(mixture_waveform)  # numpy array at 22.05 kHz
```

The `SeparationModel` class wraps a TF1 session. It loads the MetaGraph, restores the checkpoint, and runs a single `sess.run()` call for inference.

### bioacoustics-model-zoo Integration

The [bioacoustics-model-zoo](https://github.com/kitzeslab/bioacoustics-model-zoo) provides a Python wrapper:

```python
import bioacoustics_model_zoo as bmz

# Initialize with 4-source checkpoint
model = bmz.SeparationModel(
    checkpoint='bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090'
)

# Separate an opensoundscape Audio object
# Works best on 5-second segments
separated_channels = model.separate_audio(audio_object)

# Or process files directly (writes _stem0.wav, _stem1.wav, ...)
model.load_separate_write(input_path, output_dir)
```

**Dependencies for bioacoustics-model-zoo**: `tensorflow`, `tensorflow_hub`, `opensoundscape`

## Implementation Notes

1. **TF1 MetaGraph format**: Uses `tf.compat.v1` APIs. Model saved as `inference.meta` (graph definition) + `model.ckpt-*` (weights).
2. **No training code**: Only inference is released for the bird model. The [NeurIPS 2020 MixIT repo](https://github.com/google-research/sound-separation/tree/master/models/neurips2020_mixit) has training code for FUSS dataset, but not the bird-specific variant.
3. **No TDCN++ source code**: Architecture is frozen in the graph. Google has not released the TDCN++ implementation.
4. **Mixture consistency**: Built into the graph — separated sources always sum exactly to the input.
5. **Memory**: The model processes the entire input in one pass. For long recordings, you may need to chunk and process segments.

## Limitations and Failure Modes

### Known Issues (from the paper)

1. **Prominent species probability decreases after separation**: The most dominant species in a recording often gets lower classifier confidence after separation, likely due to loss of contextual audio information. Mitigated by including the original mixture as an extra channel.

2. **Separation-only classification degrades performance**: Using only separated channels (without the original) consistently underperforms the original mixture on CMAP. This may be due to over-separation, domain shift between separated and original audio, or loss of audio context.

3. **Over-separation of vocalizations**: The model sometimes splits a single bird's song across multiple channels, isolating individual notes. These isolated notes can resemble calls of a different species, causing misclassification.

4. **Co-occurrence dependencies**: MixIT assumes sources across the two input mixtures are independent. When sources have strong co-occurrence patterns (e.g., specific species always calling together), the model may avoid separating them, treating them as a single source.

5. **Fixed number of output channels**: The model always produces M outputs even when fewer sources exist. Inactive channels produce near-zero signals but are not explicitly identified.

### What It Separates Well

- Isolated vocalizations with low SNR (background noise removal)
- Overlapping vocalizations from different species, even with overlapping frequency ranges
- Common species that appear as unlabeled background in focal recordings
- Dawn chorus scenes with multiple simultaneous vocalizers

### What It Struggles With

- Very similar vocalizations from closely related species
- Continuous broadband noise that overlaps spectrally with calls
- Very long recordings (memory scales with audio length, no built-in chunking)
- Insect/amphibian choruses (model trained primarily on bird data)

## PyTorch Porting Considerations

### Option A: Inference-only (no port needed)

- Run the TF1 model as a preprocessing step via `tensorflow.compat.v1`
- Or convert frozen graph to ONNX via [tf2onnx](https://github.com/onnx/tensorflow-onnx), run via `onnxruntime`
- Pros: immediate use, no reimplementation
- Cons: TF1 dependency, no gradient flow, can't fine-tune

### Option B: PyTorch reimplementation

Would need to build:

| Component | PyTorch equivalent | Reference |
|-----------|-------------------|-----------|
| Learnable basis (encoder/decoder) | Conv1D + ConvTranspose1D | Conv-TasNet (Luo & Mesgarani, 2019) |
| TDCN++ separator | Stacked dilated conv blocks with skip connections | [Asteroid library](https://github.com/asteroid-team/asteroid) has TDCN/ConvTasNet |
| Sigmoid masks | `torch.sigmoid(...)` | Trivial |
| Mixture consistency | Project sources to sum to input | `sources -= (sources.sum(0) - mixture) / M` |
| MixIT loss | Binary assignment search over 2^M | M=4 → only 16 assignments (brute-force) |

**Key resources for PyTorch implementation**:
- [Asteroid](https://github.com/asteroid-team/asteroid) — mature PyTorch audio separation library with ConvTasNet, DPRNN, etc. Has MixIT-compatible losses.
- [SpeechBrain](https://github.com/speechbrain/speechbrain) — has separation recipes including Conv-TasNet.
- Weight transfer from TF1 → PyTorch is possible but requires mapping layer names between the frozen graph and the PyTorch model.

### Option C: Train from scratch in PyTorch

Use Asteroid's ConvTasNet or TDCN++ implementation with MixIT loss, trained on BirdCLEF 2025 focal recordings. The paper showed **+5 dB** from domain-matched training (Xeno-Canto vs AudioSet), so BirdCLEF-specific training could further improve separation quality for our species set.

## Comparison with Biodenoising (Demucs)

| Feature | Bird MixIT | Biodenoising |
|---------|-----------|--------------|
| **Architecture** | TDCN++ (time-domain convnet) | HT-Demucs (hybrid transformer) |
| **Training** | Unsupervised (MixIT, no clean refs) | Self-supervised (pseudo-clean targets) |
| **Task** | Source separation (M channels) | Denoising (1 clean output) |
| **Sample rate** | 22.05 kHz | 16 kHz (default) |
| **Domain** | Bird-specific (Xeno-Canto) | General animal vocalizations |
| **Outputs** | 4 or 8 separated sources | Single denoised waveform |
| **Domain adaptation** | Retrain from scratch on new data | Fine-tune with `biodenoise-adapt` (minutes) |
| **License** | Apache 2.0 | MIT (code), CC-BY-NC 4.0 (weights) |
| **Framework** | TensorFlow 1.x compat | PyTorch |

**Key differences**:
- Bird MixIT produces **multiple separated channels** that can be individually classified, enabling the max-channel aggregation strategy. Biodenoising produces a single denoised output.
- Biodenoising supports **quick domain adaptation** (5-10 min of target audio, a few minutes of training). Bird MixIT requires full retraining for new domains.
- Bird MixIT uses the legacy **TensorFlow 1.x** graph API, while biodenoising uses modern **PyTorch**, making it easier to integrate into typical training pipelines.
- Biodenoising is designed for **general animal vocalizations** (birds, whales, bats, insects), while Bird MixIT is optimized for birds specifically.

## Computational Cost

- **Training**: 4 Google Cloud TPUs (16 chips), batch size 256 (from original MixIT paper)
- **Training the classifier**: 3.5 hours on 32 v2 TPUs per model, 200K steps (5-model ensemble)
- **Inference**: Single forward pass through TF graph; no explicit chunking. Memory scales linearly with audio length.
- **GPU**: Works on any TensorFlow-compatible GPU; no specialized CUDA kernels required
- **CPU inference**: Possible but slow for long recordings

## BirdCLEF Competition Usage

### Direct Usage in Competitions

Bird MixIT has not been widely adopted in top BirdCLEF competition solutions. The 2025 winning solutions relied primarily on:
- EfficientNet/RegNet/NFNet backbones with SED heads
- Iterative pseudo-labeling (3-4 rounds)
- Temporal smoothing and power adjustment at inference

The separation-then-classify pipeline adds computational cost at inference time (running separation + classification on M+1 channels), which is a concern under Kaggle's runtime constraints.

### Potential Applications for BirdCLEF 2026

1. **Training data augmentation**: Separate focal recordings to isolate the target species, creating cleaner training examples. Use other channels as negative/background examples.
2. **Multi-label generation**: Run separation + classification on unlabeled soundscapes to generate pseudo-labels for background species.
3. **Acoustic index improvement**: Separate bird/insect/noise into distinct channels before computing acoustic indices (NDSI, BI, etc.) for more meaningful features.
4. **Targeted denoising**: Apply separation to soundscape evaluation data at inference time, classify each channel, and aggregate predictions.

## Relevance to Pipeline

| Use Case | Description |
|----------|-------------|
| **Inference preprocessing** | Separate soundscapes → classify each channel + original → take max per species. Paper's recommended approach. |
| **Pseudo-label denoising** | Run MixIT on 9,726 unlabeled soundscapes before teacher ensemble inference for cleaner pseudo-labels. |
| **Training augmentation** | Separate focal recordings → remix sources at different SNRs → synthetic multi-label training data. |
| **Domain-matched retraining** | Train new MixIT model on BirdCLEF 2025 data for better domain match than the released Xeno-Canto model. |

### Relationship to other tools

| Tool | Relationship |
|------|-------------|
| [Biodenoising](biodenoising.md) | Denoising (noise removal) vs. separation (isolates individual sources). Complementary. |
| [BamBird/scikit-maad](bambird-scikit-maad.md) | BamBird gives temporal ROIs; MixIT gives actual waveform-level source separation within those segments. |
| Teacher ensemble ([EAT](eat.md), ConvNeXt, [ProtoCLR](protoclr.md)) | MixIT preprocesses audio before teacher inference for cleaner predictions. |

## Dependencies

- `tensorflow` (1.x or `tensorflow.compat.v1`)
- `numpy`
- `scipy` (for WAV I/O)
- `gsutil` (for checkpoint download)
- For PyTorch port: `asteroid` or `speechbrain`

## Notes

- The 22.05 kHz sample rate is unusual (most audio ML uses 16 kHz or 32 kHz). Ensure resampling matches when integrating with other models.
- The TF1 compat mode requirement may conflict with modern PyTorch-based pipelines. Consider running separation as a preprocessing step in a separate environment.
- The 4-source model is recommended over the 8-source model unless dealing with very dense soundscapes (dawn chorus).
- For BirdCLEF pipeline integration, the **max-channel strategy** (max probability over separated channels + original) is essential -- never use separation-only.
- The model processes full audio files in one pass. For long recordings, manual chunking with overlap (e.g., 5-10 second windows) is recommended to manage memory.
- The bioacoustics-model-zoo reports that the model "works best on 5 second segments."
- Audio demos: https://bird-mixit.github.io
- Code repository: https://github.com/google-research/sound-separation/tree/master/models/bird_mixit
- See also: [Biodenoising](biodenoising.md) for single-output denoising, [Source Separation Augmentation concept](../concepts/source-separation-augmentation.md)

## References

1. Denton, Wisdom, Hershey. "Improving Bird Classification with Unsupervised Sound Separation." ICASSP 2022. [arXiv:2110.03209](https://arxiv.org/abs/2110.03209)
2. Wisdom et al. "Unsupervised Sound Separation Using Mixture Invariant Training." NeurIPS 2020. [arXiv:2006.12701](https://arxiv.org/abs/2006.12701)
