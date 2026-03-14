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

## Implementation Notes

1. **TF1 MetaGraph format**: Uses `tf.compat.v1` APIs. Model saved as `inference.meta` (graph definition) + `model.ckpt-*` (weights).
2. **No training code**: Only inference is released for the bird model. The [NeurIPS 2020 MixIT repo](https://github.com/google-research/sound-separation/tree/master/models/neurips2020_mixit) has training code for FUSS dataset, but not the bird-specific variant.
3. **No TDCN++ source code**: Architecture is frozen in the graph. Google has not released the TDCN++ implementation.
4. **Mixture consistency**: Built into the graph — separated sources always sum exactly to the input.
5. **Memory**: The model processes the entire input in one pass. For long recordings, you may need to chunk and process segments.

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
- `gsutil` (for checkpoint download)
- For PyTorch port: `asteroid` or `speechbrain`

## References

1. Denton, Wisdom, Hershey. "Improving Bird Classification with Unsupervised Sound Separation." ICASSP 2022. [arXiv:2110.03209](https://arxiv.org/abs/2110.03209)
2. Wisdom et al. "Unsupervised Sound Separation Using Mixture Invariant Training." NeurIPS 2020. [arXiv:2006.12701](https://arxiv.org/abs/2006.12701)
