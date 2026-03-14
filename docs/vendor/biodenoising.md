---
name: "biodenoising"
source: "https://github.com/earthspecies/biodenoising"
license: "MIT (code), CC-BY-NC 4.0 (model weights)"
category: "utility"
pipeline_stage:
  - "data-processing"
  - "augmentation"
relevance: "Audio denoising for animal vocalizations; foreground/background decomposition for domain adaptation"
status: "active"
---

# Biodenoising

## Overview

Biodenoising is an audio denoising tool from the Earth Species Project, specifically designed for animal vocalizations. Based on the Demucs architecture (originally for music source separation), it produces a single denoised output — effectively decomposing audio into foreground (vocalization) and background (noise). The model uses a self-supervised approach with **pseudo-clean targets** generated from speech enhancement models, requiring no clean reference audio. Supports rapid domain adaptation to fine-tune for specific species or acoustic environments.

Paper: [Biodenoising: animal vocalization denoising without access to clean data](https://arxiv.org/abs/2410.03427) (2024)

## How It Works

### Pseudo-Clean Target Training

The key insight: **speech enhancement models contain useful priors** about temporal structure that transfer to animal vocalizations. The training pipeline:

1. **Generate pseudo-clean targets**: Apply a pre-trained speech enhancement model (Demucs dns48) to noisy animal recordings
2. **Time-scaling enhancement**: Slow audio by factors of 2, 3, and 4 (exploiting correlation between body size and pitch/tempo), denoise each, then average. This prevents the speech model from outputting silence on non-speech frequencies
3. **Synthetic mixture training**: Create (noisy vocalization + noise) → (pseudo-clean target) training pairs
4. **Post-processing**: Peak detection on RMS curves, 4-second windowed extraction, 50% random convolution with room impulse responses

**Critical result**: "The estimations obtained from training with pseudo-clean targets have better quality than the pseudo-clean targets themselves" — the model learns to generalize beyond its noisy supervision.

### Architecture Comparison (from paper)

| Model | Training Loss | SI-SDR (dB) | Notes |
|-------|--------------|-------------|-------|
| noisereduce (baseline) | N/A | 6.92 | Traditional spectral subtraction |
| Demucs + pseudo-clean (noisereduce) | L1 | **12.85** | Best overall |
| Demucs + pseudo-clean (demucs speech) | L1 | 12.48 | Competitive |
| Demucs + pseudo-clean (time-scaled) | L1 | 12.74 | Time-scaling helps |
| CleanUNet + pseudo-clean | L1 + multi-STFT | 7.48 | Complex attention overfits |

Demucs outperforms CleanUNet by **5.37 dB** — CleanUNet's attention mechanisms overfit on sparse animal signals.

### Training Data

- **120,688 noisy vocalizations** + **1,770,826 noise excerpts** (4-second each)
- Species: primates (gelada, macaques, gibbons, lemurs), birds, marine mammals (orcas, humpbacks, minke whales), seals
- Sources: Xeno-Canto, NOAA passive acoustic monitoring, multiple research institutions
- After filtering silent segments: 15,000 training excerpts; 30,000 with time-scaling augmentation
- Hardware: V100 GPU, Demucs batch size 32

### Evaluation

Benchmark: 23 species including quail, nightingale, gorilla, dolphin, whale, frog
- SI-SDR improvement: **~9 dB** consistent across SNR conditions (-5 to 10 dB)
- Robust to training-test SNR mismatch (train: 0-20 dB, test: -5 to 10 dB)

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `denoise.py` | Main denoising inference script | `denoise.py` |
| `adapt.py` | Domain adaptation/fine-tuning script | `adapt.py` |
| `biodenoising/denoiser/` | Core denoiser model and utilities | `biodenoising/denoiser/` |
| `biodenoising/datasets/` | Dataset handling and audio loading | `biodenoising/datasets/` |
| `selection_table.py` | Event-based processing with annotation tables | `biodenoising/selection_table.py` |
| `htdemucs.py` | HT-Demucs model implementation | `biodenoising/htdemucs.py` |

## Relevance to Pipeline

- **Foreground/background decomposition**: Subtract denoised output from original to obtain the background noise layer. This gives a 2-channel decomposition (vocalization + noise) that can be remixed at different SNRs for domain adaptation.
- **Domain adaptation to soundscapes**: Fine-tune the denoiser on BirdCLEF soundscape data (5-10 min of target audio) to handle domain-specific noise (wind, rain, anthropogenic sounds, insect choruses).
- **Noise extraction for augmentation**: Apply to soundscape recordings to extract clean backgrounds. Mix these backgrounds with focal bird recordings to simulate soundscape conditions.
- **Event-aware processing**: Use selection tables to focus denoising on annotated bird vocalizations while extracting inter-event noise separately.
- **Complementary to Bird MixIT**: Biodenoising gives 1 clean output (foreground vs background split). MixIT gives 4-8 separated sources (individual bird voices). Use biodenoising for noise removal, MixIT for multi-source decomposition.

## Usage

```bash
# Basic denoising
biodenoise \
  --method biodenoising16k_dns48 \
  --noisy_dir /path/to/noisy_audio \
  --out_dir /path/to/output \
  --device cuda

# Domain adaptation (fine-tune on your data)
biodenoise-adapt \
  --method biodenoising16k_dns48 \
  --noisy_dir /path/to/domain_audio \
  --out_dir /path/to/adapted_model \
  --steps 3 \
  --epochs 10 \
  --device cuda

# With selection tables for event-based processing
biodenoise \
  --method biodenoising16k_dns48 \
  --noisy_dir /path/to/audio \
  --out_dir /path/to/output \
  --selection_table
```

```python
# Programmatic usage via CLI wrapper (recommended)
# The biodenoise CLI handles sample rate conversion, chunking, and batching internally.
# For programmatic use, the load_and_denoise function is the main entry point:
from biodenoising.denoiser.denoiser import load_and_denoise

# For foreground/background decomposition (conceptual -- requires matching sample rates):
import torchaudio

noisy, sr = torchaudio.load("recording.wav")
# IMPORTANT: biodenoising operates at 16 kHz internally.
# If input is 32 kHz (BirdCLEF), you must resample before/after:
noisy_16k = torchaudio.functional.resample(noisy, sr, 16000)
# Run denoising (via CLI or load_and_denoise for full pipeline)
# denoised_16k = ...  # output from biodenoising at 16 kHz
# background_16k = noisy_16k - denoised_16k

# Remix at different SNR for augmentation
# augmented = denoised_16k + 0.3 * background_16k  # 0.3 = ~-10 dB gain
```

### Domain Adaptation Details

```bash
biodenoise-adapt \
  --method biodenoising16k_dns48 \
  --noisy_dir /path/to/soundscape_audio \  # 5-10 min of target domain
  --out_dir /path/to/adapted_model \
  --steps 5 \           # adaptation passes (default 5)
  --epochs 5 \          # training epochs per step (default 5)
  --highpass 20 \       # frequency filtering (Hz)
  --device cuda
```

- Single-step teacher-student: pre-trained dns48 generates pseudo-clean targets, then model fine-tunes
- Starting from random weights degrades by 1.2-2.15 dB — speech priors are essential
- `--interactive` mode enables manual quality control during adaptation

## Dependencies

- torch >= 1.5 (on PACE with CUDA 12.1, use torch 2.x compiled for cu121)
- torchaudio >= 0.5
- librosa
- asteroid (audio source separation toolkit)
- openunmix
- noisereduce
- omegaconf == 1.4.1 (**conflicts with project-level omegaconf>=2.0 and EAT's 2.0.6 -- requires isolated venv**)
- einops

## Comparison with Bird MixIT

| Feature | Biodenoising | Bird MixIT |
|---------|-------------|-----------|
| **Task** | Denoising (1 output) | Source separation (4-8 outputs) |
| **Architecture** | Demucs dns48 | TDCN++ |
| **Training** | Distillation-based (pseudo-clean targets from speech model) | Unsupervised (MoM) |
| **Sample rate** | 16 kHz | 22.05 kHz |
| **Domain** | General animal vocalizations | Bird-specific (Xeno-Canto) |
| **Domain adaptation** | Yes (5-10 min, fast) | No (requires retraining) |
| **Framework** | PyTorch | TensorFlow 1.x compat |
| **License** | MIT (code), CC-BY-NC (weights) | Apache 2.0 |
| **Best for** | Noise removal, SNR improvement | Multi-source decomposition, remixing |

**Use together**: Biodenoising for foreground/background split. MixIT for separating individual bird voices within the foreground. The combination enables fine-grained source manipulation for augmentation.

## Limitations

- **16 kHz default**: Loses high-frequency detail above 8 kHz (use `--keep_original_sr` for bats, belugas, insects)
- **CC-BY-NC 4.0 weights**: Non-commercial use only for pre-trained weights (code is MIT)
- **General model, not bird-specific**: May underperform vs Bird MixIT for bird-only denoising, but handles multi-taxa (birds, amphibians, insects, mammals) that BirdCLEF+ 2026 requires
- **Single output**: Cannot separate overlapping bird species — use MixIT for that
- **CleanUNet artifacts**: The alternative CleanUNet architecture produced "blabber-like speech" artifacts with time-scaling; only Demucs variant is recommended
- The paper notes: "Biodenoising is a generic tool that may fail in some cases" — domain adaptation is essential for specialized environments

## Notes

- Default model operates at 16 kHz; use `--keep_original_sr` for high-frequency vocalizations (bats, belugas, insects)
- Domain adaptation typically needs 5-10 minutes of target domain audio
- Selection tables support CSV/TSV/TXT formats with auto-detected start/end columns
- The MixIT paper tried applying unsupervised separation (MixIT-style) to biodenoising but found "the heuristics necessary...did not generalize well" — pseudo-clean target approach won
- arXiv paper: https://arxiv.org/abs/2410.03427
- Web demo: https://earthspecies.github.io/biodenoising/
- See also: [Bird MixIT](bird-mixit.md) for multi-source separation
