---
name: "biodenoising"
source: "https://github.com/earthspecies/biodenoising"
license: "MIT (code), CC-BY-NC 4.0 (model weights)"
category: "utility"
pipeline_stage:
  - "data-processing"
relevance: "Audio denoising for animal vocalizations without requiring clean training data"
status: "active"
---

# Biodenoising

## Overview

Biodenoising is an audio denoising tool specifically designed for animal vocalizations. Based on the Demucs architecture (originally for music source separation), it can denoise wildlife recordings without requiring access to clean reference data. The model uses a self-supervised approach with pseudo-clean targets and supports domain adaptation to fine-tune for specific species or acoustic environments.

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

- **Data Processing**: Clean noisy field recordings before feature extraction or training. Improved SNR can benefit both discrete tokenization and downstream classification.
- **Domain Adaptation**: Fine-tune the denoiser on BirdCLEF soundscape data to handle domain-specific noise characteristics (wind, rain, anthropogenic sounds).
- **Event-Aware Processing**: Use selection tables to focus denoising on annotated bird vocalizations while extracting inter-event noise for data augmentation.

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
# Programmatic usage
from biodenoising.denoiser import pretrained
from biodenoising.denoiser.denoiser import load_and_denoise

# Load pretrained model
model = pretrained.dns48()

# Or use the full inference pipeline
# load_and_denoise handles file I/O, chunking, and GPU batching
```

## Dependencies

- torch >= 1.5
- torchaudio >= 0.5
- librosa
- asteroid (audio source separation toolkit)
- openunmix
- noisereduce
- omegaconf == 1.4.1
- einops

## Notes

- Default model operates at 16kHz; use `--keep_original_sr` for high-frequency vocalizations (bats, belugas)
- Model weights are CC-BY-NC 4.0 (non-commercial use only); code is MIT licensed
- Domain adaptation typically needs 5-10 minutes of target domain audio
- Selection tables support CSV/TSV/TXT formats with auto-detected start/end columns
- arXiv paper: https://arxiv.org/abs/2410.03427
- Web demo: https://earthspecies.github.io/biodenoising/
