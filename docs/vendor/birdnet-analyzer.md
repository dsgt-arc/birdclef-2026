---
name: "BirdNET-Analyzer"
source: "https://github.com/birdnet-team/BirdNET-Analyzer"
license: "MIT (code), CC BY-NC-SA 4.0 (models)"
category: "pretrained-model"
pipeline_stage:
  - "inference"
  - "embedding"
relevance: "Production-ready bird species classifier supporting 6500+ species"
status: "reference"
---

# BirdNET-Analyzer

## Overview

BirdNET-Analyzer is a deep learning solution for avian diversity monitoring developed by the Cornell Lab of Ornithology and Chemnitz University of Technology. It provides state-of-the-art bird species classification supporting over 6,500 species globally, with both GUI and CLI interfaces for processing audio data.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `birdnet_analyzer.model` | Core inference model and prediction logic | `birdnet_analyzer/model.py` |
| `birdnet_analyzer.audio` | Audio loading and preprocessing | `birdnet_analyzer/audio.py` |
| `birdnet_analyzer.embeddings` | Feature embedding extraction | `birdnet_analyzer/embeddings/` |
| `birdnet_analyzer.train` | Custom classifier training | `birdnet_analyzer/train/` |
| `birdnet_analyzer.cli` | Command-line interface | `birdnet_analyzer/cli.py` |

## Relevance to Pipeline

- **Inference**: Primary reference for production bird classification; our pipeline aims to achieve comparable accuracy with lower computational cost
- **Embedding**: Can extract 1024-dimensional embeddings for transfer learning or downstream tasks
- **Tokenization**: Not directly applicable - uses end-to-end neural approach rather than discrete tokens

## Usage

```python
from birdnet_analyzer import analyze

# Analyze audio files
results = analyze.analyze_file(
    audio_path="recording.wav",
    min_conf=0.1,
    sensitivity=1.0,
    overlap=0.0
)

# Extract embeddings
from birdnet_analyzer.embeddings import extract_embeddings
embeddings = extract_embeddings("recording.wav")
```

## Dependencies

- Python 3.12
- TensorFlow / TFLite
- NumPy, SciPy
- Librosa for audio processing

## Notes

- Models are CC BY-NC-SA 4.0 licensed; educational and research use is considered non-commercial
- Supports 16-bit PCM WAV at 48kHz sample rate natively
- Pre-trained models available on Zenodo
- Can be accessed via the bioacoustics-model-zoo wrapper for standardized API
