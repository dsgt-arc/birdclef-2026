---
name: "hubert"
source: "https://github.com/bshall/hubert"
license: "MIT"
category: "self-supervised"
pipeline_stage:
  - "tokenization"
  - "representation-learning"
relevance: "Discrete and soft speech units via iterative clustering for voice conversion"
status: "reference"
---

# HuBERT

## Overview

This repository provides training and inference scripts for HuBERT content encoders, focused on discrete vs soft speech unit comparison for voice conversion. HuBERT (Hidden-Unit BERT) learns speech representations by predicting cluster assignments of masked audio segments, using offline k-means clustering to generate targets. The approach enables both discrete tokenization and soft unit extraction.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `hubert_soft` | Soft speech unit encoder | via `torch.hub` |
| `hubert_discrete` | Discrete speech unit encoder | via `torch.hub` |
| `encode.py` | Batch encoding script | `encode.py` |
| `train.py` | Training script for HuBERT-Soft | `train.py` |

## Relevance to Pipeline

- **Tokenization**: HuBERT-Discrete provides discrete token extraction via k-means clustering
- **Representation Learning**: Core reference for masked prediction with iterative clustering targets; Phase 2 baseline approach
- **Embedding**: HuBERT-Soft provides continuous embeddings that predict discrete units

## Usage

```python
import torch
import torchaudio

# Load pre-trained model from torch.hub
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
hubert = hubert.cuda()

# Load audio (must be 16kHz)
wav, sr = torchaudio.load("audio.wav")
assert sr == 16000
wav = wav.unsqueeze(0).cuda()

# Extract speech units
units = hubert.units(wav)
```

```bash
# Batch encode a dataset
python encode.py soft path/to/LibriSpeech/wavs path/to/output
python encode.py discrete path/to/LibriSpeech/wavs path/to/output
```

## Dependencies

- Python 3.x
- PyTorch
- torchaudio
- fairseq (for official HuBERT)

## Notes

- Iterative refinement: first iteration uses MFCC k-means, subsequent iterations use model features
- Discrete units: quantized cluster indices (tokenization)
- Soft units: continuous predictions that approximate discrete targets
- 16kHz input sample rate required
- Training requires pre-extracted discrete units as targets
- Related to official Facebook HuBERT but focused on voice conversion use case
- Paper: "A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"
