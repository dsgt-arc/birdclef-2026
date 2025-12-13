---
name: "ProtoCLR"
source: "https://github.com/ilyassmoummad/ProtoCLR"
license: "Unspecified"
category: "self-supervised"
pipeline_stage:
  - "representation-learning"
relevance: "Domain-invariant representation learning for few-shot bird sound classification"
status: "reference"
---

# ProtoCLR

## Overview

ProtoCLR (Prototypical Contrastive Learning) is a self-supervised learning approach for domain-invariant representation learning of bird sounds. It is designed for few-shot classification scenarios where models must generalize from focal recordings (training) to soundscape recordings (evaluation), addressing the domain shift challenge in bioacoustics.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `cvt13` | CvT-13 vision transformer encoder architecture | `cvt.py` |
| `MelSpectrogramProcessor` | Audio to mel-spectrogram preprocessing | `melspectrogram.py` |
| `train_encoder.py` | Training script for ProtoCLR/SimCLR/SupCon | `train_encoder.py` |
| `test_fewshot.py` | Few-shot evaluation script | `test_fewshot.py` |
| Pre-trained weights | CE, SimCLR, SupCon, ProtoCLR checkpoints | HuggingFace: `ilyassmoummad/ProtoCLR` |

## Relevance to Pipeline

- **Representation Learning**: Demonstrates contrastive learning approaches (SimCLR, SupCon, ProtoCLR) for learning robust audio representations
- **Tokenization**: Uses mel-spectrograms as input rather than discrete tokens
- **Inference**: Achieves strong few-shot classification performance with only 19M parameters

## Usage

```python
import torch
from cvt import cvt13
from melspectrogram import MelSpectrogramProcessor

# Initialize preprocessor and model
preprocessor = MelSpectrogramProcessor()
model = cvt13()

# Load pre-trained weights
model.load_state_dict(torch.load("protoclr.pth", map_location="cpu"))
model.eval()

# Process audio (must be 16kHz, 6 seconds)
input_tensor = preprocessor.process(waveform).unsqueeze(0)

with torch.no_grad():
    embeddings = model(input_tensor)
```

## Dependencies

- Python 3.x
- PyTorch
- torchaudio
- timm (for CvT architecture)

## Notes

- Audio should be mono, 16kHz, 6 seconds (pad or chunk as needed)
- Training uses Xeno-Canto dataset (~685k segments, 10k species)
- Evaluation on BIRB benchmark soundscape datasets (SSW, NES, UHH, etc.)
- Smaller model (19M params) compared to BirdAVES (95-316M) or BioLingual (153M)
- Focus on domain invariance for focal-to-soundscape transfer
