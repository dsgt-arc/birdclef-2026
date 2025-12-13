---
name: "opensoundscape"
source: "https://github.com/kitzeslab/opensoundscape"
license: "MIT"
category: "utility"
pipeline_stage:
  - "data-processing"
  - "inference"
relevance: "Core bioacoustics analysis library for audio loading, spectrograms, and CNN training"
status: "active"
---

# OpenSoundscape

## Overview

OpenSoundscape (OPSO) is a free and open-source Python library for analyzing bioacoustic data. It provides utilities for loading and manipulating audio files, creating spectrograms, training deep learning models, running pre-trained CNNs, and detecting periodic vocalizations. Developed by the Kitzes Lab at University of Pittsburgh.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `Audio` | Audio file loading and manipulation | `opensoundscape/audio.py` |
| `Spectrogram` | Spectrogram creation and visualization | `opensoundscape/spectrogram.py` |
| `CNN` | CNN training and inference | `opensoundscape/ml/cnn.py` |
| `BoxedAnnotations` | Raven annotation loading | `opensoundscape/annotations.py` |
| `ribbit` | Periodic vocalization detection | `opensoundscape/ribbit.py` |
| `LightningSpectrogramModule` | PyTorch Lightning integration | `opensoundscape/ml/lightning.py` |

## Relevance to Pipeline

- **Data Processing**: Primary library for audio loading, trimming, and spectrogram generation
- **Inference**: Standardized interface for running trained models on audio files
- **Representation Learning**: Provides CNN base classes and training infrastructure

## Usage

```python
from opensoundscape import Audio, Spectrogram, CNN

# Load and process audio
audio = Audio.from_file("recording.wav")
clip = audio.trim(0, 5)  # 5 second clip

# Create spectrogram
spec = Spectrogram.from_audio(clip)
spec.plot()

# Train a CNN
model = CNN(architecture='resnet18', sample_duration=2, classes=['species_a', 'species_b'])
model.train(train_df, validation_df, epochs=20, batch_size=256)

# Run inference
scores = model.predict(audio_files)

# Use with bioacoustics-model-zoo
import bioacoustics_model_zoo as bmz
perch = bmz.Perch()
embeddings = perch.embed(files)
```

## Dependencies

- Python 3.10-3.13
- PyTorch, torchaudio
- librosa
- pandas, numpy

## Notes

- Integrates tightly with bioacoustics-model-zoo
- Supports Raven annotation format
- RIBBIT algorithm for detecting periodic vocalizations (frogs, insects)
- Timestamp-aware audio loading from AudioMoth recordings
- Supports transfer learning and fine-tuning of pre-trained models
- Documentation at opensoundscape.org
