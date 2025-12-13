---
name: "bioacoustics-model-zoo"
source: "https://github.com/kitzeslab/bioacoustics-model-zoo"
license: "MIT"
category: "utility"
pipeline_stage:
  - "inference"
  - "embedding"
relevance: "Unified API for pre-trained bioacoustics models; integrates with OpenSoundscape"
status: "active"
---

# Bioacoustics Model Zoo

## Overview

The Bioacoustics Model Zoo provides a unified Python interface for accessing pre-trained models for bioacoustic classification tasks. It wraps models like BirdNET, Perch, HawkEars, and BirdSet models with a consistent API for prediction, embedding extraction, and transfer learning. Developed by the Kitzes Lab at University of Pittsburgh.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `bmz.BirdNET()` | BirdNET classifier wrapper (6K species) | `bioacoustics_model_zoo/birdnet.py` |
| `bmz.Perch()` | Google Perch embedding model | `bioacoustics_model_zoo/perch.py` |
| `bmz.Perch2()` | Perch v2 with improved architecture | `bioacoustics_model_zoo/perch.py` |
| `bmz.HawkEars()` | 5-model ensemble for N. American birds | `bioacoustics_model_zoo/hawkears.py` |
| `bmz.BirdSetConvNeXT()` | BirdSet pre-trained ConvNeXT | `bioacoustics_model_zoo/bmz_birdset/` |
| `bmz.list_models()` | List available models | `bioacoustics_model_zoo/__init__.py` |

## Relevance to Pipeline

- **Inference**: Standardized interface for running multiple pre-trained models; useful for baseline comparisons
- **Embedding**: All models provide `embed()` method for feature extraction
- **Representation Learning**: Supports transfer learning via `train()` method on custom datasets

## Usage

```python
import bioacoustics_model_zoo as bmz

# List available models
bmz.list_models()

# Load and use BirdNET
model = bmz.BirdNET()
scores = model.predict(['audio.wav'], activation_layer='softmax')
embeddings = model.embed(['audio.wav'])

# Fine-tune on custom classes
model.initialize_custom_classifier(classes=['species_a', 'species_b'])
model.train(train_df, val_df, n_augmentation_variants=5)
model.save('custom_model.pkl')
```

## Dependencies

- opensoundscape
- Model-specific: tensorflow (Perch/BirdNET), timm (HawkEars), transformers (BirdSet)
- Install extras: `pip install bioacoustics-model-zoo[all]`

## Notes

- Integrates seamlessly with OpenSoundscape for data loading and preprocessing
- Supports PyTorch Lightning via `LightningSpectrogramModule.from_model()`
- BirdNET feature extractor is frozen (not open-source); only classifier head is trainable
- Perch2 uses TensorFlow >= 2.20.0
- HawkEars is an ensemble of 5 CNNs for 314 North American species
