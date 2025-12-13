---
name: "perch"
source: "https://github.com/google-research/perch"
license: "Apache-2.0"
category: "pretrained-model"
pipeline_stage:
  - "embedding"
  - "inference"
relevance: "Google's bird species classifier trained on 10k+ species; produces high-quality embeddings"
status: "reference"
---

# Perch

## Overview

Perch is a bioacoustics research project from Google Research that produces bird species classifiers trained on over 10,000 species. The model uses an EfficientNet backbone with PCEN mel-spectrogram frontend and provides both classification and embedding capabilities. Perch embeddings enable superior transfer learning for bioacoustic classification tasks.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| PCEN frontend | Mel-spectrogram with per-channel energy normalization | `chirp/models/frontend.py` |
| EfficientNet | Main classification backbone | `chirp/models/efficientnet.py` |
| Perch 2.0 | Updated architecture with improved heads | `chirp/models/perch_2.py` |
| Agile modeling | Search + active learning for custom classifiers | `chirp/inference/` |
| Call density | Proportion estimation without thresholds | `chirp/inference/call_density.py` |
| BIRB benchmark | Evaluation protocol for bioacoustics | `chirp/eval/` |

## Relevance to Pipeline

- **Embedding**: High-quality embeddings for transfer learning; baseline for representation quality
- **Inference**: State-of-the-art bird classification; target for our pipeline to match
- **Representation Learning**: Reference for training objectives and evaluation protocols

## Usage

```python
# Via bioacoustics-model-zoo (recommended)
import bioacoustics_model_zoo as bmz

model = bmz.Perch2()
predictions = model.predict(['audio.wav'])
embeddings = model.embed(['audio.wav'])

# Agile modeling notebooks
# embed_audio.ipynb - mass embedding
# agile_modeling.ipynb - search and active learning
# analysis.ipynb - inference and call density
```

## Dependencies

- TensorFlow 2.x
- tensorflow-hub
- JAX (for training)
- Poetry for package management

## Notes

- Models available on Kaggle: `google/bird-vocalization-classifier`
- SurfPerch variant trained on birds + coral reef + general audio
- BIRB benchmark for focal-to-soundscape transfer evaluation
- Perch 2.0 preprint: "Perch 2.0: The Bittern Lesson for Bioacoustics"
- Supports multi-species whale classification
- Not officially supported Google product
- TensorFlow pip install can be tricky; see documentation
