---
name: "BioLingual"
source: "https://github.com/david-rx/BioLingual"
license: "Apache-2.0"
category: "pretrained-model"
pipeline_stage:
  - "representation-learning"
  - "inference"
relevance: "Zero-shot bioacoustic classification via language-audio alignment"
status: "reference"
---

# BioLingual

## Overview

BioLingual is a contrastive language-audio model for bioacoustics, enabling zero-shot audio classification, text-to-audio search, and transfer learning to new bioacoustic tasks. It aligns natural language descriptions with audio embeddings using contrastive learning (similar to CLIP for images). Trained on AnimalSpeak, a large-scale captioned bioacoustic dataset from Xeno-canto and iNaturalist.

**Paper**: [Transferable Models for Bioacoustics with Human Language Supervision](https://arxiv.org/abs/2308.04978)

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `BioLingual` | Main model class for language-audio alignment | `biolingual/` |
| `AnimalSpeak` | Training dataset with captioned bioacoustic recordings | [HuggingFace](https://huggingface.co/datasets/davidrrobinson/AnimalSpeak) |
| Pre-trained weights | Model checkpoint | [davidrrobinson/BioLingual](https://huggingface.co/davidrrobinson/BioLingual) |

## Relevance to Pipeline

- **Zero-Shot Classification**: Can classify species using text prompts without training, e.g., "a recording of a black-capped chickadee singing"
- **Feature Extraction**: Audio encoder produces embeddings usable for downstream tasks
- **Comparison Baseline**: Mentioned in [Foundation Models for Bioacoustics review](https://arxiv.org/abs/2508.01277) as one of the evaluated models

### Relationship to NatureLM-audio

BioLingual is a predecessor to NatureLM-audio (also from Earth Species Project). Key differences:
- BioLingual: Contrastive learning (CLIP-style), smaller model
- NatureLM-audio: Generative LLM (Llama-3.1-8B), more flexible task specification

## Usage

```python
# Zero-shot classification example
from biolingual import BioLingual

model = BioLingual.from_pretrained("davidrrobinson/BioLingual")

# Classify audio using text prompts
labels = ["black-capped chickadee", "american robin", "house sparrow"]
probs = model.classify_audio(audio_path, labels)
```

## Dependencies

- PyTorch
- Transformers
- CLAP (base architecture)

## Notes

- Built on CLAP (Contrastive Language-Audio Pretraining) architecture
- Uses BEANS benchmark for evaluation
- Zero-shot approach is interesting for rare species with few training examples
- May be too slow for Kaggle CPU inference (requires running both audio and text encoders)
- Consider using just the audio encoder as a feature extractor if zero-shot not needed
