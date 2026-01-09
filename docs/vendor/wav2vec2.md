---
name: "Wav2vec 2.0"
source: "https://huggingface.co/docs/transformers/model_doc/wav2vec2"
license: "Apache-2.0"
category: "self-supervised"
pipeline_stage:
  - "tokenization"
  - "representation-learning"
relevance: "Foundation model for discrete audio tokens and efficient transformer inference"
status: "reference"
---

# Wav2vec 2.0

## Overview

Wav2vec 2.0 is a framework for self-supervised learning of speech representations. It masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. This architecture is crucial for our research into discrete audio tokens and static embeddings.

**Paper**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `Wav2Vec2FeatureExtractor` | Convolutional feature encoder (raw audio to latent) | `transformers` |
| `Wav2Vec2Model` | Transformer encoder with quantization module | `transformers` |
| `Wav2Vec2ForPreTraining` | Model with contrastive loss head | `transformers` |
| `Wav2Vec2Quantizer` | Gumbel-Softmax vector quantization | `transformers` internal |

## Relevance to Pipeline

- **Discrete Tokens**: The model's vector quantization module is a primary candidate for generating discrete audio tokens (codes) for our efficiency experiments.
- **Static Embeddings**: We plan to distill the transformer's contextual embeddings into static embeddings (Model2Vec style).
- **CPU Inference**: A key target for optimization; we will investigate pruning or distilling this model for CPU constraints.

## Usage

```python
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# 1. Load Pre-trained Model
model_name = "facebook/wav2vec2-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

# 2. Process Audio
# dummy audio: 16kHz, 1 second
audio_input = torch.randn(1, 16000)
inputs = feature_extractor(audio_input, return_tensors="pt", sampling_rate=16000)

# 3. Forward Pass
with torch.no_grad():
    outputs = model(**inputs)

# Last hidden key states (contextual embeddings)
embeddings = outputs.last_hidden_state  # shape: (1, seq_len, 768)

# Accessing quantized tokens (requires ForPreTraining model usually, or hook)
# This is key for our "Discrete Token" experiments
```

## Dependencies

- `transformers`
- `torch`
- `librosa` (for audio loading)

## Notes

- **Efficiency**: Standard wav2vec 2.0 is heavy (95M+ params). We need to explore smaller variants or distillation.
- **BirdCLEF**: Pre-trained weights on human speech may need domain adaptation (fine-tuning) on bird calls (e.g., BirdAVES starts from similar architectures).
