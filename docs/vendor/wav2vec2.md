---
name: "Wav2vec2.0"
source: "https://github.com/HarunoriKawano/Wav2vec2.0"
license: "Apache-2.0"
category: "self-supervised"
pipeline_stage:
  - "tokenization"
  - "representation-learning"
relevance: "Reference implementation of contrastive self-supervised learning with quantized targets"
status: "reference"
---

# Wav2vec2.0

## Overview

This is a PyTorch implementation of wav2vec 2.0, a framework for self-supervised learning of speech representations. The model learns representations by solving a contrastive task over masked latent speech representations, using quantized targets from a learned codebook. This architecture is foundational for understanding modern audio self-supervised learning.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| Feature encoder | Convolutional network for raw waveform encoding | Implementation files |
| Quantization module | Gumbel-softmax quantization for discrete targets | Implementation files |
| Transformer encoder | Contextualized representation learning | Implementation files |
| Contrastive loss | Self-supervised training objective | Implementation files |

## Relevance to Pipeline

- **Tokenization**: Core reference for learned quantization/tokenization of audio; the quantization module creates discrete token targets
- **Representation Learning**: Demonstrates masked prediction over discrete tokens, similar to our Phase 2 transformer training
- **Embedding**: Produces contextual embeddings that can be used for downstream tasks

## Usage

```python
# See example.py for usage
python example.py
```

## Dependencies

- Python 3.x
- PyTorch
- fairseq (for pre-trained models)

## Notes

- Key paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (Baevski et al., 2020)
- Uses Gumbel-softmax for differentiable quantization during training
- Codebook is learned jointly with the model
- HuBERT and DinoSR build on these concepts with different target strategies
- Pre-trained models available via HuggingFace Transformers or fairseq
