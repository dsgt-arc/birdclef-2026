---
name: "EAT (Efficient Audio Transformer)"
source: "https://github.com/cwx-worst-one/EAT"
license: "MIT"
category: "self-supervised"
pipeline_stage:
  - "tokenization"
  - "representation-learning"
relevance: "15x more efficient than BEATs; primary candidate for distillation"
status: "active"
---

# EAT (Efficient Audio Transformer)

## Overview

EAT is a self-supervised audio pre-training framework that achieves BEATs-level performance with 15× less compute. It uses bootstrap self-supervised training with an Utterance-Frame Objective (UFO) and sparse encoding with large inverse block masks.

**Paper**: [EAT: Self-Supervised Pre-Training with Efficient Audio Transformer](https://arxiv.org/abs/2401.03497) (IJCAI 2024)

## Key Innovations

| Innovation | Description |
|------------|-------------|
| **Sparse Encoding** | Only processes visible patches (20-30%), discards masked tokens at encoder input |
| **Block Masking** | Large contiguous masks (70-80%) force global context usage |
| **UFO Loss** | Frame-level + utterance-level objectives for local and global learning |
| **10 epochs** | vs BEATs' 342 epochs for similar performance |

## Why EAT over BEATs

- **15× cheaper** to train
- **Same quality** on downstream tasks
- **Simpler** training loop (no iterative tokenizer refinement)
- **HuggingFace integration** available (May 2025)

## Architecture

```
Audio → Mel Spectrogram → Patch Embed → [MASK 70-80%]
                                              ↓
                              Transformer (visible patches only)
                                              ↓
                              Frame Loss + Utterance Loss (UFO)
```

## Relevance to Pipeline

- **Tokenization**: Uses discrete targets like BEATs
- **Distillation source**: Strong teacher for model2vec-style distillation
- **Efficiency**: Already optimized, may be fast enough without distillation

## Local Path

```
vendor/EAT/
├── src/           # Model implementation
├── config/        # Training configs
├── scripts/       # Training scripts
├── inference/     # Inference utilities
└── models/        # Model definitions
```

## Usage

```python
# Feature extraction (from their README)
from src.eat import EATModel

model = EATModel.from_pretrained("eat-base")
features = model.extract_features(audio_tensor)
```

## Dependencies

- PyTorch
- torchaudio
- fairseq (for some components)

## Notes

- Pre-trained checkpoints available on HuggingFace
- Trained on AudioSet-2M
- Good transfer to bioacoustics per Foundation Models review
