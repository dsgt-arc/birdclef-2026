---
name: "BEATs"
source: "https://github.com/microsoft/unilm/tree/master/beats"
license: "MIT"
category: "self-supervised"
pipeline_stage:
  - "tokenization"
  - "representation-learning"
relevance: "Discrete acoustic tokenizer + transformer; baseline for distillation"
status: "active"
---

# BEATs (Bidirectional Encoder representation from Audio Transformers)

## Overview

BEATs is an audio pre-training framework that iteratively trains an acoustic tokenizer and audio transformer. Unlike continuous approaches, BEATs produces **discrete tokens** from a learned codebook, making it ideal for model2vec-style distillation.

**Paper**: [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) (ICML 2023)

## Key Components

| Component | Description | File |
|-----------|-------------|------|
| `BEATs` | Main model (tokenizer + transformer) | `BEATs.py` |
| `Tokenizers` | Acoustic tokenizer with codebook | `Tokenizers.py` |
| `quantizer` | Vector quantization module | `quantizer.py` |

## Pretrained Checkpoints

| Checkpoint | Description | Download |
|------------|-------------|----------|
| BEATs_iter1 | First iteration tokenizer | [Link](https://aka.ms/beats-iter1) |
| BEATs_iter2 | Second iteration (better) | [Link](https://aka.ms/beats-iter2) |
| BEATs_iter3 | Third iteration (best) | [Link](https://aka.ms/beats-iter3) |
| BEATs_iter3+ | Fine-tuned on AS-2M | [Link](https://aka.ms/beats-iter3-as2m) |

## Usage

```python
import torch
from beats.BEATs import BEATs, BEATsConfig

# Load pretrained
checkpoint = torch.load('BEATs_iter3.pt')
cfg = BEATsConfig(checkpoint['cfg'])
model = BEATs(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# Extract features (continuous)
audio = torch.randn(1, 16000)  # 1 second @ 16kHz
padding_mask = torch.zeros(1, 16000).bool()
features = model.extract_features(audio, padding_mask)[0]  # [1, T, 768]

# Get discrete tokens (from tokenizer)
from beats.Tokenizers import TokenizersConfig, Tokenizers

tokenizer_ckpt = torch.load('Tokenizers_iter3.pt')
tokenizer = Tokenizers(TokenizersConfig(tokenizer_ckpt['cfg']))
tokenizer.load_state_dict(tokenizer_ckpt['model'])

tokens = tokenizer.extract_labels(audio, padding_mask)  # discrete IDs
```

## Discrete Tokenization

BEATs uses **iterative refinement**:

1. **Iter 1**: Random projection tokenizer (like BEST-RQ)
2. **Iter 2**: Tokenizer trained on iter 1 encoder features
3. **Iter 3**: Tokenizer trained on iter 2 encoder features

Each iteration produces more semantic tokens.

## Relevance to Pipeline

- **Discrete tokens**: Direct input for model2vec-style distillation
- **Pretrained**: Ready to use, no training needed for baseline
- **Codebook**: Finite vocabulary (~8192 tokens)
- **SOTA**: Strong bioacoustics performance per literature review

## Local Path

```
vendor/unilm/beats/
├── BEATs.py           # Main model
├── Tokenizers.py      # Acoustic tokenizer
├── quantizer.py       # Vector quantization
├── backbone.py        # Transformer backbone
└── modules.py         # Building blocks
```

## Notes

- 15× slower to **train** than EAT, but inference is comparable
- Tokenizer can be exported separately for fast inference
- Consider EAT for training efficiency, BEATs for pretrained baseline
