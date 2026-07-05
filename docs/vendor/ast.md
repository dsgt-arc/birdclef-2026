---
name: "AST"
source: "https://github.com/YuanGongND/ast"
license: "BSD-3-Clause"
category: "pretrained-model"
pipeline_stage:
  - "representation-learning"
  - "embedding"
  - "inference"
relevance: "Audio Spectrogram Transformer; supervised baseline for audio classification"
status: "active-reference"
---

# AST (Audio Spectrogram Transformer)

## Overview

AST is the first convolution-free, purely attention-based model for audio classification. It applies Vision Transformer (ViT) architecture to audio spectrograms, treating them as images and splitting them into 16x16 patches. AST achieves state-of-the-art results on AudioSet (0.485 mAP), ESC-50 (95.6%), and Speech Commands V2 (98.1%).

**Paper**: [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) (Interspeech 2021)

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `ASTModel` | Main transformer model | upstream `src/models/ast_models.py` |
| `dataloader.py` | Audio loading and preprocessing | upstream `src/dataloader.py` |
| `traintest.py` | Training and evaluation loops | upstream `src/traintest.py` |
| `egs/` | Recipes for ESC-50, AudioSet, SpeechCommands | upstream `egs/` |

### Model Variants

| Variant | Patch Size | Input Size | Params | Notes |
|---------|------------|------------|--------|-------|
| tiny224 | 16x16 | 224x224 | ~6M | Smallest |
| small224 | 16x16 | 224x224 | ~22M | - |
| base224 | 16x16 | 224x224 | ~87M | - |
| base384 | 16x16 | 384x384 | ~87M | Best performance (default) |

### Pretrained Weights

Available via HuggingFace or auto-download:
- `MIT/ast-finetuned-audioset-10-10-0.4593` - AudioSet finetuned (HuggingFace)
- ImageNet pretrained weights (auto-downloaded via timm)

## Relevance to Pipeline

- **Supervised Baseline**: AST with ImageNet+AudioSet pretraining achieves 95.6% on ESC-50
- **Architecture Reference**: SSAST and SSAMBA build on AST's patch-based spectrogram processing
- **Inference Comparison**: Transformer baseline for measuring SSAMBA's speedup claims

### Published Results

| Dataset | Metric | Result |
|---------|--------|--------|
| AudioSet | mAP | 0.485 |
| ESC-50 | Accuracy | 95.6% |
| Speech Commands V2 | Accuracy | 98.1% |

## Usage

```python
import os
import torch
import sys
sys.path.append('/path/to/ast/src')
from models.ast_models import ASTModel

# Set pretrained model download location
os.environ['TORCH_HOME'] = '/path/to/checkpoints/ast/pretrained_models'

# Create model
ast_mdl = ASTModel(
    label_dim=527,             # AudioSet classes (or 50 for ESC-50)
    fstride=10, tstride=10,    # patch stride (10 = overlap of 6)
    input_fdim=128,            # mel frequency bins
    input_tdim=1024,           # time frames (~10.24s at 100 fps)
    imagenet_pretrain=True,    # use ImageNet weights
    audioset_pretrain=False,   # set True for AudioSet-pretrained
    model_size='base384'       # model variant
)

# Input: [batch, time_frames, freq_bins]
# Note: should be normalized with dataset mean/std
test_input = torch.rand([10, 1024, 128])

# Forward pass - returns raw logits
output = ast_mdl(test_input)  # [10, 527]
probs = torch.sigmoid(output)  # for multi-label
```

### Using HuggingFace Checkpoint

```python
from transformers import ASTModel, ASTFeatureExtractor

# Load from HuggingFace
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Process audio
inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs)

# Get embeddings (mean pool)
embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
```

## Dependencies

From `requirements.txt`:
- `torch`
- `torchaudio`
- `timm` (for ViT backbone)
- `numpy`
- `scikit-learn`

## Notes

- **Input Normalization**: Spectrograms must be normalized with dataset mean/std (see dataloader.py line 191)
- **Variable Length**: Supports variable input lengths via positional embedding interpolation
- **Patch Overlap**: Use `fstride=10, tstride=10` for best performance (6-pixel overlap)
- **Colab Demo**: One-click inference demo available at the repo
- **PSLA Training**: Best results use PSLA training pipeline (see separate repo)

## Source Checkout

The AST repo is no longer retained as a submodule. For source-level inspection, clone upstream into scratch or another external workspace:

```bash
git clone https://github.com/YuanGongND/ast /path/to/ast
```

Project code generally uses the HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593` checkpoint instead of importing this checkout directly.

## Related Documentation

- **Concept**: [SSAMBA Audio Mamba](../concepts/ssamba-audio-mamba.md) - Efficient Mamba alternative
- **Vendor**: [SSAST](ssast.md) - Self-supervised version
- **Vendor**: [SSAMBA](ssamba.md) - Mamba-based replacement (92.7% faster)
- **Vendor**: [BEATs](beats.md) - Discrete token approach
- **Experiment**: [001-encoder-benchmark](../../user/anthony/experiments/001-encoder-benchmark/proposal.md) Part C

## Related Work

- **SSAST**: Self-supervised version (no ImageNet pretraining needed)
- **SSAMBA**: Mamba-based replacement (92.7% faster inference)
- **PSLA**: Training pipeline for SOTA AudioSet results
- **CMKD**: Knowledge distillation to improve AST
