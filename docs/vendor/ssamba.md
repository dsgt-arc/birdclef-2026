---
name: "SSAMBA"
source: "https://github.com/SiavashShams/ssamba"
license: "BSD-3-Clause"
category: "self-supervised"
pipeline_stage:
  - "representation-learning"
  - "embedding"
relevance: "Efficient Mamba-based audio encoder; 92.7% faster than transformers with comparable accuracy"
status: "active"
---

# SSAMBA (Self-Supervised Audio Mamba)

## Overview

SSAMBA is the first self-supervised, attention-free audio representation model based on bidirectional Mamba state space models. It replaces the quadratic-complexity transformer in SSAST with linear-complexity Mamba blocks, achieving **92.7% faster inference** and **95.4% better memory efficiency** while maintaining competitive accuracy on audio classification tasks.

**Paper**: [SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model](https://arxiv.org/abs/2405.11831) (IEEE SLT 2024)

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `AMBAModel` | Main Mamba-based audio model | `vendor/ssamba/src/models/both_models.py` |
| `ASTModel` | Transformer baseline (SSAST) | `vendor/ssamba/src/models/both_models.py` |
| `inference_amba.py` | Inference benchmarking script | `vendor/ssamba/src/inference/` |
| `finetune/` | Finetuning recipes for ESC-50, AudioSet, etc. | `vendor/ssamba/src/finetune/` |

### Model Sizes

| Size | Embed Dim | Depth | Params | ESC-50 | AudioSet-20K mAP |
|------|-----------|-------|--------|--------|------------------|
| tiny | 192 | 24 | 7M | 80.1% | 23.3 |
| small | 384 | 24 | 26M | 85.5% | 25.6 |
| base | 768 | 24 | 99M | 89.3% | 28.3 |

### Pretrained Weights

Available on [Google Drive](https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX):
- `amba_tiny_400.pth`, `amba_tiny_300.pth`, `amba_tiny_250.pth`
- `amba_small_400.pth`, `amba_small_300.pth`, `amba_small_250.pth`
- `amba_base_400.pth`, `amba_base_300.pth`, `amba_base_250.pth`

(Number indicates masked patches during pretraining; 400 performs best for audio tasks)

## Relevance to Pipeline

- **Representation Learning**: Pretrained on AudioSet + LibriSpeech using masked patch modeling (same objective as SSAST)
- **Inference Efficiency**: Linear O(n) complexity vs transformer's O(n²) - critical for BirdCLEF inference budget
- **Accuracy vs Speed**: SSAMBA-small (85.5% ESC-50) at ~10x faster than BEATs may be acceptable tradeoff

### Comparison to Other Encoders

| Encoder | ESC-50 | Complexity | Speed vs SSAST |
|---------|--------|------------|----------------|
| BEATs | 95.8% | O(n²) | ~1x |
| SSAST | ~88% | O(n²) | 1x (baseline) |
| SSAMBA-base | 89.3% | O(n) | ~10x faster |
| SSAMBA-small | 85.5% | O(n) | ~11x faster |
| SSAMBA-tiny | 80.1% | O(n) | ~13x faster |

## Usage

```python
import torch
import sys
sys.path.append('vendor/ssamba/src')
from models.both_models import AMBAModel

# Model configuration
vision_mamba_config = {
    'img_size': (128, 1024),  # (freq_bins, time_frames)
    'patch_size': 16,
    'stride': 16,
    'embed_dim': 384,  # small
    'depth': 24,
    'bimamba_type': "v2",
    'if_bidirectional': True,
    'if_cls_token': True,
}

# Load pretrained model
model = AMBAModel(
    label_dim=50,  # number of classes
    fshape=16, tshape=16,
    fstride=16, tstride=16,
    input_fdim=128,
    input_tdim=1024,
    model_size='small',
    pretrain_stage=False,
    load_pretrained_mdl_path='path/to/amba_small_400.pth',
    vision_mamba_config=vision_mamba_config
)

# Input: (batch, time_frames, freq_bins) - 128-dim mel filterbank
feats = torch.randn(1, 1024, 128)

# Forward pass
with torch.no_grad():
    output = model(x=feats, task="ft_avgtok")  # mean pooling over tokens
```

### Feature Extraction (for linear probe)

```python
# To extract embeddings without classification head:
# Modify finetuningavgtok() to return before mlp_head
# Or use intermediate representations from self.v
```

## Dependencies

Core requirements from `requirements.txt`:
- `torch>=2.2.2`
- `torchaudio>=2.2.2`
- `mamba_ssm>=1.2.0.post1` (requires CUDA)
- `causal_conv1d>=1.2.0.post2`
- `timm>=1.0.3`
- `einops>=0.8.0`

**Additional setup**:
```bash
# Clone Vision Mamba (required dependency)
git clone https://github.com/hustvl/Vim.git vendor/ssamba/Vim
```

## Notes

- **CUDA Required**: `mamba_ssm` needs CUDA for full speed; pure PyTorch fallback exists but is slower
- **Bidirectional**: Uses forward + backward Mamba passes (unlike causal-only language models)
- **Pretraining**: Same masked patch modeling as SSAST (discriminative + generative objectives)
- **Input Format**: Expects `(batch, time, freq)` with 128-dim mel filterbank features
- **Checkpoints**: Wrapped in `DataParallel`, need to handle `module.` prefix when loading
- **bimamba_type Issue**: May need workaround per [GitHub issue](https://github.com/hustvl/Vim/issues/14#issuecomment-1964685563)

## Related Documentation

- **Concept**: [SSAMBA Audio Mamba](../concepts/ssamba-audio-mamba.md) - Research concept and hypotheses
- **Concept**: [SSAST Audio Transformer](../concepts/ssast-audio-transformer.md) - Transformer baseline
- **Vendor**: [SSAST](ssast.md) - Self-supervised transformer baseline
- **Vendor**: [AST](ast.md) - Supervised transformer baseline
- **Vendor**: [BEATs](beats.md) - Discrete token prediction baseline
- **Experiment**: [001-encoder-benchmark](../../user/anthony/experiments/001-encoder-benchmark/proposal.md) Part C

## Local Path

```
vendor/ssamba/
├── src/
│   ├── models/
│   │   └── both_models.py    # AMBAModel, ASTModel
│   ├── inference/
│   │   └── inference_amba.py # Benchmarking
│   ├── finetune/
│   │   ├── esc50/            # ESC-50 recipes
│   │   ├── audioset/         # AudioSet recipes
│   │   └── speechcommands_v2/
│   └── pretrain/             # Pretraining scripts
├── requirements.txt
└── README.md
```
