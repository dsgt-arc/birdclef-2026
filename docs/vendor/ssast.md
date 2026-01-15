---
name: "SSAST"
source: "https://github.com/YuanGongND/ssast"
license: "BSD-3-Clause"
category: "self-supervised"
pipeline_stage:
  - "representation-learning"
  - "embedding"
relevance: "Self-supervised Audio Spectrogram Transformer; baseline for SSAMBA comparison"
status: "active"
---

# SSAST (Self-Supervised Audio Spectrogram Transformer)

## Overview

SSAST is the first patch-based joint discriminative and generative self-supervised learning framework for Audio Spectrogram Transformers. It enables pretraining AST on unlabeled audio (AudioSet + LibriSpeech) using masked spectrogram patch modeling, achieving similar or better performance than supervised ImageNet pretraining without requiring labeled data.

**Paper**: [SSAST: Self-Supervised Audio Spectrogram Transformer](https://arxiv.org/abs/2110.09784) (AAAI 2022)

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `ASTModel` | Main model with pretraining and finetuning modes | `vendor/ssast/src/models/ast_models.py` |
| `dataloader.py` | Audio loading and spectrogram preprocessing | `vendor/ssast/src/dataloader.py` |
| `finetune/` | Finetuning recipes for ESC-50, SpeechCommands, etc. | `vendor/ssast/src/finetune/` |
| `pretrain/` | Self-supervised pretraining scripts | `vendor/ssast/src/pretrain/` |

### Model Sizes

| Size | Heads | Depth | Embed Dim | Params |
|------|-------|-------|-----------|--------|
| tiny | 3 | 12 | 192 | ~6M |
| small | 6 | 12 | 384 | ~22M |
| base | 12 | 12 | 768 | ~87M |

### Pretrained Weights

Available in `pretrained_model/` directory or downloadable:
- `SSAST-Base-Patch-400.pth` - 400 masked patches, patch-based (recommended for audio)
- `SSAST-Base-Frame-400.pth` - 400 masked patches, frame-based (better for speech)

## Relevance to Pipeline

- **Baseline Comparison**: SSAST is the transformer baseline that SSAMBA improves upon (92.7% faster)
- **Pretraining Framework**: Same masked patch modeling objective used by SSAMBA
- **Flexible Patches**: Unlike ImageNet pretraining, supports arbitrary patch sizes (16x16 or 128x2 frames)

### Performance on ESC-50

| Model | Accuracy | Notes |
|-------|----------|-------|
| AST (ImageNet pretrain) | 88.7% | Supervised pretrain |
| SSAST-Patch | 88.8% | Self-supervised |
| SSAST-Frame | 87.0% | Self-supervised |
| AST (finetuned) | 95.6% | Full finetuning |

## Usage

```python
import sys
sys.path.append('vendor/ssast/src')
from models.ast_models import ASTModel

# Pretraining stage
ast_mdl = ASTModel(
    fshape=16, tshape=16,      # patch shape
    fstride=16, tstride=16,    # no overlap in pretraining
    input_fdim=128,            # mel bins
    input_tdim=1024,           # time frames (~10s)
    model_size='base',
    pretrain_stage=True
)

# Pretraining forward pass
test_input = torch.zeros([10, 1024, 128])  # [batch, time, freq]
acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=400)
mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=400)
loss = nce_loss + 10 * mse_loss

# Fine-tuning stage
ast_mdl = ASTModel(
    label_dim=50,              # ESC-50 classes
    fshape=16, tshape=16,
    fstride=10, tstride=10,    # overlap in finetuning
    input_fdim=128,
    input_tdim=512,            # can differ from pretraining
    model_size='base',
    pretrain_stage=False,
    load_pretrained_mdl_path='path/to/SSAST-Base-Patch-400.pth'
)

# Finetuning forward pass
output = ast_mdl(test_input, task='ft_avgtok')  # mean pool all tokens
```

### Tasks

| Task | Description |
|------|-------------|
| `ft_avgtok` | Finetune using mean of all token outputs (recommended) |
| `ft_cls` | Finetune using CLS token output |
| `pretrain_mpc` | Masked patch classification (discriminative) |
| `pretrain_mpg` | Masked patch generation (generative) |

## Dependencies

From `requirements.txt`:
- `torch`
- `torchaudio`
- `timm==0.4.5` (specific version required)
- `numpy`
- `scikit-learn`

## Notes

- **Input Format**: `[batch, time_frames, freq_bins]` - note the order differs from some models
- **Normalization**: Input spectrograms should be normalized with dataset mean/std
- **timm Version**: Requires `timm==0.4.5` specifically; newer versions may not be compatible
- **DataParallel**: Checkpoints are saved with DataParallel wrapper, need to handle `module.` prefix
- **Patch vs Frame**: Patch-based (16x16) better for audio events, frame-based (128x2) better for speech

## Related Documentation

- **Concept**: [SSAST Audio Transformer](../concepts/ssast-audio-transformer.md) - Research concept and hypotheses
- **Concept**: [SSAMBA Audio Mamba](../concepts/ssamba-audio-mamba.md) - Efficient Mamba alternative
- **Vendor**: [AST](ast.md) - Supervised baseline
- **Vendor**: [SSAMBA](ssamba.md) - Mamba-based successor
- **Vendor**: [BEATs](beats.md) - Discrete token approach
- **Experiment**: [001-encoder-benchmark](../../user/anthony/experiments/001-encoder-benchmark/proposal.md) Part C

## Local Path

```
vendor/ssast/
├── src/
│   ├── models/
│   │   └── ast_models.py     # Main model
│   ├── dataloader.py         # Data loading
│   ├── finetune/             # Finetuning recipes
│   │   ├── esc50/
│   │   └── speechcommands_v2_35/
│   ├── pretrain/             # Pretraining scripts
│   └── prep_data/            # Data preparation
├── pretrained_model/         # Checkpoint info
├── requirements.txt
└── README.md
```
