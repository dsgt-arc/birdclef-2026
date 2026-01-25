# ssamba-hf

Pure PyTorch implementation of SSAMBA (Self-Supervised Audio Mamba) using HuggingFace's `transformers.MambaMixer`.

## Goal

Load pretrained SSAMBA weights and run inference on CPU without requiring:
- `mamba_ssm` CUDA kernels
- `causal_conv1d` CUDA kernels
- VisionMamba from Vim repository

## Installation

```bash
cd user/anthony/experiments/004-ssamba-hf-port/ssamba_hf
uv sync
```

## Usage

```python
from ssamba_hf import SSAMBAModel, SSAMBAConfig

# Load pretrained model
config = SSAMBAConfig.small()
model = SSAMBAModel(config)
model.load_pretrained("path/to/amba_small_400.pth")

# Extract embeddings
embeddings = model.get_embeddings(mel_spectrogram)  # (B, 384)
```

## Architecture

See `../design.md` for full architecture diagram and weight mapping.
