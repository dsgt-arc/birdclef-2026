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

### Load from pretrained checkpoint

```python
from ssamba_hf import load_from_ssamba_checkpoint

# Load pretrained model (config is inferred from checkpoint)
model, config = load_from_ssamba_checkpoint("path/to/amba_small_400.pth")

# Extract embeddings from spectrogram
import torch
spectrogram = torch.randn(1, 1024, 128)  # (batch, time_frames, freq_bins)
embeddings = model.get_embeddings(spectrogram)  # (batch, 384)
```

### Create model from config

```python
from ssamba_hf import SSAMBAModel, SSAMBAConfig

# Create model with small config
config = SSAMBAConfig.small()
model = SSAMBAModel(config)

# Or customize config
config = SSAMBAConfig(
    embed_dim=384,
    depth=24,
    img_size=(128, 1024),
    patch_size=16,
)
model = SSAMBAModel(config)
```

### Input format

The model expects spectrograms in shape `(batch, time_frames, freq_bins)`:
- `time_frames`: Number of time frames (default: 1024)
- `freq_bins`: Number of mel frequency bins (default: 128)

```python
# 5-second audio at 100 frames/sec = 500 time frames
# 128 mel bins
spectrogram = torch.randn(batch_size, 500, 128)
embeddings = model.get_embeddings(spectrogram)  # (batch, 384)
```

## Architecture

See `../design.md` for full architecture diagram and weight mapping.

## Components

- `SSAMBAConfig`: Model configuration dataclass
- `SSAMBAModel`: Main model class
- `AudioPatchEmbed`: Converts spectrogram to patch embeddings
- `BidirectionalMambaStack`: Stack of bidirectional Mamba blocks
- `RMSNorm`: Root Mean Square Layer Normalization
- `load_from_ssamba_checkpoint`: Load weights from original checkpoint
- `convert_ssamba_weights`: Convert original weight names to HF format

## Verification

Run the verification workflow to ensure numerical equivalence with the original implementation:

```bash
cd ../sbatch
sbatch download-weights.sbatch      # Download pretrained weights
sbatch setup-original-ssamba.sbatch # Set up original SSAMBA environment
sbatch run-verification.sbatch      # Compare outputs
```
