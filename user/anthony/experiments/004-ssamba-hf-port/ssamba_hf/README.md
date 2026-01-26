# ssamba-hf

Pure PyTorch implementation of SSAMBA (Self-Supervised Audio Mamba) using HuggingFace's Mamba primitives.

**Status**: Verified (cosine similarity = 1.0 with original)

## Why This Exists

Load pretrained SSAMBA weights and run inference without CUDA dependencies:
- No `mamba_ssm` CUDA kernels required
- No `causal_conv1d` CUDA kernels required
- No VisionMamba from Vim repository required
- Works on CPU, MPS, or CUDA

## Installation

```bash
cd user/anthony/experiments/004-ssamba-hf-port/ssamba_hf
uv sync
```

## Quick Start

```python
import torch
from ssamba_hf import load_from_ssamba_checkpoint

# Load pretrained SSAMBA-Small (384-dim, 24 layers)
model, config = load_from_ssamba_checkpoint(
    "path/to/ssamba_small_400.pth",
    device="cpu"  # or "cuda", "mps"
)

# Prepare spectrogram: (batch, time_frames, freq_bins)
# Default expects 128 mel bins, variable time frames
spectrogram = torch.randn(1, 1024, 128)

# Extract embeddings
with torch.no_grad():
    embeddings = model.get_embeddings(spectrogram)  # (1, 384)
```

## Pretrained Checkpoints

Download from the [SSAMBA release](https://github.com/SiavashShams/ssamba):

| Checkpoint | Embed Dim | Layers | Params |
|------------|-----------|--------|--------|
| `ssamba_tiny_400.pth` | 192 | 24 | ~6M |
| `ssamba_small_400.pth` | 384 | 24 | ~25M |
| `ssamba_base_400.pth` | 768 | 24 | ~100M |

The checkpoint config is automatically inferred - just pass the path.

## Usage Examples

### Batch Processing

```python
# Process multiple spectrograms
batch = torch.randn(8, 1024, 128)
embeddings = model.get_embeddings(batch)  # (8, 384)
```

### Variable Length Audio

```python
# 5-second audio at 100 frames/sec = 500 frames
short_audio = torch.randn(1, 500, 128)
embeddings = model.get_embeddings(short_audio)  # (1, 384)

# 30-second audio = 3000 frames
long_audio = torch.randn(1, 3000, 128)
embeddings = model.get_embeddings(long_audio)  # (1, 384)
```

### Create Model from Config

```python
from ssamba_hf import SSAMBAModel, SSAMBAConfig

# Use preset config
config = SSAMBAConfig.small()  # or .tiny(), .base()
model = SSAMBAModel(config)

# Or customize
config = SSAMBAConfig(
    embed_dim=384,
    depth=24,
    img_size=(128, 1024),  # (freq_bins, time_frames)
    patch_size=16,
    d_state=16,
    d_conv=4,
)
model = SSAMBAModel(config)
```

### Inspect Checkpoint

```python
from ssamba_hf import analyze_checkpoint

# See what's in a checkpoint
analyze_checkpoint("path/to/ssamba_small_400.pth")
# Prints: layer count, embed_dim, bidirectional weights, etc.
```

## Input Format

The model expects mel spectrograms:

| Dimension | Description | Typical Values |
|-----------|-------------|----------------|
| Batch | Number of samples | 1-32 |
| Time frames | Temporal dimension | 500-3000 |
| Freq bins | Mel frequency bins | 128 |

The spectrogram is internally:
1. Transposed to (batch, 1, freq, time)
2. Split into 16x16 patches with stride 8
3. Flattened to sequence with CLS token in middle
4. Processed through 24 bidirectional Mamba layers
5. Mean-pooled (excluding CLS) to produce embeddings

## Verification

The implementation has been verified against the original SSAMBA:

| Metric | Value |
|--------|-------|
| Cosine Similarity | 1.000000 |
| Max Absolute Diff | 5.25e-05 |
| Mean Absolute Diff | 4.72e-06 |

To re-run verification:

```bash
cd ../sbatch
sbatch run-verification.sbatch
tail -f logs/run-verification-*.out
```

## API Reference

### Functions

- `load_from_ssamba_checkpoint(path, device, verbose)` - Load model from original checkpoint
- `analyze_checkpoint(path)` - Print checkpoint structure info
- `convert_ssamba_weights(state_dict, config)` - Convert weight names

### Classes

- `SSAMBAConfig` - Model configuration (embed_dim, depth, patch_size, etc.)
- `SSAMBAModel` - Main model with `get_embeddings()` method
- `BidirectionalMamba` - Single bidirectional Mamba layer
- `BidirectionalMambaStack` - Stack of layers with residual connections
- `AudioPatchEmbed` - Spectrogram to patch embedding
- `RMSNorm` - Root Mean Square normalization

## Architecture

```
Spectrogram (B, T, F)
    │
    ▼ transpose + unsqueeze
(B, 1, F, T)
    │
    ▼ AudioPatchEmbed (Conv2d 16x16, stride 8)
(B, num_patches, embed_dim)
    │
    ▼ Insert CLS token at middle
(B, num_patches+1, embed_dim)
    │
    ▼ Add positional embedding
    │
    ▼ BidirectionalMambaStack (24 layers)
    │   Each layer:
    │   ├─ Forward Mamba (A, D, conv1d, projections)
    │   ├─ Backward Mamba (A_b, D_b, conv1d_b, shared projections)
    │   └─ Combine: forward*D + backward*D_b
    │
    ▼ RMSNorm
    │
    ▼ Mean pool (exclude CLS token)
    │
(B, embed_dim)
```

See `../design.md` for detailed weight mapping.
