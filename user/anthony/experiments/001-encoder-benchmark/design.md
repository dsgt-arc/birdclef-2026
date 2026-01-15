# Design: Part C - AST/SSAST/SSAMBA Encoder Integration

## Overview

Add three new encoder families to the benchmark to compare transformer vs Mamba architectures:

| Family | Models | Architecture | Complexity |
|--------|--------|--------------|------------|
| AST | AST-base | ViT (12L, 768d) | O(n²) |
| SSAST | SSAST-base | ViT (12L, 768d) | O(n²) |
| SSAMBA | tiny/small/base | Mamba (bidirectional) | O(n) |

## Encoder Implementations

### AST (Audio Spectrogram Transformer)

**Source**: HuggingFace `MIT/ast-finetuned-audioset-10-10-0.4593`

```python
# encoders/ast.py
from transformers import ASTModel, ASTFeatureExtractor

class ASTEncoder(AudioEncoder):
    def __init__(self):
        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        # AST expects 16kHz audio
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        outputs = self.model(**inputs)
        # Mean pool over sequence dimension
        return outputs.last_hidden_state.mean(dim=1)  # [B, 768]
```

**Notes**:
- AST uses 128-dim mel filterbank, 10ms hop, 25ms window
- Patch size: 16x16 with 10x10 overlap during finetuning
- Pretrained on ImageNet, finetuned on AudioSet

### SSAST (Self-Supervised AST)

**Source**: HuggingFace or official repo checkpoints

```python
# encoders/ssast.py
# SSAST uses same architecture as AST but different pretraining

class SSASTEncoder(AudioEncoder):
    def __init__(self, checkpoint_path: str):
        # Load from official checkpoint
        self.model = load_ssast_model(checkpoint_path)

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        fbank = self._compute_fbank(waveform)  # [B, T, 128]
        outputs = self.model(fbank)
        return outputs.mean(dim=1)  # [B, 768]
```

**Checkpoint options**:
- `ssast-base-patch-400` - 400 masked patches during pretraining
- `ssast-base-frame-400` - Frame-based (better for speech)

### SSAMBA (Self-Supervised Audio Mamba)

**Source**: [Google Drive](https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX)

```python
# encoders/ssamba.py
from mamba_ssm import Mamba

class SSAMBAEncoder(AudioEncoder):
    def __init__(self, size: str = "small"):
        # Load pretrained SSAMBA
        self.model = load_ssamba_model(size)  # tiny/small/base

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        fbank = self._compute_fbank(waveform)
        patches = self._patchify(fbank)  # Same as SSAST
        outputs = self.model(patches)
        return outputs.mean(dim=1)
```

**Model sizes**:
| Size | Params | Layers | Hidden Dim |
|------|--------|--------|------------|
| tiny | 7M | 12 | 192 |
| small | 26M | 24 | 384 |
| base | 99M | 24 | 768 |

## Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
ssamba = [
    "mamba-ssm>=1.0.0",
    "causal-conv1d>=1.0.0",
]
```

**Note**: SSAMBA requires CUDA for full speed. Pure PyTorch fallback exists but is slower.

## Preprocessing Pipeline

All models use similar preprocessing:

```
Audio (16kHz)
    ↓ [mel filterbank]
Spectrogram (T × 128)
    ↓ [patchify 16×16]
Patches (N × 256)
    ↓ [linear projection]
Embeddings (N × D)
    ↓ [encoder]
Features (N × D)
    ↓ [mean pool]
Final (D,)
```

| Model | Patch Size | Overlap | Embed Dim |
|-------|------------|---------|-----------|
| AST | 16×16 | 10×10 | 768 |
| SSAST | 16×16 | 6×6 | 768 |
| SSAMBA | 16×16 | 6×6 | varies |

## Benchmark Protocol

### Accuracy Evaluation

Same as Part A:
1. Extract features for all ESC-50 samples
2. 5-fold cross-validation
3. Train sklearn LogisticRegression
4. Report mean ± std accuracy

### Inference Time Evaluation

```python
def benchmark_inference(encoder, samples, device="cpu"):
    """Measure inference time per sample."""
    encoder.to(device).eval()

    # Warmup
    for _ in range(10):
        encoder.extract_features(samples[0])

    # Timed runs
    times = []
    for sample in samples[:100]:
        start = time.perf_counter()
        encoder.extract_features(sample)
        times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "throughput": 1 / np.mean(times),
    }
```

Benchmark on:
- GPU (L40S or V100)
- CPU (4 threads, matching Exp 002)

## File Structure

```
src/birdclef/encoders/
├── __init__.py
├── base.py          # AudioEncoder base class
├── wav2vec2.py      # ✅ Existing
├── beats.py         # ✅ Existing
├── eat.py           # ✅ Existing
├── ast.py           # NEW
├── ssast.py         # NEW
└── ssamba.py        # NEW
```

## Checkpoint Management

```
~/scratch/birdclef-2026/checkpoints/
├── beats/           # ✅ Existing
├── eat/             # ✅ Existing
├── ast/             # NEW (auto-downloaded by HF)
├── ssast/           # NEW
│   └── ssast-base-patch-400.pth
└── ssamba/          # NEW
    ├── ssamba-tiny-400.pth
    ├── ssamba-small-400.pth
    └── ssamba-base-400.pth
```

## Expected Results

| Encoder | ESC-50 (expected) | CPU Time (expected) | Params |
|---------|-------------------|---------------------|--------|
| BEATs | 84% (measured) | 150ms | 90M |
| AST | ~85% | ~150ms | 87M |
| SSAST | ~82% | ~150ms | 87M |
| SSAMBA-base | ~85% | ~20ms | 99M |
| SSAMBA-small | ~80% | ~10ms | 26M |
| SSAMBA-tiny | ~75% | ~5ms | 7M |

## Risks

1. **SSAMBA CUDA dependency**: May need to test on GPU-only first
2. **Checkpoint compatibility**: SSAMBA checkpoints may need custom loading
3. **Preprocessing mismatch**: Ensure fbank settings match original papers

## Next Steps

1. Download SSAMBA checkpoints
2. Implement `SSAMBAEncoder` wrapper
3. Verify preprocessing matches original
4. Run benchmark and compare to published results
