# Design: Model Distillation for Context Recovery

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DISTILLATION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                            │
│  │  Raw Audio   │  (10s @ 16kHz from AudioSet/BirdSet)                       │
│  │  Waveform    │                                                            │
│  └──────┬───────┘                                                            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     BEATs Frontend (Shared)                           │   │
│  │  ┌─────────┐   ┌─────────────┐   ┌──────────┐   ┌─────────────────┐  │   │
│  │  │  fbank  │ → │ patch_embed │ → │ layer_norm│ → │ proj (768-dim)  │  │   │
│  │  │ (128mel)│   │ (16×16 CNN) │   │           │   │                 │  │   │
│  │  └─────────┘   └─────────────┘   └──────────┘   └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │ Patch Embeds │  (B, 248, 768) - Pre-transformer representations           │
│  └──────┬───────┘                                                            │
│         │                                                                    │
│    ┌────┴────────────────────────────────────────┐                          │
│    │                                              │                          │
│    ▼                                              ▼                          │
│  ┌──────────────────────┐              ┌──────────────────────────────────┐ │
│  │   TEACHER PATH       │              │      STUDENT PATH                 │ │
│  │   (Offline/Cached)   │              │      (Training Target)            │ │
│  ├──────────────────────┤              ├──────────────────────────────────┤ │
│  │                      │              │                                   │ │
│  │  BEATs Transformer   │              │  ┌─────────────────────────────┐ │ │
│  │  (12 layers, 768d)   │              │  │    Tokenizer Quantization   │ │ │
│  │                      │              │  │    patch → token_id (1024)  │ │ │
│  │  Self-Attention      │              │  └──────────────┬──────────────┘ │ │
│  │  across 248 patches  │              │                 │                │ │
│  │                      │              │                 ▼                │ │
│  │                      │              │  ┌─────────────────────────────┐ │ │
│  │                      │              │  │   Static Embedding Lookup   │ │ │
│  │                      │              │  │   token_id → embed (768d)   │ │ │
│  │                      │              │  └──────────────┬──────────────┘ │ │
│  │                      │              │                 │                │ │
│  │                      │              │                 ▼                │ │
│  │                      │              │  ┌─────────────────────────────┐ │ │
│  │                      │              │  │     Student Sequence Model  │ │ │
│  │                      │              │  │  ┌───────────────────────┐  │ │ │
│  │                      │              │  │  │ Mamba (SSM, O(n))     │  │ │ │
│  │                      │              │  │  ├───────────────────────┤  │ │ │
│  │                      │              │  │  │ MLP-Mixer             │  │ │ │
│  │                      │              │  │  ├───────────────────────┤  │ │ │
│  │                      │              │  │  │ CNN-Pooler (1D Conv)  │  │ │ │
│  │                      │              │  │  ├───────────────────────┤  │ │ │
│  │                      │              │  │  │ TinyTransformer (2L)  │  │ │ │
│  │                      │              │  │  └───────────────────────┘  │ │ │
│  │                      │              │  └──────────────┬──────────────┘ │ │
│  │                      │              │                 │                │ │
│  └──────────┬───────────┘              └─────────────────┼────────────────┘ │
│             │                                            │                  │
│             ▼                                            ▼                  │
│  ┌──────────────────────┐              ┌──────────────────────────────────┐ │
│  │  Teacher Embeddings  │              │      Student Embeddings          │ │
│  │  (B, 248, 768)       │──── Loss ────│      (B, 248, 768)               │ │
│  │  Contextualized      │   MSE/Cos    │      Predicted Context           │ │
│  └──────────────────────┘              └──────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              INFERENCE PIPELINE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Audio → Frontend (ONNX) → Quantize (FAISS) → Static Lookup → Student → Pool│
│                                                                              │
│  Latency Target: <10ms on CPU (batch=1)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Strategy

> **Scope**: This experiment focuses on standard audio benchmarks (AudioSet, ESC-50).
> Bioacoustics evaluation (BirdSet, BirdCLEF) will be a separate experiment.

### AudioSet (Training & Evaluation)

**Source**: [confit/audioset-16khz-wds](https://huggingface.co/datasets/confit/audioset-16khz-wds)

| Version | Clips | Duration | Storage | Use Case |
|---------|-------|----------|---------|----------|
| **20k** | 20,550 | 56 hrs | ~2 GB | Pilot experiments |
| **500k** | 497,982 | 1,371 hrs | ~50 GB | Main distillation |
| **2m** | 1,912,024 | 5,264 hrs | ~200 GB | Full scale (optional) |
| **test** | 18,886 | 51 hrs | ~2 GB | Evaluation |

**Why AudioSet**:
- BEATs was pretrained on AudioSet → minimal domain shift for distillation
- WebDataset format for efficient streaming
- 527 sound classes for diverse acoustic patterns
- Standard benchmark for audio representation learning

**Download**:
```bash
# Pilot: 20k version (~2 GB)
huggingface-cli download confit/audioset-16khz-wds \
  --include 20k/train/*.tar 20k/test/*.tar \
  --repo-type=dataset \
  --local-dir ~/scratch/datasets/audioset-wds

# Main: 500k version (~50 GB)
huggingface-cli download confit/audioset-16khz-wds \
  --include 500k/train/*.tar \
  --repo-type=dataset \
  --local-dir ~/scratch/datasets/audioset-wds
```

### ESC-50 (Transfer Evaluation)

**Source**: Already available from Experiment 001/002

| Split | Samples | Classes | Duration |
|-------|---------|---------|----------|
| Full | 2,000 | 50 | 5s each |

**Why ESC-50**:
- Small, fast to evaluate
- Standard environmental sound benchmark
- Direct comparison with Exp 002 baselines (36% static, 58% LightCNN, 84% BEATs)

## Student Architecture Candidates

> See research docs for details:
> - [research/design-principles.md](research/design-principles.md) - Key architectural principles
> - [research/student-implementations.md](research/student-implementations.md) - Code for all students

### Recommended: ConvNeXt1D (Primary)

Modern CNN architecture combining best practices from ConvNeXt, MobileNet, and WaveNet:

```
ConvNeXt1D Block:
┌────────────────────────────────────────┐
│  DepthwiseConv1D(kernel=7, dilation=d) │  ← Large kernel, dilated
│  LayerNorm                              │
│  Linear(C → 4C)                         │  ← Inverted bottleneck
│  GELU                                   │
│  Linear(4C → C)                         │
│  + Residual                             │
└────────────────────────────────────────┘
```

**Key design choices**:
- **Depthwise separable convolutions**: 8-9x fewer FLOPs ([MobileNet](https://arxiv.org/abs/1905.02244))
- **Inverted bottleneck**: Expand→Process→Compress ([ConvNeXt](https://github.com/facebookresearch/ConvNeXt))
- **Large kernels (7)**: Better than 3×3 with minimal cost increase
- **Dilated convolutions**: Exponential receptive field ([WaveNet](https://arxiv.org/pdf/1609.03499))
- **LayerNorm**: Works better than BatchNorm for this task

| Config | hidden_dim | blocks | dilations | params | receptive field |
|--------|------------|--------|-----------|--------|-----------------|
| Tiny | 192 | 4 | [1,2,4,8] | ~1.2M | 91 patches |
| **Small** | 256 | 6 | [1,2,4,8,16,32] | ~2.5M | **379 patches** ✓ |
| Base | 384 | 8 | [1,2,4,8,16,32,64,128] | ~6M | 1000+ patches |

**Recommendation**: ConvNeXt1D-Small covers all 248 patches with 2.5M params.

### Alternative 1: Mamba2 (State Space Model) - via HuggingFace

Use `transformers` library for Mamba2 - pure PyTorch, no custom CUDA kernels needed.

```python
from transformers import Mamba2Config, Mamba2Model

class Mamba2Contextualizer(nn.Module):
    def __init__(self, in_dim=768, hidden_size=256, num_layers=4):
        self.proj_in = nn.Linear(in_dim, hidden_size)
        config = Mamba2Config(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            state_size=64,
            head_dim=32,
            num_heads=hidden_size // 32,
        )
        self.mamba = Mamba2Model(config)
        self.proj_out = nn.Linear(hidden_size, in_dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.mamba(inputs_embeds=x).last_hidden_state
        return self.proj_out(x)
```

| Config | hidden_size | layers | params |
|--------|-------------|--------|--------|
| Mamba2-Tiny | 256 | 2 | ~2M |
| Mamba2-Small | 256 | 4 | ~4M |
| Mamba2-Base | 384 | 6 | ~8M |

**Source**: [HuggingFace Mamba2](https://huggingface.co/docs/transformers/model_doc/mamba2)

### Alternative 2: MLP-Mixer

```
Input: (B, 248, 768) → Mixer Blocks × N → Output: (B, 248, 768)
  Token Mixing: MLP across sequence dim (248)
  Channel Mixing: MLP across feature dim (768)
```

| Config | hidden_dim | layers | params |
|--------|------------|--------|--------|
| Mixer-Tiny | 256 | 2 | ~1.5M |
| Mixer-Small | 384 | 4 | ~4M |

**Pros**: Pure PyTorch, fast on CPU, proven architecture
**Cons**: Fixed sequence length

### Alternative 3: TinyTransformer

```
Input: (B, 248, 768) → proj → Transformer × 2 → proj → Output
```

| Config | d_model | heads | layers | params |
|--------|---------|-------|--------|--------|
| Tiny-2L | 256 | 4 | 2 | ~2M |

**Note**: O(n²) attention may be slow on CPU for 248 tokens.

### Architecture Comparison

| Architecture | Params | Receptive Field | Complexity | Library |
|--------------|--------|-----------------|------------|---------|
| **Mamba2-Small** | 4M | Full (O(n) state) | O(n) | transformers |
| ConvNeXt1D-S | 2.5M | Full (379 patches) | O(n) | custom |
| TCN | 2-4M | Full (dilated) | O(n) | pytorch-tcn |
| MLP-Mixer-S | 4M | Full | O(n) | custom |
| TinyTransformer | 2M | Full | O(n²) | custom |

**Recommendation**: Start with Mamba2 (state-of-the-art SSM) and ConvNeXt1D (efficient CNN) as primary candidates.

## Distillation Loss

### Primary: MSE Loss
```python
loss = F.mse_loss(student_embeds, teacher_embeds)
```

### Alternative: Cosine Similarity
```python
# Per-patch cosine similarity
cos_sim = F.cosine_similarity(student_embeds, teacher_embeds, dim=-1)
loss = 1 - cos_sim.mean()
```

### Hybrid (Recommended)
```python
loss = 0.5 * mse_loss + 0.5 * (1 - cosine_sim.mean())
```

## Training Configuration

```yaml
# Pilot (quick validation)
pilot:
  dataset: audioset-20k
  batch_size: 32
  epochs: 10
  samples: 20,000

# Main experiment
main:
  dataset: audioset-500k
  batch_size: 64
  epochs: 3
  samples: ~500,000
  gradient_accumulation: 4

# Full scale (if needed)
full:
  dataset: audioset-2m
  batch_size: 128
  epochs: 1
  samples: ~2,000,000
```

## Evaluation Protocol

### 1. Distillation Quality
- **Metric**: MSE / Cosine similarity on held-out AudioSet
- **Baseline**: Random embeddings, Mean pooling

### 2. AudioSet mAP (Optional)
- Freeze student backbone
- Train linear classifier on AudioSet Balanced
- Evaluate on AudioSet Eval

### 3. ESC-50 Transfer (Primary)
- Freeze student backbone
- Train linear classifier (5-fold CV)
- Compare: Static (36%) < LightCNN (58%) < Student < BEATs (84%)

### 4. Inference Speed
- CPU latency for 5s audio (batch=1)
- Target: <10ms total pipeline

## File Structure

```
003-model-distillation/
├── proposal.md
├── design.md          # This file
├── tasks.md
├── artifacts/
│   ├── checkpoints/   # Student model weights
│   ├── logs/          # Training logs
│   └── results/       # Evaluation JSONs
├── sbatch/
│   ├── download-audioset.sbatch
│   ├── run-distillation.sbatch
│   └── run-evaluation.sbatch
└── results.md         # Final analysis
```

## Future Work (Separate Experiment)

Bioacoustics evaluation will be a follow-up experiment using:
- [BirdSet](https://huggingface.co/datasets/DBD-research-group/BirdSet) - 6,800+ hrs, 10k species
- BirdCLEF competition data
- Domain adaptation from general audio → bird sounds

## References

- [AudioSet 16kHz WebDataset](https://huggingface.co/datasets/confit/audioset-16khz-wds)
- [AudioSet Ontology](https://github.com/audioset/ontology)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [MLP-Mixer](https://arxiv.org/abs/2105.01601)
- [EAT: Self-Supervised Pre-Training with Efficient Audio Transformer](https://arxiv.org/abs/2401.03497)
