# Data and Compute Scale Analysis

**Status**: historical

Understanding the scale of data and compute required for different approaches to BirdCLEF.

## BirdCLEF Competition Data (2022-2025)

| Year | Region | Training Recordings | Species | Test Format | Special Constraints |
|------|--------|---------------------|---------|-------------|---------------------|
| [2022](https://www.kaggle.com/competitions/birdclef-2022) | Hawaii | ~15,000 | 152 (21 target) | Soundscapes | Few-shot for endangered species |
| [2023](https://www.kaggle.com/competitions/birdclef-2023) | Kenya | 16,900 | 264 | 191 × 10-min soundscapes | 2-hour inference limit |
| [2024](https://www.kaggle.com/competitions/birdclef-2024) | Western Ghats, India | ~240,000 | ~180 | Soundscapes | **CPU-only, 120-min limit** |
| [2025](https://www.kaggle.com/c/birdclef-2025) | Colombia | TBD | Multi-taxa* | Soundscapes | CPU-only, $50K prize |

*BirdCLEF+ 2025 includes birds, amphibians, mammals, and insects.

**Key trend**: Competitions increasingly constrain inference (CPU-only, time limits) while expanding taxonomic scope.

## BirdSet Dataset

| Dataset | Purpose | Recordings | Species | Size (GB) | Hours |
|---------|---------|------------|---------|-----------|-------|
| **XCL** | Large-scale pre-training | 528,434 | 9,734 | ~480 | ~6,800 |
| **XCM** | Targeted training (test species) | 89,798 | 409 | ~90 | ~1,100 |
| **POW** | Eval - Powdermill | 14,911 train / 4,560 test | 48 | - | - |
| **PER** | Eval - Amazon Basin | 16,802 train / 15,120 test | 132 | - | - |
| **NES** | Eval - Costa Rica | 16,117 train / 24,480 test | 89 | - | - |
| **UHH** | Eval - Hawaii | 3,626 train / 36,637 test | 25 | - | - |
| **HSN** | Eval - High Sierras | 5,460 train / 12,000 test | 21 | - | - |
| **SSW** | Eval - Sapsucker Woods | 28,403 train / 205,200 test | 81 | - | ~285 |
| **SNE** | Eval - Sierra Nevada | 19,390 train / 23,756 test | 56 | - | - |
| **NBP** | Eval - NIPS4BPlus | 24,327 train / 563 test | 51 | - | - |

**Total**: 6,800+ training hours, 400+ test hours across 8 evaluation datasets.

Source: [BirdSet NeurIPS 2024](https://arxiv.org/abs/2403.10380)

## Model Parameter Counts

### Production Bird Classifiers

| Model | Architecture | Parameters | Size (FP32) | GFLOPs | Species |
|-------|--------------|------------|-------------|--------|---------|
| [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer) | EfficientNet-like | ~9M | 36 MB | 0.7 | 6,500+ |
| [Perch 1.0](https://www.kaggle.com/models/google/bird-vocalization-classifier) | EfficientNet-B1 | ~78M | - | - | 10,000+ |
| [Perch 2.0](https://arxiv.org/abs/2508.04665) | EfficientNet-B3 | ~120M | - | ~1.8 | 15,000 |

### EfficientNet Family (Backbone Options)

| Model | Parameters | GFLOPs | Top-1 ImageNet | CPU Inference* |
|-------|------------|--------|----------------|----------------|
| EfficientNet-B0 | 5.3M | 0.39 | 77.1% | ~15ms |
| EfficientNet-B1 | 7.8M | 0.70 | 79.1% | ~25ms |
| EfficientNet-B3 | 12M | 1.8 | 81.6% | ~50ms |
| EfficientNet-B5 | 30M | 9.9 | 83.6% | ~150ms |

*Rough estimates for single image on modern CPU

### Self-Supervised / ViT Models

| Model | Architecture | Parameters | Pre-training Data |
|-------|--------------|------------|-------------------|
| [ProtoCLR](https://arxiv.org/abs/2409.08589) | CvT-13 | 19M | Xeno-Canto (685K segments) |
| [BirdAVES-base](https://github.com/earthspecies/biodenoising) | ViT | 95M | Animal vocalizations |
| [BirdAVES-large](https://github.com/earthspecies/biodenoising) | ViT | 316M | Animal vocalizations |
| wav2vec 2.0 base | Transformer | 95M | 960h LibriSpeech |
| wav2vec 2.0 large | Transformer | 317M | 60,000h LibriVox |
| HuBERT base | Transformer | 95M | 960h LibriSpeech |
| HuBERT large | Transformer | 317M | 60,000h LibriVox |

### MLP-Mixer (Efficient Alternative)

| Model | Parameters | GFLOPs | ImageNet Top-1 |
|-------|------------|--------|----------------|
| Mixer-S/16 | 18M | 3.8 | 73.8% |
| Mixer-B/16 | 59M | 12.6 | 78.5% |

### Neural Audio Codecs

| Codec | Parameters | Tokens/sec | Bitrate | Training Data |
|-------|------------|------------|---------|---------------|
| [WavTokenizer-small](https://github.com/jishengpeng/WavTokenizer) | ~50M | 75 | 0.9 kbps | 585h LibriTTS |
| [WavTokenizer-large](https://github.com/jishengpeng/WavTokenizer) | ~100M | 40-75 | 0.9 kbps | 80,000h mixed |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | ~70M | 86 | 8 kbps | Large mixed |
| [EnCodec](https://github.com/facebookresearch/encodec) | ~50M | 75 | 1.5-24 kbps | Large mixed |

## Training Time Estimates

### Self-Supervised Pre-training (from scratch)

| Model | GPU Hours | Hardware | Data |
|-------|-----------|----------|------|
| wav2vec 2.0 base | 2,400 | V100 | 960h audio |
| HuBERT base | 1,000 | A100 | 960h audio |
| data2vec 2.0 | 700 | A100 | 960h audio |
| BEST-RQ | ~500 | A100 | 960h audio |

Source: [Open Implementation of BEST-RQ](https://arxiv.org/abs/2405.04296)

### Supervised Training Estimates

| Approach | Dataset | GPU Hours (est.) | Hardware | Notes |
|----------|---------|------------------|----------|-------|
| EfficientNet-B1 on XCM | 90K recordings | 20-40 | 1× A100 | Fine-tune from ImageNet |
| EfficientNet-B3 on XCM | 90K recordings | 40-80 | 1× A100 | Fine-tune from ImageNet |
| EfficientNet-B3 on XCL | 530K recordings | 200-400 | 4× A100 | Full Perch-style training |
| ConvNeXT-base on XCM | 90K recordings | 60-100 | 1× A100 | Fine-tune from ImageNet |
| ViT-B on XCM | 90K recordings | 80-120 | 1× A100 | Fine-tune from ImageNet/AudioSet |

### Our Pipeline Estimates

| Phase | Approach | GPU Hours (est.) | Notes |
|-------|----------|------------------|-------|
| **Phase 1**: Tokenizer | Use pre-trained WavTokenizer | 0 | Just inference |
| **Phase 1**: Tokenizer | Train WavTokenizer on bird audio | 100-500 | 1× A100, depends on data |
| **Phase 2**: Transformer | HuBERT-style on XCM tokens | 200-400 | 4× A100, masked prediction |
| **Phase 2**: Transformer | Smaller transformer on XCM | 50-100 | 1× A100 |
| **Phase 3**: Distillation | model2vec style | 1-2 | CPU, 30 seconds per vocab |
| **Phase 4**: Classifier | Linear probe on embeddings | 1-5 | 1× GPU |

### MLP-Mixer Distillation Path

| Step | GPU Hours (est.) | Notes |
|------|------------------|-------|
| Fine-tune ProtoCLR on XCM | 20-40 | 1× A100 |
| Distill to Mixer-S/16 | 40-80 | 1× A100 |
| Quantize to INT8 | <1 | CPU |
| **Total** | 60-120 | |

## Kaggle Inference Constraints

### BirdCLEF 2024 Requirements
- **CPU only** (no GPU)
- **120 minutes** total inference time
- Must process all test soundscapes within limit

### What Fits in 120 Minutes?

Assuming ~500 test files × 10 min each = 5000 minutes of audio to process.
With 120 min limit → need to process ~42 min of audio per minute of wall time.

| Model | Est. Time per 5s Clip | Clips in 120 min | Feasible? |
|-------|----------------------|------------------|-----------|
| EfficientNet-B0 (FP32) | ~20ms | 360,000 | ✅ Yes |
| EfficientNet-B0 (INT8) | ~10ms | 720,000 | ✅ Yes |
| EfficientNet-B3 (FP32) | ~60ms | 120,000 | ✅ Tight |
| EfficientNet-B3 (INT8) | ~30ms | 240,000 | ✅ Yes |
| Mixer-S/16 (FP32) | ~50ms | 144,000 | ✅ Yes |
| Mixer-S/16 (INT8) | ~25ms | 288,000 | ✅ Yes |
| ViT-B/16 (FP32) | ~500ms | 14,400 | ❌ No |
| wav2vec 2.0 base | ~200ms | 36,000 | ⚠️ Maybe |

**Winning strategies from BirdCLEF 2024**:
- 3rd place: EfficientViT-B0 + MNASNet, ONNX, 5 folds in 40 min
- 4th place: OpenVINO + INT8 quantization, 30-40% speedup
- Small ensembles of fast models beat single large models

## Recommendations

### Compute Budget: Low (1× consumer GPU, ~100 GPU hours)
- Fine-tune EfficientNet-B0/B1 on XCM
- Use pre-trained BirdNET/Perch embeddings + linear probe
- Skip self-supervised pre-training

### Compute Budget: Medium (1× A100, ~500 GPU hours)
- Fine-tune EfficientNet-B3 or ConvNeXT on XCM
- Try ProtoCLR → Mixer distillation path
- Experiment with pre-trained WavTokenizer + small transformer

### Compute Budget: High (4× A100, ~2000 GPU hours)
- Train on full XCL dataset
- Self-supervised pre-training (HuBERT-style)
- Train custom neural codec on bird audio
- Approach Perch 2.0 scale

### For Kaggle Specifically
1. **Must have**: INT8 quantization, ONNX/OpenVINO export
2. **Sweet spot**: EfficientNet-B0/B1 or EfficientViT-B0
3. **Ensemble**: 3-5 small fast models > 1 large slow model
4. **Avoid**: ViT-B/L, unquantized large models, complex pipelines

## References

- [BirdSet: A Large-Scale Dataset for Audio Classification](https://arxiv.org/abs/2403.10380)
- [Perch 2.0: The Bittern Lesson for Bioacoustics](https://arxiv.org/abs/2508.04665)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [wav2vec 2.0: Self-Supervised Learning of Speech](https://arxiv.org/abs/2006.11477)
- [Open Implementation of BEST-RQ](https://arxiv.org/abs/2405.04296)
- [BirdCLEF 2024 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2024)
- [BirdCLEF+ 2025 Competition](https://www.imageclef.org/BirdCLEF2025)
