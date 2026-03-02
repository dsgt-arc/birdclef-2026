# Experiment 009: ProtoCLR (CvT-13) LoRA Fine-Tuning

## Hypothesis

ProtoCLR's domain-invariant representations (trained on Xeno-Canto with prototypical
contrastive learning) will provide complementary features to EAT and ConvNeXt for bird
sound classification, making it a valuable addition to the teacher model ensemble for
pseudo-labeling unlabeled soundscapes.

## Background

BirdCLEF's main challenge is the domain shift between clean labeled training data (focal
recordings) and noisy unlabeled soundscapes. Having diverse teacher models with different
inductive biases improves pseudo-label quality. Previous experiments tested:
- **Exp 006**: EAT (Transformer, AudioSet pretrained) -- best cmAP: 0.536
- **Exp 007**: ConvNeXt-Base (CNN, ImageNet pretrained) -- best cmAP: 0.527

ProtoCLR is a compact (19M params) CvT-13 model trained with prototypical contrastive
learning on Xeno-Canto (~685K segments, 10K species). It was designed for domain-invariant
representation learning -- exactly the focal-to-soundscape transfer problem in BirdCLEF.

**Key architectural differences**:
- CvT-13 uses convolutional tokenization (no fixed patch size) with progressive downsampling
- 3-stage architecture: [1, 2, 10] blocks at [64, 192, 384] dims
- Separate Q/K/V projections with depthwise convolutional preprocessing
- No positional embeddings (position-agnostic, handles variable-length inputs)
- 384-dim output embeddings (vs 768 for EAT, 1024 for ConvNeXt)

## Method

### Data Flow

```
16kHz waveform -> ProtoCLR mel spectrogram (CPU, dataloader workers)
    -> torchaudio MelSpectrogram (n_fft=1024, hop=320, 128 mels, 50-8000 Hz)
    -> AmplitudeToDB -> min-max norm -> standardize (mean=0.5347, std=0.0772)
    -> (batch, 1, 128, T) tensor
    -> LoRA-adapted CvT-13 encoder
    -> (batch, 384) embeddings
    -> classification head
    -> (batch, 206) logits
```

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Bias | none |

**Configs**:
1. `protoclr_lora_attn.yaml` -- attention projections only
   - Target modules: `attn.proj_q`, `attn.proj_k`, `attn.proj_v`, `attn.proj`
2. `protoclr_lora_all.yaml` -- attention + MLP layers
   - Target modules: `attn.proj_q`, `attn.proj_k`, `attn.proj_v`, `attn.proj`, `mlp.fc1`, `mlp.fc2`

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 128 | CvT-13 is small (19M), similar to ConvNeXt setup |
| Effective batch | 256 | accumulate_grad_batches=2 |
| Learning rate | 3e-4 | Same as exp 006/007 |
| Max epochs | 50 | |
| Early stopping | patience=10 | Monitor val/cmap |
| Precision | 16-mixed | |
| Chunk duration | 5.0s | Consistent with exp 006/007 for fair comparison |
| Sample rate | 16000 | ProtoCLR native rate |

## Success Criteria

- cmAP > 0.4 demonstrates ProtoCLR is viable for BirdCLEF teacher models
- Interesting to see if domain-invariant pretraining helps despite smaller model size
- Comparison with EAT (0.536) and ConvNeXt (0.527) reveals inductive bias effects

## Model References

- HuggingFace: [ilyassmoummad/ProtoCLR](https://huggingface.co/ilyassmoummad/ProtoCLR)
- GitHub: [ilyassmoummad/ProtoCLR](https://github.com/ilyassmoummad/ProtoCLR)
- Paper: [ProtoCLR: Prototypical Contrastive Learning for Domain-Invariant Representations](https://arxiv.org/abs/2409.08589)
