# Experiment 007: ConvNeXt-Base LoRA Fine-Tuning on BirdCLEF 2025

**Status**: Completed
**Created**: 2026-02-26
**Author**: Murilo Gustineli

## Hypothesis

ConvNeXt-Base (~89M params, ImageNet-pretrained) with LoRA fine-tuning will produce
comparable or lower cmAP than EAT-base (Exp 006, cmAP=0.536) on BirdCLEF 2025, because
ImageNet pretraining captures visual features while AudioSet pretraining (EAT) captures
audio-specific features directly relevant to bird vocalization classification.

## Background

Exp 006 demonstrated that LoRA fine-tuning EAT-base (~90M params, AudioSet-pretrained)
on BirdCLEF 2025 reaches cmAP=0.536 (best config: `eat_lora_all` with PEFT). The natural
follow-up is: how does a parameter-matched CNN architecture compare? ConvNeXt-Base (~89M)
provides a direct comparison — same param budget, same LoRA approach, but:
- **Pretraining domain**: ImageNet (vision) vs AudioSet (audio)
- **Architecture family**: CNN vs Transformer

ConvNeXt uses `nn.Linear` for its pointwise convolutions (`mlp.fc1`, `mlp.fc2`), so PEFT
LoRA applies directly. No attention layers exist (no qkv/proj), so LoRA targets are
MLP-only. This also means ConvNeXt's spatial mixing (depthwise conv) is NOT adapted by
LoRA — only the channel-mixing MLP layers are.

## Method

Apply PEFT LoRA to ConvNeXt-Base's MLP layers (`fc1`, `fc2`) across all 36 blocks.
Use `NNAudioMelSpectrogram` to convert 32kHz waveforms to mel spectrograms on CPU
(same transform pipeline as exp 006's fbank for EAT).

### Data Flow

```
BirdCLEF2025DataModule (32kHz waveforms, 5s chunks)
  → NNAudioMelSpectrogram transform in dataset workers (CPU)
  → (B, 1, 128, 500) mel spectrograms
  → ConvNeXtEncoder (timm backbone, 1-channel adapted, PEFT-wrapped)
  → (B, 1024) features
  → LoRAFinetuneModule head
  → (B, 206) logits
```

### LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| rank (r) | 8 | Same as exp 006 for fair comparison |
| alpha | 16 | Standard 2x rank scaling |
| dropout | 0.1 | Same as exp 006 |
| target_modules | `["fc1", "fc2"]` | MLP layers (only nn.Linear targets in ConvNeXt) |
| bias | "none" | Same as exp 006 |

Estimated LoRA params: ~1.4M across 36 blocks (compare: EAT attn-only ~451K, EAT all ~1.2M).

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 128 | CNN allows larger batches than transformer (no O(n²) attention maps) |
| lr | 3e-4 | Standard LoRA LR |
| hidden_dim | 0 | Linear head (primary) |
| max_epochs | 50 | With early stopping patience=10 |
| accumulate_grad_batches | 2 | effective batch_size = 128 * 2 = 256 |

### Batch Size Tuning

Exp 006 used `bs=32, accum=8` because EAT's transformer attention stores O(n²) activation
maps per layer, causing OOM at larger batch sizes. ConvNeXt is a CNN with no attention, so
activation memory is much lower per sample.

| Config | GPU Utilization | VRAM Usage | Notes |
|--------|-----------------|------------|-------|
| bs=32, accum=8 | ~45-50% (spiky) | ~8 GB | GPU starving for data |
| bs=128, accum=2 | ~60-70% (stable) | 22.8/24 GB | Near VRAM ceiling |
| bs=128, workers=12 vs 8 | No meaningful difference | 22.8/24 GB | Compute-bound, not data-bound |

Settled on `bs=128, accum=2` — uses available VRAM efficiently while keeping 256 effective
batch size for fair comparison with exp 006.

### Run Comparison

| | `convnext_lora_mlp` (primary) | `convnext_lora_mlp_mlphead` (ablation) |
|---|---|---|
| **LoRA targets** | fc1, fc2 | fc1, fc2 |
| **Head** | Linear (1024→206) | MLP (1024→512→206) |
| **LoRA params** | ~1.4M | ~1.4M |
| **Head params** | ~211K | ~630K |
| **Total trainable** | ~1.6M (~1.8%) | ~2.0M (~2.3%) |

**What each tests:**
1. **`convnext_lora_mlp`** — Baseline: MLP-only LoRA + linear head. Direct comparison with exp 006's `eat_lora_all` (which also targets MLP layers).
2. **`convnext_lora_mlp_mlphead`** — Does a richer classification head compensate for the domain gap (ImageNet vs AudioSet pretraining)?

## Comparison Baseline

| Method | Source | test cmAP |
|--------|--------|-----------|
| EAT LoRA (attn-only, PEFT) | Exp 006 | 0.470 |
| EAT LoRA (all layers, PEFT) | Exp 006 | 0.536 |
| EAT LoRA (attn, MLP head, PEFT) | Exp 006 | 0.532 |
| Frozen linear probe | Exp 005 | 0.301 |

## Evaluation

- **Primary metric**: test cmAP (class-mean average precision)
- **Secondary**: AUROC, F1 macro/micro, top-1/top-5 accuracy
- **Efficiency**: training time, GPU memory, trainable param count

## Success Criteria

- **Interesting if**: ConvNeXt cmAP > 0.4 (ImageNet features transfer meaningfully to audio)
- **Expected**: ConvNeXt cmAP < EAT cmAP (domain-specific pretraining advantage)
- **Surprising if**: ConvNeXt cmAP > EAT cmAP (ImageNet pretraining suffices for audio)

## Limitations

- Only PEFT backend (no adapters comparison — ConvNeXt lacks attention for meaningful adapters exploration)
- LoRA can only target MLP layers in ConvNeXt (no attention to adapt), so the comparison with EAT's attention-only LoRA is somewhat asymmetric
- ImageNet-pretrained weights may not transfer well to 1-channel mel spectrograms despite channel averaging
