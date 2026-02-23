# Experiment 006: LoRA Fine-Tuning EAT on BirdCLEF 2025

**Status**: Proposed
**Created**: 2026-02-23
**Author**: Murilo Gustineli

## Hypothesis

LoRA fine-tuning of EAT-base (~600K trainable params, 0.7% of model) will outperform
the frozen linear probe baseline (cmAP=0.301 from Exp 005) while training in <4 hours
with batch_size=256 (no gradient accumulation needed).

## Background

Exp 005's full fine-tuning approach is inefficient: unfreezing 90M EAT encoder params
caused OOM (requiring batch_size=32 + gradient accumulation 8x) and 12h+ training times
that exceeded SLURM limits. LoRA adapts ~0.5% of params instead, solving both memory and
time constraints while potentially matching full fine-tuning performance.

Anthony's existing code (`user/anthony/src/birdclef/finetune_lora.py`) validates that
PEFT LoRA works with EAT's custom architecture. The concept doc
(`docs/concepts/lora-finetuning-audio.md`) maps EAT's layer structure for targeting.

## Method

Apply PEFT LoRA externally to `EATEncoder.model` — no shared `src/` modifications needed.

### LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| rank (r) | 8 | Proven on ESC-50 (Anthony's code) |
| alpha | 16 | Standard 2× rank scaling |
| dropout | 0.1 | Regularization |
| target_modules | `["qkv", "proj"]` | Attention layers only (~442K LoRA params) |
| bias | "none" | Don't train biases |

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 256 | Fits in 24GB — only ~600K trainable params |
| lr | 3e-4 | Higher than full finetune, standard for LoRA |
| hidden_dim | 0 | Linear head (158K params) |
| max_epochs | 50 | With early stopping patience=10 |
| accumulate_grad_batches | 1 | No gradient accumulation needed |

### Ablations

1. **LoRA on all layers** (`eat_lora_all`): `target_modules: ["qkv", "proj", "fc1", "fc2"]` (~1.18M LoRA params). Tests whether adapting MLP layers adds capacity.
2. **LoRA attention + MLP head** (`eat_lora_attn_mlphead`): Same LoRA as primary but `hidden_dim: 512`. Fair comparison with Exp 005 finetune's MLP head.

## Comparison Baseline

| Method | Source | test cmAP |
|--------|--------|-----------|
| Frozen linear probe | Exp 005 | 0.301 |
| Full fine-tune | Exp 005 | OOM / exceeded SLURM time |

## Evaluation

- **Primary metric**: test cmAP (class-mean average precision)
- **Secondary**: AUROC, F1 macro/micro, top-1/top-5 accuracy
- **Efficiency**: training time, GPU memory, trainable param count

## Success Criteria

- test cmAP > 0.301 (frozen baseline)
- Training completes in <4 hours on single RTX 6000
- batch_size=256 fits without gradient accumulation
