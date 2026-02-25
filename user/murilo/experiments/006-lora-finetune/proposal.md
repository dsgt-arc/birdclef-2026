# Experiment 006: LoRA Fine-Tuning EAT on BirdCLEF 2025

**Status**: Proposed
**Created**: 2026-02-23
**Author**: Murilo Gustineli

## Hypothesis

LoRA fine-tuning of EAT-base (~608K trainable params, 0.67% of model) will outperform
the frozen linear probe baseline (cmAP=0.301 from Exp 005) while training in <4 hours
on a single RTX 6000.

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
| target_modules | `["qkv", "proj"]` | Attention layers only (450K LoRA params) |
| bias | "none" | Don't train biases |

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 32 | Activations still backprop through full encoder |
| lr | 3e-4 | Higher than full finetune, standard for LoRA |
| hidden_dim | 0 | Linear head (158K params) |
| max_epochs | 50 | With early stopping patience=10 |
| accumulate_grad_batches | 8 | effective batch_size = 32 * 8 = 256 |

### Run Comparison

| | `eat_lora_attn` (primary) | `eat_lora_all` (ablation) | `eat_lora_attn_mlphead` (ablation) |
|---|---|---|---|
| **LoRA targets** | `qkv`, `proj` | `qkv`, `proj`, `fc1`, `fc2` | `qkv`, `proj` |
| **Head** | Linear (`hidden_dim: 0`) | Linear (`hidden_dim: 0`) | MLP 512 (`hidden_dim: 512`) |
| **LoRA params** | 451K | 1,188K | 451K |
| **Head params** | 158K | 158K | 499K |
| **Total trainable** | 608K (0.67%) | 1,346K (1.47%) | 949K (1.04%) |

**What each tests:**
1. **`eat_lora_attn`** — Baseline LoRA: attention-only adapters + linear head. Cheapest config.
2. **`eat_lora_all`** — Does adapting MLP layers inside transformer blocks (fc1, fc2) add useful capacity beyond attention-only LoRA? Same linear head, so the only variable is LoRA scope.
3. **`eat_lora_attn_mlphead`** — Does a richer classification head help? Same LoRA as primary but with 768→512→206 MLP head. Fair comparison with Exp 005's full finetune which also used `hidden_dim: 512`.

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

> **Note**: Original proposal assumed batch_size=256 without gradient accumulation.
> Fast-dev-run OOM'd — LoRA adapters still require backprop through all encoder layers,
> so activation memory matches full fine-tuning. Fixed to batch_size=32 × accum 8 = 256 effective.
