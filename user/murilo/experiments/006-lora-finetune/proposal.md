# Experiment 006: LoRA Fine-Tuning EAT on BirdCLEF 2025

**Status**: In-progress
**Created**: 2026-02-23
**Updated**: 2026-02-25
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

Apply LoRA externally to `EATEncoder.model` — no shared `src/` modifications needed.
Each config is run with two independent backends (PEFT and adapters) for a total of 6 runs.

### LoRA Backends

**PEFT** (`birdclef.lora.peft`): HuggingFace PEFT library. Wraps `encoder.model` via
`get_peft_model()`. Straightforward, widely used.

**Adapters** (`birdclef.lora.adapters`): adapter-hub/adapters library. Uses
`AdapterModelInterface` to map EAT's layer structure to the adapters plugin system.
Required three workarounds for EAT compatibility (see "Upstream Contributions" below).
Provides an alternative backend that produces comparable or better results.

### LoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| rank (r) | 8 | Proven on ESC-50 (Anthony's code) |
| alpha | 16 | Standard 2× rank scaling |
| dropout | 0.1 | Regularization |
| target_modules | `["qkv", "proj"]` | Attention layers only (450K LoRA params) |
| bias | "none" | Don't train biases (PEFT); adapters defaults |

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 32 | Activations still backprop through full encoder |
| lr | 3e-4 | Higher than full finetune, standard for LoRA |
| hidden_dim | 0 | Linear head (158K params) |
| max_epochs | 50 | With early stopping patience=10 |
| accumulate_grad_batches | 8 | effective batch_size = 32 * 8 = 256 |

### Run Comparison

Each config below is run once with PEFT and once with adapters (6 total runs).

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

**What PEFT vs adapters tests:**
Running identical configs on both backends isolates library-level differences (fused QKV
handling, LoRA initialization, parameter counting) from architectural choices.

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

## Upstream Contributions (adapter-hub/adapters)

Integrating EAT with the adapters library required three workarounds that point to
upstream fixes:

1. **Fused QKV LoRA shape mismatch** — `LoRAMergedLinear.get_n_heads()` counts all
   `attn_matrices` including `"o"`, but the fused QKV layer only has Q/K/V. This
   produces a grouped conv1d sized for 4 heads with `enable_lora` for 3, causing a shape
   error. Fix: only count `{"q", "k", "v"}` in `get_n_heads` for merged layers.

2. **`get_input_embeddings()` NotImplementedError** — EAT is an audio model without a
   token vocabulary. The adapters `EmbeddingAdaptersMixin` calls
   `get_input_embeddings()` during init, which raises `NotImplementedError` on
   `EATModel`. Workaround: monkey-patch to return `model.local_encoder` (patch embedding).

3. **EAT as a supported model** — EAT follows standard ViT structure (fused QKV, same
   as ModernBERT). An `AdapterModelInterface` plugin makes it work without modifying
   the adapters library. This could be contributed as a built-in model support.

**Planned PRs:**
- Fix `LoRAMergedLinear.get_n_heads` for fused QKV layers (bug fix)
- Add EAT model support via `AdapterModelInterface` (new model)

> **Note**: Original proposal assumed batch_size=256 without gradient accumulation.
> Fast-dev-run OOM'd — LoRA adapters still require backprop through all encoder layers,
> so activation memory matches full fine-tuning. Fixed to batch_size=32 × accum 8 = 256 effective.
