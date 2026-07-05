# Concept: Pretraining Compute Feasibility

**Status**: historical
**Created**: 2026-01-30
**Updated**: 2026-01-30

## Idea

Given a budget of a few hundred GPU-hours on PACE (V100/RTX 6000/L40s), what is the most effective way to train an audio model with a custom architecture (e.g., non-square patches, Mamba backbone) for BirdCLEF? Full SSL pretraining à la BEATs/SSAST on AudioSet is out of reach at 1000-20000 GPU-hours, but there's a spectrum of cheaper alternatives: distillation from a pretrained teacher, patch embedding surgery on existing checkpoints, or small-scale SSL pretraining on a data subset. The question is which approach gives the best accuracy-per-GPU-hour, and whether any of them can justify the compute over simply finetuning existing pretrained models.

## Motivation

We want to explore non-square patch shapes (128×2 frame-level tokenization) for bioacoustics, but all published pretrained weights use 16×16 patches. This means we can't just download a checkpoint -- we need to train something. With a hard compute budget, we need to understand upfront whether the expected accuracy gain justifies the cost, and which training strategy maximizes return on that budget.

The risk of getting this wrong is real: Exp 002 showed that a clever inference optimization (static embeddings) looked promising on paper but failed in practice (48% accuracy drop). We don't want to burn hundreds of GPU-hours on pretraining only to find it underperforms a simple finetuned SSAST-base with square patches.

## Background

### Published pretraining costs

| Model | Params | Data | Hardware | Est. GPU-Hours | ESC-50 |
|-------|--------|------|----------|---------------|--------|
| BEATs (3 iterations) | 90M | AudioSet-2M (5.8k hrs) | 32-64× A100 | ~20,000 | 95.8% (linear) |
| SSAST-base | 87M | AudioSet + LibriSpeech | 4× GTX Titan | ~500-1000 (est.) | ~88% (linear) |
| SSAST-tiny | 6M | AudioSet + LibriSpeech | 4× GTX Titan | ~50-100 (est.) | lower |
| EAT | 87M | AudioSet-2M | A100s | ~1,300 (15× faster than BEATs) | 96.2% |
| SSAMBA-base | 99M | AudioSet + LibriSpeech | unknown | unknown | 89.3% |

Note: SSAST/SSAMBA GPU-hours are estimates -- not published by the authors.

### Available compute on PACE

- GPU types: V100-16GB, V100-32GB, RTX 6000-24GB, L40s
- Typical allocation: single-GPU jobs, 4-8 hour walltime
- V100-16GB OOMs on full SSAMBA finetuning (Exp 001)
- RTX 6000-24GB hits walltime limits on longer training
- Practical budget: ~200-500 GPU-hours total

### What we already have cached

- 500k AudioSet clips with BEATs teacher embeddings and token indices (350GB, from Exp 003)
- ESC-50 benchmarks for all candidate architectures
- SSAMBA-HF pure PyTorch implementation (Exp 004)

### Training strategies (cheapest to most expensive)

| Strategy | Est. GPU-Hours | What You Get |
|----------|---------------|-------------|
| **A. Distillation** (student from scratch, teacher embeddings cached) | 10-50 | Small model, any architecture/patch shape, but limited to teacher's representation quality |
| **B. Patch embedding surgery** (replace Conv2d in SSAST, finetune all) | 5-20 | Keeps pretrained transformer layers, new patch shape, but patch embedding is random |
| **C. Small SSL pretrain** (SSAST-tiny, 128×2, AudioSet-500k) | 50-200 | Learned representations with custom patches, but small model + small data |
| **D. Full SSL pretrain** (SSAST-base, 128×2, AudioSet-2M) | 500-1000+ | Best representations with custom patches, but may exceed budget |

### References

- [BEATs architecture analysis](../references/deep-research/beats-architecture.md)
- [SSAST paper (AAAI 2022)](https://sls.csail.mit.edu/publications/2022/YGong_AAAI-2022.pdf) - flexible fshape/tshape, 4× Titan training
- [EAT paper (IJCAI 2024)](https://arxiv.org/abs/2401.03497) - 15× speedup over BEATs pretraining
- [AST paper](https://github.com/YuanGongND/ast) - 128×2 > 16×16 from scratch

## Open Questions

- **What's the minimum pretraining data size that still produces useful representations?** AudioSet-2M is standard but AudioSet-500k (already cached) would cut cost ~4×. Is there a cliff where SSL pretraining stops working?
- **How much does model size matter for SSL?** SSAST-tiny (6M) vs base (87M) -- does the smaller model learn useful representations at all, or is there a minimum capacity needed for masked prediction to work?
- **Is distillation + non-square patches strictly better than SSL + non-square patches?** Distillation inherits the teacher's representations (which were learned with 16×16 patches). SSL learns its own representations with the new patch shape. Which produces better features for downstream classification?
- **Can we estimate expected accuracy before committing compute?** E.g., run a quick 1-epoch pretraining, evaluate linear probe, and extrapolate the learning curve to decide whether to continue.
- **How does patch embedding surgery compare?** Replacing only the Conv2d layer in a pretrained SSAST-base is almost free compute-wise. If it gets within 5% of full retraining, there's no reason to pretrain from scratch.
- **What is the actual GPU-hour cost of SSAST-tiny pretraining on PACE hardware?** Published numbers are on Titan GPUs. V100/L40s have different throughput. Need a concrete estimate before committing.

## Testable Hypotheses

1. **Distillation is more compute-efficient than SSL pretraining**: A distilled student (Strategy A, ~20 GPU-hours) will achieve higher ESC-50 accuracy than an SSL-pretrained model of the same size (Strategy C, ~100 GPU-hours), because the teacher provides a stronger learning signal than masked prediction.

2. **Patch embedding surgery preserves most of the pretrained quality**: Replacing the patch embedding in SSAST-base from 16×16 to 128×2 and finetuning (Strategy B) will retain >90% of the original model's linear probe accuracy on ESC-50, because the transformer layers' learned attention patterns transfer across patch shapes.

3. **There's a data efficiency threshold for SSL**: SSL pretraining on AudioSet-500k will produce representations within 5% of AudioSet-2M pretraining on downstream ESC-50, because the marginal value of additional unlabeled data diminishes.

4. **Early stopping is predictive**: Linear probe accuracy after 10% of pretraining epochs correlates >0.8 with final accuracy, enabling a cheap "go/no-go" decision before committing full budget.

## Connections

- **Based on**: [temporal-resolution-sequence-length](temporal-resolution-sequence-length.md) (non-square patch motivation), [Exp 003 distillation](../../user/anthony/experiments/003-model-distillation/proposal.md) (cached teacher embeddings), [Exp 001 benchmarks](../../user/anthony/experiments/001-encoder-benchmark/results.md)
- **Related to**: [distilled-sequence-model](distilled-sequence-model.md), [context-recovery](context-recovery.md), [ssamba-audio-mamba](ssamba-audio-mamba.md)
- **Could lead to**: Experiment comparing strategies A-D on ESC-50 with fixed compute budget; go/no-go pilot for SSAST-tiny SSL pretraining with 128×2 patches

## Notes

### The "is it worth it" decision framework

Before committing hundreds of GPU-hours, we should establish baselines that cost almost nothing:

1. **Baseline 0**: Existing SSAST-base with 16×16 patches, finetuned on ESC-50 (already done in Exp 001)
2. **Baseline 1**: Patch embedding surgery (Strategy B) -- ~5 GPU-hours, tells us the ceiling for "free" adaptation
3. **Baseline 2**: Distillation with 128×2 student (Strategy A) -- ~20 GPU-hours, tests whether the teacher signal + new patches already gets us where we want

Only if baselines 1 and 2 are clearly insufficient should we invest in SSL pretraining (Strategies C/D). This is a sequential decision: cheapest experiments first, escalate only if needed.

### Compute budget allocation sketch

With ~300 GPU-hours total:
- 20 hrs: Baseline experiments (surgery + distillation)
- 30 hrs: Hyperparameter sweeps on winning baseline
- 100 hrs: If SSL looks necessary, pilot SSAST-tiny pretraining with early stopping
- 150 hrs: Reserve for final training run or BirdCLEF-specific finetuning

### V100 vs L40s throughput

Need to benchmark actual training throughput on PACE hardware. Published numbers on Titan/A100 don't directly translate. Key variables:
- Memory bandwidth (V100: 900 GB/s, L40s: 864 GB/s)
- FP16/BF16 throughput
- Batch size limits per GPU memory
- Data loading bottlenecks from scratch storage vs home directory
