# Project Status

> **Last Updated**: January 2026

## Research Thesis

Foundation models like BEATs/AudioMAE learn superior bioacoustic representations but are excluded from BirdCLEF due to CPU inference constraints. This project explores strategies to unlock foundation model quality under CPU limits.

## Current State

**Phase**: Foundation Model Evaluation

Experiment 001 (encoder benchmark) is complete. Key result: **BEATs is best for frozen features; EAT+LoRA is best for finetuning.**

## Key Findings So Far

| Finding | Source | Implication |
|---------|--------|-------------|
| BEATs achieves 95.8% on ESC-50 (linear probe) | [Exp 001](../user/anthony/experiments/001-encoder-benchmark/results.md) | Best frozen feature extractor |
| EAT+LoRA achieves 94.6% on ESC-50 | [Exp 001](../user/anthony/experiments/001-encoder-benchmark/results.md) | LoRA closes gap with published results |
| wav2vec2 fails on environmental sounds | [Exp 001](../user/anthony/experiments/001-encoder-benchmark/results.md) | Speech models don't transfer to ESC |
| **Static embeddings lose 48% accuracy** | [Exp 002](../user/anthony/experiments/002-beats-inference-optimization/results.md) | Model2vec approach fails for transformers |
| FAISS gives 32x CPU speedup | [Exp 002](../user/anthony/experiments/002-beats-inference-optimization/results.md) | But accuracy tradeoff unacceptable |
| **Contextual pooling recovers to 63%** | [Exp 002](../user/anthony/experiments/002-beats-inference-optimization/results.md) | CNN1DPooler best, but still 21% gap |
| EAT is 15x more efficient than BEATs | [literature.md](references/literature.md) | May be better distillation target |
| OpenBEATs (300M) is SOTA on bioacoustics | [literature.md](references/literature.md) | Best available teacher model |
| BirdMAE wins on BirdSet benchmark | [literature.md](references/literature.md) | Domain-specific training helps |
| Focal→soundscape domain gap is critical | [soundscape-domain-shift.md](concepts/soundscape-domain-shift.md) | Must address in pipeline |

## Active Research Directions

1. **Distillation target selection** - EAT vs OpenBEATs vs BirdMAE
2. **CPU-viable architectures** - MLP-Mixer, model2vec, quantization
3. **Domain adaptation** - ProtoCLR, MixIT denoising
4. **ViT for bioacoustics** - Validating supervised ViT baseline on BirdSet (Exp 002)

See [concepts/research-leads.md](concepts/research-leads.md) for full list.

## What Works (Baselines)

| System | Performance | CPU Viable? |
|--------|-------------|-------------|
| BirdNET | Production-grade, 6500 species | Yes |
| Perch | 15k species, EfficientNet-B3 | Yes |
| BEATs | SOTA representations | No (too slow) |

## Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| anthony/001 | [Encoder Benchmark](../user/anthony/experiments/001-encoder-benchmark/) | Complete | BEATs 95.8%, EAT+LoRA 94.6% |
| anthony/002 | [BEATs Inference Optimization](../user/anthony/experiments/002-beats-inference-optimization/) | Complete | **Rejected** - 32x speedup, 63% best accuracy (21% gap) |
| anthony/003 | [Model Distillation](../user/anthony/experiments/003-model-distillation/) | Proposed | Distill BEATs → lightweight student on AudioSet |
| murilo/001 | [BirdSet Bioacoustics Benchmark](../user/murilo/experiments/001-birdset-bioacoustics-benchmark/) | **Phase 2A Complete** | ConvNeXT/EfficientNetB1 on 8 datasets, mean cmAP 0.36/0.35. Phase 2B deferred. |
| murilo/002 | [Supervised ViT Baseline](../user/murilo/experiments/002-supervised-vit-baseline/) | Ready for execution | HSN single-GPU baseline |
| murilo/003 | Audio-MAE Pre-training | **Deferred** (2026-02-01) | Planning complete, archived |

Experiments are tracked per-user in `user/<name>/experiments/`.

## Next Steps

1. ~~Benchmark foundation models (BEATs, EAT, wav2vec2) on ESC-50~~ Done (Exp 001)
2. ~~Optimize BEATs inference for CPU~~ Done (Exp 002) - static embeddings rejected (63% best, 21% gap)
3. **[Active]** Knowledge distillation on AudioSet (Exp 003) - Mamba2, ConvNeXt1D, TCN students
4. **[Active]** Execute Experiment 002 on HSN (single GPU, 2-4 hours)
5. Evaluate HSN results and compare to BirdSet baseline (ConvNeXT: 0.38 cmAP)
6. Decide next direction: scale to XCM, iterate architecture, or explore BEATS/EATS

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Literature | [references/literature.md](references/literature.md) |
| Vendor Docs | [vendor/](vendor/) |
| Templates | [_templates/](_templates/) |
