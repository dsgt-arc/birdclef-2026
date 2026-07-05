# Concept: Bi-Mamba (1-Bit State Space Models)

**Status**: exploratory/future-work seed
**Created**: 2026-03-31
**Updated**: 2026-03-31

## Idea

Bi-Mamba is a 1-bit quantization scheme for Mamba state space models that binarizes linear layer weights to {+1, -1} with learnable scaling factors, achieving competitive performance with full-precision models at ~32x memory reduction. Applied to SSAMBA for BirdCLEF, this could enable deploying larger models (e.g., SSAMBA-base 99M) within Kaggle's resource constraints, or compress SSAMBA-small for edge deployment on field recorders.

The core technique replaces standard `nn.Linear` layers with FBI-Linear modules: `W_binary = sign(W) * alpha`, where `alpha` is a learnable per-channel scale. Gradients propagate through the non-differentiable `sign()` via Straight-Through Estimator (STE). Models are trained from scratch using autoregressive distillation from a full-precision teacher.

## Motivation

The current SSAMBA pipeline faces a tension between representation quality and deployment constraints:

- **SSAMBA-small (26M)** fits Kaggle easily but may lack capacity for fine-grained bird call discrimination
- **SSAMBA-base (99M)** has better benchmark performance (89.3% vs 85.5% ESC-50) but is heavier at inference
- Kaggle submission environment has strict memory/compute limits (16GB GPU, 9-hour runtime)
- Future deployment on edge devices (e.g., AudioMoth, Raspberry Pi) would require extreme compression

Bi-Mamba could break this tradeoff: train a large model with rich representations, then compress to 1-bit for deployment without proportional quality loss.

## Background

**Bi-Mamba paper** (Tang et al., Nov 2024, [arxiv 2411.11843](https://arxiv.org/abs/2411.11843)):
- Tested at LLM scale: 780M, 1.3B, 2.7B parameters
- FBI-Linear binarization with STE gradient estimation
- Trained from scratch with distillation loss from full-precision teacher
- Maintains linear O(L) complexity advantage of Mamba
- GitHub: [Tangshengku/Bi-Mamba](https://github.com/Tangshengku/Bi-Mamba)

**Related quantization work:**
- BitNet (1-bit Transformers) showed that 1-bit can work for attention-based LLMs
- Q-MambaIR ([arxiv 2503.21970](https://arxiv.org/abs/2503.21970)) applies quantization to Mamba for image restoration
- No existing work applies 1-bit quantization to Mamba for audio or SSL pretraining

**Current SSAMBA context:**
- Exp 103: SSL pretraining with SSAMBA-small, 24 BiMamba layers, 384 embed_dim
- Exp 004: HF port with pure PyTorch fallback (no CUDA kernel dependency)
- Backbone uses Mamba-1 selective scan with d_state=16

## Open Questions

- Does binarization preserve the SSM state dynamics (A, B, C, D matrices) that are critical for temporal modeling of bird vocalizations? These parameters control continuous-valued recurrence — 1-bit may be too coarse.
- Can Bi-Mamba work with SSL objectives (masked patch prediction), or does it require the autoregressive distillation setup from the paper?
- Should all linear layers be binarized, or should SSM-specific projections (dt_proj, x_proj for B/C computation) remain full-precision? The paper targets LLM architectures where all linears are interchangeable; Mamba's SSM projections have different functional roles.
- How does 1-bit interact with the bidirectional scan in SSAMBA? Forward and backward SSMs share in_proj/out_proj — binarizing these shared layers affects both directions simultaneously.
- What is the minimum model size where 1-bit compression is worthwhile? At 26M params (SSAMBA-small), the full-precision model is already only ~100MB. Compression to ~3MB matters for edge but not for Kaggle.
- Does ONNX export support 1-bit custom ops, or would inference need a custom runtime?

## Testable Hypotheses

1. **SSAMBA-base (99M) with Bi-Mamba compression achieves higher BirdCLEF cmAP than full-precision SSAMBA-small (26M) at comparable inference cost.** This is the key value proposition: use quantization to unlock a larger model that would otherwise be too expensive.

2. **Selective binarization (FBI-Linear on in_proj/out_proj only, keep SSM projections full-precision) outperforms full binarization for audio tasks.** The SSM state dynamics may need higher precision than the mixing/projection layers.

3. **Bi-Mamba distillation from a pretrained SSAMBA teacher produces better features than training Bi-Mamba from scratch with SSL.** The paper's distillation approach may be the only viable training strategy, since 1-bit weights may not support the nuanced gradient signal from masked patch reconstruction.

## Connections

- **Based on**: [Bi-Mamba paper](https://arxiv.org/abs/2411.11843), SSAMBA vendor docs (`docs/vendor/ssamba.md`)
- **Related to**: `docs/concepts/ssamba-audio-mamba.md`, `docs/concepts/distilled-sequence-model.md`, exp 103 (SSL pretraining), exp 004 (HF port)
- **Could lead to**: Experiment proposal for Bi-Mamba distillation of trained SSAMBA-base; edge deployment pipeline for field monitoring

## Notes

**Priority assessment**: This is a deployment/compression optimization, not a representation quality improvement. The current bottleneck is getting SSAMBA to produce good features (exp 103 success gate: linear probe cmAP > 0.20). Bi-Mamba becomes relevant only after we have a strong full-precision teacher model to distill from.

**Sequencing**:
1. First: complete exp 103 SSL pretraining, evaluate linear probe quality
2. If SSAMBA-base is needed: train full-precision SSAMBA-base (exp 103-F)
3. Then: distill SSAMBA-base into Bi-Mamba-base for Kaggle deployment
4. Alternative: skip Bi-Mamba, use standard INT8/FP16 quantization which is simpler and well-supported by ONNX

**Comparison to simpler quantization**: Standard post-training quantization (PTQ) to INT8 gives ~4x compression with minimal quality loss and full ONNX support. Bi-Mamba's ~32x compression is only justified if INT8 is insufficient for the deployment target. For Kaggle, INT8 or even FP16 is likely enough.

**Mamba-2 interaction**: If we upgrade to Mamba-2 SSD (structured state space duality), the binarization approach would need adaptation — Mamba-2 uses chunked matrix multiplies rather than sequential scans, and the weight structure differs (multi-head SSM). The Bi-Mamba paper targets Mamba-1 only.
