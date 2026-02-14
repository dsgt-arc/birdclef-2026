# Reference: Self-Supervised Pretraining for Frame-Level Audio Tokenization

**Source**: Multi-paper literature review (wav2vec 2.0, HuBERT, data2vec, BEATs, Audio-MAE, Bird-MAE, BEST-RQ, SSAMBA, Yadav 2025 survey)
**Date Captured**: 2026-02-14
**Type**: deep-research

## Summary

Standard masked patch contrastive (MPC/NCE) pretraining fails with frame-level tokenization (128x1 patches, 1 mel frame per token) because adjacent mel frames are nearly identical, making the contrastive "which patch does this prediction belong to?" task degenerate. This document surveys how the speech and audio SSL literature solves this problem, and evaluates each approach for our SSAMBA Tiny (7M params) + Mamba backbone + 1000 tokens/clip setup on bird audio.

## Key Points

- **The core problem**: NCE contrastive loss requires negative samples to be distinguishable from positives. With 128x1 patches, all frames within a bird call have nearly identical mel spectra, so the 780-way classification task has ~uniform similarity matrix. MPC accuracy stays at random (1/780 = 0.00128).
- **Three families of solutions**: (1) discrete targets that cluster similar frames (HuBERT, BEST-RQ, wav2vec 2.0), (2) contextualized regression targets from a teacher (data2vec), (3) pure reconstruction without contrastive (MPG-only / Audio-MAE).
- **Yadav et al. 2025 explicitly validated MSE-only with Mamba** on masked spectrogram modeling, finding it sufficient and that Mamba substantially outperforms Transformers (87.9 vs 68.2 aggregated score across 10 tasks).
- **Bird-MAE achieves 55 cmAP on BirdSet HSN** using pure MSE reconstruction with ViT-Large + 16x16 patches — current bird audio SOTA.

## The Problem in Detail

Our SSAMBA pretraining uses joint MPC + MPG (from the original SSAMBA paper):

```
L = L_MPC + 10 * L_MPG
```

**MPC (Masked Patch Classification)**: InfoNCE contrastive loss. For each masked position, predict a vector that matches the raw mel patch at that position vs all other masked positions. The similarity matrix is `raw_patches @ predictions^T` with shape `(M, M)` where M=780 masked tokens.

**Why MPC fails at 128x1**: With 16x16 patches (original SSAMBA), each patch captures a distinct time-frequency region — a low-frequency patch looks nothing like a high-frequency patch. But with 128x1, every patch spans ALL frequencies for a single frame. Bird calls produce sustained spectral patterns, so frame 247 and frame 248 have nearly identical 128-dim mel vectors. The 780x780 similarity matrix becomes near-uniform.

**Empirical evidence**: After 50 epochs on 4 repeated batches, MPC accuracy = 0.002 vs random = 0.00128. The model cannot learn this task.

**MPG (Masked Patch Generation)**: MSE reconstruction. This IS learning (loss 10.4 → 6.7) because it doesn't require distinguishing between positions — each masked frame is independently reconstructed.

## Approach 1: Drop MPC, Keep MPG Only

**Source**: Yadav et al. 2025, "Neural architectures for self-supervised audio from masked spectrograms" ([arXiv:2509.18691](https://arxiv.org/abs/2509.18691)); Audio-MAE (Huang et al., NeurIPS 2022, [arXiv:2207.06405](https://arxiv.org/abs/2207.06405))

**Method**: Pure MSE reconstruction on masked positions. No contrastive loss.

```
L = MSE(predicted_patches, true_patches)  # only at masked positions
```

**Evidence for Mamba**: Yadav et al. tested exactly this setup — MSE-only masked spectrogram modeling with a Mamba (SSAM) backbone. They found:
- Mamba substantially outperforms Transformers: SSAM-Base 87.9 vs SSAST-Base 68.2 (aggregated across 10 audio tasks)
- MSE reconstruction alone is sufficient; they dropped NCE contrastive entirely
- They tested `f=F, t=1` (full-frequency, single-frame patches) and found it effective for keyword spotting

**Audio-MAE ablation**: Explicitly tested adding NCE contrastive to MSE reconstruction and found **no benefit** (Appendix C.3).

**Compatibility with Mamba**: SSAST-style masking (replace with mask tokens, keep sequence length) is required for Mamba, not MAE-style (drop masked tokens). Our current implementation already does this correctly.

**Complexity**: Negative — simpler than current code (delete NCE computation).

**Risk**: MSE focuses on low-level spectral reconstruction. Representations may be less semantically rich than discrete-target approaches. However, Yadav et al. showed strong downstream performance with Mamba + MSE.

## Approach 2: BEST-RQ (Random Projection Discrete Targets)

**Source**: Chiu et al., ICML 2022, [arXiv:2202.01855](https://arxiv.org/abs/2202.01855); also used as BEATs iteration-1 tokenizer

**Method**: Replace raw patch targets with discrete labels from a frozen random projection:

1. Initialize frozen random matrix `A` (128 → 16 dim) and frozen random codebook `C` (8192 entries, 16-dim each)
2. For each mel frame `x_t`: project `q = A * x_t`, L2-normalize, find nearest codebook entry `z_t = argmin_i ||L2(C_i) - L2(q)||^2`
3. Train with cross-entropy at masked positions: `L = -sum_{t in masked} log p(z_t | x_masked)`

**Why it solves adjacent-frame degeneracy**:
- The random projection maps similar 128-dim vectors to the same discrete code — this is correct behavior, not a bug
- The model predicts "what type of sound is at this position?" (discrete class) not "which specific frame is this?" (contrastive identity)
- Even if frames 247 and 248 map to the same code (code 42), predicting "code 42" at both positions is a valid, learnable task
- The cross-entropy loss only needs the model to predict the right cluster, not distinguish between identical frames

**Complexity**: ~30 lines of code. Frozen random matrix + codebook, no training, no offline clustering.

**Published results**: Outperforms wav2vec 2.0 and HuBERT on speech benchmarks. Recent optimizations: 23.8-30.6% relative WER improvement.

## Approach 3: HuBERT-Style K-Means Discrete Targets

**Source**: Hsu et al., 2021, [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)

**Method**: Offline k-means clustering on mel features, then masked cross-entropy prediction:

1. **Iteration 1**: Run k-means (K=100-512) on 128-dim mel frames from the training data
2. Assign each frame a cluster label `z_t`
3. Train with cross-entropy on masked positions: `L = -sum_{t in masked} log p(z_t | x_masked)`
4. **Iteration 2** (optional): Re-cluster using learned representations from the backbone's intermediate layer, increase K to 500-1000

**Why it solves adjacent-frame degeneracy**: Same reasoning as BEST-RQ — similar frames get the same cluster label, and the model predicts cluster identity, not frame identity. Cross-entropy is well-defined even when adjacent frames share the same target.

**Key insight from the paper**: "Cluster label consistency matters more than cluster quality." Even noisy initial clusters (from raw MFCC/mel) teach the model useful structure.

**Masking**: SpanBERT-style — 8% probability of starting a mask span, each span 10 frames long, resulting in ~49% masking. This is lower than our 78% — worth ablating.

**Complexity**: ~50 lines + offline k-means pass (scikit-learn MiniBatchKMeans, minutes to run).

**Published results**: Matches or exceeds wav2vec 2.0 on all LibriSpeech benchmarks. HuBERT-Large: 1.5/2.5 WER.

## Approach 4: wav2vec 2.0 Quantized Contrastive

**Source**: Baevski et al., NeurIPS 2020, [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)

**Method**: Online learned quantization + contrastive learning:

```
L = L_contrastive + 0.1 * L_diversity
```

- Contrastive: `L_m = -log[exp(sim(c_t, q_t) / 0.1) / sum_{q~} exp(sim(c_t, q~) / 0.1)]`
  - `c_t` = context network output at masked position
  - `q_t` = quantized target (from product codebook, G=2 groups × V=320 entries)
  - 100 negative distractors sampled from other masked positions
- Diversity: `L_d = (1/GV) * sum p_bar log p_bar` — maximizes entropy of codebook usage

**Quantization**: Gumbel-Softmax with straight-through estimator. Temperature annealed from 2 → 0.5.

**How it handles similarity**: The quantization module maps continuous features to discrete codes before computing contrastive targets. Even if raw features are similar, the learned codebook creates meaningful distinctions.

**Caveat**: wav2vec-C (Sadhu et al., 2021) found that the codebook can collapse to **temporally invariant codes** (codes map to positions, not content). Added VQ-VAE-style reconstruction to fix this.

**Complexity**: Medium-high (~200 lines for quantizer + diversity loss + temperature schedule).

**Assessment**: More complex than BEST-RQ or HuBERT for a similar outcome. The learned quantizer is powerful but adds training instability risk.

## Approach 5: data2vec EMA Teacher Regression

**Source**: Baevski et al., ICML 2022, [arXiv:2202.03555](https://arxiv.org/abs/2202.03555)

**Method**: Student predicts the teacher's contextualized representations at masked positions:

```
L = SmoothL1(student_pred, teacher_output)  # at masked positions only
```

- Teacher = EMA of student: `theta_teacher <- tau * theta_teacher + (1-tau) * theta_student`
- `tau` annealed from 0.999 → 0.9999 over 30K updates
- Targets = average of teacher's top-K layer outputs, instance-normalized

**Why it solves adjacent-frame degeneracy**: The teacher processes the **full unmasked input**, so its representations at each position encode global context. Even if raw mel values at frames 247 and 248 are identical, the teacher's representation at each position integrates the full surrounding context (via bidirectional Mamba), producing differentiated targets.

**Concern for Mamba**: Transformer teachers have global attention — every position "sees" every other position. Mamba processes sequentially (forward + backward), so the contextualization is more local/sequential. The teacher outputs at adjacent positions might still be more similar than with a Transformer teacher. However, bidirectional processing should provide sufficient differentiation.

**Complexity**: ~300 lines. Second model copy (2x VRAM), EMA update logic, multi-layer target averaging.

**Published results**: 5.5 WER on LibriSpeech test-other (Base model). 20% improvement over wav2vec 2.0 in low-resource settings.

## Approach 6: Bird-MAE (Domain Reference)

**Source**: Bird-MAE, 2025, [arXiv:2504.12880](https://arxiv.org/abs/2504.12880)

**Architecture**: ViT-Large (~300M params), 16x16 patches, MSE reconstruction, 75% masking.

**Key details**:
- Pretrained on 1.6M curated bird vocalizations from Xeno-Canto (BirdSet pretraining corpus)
- Mixup augmentation (alpha=0.3) during pretraining
- 150 epochs, batch size 1024
- ViT decoder (not Swin shifted attention)

**Results on BirdSet HSN**: **55.28 cmAP** — current state-of-the-art for bird audio classification.

**Relevance**: Shows that (a) MSE-only pretraining works excellently for birds, (b) domain-specific pretraining data (bird audio) matters enormously, (c) model scale matters (ViT-Large vs our Tiny). However, Bird-MAE uses 16x16 patches and a ViT — different architecture from our Mamba + 128x1 setup.

## Approach 7: Temporal Pretext Tasks

**Source**: JiTTER (2025, [arXiv:2502.20857](https://arxiv.org/html/2502.20857v1))

**Methods**:
- **Temporal shuffling/jigsaw**: Shuffle blocks of frames, reconstruct original order. JiTTER outperforms masked prediction on DESED sound event detection by 5.89% PSDS.
- **Arrow of time**: Binary forward vs. reversed. Weak signal.
- **Temporal distance prediction**: Predict temporal gap between sampled frames.

**Problem for 128x1**: If adjacent frames are nearly identical, shuffling them is undetectable — the model can reconstruct the original by ignoring the shuffle. Temporal pretext tasks likely fail for the same reason contrastive learning fails.

**Assessment**: Best as auxiliary losses, not primary objectives. JiTTER has not been validated with Mamba.

## Recommendations for Our Setup

### Configuration
- SSAMBA Tiny: 7M params, 192-dim, 24 layers
- Mamba (bidirectional SSM) backbone
- 128x1 patches → 1000 tokens per 10s clip
- Bird audio domain (BirdCLEF 2025 soundscapes)

### Tier 1: Try First

| Approach | Complexity | Change |
|----------|-----------|--------|
| **MPG-only** | Delete code | Drop NCE loss, keep MSE. Validated by Yadav 2025 with Mamba. |
| **BEST-RQ** | ~30 lines | Frozen random projection + codebook → cross-entropy. Simplest discrete target. |
| **HuBERT k-means** | ~50 lines + offline step | k-means on mel frames → cross-entropy. Proven, simple. |

### Tier 2: If Tier 1 Plateaus

| Approach | Complexity | Change |
|----------|-----------|--------|
| **data2vec EMA** | ~300 lines, 2x VRAM | Contextualized teacher targets. Richest signal. |
| **wav2vec 2.0 quantizer** | ~200 lines | Learned Gumbel-Softmax codebook + diversity loss. |

### Tier 3: Different Paradigm

| Approach | Complexity | Change |
|----------|-----------|--------|
| **Switch to 16x16 patches** | Architecture change | Makes original MPC work, but loses temporal resolution. |
| **Bird-MAE style** | Architecture change | ViT + MAE + decoder. Different model entirely. |

### Suggested Experiment Plan

1. **Baseline**: MPG-only (just remove MPC from the loss). Minimal change, establishes reconstruction-only baseline.
2. **BEST-RQ**: Add frozen random projection + codebook. Replace MPC with cross-entropy on discrete labels. Compare against MPG-only.
3. **HuBERT k-means**: Offline cluster mel frames, cross-entropy on cluster IDs. Compare against BEST-RQ (data-informed clusters vs random projection).
4. **Evaluate**: Fine-tune each pretrained checkpoint on BirdCLEF labeled data, compare cmAP. The pretraining approach that produces the best downstream classifier wins.

## Connections

- **Related concepts**: frame-level tokenization (exp 005), SSAMBA architecture (exp 004)
- **Informs experiments**: experiment 009 pretraining stage (MSPM loss redesign)
- **BirdSet benchmarks**: Bird-MAE 55 cmAP sets the domain SOTA reference point

## Key References

| Paper | Year | Key Contribution | Link |
|-------|------|-----------------|------|
| wav2vec 2.0 | 2020 | Quantized contrastive for frame-level SSL | [arXiv:2006.11477](https://arxiv.org/abs/2006.11477) |
| HuBERT | 2021 | Offline k-means discrete targets | [arXiv:2106.07447](https://arxiv.org/abs/2106.07447) |
| BEST-RQ | 2022 | Random projection discrete targets | [arXiv:2202.01855](https://arxiv.org/abs/2202.01855) |
| data2vec | 2022 | EMA teacher contextualized regression | [arXiv:2202.03555](https://arxiv.org/abs/2202.03555) |
| Audio-MAE | 2022 | MSE-only reconstruction, NCE adds nothing | [arXiv:2207.06405](https://arxiv.org/abs/2207.06405) |
| BEATs | 2023 | Iterative acoustic tokenizer | [arXiv:2212.09058](https://arxiv.org/abs/2212.09058) |
| SSAMBA | 2024 | Bidirectional Mamba for audio SSL | [arXiv:2405.11831](https://arxiv.org/abs/2405.11831) |
| Bird-MAE | 2025 | Domain-specific MAE for bird audio | [arXiv:2504.12880](https://arxiv.org/abs/2504.12880) |
| Yadav et al. | 2025 | MSE-only works with Mamba, outperforms Transformer | [arXiv:2509.18691](https://arxiv.org/abs/2509.18691) |
| JiTTER | 2025 | Temporal shuffling pretext task | [arXiv:2502.20857](https://arxiv.org/html/2502.20857v1) |
