# Training Insights

Cross-experiment learnings about what works and doesn't work for model training.
Observations from our own experiments plus competitor analysis.

Last updated: 2026-03-26

## Loss Functions

| Loss | Our Result | Competitor Evidence | Verdict |
|------|-----------|--------------------| --------|
| **soft CE** | v4b 0.859 LB (best) | — | Our default, works well |
| **BCE** | v4g 0.868 val (LB pending) | Tom's default, Tawara 0.859 LB | Natural for multi-label SED |
| **ASL** | v4d 0.808, v4h 0.858 val | Tom +0.023 over BCE | Converges fast, plateaus low in our setup |
| **CE** (standard) | v2 regressed (exp 016) | 2024 1st place: CE >> BCE | Need sigmoid at inference, not softmax |
| **Focal** | — | 2023 1st place used it | Untested |
| **SoftAUCLoss** | — | 2025 1st place, later rounds | Directly optimizes competition metric |

**Key insight**: Our soft CE (normalize multi-hot → F.cross_entropy) is unique to us.
It works well but is different from both standard CE and BCE. When testing competitor
techniques (AttBlockV2, ASL), we must use their native loss pairing — not ours.

**ASL pattern**: In both v4d and v4h, ASL peaked early (epoch 12-16) then plateaued.
Tom's +0.023 was over BCE; our soft CE already handles easy negatives via normalization.
ASL may be redundant with soft CE.

References:
- [Tom's SED comparison](../../docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md)
- [2024 1st place](../../docs/references/kaggle/birdclef-2024/birdclef2024-1st-place-solution.md) — CE > BCE
- [Tawara HGNetV2-B0](../../docs/references/kaggle/birdclef-2026/discussions/hgnetv2-b0-baseline-tawara.md) — BCE + LogSumExp

## SED Head Design

| Head | Our Result | Competitor Evidence |
|------|-----------|---------------------|
| **AttHead** (softmax att) | v4b 0.859 LB | Our custom design |
| **AttBlockV2** (tanh+softmax, sigmoid cls) | v4c 0.815 (soft CE), v4g 0.868 val (BCE) | Tom + PANNs default |

**Critical finding**: AttBlockV2 **requires BCE** (not soft CE). The sigmoid→logit
roundtrip in our AttBlockV2 implementation breaks with soft CE's softmax-based loss.
With BCE (v4g), AttBlockV2 achieves 0.868 val — matching our best offline score.

**Tawara's approach**: Uses LogSumExp (temperature=0.1) for frame aggregation instead
of hard max. This is a smooth differentiable approximation that may be more stable.
We use `framewise_logits.max(dim=-1)` — could try LSE.

## Mel Spectrogram Configuration

| Parameter | Our Best (v4b) | Tom | Tawara | 2025 1st |
|-----------|---------------|-----|--------|----------|
| n_fft | **4096** | 2048 | 2048 | **4096** |
| n_mels | **224** | 224 | 256 | **224** |
| hop_length | 320 | 512 | 313 | 1252 |
| f_min | 0 | 0 | 20 | 0 |
| Normalization | **AmplitudeToDB + min-max** | **AmplitudeToDB + min-max** | **AmplitudeToDB + min-max** | **AmplitudeToDB + min-max** |
| Input shape | (1, 224, 1000) | (1, 224, ~) | (1, 256, 256) resize | (1, 224, 512) |
| Segment | 10s | 5s | 5s | 20s |

**AmplitudeToDB is universal among top performers.** Every strong SED baseline uses it.
Our biggest single-technique gain was switching to it (+0.034 LB).

**n_fft=4096 vs 2048**: Both work well. Tom uses 2048 with n_mels=224 and gets strong
results. The 2025 1st place uses 4096. Our v4a showed +0.030 LB from the n_fft upgrade.

**Resize vs native resolution**: Tawara resizes to 256x256, losing temporal info.
We keep native 224x1000 (10s at hop=320). Our longer context may help.

## Normalization Impact

| Normalization | LB | vs v4a (audioset) |
|---------------|-----|-------------------|
| **AmplitudeToDB + standardize + min-max** (v4b) | **0.859** | **+0.034** |
| AudioSet (log + fixed mean/std) (v4a) | 0.825 | baseline |

This was the largest single-technique gain in the entire experiment history.
AmplitudeToDB preserves more dynamic range than log+fixed stats.

## Regularization

| Technique | Result | Context |
|-----------|--------|---------|
| **StochasticDepth (drop_path=0.15)** | v4e: +0.015 LB (audioset norm), v4f: −0.027 LB (AmplitudeToDB) | Helps with audioset, hurts with a2db |
| **Dropout (0.5)** | Used in all variants | Standard, in attention head |
| **Spectrogram MixUp** | Used in all variants (prob=0.5, alpha=0.5) | Standard |
| **Raw waveform MixUp** | v3b: +0.007 LB (exp 016) | Marginal improvement |

**StochasticDepth + AmplitudeToDB conflict**: v4f regressed −0.027 LB despite only
−0.003 val gap. The regularization that helps with audioset norm becomes counterproductive
with AmplitudeToDB — possibly because a2db's per-sample normalization acts as implicit
regularization, and adding drop_path causes underfitting.

## Augmentation

| Technique | Our Usage | Competitor Evidence |
|-----------|-----------|---------------------|
| **Spectrogram MixUp** | prob=0.5, alpha=0.5 | Tom uses, Tawara alpha=1.0 theta=0.8 |
| **Raw waveform MixUp** | Tested in v3b (+0.007) | 2025 2nd place: +0.036 private LB |
| **SpecAugment** | Not used | 2023 1st place: freq+time masking |
| **Background noise** | Not used | Tom: −0.041 (hurts with pseudo-labels) |
| **CutMix** | Not used | Tom: −0.047 (destroys spatial coherence) |
| **Rating filter ≥3** | Not used | Tom: −0.128 (severe data reduction) |

**Tawara's MixUp**: Alpha=1.0 (uniform distribution) with theta=0.8 label binarization.
More aggressive than our alpha=0.5. Also disables during first 5 warmup epochs. Worth
testing higher alpha.

## Training Configuration

| Parameter | Our v4b | Tawara | Tom |
|-----------|---------|--------|-----|
| Epochs | 50 (patience=10) | 20 (no patience) | ~20 |
| LR | 1e-4 (head 1e-3) | 5e-4 | — |
| Optimizer | AdamW (wd=0.01) | AdamW (wd=1e-4) | — |
| Scheduler | CosineAnnealing | OneCycleLR | — |
| Batch | 64 (accum=4, eff=256) | 64 | — |
| AMP | fp16 | fp16 | — |
| CV | 90/10 single split | 4-fold MSGKF | — |

**K-fold observation**: Tawara achieves 0.859 LB with 4-fold ensemble on HGNetV2-B0.
We match that with a single v4b model. 4-fold of our v4b could potentially reach
~0.865-0.870 LB. Free improvement but 4x training cost.

## Pseudo-Labeling Lessons

Our exp 015 (student distillation) **failed** — both ProtoCLR (−0.005) and EAT (−0.024)
regressed despite flat offline metrics.

**Root causes**:
1. Pseudo-label weight too high (w=0.5) — likely need w=0.1-0.3
2. EAT excluded from teacher ensemble (poor insect AUROC 0.673) → pseudo-labels
   conflicted with EAT's representations
3. Temperature T=1.0 may not be right — 2025 1st place uses power transform (p^γ)
   with γ=2.0 to sharpen pseudo-labels across rounds

**What the winners do**:
- 2025 1st place: 4-round self-training with power transform (+0.032 LB)
- Tom: multi-round pseudo-labels + Perch soft KD (+0.025 val)
- Key: iterative refinement, not single-shot

References:
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md)

## Inference Optimization

Our ONNX inference optimization (v4a onward):
1. **Full-signal mel**: 1 STFT per 60s file instead of 36 → ~10-20x mel speedup
2. **Batched ONNX**: All slices in one session.run() call
3. **Session options**: ORT_ENABLE_ALL + multi-threaded intra-op

Result: >1h → <32min for n_fft=4096 models. Faster than v1 (n_fft=1024) despite
1.75x more mel bins.

**Competitors use OpenVINO** (every winner since 2023). Could be faster than ONNX
Runtime for CPU inference. Worth exploring if we need more inference budget.

Reference: [ONNX inference optimization](../experiments/017-mel-overhaul/onnx-inference-optimization.md)
