---
experiment: "017-mel-overhaul"
status: proposed
date: 2026-03-25
author: Murilo Gustineli
---

# Experiment 017: Mel Spectrogram Overhaul

## Hypothesis

Changing from (n_fft=1024, n_mels=128) to (n_fft=4096, n_mels=224) while keeping
all other v1 parameters identical will improve LB by >=0.02. This is 4x frequency
resolution and 1.75x spectral detail — the largest untested gap vs the 2025 1st
place solution (0.937).

## Motivation

The v3 ablation (exp 016) proved no single recipe tweak (weights, loss, mixup domain)
helped. Two mel paradigms exist in BirdCLEF:
- SED-native: n_fft=1024, 128 mels (public notebooks 0.857-0.878)
- High-res + SED: n_fft=4096, 224 mels (2025 1st place 0.937)

Our v1 uses the SED-native paradigm. The 1st place combined high-res mel WITH SED
head — an untested combination. With 234 species including spectrally similar calls,
amphibian harmonics, and insect stridulation, the model can't distinguish signals
that a higher-resolution spectrogram would separate.

### Tom's 28-experiment ablation validates subsequent steps

[Tom's SED model comparison](../../docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md)
(28 experiments, best 0.8139 soundscape val AUC) confirms several directions for
our Steps 2-3:

| Technique | Tom's impact | Our plan |
|-----------|-------------|----------|
| ASL loss (γ⁻=4, γ⁺=0) | **+0.023 AUC** over BCE | Step 3 (v4f) |
| Soft KD from Perch | **+0.030 AUC** cumulative | Future opportunity |
| AmplitudeToDB + min-max | Used across all experiments | Step 2 (v4b) |
| tanh+softmax attention | Used across all experiments | Step 2 (v4c) |
| B0 > B3/V2S | B3 unstable, V2S underperforms | Confirms our B0 choice |
| BG noise augmentation | **−0.041** (hurts) | Avoid |
| Rating filter ≥3 | **−0.128** (hurts badly) | Don't filter |

Tom uses n_fft=2048 (not 4096) with n_mels=224 and still gets strong results,
suggesting n_mels=224 may matter more than n_fft for spectrogram resolution.

## Approach

### v4a: Mel resolution only (single variable from v1)

| Parameter | v1 | v4a |
|-----------|-----|-----|
| n_fft | 1024 | **4096** |
| n_mels | 128 | **224** |
| hop_length | 320 (10ms) | 320 (10ms) — keep our temporal resolution |
| Everything else | v1 recipe | Identical |

Input shape changes: (1, 128, 1000) -> (1, 224, 1000) for 10s windows.

### If OOM or too slow

- Reduce n_mels to 160 (intermediate between 128 and 224)
- Or increase hop to 640 (20ms) -> (1, 224, 500) for 10s

## Baseline

Best from exp 016: v1-10s+pp2 (0.826 LB baseline, pending pp2 result).

## Success Criteria

- **Confirm if**: LB improves by >=0.02 over baseline
- **Strong confirm if**: LB exceeds 0.85
- **Reject if**: LB shows no improvement or regresses

## Dependencies

- Exp 016 v1 training infrastructure (complete) — fork train.py, configs, SLURM scripts
- NNAudioMelSpectrogram needs to support n_fft=4096 and n_mels=224
- May need batch_size reduction if (224, 1000) causes OOM on RTX 6000 (24GB)

## References

- [Strong baseline concept](../../docs/concepts/strong-baseline-mel-overhaul.md)
- [Tom's SED model comparison (28 experiments)](../../docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md) — ASL +0.023, Perch KD +0.030, validates Steps 2-3
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md) — technique rankings and code snippets
- [2025 1st place inference](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md): n_fft=4096, 224 mels, hop=1252
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md)
- [Exp 016 summary](../016-sed-architecture/summary.md)

## ONNX Inference Optimization (v4a)

The n_fft=4096 mel upgrade created a major inference bottleneck on Kaggle CPU — the
original per-segment approach took >1h (vs ~38min for v1). Three optimizations brought
v4a inference to <32min (~2.5-3× faster), making it faster than v1 despite 1.75× more
mel bins:

### 1. Full-signal mel (biggest win)

Instead of computing an independent STFT for each 5s segment (with 10s context windows
and ±2.5s TTA shifts = many redundant FFTs), compute the mel spectrogram **once** for
the entire 60s file, then slice frames for each segment/TTA window. This eliminates
redundant STFT computation across overlapping segments.

```
Before: 12 segments × 3 TTA shifts × n_fft=4096 STFT each = 36 STFTs per file
After:  1 STFT for 60s padded audio + 36 numpy slices (near-free)
```

Implemented as `compute_full_signal_mel()` + `_slice_mel_window()` in submission.py.
Activated automatically for SED models with `architecture=sed` and `type=mel`.

### 2. Batched ONNX inference

Instead of running ONNX session.run() once per segment, concatenate all mel slices
into a single batch `(N, 1, n_mels, T)` and run one batched inference call per file.
This amortizes ONNX session overhead and enables internal parallelism.

### 3. ONNX session optimization

- `graph_optimization_level = ORT_ENABLE_ALL` (operator fusion, constant folding)
- `intra_op_num_threads = os.cpu_count()` (parallelize within ops)
- `inter_op_num_threads = 1` (single execution stream, avoids contention)

### Kaggle LB verification

Re-submitted v4a with optimized inference: **0.826 LB** (vs 0.825 original). The ±0.001
variance is expected — Kaggle rotates/subsamples test data between submissions to
prevent overfitting. The score is functionally identical, confirming the optimization
is lossless.

These optimizations apply to all future v4* submissions automatically.

## Known Risks & Mitigations (Step 2 Ablations)

### v4c (AttBlockV2) + soft CE loss mismatch

AttBlockV2 computes `sigmoid(frame_logits)` internally and produces probabilities
via attention-weighted sum. We convert back to logits via `torch.logit()` for
compatibility with the existing soft CE loss. But AttBlockV2 is designed for **BCE
loss** — Tom and PANNs use BCE with this head, not CE. The soft CE + softmax applied
to logit-converted probabilities is a different computation path.

**Mitigation**: If v4c underperforms, retry with BCE loss before concluding
AttBlockV2 doesn't help. The loss pairing may be the issue, not the head.

### v4b-v4e vs v4a comparison confound

v4a used 80/10/10 split + 40 epochs. v4b-v4e use 90/10 + 50 epochs (~11% more
training data). Small LB deltas (≤+0.003) may be the extra data, not the technique.

### PP2 diminishing returns on v4a

PP2 gained only +0.005 on v4a vs +0.032 on v1. The temperature (T=1.10) and TTA
params were tuned for v1's output distribution. The v4a model is more confident on
correct predictions, so the current T over-softens. **A temperature sweep on the
existing v4a model is a zero-cost opportunity** — sweep T ∈ {0.90, 0.95, 1.00, 1.05,
1.10, 1.15} before retraining anything.

### Audio-domain MixUp as backup

v3b in exp 016 tested waveform mixup at n_fft=1024/128 mels and it didn't help. But
with n_fft=4096/224 mels the mel resolution is fundamentally different — mixing in the
waveform domain may preserve more meaningful signal. The SED improvement plan reports
audio-domain MixUp at +0.036 Private LB (2025 2nd place). Infrastructure already
exists (`raw_waveform_mixup=true`). Consider if v4e (StochasticDepth) disappoints.

## Leads to

- If positive: Exp 018 (recipe alignment on top of improved mel)
- If neutral/negative: Pivot to Perch ensemble or metadata-focused approach
