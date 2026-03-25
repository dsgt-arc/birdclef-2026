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

## Leads to

- If positive: Exp 018 (recipe alignment on top of improved mel)
- If neutral/negative: Pivot to Perch ensemble or metadata-focused approach
