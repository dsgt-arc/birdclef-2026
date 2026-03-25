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
- [2025 1st place inference](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md): n_fft=4096, 224 mels, hop=1252
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md)
- [Exp 016 summary](../016-sed-architecture/summary.md)

## Leads to

- If positive: Exp 018 (recipe alignment on top of improved mel)
- If neutral/negative: Pivot to Perch ensemble or metadata-focused approach
