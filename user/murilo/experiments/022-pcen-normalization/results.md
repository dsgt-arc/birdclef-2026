---
experiment: "022-pcen-normalization"
status: in-progress
date: 2026-04-07
author: Murilo Gustineli
---

# Experiment 022: PCEN Normalization — Results

## Offline Metrics

All variants identical to v5j except mel normalization. HGNetV2-B0, LSEHead,
BCE, OneCycleLR, 20 epochs, 90/10 split, spectrogram MixUp.

| Model | Normalization | PCEN Trainable | val/auroc | val/cmap | val/f1_macro | Best Epoch | LB |
|-------|---------------|----------------|-----------|----------|-------------|------------|-----|
| v5j (baseline) | AmplitudeToDB | — | **0.874** | — | — | — | **0.906** |
| v10a | PCEN → min-max | yes (1024 params) | 0.871 | **0.620** | **0.532** | 18 | **0.880** (−0.026) |
| v10b | PCEN → min-max | no (frozen) | 0.870 | 0.610 | 0.515 | 18 | — |
| v10c | PCEN → AmpToDB | no (frozen) | 0.870 | 0.602 | 0.514 | 17 | — |

### Key Observations

1. **v10a (learnable PCEN) scored 0.880 LB — a −0.026 regression from v5j (0.906).**
   PCEN normalization hurts soundscape generalization despite theoretical motivation.

2. **val/auroc was ~0.003–0.004 below v5j** across all PCEN variants (0.870–0.871
   vs 0.874). The offline drop correctly predicted the LB regression this time.

3. **v10a (learnable) was best across all offline metrics** — highest val/auroc
   (0.871), val/cmap (0.620), and val/f1_macro (0.532). v10b/v10c not submitted
   since they had worse offline metrics and v10a already regressed significantly.

4. **v10c (PCEN+AmplitudeToDB) was worst offline** — PCEN adaptive gain control
   followed by dB compression doesn't add value over PCEN alone.

5. **v5j-repro (AmplitudeToDB baseline retrained through exp 022 pipeline) is
   training** to verify the regression is from PCEN itself and not a pipeline bug.

## LB Submissions

| Model | LB | Delta vs v5j | Date | Notes |
|-------|-----|-------------|------|-------|
| v10a | **0.880** | **−0.026** | 2026-04-07 | Learnable PCEN regressed |
| v10b | — | — | — | Not submitted (worse offline than v10a) |
| v10c | — | — | — | Not submitted (worse offline than v10a) |

## Next Steps

- v5j-repro training to verify pipeline correctness (in progress)
- If v5j-repro ≈ 0.906: PCEN is the problem, experiment rejected
- If v5j-repro << 0.906: pipeline bug, investigate
