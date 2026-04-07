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
| **v10a** | PCEN → min-max | yes (1024 params) | 0.871 | **0.620** | **0.532** | 18 | pending |
| v10b | PCEN → min-max | no (frozen) | 0.870 | 0.610 | 0.515 | 18 | pending |
| v10c | PCEN → AmpToDB | no (frozen) | 0.870 | 0.602 | 0.514 | 17 | pending |

### Key Observations

1. **val/auroc is ~0.003–0.004 below v5j** across all PCEN variants (0.870–0.871
   vs 0.874). Consistent with the pattern that offline val/auroc is anti-correlated
   with LB — lower offline may mean better soundscape generalization.

2. **v10a (learnable) is best across all offline metrics** — highest val/auroc
   (0.871), val/cmap (0.620), and val/f1_macro (0.532). Learnable PCEN parameters
   adapt better than fixed defaults.

3. **v10c (PCEN+AmplitudeToDB) is worst** — suggests PCEN's adaptive gain control
   followed by another dB compression doesn't add value over PCEN alone.

4. **All variants converged normally** — best epochs at 17–18 (of 20), no training
   instability from PCEN.

5. **Offline metrics are unreliable predictors of LB** — per pattern analysis,
   offline improvements from training recipe changes consistently failed to translate
   to LB gains. The ~0.003 offline drop could mean anything on LB. Must submit to know.

## LB Submissions

| Model | LB | Delta vs v5j | Date | Notes |
|-------|-----|-------------|------|-------|
| v10a | pending | — | — | Best offline; submit first |
| v10b | pending | — | — | — |
| v10c | pending | — | — | — |

## Next Steps

- Export v10a to ONNX and submit to LB (highest priority — best offline metrics)
- If v10a ≥ 0.906: submit v10b and v10c to compare
- If v10a ≥ 0.910: 4-fold ensemble + priors
