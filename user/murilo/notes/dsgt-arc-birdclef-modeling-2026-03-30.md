# DSGT ARC Meeting — BirdCLEF 2026 Modeling Update

**Date**: 2026-03-30
**Presenter**: Murilo Gustineli
**Competition**: BirdCLEF 2026 — 234 species, 60s soundscapes, 90-min CPU inference budget
**Current #1**: 0.939 LB | **Our best**: 0.918 LB | **Deadline**: June 3, 2026

---

## Slide 1: Clip-Level Models (EAT / ConvNeXt / ProtoCLR)

Focal training → soundscape fine-tuning → pseudo-labels (failed)

| Model | Exp | Training Data | Params | Test AUROC | LB |
|-------|-----|--------------|--------|-----------|-----|
| EAT-base | 011 | Focal only | 86M | 0.929 | 0.701 |
| ConvNeXt-Base | 011 | Focal only | 88M | 0.938 | 0.720 |
| ProtoCLR/CvT-13 | 011 | Focal only | 19M | 0.937 | 0.721 |
| EAT-base | 012 | + Soundscape | 86M | 0.838 | 0.720 |
| ConvNeXt-Base | 012 | + Soundscape | 88M | 0.844 | 0.755 |
| **ProtoCLR/CvT-13** | **012** | **+ Soundscape** | **19M** | **0.845** | **0.767** |
| ProtoCLR + pseudo-labels | 015 | + 127K pseudo (w=0.5) | 19M | 0.846 | 0.762 (-0.005) |

**Key takeaways:**

- Soundscape fine-tuning: +0.019 to +0.046 LB across all encoders
- Pseudo-labels regressed on LB despite flat offline metrics — motivated architecture pivot
- Best clip-level: **0.767 LB** (ProtoCLR)

---

## Slide 2: SED Architecture (EfficientNet-B0)

Architecture change → mel resolution → ablations

| Model | Exp | Key Change | val/auroc | LB |
|-------|-----|-----------|-----------|-----|
| ProtoCLR (best clip-level) | 012 | — (baseline) | 0.845 | 0.767 |
| SED v1 | 016 | SED architecture (3.6M params) | 0.860 | 0.795 |
| SED v1 + post-processing | 016 | + TTA + smoothing + TopN | — | 0.827 |
| SED v4a | 017 | + Hi-res mel (4096 FFT, 224 mels) | 0.862 | 0.825 / 0.830 (+pp) |
| **SED v4b** | **017** | **+ AmplitudeToDB norm** | **0.865** | **0.859 / 0.860 (+pp)** |
| SED v4e | 017 | + StochasticDepth | 0.868 | 0.840 |
| SED v4j | 017 | + Waveform MixUp | 0.868 | 0.853 |
| SED v5b | 019 | + Stronger MixUp (α=1.0) | **0.869** | pending |

**Key takeaways:**

- SED architecture: 5x fewer params, +0.028 LB over best clip-level
- Mel resolution: +0.030 raw LB (v4a raw nearly matched v1+PP)
- AmplitudeToDB: single-variable +0.034 LB (0.825 → 0.859)
- Offline val/auroc is a poor LB predictor (v4e: 0.868 val → 0.840 LB)
- Current best: **0.860 LB**, 5 new ablations in flight targeting 0.870+

---

## Slide 3: Perch ProtoSSM (Exp 018)

Frozen foundation model → probes → OOF → temporal modeling

| Pipeline | Technique Added | LB | Delta |
|----------|----------------|-----|-------|
| Perch raw | Logit mapping only | 0.825 | — |
| + genus proxy | Genus-level proxy for missing species | 0.836 | +0.011 |
| + LogReg probes | 52 per-species probes (39-dim features) | 0.893 | +0.057 |
| + OOF + fold-safe priors | 5-fold GroupKFold, 6-category fusion | 0.911 | +0.018 |
| **+ ProtoSSM v4** | **BiSSM + cross-attn + ResidualSSM + TTA + 6-stage PP** | **0.918** | **+0.007** |
| ProtoSSM v4.2 | Taxonomy fix, ensemble weight fix | 0.899 (-0.019) | regressed |
| Target | Match top LB (Yuriy/Yaroslav) | 0.925 | — |

**Key takeaways:**

- No GPU training — probes train in ~2s on CPU, all inference on Kaggle CPU
- +0.093 LB from raw Perch to ProtoSSM through systematic feature engineering
- Current best: **0.918 LB** (v4.2 regressed to 0.899 — reverting to v4 baseline next)

---

## Next Steps

- **SED track (exp 019)**: 5 ablations training now (BCE, MixUp, weight decay, LSEHead, HGNetV2-B0), targeting 0.870+ single fold → 0.885+ with 4-fold ensemble
- **Perch track (exp 018)**: v4.2 regressed to 0.899; next step: revert to v4 (0.918) or port Yuriy's pipeline directly
- **Pseudo-labels (exp 019 Phase C)**: Use Perch ProtoSSM (~0.920 teacher) to label 127K unlabeled soundscape windows → train SED, targeting 0.900–0.915
- **Endgame**: Perch + SED ensemble in a single 90-min kernel
