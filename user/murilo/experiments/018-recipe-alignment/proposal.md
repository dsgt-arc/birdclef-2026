---
experiment: "018-recipe-alignment"
status: proposed
date: 2026-03-25
author: Murilo Gustineli
---

# Experiment 018: Training Recipe Alignment

## Hypothesis

Cumulative training recipe changes — ASL loss, AttBlockV2 head, StochasticDepth
regularization, and backbone scaling — will close the remaining gap to 0.90 LB
when built on top of the best mel config from exp 017.

## Gate Condition

**Only proceeds if exp 017 (mel overhaul) shows >=+0.01 LB improvement.** If mel
resolution doesn't help, the gap is elsewhere and recipe alignment won't fix it.

## Motivation

The 2025 1st place and competitor solutions differ from our v1 recipe in multiple
ways beyond mel parameters:

| Change | Evidence | Est. Impact |
|--------|----------|-------------|
| ASL loss (gamma_neg=4, gamma_pos=0, clip=0.05) | Competitor ablation: +0.023 vs BCE | +0.01-0.02 |
| AttBlockV2 head (tanh+sigmoid dual activation) | Competition standard (5+ top solutions) | +0.005-0.01 |
| StochasticDepth (drop_path=0.15) | 2025 1st place uses this | +0.005 |
| Larger backbone (B3/B4/RegNetY) | 2025 1st place: B0/B3/B4/RegNetY mix | +0.01-0.02 |
| AmplitudeToDB + min-max normalization | 2025 1st place | Unknown |

## Approach

Ablate one change at a time from exp 017 best config:

1. **v5a**: ASL loss (gamma_neg=4, gamma_pos=0, clip=0.05)
2. **v5b**: AttBlockV2 head + framewise-max inference (not blended 0.75/0.25)
3. **v5c**: StochasticDepth (drop_path=0.15)
4. **v5d**: EfficientNet-B3 backbone (12M params vs B0's 3.6M)
5. **v5e**: AmplitudeToDB + standardize + min-max normalization

If time is limited, combine most-promising changes based on early results.

## Baseline

Best from exp 017.

## References

- [Strong baseline concept](../../docs/concepts/strong-baseline-mel-overhaul.md)
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md)
- ASL paper: Ridnik et al., "Asymmetric Loss For Multi-Label Classification" (2021)

## Leads to

- Strong labeled-data baseline (target: 0.85+)
- Phase 2: Pseudo-labeling with PowerTransform on the strong baseline
- Multi-model ensemble with OpenVINO optimization
