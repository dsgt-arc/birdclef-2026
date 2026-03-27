---
experiment: "021-pseudo-labels"
status: proposed
date: 2026-03-27
author: Murilo Gustineli
---

# Proposal: Multi-Round Pseudo-Labeling

**Status**: proposed
**Created**: 2026-03-27
**Author**: Murilo Gustineli

## Hypothesis

Multi-round pseudo-labeling with a strong SED+Perch teacher ensemble, power transform sharpening, and low pseudo-label weight will yield +0.02-0.04 LB improvement, pushing us toward 0.93+ LB.

## Background

The biggest remaining gap to 0.93+ is leveraging the 10,592 unlabeled soundscapes. Every winning solution uses pseudo-labeling:
- 2025 1st place: 4-round PowerTransform pseudo-labels -> +0.06 private LB (0.872 -> 0.930)
- 2025 2nd place: 3-round OOF pseudo-labels -> +0.024 private LB
- Tom (0.929): r1-r5 pseudo-labels with Perch soft KD -> +0.034 val cumulative
- Our exp 015 (failed): single round, w=0.5, weak teachers -> regression

### Why Previous Attempt (Exp 015) Failed

1. Weight too aggressive (0.5 on pseudo-labels vs 0.1-0.2 recommended)
2. Single round (no iterative refinement with power transform)
3. Weak teachers (clip-level ProtoCLR+ConvNeXt, not SED or Perch)
4. No power transform to suppress confident noise between rounds

## Method

### Approach

3-4 rounds of iterative pseudo-labeling using the best ensemble (SED + Perch) as teacher, with power transform sharpening between rounds and low pseudo-label weight.

### Procedure

1. Generate soft pseudo-labels on all 10,592 unlabeled soundscapes using ensemble (SED + Perch)
2. Apply power transform: `p_new = p ** gamma` with gamma=2.0 to sharpen predictions
3. Train SED student model on: labeled data (weight=1.0) + pseudo-labeled data (weight=0.1-0.2)
4. Use trained student as new teacher -> repeat for 3-4 rounds
5. Progressive sharpening: increase gamma each round (1.5 -> 2.0 -> 2.5)

### Key Differences from Exp 015

| Aspect | Exp 015 (failed) | Exp 021 (proposed) |
|--------|-------------------|---------------------|
| Teacher strength | Clip-level 0.767 | SED (0.859) + Perch (0.910) ensemble |
| Pseudo-label weight | 0.5 | 0.1-0.2 |
| Rounds | 1 | 3-4 |
| Power transform | None | gamma 1.5 -> 2.0 -> 2.5 |
| Architecture | Clip-level student | SED student (framewise) |

### Potential Additions

- SoftAUCLoss in later rounds (directly optimizes competition metric)
- WeightedRandomSampler proportional to prediction density
- Perch soft KD alongside pseudo-labels (Tom's approach: +0.030 val)

### Setup

| Component | Details |
|-----------|---------|
| Data | 10,592 unlabeled soundscapes + labeled training set |
| Compute | 1x RTX 6000, ~4h per round (12-16h total for 3-4 rounds) |
| Dependencies | Exp 018 (ensemble teacher), exp 017 SED training infrastructure |
| Code | `user/murilo/experiments/017-mel-overhaul/` (SED training), new pseudo-label generation script |

### Variables

- **Independent**: Number of rounds (3-4), pseudo-label weight (0.1-0.2), power transform gamma (1.5-2.5)
- **Dependent**: Kaggle LB score (AUROC), validation loss
- **Controlled**: Architecture (EfficientNet-B0), mel config, labeled data splits

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Kaggle LB (AUROC) | Primary competition metric |
| Per-round LB delta | Track improvement across pseudo-label rounds |
| Validation loss | Training stability per round |

### Baseline

Best from exps 018-020 (expected ~0.87-0.88 LB). Target: 0.93+ LB.

### Success Criteria

- **Confirm if**: Final round achieves >= 0.91 LB (clear improvement trajectory toward 0.93+)
- **Reject if**: Round 1 student regresses below teacher LB (same failure mode as exp 015)

## Limitations

- Gate condition: Exp 018 ensemble must be working (need strong teacher)
- Each round requires ~4h training + ~30min pseudo-label generation
- Power transform gamma selection is heuristic — may need tuning
- Risk of confirmation bias: pseudo-labels can reinforce teacher errors across rounds
- Unlabeled soundscape distribution may differ from labeled data

## References

- `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` — PowerTransform
- `docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md` — Multi-round pseudo
- `docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md` — Technique rankings
- `docs/references/deep-research/2026-03-24-gemini-strategic-framework.md` — Phase 3
- `user/murilo/notes/gap-analysis-2026-03-26.md` — Gap 1 analysis
