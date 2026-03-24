# Experiment Ladder

Progressive runs building toward a competitive single model. Each step adds
one change. Earlier steps inform later ones — if a step doesn't help, skip
the downstream steps that depend on it.

## Phase 1: Architecture

Find the best model architecture on focal data only.

| Run | Change | Config | Depends on |
|-----|--------|--------|------------|
| A | Frozen backbone + avg pool | `run-a-frozen.yaml` | — |
| B | LoRA r=8 + avg pool | `run-b-lora.yaml` | — |
| C | LoRA r=8 + attention + GEMM(512) | `run-c-lora-attention.yaml` | — |

**Decision gate**: Pick best architecture by val/cmap. If B ≈ A, LoRA isn't
helping and we should investigate why before proceeding. If C > B, attention
pooling is worth the complexity.

### LoRA rank sweep (optional, after C)

If LoRA helps, test whether more capacity helps further.

| Run | Rank | Alpha |
|-----|------|-------|
| C-r16 | 16 | 32 |
| C-r32 | 32 | 64 |

## Phase 2: Sample Weighting

Use activity scores to weight focal training samples. Applied to the best
architecture from Phase 1.

| Run | Change | Depends on |
|-----|--------|------------|
| E | Activity score sample weighting | Best from Phase 1 |

Uses `activity_score` from `focal_activity.parquet` as per-sample weight
in BCE loss. High-activity chunks (likely containing calls) contribute more
gradient. Low-activity chunks (likely silence) contribute less.

**Decision gate**: If val/cmap improves, keep weighting for all subsequent runs.

## Phase 3: Class Weighting

Address the 499x class imbalance. Rare species (1-3 recordings) should
contribute more to the loss than common species (499 recordings).

| Run | Change | Depends on |
|-----|--------|------------|
| F | Per-species class weights in BCE loss | E |

Class weights from exp 100 `class_weights.json` (inverse frequency,
normalized to mean 1.0). Applies per-class weighting to the loss so rare
amphibians/insects get proportionally more gradient.

**Decision gate**: Check if rare species val/auroc improves without hurting
common species. If rare species improve by >0.02 auroc, keep.

## Phase 4: Focal Mixup

Mix focal recordings to create synthetic training examples. Increases
diversity without adding new data sources.

| Run | Change | Depends on |
|-----|--------|------------|
| G-2 | Mixup 2 focal clips | F |
| G-4 | Mixup 4 focal clips | F |
| G-8 | Mixup 8 focal clips | F |

Standard audio mixup: sample N clips, mix waveforms with random weights
(Dirichlet-sampled), union the species labels. This creates multi-species
training examples that better match the soundscape domain (which has 8+
species per window).

**Decision gate**: Pick best N by val/cmap. If none help, mixup may not be
useful with focal-only data. If 4 or 8 source mixup helps, the model is
learning to handle multi-species inputs.

## Phase 5: Cross-Taxa Mixup

Mix focal recordings across taxonomic groups to create realistic multi-taxa
training examples matching the Pantanal soundscape distribution.

| Run | Change | Depends on |
|-----|--------|------------|
| H | Cross-taxa mixup (bird + frog + insect) | Best from G |

Instead of random mixup, explicitly combine recordings from different taxa:
- Sample 1 bird + 1 frog + 1 insect recording
- Mix waveforms, union labels
- This directly addresses the soundscape composition: 76% amphibian + 29%
  bird + 23% insect windows

**Decision gate**: Compare against same-taxa mixup. If cross-taxa is better,
it confirms the model needs exposure to multi-taxa scenes.

## Phase 6: No-Call Class

Add a dedicated negative class from soundscape silence. Teaches the model
when nothing is vocalizing.

| Run | Change | Depends on |
|-----|--------|------------|
| I | Add no-call class from soundscape noise | Best from H |

Source: low-activity windows from unlabeled soundscapes (identified by
spectral contrast < threshold). These become training examples with all-zero
labels (no species present).

This is a separate experiment because it changes the label space — all-zero
targets are currently not seen in focal training (every clip has at least
one species).

**Decision gate**: Does adding no-call examples improve precision on
soundscape evaluation (exp 101 windows)?

## Phase 7: Soundscape Mixup

Mix focal recordings with real soundscape audio. Maximum domain diversity.

| Run | Change | Depends on |
|-----|--------|------------|
| J | Focal + soundscape mixup | Best from I |

Mix focal clips (clean, single-species, labeled) with soundscape windows
(noisy, multi-species, pseudo-labeled or unlabeled). This bridges the
focal-to-soundscape domain gap directly.

Options:
- Mix with labeled soundscapes (739 windows, ground truth labels)
- Mix with unlabeled soundscapes (as noise augmentation, no label contribution)
- Mix with pseudo-labeled soundscapes (from exp 101 ensemble, soft labels)

**Decision gate**: Compare focal val metrics AND soundscape val metrics. If
soundscape auroc improves, the domain gap is being addressed.

## Summary

```
Phase 1: Architecture     A → B → C (→ rank sweep)
Phase 2: Sample weight    E
Phase 3: Class weight     F
Phase 4: Focal mixup      G-2 → G-4 → G-8
Phase 5: Cross-taxa       H
Phase 6: No-call          I
Phase 7: Soundscape mix   J
```

Each phase has a decision gate. The ladder can be short-circuited — if Phase 1
shows frozen backbone is sufficient, skip LoRA-dependent phases. If mixup
doesn't help in Phase 4, skip Phases 5 and 7.

## What's NOT in this ladder

- **Iterative pseudo-labeling**: Requires running inference on all 10K
  soundscapes and retraining. Separate experiment once single-model is strong.
- **Ensemble**: Combining multiple architectures (ProtoCLR + EAT + ConvNeXt +
  EfficientNet). Separate from single-model optimization.
- **SED with frame labels**: Requires temporal annotations we don't have.
  Activity scores are the closest proxy.
- **Different backbones**: Only testing EfficientNet-B1. Could test B3, B5,
  or entirely different architectures.
- **Test-time augmentation**: Inference-time flipping, multi-crop, etc.
  Applied at submission time, not training.
