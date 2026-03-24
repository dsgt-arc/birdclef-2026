# Experiment Ladder

Progressive runs building toward a strong single model. Each step adds one
change. Each has a decision gate — if it doesn't help, skip it.

## Phase 1: Architecture

| Run | Change | Config |
|-----|--------|--------|
| A | Frozen backbone + avg pool | `run-a-frozen.yaml` |
| B | LoRA r=8 + avg pool | `run-b-lora.yaml` |
| C | LoRA r=8 + attention + GEMM(512) | `run-c-lora-attention.yaml` |

**Gate**: Pick best by val/cmap.

## Phase 2: Class Weighting

Apply per-species weights to the loss so rare species contribute more.
Use on best architecture from Phase 1.

| Formula | Description |
|---------|-------------|
| Inverse frequency | `w = 1/count`, normalized to mean 1.0 |
| Inverse square root | `w = 1/sqrt(count)`, normalized to mean 1.0 |

The 2025 1st place used `count^(-0.5)` (inverse square root). Full inverse
can over-amplify singleton species (1 recording → huge weight). Test both.

Class weights already computed in exp 100 `class_weights.json` (inverse
frequency). Need to add inverse square root variant.

**Gate**: Check rare species val/auroc. Keep if rare species improve >0.02
without hurting common species.

## Phase 3: Sample Weighting / Filtering

Use activity scores to handle focal chunks that are mostly silence.

| Variant | Description |
|---------|-------------|
| Soft weight | `activity_score` as sample weight in BCE loss |
| Hard filter | Drop chunks with `activity_score < threshold` |

The research suggests hard filtering (dropping bad chunks) may work
better than soft weighting. Test both.

**Gate**: Compare val/cmap. If neither helps, focal chunks are fine as-is.

## Phase 4: Focal Mixup

Mix focal recordings to create multi-species training examples.

| Variant | Description |
|---------|-------------|
| 2-clip, alpha=0.3 | Standard mixup, conservative |
| 2-clip, alpha=1.0 | Standard mixup, uniform weights |
| 8-clip, Dirichlet | Match soundscape density (~8 species/window) |

Start with 2-clip alpha=0.3 (close to what the 2025 1st place used at
alpha=0.15). Only try higher if 2-clip helps.

**Gate**: If 2-clip doesn't help, skip higher counts.

## Phase 5: No-Call Class

Add soundscape silence as negative training examples.

Mine low-activity windows from unlabeled soundscapes (spectral contrast
below threshold). Train with all-zero labels. This teaches the model
that silence exists — currently every training sample has at least one
species.

**Gate**: Does precision improve on soundscape evaluation (exp 101)?

## Phase 6: Label Smoothing (optional)

Smooth hard labels from 1.0 to 0.9. Prevents overconfidence, especially
useful with mixup. Simple to add, low risk.

## Out of Scope (separate experiments)

| What | Why separate | Experiment |
|------|-------------|------------|
| Pseudo-labeling | Changes training data, needs inference on 10K soundscapes, iterative | Exp 103 |
| Soundscape-realistic mixup | Complex, depends on pseudo-label quality. See `docs/soundscape-mixup.md` | Exp 104 |
| Model ensembling | Multiple architectures, orthogonal to single-model optimization | Later |
| External data (Xeno-Canto) | Data pipeline work, legal/license considerations | Later |
| SoftAUCLoss | Alternative loss function, could test as Phase 1 variant | Optional |
| TTA / inference tricks | Applied at submission time, not training | Submission pipeline |

## Research Notes (from BirdCLEF 2025 solutions)

**1st place (Nikita Babych)**:
- EfficientNet ensemble + CE/AdamW baseline
- MixUp (alpha=0.15) + StochasticDepth → 0.872 → 0.898
- Iterative pseudo-labeling (3-4 rounds) → 0.898 → 0.930
- Used `count^(-0.5)` class weighting + SoftAUCLoss

**Key insight**: Pseudo-labeling was worth +3-6 AUC points — more than any
architectural change. But it requires a decent single model to start.
Exp 102 builds that single model; exp 103 does the pseudo-labeling.

**What winners did NOT use**:
- Cross-taxa explicit mixup (random mixup was sufficient)
- Activity-weighted training (used hard filtering instead)
- Attention pooling (some did, some used simple SED)
