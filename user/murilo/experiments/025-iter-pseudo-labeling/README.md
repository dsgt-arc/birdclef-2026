# Experiment 025: Iterative Pseudo-Labeling

Train SED models on pseudo-labeled soundscape data to close the LB gap caused by
159 species having zero soundscape representation. Uses a tiered teacher strategy:
SED max-folds for 184 species with reliable predictions, and Perch with per-species
z-normalization for 44 species where SED is blind.

See [proposal.md](proposal.md) for full methodology, step-by-step walkthrough, and
parameter rationale (12 sections including key concepts explained from first principles).
See [tasks.md](tasks.md) for the task checklist per phase.
See [results.md](results.md) for results per iteration round.

**Branch**: TBD
**Status**: PROPOSED
**Current best**: 0.929 LB (Perch+SED ensemble, exp 023)
**Target**: 0.935+ LB

## Why This Experiment

Exp 024 established that the LB gap is a **data coverage problem**:
- 0.9958 AUC on 75 species with soundscape data (saturated)
- 159 species have ZERO soundscape data (the gap)
- SED max-folds covers 109/159 at >0.5 with 94% precision
- Perch covers 149/159 but with 6% precision (needs calibration)
- Only 6 species are truly invisible (1-5 focal recordings each)

Pseudo-labeling gives the model soundscape-domain training data for these 159 species.
Exp 015 tried this and failed, but conditions are fundamentally different now: 5× stronger
teacher, SED architecture, domain-matched data, power-transform filtering.

## Multi-Iteration Design

This experiment is designed for **multiple rounds** of pseudo-labeling, where each round's
student becomes the next round's teacher. Every round produces a versioned config and
output set for full traceability.

| Round | Config | Teacher | Goal |
|-------|--------|---------|------|
| **v1** | `configs/v1.yaml` | SED max-folds + Perch z-normalized | Close data coverage gap |
| **v2** | `configs/v2.yaml` | Round 1 student (better calibrated) | Improve PL quality |
| **v3** | `configs/v3.yaml` | Round 2 + backbone diversity | Diverse ensemble teacher |

Each round:
1. Generate pseudo-labels (config-driven, reproducible)
2. Train v5j student on focal + labeled + pseudo-labeled data
3. Evaluate on LB
4. If gain → student becomes teacher → next round

## Species Tiers

| Tier | N | Description | Teacher | Label Source |
|------|---|-------------|---------|-------------|
| **A** | 75 | In labeled soundscapes | SED max-folds | High quality (0.9987 AUC) |
| **B** | 109 | Unseen, SED-covered | SED max-folds | Good (94% precision) |
| **C** | 44 | Unseen, SED-blind | Perch z-normalized | Uncertain (needs filtering) |
| **D** | 6 | Invisible to both | None | Can't help |

## Directory Structure

```
025-iter-pseudo-labeling/
├── README.md                           # This file
├── proposal.md                         # Full methodology (12 sections)
├── tasks.md                            # Task tracking per phase
├── results.md                          # Results per round
├── configs/
│   ├── v1.yaml                         # Round 1: SED + Perch teachers
│   ├── v2.yaml                         # Round 2: student-as-teacher (TBD)
│   └── ...
├── artifacts/
│   └── training/
│       ├── hgnet_b0_sed_v5j_pl_v1_fold0/
│       │   └── metrics.json
│       └── ...
├── outputs/
│   └── tensorboard/
└── sbatch/
```

**Shared outputs** (large, not version-controlled):

```
shared/pseudo-labels/birdclef-2026/
├── v1/                                 # Round 1 pseudo-labels
│   ├── config.yaml                     # Copy of configs/v1.yaml
│   ├── pseudo_labels.parquet           # Merged (127K × 234 soft probs)
│   ├── sed_pseudo_labels.parquet       # SED component
│   ├── perch_pseudo_labels.parquet     # Perch component
│   ├── tier_assignment.csv             # Per-species tier
│   └── stats.json                      # Coverage statistics
├── v2/                                 # Round 2 (TBD)
└── ...

shared/artifacts/025-iter-pseudo-labeling/
├── v1-fold0/                           # Model checkpoints
└── ...

shared/inference/birdclef-2026/
├── hgnet-b0-sed-v5j-pl-v1/            # Student v1 inference
├── hgnet-b0-sed-v5j-pl-v1-4fold/      # Student v1 4-fold
└── ...
```

## Config Comparison

| Config | Teacher | Temperature | Power | Perch Z | PL Weight | Parent |
|--------|---------|-------------|-------|---------|-----------|--------|
| **v1** | SED 4-fold + Perch | 0.6 | 1.5 | 2.0 | 0.5 | — |
| v2 | v1 student | TBD | TBD | TBD | TBD | v1 |

## Commands

### Generate pseudo-labels

```bash
cd user/murilo/experiments/025-iter-pseudo-labeling

# Generate v1 pseudo-labels (reads configs/v1.yaml)
python scripts/generate_pseudo_labels.py --config configs/v1.yaml

# Verify output
python scripts/generate_pseudo_labels.py --config configs/v1.yaml --stats-only
```

### Training

Training extends the exp 021 pipeline. Commands TBD once data loader is extended.

```bash
# Train fold 0 (sanity check)
bash sbatch/run.sh configs/v1.yaml --fold 0

# Train all 4 folds
bash sbatch/run.sh configs/v1.yaml --all-folds
```

### Kaggle submission

Uses exp 010 submission pipeline. See
[adding-new-models.md](../010-submission-pipeline/adding-new-models.md).

```bash
cd user/murilo/experiments/010-submission-pipeline
python scripts/upload_kaggle.py --model hgnet-b0-sed-v5j-pl-v1 --push-kernel
```

## Key References

| Document | Purpose |
|----------|---------|
| [Exp 024 Results](../024-data-centric-exploration/results.md) | All EDA findings that motivate this experiment |
| [EDA Deep Dive](../../notes/2026-04-09-inference-eda-deep-dive.md) | 14-section analysis (coverage, calibration, ecology) |
| [Data-Centric Pivot](../../notes/2026-04-09-data-centric-pivot.md) | Strategic rationale for pseudo-labeling |
| [FINDINGS.md](../../FINDINGS.md) | Cross-experiment patterns (recipe changes never help) |
| [Nikita 1st Place](../../../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md) | Reference: 3-4 PL iterations, T=0.55-0.6 |
| [Exp 015 Results](../015-student-training/tasks.md) | Previous PL failure analysis |
| [Exp 021 SED Training](../021-sed-training/) | v5j training recipe (base for student) |
| [Exp 023 Ensemble](../023-perch-sed-ensemble/) | Current best (0.929 LB) |
