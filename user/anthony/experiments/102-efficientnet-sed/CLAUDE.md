# Experiment 102: EfficientNet with Unfrozen Backbone

## What this experiment does

Unfreeze the EfficientNet-B1 backbone (frozen in exp 012, 0.649 Kaggle LB) and
train on BirdCLEF 2026 focal data. This is the single highest-impact Track 1
change not yet tried. Progressive runs: frozen baseline → unfrozen + attention
→ mixup with soundscapes.

## Current status

- Proposal written, activity scores computed
- Model implementation not yet started

## Run matrix

| Run | Backbone | Head | Data | Purpose |
|-----|----------|------|------|---------|
| A | Frozen | Linear + GAP | Focal only | Reproduce exp 012 baseline on 2026 |
| B | Unfrozen | Attention + GAP | Focal only | Main hypothesis |
| C | Unfrozen | Attention + GAP | Focal + soundscape mixup | Domain adaptation |

## Key data

- Focal recordings: 35,549 files, 206 species, 5s chunks @ 32kHz
- Activity scores: `artifacts/focal_activity.parquet` (244,875 chunks)
  - Per-clip normalized spectral contrast (`activity_score`)
  - Global normalized (`activity_score_global`)
  - See `docs/activity-scores.md` for details and known issues
- Labeled soundscapes (for Run C): 739 windows from exp 100/101
- Evaluation: 739 labeled soundscape windows (exp 101 pipeline)

## Dependencies

- Forks exp 012 model code (`birdclef_012.model.EfficientNetClassifier`)
- Uses exp 009 data module (`birdclef_009.data.finetune.BirdCLEFDataModule`)
- Uses exp 010-2026 acoustic indices (pre-computed, in scratch)
- Evaluation via exp 101 labeled soundscape pipeline

## Baselines

| Method | val/cmap | Kaggle LB |
|--------|----------|-----------|
| Frozen EfficientNet + linear (exp 012) | 0.336 | 0.649 |
| ProtoCLR LoRA + soundscapes (murilo/012) | 0.481 | 0.767 |

## Target

Kaggle LB ≥ 0.75 (match Murilo's best single model).

## Directory structure

```
├── proposal.md
├── tasks.md
├── CLAUDE.md
├── docs/
│   └── activity-scores.md
├── scripts/
│   ├── build_activity_scores.py
│   └── spot_check_activity.py
├── sbatch/
│   ├── build-activity.sbatch
│   └── spot-check.sbatch
├── src/birdclef_102/          (not yet created)
├── configs/                   (not yet created)
└── artifacts/
    └── focal_activity.parquet
```
