# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Architecture

This experiment forks exp 012's `EfficientNetClassifier`. The key code relationships:

- **Model**: `src/birdclef_102/model.py` — forked from `012-efficientnet-linear-probe/src/birdclef_012/model.py`
- **Data module**: reuses `birdclef_009.data.finetune.BirdCLEFDataModule` from exp 009
- **Training loop**: reuses `birdclef_009.training.finetune.FinetuneLightningModule` from exp 009
- **Activity scores**: pre-computed from exp 010-2026 acoustic indices
- **Evaluation**: exp 101 labeled soundscape pipeline (739 windows, 42 evaluable species)

The exp 012 model does: `waveform → nnAudio mel → frozen EfficientNet-B1 → GAP → Linear`.
This experiment modifies it to: `waveform → nnAudio mel → UNFROZEN EfficientNet-B1 → attention head`.

## Running

Analysis scripts run via sbatch (CPU-only, use exp 010 venv):
```bash
sbatch sbatch/build-activity.sbatch    # Build activity scores parquet
sbatch sbatch/spot-check.sbatch        # Spot check activity scores
```

Training (GPU, needs own venv — not yet implemented):
```bash
# Will be: sbatch sbatch/train-{a,b,c}.sbatch
```

Venv for analysis: `~/scratch/birdclef-2026/.venv-010`
Venv for training: needs GPU venv with torch + transformers (create per CLAUDE.md venv policy)

## Key context

- BirdCLEF 2026 focal data has **clip-level labels only** — no per-frame annotations
- No nocall/background class — all 206 dirs are species
- Activity scores have known issues with short recordings (1-2 chunks get degenerate scores)
- Murilo's exp 015 showed pseudo-labels + GAP didn't help → possible GAP bottleneck
- Murilo's branch `origin/mg/012-soundscape-finetune` has related SED work (exp 016)

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
