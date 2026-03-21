---
experiment: "011-lora-baseline-2026"
status: ready
date: 2026-03-14
author: Murilo Gustineli
---

# Experiment 011: LoRA Baseline on BirdCLEF 2026

## Hypothesis

The proven LoRA fine-tuning pipeline from BirdCLEF 2025 (exps 006/007/009) will produce
competitive baselines on BirdCLEF 2026 focal data by simply retraining on the new dataset
(35,549 recordings, ~206 focal species) with identical hyperparameters.

## Motivation

BirdCLEF 2025 models score poorly on 2026 because only 41/234 species overlap. We need
baselines retrained on the 2026 focal training data. The LoRA pipeline is validated and
the data format is identical — only the data path changes.

## Approach

- **Encoders**: EAT-base, ConvNeXt-Base, ProtoCLR/CvT-13
- **LoRA config**: rank=8, alpha=16, dropout=0.1 (same as 2025)
- **Data**: BirdCLEF 2026 `train_audio/` (focal recordings only; 28 soundscape-only
  insect sonotypes will not have training data)
- **Unified script**: Single `train.py` handles all 3 encoders via config dispatch
- **BirdCLEFDataModule**: Reused unchanged (format-compatible with 2026 data)

## What Changes from 2025

| Item | From (2025) | To (2026) |
|------|-------------|-----------|
| `data.data_dir` | `~/ps-clef2026_birdclef-0/data/birdclef-2025` | `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026` |
| Recordings | 28,564 | 35,549 |
| W&B project | per-encoder projects | `birdclef-2026-lora-finetune` (unified) |
| Training script | 3 separate scripts | 1 unified `train.py` |

## Success Criteria

- Pipeline runs end-to-end on all 3 configs (fast-dev-run passes)
- `build_species_list()` detects ~206 focal species
- cmAP/AUROC in similar range as 2025 results (0.5-0.9)
- All 3 models train to convergence within 50 epochs

## Risks

- 2026 may have different class balance / geography → different optimal hyperparameters
  (mitigation: tune in a follow-up experiment)
- 28 soundscape-only species get near-zero predictions (expected, separate experiment)
