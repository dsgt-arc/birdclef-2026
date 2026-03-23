---
experiment: "014-pseudo-labeling"
status: proposed
date: 2026-03-22
author: Murilo Gustineli
---

# Experiment 014: Teacher Pseudo-Labeling Pipeline

## Hypothesis

Running teacher models on unlabeled soundscapes (with and without MixIT-separated
channels) produces soft pseudo-labels whose quality exceeds single-model direct
prediction, enabling downstream student training (exp 015) to improve LB scores
beyond the exp 012 baseline (0.767).

## Motivation

BirdCLEF 2026 provides 10,592 unlabeled soundscapes — 160× more soundscape data
than the 66 labeled files used in exp 012. Every recent BirdCLEF winner (2024 1st,
2025 1st) uses pseudo-labeling to leverage this data. The 2025 1st place solution
used 5/7 models trained with iterative pseudo-labels.

This experiment is **decoupled from student training** (exp 015) so that:
- Pseudo-labels can be regenerated with stronger teachers without retraining
- Multiple student architectures/configurations can consume the same labels
- MixIT-enhanced vs direct pseudo-labeling can be compared as an ablation

## Approach

### Inputs

- **Teacher models**: Exp 012 ONNX models (ProtoCLR, ConvNeXt, EAT — 234 classes)
- **Unlabeled audio**: 10,592 soundscape files in `shared/data/birdclef-2026/train_soundscapes/`
- **MixIT separated audio** (optional): 4 channels per file from exp 013
  (`shared/mixit/pretrained/soundscapes/`)
- **Labeled soundscapes**: 66 files for quality validation

### Pipeline

```
For each unlabeled soundscape (12 × 5s segments per file):

  Mode A — Direct (no MixIT):
    Original audio → ProtoCLR → probs_pcr
                   → ConvNeXt → probs_cnx
                   → EAT     → probs_eat (exclude for insect species)
    Per species: weighted_mean(probs_pcr, probs_cnx, [probs_eat])

  Mode B — MixIT-enhanced:
    Original + 4 separated channels → each teacher model
    Per model: max-pool across 5 inputs (4 channels + original)
    Per species: weighted_mean across models

  Final: soft pseudo-labels (234 probabilities per 5s segment)
```

### Key Design Decisions

1. **Weighted ensemble, not simple mean**: Error analysis (exp 013 error-analysis.md)
   showed EAT fails on insects (0.673 AUROC). Exclude EAT for insect species or
   use per-taxon weights (e.g., ProtoCLR 0.4 + ConvNeXt 0.4 + EAT 0.2 for birds;
   ProtoCLR 0.5 + ConvNeXt 0.5 for insects).

2. **Temperature scaling**: Following 2025 1st place (T=0.55–0.6), apply temperature
   scaling to sharpen predictions before averaging. Lower T → higher confidence on
   strong predictions, less noise.

3. **Confidence thresholding**: Sweep thresholds (0.3, 0.5, 0.7) on the 66 labeled
   soundscapes to find the optimal precision/recall tradeoff.

4. **Hard labels for labeled data**: Per Anthony's exp A-014 finding, never replace
   ground truth with pseudo-labels. Pseudo-labels are only for the ~10,526 unlabeled
   soundscapes.

5. **Mode A vs Mode B comparison**: Run both direct and MixIT-enhanced pseudo-labeling
   on the 66 labeled files. Compare pseudo-label quality against ground truth to
   measure MixIT's actual contribution.

### Output

```
shared/pseudo-labels/birdclef-2026/{run_name}/
├── pseudo_labels.parquet    # (n_segments, 236 cols: filename, end_sec, 234 species probs)
├── config.json              # teacher models, weights, temperature, threshold
└── validation/
    └── labeled_quality.json # precision/recall on 66 labeled files
```

## Evaluation

### Metrics
- **Pseudo-label quality on labeled soundscapes**: Per-species precision/recall/AUROC
  against ground truth labels
- **Per-taxon quality**: Birds vs amphibians vs insects breakdown
- **MixIT gain**: Mode B quality − Mode A quality
- **Coverage**: % of unlabeled segments with ≥1 species above threshold

### Baseline
- Exp 012 models' direct predictions (no ensemble, no temperature scaling)

### Success Criteria
- **Confirm if**: Ensemble pseudo-labels have higher AUROC than any single model
  on labeled soundscapes
- **Confirm if**: MixIT-enhanced (Mode B) outperforms direct (Mode A) on labeled set
- **Reject if**: No quality improvement from ensemble or MixIT over single-model predictions

## Dependencies

- Exp 012 ONNX models (complete)
- Exp 013 MixIT separated audio (running, ~10,500/10,592 files done)
- Inference script: `user/murilo/scripts/infer_soundscapes.py` (complete)

## Leads to

Exp 015 (student training) consumes the pseudo-labels produced here.
Future iterations: regenerate pseudo-labels with stronger teachers (e.g., SED models)
and re-run exp 015.
