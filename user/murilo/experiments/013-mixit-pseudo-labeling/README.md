# Experiment 013: MixIT Source Separation + Pseudo-Labeling

Use Bird MixIT source separation as offline preprocessing to generate cleaner
pseudo-labels from 10,592 unlabeled soundscapes, then fine-tune models on
pseudo-labeled data.

## Status: Proposed

Depends on exp 012 (labeled soundscape fine-tuning) for teacher models.

## Overview

```
Unlabeled soundscapes (10,592 files)
    ↓
Bird MixIT separation → 4 channels + original
    ↓
Exp 012 teacher ensemble (EAT + ConvNeXt + ProtoCLR)
    ↓
Aggregation: max(channels) per model → mean(models)
    ↓
Soft pseudo-labels (234 species probabilities per 5s segment)
    ↓
Student training on focal + labeled soundscape + pseudo-labeled data
```

## Pseudo-Labeling Pipeline (Phase 3)

```
For each unlabeled soundscape (12 × 5s segments per file):
    ├── Original audio → EAT → probs_eat_orig
    │                  → ConvNeXt → probs_cnx_orig
    │                  → ProtoCLR → probs_pcr_orig
    │
    ├── Channel 1 → EAT/ConvNeXt/ProtoCLR → probs per model
    ├── Channel 2 → same
    ├── Channel 3 → same
    └── Channel 4 → same

Per model: max-pool across 5 inputs (4 channels + original)
Final: average across 3 models → soft pseudo-labels per segment
```

The original audio is always included — if separation fails (e.g., for non-bird taxa),
the max-pool falls back to the original prediction.

## Phases

1. **Convert Bird MixIT TF1 → ONNX** (`scripts/convert_mixit_onnx.py`)
2. **Offline separation** (`scripts/separate_soundscapes.py`) — CPU batch job, soundscapes only (not focal)
3. **Teacher pseudo-labeling** (`scripts/pseudo_label.py`) — using exp 012 ONNX models
4. **Student training** (`train.py`) — fine-tune on mixed data with soft BCE loss

## Known Limitation: Non-Bird Species

The pretrained Bird MixIT was trained on bird audio only. BirdCLEF 2026 includes
amphibians (35 species), insects (28 species), mammals, and reptiles. Separation
quality will likely degrade for non-bird taxa. If Phase 3 validation shows poor
per-taxon performance, a domain-matched MixIT model trained on BirdCLEF data
(via Asteroid/PyTorch) would be the follow-up experiment.

## Dependencies

- Exp 012 completed (234-class ONNX teacher models)
- Bird MixIT TF1 checkpoint from Google Cloud Storage
- `tf2onnx` (Phase 1 only), `onnxruntime` (Phases 2-4)

## Data

| Source | Files | Purpose |
|--------|-------|---------|
| Unlabeled soundscapes | 10,592 | Pseudo-label targets |
| Labeled soundscapes | 66 | Pseudo-label quality validation |
| Focal recordings | 35,549 | Anti-forgetting anchor |

## References

- [Bird MixIT paper](../../docs/references/bird-mixit-paper.md)
- [Bird MixIT vendor doc](../../docs/vendor/bird-mixit.md)
- [BirdCLEF 2026 competition](../../docs/references/birdclef-2026-competition.md)

## Directory Structure

```
013-mixit-pseudo-labeling/
├── README.md
├── proposal.md
├── tasks.md
├── pyproject.toml
├── config/              # Training configs (Phase 4)
├── scripts/
│   ├── convert_mixit_onnx.py    # Phase 1: TF1 → ONNX
│   ├── separate_soundscapes.py  # Phase 2: batch separation
│   └── pseudo_label.py          # Phase 3: teacher ensemble
├── sbatch/              # SLURM jobs
├── artifacts/           # Separated audio, pseudo-labels
└── outputs/             # Training outputs
```
