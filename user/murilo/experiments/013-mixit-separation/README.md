# Experiment 013: MixIT Source Separation

Use Bird MixIT source separation as offline preprocessing to separate 10,592
unlabeled soundscapes into 4 channels per file. The separated audio is consumed
by downstream experiments (exp 014 pseudo-labeling, exp 015 student training).

## Status: Phase 2 Complete (separation running)

- Phase 1 (TF1 → ONNX conversion): Complete
- Phase 2 (offline separation): Complete for labeled (66 files), running for unlabeled (~10,592 files)

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

## Quickstart

All commands assume you are in the experiment directory:

```bash
cd user/murilo/experiments/013-mixit-separation
```

### Phase 1: Convert Bird MixIT TF1 → ONNX

Converts the pretrained TF1 checkpoint to ONNX format for portable inference.
Requires `tensorflow` + `tf2onnx` (installed automatically in the SLURM job venv).

```bash
# Submit conversion job (cpu-small, ~15 min)
bash sbatch/run.sh convert

# Dry run (preview command)
bash sbatch/run.sh convert --dry-run
```

Output: `/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx`

### Phase 2: Run MixIT separation on soundscapes

Runs the ONNX model on soundscape files, saving 4 separated channels per file as
compressed NPZ arrays. Supports resume (skips already-processed files).

```bash
# Smoke test (10 files, verify separation works)
bash sbatch/run.sh separate --test

# Separate labeled soundscapes only (for Phase 3 validation)
bash sbatch/run.sh separate --labeled

# Separate all unlabeled soundscapes (~10,592 files, ~2 hours on 10 GPUs)
bash sbatch/run.sh separate --num-workers 10

# Single GPU (slower, ~20 hours)
bash sbatch/run.sh separate

# Dry run
bash sbatch/run.sh separate --num-workers 10 --dry-run
```

Output: `artifacts/separated/` (unlabeled), `artifacts/separated_labeled/` (labeled),
`artifacts/separated_test/` (test).

### Phase 3: Teacher pseudo-labeling (depends on exp 012)

*Not yet implemented.* Will run exp 012 ONNX models on separated + original audio.

```bash
# (future) bash sbatch/run.sh pseudo-label
```

### Phase 4: Student training (depends on Phase 3)

*Not yet implemented.* Will fine-tune from exp 012 checkpoint on pseudo-labeled data.

```bash
# (future) bash sbatch/run.sh train --config config/...
```

## Known Limitation: Non-Bird Species

The pretrained Bird MixIT was trained on bird audio only. BirdCLEF 2026 includes
amphibians (35 species), insects (28 species), mammals, and reptiles. Separation
quality will likely degrade for non-bird taxa. If Phase 3 validation shows poor
per-taxon performance, a domain-matched MixIT model trained on BirdCLEF data
(via Asteroid/PyTorch) would be the follow-up experiment.

## Dependencies

- Bird MixIT TF1 checkpoint: `/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/`
- `tf2onnx` (Phase 1 only, installed in job venv), `onnxruntime` (Phases 2+)
- Exp 012 completed (Phases 3–4 only: 234-class ONNX teacher models)

## Data

| Source | Files | Purpose |
|--------|-------|---------|
| Unlabeled soundscapes | 10,592 | Pseudo-label targets (Phase 2–3) |
| Labeled soundscapes | 66 | Pseudo-label quality validation (Phase 3) |
| Focal recordings | 35,549 | Anti-forgetting anchor (Phase 4) |

## Checkpoint Paths

| Model | Path |
|-------|------|
| Bird MixIT (TF1) | `/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/model.ckpt-3223090` |
| Bird MixIT (ONNX) | `/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx` |

## References

- [Bird MixIT paper](../../docs/references/bird-mixit-paper.md)
- [Bird MixIT vendor doc](../../docs/vendor/bird-mixit.md)
- [Bird MixIT GitHub](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit)
- [BirdCLEF 2026 competition](../../docs/references/birdclef-2026-competition.md)

## Directory Structure

```
013-mixit-separation/
├── README.md
├── proposal.md
├── tasks.md
├── pyproject.toml
├── sbatch/
│   ├── run.sh                         # Launcher (convert, separate, ...)
│   ├── convert.sh                     # Phase 1: TF1 → ONNX (cpu-small, 1hr)
│   └── separate.sh                    # Phase 2: batch separation (cpu-small, 8hr)
├── scripts/
│   ├── convert_mixit_onnx.py          # Phase 1: TF1 → ONNX conversion + validation
│   ├── separate_soundscapes.py        # Phase 2: batch separation with resume
│   └── pseudo_label.py                # Phase 3: teacher ensemble (not yet implemented)
├── config/                            # Phase 4: training configs
├── artifacts/
│   ├── separated/                     # NPZ files for unlabeled soundscapes
│   ├── separated_labeled/             # NPZ files for labeled soundscapes (validation)
│   └── separated_test/                # NPZ files for smoke test
└── outputs/                           # Training outputs (Phase 4)
```
