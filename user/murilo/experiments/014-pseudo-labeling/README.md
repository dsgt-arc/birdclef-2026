# Experiment 014: Teacher Pseudo-Labeling Pipeline

Pseudo-label unlabeled soundscapes using exp 012 teacher ensemble. Supports direct
ensemble (Mode A) and MixIT-enhanced inference (Mode B, Bird MixIT paper Section 7.1).

## Quickstart

```bash
cd user/murilo/experiments/014-pseudo-labeling

# Validation on 66 labeled files (quick, ~5 min on GPU)
bash sbatch/run.sh config/mode_a.yaml --split labeled

# Full run on all unlabeled soundscapes (~1-2h Mode A, ~4-6h Mode B)
bash sbatch/run.sh config/mode_a.yaml --split unlabeled

# MixIT-enhanced (requires exp 013 separation complete)
bash sbatch/run.sh config/mode_b.yaml --split labeled

# Test with 5 files
bash sbatch/run.sh config/mode_a.yaml --split labeled --max-files 5

# Dry run
bash sbatch/run.sh config/mode_a.yaml --split labeled --dry-run
```

## Modes

### Mode A: Direct Ensemble

Each teacher runs on the original audio. Weighted average across models with
per-taxon weights (EAT excluded for insects per error analysis).

### Mode B: MixIT-Enhanced (Bird MixIT Paper Section 7.1)

Each teacher runs on original + 4 MixIT separated channels (5 forward passes).
Max-pool across channels per model, then weighted average across models.
Falls back to Mode A for files without MixIT separation.

## Output

```
shared/pseudo-labels/birdclef-2026/{run_name}/
├── logits/
│   ├── protoclr.npz         # (n_segments, 234) raw logits
│   ├── convnext.npz
│   └── eat.npz
├── pseudo_labels.parquet     # (n_segments, 236 cols: filename, end_sec, 234 probs)
└── config.json               # run metadata
```

## Dependencies

- Exp 012 ONNX models (in `010-submission-pipeline/artifacts/weights/`)
- Exp 013 MixIT separation (for Mode B only, in `shared/mixit/pretrained/soundscapes/`)
