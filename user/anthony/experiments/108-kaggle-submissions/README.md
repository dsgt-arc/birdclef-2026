# Experiment 108: Kaggle single-model submissions

Ship the best available single models individually to Kaggle to measure
the local ↔ LB gap, then decide whether to ensemble.

## Submission matrix

| # | Model | Source | Local metric | Status |
|---|---|---|---|---|
| ref | Exp 102 Run C (EfficientNet LoRA+attn) | exp 102 | 0.794 LB | already on LB |
| 1 | Exp 106 ridge | `exp106/artifacts/ridge_model.npz` | 0.749 LOSO AUROC | pending |
| 2 | Exp 107 v4 (all-data retrain) | new retrain | ~0.707 LOSO | pending |
| 3 | Exp 107 v1-no-gate | new retrain | 0.716 S22 AUROC | pending |

## Shared inference module: `birdclef_kaggle`

| Module | Purpose |
|---|---|
| `perch_session` | Perch v2 ONNX wrapper (1536-d embedding + 14795 logits) |
| `row_id` | Parse `BC2026_Test_NNNN_S##_..._W` → `(soundscape, site_idx, window_idx)` with OOD-site flagging |
| `species` | Canonical 234-species ordering (proxies `exp106.data_loading.build_species_list`) |
| `audio` | Soundfile loader + deterministic 60s → 12×5s window stacker |
| `submission` | CSV writer aligned against Kaggle's sample_submission.csv |
| `ood` | OOD-site fallback (site_idx ∉ S01..S23 → site_score=1) |

## Per-submission layout

Each `submissions/<name>/` contains:

- `infer.py` — pure-Python entrypoint: load weights → Perch embed → model forward → submission.csv
- `kernel.ipynb` — Kaggle kernel wrapping `infer.py` with offline wheel install
- `README.md` — upload checklist + LB result notes

## Run

```bash
# Test the shared module (CPU, ~5 min)
sbatch sbatch/test.sbatch

# Dry-run inference locally on a training soundscape
sbatch sbatch/dry_run_infer.sbatch
```

## Directory structure

```
108-kaggle-submissions/
├── pyproject.toml              # tool.uv.sources -> exp106, exp107 editable
├── src/birdclef_kaggle/        # shared inference utilities
├── submissions/                # per-model Kaggle kernels
├── sbatch/                     # test + dry-run jobs
└── tests/                      # pytest for shared module
```
