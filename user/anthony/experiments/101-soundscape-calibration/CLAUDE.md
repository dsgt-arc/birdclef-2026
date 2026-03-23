# Experiment 101: Soundscape Label Calibration

## What this experiment does

Evaluates teacher models against 739 labeled BirdCLEF 2026 soundscape windows
to determine the best pseudo-labeling strategy for 176.5h unlabeled soundscapes.

## Key results

- **Ensemble AUROC = 0.880** (ProtoCLR + ConvNeXt + EAT mean)
- **Optimal threshold = 0.12–0.20**, not 0.5 (compressed probabilities from multi-species windows)
- **Best F1 = 0.756** at threshold 0.12 on S22 test set
- **Spectral contrast = 0.888 AUROC** for call/no-call (no ML needed)
- **Linear probes are broken** on soundscapes (0.65–0.71 AUROC, domain shift)
- **Decision**: Strategy A — global threshold + confidence weighting for pseudo-labeling

## Data

- Ground truth: `100-eda/artifacts/soundscape_labels.parquet` (739 windows, 66 files)
- Murilo's predictions: `shared/inference/birdclef-2026/{protoclr,convnext,eat}-2026-soundscape/labeled.npz`
- BirdNET/Perch embeddings: `~/scratch/birdclef-2026/outputs/exp010-2026/labels/`
- Probe weights: `~/scratch/birdclef-2026/outputs/exp101/probes/`

## Running scripts

All scripts run via sbatch (CPU-only, use exp 010 venv):

```bash
cd user/anthony/experiments/101-soundscape-calibration
sbatch sbatch/evaluate.sbatch           # Phase 2: raw model AUROC
sbatch sbatch/evaluate-probes.sbatch    # Probe comparison
sbatch sbatch/detailed.sbatch           # Per-species TP/FP breakdown
sbatch sbatch/threshold-sweep.sbatch    # Threshold selection via LOSO CV
sbatch sbatch/call-nocall.sbatch        # Model-based call detection
sbatch sbatch/indices-callnocall.sbatch # Acoustic indices call detection
sbatch sbatch/train-probes.sbatch       # Train BirdNET/Perch probes (needs temp venv)
```

Venv: `~/scratch/birdclef-2026/.venv-010` (has numpy, pandas, sklearn, pyarrow).
Probe training creates its own temp venv (needs torch).

## Key gotcha

Murilo's npz `seg_keys` use **end times** (5, 10, ..., 60), not start times.
Convert: `start_sec = seg_key_window_id - 5.0`. See `scripts/debug_alignment.py`.

## Directory structure

```
├── proposal.md              # Hypothesis, method, decision gates
├── results.md               # Full results with all analyses
├── tasks.md                 # Task checklist
├── CLAUDE.md                # This file
├── docs/
│   ├── available-predictions.md   # NPZ format and paths
│   ├── ground-truth-summary.md    # 739 windows, species counts
│   ├── split-strategy.md          # S22 test, 8-site LOSO CV
│   ├── exp013-diagnosis.md        # FlyingSquid failure hypotheses
│   ├── pseudo-label-strategies.md # 5 strategies compared, decision record
│   └── call-nocall-detector.md    # Acoustic indices for activity detection
├── scripts/                 # Analysis scripts (python, run via sbatch)
├── sbatch/                  # SLURM job scripts
└── artifacts/               # Output CSVs, JSONs (gitignored)
```
