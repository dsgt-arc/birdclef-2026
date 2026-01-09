# Anthony's Workspace

BirdCLEF 2026 experiments and code.

## Structure

```
.
├── src/birdclef/          # Python package
├── experiments/           # Experiment tracking + job configs
│   └── NNN-name/
│       ├── proposal.md    # Hypothesis and method
│       ├── results.md     # Outcomes (after completion)
│       ├── config.yaml    # Training configuration
│       ├── train.sbatch   # SLURM job script
│       └── artifacts/     # Logs, checkpoints
└── pyproject.toml         # Package dependencies
```

## Quick Start

```bash
# Install dependencies
cd user/anthony
uv sync

# Run an experiment
sbatch experiments/001-baseline/train.sbatch

# Check job status
squeue -u $USER
```

## Creating a New Experiment

1. Copy the template:
   ```bash
   cp -r experiments/000-template experiments/NNN-name
   ```

2. Edit `proposal.md` with your hypothesis

3. Configure `config.yaml` for your experiment

4. Update `train.sbatch` with correct paths and resources

5. Submit: `sbatch experiments/NNN-name/train.sbatch`

6. After completion, fill in `results.md`

## Current Experiments

| ID | Name | Status | Description |
|----|------|--------|-------------|
| 000 | template | - | Template for new experiments |
