# Experiment 001: Inference Benchmark

Benchmark inference timing for OpenSoundscape and BEATs models.

## Overview

- **Goal**: Measure inference timing for different audio models on 1-minute audio.
- **Models**: BirdNET, Perch (OpenSoundscape), BEATs.

## Quick Start

### Prerequisites
1. Set up scratch directories (see `docs/operations.md`)
2. Download BEATs checkpoint to `~/scratch/birdclef-2026/checkpoints/beats/BEATs_iter3_plus_AS2M.pt`

### Run via SLURM
```bash
# Submit the benchmark job
cd ~/arc/birdclef-2026
sbatch user/adrian/experiments/001-inference-benchmark/sbatch/run_benchmark.sbatch

# Monitor job
squeue -u $USER
tail -f logs/benchmark-<job_id>.out
```

### Run locally (dry run)
```bash
export UV_PROJECT_ENVIRONMENT=~/scratch/birdclef-2026/.venv
uv run python -m birdclef.benchmark_inference --dry-run
```

## Configuration

Environment variables for `sbatch`:
- `MODELS`: Models to benchmark (default: "BirdNET beats")
- `DURATION`: Audio duration in seconds (default: 60.0)
- `NUM_RUNS`: Number of benchmark runs (default: 10)
- `WARMUP_RUNS`: Number of warmup runs (default: 2)

## Output

Results saved to `~/scratch/birdclef-2026/outputs/001-inference-benchmark/benchmark_results.json`.
