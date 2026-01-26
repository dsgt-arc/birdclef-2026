# Experiment 001: Inference Benchmark

Benchmark inference timing for OpenSoundscape, BEATs, and Bioacoustics Model Zoo models.

## Overview

- **Goal**: Measure inference timing for different audio models on 1-minute audio (16kHz).
- **Models**: BirdNET, Perch, BEATs, BirdSet (ConvNeXT, EfficientNet), HawkEars, YAMNet, RanaSierraeCNN.
- **Hardware**:
    - CPU: Intel Xeon Gold 6226 (Cascade Lake), 4 cores / 4GB RAM
    - GPU: NVIDIA Tesla V100 16GB

## Results (60s Audio)

| Model | CPU Mean (s) | GPU Mean (s) | Speedup | Notes |
|-------|-------------:|-------------:|--------:|-------|
| **YAMNet** | 0.14 | 0.11 | 1.3x | Extremely fast on both |
| **BirdSet (EfficientNetB1)** | 0.85 | 0.21 | 4.0x | |
| **BirdSet (ConvNeXT)** | 1.54 | 0.21 | 7.3x | |
| **RanaSierraeCNN** | 1.06 | 0.23 | 4.6x | |
| **HawkEars** | 3.17 | 0.27 | 11.7x | Significant GPU acceleration |
| **BEATs** | 5.51 | 0.29 | 19.0x | Massive GPU acceleration |
| **HawkEars (Embedding)** | 0.87 | - | - | |
| **BirdNET** | 1.92 | 1.91 | 1.0x | TFLite runtime (CPU bound) |
| **HawkEars (v0.10)** | 2.87 | - | - | |
| **Perch** | 4.15 | *Failed* | - | Tensorflow XLA issues on GPU |
| **Perch2** | 8.76 | *Failed* | - | Tensorflow XLA issues on GPU |

*Note: GPU times include data transfer but models are pre-loaded.*

## Quick Start

### Prerequisites
1. Set up scratch directories (see `docs/operations.md`)
2. Download checkpoints (handled automatically or via `download_checkpoints.py`)

### Run via SLURM

**CPU Benchmark:**
```bash
sbatch user/adrian/experiments/001-inference-benchmark/sbatch/run_benchmark_cpu.sbatch
```

**GPU Benchmark:**
```bash
sbatch user/adrian/experiments/001-inference-benchmark/sbatch/run_benchmark_gpu.sbatch
```

### Run locally (dry run)
```bash
source ~/scratch/birdclef-2026/.venv/bin/activate
uv sync --active
uv pip install -e user/adrian

# Run specific models
python -m birdclef_adrian.benchmark_inference --models BirdNET BEATs --device cpu --dry-run
```

## Configuration

Environment variables for `sbatch`:
- `MODELS`: Space-separated list of models (default: all available)
- `DURATION`: Audio duration in seconds (default: 60.0)
- `NUM_RUNS`: Number of benchmark runs (default: 10)
- `WARMUP_RUNS`: Number of warmup runs (default: 1 or 2)

## Output

Results saved to:
- CPU: `~/scratch/birdclef-2026/outputs/001-inference-benchmark-cpu/benchmark_results.json`
- GPU: `~/scratch/birdclef-2026/outputs/001-inference-benchmark/benchmark_results.json`
