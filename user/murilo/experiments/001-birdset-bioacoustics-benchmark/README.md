# Experiment 001: BirdSet Bioacoustics Model Benchmark

Benchmarking pre-trained bioacoustics models from the Bioacoustics Model Zoo on the BirdSet dataset using linear probing.

## Quick Start

### 1. Interactive Testing (Recommended First Step)

Request an interactive GPU session and run the test script:

```bash
# Request interactive GPU session
salloc -p gpu-rtx6000 --gres=gpu:1 -c 4 --mem=32G -t 2:00:00

# Once allocated, run test script
cd ~/arc/birdclef-2026/user/murilo/experiments/001-birdset-bioacoustics-benchmark
bash test_setup.sh
```

This will:
- ✅ Verify GPU availability
- ✅ Create isolated venv in `$TMPDIR` (fast local I/O)
- ✅ Install all dependencies via `uv`
- ✅ Test PyTorch CUDA support
- ✅ Load and test bioacoustics-model-zoo models
- ✅ Verify BirdSet dataset accessibility
- ✅ Run quick embedding extraction test
- ✅ Test scikit-learn multi-label classifier

**Expected runtime**: ~10-15 minutes (first install)

### 2. Full Experiment (Coming Soon)

After verifying the environment works:

```bash
# Stage 1: Extract embeddings (GPU job, 12 hours)
sbatch sbatch/extract_embeddings.sbatch

# Stage 2: Train linear probes (GPU job, 2 hours)
sbatch sbatch/train_probe.sbatch
```

## Project Structure

```
001-birdset-bioacoustics-benchmark/
├── README.md              # This file
├── proposal.md            # Hypothesis and methodology
├── design.md              # Technical decisions
├── tasks.md               # Implementation checklist
├── config.yaml            # Experiment configuration
├── test_setup.sh          # Interactive environment test (executable)
├── sbatch/                # SLURM job scripts (TODO)
│   ├── extract_embeddings.sbatch
│   └── train_probe.sbatch
└── artifacts/             # Results (created during run)
    ├── embeddings/
    ├── classifiers/
    ├── results_summary.json
    └── plots/
```

## Models Under Test

All models from `bioacoustics-model-zoo`:

| Model | Window (sec) | Embed Size | Notes |
|-------|--------------|------------|-------|
| BirdNET | 3.0 | 1024 | ~6,500 bird species |
| YAMNet | 0.96 | 1024 | General audio classification |
| Perch | 5.0 | 1280 | Google's bioacoustics model |
| HawkEars | 3.0 | 2048 | Raptor-focused |
| BirdSetConvNeXT | 5.0 | 1024 | Pre-trained on BirdSet |
| BirdSetEfficientNetB1 | 5.0 | 1280 | Pre-trained on BirdSet |
| RanaSierraeCNN | 2.0 | 512 | Frog-focused |

## Environment Details

**Virtual Environment Setup**:
- Location: `$TMPDIR/birdset-benchmark-$SLURM_JOB_ID/.venv` (SLURM jobs)
- Location: `~/tmp/birdset-benchmark-$$/.venv` (interactive sessions)
- Python: 3.11
- Package manager: `uv` (fast, deterministic)

**Key Dependencies**:
- `torch` (CUDA 12.1)
- `bioacoustics-model-zoo` (from GitHub)
- `datasets` (HuggingFace)
- `scikit-learn` (linear probes)
- `polars` (data processing)

See `../pyproject.toml` for complete dependency list.

## Metrics

- **cmAP**: Class-mean Average Precision (primary, BirdSet standard)
- **F1 Micro**: Overall classification quality
- **F1 Macro**: Per-class average (handles imbalance)
- **Top-1/Top-5 Accuracy**: Ranking quality

## Methodology

**Two-Stage Pipeline**:
1. **Stage 1 - Embedding Extraction** (GPU, 12h):
   - Load BirdSet HSN train + test_5s splits
   - For each model: extract embeddings, save to `.npz`
   - Handle variable audio lengths via max-pooling

2. **Stage 2 - Linear Probing** (GPU, 2h):
   - Load pre-extracted embeddings
   - Train `OneVsRestClassifier(LogisticRegression)`
   - Evaluate with cmAP, F1, accuracy metrics
   - Generate visualizations and per-class analysis

## Status

- [x] Proposal written
- [x] Design document completed
- [x] Tasks defined
- [x] `pyproject.toml` created
- [x] Test script created (`test_setup.sh`)
- [ ] Environment tested on GPU (run `test_setup.sh`)
- [ ] SLURM scripts created
- [ ] Code implementation
- [ ] Embeddings extracted
- [ ] Linear probes trained
- [ ] Results analyzed

## Next Steps

1. **Test environment** (you are here):
   ```bash
   salloc -p gpu-rtx6000 --gres=gpu:1 -c 4 --mem=32G -t 2:00:00
   bash test_setup.sh
   ```

2. **Verify test passes**, then proceed to code implementation

3. **Implement data pipeline** (Phase 2 in `tasks.md`)

4. **Create SLURM scripts** (Phase 4-5 in `tasks.md`)

## Troubleshooting

**Issue**: `bioacoustics-model-zoo` import fails
- **Fix**: Ensure installed from GitHub: `uv pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo.git@main`

**Issue**: BirdSet dataset not accessible
- **Fix**: Authenticate with HuggingFace: `huggingface-cli login`

**Issue**: CUDA out of memory
- **Fix**: Reduce batch size in `config.yaml` (default: 16 → try 8)

**Issue**: PyTorch not seeing GPU
- **Fix**: Verify CUDA version matches: `nvidia-smi` vs `torch.version.cuda`

## Contact

**Author**: Murilo Gustineli (mgustineli3@gatech.edu)
**Created**: 2026-01-24
**Status**: Proposed (Phase 1: Environment Setup)
