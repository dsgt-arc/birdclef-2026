# Tasks: BirdSet Bioacoustics Model Benchmark

**Status**: Phase 1 complete, Phase 2 in progress
**Current Phase**: Phase 2.1 (Implement embedding extraction)
**Last Updated**: 2026-01-24

---

## Overview

This experiment evaluates 4 pre-trained bioacoustics models across 8 BirdSet datasets using linear probing.

**Models**: BirdNET, Perch, BirdSetConvNeXT, BirdSetEfficientNetB1
**Datasets**: HSN, PER, NES, UHH, POW, SSW, SNE, NBP
**Total Combinations**: 4 models × 8 datasets = 32

## Phase 0: Planning and Documentation ✅ COMPLETED

- [x] 0.1 Create experiment proposal
- [x] 0.2 Create technical design document
- [x] 0.3 Create implementation task list
- [x] 0.4 Create configuration and documentation
- [x] 0.5 Commit and push to repository

## Phase 1: Environment Setup ✅ COMPLETED

- [x] 1.1 Create Python package structure
  - [x] `user/murilo/pyproject.toml` with all dependencies
  - [x] `user/murilo/src/birdset_benchmark/__init__.py`
  - [x] Fixed git URL dependency (moved to `[tool.uv.sources]`)
  - [x] Added `ai-edge-litert` for TensorFlow Lite (BirdNET requirement)
  - [x] Added `tensorflow==2.17` and `tensorflow-hub` for Perch model support
  - [x] Made `pytorch-lightning` and `torchmetrics` explicit dependencies
  - [x] Added `pytest` to dependencies
  - [x] Created comprehensive pytest test suite (28 tests total)

- [x] 1.2 Install and test bioacoustics-model-zoo
  - [x] Test `import bmz; bmz.list_models()` works (13 models available)
  - [x] Verify BirdNET can be instantiated and extract embeddings
  - [x] Verify Perch model loading (requires TensorFlow)
  - [x] Test Perch embedding extraction (1280-dim embeddings)
  - [x] Created pytest tests: `tests/test_models.py`

- [x] 1.3 Test BirdSet dataset loading
  - [x] Load HSN subset from `DBD-research-group/BirdSet`
  - [x] Verify train and test_5s splits exist
  - [x] Document dataset statistics
  - [x] Created 6 comprehensive pytest tests

- [x] 1.4 Create experiment directory structure
  - [x] Base directory and config files
  - [x] `test_setup.sh` for environment setup and pytest validation
  - [x] Test suite organized by experiment number
  - [x] `sbatch/` and `artifacts/` directories

## Phase 2: Embedding Extraction Pipeline 🔄 IN PROGRESS

### 2.1 Implement Data Pipeline
- [ ] Create `src/birdset_benchmark/data.py`
  - [ ] Function: `load_birdset_dataset(dataset_code)` - Load any of 8 datasets
  - [ ] Function: `resample_audio(audio, source_sr, target_sr)`
  - [ ] Function: `pad_or_crop(audio, target_duration_sec, sr)`
  - [ ] Function: `max_pool_embeddings(embeddings_sequence)` for long clips
  - [ ] Class: `BirdSetDataset` (PyTorch Dataset wrapper)
  - [ ] Function: `collate_fn()` for batching variable-length audio
  - [ ] Unit tests for edge cases

### 2.2 Implement Model Encoders
- [ ] Create `src/birdset_benchmark/encoders.py`
  - [ ] Class: `BioacousticsEncoder` (abstract base class)
  - [ ] Class: `BirdNETEncoder` (wrapper for bmz.BirdNET)
  - [ ] Class: `PerchEncoder` (wrapper for bmz.Perch)
  - [ ] Class: `BirdSetConvNeXTEncoder` (wrapper for bmz.BirdSetConvNeXT)
  - [ ] Class: `BirdSetEfficientNetEncoder` (wrapper for bmz.BirdSetEfficientNetB1)
  - [ ] Each encoder: `.embed(audio_batch)` → `[batch, embedding_dim]`
  - [ ] Handle sample rate differences (48kHz for BirdNET, 32kHz for others)
  - [ ] Unit tests for each encoder

### 2.3 Implement Embedding Extraction Script
- [ ] Create `src/birdset_benchmark/extract_embeddings.py`
  - [ ] CLI with typer: `--model`, `--dataset`, `--batch-size`, `--output-dir`
  - [ ] Load BirdSet dataset for specified split
  - [ ] Initialize model encoder
  - [ ] Loop over DataLoader with tqdm progress bar
  - [ ] Save embeddings to `.npz` format
  - [ ] Save metadata to `metadata.json`
  - [ ] Logging: model name, sample rate, embedding dim, throughput

### 2.4 Create SLURM Job Scripts
- [ ] Create `sbatch/common.sh`
  - [ ] Function: `setup_environment()` - venv in TMPDIR
  - [ ] Function: `get_artifact_dir()` - path construction
  - [ ] Environment variables: PROJECT_ROOT, etc.

- [ ] Create `sbatch/extract_embeddings/_job.sh`
  - [ ] SLURM configuration: 1 GPU, 4 CPUs, 32GB RAM, 4 hours
  - [ ] Source common.sh
  - [ ] Run `python -m birdset_benchmark.extract_embeddings`
  - [ ] Use MODEL and DATASET environment variables

- [ ] Create `sbatch/extract_embeddings/run.sh`
  - [ ] Parse `--model`, `--dataset`, `--dry-run`, `--force` flags
  - [ ] Loop over models and datasets
  - [ ] Check for existing `metadata.json` to skip completed jobs
  - [ ] Submit jobs with sbatch

### 2.5 Validate Single Model + Single Dataset
- [ ] Test extraction locally on small subset (10 samples)
- [ ] Run `bash sbatch/extract_embeddings/run.sh --model birdnet --dataset HSN`
- [ ] Verify `.npz` files created with correct shapes
- [ ] Verify `metadata.json` has correct statistics
- [ ] Check GPU utilization and throughput

### 2.6 Scale to All Datasets (One Model)
- [ ] Run `bash sbatch/extract_embeddings/run.sh --model birdnet`
- [ ] Monitor jobs: `squeue -u $USER`
- [ ] Verify all 8 datasets completed
- [ ] Document any failures or issues

### 2.7 Scale to All Models
- [ ] Run for Perch: `run.sh --model perch`
- [ ] Run for BirdSetConvNeXT: `run.sh --model birdset_convnext`
- [ ] Run for BirdSetEfficientNetB1: `run.sh --model birdset_efficientnet_b1`
- [ ] Verify all 32 embedding jobs completed
- [ ] Total embeddings: 4 models × 8 datasets × 2 splits = 64 .npz files

## Phase 3: Classification Pipeline

### 3.1 Implement Training Script
- [ ] Create `src/birdset_benchmark/train_classifier.py`
  - [ ] CLI with typer: `--model`, `--dataset`, `--epochs`, `--lr`
  - [ ] Load embeddings from `artifacts/embeddings/{model}/{dataset}/`
  - [ ] Create PyTorch DataLoader from embeddings
  - [ ] Initialize `MultiLabelLinearProbe` from `models.py`
  - [ ] Initialize PyTorch Lightning Trainer with:
    - GPU accelerator
    - Early stopping callback (monitor val_loss)
    - Model checkpointing (save best)
    - TensorBoard logger
  - [ ] Train and save checkpoint to `artifacts/classifiers/`
  - [ ] Evaluate on test embeddings
  - [ ] Save metrics to `metrics.json`
  - [ ] Save predictions to `predictions.npz`

### 3.2 Create SLURM Job Scripts
- [ ] Create `sbatch/train_classifier/_job.sh`
  - [ ] SLURM configuration: 1 GPU, 4 CPUs, 32GB RAM, 2 hours
  - [ ] Run `python -m birdset_benchmark.train_classifier`

- [ ] Create `sbatch/train_classifier/run.sh`
  - [ ] Similar structure to extract_embeddings/run.sh
  - [ ] Check for existing `metrics.json` to skip completed jobs

### 3.3 Validate Single Model + Single Dataset
- [ ] Train classifier on BirdNET + HSN embeddings
- [ ] Verify checkpoint saved
- [ ] Verify metrics.json contains: cmAP, F1 micro/macro, Top-1/5
- [ ] Verify predictions.npz has correct shape

### 3.4 Scale to All Combinations
- [ ] Run `bash sbatch/train_classifier/run.sh`
- [ ] Monitor jobs and handle failures
- [ ] Verify all 32 classifier jobs completed
- [ ] Document training times and convergence

## Phase 4: Results Aggregation

### 4.1 Implement Aggregation Script
- [ ] Create `src/birdset_benchmark/aggregate_results.py`
  - [ ] Collect all `metrics.json` files
  - [ ] Generate `results/summary.json`
  - [ ] Generate `results/results_matrix.csv` (cmAP by model × dataset)
  - [ ] Calculate mean cmAP per model

### 4.2 Generate Visualizations
- [ ] Heatmap: cmAP by model × dataset
- [ ] Bar chart: Mean cmAP per model
- [ ] Bar chart: Per-dataset comparison (all models)
- [ ] Per-class AP analysis for best/worst classes
- [ ] Save to `artifacts/results/plots/`

### 4.3 Compare to Baselines
- [ ] Look up BirdSet paper results for EfficientNet, ConvNeXt
- [ ] Add baseline comparison to summary
- [ ] Note differences in methodology (linear probe vs full training)

### 4.4 Write Results Summary
- [ ] Create `results.md` with:
  - Summary table (all 32 combinations)
  - Key findings
  - Comparison to baselines
  - Per-model analysis
  - Per-dataset analysis
  - Limitations and caveats

## Phase 5: Cleanup and Documentation

### 5.1 Code Quality
- [ ] Run `ruff format` on all Python files
- [ ] Run `ruff check` and fix issues
- [ ] Add type hints to public functions
- [ ] Add docstrings to modules and classes

### 5.2 Update Documentation
- [ ] Update README with final results
- [ ] Update proposal.md with actual results
- [ ] Archive large artifacts (embeddings) to scratch
- [ ] Document artifact locations

### 5.3 Final Commit
- [ ] Commit results and documentation
- [ ] Tag release: `exp-001-v1.0`
- [ ] Push to repository

---

## Appendix: File Inventory

### Source Files to Create
```
src/birdset_benchmark/
├── __init__.py          (exists)
├── models.py            (exists - LinearProbe, metrics)
├── data.py              (new - dataset loading, preprocessing)
├── encoders.py          (new - model wrappers)
├── extract_embeddings.py (new - CLI script)
├── train_classifier.py   (new - CLI script)
├── aggregate_results.py  (new - results collection)
└── utils.py              (new - utilities)
```

### SLURM Scripts to Create
```
sbatch/
├── common.sh
├── extract_embeddings/
│   ├── run.sh
│   └── _job.sh
└── train_classifier/
    ├── run.sh
    └── _job.sh
```

### Artifact Structure
```
artifacts/
├── embeddings/{model}/{dataset}/
│   ├── train.npz
│   ├── test.npz
│   └── metadata.json
├── classifiers/{model}/{dataset}/
│   ├── model.ckpt
│   ├── metrics.json
│   └── predictions.npz
├── results/
│   ├── summary.json
│   ├── results_matrix.csv
│   └── plots/
└── logs/
```

---

## Quick Reference

**Run embedding extraction (single)**:
```bash
cd experiments/001-birdset-bioacoustics-benchmark
bash sbatch/extract_embeddings/run.sh --model birdnet --dataset HSN
```

**Run embedding extraction (all datasets, one model)**:
```bash
bash sbatch/extract_embeddings/run.sh --model birdnet
```

**Run classification (single)**:
```bash
bash sbatch/train_classifier/run.sh --model birdnet --dataset HSN
```

**Check job status**:
```bash
squeue -u $USER
```

**Dry run (preview jobs without submitting)**:
```bash
bash sbatch/extract_embeddings/run.sh --dry-run
```

---

**Estimated Timeline**:
- Phase 0: ✅ Complete
- Phase 1: ✅ Complete
- Phase 2: ~8-12 hours (implementation + GPU time)
- Phase 3: ~4-6 hours (implementation + GPU time)
- Phase 4: ~2-3 hours (aggregation + visualization)
- Phase 5: ~2-3 hours (cleanup + documentation)
- **Total Remaining**: ~16-24 hours + GPU queue time

**Dependencies**:
- BirdSet dataset availability on HuggingFace (verified)
- SLURM access with GPU allocation
- bioacoustics-model-zoo installation (verified)
