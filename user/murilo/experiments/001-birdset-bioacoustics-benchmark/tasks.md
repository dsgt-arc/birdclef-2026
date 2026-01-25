# Tasks: BirdSet Bioacoustics Model Benchmark

**Status**: Phase 1 complete, Phase 2A partially complete (2A.1-2A.2 done, 2A.3-2A.5 pending)
**Current Phase**: Phase 2A.3-2A.5 (SLURM scripts and production runs)
**Last Updated**: 2026-01-25

---

## Overview

This experiment evaluates pre-trained bioacoustics models on the BirdSet benchmark using two approaches:

**Part A - Direct Inference (No Training)**:
- BirdSetConvNeXT and BirdSetEfficientNetB1 (trained on BirdSet)
- Run inference directly, compute cmAP
- Reproduces/validates BirdSet paper results

**Part B - Transfer Learning (Requires Training or Mapping)**:
- BirdNET and Perch (external pre-trained models)
- Option 1: Linear probe on embeddings
- Option 2: Taxonomy mapping (BirdNET predictions → eBird codes)

**Models**: BirdNET, Perch, BirdSetConvNeXT, BirdSetEfficientNetB1
**Datasets**: HSN, PER, NES, UHH, POW, SSW, SNE, NBP
**Total Combinations**: 4 models × 8 datasets = 32

---

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

---

## Phase 2A: Direct Inference (BirdSet-Native Models) 🔄 IN PROGRESS

**Goal**: Run inference with BirdSetConvNeXT and BirdSetEfficientNetB1 directly on BirdSet test data. No training required since these models were trained on BirdSet.

### 2A.1 Implement Data Pipeline ✅ COMPLETED
- [x] Create `src/birdset_benchmark/data.py`
  - [x] Function: `load_birdset_dataset(dataset_code)` - Load any of 8 datasets
  - [x] Function: `resample_audio(audio, source_sr, target_sr)`
  - [x] Function: `pad_or_crop(audio, target_duration_sec, sr)` - supports pad/repeat modes
  - [x] Class: `BirdSetDataset` (PyTorch Dataset wrapper)
  - [x] Function: `collate_fn()` for batching
  - [x] Dataset-agnostic design with `BIRDSET_NUM_CLASSES` constant mapping
  - [x] Function: `get_num_classes(dataset_code)` for proper multi-hot encoding
  - [x] Multi-hot label encoding from `ebird_code_multilabel` indices
  - [x] Audio loading from file paths (BirdSet provides paths, not arrays)
  - [x] Unit tests for edge cases (16 tests passing in `test_data_pipeline.py`)

**Implementation Notes**:
- BirdSet provides audio as file paths in `sample["audio"]["path"]`, not arrays
- Uses `soundfile.read()` to load audio, then resamples and pads/crops
- Dataset-agnostic: constant class counts ensure correct multi-hot vector size even with subsets
- Comprehensive test coverage: resampling, padding, cropping, batching, metadata handling

### 2A.2 Implement Inference Script ✅ COMPLETED
- [x] Create `src/birdset_benchmark/run_inference.py`
  - [x] CLI with typer: `--model`, `--dataset`, `--batch-size`, `--output-dir`
  - [x] Support models: `birdset_convnext`, `birdset_efficientnet_b1`
  - [x] Load BirdSet test split
  - [x] Run model.predict() to get class predictions
  - [x] Compute metrics: cmAP, F1 micro/macro, Top-1/5
  - [x] Save results to `metrics.json`
  - [x] Save predictions to `predictions.npz`
  - [x] Simplified implementation using BMZ API correctly (model handles batching internally)
  - [x] Sigmoid activation to convert logits to probabilities
  - [x] Default output to `experiments/001-birdset-bioacoustics-benchmark/artifacts/`
  - [x] Unit tests for workflow (6 tests passing in `test_inference.py`)

**Implementation Notes**:
- Simplified design: single call to `model.predict(audio_paths, batch_size=batch_size)`
- Model handles all batching internally (no manual batching required)
- Predictions are logits, apply sigmoid: `y_pred = 1.0 / (1.0 + np.exp(-y_pred))`
- Species mapping from dataset features to align predictions with ground truth
- Test suite validates full workflow end-to-end on small subsets

**⚠️ Important**: Code is implemented and tested on small subsets (5-50 samples). Full dataset inference runs (Phase 2A.4-2A.5) have NOT been executed yet.

### 2A.3 Create SLURM Job Scripts
- [ ] Create `sbatch/common.sh`
  - [ ] Function: `setup_environment()` - venv in TMPDIR
  - [ ] Function: `get_artifact_dir()` - path construction
  - [ ] Environment variables: PROJECT_ROOT, etc.

- [ ] Create `sbatch/inference/_job.sh`
  - [ ] SLURM configuration: 1 GPU, 4 CPUs, 32GB RAM, 2 hours
  - [ ] Run `python -m birdset_benchmark.run_inference`
  - [ ] Use MODEL and DATASET environment variables

- [ ] Create `sbatch/inference/run.sh`
  - [ ] Parse `--model`, `--dataset`, `--dry-run` flags
  - [ ] Default: run both BirdSet-native models on all datasets
  - [ ] Check for existing `metrics.json` to skip completed jobs

### 2A.4 Validate Single Model + Single Dataset
- [ ] Test inference locally on small subset
- [ ] Run `bash sbatch/inference/run.sh --model birdset_convnext --dataset HSN`
- [ ] Verify metrics.json contains: cmAP, F1, Top-1/5
- [ ] Compare to expected BirdSet paper results

### 2A.5 Scale to All Datasets
- [ ] Run `bash sbatch/inference/run.sh --model birdset_convnext`
- [ ] Run `bash sbatch/inference/run.sh --model birdset_efficientnet_b1`
- [ ] Monitor jobs and verify completion
- [ ] Total: 2 models × 8 datasets = 16 inference runs

---

## Phase 2B: Transfer Learning (External Models)

**Goal**: Evaluate BirdNET and Perch on BirdSet. Two options:
1. **Linear probe**: Extract embeddings, train classifier on BirdSet labels
2. **Taxonomy mapping**: Map BirdNET's native predictions to eBird codes

### 2B.1 Research Taxonomy Mapping (Option 2)
- [ ] Investigate if BirdNET → eBird code mapping exists
  - [ ] Check BirdNET documentation for taxonomy
  - [ ] Check eBird API for species code mappings
  - [ ] Determine overlap between BirdNET species and BirdSet species
- [ ] If mapping exists:
  - [ ] Implement `src/birdset_benchmark/taxonomy_mapping.py`
  - [ ] Create mapping table (BirdNET class → eBird code)
  - [ ] Handle unmapped species (assign to "unknown" or drop)
- [ ] Decision: Use taxonomy mapping OR fall back to linear probe

### 2B.2 Implement Embedding Extraction (for Linear Probe)
- [ ] Create `src/birdset_benchmark/encoders.py`
  - [ ] Class: `BioacousticsEncoder` (abstract base class)
  - [ ] Class: `BirdNETEncoder` (wrapper for bmz.BirdNET)
  - [ ] Class: `PerchEncoder` (wrapper for bmz.Perch)
  - [ ] Each encoder: `.embed(audio_batch)` → `[batch, embedding_dim]`
  - [ ] Handle sample rate differences (48kHz for BirdNET, 32kHz for Perch)

- [ ] Create `src/birdset_benchmark/extract_embeddings.py`
  - [ ] CLI with typer: `--model`, `--dataset`, `--batch-size`, `--output-dir`
  - [ ] Support models: `birdnet`, `perch`
  - [ ] Extract embeddings for train and test splits
  - [ ] Save to `.npz` format with labels
  - [ ] Save metadata to `metadata.json`

- [ ] Create SLURM job scripts for embedding extraction
  - [ ] `sbatch/extract_embeddings/_job.sh`
  - [ ] `sbatch/extract_embeddings/run.sh`

### 2B.3 Implement Linear Probe Training
- [ ] Create `src/birdset_benchmark/train_classifier.py`
  - [ ] CLI with typer: `--model`, `--dataset`, `--epochs`, `--lr`
  - [ ] Load embeddings from `artifacts/embeddings/{model}/{dataset}/`
  - [ ] Train `MultiLabelLinearProbe` with PyTorch Lightning
  - [ ] Evaluate and save metrics to `metrics.json`
  - [ ] Save predictions to `predictions.npz`

- [ ] Create SLURM job scripts for training
  - [ ] `sbatch/train_classifier/_job.sh`
  - [ ] `sbatch/train_classifier/run.sh`

### 2B.4 Run External Model Evaluation
- [ ] If using taxonomy mapping:
  - [ ] Run direct inference with BirdNET/Perch
  - [ ] Apply taxonomy mapping to predictions
  - [ ] Compute cmAP on mapped predictions
- [ ] If using linear probe:
  - [ ] Extract embeddings for all datasets
  - [ ] Train classifiers on embeddings
  - [ ] Evaluate and save results
- [ ] Total: 2 models × 8 datasets = 16 evaluations

---

## Phase 3: Results Aggregation

### 3.1 Implement Aggregation Script
- [ ] Create `src/birdset_benchmark/aggregate_results.py`
  - [ ] Collect all `metrics.json` files
  - [ ] Generate `results/summary.json`
  - [ ] Generate `results/results_matrix.csv` (cmAP by model × dataset)
  - [ ] Calculate mean cmAP per model
  - [ ] Note which models used direct inference vs linear probe

### 3.2 Generate Visualizations
- [ ] Heatmap: cmAP by model × dataset
- [ ] Bar chart: Mean cmAP per model
- [ ] Bar chart: Per-dataset comparison (all models)
- [ ] Separate colors for direct inference vs linear probe results
- [ ] Save to `artifacts/results/plots/`

### 3.3 Compare to Baselines
- [ ] Look up BirdSet paper results
- [ ] Compare our BirdSet-native model results to paper
- [ ] Note differences in methodology
- [ ] Document any discrepancies

### 3.4 Write Results Summary
- [ ] Create `results.md` with:
  - Summary table (all 32 combinations)
  - Key findings
  - Comparison to BirdSet paper baselines
  - Analysis: direct inference vs linear probe
  - Per-model and per-dataset analysis
  - Limitations and caveats

---

## Phase 4: Cleanup and Documentation

### 4.1 Code Quality
- [ ] Run `ruff format` on all Python files
- [ ] Run `ruff check` and fix issues
- [ ] Add type hints to public functions
- [ ] Add docstrings to modules and classes

### 4.2 Update Documentation
- [ ] Update README with final results
- [ ] Update proposal.md with actual results
- [ ] Archive large artifacts (embeddings) to scratch
- [ ] Document artifact locations

### 4.3 Final Commit
- [ ] Commit results and documentation
- [ ] Tag release: `exp-001-v1.0`
- [ ] Push to repository

---

## Appendix: File Inventory

### Source Files to Create
```
src/birdset_benchmark/
├── __init__.py              ✅ exists
├── models.py                ✅ exists - LinearProbe, metrics
├── data.py                  ✅ exists - dataset loading, preprocessing (356 lines)
├── run_inference.py         ✅ exists - direct inference for BirdSet models (239 lines)
├── encoders.py              (new - embedding extractors for BirdNET/Perch)
├── extract_embeddings.py    (new - CLI for embedding extraction)
├── train_classifier.py      (new - CLI for linear probe training)
├── taxonomy_mapping.py      (new - BirdNET → eBird mapping, if viable)
├── aggregate_results.py     (new - results collection)
└── utils.py                 (new - utilities)
```

### SLURM Scripts to Create
```
sbatch/
├── common.sh
├── inference/               # Phase 2A: Direct inference
│   ├── run.sh
│   └── _job.sh
├── extract_embeddings/      # Phase 2B: Embedding extraction
│   ├── run.sh
│   └── _job.sh
└── train_classifier/        # Phase 2B: Linear probe training
    ├── run.sh
    └── _job.sh
```

### Artifact Structure
```
artifacts/
├── inference/{model}/{dataset}/        # Phase 2A results
│   ├── metrics.json
│   └── predictions.npz
├── embeddings/{model}/{dataset}/       # Phase 2B embeddings
│   ├── train.npz
│   ├── test.npz
│   └── metadata.json
├── classifiers/{model}/{dataset}/      # Phase 2B classifiers
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

**Run direct inference (BirdSet-native models)**:
```bash
cd experiments/001-birdset-bioacoustics-benchmark
bash sbatch/inference/run.sh --model birdset_convnext --dataset HSN
```

**Run embedding extraction (external models)**:
```bash
bash sbatch/extract_embeddings/run.sh --model birdnet --dataset HSN
```

**Run linear probe training**:
```bash
bash sbatch/train_classifier/run.sh --model birdnet --dataset HSN
```

**Dry run (preview jobs)**:
```bash
bash sbatch/inference/run.sh --dry-run
```

---

## Estimated Timeline

| Phase | Description | Time Estimate |
|-------|-------------|---------------|
| Phase 0 | Planning | ✅ Complete |
| Phase 1 | Environment Setup | ✅ Complete |
| Phase 2A.1-2A.2 | Data Pipeline & Inference Script | ✅ Complete |
| Phase 2A.3-2A.5 | SLURM Scripts & Production Runs | ~2-3 hours + GPU queue |
| Phase 2B | Transfer Learning (BirdNET/Perch) | ~8-12 hours |
| Phase 3 | Results Aggregation | ~2-3 hours |
| Phase 4 | Cleanup & Documentation | ~2-3 hours |
| **Total Remaining** | | ~14-20 hours + GPU queue |

**Dependencies**:
- BirdSet dataset availability on HuggingFace (verified)
- SLURM access with GPU allocation
- bioacoustics-model-zoo installation (verified)
