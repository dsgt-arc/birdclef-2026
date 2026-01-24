# Tasks: BirdSet Bioacoustics Model Benchmark

**Status**: Phase 0 complete (Planning & Documentation) ✅
**Current Phase**: Phase 1 (Environment Setup) 🔄
**Last Updated**: 2026-01-24

---

## Phase 0: Planning and Documentation ✅ COMPLETED

- [x] 0.1 Create experiment proposal
  - [x] Write proposal.md with hypothesis, methodology, evaluation criteria
  - [x] Define success metrics (cmAP, F1 micro/macro, Top-1/5)
  - [x] Identify all 7 models to benchmark
  - [x] Document limitations and future work

- [x] 0.2 Create technical design document
  - [x] Write design.md with technical decisions
  - [x] Define two-stage pipeline architecture
  - [x] Decide on max-pooling strategy for variable-length audio
  - [x] Choose linear probe implementation (scikit-learn)
  - [x] Document data flow and code structure

- [x] 0.3 Create implementation task list
  - [x] Break down work into 7 phases
  - [x] Define ~40 detailed subtasks
  - [x] Estimate timeline (16-24 hours)

- [x] 0.4 Create configuration and documentation
  - [x] Write config.yaml with all model specs and SLURM settings
  - [x] Create pyproject.toml with all dependencies
  - [x] Write README.md with quick start guide
  - [x] Create test_setup.sh for interactive environment testing

- [x] 0.5 Commit and push to repository
  - [x] Add all files to git
  - [x] Commit with descriptive message
  - [x] Push to origin/mg-birdset branch

## Phase 1: Environment Setup

- [ ] 1.1 Create Python package structure
  - [x] `user/murilo/pyproject.toml` with all dependencies
  - [ ] `user/murilo/src/birdset_benchmark/__init__.py`
  - [ ] Add to main project `pyproject.toml` as optional dependency (if needed)
  - [ ] Test `uv` environment setup via `test_setup.sh`

- [ ] 1.2 Install and test bioacoustics-model-zoo
  - [ ] `uv pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo.git`
  - [ ] Test `import bmz; bmz.list_models()` works
  - [ ] Verify BirdNET and Perch can be instantiated

- [ ] 1.3 Test BirdSet dataset loading
  - [ ] Load HSN subset from `DBD-research-group/BirdSet`
  - [ ] Verify train and test_5s splits exist
  - [ ] Inspect sample: check audio format, labels, metadata
  - [ ] Document dataset statistics (num samples, species, duration)

- [ ] 1.4 Create experiment directory structure
  - [x] `user/murilo/experiments/001-birdset-bioacoustics-benchmark/` (base directory)
  - [x] `config.yaml` with all model specs and SLURM settings
  - [x] `test_setup.sh` for interactive environment testing
  - [ ] `sbatch/` directory for SLURM job scripts
  - [ ] `artifacts/` directory for results and outputs

## Phase 2: Data Pipeline Implementation

- [ ] 2.1 Implement audio preprocessing (`data.py`)
  - [ ] Function: `resample_audio(audio, source_sr, target_sr)`
  - [ ] Function: `pad_or_crop(audio, target_duration_sec, sr)`
  - [ ] Function: `max_pool_embeddings(embeddings_sequence)` for long clips
  - [ ] Unit tests for edge cases (very short, very long, exact 5s)

- [ ] 2.2 Create BirdSet dataset wrapper
  - [ ] Class: `BirdSetDataset` (PyTorch Dataset)
  - [ ] Lazy loading (don't load all audio into memory)
  - [ ] Resample on-the-fly to target sample rate
  - [ ] Return: `{'audio': tensor, 'labels': multi_hot, 'id': str}`
  - [ ] Handle FLAC/OGG decoding correctly

- [ ] 2.3 Implement batching with collate function
  - [ ] `collate_fn()` to pad batch to max length
  - [ ] Test DataLoader with batch_size=16
  - [ ] Profile: samples/sec on CPU vs GPU

- [ ] 2.4 Create multi-label encoding utilities
  - [ ] Function: `encode_multilabel(ebird_codes, all_classes)` → one-hot
  - [ ] Function: `decode_predictions(logits, threshold=0.5)` → species list
  - [ ] Save label mapping to JSON for reproducibility

## Phase 3: Model Wrappers

- [ ] 3.1 Create unified model interface (`models.py`)
  - [ ] Abstract class: `BioacousticsEncoder` with `.extract_embeddings(audio_batch)`
  - [ ] Wrapper: `BirdNETEncoder` (handles 48kHz resampling)
  - [ ] Wrapper: `PerchEncoder` (handles 32kHz resampling)
  - [ ] Each wrapper returns numpy array [batch, embedding_dim]

- [ ] 3.2 Test model wrappers
  - [ ] Load BirdNET, extract embeddings for 1 sample → verify shape (1024,)
  - [ ] Load Perch, extract embeddings for 1 sample → verify shape (1280,)
  - [ ] Test batched extraction (batch_size=16) → verify no errors
  - [ ] Measure throughput: samples/sec on GPU

- [ ] 3.3 Handle edge cases
  - [ ] Empty audio (all zeros) → return zero embeddings
  - [ ] Corrupted audio → catch exceptions, log warning
  - [ ] Very long audio (>60s) → chunk and max-pool

## Phase 4: Embedding Extraction (Stage 1 - GPU)

- [ ] 4.1 Implement extraction script (`extract_embeddings.py`)
  - [ ] CLI arguments: `--model {birdnet,perch}`, `--split {train,test}`, `--output-dir`
  - [ ] Load BirdSet dataset for specified split
  - [ ] Initialize model wrapper
  - [ ] Loop over DataLoader with tqdm progress bar
  - [ ] Accumulate embeddings + labels in lists
  - [ ] Save to `.npz` with keys: `embeddings`, `labels`, `ids`

- [ ] 4.2 Add metadata logging
  - [ ] Log: model name, sample rate, embedding dim, num samples
  - [ ] Log: extraction start/end time, throughput (samples/sec)
  - [ ] Save metadata to `artifacts/metadata.json`

- [ ] 4.3 Create SLURM batch script (`sbatch/extract_embeddings.sbatch`)
  - [ ] Request: 1 GPU (L40S preferred), 4 CPU cores, 32GB RAM
  - [ ] Time limit: 12 hours
  - [ ] Load environment: `module load anaconda3; source activate birdclef-env`
  - [ ] Run extraction for all combinations: {birdnet, perch} × {train, test}
  - [ ] Redirect output to logs directory

- [ ] 4.4 Dry run and validation
  - [ ] Run extraction on 10 samples (subset) to test pipeline
  - [ ] Verify `.npz` files created with correct shapes
  - [ ] Check GPU utilization (should be >50%)
  - [ ] Submit full GPU job to SLURM

## Phase 5: Linear Probe Training (Stage 2 - CPU)

- [ ] 5.1 Implement training script (`train_probe.py`)
  - [ ] CLI arguments: `--model {birdnet,perch}`, `--embeddings-dir`, `--output-dir`
  - [ ] Load embeddings: `embeddings_train_*.npz`, `embeddings_test_*.npz`
  - [ ] Initialize: `OneVsRestClassifier(LogisticRegression(...))`
  - [ ] Train on train embeddings
  - [ ] Save trained classifier to `artifacts/classifiers/{model}_probe.pkl`

- [ ] 5.2 Add optional preprocessing
  - [ ] Optional: L2-normalize embeddings (`normalize(X, norm='l2')`)
  - [ ] Log: whether normalization was used
  - [ ] Compare results with/without normalization

- [ ] 5.3 Implement evaluation (`evaluate.py`)
  - [ ] Function: `compute_cmap(y_true, y_pred_proba)`
  - [ ] Function: `compute_f1_scores(y_true, y_pred_binary)` → (f1_micro, f1_macro)
  - [ ] Function: `compute_top_k_accuracy(y_true, y_pred_proba, k=5)`
  - [ ] Function: `per_class_average_precision(y_true, y_pred_proba, class_names)`
  - [ ] Generate predictions on test set
  - [ ] Compute all metrics (cmAP, F1 micro/macro, Top-1/5)

- [ ] 5.4 Create SLURM batch script (`sbatch/train_probe.sbatch`)
  - [ ] Request: 1 GPU (gpu-rtx6000), 4 cores, 32GB RAM
  - [ ] Time limit: 2 hours
  - [ ] Train probes for all 7 models
  - [ ] Run evaluation and save results

- [ ] 5.5 Dry run and validation
  - [ ] Train on subset of embeddings (1000 samples)
  - [ ] Verify classifier trains without errors
  - [ ] Check convergence (should be <1000 iterations)
  - [ ] Submit full GPU job to SLURM

## Phase 6: Results Analysis

- [ ] 6.1 Generate results summary
  - [ ] Create `results_summary.json`:
    ```json
    {
      "birdnet": {
        "cmap": 0.XX,
        "f1_micro": 0.XX,
        "f1_macro": 0.XX,
        "top1": 0.XX,
        "top5": 0.XX
      },
      "perch": {
        "cmap": 0.XX,
        "f1_micro": 0.XX,
        "f1_macro": 0.XX,
        "top1": 0.XX,
        "top5": 0.XX
      }
    }
    ```
  - [ ] Create `per_class_ap.csv` with columns: `[species, birdnet_ap, perch_ap]`
  - [ ] Log extraction and training times

- [ ] 6.2 Generate visualizations (`plots/`)
  - [ ] Bar chart: cmAP comparison (BirdNET vs Perch)
  - [ ] Scatter plot: per-class AP (BirdNET vs Perch) - identify which species benefit from which model
  - [ ] Histogram: distribution of per-class APs
  - [ ] Top-10 and Bottom-10 species by AP (identify failure cases)

- [ ] 6.3 Confusion analysis (optional, if time permits)
  - [ ] For top-5 misclassified species, create confusion matrix
  - [ ] Identify common confusions (e.g., similar vocalizations)

- [ ] 6.4 Write results.md
  - [ ] Summary table with all metrics
  - [ ] Comparison to baseline expectations
  - [ ] Key observations (qualitative)
  - [ ] Failure analysis
  - [ ] Recommendations for next steps

## Phase 7: Cleanup and Documentation

- [ ] 7.1 Code quality
  - [ ] Run `ruff format` on all Python files
  - [ ] Run `ruff check` and fix issues
  - [ ] Add type hints to public functions
  - [ ] Add docstrings to modules and classes

- [ ] 7.2 Update README with final results
  - [x] Initial README created with quick start guide
  - [ ] Update with final reproduction commands
  - [ ] Link to artifacts and results
  - [ ] Note any deviations from proposal

- [ ] 7.3 Archive artifacts
  - [ ] Move large files to scratch if needed
  - [ ] Keep only essential files in git (metadata, configs)
  - [ ] Document artifact locations in README

- [ ] 7.4 Update proposal status
  - [ ] Change status from "proposed" to "completed"
  - [ ] Update dates in proposal.md

## Optional Extensions (Out of Scope for v1)

- [ ] 8.1 Expand to all BirdSet datasets (PER, NES, UHH, POW, SSW, SNE, NBP)
- [ ] 8.2 Test different audio preprocessing strategies (random crop vs max-pool)
- [ ] 8.3 Hyperparameter tuning for LogisticRegression (C parameter grid search)
- [ ] 8.4 Compare with other models from bioacoustics-model-zoo (if available)
- [ ] 8.5 Ensemble predictions from BirdNET + Perch
- [ ] 8.6 Fine-tune models end-to-end (not just linear probe)

---

**Estimated Timeline**:
- Phase 0: ✅ Complete (2-3 hours - planning and documentation)
- Phase 1-2: 4-6 hours (environment setup, data pipeline)
- Phase 3-4: 6-8 hours (model wrappers, embedding extraction)
- Phase 5: 2-4 hours (linear probe training)
- Phase 6-7: 4-6 hours (analysis, documentation)
- **Remaining**: 16-24 hours of development + GPU job time (~14 hours)

**Dependencies**:
- BirdSet dataset availability on HuggingFace
- SLURM access with GPU allocation
- bioacoustics-model-zoo installation working correctly
