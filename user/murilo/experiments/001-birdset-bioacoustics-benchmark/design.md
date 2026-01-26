# Design: BirdSet Bioacoustics Model Benchmark

## Context

We are evaluating pre-trained bioacoustics models (BirdNET, Perch, BirdSetConvNeXT, BirdSetEfficientNetB1) on the full BirdSet benchmark (8 datasets) using linear probing. This complements the original BirdSet paper by testing external pre-trained models alongside BirdSet-native models.

The challenge is creating a scalable pipeline that can handle 4 models × 8 datasets = 32 combinations efficiently on the PACE cluster.

**Update (2026-01-26)**: Phase 2A (direct inference for BirdSet-native models) is complete. All 16 inference runs finished with cmAP, AUROC, and T1-Acc metrics. Results documented in RESULTS_SUMMARY.md. Ready for Phase 2B (transfer learning with external models).

## Goals / Non-Goals

**Goals:**
- Extract embeddings from 4 models across all 8 BirdSet datasets
- Train reproducible linear probes that isolate embedding quality
- Produce cmAP scores comparable to BirdSet published baselines
- Create reusable SLURM job scripts for scalable execution
- Generate comprehensive results matrix (model × dataset)

**Non-Goals:**
- Fine-tuning models end-to-end (linear probe only)
- Zero-shot classification (taxonomies don't align)
- Real-time inference optimization (batch processing acceptable)
- Hyperparameter tuning for linear probe (use sensible defaults)

## Technical Decisions

### Decision 1: Modular SLURM Job Architecture

**Choice**: Use `run.sh` controller + `_job.sh` worker pattern (similar to Fast-dLLM repo).

**Rationale**:
- Separates job orchestration from execution logic
- Supports parallel submission of independent jobs
- Tracks completion via artifact files
- Enables dry-run and force-rerun options
- Familiar pattern from existing workflow

**Implementation**:
```
sbatch/
├── common.sh                    # Shared functions (venv setup, paths)
├── inference/                   # Phase 2A: Direct inference (BirdSet-native)
│   ├── run.sh                   # Controller: submit jobs for model × dataset
│   └── _job.sh                  # Worker: run inference for one combination
├── extract_embeddings/          # Phase 2B: Embedding extraction (external)
│   ├── run.sh                   # Controller: submit jobs for model × dataset
│   └── _job.sh                  # Worker: extract embeddings for one combination
└── train_classifier/            # Phase 2B: Linear probe training (external)
    ├── run.sh                   # Controller: submit jobs for model × dataset
    └── _job.sh                  # Worker: train classifier for one combination
```

### Decision 2: Two-Part Evaluation Strategy

**Choice**: Split evaluation into Part A (direct inference) and Part B (transfer learning).

**Part A - Direct Inference (BirdSet-Native Models)**:
- BirdSetConvNeXT and BirdSetEfficientNetB1 output BirdSet class predictions directly
- No training required - just run inference and compute metrics
- Reproduces BirdSet paper methodology

**Part B - Transfer Learning (External Models)**:
- BirdNET and Perch use different taxonomies than BirdSet
- Two options:
  1. **Taxonomy mapping**: Map model predictions to eBird codes (if mapping exists)
  2. **Linear probe**: Extract embeddings, train classifier on BirdSet labels

**Rationale**:
- Avoids unnecessary training for models that already output BirdSet classes
- Provides flexibility for external models with taxonomy mismatch
- Allows direct comparison of different evaluation approaches

**Alternatives Considered**:
- **Linear probe for all models**: Would work but wastes effort for BirdSet-native models
- **Only direct inference**: Wouldn't work for BirdNET/Perch due to taxonomy mismatch

### Decision 3: CLI Interface with Flags

**Choice**: Use `typer` CLI with `--model` and `--dataset` flags.

**Scripts**:
```bash
# Extract embeddings for one model + one dataset
python -m birdset_benchmark.extract_embeddings --model birdnet --dataset HSN

# Train classifier on extracted embeddings
python -m birdset_benchmark.train_classifier --model birdnet --dataset HSN
```

**Default Behavior**:
- If no `--model` specified: use BirdNET
- If no `--dataset` specified: run on all 8 datasets (separate jobs)

### Decision 4: Vectorized Prediction Conversion

**Choice**: Use pandas `.reindex()` for O(n) prediction alignment instead of nested loops.

**Problem Encountered (2026-01-25)**:
- Original implementation used nested loops to match file paths: O(n²m) complexity
- For SSW: 205,200 samples × 81 species = ~42 billion operations
- Inference completed in 2.3 hours, but conversion phase hung for 1+ hours

**Solution**:
```python
# Old approach: O(n²m)
for i, audio_path in enumerate(audio_paths):  # 205K iterations
    matching_rows = pred_df.loc[pred_df.index == audio_path]  # 205K comparisons each
    ...

# New approach: O(n)
pred_df_indexed.index = pred_df.index.get_level_values('file')
pred_df_aligned = pred_df_indexed.reindex(audio_paths, fill_value=0.0)  # Single vectorized op
for species_code in pred_df_aligned.columns:  # Only 81 iterations
    y_pred[:, class_idx] = pred_df_aligned[species_code].values  # Vectorized assignment
```

**Impact**:
- ~210,000x speedup for SSW
- Conversion phase: hours → seconds
- Comprehensive test suite validates correctness across edge cases

**Rationale**: Pandas is optimized for vectorized operations on indexed data. Using `.reindex()` leverages hash-based lookups instead of linear scans.

### Decision 5: Artifact Organization

**Choice**: Organize artifacts by model and dataset with completion markers.

**Structure**:
```
artifacts/
├── embeddings/
│   ├── birdnet/
│   │   ├── HSN/
│   │   │   ├── train.npz          # Embeddings + labels
│   │   │   ├── test.npz
│   │   │   └── metadata.json      # Extraction stats
│   │   ├── PER/
│   │   └── ...
│   ├── perch/
│   ├── birdset_convnext/
│   └── birdset_efficientnet_b1/
├── classifiers/
│   ├── birdnet/
│   │   ├── HSN/
│   │   │   ├── model.ckpt         # PyTorch Lightning checkpoint
│   │   │   ├── metrics.json       # cmAP, F1, Top-K
│   │   │   └── predictions.npz    # Test predictions
│   │   └── ...
│   └── ...
├── results/
│   ├── summary.json               # All results aggregated
│   ├── results_matrix.csv         # cmAP by model × dataset
│   └── plots/
│       ├── heatmap.png
│       └── per_dataset_comparison.png
└── logs/
    └── slurm/                     # SLURM job logs
```

**Completion Tracking**:
- `run.sh` checks for `metrics.json` to determine if job completed
- Enables skip-if-done and force-rerun semantics

### Decision 5: Audio Preprocessing Strategy

**Choice**: Use **max-pooling over time** for train samples longer than 5s.

**Rationale**:
- BirdSet train samples are 1-60s (variable); test is 5s (fixed)
- Max-pooling preserves most salient features
- Single embedding per sample simplifies linear probe
- Translation-invariant (bird call can occur anywhere)

**Implementation**:
```python
def preprocess_audio(audio, target_duration=5.0, sr=48000):
    target_samples = int(target_duration * sr)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        embeddings = model.extract_windows(audio, window_size=target_samples)
        return np.max(embeddings, axis=0)
    return model.extract(audio)
```

### Decision 6: Linear Probe Implementation

**Choice**: Use PyTorch Lightning `MultiLabelLinearProbe` (2-layer MLP).

**Rationale**:
- GPU acceleration for faster training
- Built-in logging, checkpointing, early stopping
- Consistent with birdclef-2025 codebase
- Already implemented and tested in Phase 1

**Architecture**:
```python
class MultiLabelLinearProbe(pl.LightningModule):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
```

### Decision 7: Embedding Storage Format

**Choice**: Save as `.npz` (NumPy compressed) with metadata JSON.

**File Contents**:
```python
# train.npz
{
    'embeddings': np.array([N, D]),    # Embedding vectors
    'labels': np.array([N, C]),        # Multi-hot labels
    'ids': np.array([N]),              # Sample identifiers
}

# metadata.json
{
    'model': 'birdnet',
    'dataset': 'HSN',
    'embedding_dim': 1024,
    'num_train_samples': 5460,
    'num_test_samples': 12000,
    'extraction_time_sec': 1234.5,
    'samples_per_sec': 4.4,
    'timestamp': '2026-01-24T12:00:00Z'
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│            PHASE 2A: DIRECT INFERENCE (BirdSet-Native)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Models: BirdSetConvNeXT, BirdSetEfficientNetB1                │
│  (These models were trained on BirdSet - no training needed)   │
│                                                                 │
│  run.sh --model birdset_convnext                               │
│      │                                                          │
│      ├── sbatch _job.sh (MODEL=birdset_convnext, DATASET=HSN)  │
│      ├── sbatch _job.sh (MODEL=birdset_convnext, DATASET=PER)  │
│      └── ... (8 parallel GPU jobs)                             │
│                                                                 │
│  Each _job.sh:                                                  │
│      BirdSet(test) → Model.predict() → metrics.json            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            PHASE 2B: TRANSFER LEARNING (External Models)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Models: BirdNET, Perch                                        │
│  (Different taxonomy - need mapping or linear probe)           │
│                                                                 │
│  Option 1: Taxonomy Mapping (if BirdNET→eBird mapping exists)  │
│      BirdSet(test) → Model.predict() → Map → metrics.json      │
│                                                                 │
│  Option 2: Linear Probe (if mapping not viable)                │
│      Step 1: Extract embeddings                                │
│          BirdSet(train+test) → Model.embed() → .npz            │
│      Step 2: Train classifier                                  │
│          Load .npz → Train LinearProbe → metrics.json          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 3: AGGREGATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  python aggregate_results.py                                    │
│      │                                                          │
│      ├── Collect all metrics.json files                        │
│      ├── Generate results_matrix.csv                           │
│      ├── Create visualizations                                 │
│      ├── Compare direct inference vs linear probe              │
│      └── Compare to BirdSet paper baselines                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Code Structure

```
user/murilo/
├── src/birdset_benchmark/
│   ├── __init__.py
│   ├── data.py                    # BirdSet loading, preprocessing
│   ├── models.py                  # LinearProbe, metrics (existing)
│   ├── run_inference.py           # CLI: Direct inference for BirdSet-native models
│   ├── encoders.py                # Embedding extractors for BirdNET/Perch
│   ├── extract_embeddings.py      # CLI: Embedding extraction for external models
│   ├── train_classifier.py        # CLI: Linear probe training
│   ├── taxonomy_mapping.py        # BirdNET → eBird code mapping (if viable)
│   ├── aggregate_results.py       # Collect and visualize results
│   └── utils.py                   # Audio resampling, batching
│
├── experiments/001-birdset-bioacoustics-benchmark/
│   ├── proposal.md
│   ├── design.md                  # This file
│   ├── tasks.md
│   ├── config.yaml                # Model configs, paths
│   ├── sbatch/
│   │   ├── common.sh              # Shared setup (venv, modules)
│   │   ├── inference/             # Phase 2A: Direct inference
│   │   │   ├── run.sh
│   │   │   └── _job.sh
│   │   ├── extract_embeddings/    # Phase 2B: Embedding extraction
│   │   │   ├── run.sh
│   │   │   └── _job.sh
│   │   └── train_classifier/      # Phase 2B: Linear probe training
│   │       ├── run.sh
│   │       └── _job.sh
│   └── artifacts/                 # Output directory (gitignored)
│
└── tests/001-birdset-bioacoustics-benchmark/
    ├── test_environment.py
    ├── test_models.py
    ├── test_classifiers.py
    ├── test_datasets.py
    └── test_pipeline.py           # Integration tests (new)
```

## SLURM Job Scripts

### `sbatch/common.sh` (Shared Setup)
```bash
#!/bin/bash
# Shared functions for all SLURM jobs

setup_environment() {
    # Create venv in TMPDIR for fast I/O
    export JOB_VENV="$TMPDIR/birdset-benchmark-$SLURM_JOB_ID/.venv"

    # Install dependencies
    uv venv "$JOB_VENV" --python 3.11
    source "$JOB_VENV/bin/activate"
    uv pip install -e "$PROJECT_ROOT/user/murilo"
}

get_artifact_dir() {
    local model=$1
    local dataset=$2
    local stage=$3  # embeddings or classifiers
    echo "$PROJECT_ROOT/user/murilo/experiments/001-birdset-bioacoustics-benchmark/artifacts/$stage/$model/$dataset"
}
```

### `sbatch/inference/run.sh` (Phase 2A)
```bash
#!/bin/bash
# Submit direct inference jobs for BirdSet-native models

MODELS=(birdset_convnext birdset_efficientnet_b1)
DATASETS=(HSN PER NES UHH POW SSW SNE NBP)

# Parse flags
MODEL_FILTER=""
DATASET_FILTER=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_FILTER="$2"; shift 2 ;;
        --dataset) DATASET_FILTER="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Submit jobs
for model in "${MODELS[@]}"; do
    [[ -n "$MODEL_FILTER" && "$model" != "$MODEL_FILTER" ]] && continue

    for dataset in "${DATASETS[@]}"; do
        [[ -n "$DATASET_FILTER" && "$dataset" != "$DATASET_FILTER" ]] && continue

        # Check if already completed
        artifact_dir="artifacts/inference/$model/$dataset"
        if [[ -f "$artifact_dir/metrics.json" ]]; then
            echo "SKIP: $model/$dataset (already completed)"
            continue
        fi

        echo "SUBMIT: $model/$dataset"
        if [[ "$DRY_RUN" == "false" ]]; then
            sbatch \
                --job-name="infer-$model-$dataset" \
                --output="artifacts/logs/infer-$model-$dataset-%j.log" \
                --export=ALL,MODEL="$model",DATASET="$dataset" \
                _job.sh
        fi
    done
done
```

## Environment Setup

**Dependencies** (in `pyproject.toml`):
```toml
dependencies = [
    "torch>=2.0.0",
    "pytorch-lightning>=2.0.0",
    "torchmetrics>=1.0.0",
    "tensorflow==2.17",
    "tensorflow-hub",
    "datasets>=2.14.0",
    "bioacoustics-model-zoo",
    "opensoundscape>=0.10.0",
    "ai-edge-litert>=1.0.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "polars>=0.19.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]
```

## Implementation Findings (Phase 2A)

### Decision 8: Data Module Organization

**Choice**: Split data module into separate files by purpose.

**Structure**:
```
src/birdset_benchmark/
├── datasets.py      # HuggingFace loading, constants (Phase 2A & 2B)
├── torch_data.py    # PyTorch Dataset/DataLoader (Phase 2B only)
└── data.py          # Backwards compatibility re-exports
```

**Rationale**:
- Phase 2A (Direct Inference) only needs HuggingFace dataset loading
- Phase 2B (Transfer Learning) needs PyTorch DataLoader for embedding extraction and linear probe training
- Clear separation makes it obvious which components are used when

### Decision 9: BMZ API Constraints and GPU Utilization

**Finding**: The bioacoustics-model-zoo `.predict()` API is **I/O bound**, not GPU bound.

**Technical Analysis**:
```
BMZ Model.predict(file_paths) Internal Flow:
┌─────────────────────────────────────────────────────────────┐
│  1. SafeAudioDataloader validates file paths exist          │
│  2. AudioFileDataset loads audio from disk (I/O bottleneck) │
│  3. BirdsetPreprocessor creates spectrograms (CPU)          │
│  4. Batch transferred to GPU                                │
│  5. Forward pass (GPU - fast)                               │
│  6. Results detached back to CPU                            │
│  7. Repeat for each batch                                   │
└─────────────────────────────────────────────────────────────┘
```

**Key Constraints**:
| Constraint | Implication |
|------------|-------------|
| File paths only | Cannot pass pre-loaded tensors to `.predict()` |
| CPU preprocessing | Spectrogram computation happens before GPU |
| Sequential I/O | Audio loaded during DataLoader iteration |
| No bypass possible | OpenSoundscape's SafeAudioDataloader enforces this |

**Mitigation**:
- Use `num_workers` parameter (default 4) to parallelize CPU preprocessing
- Increase `batch_size` to amortize CPU-GPU transfer overhead
- Accept that this is expected behavior for production audio ML pipelines

**Why BirdSetDataset is NOT used for Phase 2A**:
- BMZ models have their own internal DataLoader that only accepts file paths
- We cannot inject our own PyTorch Dataset into the BMZ prediction pipeline
- BirdSetDataset will be used for Phase 2B (embedding extraction, linear probe training)

### Decision 10: Warning Suppression Strategy

**Choice**: Comprehensive warning suppression for clean production output.

**Warnings Suppressed**:
| Source | Warning Type | Method |
|--------|--------------|--------|
| TensorFlow | cuFFT/cuDNN/cuBLAS registration | `TF_CPP_MIN_LOG_LEVEL=3` |
| TensorFlow | oneDNN messages | `TF_ENABLE_ONEDNN_OPTS=0` |
| OpenSoundscape | Architecture not in ARCH_DICT | Monkey-patch `warnings.warn` |
| PyTorch | pin_memory deprecation (C++) | `suppress_stderr()` context manager |

**Rationale**: These warnings are informational, not errors. Clean output is essential for production runs and log analysis.

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **BirdSet dataset unavailable** | Low | High | Verify all 8 datasets load before full run |
| **bioacoustics-model-zoo API changes** | Medium | Medium | Pin version; document API usage |
| **GPU OOM during extraction** | Medium | High | Reduce batch size; test on single sample first |
| **SLURM queue delays** | High | Low | Submit jobs in parallel; use inferno queue |
| **Embedding dimension mismatch** | Low | Medium | Validate dimensions in tests |
| **Model loading failures** | Medium | Medium | Graceful error handling; skip failed models |
| **I/O bottleneck during inference** | High | Low | Use num_workers; expected behavior |

## Open Questions (Resolved)

1. ~~Should we use class weights for imbalance?~~ → Start without; add if cmAP is poor
2. ~~For variable-length audio: max-pool vs random-crop?~~ → Use max-pool
3. ~~Should we normalize embeddings?~~ → Test both; document choice
4. ~~Optimal batch size?~~ → Start with 16; adjust based on GPU memory
5. ~~Why is GPU utilization low during inference?~~ → I/O bound by design; use num_workers to parallelize CPU preprocessing
6. ~~Why isn't BirdSetDataset used for Phase 2A?~~ → BMZ API only accepts file paths; BirdSetDataset is for Phase 2B

---

**Next**: Create detailed tasks.md with implementation steps.
