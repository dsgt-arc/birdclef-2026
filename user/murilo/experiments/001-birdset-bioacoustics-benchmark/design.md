# Design: BirdSet Bioacoustics Model Benchmark

## Context

We are evaluating pre-trained bioacoustics models (BirdNET, Perch, BirdSetConvNeXT, BirdSetEfficientNetB1) on the full BirdSet benchmark (8 datasets) using linear probing. This complements the original BirdSet paper by testing external pre-trained models alongside BirdSet-native models.

The challenge is creating a scalable pipeline that can handle 4 models × 8 datasets = 32 combinations efficiently on the PACE cluster.

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
├── extract_embeddings/
│   ├── run.sh           # Controller: submit jobs for model × dataset
│   └── _job.sh          # Worker: extract embeddings for one combination
├── train_classifier/
│   ├── run.sh           # Controller: submit jobs for model × dataset
│   └── _job.sh          # Worker: train classifier for one combination
└── common.sh            # Shared functions (venv setup, paths)
```

### Decision 2: Two-Stage Pipeline (Extract → Train)

**Choice**: Separate embedding extraction (GPU job) from classifier training (GPU job).

**Rationale**:
- Embeddings are expensive to compute but reusable
- Classifier training benefits from GPU acceleration with PyTorch Lightning
- Enables caching embeddings to disk for reproducibility
- Allows independent re-running of either stage

**Alternatives Considered**:
- **End-to-end pipeline**: Simpler but recomputes embeddings every time
- **CPU classifier training**: Slower; GPU with PyTorch Lightning is faster

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

### Decision 4: Artifact Organization

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
│                    PHASE 2: EMBEDDING EXTRACTION                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  run.sh --model birdnet                                        │
│      │                                                          │
│      ├── sbatch _job.sh (MODEL=birdnet, DATASET=HSN)           │
│      ├── sbatch _job.sh (MODEL=birdnet, DATASET=PER)           │
│      ├── sbatch _job.sh (MODEL=birdnet, DATASET=NES)           │
│      └── ... (8 parallel GPU jobs)                             │
│                                                                 │
│  Each _job.sh:                                                  │
│      BirdSet(dataset) → Resample → Model.embed() → .npz        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 3: CLASSIFICATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  run.sh --model birdnet                                        │
│      │                                                          │
│      ├── sbatch _job.sh (MODEL=birdnet, DATASET=HSN)           │
│      ├── sbatch _job.sh (MODEL=birdnet, DATASET=PER)           │
│      └── ... (8 parallel GPU jobs)                             │
│                                                                 │
│  Each _job.sh:                                                  │
│      Load .npz → Train LinearProbe → Evaluate → metrics.json   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 4: AGGREGATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  python aggregate_results.py                                    │
│      │                                                          │
│      ├── Collect all metrics.json files                        │
│      ├── Generate results_matrix.csv                           │
│      ├── Create visualizations                                 │
│      └── Compare to baselines                                  │
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
│   ├── encoders.py                # Wrapper for bioacoustics-model-zoo
│   ├── extract_embeddings.py      # CLI: --model --dataset
│   ├── train_classifier.py        # CLI: --model --dataset
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
│   │   ├── extract_embeddings/
│   │   │   ├── run.sh             # Submit extraction jobs
│   │   │   └── _job.sh            # Single extraction job
│   │   └── train_classifier/
│   │       ├── run.sh             # Submit training jobs
│   │       └── _job.sh            # Single training job
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

### `sbatch/extract_embeddings/run.sh`
```bash
#!/bin/bash
# Submit embedding extraction jobs

MODELS=(birdnet perch birdset_convnext birdset_efficientnet_b1)
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
        artifact_dir="artifacts/embeddings/$model/$dataset"
        if [[ -f "$artifact_dir/metadata.json" ]]; then
            echo "SKIP: $model/$dataset (already completed)"
            continue
        fi

        echo "SUBMIT: $model/$dataset"
        if [[ "$DRY_RUN" == "false" ]]; then
            sbatch \
                --job-name="embed-$model-$dataset" \
                --output="artifacts/logs/embed-$model-$dataset-%j.log" \
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

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **BirdSet dataset unavailable** | Low | High | Verify all 8 datasets load before full run |
| **bioacoustics-model-zoo API changes** | Medium | Medium | Pin version; document API usage |
| **GPU OOM during extraction** | Medium | High | Reduce batch size; test on single sample first |
| **SLURM queue delays** | High | Low | Submit jobs in parallel; use inferno queue |
| **Embedding dimension mismatch** | Low | Medium | Validate dimensions in tests |
| **Model loading failures** | Medium | Medium | Graceful error handling; skip failed models |

## Open Questions (Resolved)

1. ~~Should we use class weights for imbalance?~~ → Start without; add if cmAP is poor
2. ~~For variable-length audio: max-pool vs random-crop?~~ → Use max-pool
3. ~~Should we normalize embeddings?~~ → Test both; document choice
4. ~~Optimal batch size?~~ → Start with 16; adjust based on GPU memory

---

**Next**: Create detailed tasks.md with implementation steps.
