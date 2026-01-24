# Design: BirdSet Bioacoustics Model Benchmark

## Context

We need to establish baseline performance for pre-trained bioacoustics models (BirdNET, Perch) on the BirdSet benchmark before investing in custom model development. The challenge is integrating models with different sample rates, embedding dimensions, and preprocessing requirements into a unified evaluation pipeline.

## Goals / Non-Goals

**Goals:**
- Extract embeddings efficiently from BirdNET and Perch for all BirdSet HSN samples
- Train reproducible linear probes that isolate embedding quality
- Produce cmAP scores comparable to BirdSet published baselines
- Create reusable code for expanding to other BirdSet datasets

**Non-Goals:**
- Fine-tuning models end-to-end (linear probe only)
- Zero-shot classification (taxonomies don't align)
- Real-time inference optimization (batch processing acceptable)
- Hyperparameter tuning for linear probe (use sensible defaults)

## Technical Decisions

### Decision 1: Two-Stage Pipeline (Extract → Train)

**Choice**: Separate embedding extraction (GPU job) from classifier training (CPU job).

**Rationale**:
- Embeddings are expensive to compute (GPU) but reusable across experiments
- Classifier training is fast (CPU) and benefits from iteration
- Enables caching embeddings to disk for reproducibility
- Aligns with SLURM workflow: submit GPU job → save embeddings → submit CPU job

**Alternatives Considered**:
- **End-to-end pipeline**: Simpler but recomputes embeddings every time; wasteful for experimentation
- **In-memory pipeline**: Faster but not feasible if embeddings don't fit in RAM (~1-10GB)

**Implementation**:
```python
# Stage 1: extract_embeddings.py (GPU)
embeddings = extract_all(model, dataloader)
np.savez(f"embeddings_{model_name}.npz", **embeddings)

# Stage 2: train_probe.py (CPU)
data = np.load("embeddings_birdnet.npz")
classifier.fit(data['train'], labels_train)
```

### Decision 2: Audio Preprocessing Strategy for Variable-Length Clips

**Choice**: Use **max-pooling over time** for train samples longer than 5s.

**Rationale**:
- BirdSet train samples are 1-60s (variable); test is 5s (fixed)
- Max-pooling:
  - ✅ Preserves most salient features (peak activations)
  - ✅ Single embedding per sample (simplifies linear probe)
  - ✅ Translation-invariant (bird call can occur anywhere in clip)
- Aligns with how models handle soundscapes in production

**Alternatives Considered**:
- **Random 5s crop**: Fast but loses information; high variance across runs
- **Center crop**: Assumes bird is in middle (often untrue)
- **Multiple crops + averaging**: Better but computationally expensive
- **Pad short clips**: Necessary for <5s; use silence padding

**Implementation**:
```python
def preprocess_audio(audio, target_duration=5.0, sr=48000):
    target_samples = int(target_duration * sr)
    if len(audio) < target_samples:
        # Pad with zeros
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        # Extract embeddings for overlapping windows, then max-pool
        embeddings = model.extract_windows(audio, window_size=target_samples)
        return np.max(embeddings, axis=0)  # Max-pool across time
    return model.extract(audio)
```

### Decision 3: Linear Probe Implementation

**Choice**: Use **scikit-learn LogisticRegression with OneVsRestClassifier** for multi-label classification.

**Rationale**:
- ✅ Simple, well-tested, no GPU required
- ✅ Built-in support for multi-label via OneVsRest wrapper
- ✅ L2 regularization (C parameter) handles overfitting
- ✅ Fast training (<1 minute for typical BirdSet dataset)
- ✅ Matches methodology in anthony/001-encoder-benchmark

**Alternatives Considered**:
- **PyTorch MLP**: More flexible but overkill for linear probe; adds complexity
- **MultiOutputClassifier**: Similar to OneVsRest but less standard
- **XGBoost**: Non-linear; violates "linear probe" constraint

**Hyperparameters** (defaults, no tuning):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

classifier = OneVsRestClassifier(
    LogisticRegression(
        C=1.0,              # L2 regularization
        max_iter=1000,      # Sufficient for convergence
        random_state=42,    # Reproducibility
        n_jobs=-1           # Parallel across classes
    )
)
```

### Decision 4: Batching Strategy for Embedding Extraction

**Choice**: Batch size = 16, with dynamic padding to handle variable-length audio.

**Rationale**:
- GPU memory efficient (tested on V100/L40S with 16GB)
- Balances throughput vs latency
- Models (BirdNET, Perch) accept batched input

**Implementation**:
```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    """Pad batch to longest sample."""
    audio_list = [item['audio'] for item in batch]
    max_len = max(len(a) for a in audio_list)
    padded = [np.pad(a, (0, max_len - len(a))) for a in audio_list]
    return {
        'audio': torch.tensor(padded),
        'labels': [item['labels'] for item in batch]
    }

dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=collate_fn,
    num_workers=4
)
```

### Decision 5: Embedding Storage Format

**Choice**: Save as `.npz` (NumPy compressed) with separate files per split and model.

**File Structure**:
```
artifacts/
├── embeddings_train_birdnet.npz    # {'embeddings': [N, 1024], 'labels': [N, num_classes], 'ids': [N]}
├── embeddings_test_birdnet.npz
├── embeddings_train_perch.npz      # {'embeddings': [N, 1280], ...}
├── embeddings_test_perch.npz
└── metadata.json                    # Model configs, sample rates, extraction time
```

**Rationale**:
- ✅ Efficient compression (~50% size reduction vs .npy)
- ✅ Multiple arrays per file (embeddings + labels + metadata)
- ✅ Compatible with both NumPy and PyTorch
- ✅ Standard format used in anthony/001

### Decision 6: Evaluation Metric Implementation

**Choice**: Use BirdSet's official cmAP implementation (if available), otherwise implement from scratch. Include F1 scores for comprehensive evaluation.

**Primary Metrics**:
```python
def compute_cmap(y_true, y_pred):
    """
    Class-mean Average Precision (BirdSet standard)
    y_true: [N, C] binary labels
    y_pred: [N, C] predicted probabilities
    Returns: mean of per-class AP
    """
    aps = []
    for class_idx in range(y_true.shape[1]):
        ap = average_precision_score(y_true[:, class_idx], y_pred[:, class_idx])
        aps.append(ap)
    return np.mean(aps)  # Macro-average across classes

def compute_f1_scores(y_true, y_pred_binary):
    """
    F1 scores for multi-label classification
    y_true: [N, C] binary labels
    y_pred_binary: [N, C] binary predictions (thresholded)
    Returns: (f1_micro, f1_macro)
    """
    f1_micro = f1_score(y_true, y_pred_binary, average='micro')
    f1_macro = f1_score(y_true, y_pred_binary, average='macro')
    return f1_micro, f1_macro
```

**Rationale**:
- **cmAP**: Aligns with BirdSet benchmark standard; handles class imbalance; comparable to AudioSet mAP from anthony/001
- **F1 Micro**: Overall classification quality across all samples (gives more weight to common classes)
- **F1 Macro**: Per-class average (treats all classes equally, better for imbalanced datasets)

## Data Flow

```
BirdSet HSN (HuggingFace)
    ↓
[Download & Cache]
    ↓
Audio Files (FLAC/OGG, variable length)
    ↓
[Resample to model SR (32kHz or 48kHz)]
    ↓
Preprocessed Audio (padded/cropped to 5s)
    ↓
[Batch → GPU → Model.extract_embeddings()]
    ↓
Embeddings [N, D] + Labels [N, C]
    ↓
[Save to .npz]
    ↓
[Load embeddings → Train LogisticRegression]
    ↓
Classifier (OneVsRest)
    ↓
[Predict on test embeddings]
    ↓
cmAP Score
```

## Code Structure

```
user/murilo/src/birdset_benchmark/
├── __init__.py
├── data.py                  # BirdSet loading, preprocessing
├── models.py                # Wrapper for bioacoustics-model-zoo
├── extract_embeddings.py    # Stage 1: GPU job
├── train_probe.py           # Stage 2: CPU job
├── evaluate.py              # Compute cmAP, per-class AP
└── utils.py                 # Audio resampling, batching

user/murilo/experiments/001-birdset-bioacoustics-benchmark/
├── proposal.md
├── design.md                # This file
├── tasks.md
├── config.yaml              # Model configs, paths
├── sbatch/
│   ├── extract_embeddings.sbatch   # GPU job (L40S, 4 hours)
│   └── train_probe.sbatch          # CPU job (10 minutes)
└── artifacts/
    ├── embeddings_*.npz
    ├── classifiers/
    │   ├── birdnet_probe.pkl
    │   └── perch_probe.pkl
    ├── results_summary.json
    ├── per_class_ap.csv
    └── plots/
        ├── confusion_matrix_birdnet.png
        └── per_class_ap_comparison.png
```

## Environment Setup

**Dependencies** (add to `pyproject.toml`):
```toml
[project]
dependencies = [
    "torch>=2.0",
    "datasets>=2.0",              # HuggingFace datasets
    "bioacoustics-model-zoo",     # Install from GitHub
    "opensoundscape>=0.10",       # Audio utilities
    "scikit-learn>=1.3",
    "polars>=0.19",               # Preferred over pandas
    "librosa>=0.10",              # Audio resampling
    "numpy>=1.24",
    "matplotlib>=3.7",            # Plotting
    "tqdm>=4.65",                 # Progress bars
]
```

**Installation Command**:
```bash
uv pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo.git
```

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **BirdSet dataset unavailable/incomplete** | Low | High | Verify HF dataset loads before full run; have fallback to local files |
| **bioacoustics-model-zoo API changes** | Medium | Medium | Pin version in dependencies; document API usage |
| **GPU OOM during embedding extraction** | Medium | High | Reduce batch size to 8 or 4; use gradient checkpointing if available |
| **Class imbalance degrades cmAP** | High | Medium | Report per-class AP alongside cmAP; consider class weights |
| **Max-pooling loses temporal info** | Medium | Low | Acceptable for baseline; document limitation for future work |

## Migration Plan

N/A (new experiment, no migration)

## Open Questions

1. **Class Weighting**: Should we use `class_weight='balanced'` in LogisticRegression to handle imbalance?
   - **Recommendation**: Start without; add if cmAP is poor
2. **Embedding Normalization**: Should embeddings be L2-normalized before linear probe?
   - **Recommendation**: Test both; cosine similarity often helps
3. **Train/Val Split**: BirdSet provides train + test; should we carve out validation set?
   - **Recommendation**: No; use full train for probe, report on test only (matches benchmark protocol)
4. **Optimal Batch Size**: 16 is default; test 8/32 if GPU memory is issue?
   - **Recommendation**: Profile first batch; adjust if needed

---

**Next**: Create `tasks.md` with step-by-step implementation checklist.
