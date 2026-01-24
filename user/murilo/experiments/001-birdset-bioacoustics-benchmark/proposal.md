# Proposal: BirdSet Bioacoustics Model Benchmark

**Status**: proposed
**Created**: 2026-01-24
**Author**: murilo

## Related Documentation

- **Vendor**:
  - [BirdSet Dataset](../../../docs/vendor/birdset.md) - Standardized bioacoustics benchmark
  - [Bioacoustics Model Zoo](../../../docs/vendor/bioacoustics-model-zoo.md) - Pre-trained model library
  - [BirdNET](../../../docs/vendor/birdnet-analyzer.md) - Species-specific neural network
  - [Perch](../../../docs/vendor/perch.md) - Google's bioacoustics model
- **References**:
  - BirdSet Paper: Standardized evaluation for bioacoustics models
- **Related Experiments**:
  - [anthony/001-encoder-benchmark](../../anthony/experiments/001-encoder-benchmark/) - Similar methodology on ESC-50/AudioSet

## Hypothesis

Pre-trained bioacoustics models (BirdNET and Perch) will perform well as feature extractors on the BirdSet benchmark when using linear probing, despite taxonomic misalignment between their training data and BirdSet's eBird codes. We expect:

1. **BirdNET** (trained on ~6,500 bird species) will achieve >0.60 cmAP on HSN test set
2. **Perch** (trained on diverse soundscapes) will achieve >0.50 cmAP on HSN test set
3. Both models will outperform random embeddings by >20x
4. Domain-specific pretraining (bioacoustics) will outperform general audio models (BEATs) on fine-grained bird classification

**Rationale**: These models were trained on bird vocalizations, so their learned representations should transfer well to BirdSet despite not sharing the exact same taxonomy. This establishes a baseline for comparison with custom distillation approaches.

## Background

The BirdSet benchmark provides standardized evaluation for bioacoustics models across 8 test datasets with varying difficulty. Before developing custom models (discrete tokens, distillation), we need baseline performance from state-of-the-art pre-trained models.

**Key Challenges**:
- **Taxonomic Mismatch**: BirdNET uses its own taxonomy; BirdSet uses eBird codes → Zero-shot won't work
- **Audio Length Variation**: BirdSet train samples vary (1-60s); test is standardized to 5s soundscape clips
- **Multi-label Classification**: Test soundscapes contain multiple species simultaneously
- **Class Imbalance**: Some species have few training examples

**Why Linear Probing**: Isolates the quality of learned representations from classifier capacity, providing a fair comparison across models.

## Method

### Approach

Two-stage pipeline:
1. **Feature Extraction**: Extract embeddings from pre-trained models for all BirdSet samples
2. **Linear Probing**: Train lightweight classifier on embeddings, evaluate with cmAP metric

Start with **HSN (High Sierra Nevada)** as pilot dataset, then expand to all 8 test sets if successful.

### Models

| Model | Source | Sample Rate | Embedding Dim | Pretraining |
|-------|--------|-------------|---------------|-------------|
| BirdNET | bioacoustics-model-zoo | 48kHz | 1024 | ~6,500 bird species (proprietary dataset) |
| Perch | bioacoustics-model-zoo | 32kHz | 1280 | Diverse soundscapes (YouTube, Xeno-Canto) |

### Setup

| Component | Details |
|-----------|---------|
| **Dataset** | BirdSet HSN (DBD-research-group/BirdSet on HuggingFace) |
| **Train Split** | Focal recordings (variable length, single-label) |
| **Test Split** | test_5s (5-second soundscapes, multi-label) |
| **Compute** | GPU for embedding extraction, CPU for linear probe |
| **Dependencies** | `datasets`, `bioacoustics-model-zoo`, `opensoundscape`, `torch`, `scikit-learn`, `polars` |
| **Code** | `user/murilo/src/birdset_benchmark/` |

### Procedure

**Part A: Environment Setup**
1. Install dependencies (prefer GitHub version of bioacoustics-model-zoo)
2. Verify BirdSet HSN loads correctly from HuggingFace
3. Test audio resampling pipeline (FLAC/OGG → model sample rate)

**Part B: Feature Extraction** (Run 1 - GPU job)
1. Load BirdSet HSN train and test_5s splits
2. For each model (BirdNET, Perch):
   - Instantiate model with `bmz.BirdNET()` / `bmz.load_model("perch")`
   - Create DataLoader with batching (batch_size=16)
   - Handle variable-length audio:
     - **Train**: Max-pool embeddings across time if >5s, pad if <5s
     - **Test**: Use full 5s clip (already standardized)
   - Extract embeddings and save to disk:
     - `embeddings_train_birdnet.npz` (shape: [N_train, 1024])
     - `embeddings_test_birdnet.npz` (shape: [N_test, 1024])
     - Save corresponding labels as `labels_train.npz`, `labels_test.npz`
3. Log extraction time per model (throughput: samples/sec)

**Part C: Linear Probe Training** (Run 2 - CPU job)
1. Load saved embeddings from Part B
2. Prepare multi-label targets:
   - One-hot encode `ebird_code_multilabel` column
   - Handle class imbalance with class weights (optional)
3. Train classifier options:
   - **Option 1**: `sklearn.linear_model.LogisticRegression` with `OneVsRestClassifier`
   - **Option 2**: Single-layer PyTorch MLP (768 → num_classes) with BCE loss
4. Train on train embeddings, validate on subset if needed
5. Evaluate on test embeddings

**Part D: Evaluation**
1. Compute predictions on test set
2. Calculate **cmAP (Class-mean Average Precision)** using BirdSet's metric
3. Additional metrics:
   - Per-class AP (identify which species are well-represented)
   - Top-1/Top-5 accuracy
   - Confusion analysis (which species are confused)
4. Compare to BirdSet published baselines (if available)

### Variables

- **Independent**:
  - Pre-trained model (BirdNET vs Perch)
  - Audio preprocessing strategy (max-pool vs random-crop for long samples)
  - Classifier type (Logistic Regression vs MLP)
- **Dependent**:
  - cmAP on HSN test set
  - Embedding extraction time (samples/sec)
  - Training time for linear probe
- **Controlled**:
  - Dataset (HSN)
  - Embedding dimension (fixed per model)
  - Test protocol (5-second soundscapes, multi-label)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **cmAP** | Class-mean Average Precision (BirdSet standard, primary metric) |
| **F1 Micro** | Micro-averaged F1 score (overall classification quality) |
| **F1 Macro** | Macro-averaged F1 score (per-class average, handles imbalance) |
| **Top-1 Accuracy** | Percentage where true label is top prediction |
| **Top-5 Accuracy** | Percentage where true label is in top 5 |
| **Extraction Throughput** | Samples processed per second during embedding extraction |
| **Per-class AP** | Identify which species benefit from transfer learning |

### Baseline

| Comparison | Expected Performance |
|------------|----------------------|
| Random embeddings | ~0.05 cmAP |
| BirdSet published baseline | TBD (check leaderboard) |
| General audio models (BEATs from anthony/001) | Unknown on bird-specific task |

### Success Criteria

- **Confirm if**:
  1. BirdNET achieves cmAP >0.60 on HSN test set
  2. Perch achieves cmAP >0.50 on HSN test set
  3. Both models significantly outperform random embeddings (>10x)
  4. Embedding extraction completes in <1 hour for full HSN dataset
  5. Results are reproducible with fixed random seed

- **Reject if**:
  1. cmAP <0.30 for both models (worse than expected random performance with some signal)
  2. Embedding extraction takes >4 hours (pipeline too slow)
  3. Large variance in results across runs (>0.05 cmAP stdev)

## Deliverables

1. **Extracted Embeddings**: Saved `.npz` files for both models (train + test)
2. **Trained Classifiers**: Serialized sklearn/PyTorch models
3. **Results Table**: cmAP, Top-1/5 accuracy, timing for both models
4. **Visualization**: Per-class AP bar charts, confusion matrices for top species
5. **Artifacts**: `user/murilo/experiments/001-birdset-bioacoustics-benchmark/artifacts/`
   - `results_summary.json`
   - `per_class_ap.csv`
   - `confusion_matrix.png`
6. **Reusable Code**: Modular embedding extraction and linear probe training scripts

## Limitations

- **Taxonomic Mismatch**: Zero-shot not possible; we're measuring embedding quality, not end-to-end model performance
- **Single Dataset**: HSN only for pilot; generalization to other BirdSet tasks unknown
- **Simple Classifier**: Linear probe may underestimate benefit of fine-tuning
- **Audio Preprocessing**: Max-pooling strategy for long clips may lose temporal information
- **No Data Augmentation**: Training on embeddings precludes audio-level augmentation
- **Class Imbalance**: Some species have <10 training samples in BirdSet

## Future Work (Out of Scope)

- Expand to all 8 BirdSet test datasets
- Fine-tuning models end-to-end (not just linear probe)
- Data augmentation at audio level
- Ensemble of BirdNET + Perch
- Compare with custom distilled models from anthony/003
- Benchmark inference time on CPU for Kaggle submission feasibility

## Open Questions

1. Should we use class weights to handle imbalance during linear probe training?
2. For variable-length audio: max-pool vs random 5s crop vs center crop?
3. Should we normalize embeddings before training classifier?
4. What's the optimal batch size for embedding extraction given GPU memory?

---

**Next Steps**: Review this proposal, clarify open questions, then proceed to `design.md` and `tasks.md` before implementation.
