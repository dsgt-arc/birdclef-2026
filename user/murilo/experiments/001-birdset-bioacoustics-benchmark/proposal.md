# Proposal: BirdSet Bioacoustics Model Benchmark

**Status**: in-progress (Phase 2A final dataset running)
**Created**: 2026-01-24
**Updated**: 2026-01-25 (evening)
**Author**: murilo

## Related Documentation

- **Vendor**:
  - [BirdSet Dataset](../../../docs/vendor/birdset.md) - Standardized bioacoustics benchmark
  - [Bioacoustics Model Zoo](../../../docs/vendor/bioacoustics-model-zoo.md) - Pre-trained model library
  - [BirdNET](../../../docs/vendor/birdnet-analyzer.md) - Species-specific neural network
  - [Perch](../../../docs/vendor/perch.md) - Google's bioacoustics model
- **References**:
  - [BirdSet Paper Summary](../../../docs/references/birdset-benchmark-paper.md) - arXiv:2403.10380
- **Related Experiments**:
  - [anthony/001-encoder-benchmark](../../anthony/experiments/001-encoder-benchmark/) - Similar methodology on ESC-50/AudioSet

## Hypothesis

Pre-trained bioacoustics models from the bioacoustics-model-zoo will perform well as feature extractors on the full BirdSet benchmark when using linear probing. This provides a **complementary evaluation** to the original BirdSet paper, which trained models directly on BirdSet data.

We expect:

1. **BirdNET** (trained on ~6,500 bird species) will achieve >0.50 cmAP averaged across all 8 test datasets
2. **Perch** (trained on diverse soundscapes) will achieve >0.45 cmAP averaged across all 8 test datasets
3. **BirdSet-native models** (BirdSetConvNeXT, BirdSetEfficientNetB1) will achieve highest cmAP since they were trained on BirdSet
4. All models will significantly outperform random embeddings (>10x)
5. Performance will vary across datasets due to geographic/acoustic diversity

**Rationale**: Domain-specific pre-trained models should transfer well to BirdSet. BirdSet-native models provide an upper bound, while external models (BirdNET, Perch) show transferability of representations learned on different data distributions.

## Background

The BirdSet benchmark (arXiv:2403.10380) provides standardized evaluation for bioacoustics models across **8 test datasets** representing diverse geographic regions and acoustic environments:

| Code | Name | Region |
|------|------|--------|
| HSN | High Sierra Nevada | California, USA |
| PER | Amazon Basin | Peru |
| NES | Colombia & Costa Rica | Central/South America |
| UHH | Hawaiian Islands | Hawaii, USA |
| POW | Powdermill Nature Reserve | Pennsylvania, USA |
| SSW | Sapsucker Woods | New York, USA |
| SNE | Sierra Nevada | Spain |
| NBP | NIPS4BPlus | Various |

**Key Challenge - Covariate Shift**: Models are trained on high-quality focal recordings but tested on noisy real-world soundscapes. This tests generalization to deployment conditions.

**Why This Experiment**: The original paper benchmarks architectures trained directly on BirdSet (EfficientNet, ConvNeXt, Wav2Vec2, AST, EAT, Bird-MAE). We complement this by evaluating **external pre-trained models** via linear probing to measure embedding quality transfer.

## Method

### Approach

Two-part evaluation with modular SLURM job scripts:

**Part A - Direct Inference (No Training)**:
- BirdSetConvNeXT and BirdSetEfficientNetB1 were trained directly on BirdSet
- Run inference to get class predictions, compute cmAP
- Reproduces/validates BirdSet paper results

**Part B - Transfer Learning (External Models)**:
- BirdNET and Perch are external pre-trained models with different taxonomies
- Option 1: **Linear probe** - Extract embeddings, train classifier on BirdSet labels
- Option 2: **Taxonomy mapping** - Map BirdNET predictions to eBird codes (if mapping exists)

### Models

| Model | Source | Sample Rate | Embedding Dim | Pretraining | Notes |
|-------|--------|-------------|---------------|-------------|-------|
| BirdNET | bioacoustics-model-zoo | 48kHz | 1024 | ~6,500 bird species | External pre-trained |
| Perch | bioacoustics-model-zoo | 32kHz | 1280 | YouTube, Xeno-Canto | External pre-trained |
| BirdSetConvNeXT | bioacoustics-model-zoo | 32kHz | TBD | BirdSet XCL | BirdSet-native baseline |
| BirdSetEfficientNetB1 | bioacoustics-model-zoo | 32kHz | TBD | BirdSet XCL | BirdSet-native baseline |

### Datasets

All 8 BirdSet evaluation datasets:

| Dataset | Train Split | Test Split | Species Count |
|---------|-------------|------------|---------------|
| HSN | train | test_5s | 21 |
| PER | train | test_5s | TBD |
| NES | train | test_5s | TBD |
| UHH | train | test_5s | TBD |
| POW | train | test_5s | TBD |
| SSW | train | test_5s | TBD |
| SNE | train | test_5s | TBD |
| NBP | train | test_5s | TBD |

### Setup

| Component | Details |
|-----------|---------|
| **Dataset** | BirdSet (DBD-research-group/BirdSet on HuggingFace) |
| **Train Split** | Focal recordings (variable length, single-label) |
| **Test Split** | test_5s (5-second soundscapes, multi-label) |
| **Compute** | PACE cluster with GPU (RTX 6000) |
| **Dependencies** | `datasets`, `bioacoustics-model-zoo`, `opensoundscape`, `torch`, `pytorch-lightning`, `tensorflow` |
| **Code** | `user/murilo/src/birdset_benchmark/` |

### Procedure

**Phase 1: Environment Setup** (Complete)
- Install dependencies including TensorFlow for Perch
- Verify all 8 BirdSet datasets load correctly
- Test model instantiation for all 4 models
- Create pytest test suite

**Phase 2A: Direct Inference (BirdSet-Native Models)**
1. Implement `run_inference.py` script with CLI flags:
   - `--model`: `birdset_convnext` or `birdset_efficientnet_b1`
   - `--dataset`: BirdSet dataset code (HSN, PER, etc.)
2. Create SLURM job scripts (`sbatch/inference/run.sh`, `_job.sh`)
3. Run inference directly on test data (no training needed)
4. Compute cmAP, F1, Top-K metrics
5. Scale to all datasets (2 models × 8 datasets = 16 runs)

**Phase 2B: Transfer Learning (External Models)**
1. Research taxonomy mapping (BirdNET → eBird codes)
   - If viable: implement direct inference with prediction mapping
   - If not viable: fall back to linear probe approach
2. If using linear probe:
   - Implement `extract_embeddings.py` for BirdNET/Perch
   - Implement `train_classifier.py` for linear probe training
   - Extract embeddings, train classifiers, evaluate
3. Scale to all datasets (2 models × 8 datasets = 16 runs)

**Phase 3: Results Aggregation**
1. Collect results from all 32 model × dataset combinations
2. Generate summary tables and visualizations
3. Compare to BirdSet paper baselines
4. Analyze: direct inference vs linear probe approaches

### Variables

- **Independent**:
  - Pre-trained model (BirdNET, Perch, BirdSetConvNeXT, BirdSetEfficientNetB1)
  - Dataset (HSN, PER, NES, UHH, POW, SSW, SNE, NBP)
- **Dependent**:
  - cmAP on each test set
  - F1 Micro/Macro
  - Top-1/Top-5 accuracy
  - Extraction throughput (samples/sec)
- **Controlled**:
  - Linear probe architecture (2-layer MLP)
  - Test protocol (5-second soundscapes, multi-label)
  - Training hyperparameters

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **cmAP** | Class-mean Average Precision (BirdSet standard, primary metric) |
| **F1 Micro** | Micro-averaged F1 score (overall classification quality) |
| **F1 Macro** | Macro-averaged F1 score (per-class average, handles imbalance) |
| **Top-1 Accuracy** | Percentage where true label is top prediction |
| **Top-5 Accuracy** | Percentage where true label is in top 5 |
| **Mean cmAP** | Average cmAP across all 8 datasets |

### Baseline Comparisons

| Model Type | Expected Performance |
|------------|----------------------|
| Random embeddings | ~0.05 cmAP |
| BirdSet-native (ConvNeXT, EfficientNet) | Highest (trained on BirdSet) |
| External pre-trained (BirdNET, Perch) | Mid-range (transfer learning) |
| BirdSet paper baselines | Reference for comparison |

### Success Criteria

- **Confirm if**:
  1. BirdSet-native models achieve highest mean cmAP
  2. BirdNET and Perch achieve >0.40 mean cmAP across all datasets
  3. All models significantly outperform random embeddings
  4. Pipeline scales to all 32 model × dataset combinations
  5. Results are reproducible with fixed random seed

- **Reject if**:
  1. Mean cmAP <0.25 for all models
  2. Pipeline fails on >25% of combinations
  3. Large variance in results across runs (>0.05 cmAP stdev)

## Deliverables

1. **Extracted Embeddings**: `.npz` files for all 4 models × 8 datasets (32 files)
2. **Trained Classifiers**: PyTorch Lightning checkpoints
3. **Results Matrix**: cmAP for all model × dataset combinations
4. **Visualizations**:
   - Heatmap: cmAP by model × dataset
   - Bar charts: Per-dataset comparison
   - Per-class AP analysis for top/bottom species
5. **SLURM Scripts**: Reusable `run.sh` and `_job.sh` for embedding/classification
6. **Artifacts**: `experiments/001-birdset-bioacoustics-benchmark/artifacts/`

## Limitations

- **Linear Probe Only**: May underestimate benefit of fine-tuning
- **No Data Augmentation**: Training on embeddings precludes audio-level augmentation
- **Class Imbalance**: Some species have few training samples
- **Taxonomic Mismatch**: External models use different taxonomies than BirdSet

## Future Work (Out of Scope)

- Fine-tuning models end-to-end
- Ensemble of multiple models
- Data augmentation at audio level
- Compare with custom distilled models
- Benchmark inference time on CPU for Kaggle

---

**Next Steps**: Implement embedding extraction pipeline (Phase 2), starting with single model + single dataset validation.
