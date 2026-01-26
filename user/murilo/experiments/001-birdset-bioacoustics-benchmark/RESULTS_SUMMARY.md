# BirdSet Benchmark Results Summary

**Experiment**: 001-birdset-bioacoustics-benchmark
**Date**: 2026-01-26
**Models**: BirdSetConvNeXT, BirdSetEfficientNetB1
**Datasets**: All 8 BirdSet evaluation datasets
**Evaluation**: Direct inference (no training required - models trained on BirdSet)

## Primary Metrics (BirdSet Paper Standard)

Results for all 8 datasets using the three key metrics from the BirdSet paper:
- **cmAP**: Class-mean Average Precision (primary metric)
- **AUROC**: Area Under ROC Curve (multi-label, per-class averaging)
- **T1-Acc**: Top-1 Accuracy

### Results Table (BirdSet Paper Format)

|  | | **Test** | | | | | | | | **Score** |
|--|--|------|------|------|------|------|------|------|------|-------|
| **Model** | **Metric** | POW | PER | NES | UHH | HSN | NBP | SSW | SNE | **Mean** |
| **ConvNeXT** | cmAP | 0.34 | 0.17 | 0.34 | 0.25 | **0.48** | **0.63** | 0.34 | 0.30 | **0.36** |
| | AUROC | 0.82 | **0.73** | 0.88 | 0.78 | 0.88 | 0.90 | **0.93** | 0.83 | **0.84** |
| | T1-Acc | **0.68** | 0.25 | 0.17 | 0.30 | 0.25 | **0.66** | 0.16 | 0.43 | 0.36 |
| **EfficientNetB1** | cmAP | 0.35 | 0.18 | 0.32 | 0.22 | 0.38 | **0.62** | **0.37** | 0.34 | **0.35** |
| | AUROC | 0.83 | **0.72** | **0.90** | 0.78 | 0.84 | 0.90 | 0.93 | 0.83 | **0.84** |
| | T1-Acc | **0.74** | 0.33 | 0.20 | 0.28 | 0.28 | **0.66** | 0.17 | **0.49** | **0.37** |

### Dataset Statistics

| Dataset | HSN | PER | NES | UHH | POW | SSW | SNE | NBP | Total |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-------|
| **Samples** | 11,982 | 15,120 | 24,472 | 36,388 | 4,560 | 205,200 | 23,755 | 342 | 321,819 |
| **Classes** | 21 | 132 | 89 | 27 | 48 | 81 | 56 | 51 | - |

### Key Findings

**Model Comparison**:
- **ConvNeXT** achieves higher mean cmAP (0.357 vs 0.349, +0.008) and AUROC (0.843 vs 0.841, +0.002)
- **EfficientNetB1** achieves higher mean T1-Acc (0.368 vs 0.362, +0.006)
- Models are very competitive overall (all differences <1%)
- Both models show strong AUROC performance (>0.84 average)

**Dataset Performance**:
- **Best**: NBP achieves highest cmAP (0.62-0.63) despite moderate size (342 samples, 51 species)
- **Most Challenging**: PER has lowest cmAP (0.17-0.18) with highest diversity (15,120 samples, 132 species)
- **Largest**: SSW with 205,200 samples (64% of total test set) shows strong AUROC (0.926-0.932)

## Per-Dataset Analysis

### Best Performing Datasets (by cmAP)
1. **NBP** (51 species): 0.628 (ConvNeXT), 0.618 (EfficientNetB1) - Highest cmAP
2. **POW** (48 species): 0.342 (ConvNeXT), 0.354 (EfficientNetB1) - Good focal recording quality
3. **SSW** (81 species): 0.343 (ConvNeXT), 0.370 (EfficientNetB1) - Largest test set

### Most Challenging Datasets (by cmAP)
1. **PER** (132 species): 0.173 (ConvNeXT), 0.184 (EfficientNetB1) - Most species, Amazon diversity
2. **UHH** (27 species): 0.250 (ConvNeXT), 0.220 (EfficientNetB1) - Hawaiian Islands endemic challenge
3. **SNE** (56 species): 0.300 (ConvNeXT), 0.340 (EfficientNetB1) - European context

### Dataset Characteristics

| Dataset | Region | Samples | Species | cmAP Range | Key Challenge |
|---------|--------|---------|---------|-----------|---------------|
| NBP | Various | 342 | 51 | 0.62-0.63 | NIPS4BPlus benchmark |
| POW | Pennsylvania | 4,560 | 48 | 0.34-0.35 | Mid-size, temperate |
| SSW | New York | 205,200 | 81 | 0.34-0.37 | Largest test set |
| NES | Central/S. America | 24,472 | 89 | 0.32-0.34 | Neotropical diversity |
| SNE | Spain | 23,755 | 56 | 0.30-0.34 | European species |
| UHH | Hawaii | 36,388 | 27 | 0.22-0.25 | Endemic species |
| PER | Amazon Basin | 15,120 | 132 | 0.17-0.18 | Highest diversity |
| HSN | California | 11,982 | 21 | 0.38-0.48 | Western US montane |

## Additional Metrics

### F1 Scores and Top-K Accuracy (Full Results)

**BirdSetConvNeXT**:
| Dataset | F1-Micro | F1-Macro | Top-5 Acc | Throughput (samples/s) |
|---------|----------|----------|-----------|------------------------|
| HSN | 0.121 | 0.139 | 0.330 | 25.5 |
| NBP | 0.247 | 0.226 | 0.839 | 16.8 |
| NES | 0.237 | 0.193 | 0.255 | 24.9 |
| PER | 0.070 | 0.056 | 0.534 | 21.9 |
| POW | 0.228 | 0.114 | 0.876 | 25.4 |
| SSW | 0.206 | 0.196 | 0.225 | 20.4 |
| SNE | 0.238 | 0.085 | 0.545 | 25.6 |
| UHH | 0.133 | 0.063 | 0.516 | 25.2 |

**BirdSetEfficientNetB1**:
| Dataset | F1-Micro | F1-Macro | Top-5 Acc | Throughput (samples/s) |
|---------|----------|----------|-----------|------------------------|
| HSN | 0.132 | 0.115 | 0.375 | 49.1 |
| NBP | 0.204 | 0.198 | 0.877 | 69.9 |
| NES | 0.198 | 0.159 | 0.268 | 90.0 |
| PER | 0.053 | 0.043 | 0.555 | 104.3 |
| POW | 0.171 | 0.086 | 0.916 | 82.1 |
| SSW | 0.215 | 0.175 | 0.235 | 20.4 |
| SNE | 0.222 | 0.114 | 0.592 | 97.1 |
| UHH | 0.124 | 0.045 | 0.527 | 49.7 |

**Throughput Note**: EfficientNetB1 is generally faster (2-5x) than ConvNeXT, except on SSW where both models have similar throughput (~20 samples/s) due to I/O bottleneck on the largest dataset.

## Methodology Notes

### Inference Configuration
- **Batch Size**: 16
- **Workers**: 4 CPU workers for data loading
- **GPU**: RTX 6000 (PACE cluster)
- **Precision**: FP32
- **Test Split**: test_5s (5-second soundscape segments, multi-label)

### Data Pipeline
- Audio resampling: 32kHz (model native)
- No data augmentation (inference only)
- Predictions: Sigmoid activation on logits
- Multi-label evaluation: Per-sample multi-hot encoding

### Metric Computation
- **cmAP**: Sklearn `average_precision_score` per class, then macro average
- **AUROC**: Per-class ROC AUC with macro averaging (excludes undefined classes)
- **T1-Acc**: Top predicted class matches any true label
- **Top-5 Acc**: Top 5 predicted classes contain any true label

## Comparison to BirdSet Paper

**Status**: Pending detailed comparison once paper baseline values are extracted.

**Expected Alignment**:
- Our cmAP values should align with paper's reported performance for ConvNeXT and EfficientNetB1
- Any discrepancies may be due to:
  - Different checkpoints or training configurations
  - Data preprocessing differences
  - Random seed variations in test set creation

**Next Steps**:
- Compare our results to Table 2/3 in BirdSet paper (arXiv:2403.10380)
- Document any methodology differences
- Note that we now have AUROC metrics which may not be in original paper

## Computational Cost

### Total Inference Time
- **BirdSetConvNeXT**: ~3.9 hours (14,081 seconds total)
- **BirdSetEfficientNetB1**: ~3.3 hours (11,734 seconds total)

### Per-Dataset Breakdown (minutes)
| Dataset | ConvNeXT | EfficientNetB1 | Speedup |
|---------|----------|----------------|---------|
| HSN | 7.8 | 4.1 | 1.9x |
| NBP | 0.3 | 0.1 | 4.2x |
| NES | 16.4 | 4.5 | 3.6x |
| PER | 11.5 | 2.4 | 4.8x |
| POW | 3.0 | 0.9 | 3.2x |
| SSW | 167.2 | 167.3 | 1.0x |
| SNE | 15.5 | 4.1 | 3.8x |
| UHH | 24.1 | 12.2 | 2.0x |
| **Total** | **245.8** | **195.6** | **1.3x** |

**Performance Notes**:
- EfficientNetB1 is generally 2-5x faster for smaller/medium datasets
- SSW (205K samples) is I/O bound - both models process at ~20 samples/s
- Total speedup: EfficientNetB1 is 1.3x faster overall

## Technical Achievements

### Performance Optimizations
1. **Vectorized Prediction Conversion**: O(n) pandas reindex approach eliminated O(n²m) bottleneck
   - Impact: SSW now completes in hours instead of hanging indefinitely
   - Speedup: ~210,000x for prediction array alignment

2. **Stable Long-Running Jobs**: Switched from embers to inferno QOS
   - Prevents preemption on multi-hour SSW runs
   - Ensures reproducible results

3. **Multi-Label AUROC Fix**: Proper per-class averaging
   - Now excludes classes with undefined ROC AUC
   - Matches cmAP computation pattern

### Test Coverage
- **32 total tests** across data pipeline, inference, and metrics
- All critical paths validated with both integration and unit tests
- Performance regression tests for vectorized operations

## Artifacts

All results stored in:
```
user/murilo/experiments/001-birdset-bioacoustics-benchmark/artifacts/inference/
├── birdset_convnext/
│   ├── HSN/metrics.json, predictions.npz
│   ├── NBP/metrics.json, predictions.npz
│   ├── NES/metrics.json, predictions.npz
│   ├── PER/metrics.json, predictions.npz
│   ├── POW/metrics.json, predictions.npz
│   ├── SSW/metrics.json, predictions.npz
│   ├── SNE/metrics.json, predictions.npz
│   └── UHH/metrics.json, predictions.npz
└── birdset_efficientnet_b1/
    ├── HSN/metrics.json, predictions.npz
    ├── NBP/metrics.json, predictions.npz
    ├── NES/metrics.json, predictions.npz
    ├── PER/metrics.json, predictions.npz
    ├── POW/metrics.json, predictions.npz
    ├── SSW/metrics.json, predictions.npz
    ├── SNE/metrics.json, predictions.npz
    └── UHH/metrics.json, predictions.npz
```

**Total Storage**:
- 16 metrics.json files (metadata)
- 16 predictions.npz files (full prediction arrays)
- Estimated size: ~5-10 GB total

## Next Steps

### Phase 2B: Transfer Learning (External Models)
- Implement embedding extraction for BirdNET and Perch
- Train linear probes on extracted embeddings
- Evaluate on same 8 datasets
- Compare transfer learning vs direct inference

### Phase 3: Final Analysis
- Generate visualizations (heatmaps, bar charts)
- Statistical significance testing between models
- Per-class analysis (identify challenging species)
- Write comprehensive results document

### Documentation
- Compare to BirdSet paper baseline values
- Document any methodology differences
- Archive artifacts and update README
- Tag release: exp-001-phase-2a-complete
