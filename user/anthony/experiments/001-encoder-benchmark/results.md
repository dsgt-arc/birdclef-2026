# Results: Encoder Benchmark on ESC-50

**Date Completed**: 2026-01-10
**Author**: Anthony Miyaguchi

## Summary

BEATs significantly outperforms other encoders for environmental sound classification using frozen features with a linear probe. EAT requires mean pooling (not CLS token) for competitive performance.

**Verdict**: confirmed (BEATs is best for frozen feature extraction)

## Observations

### Quantitative

| Encoder | Mean Accuracy | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Embed Dim | Eval Time |
|---------|--------------|--------|--------|--------|--------|--------|-----------|-----------|
| BEATs | **95.8%** | 94.7% | 95.5% | 95.7% | 98.5% | 94.5% | 768 | 107s |
| EAT | 77.9% | 76.0% | 79.0% | 75.7% | 79.7% | 79.2% | 768 | 162s |
| wav2vec2 | 48.2% | 51.0% | 45.7% | 49.0% | 50.5% | 44.7% | 768 | 256s |

#### Comparison to Published Results

| Encoder | Our Result | Published | Method | Gap |
|---------|-----------|-----------|--------|-----|
| BEATs | 95.8% | 98.1% | Linear probe vs Finetuned | -2.3% |
| EAT | 77.9% | 95.9% | Linear probe vs Finetuned | -18.0% |
| wav2vec2 | 48.2% | N/A | Speech model, not designed for ESC | N/A |

### GPU Statistics (EAT run, V100-16GB)

| Metric | Value |
|--------|-------|
| Peak Memory | 1,372 MiB (8.4% of 16GB) |
| Peak GPU Utilization | 84% |
| Peak Power Draw | 248.8 W |
| Temperature Range | 45-61°C |
| Job Duration | 3m 51s |

GPU utilization was bursty (0-84%) due to alternating between:
- Feature extraction (GPU-bound, high utilization)
- Linear probe training on CPU (sklearn, 0% GPU)

### Qualitative

- BEATs frozen features are remarkably effective, achieving 95.8% with just a linear probe
- EAT initially showed 45% accuracy due to using CLS token; switching to mean pooling improved to 77.9%
- wav2vec2 (speech model) performs at near-random levels on environmental sounds, as expected
- All encoders produce 768-dimensional embeddings
- BEATs is fastest for feature extraction despite not using HuggingFace infrastructure

## Analysis

BEATs' strong frozen-feature performance suggests its pretraining objective (iterative audio tokenization with masked prediction) creates highly transferable representations. The 2.3% gap to finetuned performance is surprisingly small.

EAT's larger gap (18%) indicates the pretrained model benefits more from task-specific finetuning. The CLS token issue suggests EAT wasn't trained with CLS-based classification objectives during pretraining.

### Feature Extraction Strategy

| Encoder | Pooling Method | Why |
|---------|---------------|-----|
| BEATs | Mean pooling | Matches published implementation |
| EAT | Mean pooling (fixed) | CLS token gave 45%, mean pooling gives 78% |
| wav2vec2 | Mean pooling | Standard for sequence models |

### Confounders

- **Linear probe vs finetuning**: Published results use finetuning; we use frozen features
- **Hyperparameters**: Used 100 epochs, lr=1e-3, batch_size=32 for all; may not be optimal per-encoder
- **EAT preprocessing**: Verified against vendored code, but HuggingFace wrapper may have subtle differences
- **No data augmentation**: Published results may use SpecAugment, mixup, etc.

## Artifacts

| Type | Location |
|------|----------|
| Results JSON | `~/scratch/birdclef-2026/outputs/esc50-benchmark/*.json` |
| GPU Stats | `sbatch/logs/gpu_stats-3234084.csv` |
| Job Logs | `sbatch/logs/benchmark-*.out` |
| Checkpoints | `~/scratch/birdclef-2026/checkpoints/` |
| Code | `user/anthony/src/birdclef/` |

### Checkpoint Sources

| Model | Source | Notes |
|-------|--------|-------|
| BEATs | `lpepino/beats_ckpts` (HuggingFace) | Mirror by Leonardo Pepino; official OneDrive not programmatically accessible |
| EAT | `worstchan/EAT-base_epoch30_pretrain` (HuggingFace) | Official release from EAT authors |
| wav2vec2 | `facebook/wav2vec2-base-960h` (HuggingFace) | Standard pretrained model |

## Next Steps

- [ ] Try finetuning EAT to close the gap with published results
- [ ] Add BirdNET encoder for comparison (bird-specific pretraining)
- [ ] Run on AudioSet bird subset (stretch goal)
- [ ] Investigate EAT-large model
- [ ] Add data augmentation to linear probe training

**Leads to**: BEATs recommended as primary encoder for BirdCLEF 2026 feature extraction pipeline
