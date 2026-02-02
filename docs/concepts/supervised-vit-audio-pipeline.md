# Concept: Supervised Vision Transformer Training Pipeline for Bioacoustics

**Status**: implementing (Experiment 002)
**Created**: 2026-02-01
**Updated**: 2026-02-01

## Idea

Build an end-to-end supervised Vision Transformer (ViT) training pipeline for BirdSet that generates mel-spectrograms on-the-fly, maximizes GPU utilization through optimized data loading, and provides a strong baseline for bioacoustic classification tasks.

**Key insight**: Vision Transformers can treat mel-spectrograms as images, leveraging ImageNet-pretrained weights as initialization. By generating mel-spectrograms on-the-fly with parallel CPU workers, we avoid storing ~500GB of preprocessed data while keeping the GPU saturated. This infrastructure becomes reusable for future experiments (self-supervised pre-training, fine-tuning, distillation).

## Motivation

**Why supervised ViT baseline?**:
1. **Infrastructure validation**: Establish PyTorch Lightning pipeline before attempting complex self-supervised methods
2. **Performance baseline**: Quantify supervised ViT performance to compare against future approaches (MAE pre-training, BEATS distillation, etc.)
3. **Reusable components**: DataModule, preprocessing, and metrics can be reused across experiments
4. **Model constraints**: Competition requires CPU inference - need to validate that models fit on single GPU

**Why on-the-fly mel-spectrogram generation?**:
- **Storage savings**: XCM preprocessed spectrograms would require ~500GB; on-the-fly needs only ~90GB raw audio
- **Flexibility**: Easy to change preprocessing parameters (n_mels, hop_length) without regenerating entire dataset
- **CPU parallelism**: Modern CPUs have 16+ cores; leverage them to keep GPU saturated

**Why start with HSN before XCM?**:
- **Fast iteration**: HSN trains in 2-4 hours vs. XCM's 24-48 hours
- **Quick debugging**: Infrastructure bugs surface faster on smaller dataset
- **Methodical approach**: Validate on manageable scale before committing to large-scale training

## Background

**Vision Transformers for Audio**:
- AST (Audio Spectrogram Transformer): First major work applying ViT to audio spectrograms
- BEATs: Self-supervised ViT for audio achieves SOTA on many benchmarks
- ViT treats 16×16 patches of spectrogram as tokens (similar to word embeddings in NLP)

**BirdSet Context**:
- XCM dataset: ~160k recordings, ~9,736 species (long-tail distribution)
- HSN dataset: ~5,460 recordings, 21 species (development/debugging)
- Current best supervised: ConvNeXT cmAP 0.36, EfficientNetB1 cmAP 0.35
- Multi-label classification: Birds can overlap in soundscape recordings

**Key References**:
- ViT: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
- AST: "AST: Audio Spectrogram Transformer" (Gong et al., 2021)
- BirdSet: "BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics"
- PyTorch Lightning: Framework for clean, scalable training loops

## Architecture Overview

```
Raw Audio (32kHz, 5s)
    ↓
DataLoader Workers (8-16 parallel)
    ├─> Load & resample audio
    ├─> Generate mel-spectrogram (128 mels, 500 frames)
    └─> Normalize (AudioSet stats)
    ↓
Batch (32, 1, 128, 500)
    ↓
ViT-Base (timm)
    ├─> 1-channel patch embedding (averaged from RGB weights)
    ├─> 12 transformer layers
    └─> Linear classifier head
    ↓
Logits (32, num_classes)
    ↓
BCEWithLogitsLoss (multi-label)
    ↓
Metrics: cmAP, F1-macro/micro, Top-K accuracy
```

## Key Design Decisions

### 1. On-the-fly Mel-Spectrogram Generation

**Decision**: Generate spectrograms in DataLoader collate function, not pre-compute.

**Rationale**:
- Saves ~500GB storage (XCM preprocessed data)
- CPU parallelism (8-16 workers) can keep GPU saturated
- Flexible: change n_mels/hop_length without regenerating dataset
- Proven approach: EAT encoder uses similar pattern

**Trade-offs**:
- Requires tuning num_workers to avoid CPU bottleneck
- Slightly more complex than loading pre-computed tensors
- Must profile to ensure GPU utilization >80%

### 2. AudioSet Normalization Constants

**Decision**: Use AudioSet mean=-4.268, std=4.569 instead of BirdSet-specific stats.

**Rationale**:
- Existing BirdSet models use AudioSet normalization successfully
- Enables transfer learning from AudioSet-pretrained models
- Avoids full dataset pass to compute BirdSet-specific stats

**Trade-offs**:
- May be suboptimal for BirdSet distribution
- Can revisit if performance is unexpectedly poor

### 3. Map-Style Dataset (Download to Cache)

**Decision**: Download full dataset to local cache vs. streaming.

**Rationale**:
- XCM size (~90GB) is manageable on PACE storage
- Simpler than streaming: standard PyTorch Dataset, reliable checkpointing
- Faster iteration during development

**Trade-offs**:
- Requires disk space
- Streaming could be added later for larger datasets

### 4. Single-GPU First, Multi-GPU Optional

**Decision**: Focus on single-GPU training; keep multi-GPU as optional scaling path.

**Rationale**:
- Competition requires CPU inference (models must fit on single GPU)
- Simpler debugging and development
- Multi-GPU provides minimal benefit for HSN (already fast)
- Can enable multi-GPU for XCM if needed

**Trade-offs**:
- XCM training might take 24-36 hours on single GPU
- Multi-GPU would reduce to 12-18 hours but adds DDP complexity

### 5. HSN → XCM Development Workflow

**Decision**: Develop and debug on HSN, then scale to XCM.

**Rationale**:
- HSN provides ~10x faster iteration (2-4 hours vs. 24-48 hours)
- Infrastructure bugs (OOM, data loading) surface quickly
- Config change to scale: just update dataset_name in config.yaml

**Trade-offs**:
- Potential transfer issues (though both are BirdSet datasets)
- Must validate on XCM with fast-dev-run before full training

## Implementation Details

### Mel-Spectrogram Transform
Following EAT encoder patterns:
- Kaldi fbank: 128 mel bins, 10ms frame shift, htk_compat=True
- Normalization: (fbank - mean) / (std * 2)
- Target: 500 frames (~5 seconds at 100 FPS)

### ViT Patch Embedding Adaptation
- Load pretrained ViT-Base (ImageNet)
- Replace 3-channel conv with 1-channel: average RGB weights to initialize
- Preserves some pretrained knowledge while adapting to grayscale input

### DataLoader Optimization
- 8-16 workers for parallel mel-spec generation
- persistent_workers=True to avoid worker respawn overhead
- prefetch_factor=4 to keep GPU fed with batches
- pin_memory=True for faster GPU transfer

### Multi-Label Metrics
- **cmAP** (class-mean Average Precision): BirdSet primary metric
- **F1-macro**: Unweighted mean of per-class F1 (class-balanced)
- **F1-micro**: Global precision/recall (biased toward frequent classes)
- **Top-1/5 Accuracy**: Fraction with true label in top predictions

## Success Criteria

**Infrastructure validation**:
- GPU utilization consistently >80% during training
- Data loading overhead <10%
- Training completes without OOM on batch_size=32
- Checkpointing and W&B logging work correctly

**Performance targets**:
- **HSN**: Validation cmAP > 0.30 (better than random ~0.048)
- **XCM**: Validation cmAP > 0.25 (competitive with BirdSet baselines)
- Loss decreases monotonically
- Top-5 accuracy > 60%

**Rejection criteria**:
- GPU utilization <50% (data loading bottleneck)
- OOM on batch_size=32 (need architecture changes)
- cmAP < 0.10 on HSN (indicates implementation bug)

## Open Questions

**Data Loading**:
- What's the optimal num_workers for PACE hardware? (Start with 8, tune if GPU util <80%)
- Should we use gradient accumulation if batch_size=32 OOMs, or reduce batch_size?

**Training**:
- Is cosine annealing better than ReduceLROnPlateau for XCM?
- Early stopping patience: 10 epochs sufficient, or increase to 15-20 for XCM?
- Does HSN cmAP predict XCM cmAP, or are they uncorrelated?

**Performance**:
- How does ViT compare to CNNs (ConvNeXT, EfficientNet) on BirdSet?
- Is ImageNet pre-training helpful, or does random init perform similarly?
- Which species are hardest to classify? (Long-tail? Soundscape-only?)

**Future Directions**:
- After supervised baseline, should we pursue MAE pre-training, BEATS distillation, or alternative architectures?
- Can we use this pipeline for other bioacoustic datasets (ESC-50, AudioSet, Xeno-Canto)?

## Relationship to Other Concepts

- **Foundation**: This pipeline provides infrastructure for future experiments
- **MAE Pre-training** (deferred): Would reuse DataModule and preprocessing, add encoder-decoder architecture
- **BEATS Distillation**: Could use this pipeline as student model training loop
- **Soundscape Domain Shift**: Performance on soundscape test sets quantifies domain gap

## Related Experiments

- **Experiment 001**: BirdSet benchmark - established baseline performance for CNNs
- **Experiment 002**: This pipeline implementation - supervised ViT baseline
- **Experiment 003** (archived): Audio-MAE pre-training - deferred until Exp 002 validates infrastructure

## Next Steps

1. Complete Experiment 002 HSN training (2-4 hours)
2. Evaluate results: compare to BirdSet baselines (ConvNeXT: 0.38 cmAP)
3. Decision point:
   - If cmAP ≥ 0.30 and GPU util >80%: Scale to XCM
   - If cmAP < 0.30: Iterate on architecture/hyperparameters
   - If infrastructure issues: Debug data loading or memory
4. Document results and decide next research direction
