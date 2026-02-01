# Proposal: Supervised ViT Baseline

**Status**: approved
**Created**: 2026-01-31
**Author**: Murilo Gustineli

## Hypothesis

A Vision Transformer (ViT-Base) trained end-to-end on BirdSet XCM with on-the-fly mel-spectrogram generation can:
1. Achieve competitive multi-label classification performance (cmAP > 0.30 on HSN validation)
2. Maintain GPU utilization >80% during training with optimized data loading (persistent workers, prefetching)
3. Process batches efficiently without data loading bottlenecks (<10% overhead)

This establishes infrastructure for future self-supervised pre-training experiments.

## Background

Before implementing Audio-MAE pre-training (Experiment 003), we need a robust PyTorch Lightning training pipeline that:
- Handles BirdSet's multi-label classification task
- Generates mel-spectrograms on-the-fly to avoid storing preprocessed data
- Maximizes GPU utilization through efficient data loading
- Provides baseline supervised performance for comparison with self-supervised approaches

This experiment validates the infrastructure and provides a supervised baseline for evaluating Audio-MAE's effectiveness.

**Related concepts**:
- Vision Transformers for audio spectrograms (treating mel-spectrograms as images)
- Multi-label bird species classification on BirdSet
- On-the-fly audio preprocessing to minimize storage

## Method

### Approach

Train ViT-Base end-to-end on BirdSet HSN (quick iteration) then scale to XCM. The model processes mel-spectrograms generated on-the-fly by DataLoader workers, using:
- **Architecture**: ViT-Base with modified 1-channel patch embedding (instead of 3-channel RGB)
- **Loss**: BCEWithLogitsLoss for multi-label classification
- **Optimization**: AdamW with cosine annealing + linear warmup
- **Data**: Parallel workers generate mel-spectrograms using Kaldi fbank (following EAT encoder patterns)

### Setup

| Component | Details |
|-----------|---------|
| Data | **HSN**: ~5,460 train samples, 21 species (fast iteration)<br>**XCM**: ~160k train samples, ~9,736 species (production)<br>Map-style dataset (download to local cache ~90GB) |
| Compute | **GPU**: 1x RTX 6000 (24GB VRAM)<br>**CPUs**: 16 cores for DataLoader workers<br>**Memory**: 64GB RAM<br>**Time**: HSN ~2-4 hours, XCM ~24-48 hours |
| Dependencies | `timm>=0.9.0` (ViT), `wandb>=0.16.0` (logging), `pytorch-lightning>=2.0.0`, `torchaudio` (Kaldi fbank) |
| Code | `user/murilo/src/birdset_benchmark/`:<br>- `lightning/birdset_datamodule.py`<br>- `lightning/supervised_vit.py`<br>- `models/vit_audio.py`<br>- `preprocessing/mel_spectrogram.py` |

### Procedure

1. **Infrastructure Development (on HSN)**:
   - Implement `MelSpectrogramTransform` with Kaldi fbank + AudioSet normalization
   - Implement `BirdSetDataModule` with on-the-fly mel-spec generation in collate function
   - Implement `ViTAudio` with 1-channel patch embedding (averaging pretrained RGB weights)
   - Implement `SupervisedViTModule` with multi-label metrics (cmAP, F1, Top-K accuracy)

2. **Quick Validation (HSN, fast_dev_run=5)**:
   - Verify data loading: mel-spec shapes, label encoding, no NaN/Inf
   - Verify training loop: loss decreasing, metrics updating, no errors
   - Monitor GPU utilization in first epoch

3. **HSN Training (50 epochs)**:
   - Train with batch_size=32, 8 workers, persistent_workers=True, prefetch_factor=4
   - Monitor: GPU utilization, data loading overhead, cmAP convergence
   - Validate infrastructure: early stopping, checkpointing, W&B logging

4. **Scale to XCM**:
   - Switch dataset in config.yaml
   - Full training run (50 epochs with early stopping)
   - Compare HSN vs XCM convergence patterns

### Variables

- **Independent**:
  - Dataset (HSN → XCM)
  - DataLoader configuration (num_workers, persistent_workers, prefetch_factor)
  - Model architecture (ViT-Base fixed)

- **Dependent**:
  - cmAP (primary metric)
  - F1-macro, F1-micro
  - Top-1 and Top-5 accuracy
  - GPU utilization %
  - Data loading overhead (time per batch)

- **Controlled**:
  - Mel-spectrogram parameters (128 mels, 500 frames, 10ms shift)
  - AudioSet normalization (mean=-4.268, std=4.569)
  - Training hyperparameters (lr=1e-4, weight_decay=0.01, warmup=5 epochs)
  - Precision (16-mixed)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **cmAP** (Class-mean Average Precision) | BirdSet primary metric: macro-averaged AP across classes. Measures ranking quality for multi-label classification. |
| **F1-macro** | Macro-averaged F1 score: unweighted mean of per-class F1. Good for class-balanced evaluation. |
| **F1-micro** | Micro-averaged F1 score: global precision/recall. Biased toward frequent classes. |
| **Top-1 Accuracy** | Fraction of samples where highest-scoring prediction is a true label. |
| **Top-5 Accuracy** | Fraction of samples where at least one true label appears in top-5 predictions. |
| **GPU Utilization** | Percentage of time GPU is actively computing (target: >80%). |
| **Data Loading Overhead** | Time spent waiting for data vs. training (target: <10%). |

### Baseline

**HSN Dataset**:
- **BirdSet paper baselines**: Not directly available for HSN, but typical supervised CNN/ViT baselines achieve cmAP ~0.30-0.50 on small datasets
- **Random baseline**: cmAP ~0.048 (1/21 classes)
- **Infrastructure baseline**: Previous experiments used pre-extracted embeddings; this is first end-to-end training

**XCM Dataset**:
- **BirdSet paper**: Supervised baselines achieve cmAP ~0.25-0.35 on XCM (depends on architecture)
- **Random baseline**: cmAP ~0.0001 (1/9736 classes)

### Success Criteria

- **Confirm Infrastructure if**:
  - GPU utilization consistently >80% during training epochs
  - Data loading overhead <10% (measured via profiling or epoch timing)
  - Training completes without OOM errors on batch_size=32
  - W&B logging captures all metrics correctly
  - Checkpointing and early stopping work as expected

- **Confirm Performance if**:
  - **HSN**: cmAP > 0.30 on validation set (better than random ~0.048)
  - **XCM**: cmAP > 0.25 on validation set (competitive with BirdSet baselines)
  - Loss decreases monotonically without collapse
  - Top-5 accuracy > 60% (at least one true label in top-5 predictions)

- **Reject if**:
  - GPU utilization <50% (data loading bottleneck)
  - Training OOMs on batch_size=32 (need to reduce batch size or model size)
  - cmAP < 0.10 on HSN (worse than weak baseline, indicates bug)
  - Loss plateaus or increases (training instability)

## Limitations

- **Architecture**: ViT-Base is relatively small compared to state-of-the-art audio models; larger models may perform better but are computationally expensive
- **Preprocessing**: AudioSet normalization constants may not be optimal for BirdSet; computing dataset-specific stats would require full dataset pass
- **Data Loading**: Map-style dataset requires downloading full XCM (~90GB); streaming approach would reduce storage but adds complexity
- **Compute**: Single GPU training limits batch size and experiment throughput; multi-GPU would enable larger batches and faster iteration
- **Generalization**: Performance on HSN may not predict XCM performance due to domain differences (21 vs 9,736 classes, data distribution)
- **Comparison**: No direct baseline from BirdSet paper for supervised ViT; comparing against paper's CNN baselines may not be apples-to-apples
