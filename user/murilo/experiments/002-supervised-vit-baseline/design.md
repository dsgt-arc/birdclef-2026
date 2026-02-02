# Design: Supervised ViT Baseline

## Context

Building a robust PyTorch Lightning training pipeline for BirdSet before implementing Audio-MAE pre-training (Experiment 003). This infrastructure must:
- Handle multi-label classification with thousands of classes
- Generate mel-spectrograms on-the-fly to avoid storing preprocessed data
- Maximize GPU utilization through efficient data loading
- Provide clean abstractions for future experiments (MAE, fine-tuning, etc.)

Key constraints:
- Single GPU training (RTX 6000, 24GB VRAM)
- Map-style dataset preferred over streaming for simplicity
- Must follow existing BirdSet patterns (AudioSet normalization, Kaldi fbank)

## Goals

- Establish PyTorch Lightning pipeline: DataModule, LightningModule, metrics, callbacks
- Validate on-the-fly mel-spectrogram generation with minimal overhead
- Achieve GPU utilization >80% through optimized data loading
- Provide supervised baseline for comparing Audio-MAE performance
- Enable rapid iteration: HSN (fast) → XCM (production)

## Non-Goals

- State-of-the-art performance on BirdSet leaderboard (this is infrastructure validation)
- Multi-GPU distributed training (single GPU sufficient for now)
- Streaming dataset implementation (map-style simpler, deferred until needed)
- Dataset-specific normalization constants (use AudioSet for consistency)
- Extensive hyperparameter tuning (use reasonable defaults from ViT/MAE papers)

## Decisions

### Decision 1: Map-Style Dataset vs. Streaming

**Options considered:**
1. **Map-style**: Download full dataset to local cache, use standard PyTorch Dataset
   - Pros: Simpler, faster iteration, reliable checkpointing, easier debugging
   - Cons: Requires ~90GB disk space for XCM
2. **Streaming**: Load data on-demand with HuggingFace `load_dataset(streaming=True)`
   - Pros: No disk space needed, scales to arbitrarily large datasets
   - Cons: Complex worker coordination, custom checkpointing, slower iteration

**Decision**: Map-style dataset

**Rationale**: XCM size (~90GB) is manageable on PACE storage. Map-style is significantly simpler to implement and debug, with faster iteration during development. Streaming can be added later if needed for larger datasets.

### Decision 2: Mel-Spectrogram Generation Strategy

**Options considered:**
1. **Pre-compute and save**: Generate all mel-spectrograms upfront, save to disk
   - Pros: Fastest data loading (no compute during training)
   - Cons: Requires ~500GB+ storage, inflexible (can't change mel params without regenerating)
2. **On-the-fly in DataLoader**: Generate in `collate_fn` or worker processes
   - Pros: Minimal storage, flexible (easy to change params), leverages CPU parallelism
   - Cons: CPU overhead, requires tuning num_workers
3. **On-the-fly on GPU**: Generate mel-specs in first layer of model
   - Pros: Leverages GPU, no CPU bottleneck
   - Cons: Wastes GPU cycles, complicates model architecture

**Decision**: On-the-fly in DataLoader (option 2)

**Rationale**: CPU parallelism (8-16 workers) can generate mel-spectrograms fast enough to keep GPU saturated. Storage savings are significant. This matches patterns from existing audio models (EAT encoder) and provides flexibility for experimentation.

### Decision 3: Normalization Constants

**Options considered:**
1. **Compute BirdSet-specific stats**: Pass through training set, compute mean/std
   - Pros: Optimal for BirdSet distribution
   - Cons: Requires full dataset pass, breaks transfer learning from AudioSet-pretrained models
2. **Use AudioSet constants** (mean=-4.268, std=4.569)
   - Pros: Proven to work, matches existing models (EAT, SSAST), enables transfer learning
   - Cons: May be suboptimal for BirdSet

**Decision**: Use AudioSet constants

**Rationale**: Existing BirdSet encoders use AudioSet normalization successfully. Computing dataset-specific stats adds engineering complexity with unclear benefit. Can revisit if performance is poor.

### Decision 4: ViT Patch Embedding Modification

**Options considered:**
1. **Average RGB weights**: Initialize 1-channel conv by averaging pretrained 3-channel weights
   - Pros: Simple, preserves some pretrained information
   - Cons: Loses color channel structure (not relevant for spectrograms)
2. **Random initialization**: Start from scratch for 1-channel conv
   - Pros: Clean slate for audio
   - Cons: Loses pretrained knowledge, slower convergence
3. **Replicate channel**: Input (1, H, W) → (3, H, W) by repeating, keep 3-channel conv
   - Pros: Full pretrained weights preserved
   - Cons: Wastes parameters (2/3 of first conv), conceptually awkward

**Decision**: Average RGB weights (option 1)

**Rationale**: Standard approach for adapting RGB-pretrained models to grayscale. Provides reasonable initialization while adapting architecture for single-channel input. Used successfully in audio domain (e.g., AST, EAT).

### Decision 5: Development Workflow (HSN → XCM)

**Options considered:**
1. **Develop directly on XCM**: Full-scale from the start
   - Pros: No transfer issues, final dataset from beginning
   - Cons: Slow iteration (24-48 hour runs), expensive debugging
2. **HSN → XCM**: Develop on small HSN, scale to large XCM
   - Pros: Fast iteration (2-4 hour runs), quick debugging, efficient development
   - Cons: Potential transfer issues, need to validate on both datasets

**Decision**: HSN → XCM workflow (option 2)

**Rationale**: HSN provides ~100x faster iteration (minutes vs hours per epoch), enabling rapid debugging and validation. Infrastructure bugs (data loading, GPU OOM, etc.) will surface quickly on HSN. Scaling to XCM is straightforward (change dataset name in config).

### Decision 6: Validation Strategy (HSN Dataset Splits)

**Options considered:**
1. **Custom train/val/test split**: Manually split training data with stratification
   - Pros: Control over split ratios, guaranteed species coverage
   - Cons: Extra engineering, may not match BirdSet evaluation protocol
2. **Use BirdSet pre-defined splits**: Use `train` for training, `test_5s` for validation
   - Pros: Follows BirdSet conventions, no custom code needed, species coverage already ensured
   - Cons: Naming is confusing (`test_5s` used as validation)

**Decision**: Use BirdSet pre-defined splits (option 2)

**Rationale**: BirdSet HSN comes with pre-defined splits that already ensure species coverage:
- `train`: 5,460 samples, 21 species
- `test_5s`: 12,000 samples, 21 species (used for validation)
- All 21 species appear in both splits

No custom splitting is needed. The `test_5s` split serves as our validation set during training, following BirdSet conventions.

## Architecture / Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  BirdSetDataModule  │  (PyTorch Lightning DataModule)
│  (lightning/)       │
└──────────┬──────────┘
           │
           │ setup() loads HuggingFace datasets
           ▼
┌─────────────────────┐
│  BirdSetDataset     │  (PyTorch Dataset, from torch_data.py)
│  (existing)         │  - Loads audio from disk
└──────────┬──────────┘  - Resamples to 32kHz
           │             - Pads/crops to 5 seconds
           │             - Returns raw waveform
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  DataLoader (8 workers, persistent, prefetch_factor=4)          │
│  ├─ Worker 1: Load audio → waveform                             │
│  ├─ Worker 2: Load audio → waveform                             │
│  └─ ...                                                         │
│                                                                 │
│  Collate Function (_collate_with_melspec):                      │
│  ├─ Stack waveforms into batch                                  │
│  ├─ Apply MelSpectrogramTransform to each sample                │
│  │   └─ Kaldi fbank (128 mels, 500 frames, 10ms shift)          │
│  │   └─ AudioSet normalization (mean=-4.268, std=4.569)         │
│  └─ Return (mel_specs, labels)                                  │
└──────────┬──────────────────────────────────────────────────────┘
           │ (batch, 1, 128, 500), (batch, num_classes)
           ▼
┌─────────────────────┐
│  SupervisedViTModule│  (PyTorch Lightning Module)
│  (lightning/)       │
└──────────┬──────────┘
           │
           │ forward()
           ▼
┌─────────────────────┐
│  ViTAudio           │  (Model)
│  (models/)          │
│  ├─ ViT-Base        │  (timm, patch_size=16)
│  │  └─ Modified:    │  1-channel patch embedding (averaged from RGB)
│  └─ Linear Head     │  (768 → num_classes)
└──────────┬──────────┘
           │
           │ logits
           ▼
┌─────────────────────┐
│  Loss & Metrics     │
│  ├─ BCEWithLogitsLoss
│  ├─ cmAP (macro AP) │  BirdSet primary metric
│  ├─ F1-macro/micro  │
│  └─ Top-1/5 accuracy│
└─────────────────────┘

┌─────────────────────┐
│  Callbacks          │
│  ├─ ModelCheckpoint │  Save top-3 by val/cmap
│  ├─ EarlyStopping   │  Stop if no improvement (patience=10)
│  └─ LRMonitor       │  Log learning rate
└─────────────────────┘

┌─────────────────────┐
│  WandbLogger        │  Log to Weights & Biases
│  ├─ Metrics         │  train/val loss, cmAP, F1, accuracy
│  ├─ System          │  GPU utilization, memory
│  └─ Config          │  Hyperparameters, model config
└─────────────────────┘
```

### Code Organization

```
user/murilo/src/birdset_benchmark/
├── preprocessing/
│   └── mel_spectrogram.py       # MelSpectrogramTransform (Kaldi fbank)
├── models/
│   └── vit_audio.py             # ViTAudio (1-channel ViT-Base)
└── lightning/
    ├── birdset_datamodule.py    # DataModule with on-the-fly mel-spec
    └── supervised_vit.py        # LightningModule with multi-label metrics

user/murilo/experiments/002-supervised-vit-baseline/
├── proposal.md                  # This document
├── tasks.md                     # Implementation checklist
├── design.md                    # Technical decisions (this file)
├── config.yaml                  # Hyperparameters
├── train.py                     # Training script
└── sbatch/
    └── train.sbatch             # SLURM batch script
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU bottleneck in mel-spec generation | High (GPU underutilized, slow training) | Use 8-16 workers, persistent_workers=True, prefetch_factor=4. Profile and tune num_workers. |
| OOM with batch_size=32 | High (training fails) | Monitor GPU memory in fast_dev_run. Reduce batch_size or use gradient accumulation if needed. |
| Slow convergence on XCM (9,736 classes) | Medium (long training time) | Start with HSN to validate infrastructure. Use early stopping. Consider pre-training (Experiment 003). |
| AudioSet normalization suboptimal for BirdSet | Low (minor performance impact) | Accept for now; revisit if cmAP is unexpectedly low (<0.20 on XCM). |
| Map-style dataset fills disk | Low (PACE has sufficient storage) | Monitor disk usage. Can switch to streaming if needed. |
| HSN → XCM transfer issues | Low (both are BirdSet datasets) | Validate on XCM with fast_dev_run before full run. |

## Open Questions

- **Optimal num_workers**: Need to profile on PACE hardware. Start with 8, tune if GPU utilization <80%.
- **Batch size vs. gradient accumulation**: If batch_size=32 OOMs, should we reduce batch_size or use gradient accumulation? (Prefer accumulation to maintain effective batch size.)
- **Early stopping patience**: Is 10 epochs sufficient for XCM, or should we increase to 15-20 given larger dataset?
- **Learning rate schedule**: Cosine annealing vs. ReduceLROnPlateau? (Using cosine as default; ReduceLR might be better for XCM.)
- **Checkpoint frequency**: Save every epoch or only top-3? (Only top-3 to save disk space.)
- **HSN performance correlation**: Does HSN cmAP predict XCM cmAP? (Will find out after both runs.)

## Implementation Notes

### MelSpectrogramTransform

Following EAT encoder patterns:
```python
fbank = torchaudio.compliance.kaldi.fbank(
    waveform, htk_compat=True, sample_frequency=32000,
    num_mel_bins=128, frame_shift=10
)
normalized = (fbank - mean) / (std * 2)
```

Key details:
- `htk_compat=True`: Match HTK Mel filterbank (standard in audio)
- `frame_shift=10`: 10ms hop length (100 FPS)
- Normalize by `std * 2` (EAT pattern, more aggressive than `std`)
- Target 500 frames → ~5 seconds at 100 FPS

### ViTAudio Initialization

```python
# Load pretrained ViT-Base (ImageNet)
backbone = timm.create_model('vit_base_patch16_224', pretrained=True)

# Replace 3-channel conv with 1-channel
old_conv = backbone.patch_embed.proj
new_conv = nn.Conv2d(1, 768, kernel_size=16, stride=16)

# Initialize by averaging RGB channels
new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
new_conv.bias[:] = old_conv.bias
```

### DataLoader Optimization

```python
DataLoader(
    batch_size=32,
    num_workers=8,              # Parallel mel-spec generation
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=4,          # 4 batches prefetched per worker
    pin_memory=True,            # Faster GPU transfer
    drop_last=True,             # Stable batch sizes for training
)
```

Expected throughput: ~10-20 batches/sec on RTX 6000 (depends on CPU speed).
