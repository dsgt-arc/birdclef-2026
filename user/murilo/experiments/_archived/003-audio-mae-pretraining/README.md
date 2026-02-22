# Experiment 003: Audio-MAE Pre-training

**Status**: Proposed (awaiting Experiment 002 completion)

## Overview

Self-supervised pre-training of ViT-Base encoder using Masked Autoencoders (MAE) on BirdSet XCM. The model learns to reconstruct masked mel-spectrogram patches, forcing it to learn semantic audio representations without labels.

**Key Idea**: Mask 75% of spectrogram patches randomly, train encoder-decoder to reconstruct the masked content. After pre-training, use encoder for downstream tasks (zero-shot retrieval, fine-tuning).

## Prerequisites

- ✅ Experiment 002 completed (infrastructure validated)
- ✅ Dependencies installed (timm, wandb, einops)
- ✅ BirdSet XCM dataset accessible

## Quick Start

### 1. Quick Validation (Fast Dev Run)

Test the MAE pipeline with 5 batches on HSN:

```bash
cd experiments/003-audio-mae-pretraining
python train.py --fast-dev-run
```

This should:
- Load HSN dataset
- Generate mel-spectrograms
- Mask 75% of patches randomly
- Forward pass through encoder-decoder
- Compute reconstruction loss
- Log visualizations to W&B
- Complete in ~2-5 minutes

### 2. HSN Pre-training (Quick Test, 20 epochs)

Validate full pipeline on HSN:

```bash
python train.py --config config.yaml --max-epochs 20
```

Expected:
- Loss decreases from ~1.5 to <1.0
- Reconstructions show structure by epoch 10
- Training time: ~2-4 hours
- W&B logs reconstruction images every 5 epochs

### 3. XCM Pre-training (Production, 100 epochs)

Update `config.yaml`:
```yaml
data:
  dataset_name: "XCM"
training:
  max_epochs: 100
```

Submit SLURM job:
```bash
cd sbatch
sbatch train.sbatch
```

Monitor progress:
```bash
# Check SLURM logs
tail -f ../outputs/slurm/exp003-audio-mae-*.out

# Check W&B dashboard
# https://wandb.ai/<your-username>/birdset-audio-mae
```

Expected:
- Training time: 48-72 hours
- Final MSE loss: <0.5
- High-quality reconstructions by epoch 100

## Architecture

```
Input: Mel-spectrogram (1, 128, 500)
  ↓
Patchify (16x16 patches) → 248 patches
  ↓
Random Masking (75%) → Keep 62 visible, mask 186
  ↓
Encoder (ViT-Base) → Process only 62 visible patches
  ↓
Decoder (4-layer Transformer) → Reconstruct all 248 patches
  ↓
Loss: MSE on 186 masked patches only
```

**Encoder**: ViT-Base (12 layers, 768 dim, 12 heads)
**Decoder**: Lightweight transformer (4 layers, 512 dim, 8 heads)
**Masking**: 75% random (following MAE paper)
**Target**: Patch-normalized spectrograms (each patch normalized independently)

## Files

| File | Purpose |
|------|---------|
| `proposal.md` | Hypothesis, method, success criteria |
| `design.md` | Technical decisions, architecture diagrams |
| `tasks.md` | Implementation checklist (not started) |
| `config.yaml` | MAE hyperparameters |
| `train.py` | Training script |
| `sbatch/train.sbatch` | SLURM batch script |

## Key Features

- **Self-supervised**: No labels needed, learn from reconstruction
- **Aggressive masking**: 75% masked forces semantic understanding (not just local copying)
- **Patch normalization**: Each patch normalized independently for training stability
- **Lightweight decoder**: Only 4 layers (decoder discarded after pre-training)
- **W&B visualization**: Log [Original | Masked | Reconstructed] images every 5 epochs
- **Encoder reuse**: Save encoder for downstream tasks (fine-tuning, retrieval)

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Mask ratio | 0.75 | MAE default, proven effective |
| Batch size | 64 | 2x supervised (no labels, more fits in memory) |
| Learning rate | 1.5e-4 | Higher than supervised (MAE paper) |
| Weight decay | 0.05 | Higher than supervised (MAE paper) |
| Warmup epochs | 10 | Longer warmup for stability |
| Max epochs | 100 | 2x supervised (self-supervised needs more) |
| Decoder depth | 4 | Lightweight (MAE recommendation) |
| Decoder dim | 512 | 2/3 of encoder dim |

## Success Criteria

**Infrastructure**:
- ✅ Training completes without OOM (batch_size=64)
- ✅ W&B logging captures reconstruction images
- ✅ Loss decreases monotonically (no spikes or NaN)

**Reconstruction Quality**:
- ✅ Final MSE loss <0.5 (50% reduction from random init)
- ✅ Reconstructions show frequency structure (not gray blur)
- ✅ Reconstructions show temporal patterns
- ✅ Masked regions filled plausibly

**Visual Inspection** (W&B images over epochs):
- Epoch 1-5: Gray blur (untrained)
- Epoch 10-20: Structure emerging
- Epoch 50-100: High-quality reconstructions

## Troubleshooting

### Reconstructions are Gray Blur (not learning)
```yaml
# Try reducing mask ratio (easier task)
model:
  mask_ratio: 0.60  # Instead of 0.75

# Or increase decoder capacity
model:
  decoder_depth: 6  # Instead of 4
```

### GPU OOM
```yaml
# Reduce batch size
data:
  batch_size: 48  # Instead of 64
```

### Loss Not Decreasing
```bash
# Check patch normalization is correct
# Check loss computed only on masked patches (not visible)
# Try lower learning rate
training:
  lr: 1.0e-4  # Instead of 1.5e-4
```

### W&B Images Not Appearing
```bash
# Check MAEImageLogger is added to callbacks
# Check validation dataloader is not empty
# Check log_every_n_epochs setting in config
```

## Downstream Tasks (Future Experiments)

After pre-training, the encoder can be used for:

1. **Fine-tuning**: Initialize classification head from MAE encoder, fine-tune on labeled data
2. **Zero-shot retrieval**: Use encoder embeddings for query-by-example search
3. **Linear probe**: Freeze encoder, train linear classifier on embeddings
4. **Transfer learning**: Pre-train on XCM, fine-tune on other bird datasets

Compare MAE pre-trained encoder vs:
- Random initialization (Exp 002 baseline)
- ImageNet pre-training (transfer from vision)
- Supervised pre-training on XCM

## Visualization Guide

W&B logs reconstruction images with 3 columns:

- **Original**: Ground truth mel-spectrogram
- **Masked**: Original with 75% patches replaced by gray
- **Reconstructed**: Model's prediction for all patches

**Good reconstruction**:
- Frequency structure preserved (horizontal patterns)
- Temporal patterns preserved (vertical patterns)
- Masked regions filled plausibly (smooth transitions)

**Bad reconstruction**:
- Gray blur (averaging, not learning)
- Random noise (no structure)
- Discontinuities at patch boundaries

## Expected Results

**Training Curve**:
- Epoch 0-10: Rapid loss decrease (1.5 → 0.8)
- Epoch 10-50: Steady decrease (0.8 → 0.6)
- Epoch 50-100: Slow convergence (0.6 → 0.5)

**Reconstruction Quality**:
- Epoch 1: Gray blur
- Epoch 10: Hints of structure
- Epoch 20: Clear structure
- Epoch 50: High quality
- Epoch 100: Nearly perfect on simple spectrograms

## References

- **MAE**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)
- **AudioMAE**: "Masked Autoencoders that Listen" (Huang et al., 2022)
- **SSAMBA**: Self-supervised learning for bird vocalization
- **BirdSet**: https://github.com/DBD-research-group/BirdSet

## Code Structure

```
src/birdclef/
├── models/
│   ├── vit_audio.py          # Encoder (reuse from Exp 002)
│   └── mae_decoder.py        # NEW: Transformer decoder
└── lightning/
    ├── birdset_datamodule.py # Reuse from Exp 002
    ├── audio_mae.py          # NEW: MAE Lightning module
    └── callbacks.py          # NEW: MAEImageLogger
```

## Next Steps

After successful pre-training:

1. **Quantitative evaluation**:
   - Extract encoder embeddings for validation set
   - Compute reconstruction MSE across epochs
   - Measure encoder quality via linear probe

2. **Qualitative evaluation**:
   - Visual inspection of best/worst reconstructions
   - t-SNE/UMAP visualization of embeddings
   - Compare to supervised encoder (Exp 002)

3. **Downstream experiments**:
   - Fine-tune on classification (compare to supervised baseline)
   - Zero-shot retrieval experiments
   - Transfer learning to other datasets

4. **Documentation**:
   - Write results.md with findings
   - Save encoder checkpoints to artifacts/
   - Update Status to "completed"
