# Design: Audio-MAE Pre-training

## Context

After validating the training infrastructure with supervised ViT (Experiment 002), we now implement self-supervised pre-training using Masked Autoencoders (MAE). This approach learns representations from unlabeled audio by reconstructing masked spectrogram patches.

**Motivation**:
- BirdSet has long-tail species distribution (many rare species with few examples)
- Self-supervised pre-training can leverage all available audio (no labels needed)
- MAE has proven effective in vision (He et al., 2022) and audio (AudioMAE)

**Constraints**:
- Build on Experiment 002 infrastructure (reuse DataModule, mel-spec generation)
- Single GPU training (RTX 6000, 24GB VRAM)
- Must visualize reconstructions to validate learning (not just monitor loss)

## Goals

- Pre-train ViT-Base encoder on XCM using masked autoencoding
- Achieve visually coherent mel-spectrogram reconstructions
- Validate self-supervised learning on BirdSet dataset
- Produce encoder checkpoints for downstream tasks (future experiments)

## Non-Goals

- Downstream task evaluation (zero-shot retrieval, fine-tuning) - deferred to future experiments
- Multi-GPU distributed training - single GPU sufficient for now
- Hyperparameter tuning (mask ratio, decoder depth) - use proven defaults from MAE/AudioMAE papers
- Pre-training on external datasets (AudioSet, Xeno-Canto) - focus on BirdSet XCM only

## Decisions

### Decision 1: Mask Ratio

**Options considered:**
1. **75% masking** (MAE default for vision)
   - Pros: Proven in MAE paper, forces semantic understanding
   - Cons: May be too aggressive for audio (more redundant than images)
2. **80% masking** (AudioMAE default for audio)
   - Pros: AudioMAE found optimal for spectrograms
   - Cons: Very aggressive, may be harder to reconstruct
3. **50% masking** (conservative)
   - Pros: Easier task, faster convergence
   - Cons: May not force semantic understanding, too easy

**Decision**: 75% masking

**Rationale**: Split the difference between MAE (75%) and AudioMAE (80%). Aggressive enough to force semantic learning, but not so extreme that reconstruction is impossible. Can tune in future experiments if needed.

### Decision 2: Decoder Architecture

**Options considered:**
1. **Lightweight transformer** (4 layers, 512 dim, 8 heads)
   - Pros: Fast, proven in MAE paper, sufficient for reconstruction
   - Cons: May limit reconstruction quality
2. **Deep decoder** (8+ layers, 768 dim, matches encoder)
   - Pros: More capacity for high-quality reconstruction
   - Cons: Slower training, overfitting risk, wastes computation (decoder discarded after pre-training)
3. **Convolutional decoder** (U-Net style)
   - Pros: Spatially-aware reconstruction
   - Cons: Breaks transformer architecture consistency, harder to implement

**Decision**: Lightweight transformer (option 1)

**Rationale**: MAE paper shows decoder depth doesn't critically affect encoder quality. Lightweight decoder (4 layers, 512 dim, 8 heads) is fast and sufficient. Decoder is discarded after pre-training, so investing in deep decoder is wasteful.

### Decision 3: Reconstruction Target

**Options considered:**
1. **Raw mel-spectrogram patches**
   - Pros: Simple, direct reconstruction
   - Cons: High variance across patches (loud vs quiet), training instability
2. **Patch-normalized spectrograms** (normalize each patch by mean/std)
   - Pros: Stable training (MAE paper recommendation), removes global intensity variation
   - Cons: Loses global loudness information
3. **Log-mel spectrograms**
   - Pros: Better dynamic range for audio
   - Cons: Already using Kaldi fbank (log-like), additional log may over-compress

**Decision**: Patch-normalized spectrograms (option 2)

**Rationale**: MAE paper explicitly recommends patch normalization for training stability. Each patch is normalized by its own mean/std before computing loss. This removes global loudness variation but focuses on local structure (frequency/temporal patterns).

### Decision 4: Loss Computation

**Options considered:**
1. **MSE on all patches** (visible + masked)
   - Pros: Simple, uses all information
   - Cons: Encoder could cheat by memorizing visible patches, wastes computation
2. **MSE on masked patches only** (MAE default)
   - Pros: Forces encoder to infer masked content, prevents memorization
   - Cons: Slightly more complex implementation
3. **Perceptual loss** (e.g., SSIM, feature matching)
   - Pros: May produce better perceptual quality
   - Cons: Complex, slower, unclear benefit for spectrograms

**Decision**: MSE on masked patches only (option 2)

**Rationale**: Standard in MAE/AudioMAE papers. Computing loss only on masked patches forces encoder to learn semantic representations (can't just copy visible patches). MSE is simple and effective for normalized targets.

### Decision 5: Masking Strategy

**Options considered:**
1. **Random masking** (MAE default)
   - Pros: Simple, no spatial bias, proven in MAE
   - Cons: May create easy tasks (isolated masked patches)
2. **Block masking** (mask contiguous regions)
   - Pros: Harder task (large gaps to fill), may force global understanding
   - Cons: Spatial bias, harder to reconstruct, slower convergence
3. **Time-frequency masking** (SpecAugment-style)
   - Pros: Targets specific audio structure (time/frequency bands)
   - Cons: Complex, may not be optimal for MAE (designed for augmentation, not pre-training)

**Decision**: Random masking (option 1)

**Rationale**: MAE and AudioMAE both use random masking successfully. Simple to implement, no spatial bias, allows encoder to see diverse contexts. Block masking might be explored in future work.

### Decision 6: Positional Embeddings

**Options considered:**
1. **Learned positional embeddings** (standard ViT)
   - Pros: Flexible, learns optimal encoding
   - Cons: Fixed resolution (can't extrapolate to longer spectrograms)
2. **Sinusoidal positional embeddings**
   - Pros: Interpolatable, works for variable lengths
   - Cons: Less flexible than learned
3. **Relative positional embeddings** (Transformer-XL style)
   - Pros: Translation invariant
   - Cons: More complex, unclear benefit for spectrograms

**Decision**: Learned positional embeddings (option 1, inherited from ViT-Base)

**Rationale**: Using timm's ViT-Base which has learned positional embeddings. Keep it simple and consistent with Experiment 002. All spectrograms are fixed size (128x500), so extrapolation not needed.

## Architecture / Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                   Audio-MAE Pre-training                        │
└─────────────────────────────────────────────────────────────────┘

Input: Mel-spectrogram (batch, 1, 128, 500)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│  Patchify + Normalize                                           │
│  ├─ Split into 16x16 patches: (batch, num_patches, patch_dim)   │
│  │   num_patches = (128/16) * (500/16) = 8 * 31 = 248 patches   │
│  │   patch_dim = 16 * 16 * 1 = 256                              │
│  └─ Normalize each patch: (patch - patch_mean) / patch_std      │
└──────────┬──────────────────────────────────────────────────────┘
           │ patches_normalized: (batch, 248, 256)
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Random Masking (75%)                                           │
│  ├─ Shuffle patches randomly                                    │
│  ├─ Keep first 25% (62 patches) as visible                      │
│  ├─ Mask remaining 75% (186 patches)                            │
│  └─ Return:                                                     │
│      - x_visible: (batch, 62, 256)                              │
│      - mask: (batch, 248) binary mask [1=masked, 0=visible]     │
│      - ids_restore: (batch, 248) indices to restore order       │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Encoder (ViT-Base)                                             │
│  ├─ Patch embedding: (batch, 62, 256) → (batch, 62, 768)        │
│  ├─ Add positional embeddings (only for visible patches)        │
│  ├─ 12 Transformer blocks (768 dim, 12 heads, 3072 MLP)         │
│  └─ Output: (batch, 62, 768)                                    │
└──────────┬──────────────────────────────────────────────────────┘
           │ encoded_visible: (batch, 62, 768)
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Decoder Embedding                                              │
│  ├─ Project encoder output: (batch, 62, 768) → (batch, 62, 512) │
│  ├─ Create mask tokens: (batch, 186, 512)                       │
│  ├─ Concatenate: [encoded_visible; mask_tokens]                 │
│  ├─ Unshuffle using ids_restore                                 │
│  ├─ Add positional embeddings (full set, 248 patches)           │
│  └─ Output: (batch, 248, 512)                                   │
└──────────┬──────────────────────────────────────────────────────┘
           │ decoder_input: (batch, 248, 512)
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Decoder (Lightweight Transformer)                              │
│  ├─ 4 Transformer blocks (512 dim, 8 heads, 2048 MLP)           │
│  └─ Output: (batch, 248, 512)                                   │
└──────────┬──────────────────────────────────────────────────────┘
           │ decoded: (batch, 248, 512)
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Reconstruction Head                                            │
│  ├─ Linear: (batch, 248, 512) → (batch, 248, 256)               │
│  └─ Reshape: (batch, 248, 256) → (batch, 1, 128, 500)           │
└──────────┬──────────────────────────────────────────────────────┘
           │ reconstructed: (batch, 1, 128, 500)
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Loss Computation (MSE on masked patches only)                  │
│  ├─ Extract masked patches from reconstruction and target       │
│  ├─ Compute MSE: mean((reconstructed_masked - target_masked)²)  │
│  └─ Backpropagate only through masked patches                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Visualization (MAEImageLogger)                                 │
│  ├─ Create masked version: fill masked patches with gray        │
│  ├─ Denormalize reconstruction (reverse patch normalization)    │
│  ├─ Log to W&B: [Original | Masked | Reconstructed]             │
│  └─ Generate figure with 4 samples per validation epoch         │
└─────────────────────────────────────────────────────────────────┘
```

### Code Organization

```
user/murilo/src/birdset_benchmark/
├── models/
│   ├── vit_audio.py              # Reuse from Exp 002
│   └── mae_decoder.py            # NEW: Transformer decoder
├── lightning/
│   ├── birdset_datamodule.py     # Reuse from Exp 002 (no labels needed)
│   ├── audio_mae.py              # NEW: MAE Lightning module
│   └── callbacks.py              # NEW: MAEImageLogger

user/murilo/experiments/003-audio-mae-pretraining/
├── proposal.md
├── design.md
├── tasks.md
├── README.md
├── config.yaml                   # NEW: MAE hyperparameters
├── train.py                      # NEW: MAE training script
└── sbatch/
    └── train.sbatch              # NEW: SLURM batch script
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Reconstructions are gray blur (model averaging) | High (pre-training failed, encoder not learning) | Monitor visualizations from epoch 1. If blur persists >20 epochs, debug normalization or reduce mask ratio. |
| Loss doesn't decrease | High (training not working) | Check: (1) patch normalization correct, (2) loss computed only on masked patches, (3) learning rate not too low/high. |
| OOM with batch_size=64 | Medium (need smaller batches) | Reduce batch_size to 32 or 48. MAE doesn't need labels, so larger batches are helpful but not critical. |
| Patch normalization removes useful signal | Low (trade-off for stability) | Accept for now; MAE paper shows this works. Could try without normalization if results are poor. |
| Decoder too weak (4 layers insufficient) | Low (encoder quality may suffer) | Monitor reconstruction quality. If very poor after 100 epochs, increase decoder depth to 6-8 layers. |
| 75% masking too aggressive | Medium (task too hard) | If loss doesn't decrease below 1.0 after 50 epochs, reduce mask ratio to 60% or 50%. |
| Encoder doesn't generalize to downstream tasks | High (pre-training not useful) | Defer to future experiments. This experiment only validates reconstruction, not downstream performance. |

## Open Questions

- **Optimal mask ratio**: Is 75% best for audio spectrograms, or would 60-80% be better? (Requires ablation study.)
- **Patch normalization impact**: Does normalizing patches remove critical global intensity information? (Acceptable trade-off for stability per MAE paper.)
- **Decoder depth**: Would deeper decoder (6-8 layers) improve encoder quality, or is 4 layers sufficient? (MAE paper suggests 4 is enough.)
- **Reconstruction quality metric**: Besides visual inspection, should we compute SSIM or other perceptual metrics? (MSE sufficient for now.)
- **Positional encoding**: Are learned positional embeddings optimal for spectrograms, or would sinusoidal be better? (Learned is standard, keep it simple.)
- **Warmup duration**: Should we use longer warmup (>10 epochs) for MAE stability? (Start with 10, increase if loss spikes.)

## Implementation Notes

### Random Masking

```python
def random_masking(x, mask_ratio=0.75):
    """
    Random masking following MAE paper.

    Args:
        x: (batch, num_patches, embed_dim) - patch embeddings
        mask_ratio: Fraction of patches to mask (default 0.75)

    Returns:
        x_masked: (batch, num_visible, embed_dim) - only visible patches
        mask: (batch, num_patches) - binary mask (1=masked, 0=visible)
        ids_restore: (batch, num_patches) - indices to restore order
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    # Random shuffle
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep first subset (visible patches)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # Binary mask: 0=visible, 1=masked
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore
```

### Patch Normalization

```python
def patchify_and_normalize(images, patch_size=16):
    """
    Convert images to normalized patches (MAE paper recommendation).

    Args:
        images: (batch, 1, H, W) - mel-spectrograms
        patch_size: Patch size (default 16)

    Returns:
        patches: (batch, num_patches, patch_dim) - normalized patches
    """
    # Patchify using einops
    patches = rearrange(
        images,
        'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
        p1=patch_size,
        p2=patch_size
    )

    # Normalize each patch independently
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True) + 1e-6
    patches_normalized = (patches - mean) / std

    return patches_normalized
```

### MAE Decoder

```python
class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,         # Encoder output dim
        decoder_embed_dim=512, # Decoder internal dim
        decoder_depth=4,       # Number of transformer blocks
        decoder_num_heads=8,   # Attention heads
        num_patches=248,       # Total patches (8 * 31)
        patch_size=16,         # Patch size for reconstruction head
    ):
        super().__init__()

        # Project encoder output to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Positional embeddings (full set for all patches)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim)
        )

        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(  # timm.models.vision_transformer.Block
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
            )
            for _ in range(decoder_depth)
        ])

        # Reconstruction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * 1  # Predict 16x16x1 patch
        )
```

### W&B Visualization

```python
class MAEImageLogger(Callback):
    """Log MAE reconstructions to W&B."""

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get batch from validation
        batch = next(iter(trainer.val_dataloaders))
        mel_specs, _ = batch  # Ignore labels (unused for MAE)
        mel_specs = mel_specs[:4].to(pl_module.device)

        # Forward pass
        with torch.no_grad():
            loss, reconstructed, mask = pl_module(mel_specs)

        # Create masked version (fill masked patches with 0.5 gray)
        masked_imgs = apply_mask_to_image(mel_specs, mask, patch_size=16)

        # Denormalize reconstructions (reverse patch normalization)
        reconstructed = denormalize_patches(reconstructed, patch_size=16)

        # Create figure: [Original | Masked | Reconstructed]
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        for i in range(4):
            axes[i, 0].imshow(mel_specs[i, 0].cpu(), cmap='viridis', aspect='auto')
            axes[i, 0].set_title('Original')
            axes[i, 1].imshow(masked_imgs[i, 0].cpu(), cmap='viridis', aspect='auto')
            axes[i, 1].set_title('Masked (75%)')
            axes[i, 2].imshow(reconstructed[i, 0].cpu(), cmap='viridis', aspect='auto')
            axes[i, 2].set_title('Reconstructed')

        # Log to W&B
        trainer.logger.experiment.log({
            "reconstructions": wandb.Image(fig),
            "epoch": trainer.current_epoch,
        })
        plt.close(fig)
```

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Mask ratio | 0.75 | MAE default, proven effective |
| Decoder depth | 4 | MAE recommendation (lightweight) |
| Decoder embed dim | 512 | 2/3 of encoder dim (standard) |
| Decoder heads | 8 | Half of encoder heads (12 → 8) |
| Learning rate | 1.5e-4 | Higher than supervised (MAE paper) |
| Weight decay | 0.05 | Higher than supervised (MAE paper) |
| Warmup epochs | 10 | Longer warmup for stability |
| Batch size | 64 | 2x supervised (no labels, fits in memory) |
| Max epochs | 100 | 2x supervised (self-supervised needs more) |
| Patch size | 16 | Match supervised ViT |

## Success Metrics

**Quantitative**:
- MSE loss < 0.5 after 100 epochs (50% reduction from random init ~1.0)
- Loss decreases monotonically (no divergence or plateau >10 epochs)

**Qualitative**:
- Reconstructions show clear frequency structure (not gray blur)
- Temporal patterns visible (not just vertical averaging)
- Masked regions filled plausibly (resemble original)
- By epoch 20: structure visible (early signal of learning)
- By epoch 100: high-quality reconstructions (indistinguishable from original in some cases)

**Infrastructure**:
- W&B logging captures reconstruction images every 5 epochs
- Training completes without OOM (batch_size=64)
- GPU utilization >70% (slightly lower than supervised due to masking overhead)
