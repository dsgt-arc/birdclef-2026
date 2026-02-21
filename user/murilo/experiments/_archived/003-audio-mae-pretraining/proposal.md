# Proposal: Audio-MAE Pre-training

**Status**: proposed
**Created**: 2026-01-31
**Author**: Murilo Gustineli

## Hypothesis

A Masked Autoencoder (MAE) pre-trained on BirdSet XCM using 75% masking can:
1. Learn high-quality audio representations through self-supervised reconstruction
2. Produce visually coherent mel-spectrogram reconstructions (not just averaging)
3. Achieve lower reconstruction MSE loss than random initialization baseline
4. Provide embeddings useful for downstream tasks (zero-shot retrieval, fine-tuning)

This validates the MAE pre-training approach for audio before fine-tuning for classification or retrieval.

## Background

Self-supervised pre-training has proven effective for learning generalizable representations in vision (MAE) and audio (AudioMAE, SSAMBA). For BirdSet:
- **Current bottleneck**: Limited labeled data for rare species (long-tail distribution)
- **Opportunity**: Abundant unlabeled audio can be leveraged via self-supervision
- **Approach**: Masked Autoencoding - mask 75% of spectrogram patches, train model to reconstruct

**Key advantages**:
- No labels required → use full XCM dataset (~160k recordings)
- Forces model to learn semantic audio features (not just local patterns)
- Encoder can be used for zero-shot retrieval or fine-tuned for classification

**Related concepts**:
- Masked Autoencoders (MAE) for vision (He et al., 2022)
- Audio Masked Autoencoders (AudioMAE) for audio spectrograms
- SSAMBA: Self-supervised learning for bird vocalization

**Prerequisites**: Experiment 002 (Supervised ViT) validates the training pipeline infrastructure.

## Method

### Approach

Train an encoder-decoder architecture on XCM to reconstruct masked mel-spectrogram patches:

1. **Encoder** (ViT-Base): Processes only visible patches (25% of total)
2. **Decoder** (Lightweight transformer): Reconstructs full spectrogram from encoded visible patches + mask tokens
3. **Loss**: MSE between reconstructed and original patches (computed only on masked regions)
4. **Target**: Patch-normalized spectrograms (normalize each patch by mean/std for stability)

After pre-training, encoder provides high-quality embeddings for downstream tasks.

### Setup

| Component | Details |
|-----------|---------|
| Data | **XCM**: ~160k recordings (unlabeled, use all splits for pre-training)<br>Map-style dataset, mel-spectrograms generated on-the-fly |
| Compute | **GPU**: 1x RTX 6000 (24GB VRAM)<br>**CPUs**: 16 cores for DataLoader workers<br>**Memory**: 64GB RAM<br>**Time**: ~48-72 hours (100 epochs) |
| Dependencies | `einops>=0.7.0` (patchify/unpatchify), `timm>=0.9.0` (transformer blocks), `wandb>=0.16.0` |
| Code | `user/murilo/src/birdclef/`:<br>- `models/mae_decoder.py`<br>- `lightning/audio_mae.py`<br>- `lightning/callbacks.py` (MAEImageLogger) |

### Procedure

1. **Implementation**:
   - Implement `MAEDecoder` (lightweight transformer: 4 layers, 512 dim, 8 heads)
   - Implement random masking logic (75% patches masked)
   - Implement patch normalization for targets
   - Implement `AudioMAEModule` (PyTorch Lightning)
   - Implement `MAEImageLogger` callback for W&B (visualize reconstructions)

2. **Quick Validation (HSN, fast_dev_run=5)**:
   - Verify masking: 75% patches masked, reconstruction targets have correct shape
   - Verify forward pass: encoder processes 25% visible patches, decoder reconstructs all
   - Verify loss computation: MSE computed only on masked patches
   - Verify W&B logging: reconstruction images show [original | masked | reconstructed]

3. **HSN Pre-training (Quick Test, 20 epochs)**:
   - Train on HSN for 20 epochs to validate full pipeline
   - Monitor: reconstruction quality improves, loss decreases monotonically
   - Inspect visualizations: reconstructions should show structure (not gray blur)

4. **XCM Pre-training (Production, 100 epochs)**:
   - Train on full XCM for 100 epochs
   - Save encoder checkpoints for downstream tasks
   - Monitor reconstruction quality across epochs

### Variables

- **Independent**:
  - Mask ratio (fixed at 75%)
  - Decoder architecture (depth=4, dim=512, heads=8)
  - Patch size (16x16, matches ViT)
  - Dataset (HSN → XCM)

- **Dependent**:
  - Reconstruction MSE loss (on masked patches)
  - Visual reconstruction quality (qualitative)
  - Encoder representation quality (measured via downstream tasks)

- **Controlled**:
  - Mel-spectrogram parameters (128 mels, 500 frames, same as Exp 002)
  - Encoder architecture (ViT-Base, same as Exp 002)
  - Learning rate schedule (higher lr=1.5e-4 for MAE, following MAE paper)
  - Batch size (64, larger than supervised due to no labels)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Reconstruction MSE** | Mean squared error between predicted and true patches (computed only on masked regions). Should decrease monotonically. |
| **Visual Quality** | Qualitative assessment of reconstructions via W&B images. Good: shows structure (frequency patterns, temporal patterns). Bad: gray blur (averaging). |
| **Loss Convergence** | Training loss should decrease smoothly without plateau or divergence. |
| **Encoder Embeddings** | (Future) Zero-shot retrieval performance, linear probe accuracy on frozen embeddings. |

### Baseline

**Random Initialization**:
- Initial MSE loss: ~1.0-2.0 (normalized patches have variance ~1)
- Random encoder produces gray blur reconstructions (averages patches)

**Expected After Pre-training**:
- Final MSE loss: <0.5 (50% reduction from random)
- Reconstructions show frequency structure, temporal patterns, recognizable as spectrograms

**Comparison**:
- No direct BirdSet MAE baseline exists (novel approach for this dataset)
- AudioMAE paper (on AudioSet): achieves strong reconstruction quality, useful for downstream

### Success Criteria

- **Confirm Pre-training if**:
  - MSE loss decreases monotonically from ~1.5 to <0.5
  - Reconstructions show clear structure (not gray blur) by epoch 20
  - W&B logging captures reconstruction images correctly
  - No training instabilities (loss spikes, NaN)

- **Confirm Encoder Quality if**:
  - Reconstructed spectrograms are visually similar to originals (frequency/temporal patterns preserved)
  - Masked regions are filled in plausibly (not just noise or averaging)
  - Loss continues decreasing through 100 epochs (not plateauing early)

- **Reject if**:
  - Loss plateaus >1.0 (model not learning)
  - Reconstructions remain gray blur after 50 epochs (encoder not extracting features)
  - Loss diverges or produces NaN (instability, need to debug normalization or lr)
  - OOM with batch_size=64 (need to reduce batch size)

## Limitations

- **No downstream evaluation**: This experiment only validates reconstruction quality, not downstream task performance (retrieval, classification). Need follow-up experiments to assess encoder usefulness.
- **Single mask ratio**: Testing only 75% masking (MAE/AudioMAE default). Optimal ratio for audio may differ.
- **Patch normalization**: Normalizing each patch may remove global intensity information, but improves training stability (following MAE paper).
- **Decoder discarded**: After pre-training, only encoder is used. Decoder quality doesn't directly indicate encoder quality.
- **Single dataset**: Only pre-training on XCM. Generalization to other bird datasets (e.g., Xeno-Canto) is unknown.
- **Compute constraints**: Single GPU limits batch size to ~64. Larger batches might improve training stability.
- **Baseline comparison**: No supervised ViT pre-training baseline to compare against (e.g., ImageNet pre-trained vs. MAE pre-trained).
