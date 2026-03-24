# Experiment 016: SED Architecture — Tasks

## Goal

Replace clip-level classification with Sound Event Detection architecture to
close the gap to the competition leader. Target: LB ≥0.85 (from 0.767 baseline).

**Competition reference**: EfficientNet-B0 SED single model = 0.908 LB
**Our baseline** (exp 012): ProtoCLR 0.767, ConvNeXt 0.755, EAT 0.720

---

## 1. Setup

- [x] 1.1 Create `src/birdclef/heads/sed_head.py` (GeM + AttHead)
- [x] 1.2 Create `src/birdclef/encoders/efficientnet.py` (EfficientNet-B0 via timm)
- [x] 1.3 Create `src/birdclef/lightning/sed_finetune.py` (SEDFinetuneModule)
- [x] 1.4 Create `src/birdclef/data/sed.py` (SEDFocalDataset, SEDSoundscapeDataset, SEDDataModule)
- [x] 1.5 Update `__init__.py` package exports
- [x] 1.6 Create experiment directory, pyproject.toml, config, SLURM scripts
- [x] 1.7 Create `train.py`

### Bugs Fixed During Setup

1. **embed_dim 1280 → 320**: With `features_only=True`, EfficientNet-B0's last stage
   outputs 320 channels (not 1280 — that's the conv_head which is excluded). Fixed in
   config and train.py now uses `encoder.embed_dim` instead of config value.

2. **PyTorch 2.11+cu130 incompatibility**: PyTorch 2.11 released 2026-03-23 and
   auto-resolved from the cu121 index. CUDA 13.0 requires newer drivers than PACE
   RTX 6000 nodes have. Fixed by pinning `torch<2.11` in `_job.sh`.

## 2. Pipeline Validation

- [x] 2.1 Fast dev run: job 5430704, completed successfully
- [x] 2.2 Encoder: embed_dim=320, 5 stages (correct with features_only=True)
- [x] 2.3 SED head: no shape errors, clipwise + framewise passed through loss and metrics
- [x] 2.4 Soft CE loss: train=5.654, val=5.524 (reasonable for 234 classes, ln(234)≈5.46)
- [x] 2.5 Datasets: 28,596 focal recordings + 570 soundscape = 29,166 train (10s windows)
      Model: 3.6M params (EfficientNet-B0), all trainable

## 3. GPU Check

- [x] 3.1 Submit: job 5430838, completed
- [x] 3.2 Results: GPU 68.2% avg, **data loading 42.1% avg** (spikes to 96%), memory 0.7%
      Bottleneck: I/O spikes from loading 10s audio. Compute is fast (141ms avg).
      Recommendation: increase batch_size and num_workers to reduce data stalls.

## 4. Full Training

- [ ] 4.1 Submit: `bash sbatch/run.sh config/efficientnet_b0_sed.yaml`
- [ ] 4.2 Monitor via W&B dashboard
- [ ] 4.3 Record best val/auroc, stopped epoch

### Results

| Model | val/auroc | val/cmap | test/auroc | test/cmap | Best Epoch | Stopped |
|-------|-----------|----------|-----------|----------|------------|---------|
| EfficientNet-B0 SED | | | | | | |

## 5. ONNX Export & Kaggle Submission

SED models need a **different ONNX export** than clip-level models because
the inference logic is different (dual output + blending).

- [ ] 5.1 Create `scripts/export_efficientnet_sed.py` in exp 010 pipeline
      - Load checkpoint → build EfficientNetEncoder + AttHead
      - Wrap in SEDWrapper that takes mel input and returns blended probabilities:
        `0.75 * sigmoid(clipwise) + 0.25 * sigmoid(framewise_max)`
      - Export single ONNX with mel input → blended probs output
      - Alternative: export raw clipwise + framewise_max logits and do blending
        in submission.py (more flexible for tuning blend ratio)
- [ ] 5.2 Register in exp 010: export.sbatch, upload_kaggle.py, kernel metadata, submission.py MODEL_SLUGS
- [ ] 5.3 Verify ONNX output matches PyTorch (max diff < 1e-4)
- [ ] 5.4 Upload weights + push kernel to Kaggle
- [ ] 5.5 Verify species_mapping.json has 234 items on Kaggle before submitting
- [ ] 5.6 Record LB score, compare to baseline and competition reference (0.908)

### LB Results

| Model | Exp 012 Best | Exp 016 LB | Target |
|-------|-------------|-----------|--------|

### LB Results

| Model | Exp 012 Best | Exp 016 LB | Target |
|-------|-------------|-----------|--------|
| EfficientNet-B0 SED | 0.767 (ProtoCLR) | pending | ≥0.85 |

## 6. Extend SED to Existing Encoders (after EfficientNet-B0 validates)

If EfficientNet-B0 SED shows significant LB improvement (≥0.05 over baseline):

- [ ] 6.1 Add `forward_features()` to ProtoCLR encoder (returns (B, 384, H, W))
- [ ] 6.2 Add `forward_features()` to ConvNeXt encoder (timm already has this)
- [ ] 6.3 Add `forward_features()` to EAT encoder (returns (B, seq_len, 768) tokens — 1D temporal, skip GeM)
- [ ] 6.4 Create SED configs for ProtoCLR, ConvNeXt, EAT (with LoRA + SED head)
- [ ] 6.5 Train all 3 encoders with SED head (resume from exp 012 LoRA checkpoints)
- [ ] 6.6 Submit to Kaggle LB, compare all 4 SED models

### Expected SED Results (all models)

| Model | Params | Exp 012 LB (clip) | Exp 016 LB (SED) | Delta |
|-------|--------|-------------------|-------------------|-------|
| EfficientNet-B0 | 5.3M | — | pending | — |
| ProtoCLR (CvT-13) | 19.9M | 0.767 | pending | — |
| ConvNeXt-Base | 89M | 0.755 | pending | — |
| EAT (ViT) | 90M | 0.720 | pending | — |

### Architecture Notes per Encoder

- **ProtoCLR**: CvT produces (B, 384, H, W) feature maps. Apply GeM + AttHead directly.
  Resume from exp 012 LoRA checkpoint, replace head with AttHead, freeze backbone
  initially then unfreeze with LoRA.
- **ConvNeXt**: `backbone.forward_features()` returns (B, 1024, H, W). Straightforward
  — same pattern as EfficientNet. Resume from exp 012 LoRA checkpoint.
- **EAT**: ViT outputs (B, seq_len, 768) tokens. No 2D spatial structure — skip GeM,
  apply 1D temporal attention directly on the token sequence. May need a modified
  AttHead variant (AttHead1D) that skips GeM and takes (B, T, C) input.

## 7. Documentation

- [ ] 7.1 Update MODEL-RUNS.md
- [ ] 7.2 Create results.md

---

## Future Work

See [competition discussion analysis](../../docs/references/kaggle/birdclef-2026/competition-discussion-analysis.md)
for full details on each technique, confirmed LB impacts, and code snippets.

### Immediate Improvements (next training run)

- **NS-JFT pretrained weights**: Switch from `efficientnet_b0` (standard ImageNet) to
  `tf_efficientnet_b0.ns_jft_in1k` (Noisy Student on JFT-300M). Competitor confirmed
  it consistently outperforms standard ImageNet init for audio spectrograms.
- **ASL loss**: Consider replacing soft CE with Asymmetric Loss (γ⁻=4, γ⁺=0, clip=0.05).
  Competitor ablation: +0.023 over BCE. Handles multi-label natively without normalization.
- **Raw waveform mixup**: Current spectrogram mixup vs competitor's raw waveform mixup.
  2025 2nd place confirmed +0.036 Private LB from audio-domain mixup. Requires moving
  mel transform to model forward pass or custom collate.

### Free Post-Processing (no retraining, add to submission.py)

- **TopN postprocessing** (+0.011 LB): `preds *= preds.max(axis=0, keepdims=True)`
- **Temporal smoothing** (+0.010 LB): Gaussian [0.1, 0.2, 0.4, 0.2, 0.1] across segments
- **TTA: 2.5s shifts** (+0.012 LB): 3 predictions per segment (0, −2.5s, +2.5s), average
- **Model soup** (free stability): Average last 3-5 epoch checkpoints

### Larger Improvements

- **Perch as soft label teacher** (+0.030 val): Knowledge distillation from Google Perch
  model on soundscape clips. Perch baseline alone = 0.825 LB.
- **Multi-round pseudo-labels with PowerTransform**: p^γ (γ=2.0) sharpening across 4
  rounds. 2025 1st place core innovation: +0.032 LB.
- **SoftAUCLoss**: Directly optimizes competition metric (macro ROC-AUC) instead of proxy.
  Used in later pseudo-label rounds by 2025 1st place.
- **StochasticDepth**: drop_path_rate=0.2 during pseudo-label training rounds.
- **Separate non-bird pipeline**: Dedicated model for insects/amphibians (+0.003 LB).
- **819K prior-year pretraining**: 2025 2nd place gained +0.014 LB from pretraining on
  BirdCLEF 2021-2024 data (16,607 species, 50 epochs).
- **Extend SED to ProtoCLR/ConvNeXt/EAT**: See section 6 above.
- **Ensemble**: Multiple SED models with Optuna weight optimization.
