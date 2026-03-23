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

## 2. Pipeline Validation

- [ ] 2.1 Fast dev run: `bash sbatch/run.sh config/efficientnet_b0_sed.yaml --fast-dev-run`
- [ ] 2.2 Verify encoder output shape: (B, 1280, F', T')
- [ ] 2.3 Verify SED head output shapes: clipwise (B, 234), framewise (B, 234, T)
- [ ] 2.4 Verify soft CE loss computation (no NaN/inf)
- [ ] 2.5 Verify dataset sizes (focal recordings + soundscape segments)

## 3. GPU Check

- [ ] 3.1 Submit: `bash sbatch/run.sh config/efficientnet_b0_sed.yaml --gpu-check`
- [ ] 3.2 Record GPU utilization, memory usage, throughput

## 4. Full Training

- [ ] 4.1 Submit: `bash sbatch/run.sh config/efficientnet_b0_sed.yaml`
- [ ] 4.2 Monitor via W&B dashboard
- [ ] 4.3 Record best val/auroc, stopped epoch

### Results

| Model | val/auroc | val/cmap | test/auroc | test/cmap | Best Epoch | Stopped |
|-------|-----------|----------|-----------|----------|------------|---------|
| EfficientNet-B0 SED | | | | | | |

## 5. Evaluation

- [ ] 5.1 Export to ONNX (update exp 010 pipeline for SED inference)
- [ ] 5.2 Submit to Kaggle LB
- [ ] 5.3 Compare to exp 012 baseline and competition reference (0.908)

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

- **Temporal post-processing**: Gaussian smoothing [0.1, 0.2, 0.4, 0.2, 0.1]
- **Ensemble**: Multiple SED models (EfficientNet-B0 + our 3 encoders)
- **Pseudo-labeling with SED teachers**: Re-run exp 014/015 with SED-based teachers
- **Overlapping inference**: 20s windows with 5s step (2025 winner pattern)
