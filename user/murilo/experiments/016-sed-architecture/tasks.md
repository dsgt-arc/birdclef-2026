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

## 6. Documentation

- [ ] 6.1 Update MODEL-RUNS.md
- [ ] 6.2 Create results.md

---

## Future Work

- **Add SED to existing encoders**: ProtoCLR, ConvNeXt, EAT with SED head + LoRA
- **Temporal post-processing**: Gaussian smoothing [0.1, 0.2, 0.4, 0.2, 0.1]
- **Ensemble**: Multiple SED models (EfficientNet-B0 + our 3 encoders)
- **Pseudo-labeling with SED teachers**: Re-run exp 014/015 with SED-based teachers
- **Overlapping inference**: 20s windows with 5s step (2025 winner pattern)
