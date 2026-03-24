# Experiment 016: SED Architecture — Tasks

## Goal

Replace clip-level classification with Sound Event Detection architecture to
close the gap to the competition leader. Target: LB ≥0.85 (from 0.767 baseline).

**Competition reference**: EfficientNet-B0 SED single model = 0.908 LB
**Our baseline** (exp 012): ProtoCLR 0.767, ConvNeXt 0.755, EAT 0.720

### Current Status (2026-03-24)

- **Branch**: `mg/016-sed-improvements` (PR #18)
- **Best LB**: **0.823** (v1 + post-processing)
- **v2 training complete**: NS-JFT + CE loss **regressed** (test/auroc 0.837, LB 0.755)
- **v3 ablation complete**: v3b LB 0.802 (worse than v1+pp 0.823), v3a/v3c LB pending
- **v3d (EfficientNetV2-B0) training**: post author (0.889) used V2; 0.908 commenter ambiguous
- **Key references**:
  - [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
  - [Perch + ProtoSSM analysis](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach-analysis.md) (0.913 LB)

### Score Progression

| Version | Changes | LB |
|---------|---------|-----|
| Exp 012 ProtoCLR | Clip-level baseline | 0.767 |
| Exp 016 v1 | SED + soft CE + spectrogram mixup | 0.795 |
| Exp 016 v1 + pp | + TTA + smoothing + TopN | **0.823** |
| Exp 016 v2 + pp | + NS-JFT + CE + post-processing | 0.755 (regression) |
| Exp 016 v3b + pp | + raw waveform mixup + pp | 0.802 (regression) |
| Exp 016 v3a + pp | + TF-ported weights + pp | 0.813 (regression) |
| Exp 016 v3c + pp | + standard CE loss + pp | pending |
| Exp 016 v3d | EfficientNetV2-B0 | training... |
| Target | Close gap to competitor | ≥0.85 |

---

## 1. Setup

- [x] 1.1 Create `src/birdclef/heads/sed_head.py` (GeM + AttHead)
- [x] 1.2 Create `src/birdclef/encoders/efficientnet.py` (EfficientNet-B0 via timm)
- [x] 1.3 Create `src/birdclef/lightning/sed_finetune.py` (SEDFinetuneModule)
- [x] 1.4 Create `src/birdclef/data/sed.py` (SEDFocalDataset, SEDSoundscapeDataset, SEDDataModule)
- [x] 1.5 Update `__init__.py` package exports
- [x] 1.6 Create experiment directory, pyproject.toml, config, SLURM scripts
- [x] 1.7 Create `train.py`
- [x] 1.8 Create `save_metrics.py` (saves structured metrics to `artifacts/`)

### Bugs Fixed During Setup

1. **embed_dim 1280 → 320**: With `features_only=True`, EfficientNet-B0's last stage
   outputs 320 channels (not 1280 — that's the conv_head which is excluded). Fixed in
   config and train.py now uses `encoder.embed_dim` instead of config value.

2. **PyTorch 2.11+cu130 incompatibility**: PyTorch 2.11 released 2026-03-23 and
   auto-resolved from the cu121 index. CUDA 13.0 requires newer drivers than PACE
   RTX 6000 nodes have. Fixed by pinning `torch<2.11` in `_job.sh`.

3. **W&B config key conflict**: `encoder` key set as dict by first `config.update()`,
   then overwritten as string. Renamed to `encoder_name`.

4. **OOM at batch_size=256**: 10s mel spectrograms (128×1000) too large for 24GB RTX 6000
   with batch_size=256. Reduced to batch_size=128 with accumulate_grad_batches=2.

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
      Updated config: batch_size 64→128, num_workers 8→12, accumulate 4→2.

## 4. Full Training

- [x] 4.1 Submit: job 5432792, completed
- [x] 4.2 Monitored via W&B: `dsgt-arc/birdclef-2026-sed-architecture/runs/x5rs663d`
- [x] 4.3 Results recorded below
- [x] 4.4 Structured metrics saved to `artifacts/training/efficientnet_b0_sed/metrics.json`

### Training Results

| Model | val/auroc | test/auroc | test/cmap | Best Epoch | Stopped |
|-------|-----------|-----------|----------|------------|---------|
| EfficientNet-B0 SED | 0.845 | **0.860** | **0.593** | 24 | 34 (early stop) |

### Training Dynamics

Strong learning curve from ImageNet init (no prior BirdCLEF checkpoint):
- Epoch 0: val/auroc 0.517 (random head)
- Epoch 5: val/auroc 0.815 (rapid improvement)
- Epoch 12: val/auroc 0.839
- Epoch 18: val/auroc 0.844
- Epoch 24: val/auroc 0.845 (best)
- Epoch 34: early stopped (patience=10)

### Comparison vs Exp 012 Clip-Level (offline test set)

| Model | Architecture | Test AUROC | Test cmAP |
|-------|-------------|-----------|----------|
| **EfficientNet-B0 SED** | **SED (GeM + AttHead)** | **0.860** | **0.593** |
| ProtoCLR (exp 012) | Clip (global avg pool) | 0.845 | 0.481 |
| ConvNeXt (exp 012) | Clip (global avg pool) | 0.844 | 0.470 |
| EAT (exp 012) | Clip (global avg pool) | 0.838 | 0.452 |

SED beats all clip-level models: +0.015 AUROC, **+0.112 cmAP** over best clip-level.

## 5. ONNX Export & Kaggle Submission

- [x] 5.1 Created `scripts/export_efficientnet_sed.py` — SEDWrapper bakes blending
      into ONNX: `0.75 * sigmoid(clipwise) + 0.25 * sigmoid(framewise_max)`.
      Dynamic time axis for both 5s and 10s inference.
- [x] 5.2 Registered in exp 010: export.sbatch, upload_kaggle.py, kernel metadata,
      submission.py (MODEL_SLUGS + output_type=probs handling)
- [x] 5.3 ONNX exported: `bash sbatch/run.sh export --model effnet-b0-sed-v1`
- [x] 5.4 Uploaded + pushed kernel: `upload_kaggle.py --model effnet-b0-sed-v1 --push-kernel`
- [x] 5.5 Verified species_mapping.json has 234 items on Kaggle
- [x] 5.6 LB score: **0.795** (+0.028 over exp 012 ProtoCLR 0.767, but far from 0.908 target)

### Key Implementation Notes for SED ONNX

- ONNX outputs **probabilities** (not logits) — submission.py checks `mel_cfg["output_type"] == "probs"` to skip sigmoid
- mel_config has `target_frames: 500` (5s at 32kHz) for inference, though model trained on 1000 (10s)
- Dynamic time axis allows both 5s and 10s inference windows
- Species mapping uses 234-class identity mapping (same as soundscape/student models)

### LB Results

| Model | Exp 012 Best | Exp 016 LB | Target |
|-------|-------------|-----------|--------|
| EfficientNet-B0 SED v1 | 0.767 (ProtoCLR) | 0.795 | ≥0.85 |
| EfficientNet-B0 SED v1 + pp | — | **0.823** | — |

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
| EfficientNet-B0 | 3.6M | — | pending | — |
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

## 7. Quick Wins — Post-Processing (no retraining, est. +0.03 combined)

- [x] 7.1 Add TopN postprocessing to submission.py (+0.011)
- [x] 7.2 Add temporal smoothing to submission.py (+0.010)
- [x] 7.3 Add TTA 2.5s shifts to submission.py (+0.012)
- [ ] 7.4 Model soup: average epochs 20-24 checkpoints, re-export ONNX
- [x] 7.5 LB with post-processing: **0.823** (+0.028 over v1 baseline 0.795)
      TTA + smoothing + TopN confirmed +0.028 gain (estimated was +0.033)

## 8. Training v2 — Close the Gap (est. +0.05–0.08)

- [x] 8.1 Created v2 config: `tf_efficientnet_b0.ns_jft_in1k` + CE loss with label_smoothing=0.05
- [x] 8.2 Made loss_type configurable in SEDFinetuneModule ("soft_ce" for v1, "ce" for v2)
- [x] 8.3 v2 training submitted — job 5438912, completed
      Config: `config/efficientnet_b0_sed_v2.yaml`
      Changes: `tf_efficientnet_b0.ns_jft_in1k`, loss_type="ce", label_smoothing=0.05
- [x] 8.4 Record v2 results — **REGRESSION**: test/auroc 0.837 (−0.023), test/cmap 0.539 (−0.054)
      v2 underperformed v1 on all metrics. NS-JFT + CE hurt vs ImageNet + soft CE.
      W&B run: `covhhpck`. Best epoch 24, early stopped at 35.
- [x] 8.5 Export v2 to ONNX + submit to Kaggle LB: **0.755** (−0.068 vs v1+pp 0.823, confirmed regression)
- [ ] 8.6 Implement raw waveform mixup (move mel to model forward, mixup before mel)
- [ ] 8.7 Implement 10s overlapping inference (load 10s centered on 5s target)

## 9. v3 Ablation Runs — Isolate Individual Changes

Three parallel runs, each changing one variable from v1 baseline:

- [x] 9.1 v3a config: `tf_efficientnet_b0.in1k` (TF-ported Google weights)
- [x] 9.2 v3b config + code: raw waveform mixup (mel to GPU, mixup before mel)
      Changed: `sed_finetune.py` (mel_transform param, waveform mixup path),
      `sed.py` (eval_transform for val/test), `train.py` (raw_waveform_mixup flag)
- [x] 9.3 v3c config: `loss_type="ce"`, `label_smoothing=0.0` (standard CE)
- [x] 9.4 Tests: 29 pytest tests covering all v3 changes + backward compat
- [x] 9.5 All three submitted to SLURM (running in parallel)
- [x] 9.6 Record v3a/v3b/v3c results:
      v3a (TF weights): test/auroc 0.857, test/cmap 0.599 (flat vs v1)
      v3b (waveform mixup): test/auroc 0.862, test/cmap 0.607 (+0.014 cmAP, but LB 0.802 — worse than v1+pp 0.823)
      v3c (CE loss): test/auroc 0.850, test/cmap 0.552 (hurt, soft CE is better)
- [x] 9.7 Export v3a/v3b/v3c + submit to Kaggle LB (v3b=0.802, v3a/v3c pending)
- [x] 9.8 v3d config created: EfficientNetV2-B0 (tf_efficientnetv2_b0.in1k) — competitor
      confirmed using V2 not V1. Training submitted.
- [ ] 9.9 Record v3d results + submit to Kaggle LB
- [ ] 9.10 If v3d helps → combine with waveform mixup into v4

## 10. Documentation

- [x] 10.1 Update MODEL-RUNS.md with LB scores
- [x] 10.2 Create results.md with gap analysis
- [x] 10.3 Create summary.md with full experiment report
- [x] 10.4 Kaggle notebook reference docs (Perch+ProtoSSM, Pantanal Distill, etc.)

---

## Future Work

Detailed analysis of each item below is in [summary.md](summary.md) § "Potential Next Steps".

### High Priority (next experiments)

- [ ] **EfficientNetV2-B0** (v3d, training): Post author (0.889) used V2; 0.908
  commenter ambiguous ("SED EfficientNet B0"). V2 has Fused-MBConv, 5.6M params.
  Worth testing regardless. If successful → combine into v4.

- [ ] **Train/inference window alignment**: We train on 10s but infer on 5s. The model
  sees 1000 mel frames during training but only 500 at inference. Two options:
  - **10s overlapping inference** (no retraining): Load 10s centered on each 5s target,
    overlap-average predictions. 2024 1st place confirmed +0.015 LB.
  - **Train on 5s** (retraining): Match 2025 1st place who trained AND inferred on 5s.

- [ ] **Mel spectrogram parameters**: Our n_fft=1024, 128 mels differs significantly
  from 2025 1st place (n_fft=4096, 224 mels). Higher frequency resolution may help
  distinguish similar bird calls. Requires retraining.

### Medium Priority

- [ ] **Model soup**: Average last 3-5 epoch checkpoints. Free stability, no retraining.
- [ ] **ASL loss**: Asymmetric Loss (γ⁻=4, γ⁺=0, clip=0.05). Competitor: +0.023 over BCE.
  May work where standard CE failed (v3c) since ASL handles multi-label natively.
- [ ] **Ensemble with Perch**: Weighted average of our SED + Perch-based predictions.
  The approaches are complementary (trained features vs foundation model).

### Lower Priority / Longer Term

- [ ] **Perch as soft label teacher** (+0.030 val): Knowledge distillation on soundscapes.
- [ ] **Multi-round pseudo-labels with PowerTransform**: 2025 1st place: +0.032 LB.
- [ ] **Extend SED to ProtoCLR/ConvNeXt/EAT**: See section 6 above.
- [ ] **819K prior-year pretraining**: +0.014 LB from BirdCLEF 2021-2024 data.

### Completed / Ruled Out

- ~~NS-JFT pretrained weights~~: v2, regressed (−0.068 LB, confounded with CE loss)
- ~~Standard CE loss~~: v3c, hurt offline (−0.041 cmAP). Soft CE is better.
- ~~Raw waveform mixup~~: v3b, better offline but worse LB (0.802 vs 0.823)
- ~~TF-ported V1 weights~~: v3a, LB 0.813 (−0.010). Worse than v1.
- ~~Post-processing~~: ✅ Done (+0.028 confirmed). TTA + smoothing + TopN.
