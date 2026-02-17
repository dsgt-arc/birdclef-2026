# Results: BirdCLEF End-to-End Pipeline

**Date Completed**: 2026-02-16 (in progress)
**Author**: Anthony

## Summary

The end-to-end pipeline — SSL pretraining, fine-tuning, ONNX export, and Kaggle submission — is fully operational. The AudioSet-initialized finetune reached 0.425 val/cmap (epoch 11, still climbing) and scored 0.484 public / 0.476 private on BirdCLEF 2025. SSL pretraining from scratch and with AudioSet warm-start both converged to similar reconstruction loss after 60+ epochs. An SSL-initialized finetune is now running.

**Verdict**: confirmed (pipeline works end-to-end; performance improvements ongoing)

## Observations

### Pretraining (MPG reconstruction, L1 loss, linear decoder)

Two conditions ran for ~62-64 epochs each (~40 min/epoch on RTX 6000):

| Condition | Epochs | Loss (L1) | Monitor MSE | Monitor L1 |
|-----------|--------|-----------|-------------|------------|
| pretrain-full (from scratch) | 64 | 0.670 → 0.408 | 0.788 → 0.320 | 0.670 → 0.408 |
| pretrain-warm (AudioSet init) | 62 | 0.575 → 0.409 | 0.598 → 0.321 | 0.575 → 0.409 |

- Warm start converges faster initially (epoch 0: 0.575 vs 0.670) but both reach the same plateau by epoch 60+
- Steep improvement epochs 0-10, then gradual — diminishing returns after epoch 30
- MPC (contrastive) was removed early: degenerate with 128x1 tokens (adjacent mel frames near-identical)
- JiTTER (block shuffle) removed: fights Mamba's sequential state dynamics

### Fine-tuning

Four runs total, only the last used pretrained weights:

| Run | Init | Epochs | Best val/cmap | val/auroc | Notes |
|-----|------|--------|---------------|-----------|-------|
| v3951429 | scratch (SED head) | 15 | 0.012 | 0.044 | Old SED architecture |
| v3951522 | scratch (mean pool) | 17 | 0.014 | 0.038 | Mean pool, no pretrained |
| v3951554 | scratch (mean pool) | 200 | 0.022 | 0.048 | 200 epoch overfit test |
| **v3960497** | **AudioSet backbone** | **12** | **0.425** | **0.934** | **Timed out, still climbing** |

The AudioSet backbone run (v3960497) is the only one with meaningful performance:
- Backbone frozen for 5 epochs (head-only warmup), then unfrozen
- val/cmap: 0.009 → 0.425 over 12 epochs, no plateau — would likely improve with more time
- Train/val gap at epoch 11: train=0.412, val=0.425 — no significant overfitting yet
- SpecAugment enabled (2 freq masks + 2 time masks)

### Kaggle Submission

| Metric | Value |
|--------|-------|
| Kernel status | COMPLETE |
| Public score | **0.484** |
| Private score | **0.476** |
| Runtime | 13s total (1.5s ONNX session init) |
| GPU required | No (CPU-only) |
| Model size | 27.3 MB (ONNX) |

Pipeline: librosa mel → ONNX Runtime + mamba-onnx custom ops → windowed mean pooling → sigmoid

For context:
- EfficientNetB1 baseline: 0.778 public / 0.810 private
- BirdNET baseline: ~0.72 public
- SSAMBA Tiny (this): 0.484 public / 0.476 private

### Quantitative Summary

| Metric | Baseline (scratch) | AudioSet init | Delta |
|--------|-------------------|---------------|-------|
| val/cmap (finetune) | 0.022 (200 ep) | 0.425 (12 ep) | +0.403 |
| val/auroc (finetune) | 0.048 | 0.934 | +0.886 |
| Kaggle public | N/A | 0.484 | — |
| Pretrain throughput | ~28 samples/sec | ~28 samples/sec | same |
| Pretrain time/epoch | ~40 min | ~40 min | same |
| Inference (Kaggle) | — | 13s total | — |

### Qualitative

- Pre-training without pretrained weights is essentially useless for this model size — scratch finetune peaks at 0.022 cmAP even after 200 epochs
- AudioSet backbone provides massive jumpstart despite 128x1 patch_embed re-initialization (only 96.9% Mamba weight transfer)
- Local val/cmap (0.425) and Kaggle public (0.484) are in reasonable agreement, validating the 90/10 stratified split
- The ONNX + mamba-onnx CPU inference path works reliably with no GPU requirement
- Pretraining warm-start vs cold-start converges to the same place for reconstruction — the question is whether SSL representations transfer differently to classification

## Analysis

The pipeline hypothesis is **confirmed**: a valid Kaggle submission was produced and all wall-clock benchmarks are documented.

The performance gap to SOTA (~0.78) is expected given SSAMBA Tiny is 7M params with 128x1 tokenization. The key bottleneck identified across experiments 007-008 was the patch_embed re-initialization when going from 16x16 (AudioSet) to 128x1 (our tokenization). The SSL pretraining on BirdCLEF data with native 128x1 patches should address this — that finetune run is in progress (Job 4011637).

### Confounders

- AudioSet finetune (v3960497) timed out at epoch 11 — true peak is unknown
- SSL pretrain runs were also cancelled at epoch 62-64 (not the target 100 epochs), though curves were near plateau
- Only one finetune run per condition — no variance estimates
- The 200-epoch scratch run (v3951554) had no early stopping and used overfit_batches=0, so it trained on the full dataset but the model simply couldn't learn without pretrained representations

## Artifacts

| Type | Location |
|------|----------|
| Pretrain checkpoints (full) | `logs/pretrain-full/checkpoints/` (65 epochs) |
| Pretrain checkpoints (warm) | `logs/pretrain-warm/checkpoints/` (63 epochs) |
| Finetune checkpoints | `logs/finetune/checkpoints/` (12 epochs, AudioSet init) |
| TensorBoard logs | `logs/*/lightning_logs/` |
| ONNX model | `artifacts/weights/ssamba_classifier.onnx` (27.3 MB) |
| Species list | `artifacts/weights/species_list.json` |
| Kaggle kernel | `kaggle_kernel/submission.py` |
| SLURM logs | `logs/pretrain-full-*.out`, `logs/finetune-*.out` |

## Next Steps

- [x] Submit SSL-pretrained finetune (Job 4011637 — running)
- [ ] Compare SSL finetune val/cmap against AudioSet finetune (0.425 baseline)
- [ ] If SSL finetune improves: re-export ONNX, push to Kaggle, resubmit
- [ ] Continue pretrain to 100 epochs if SSL finetune shows promise
- [ ] Experiment with larger model (SSAMBA Small, 25.5M params)
- [ ] Integrate teacher labels from experiment 010 (BamBird, BirdNET, Perch)

**Leads to**: experiment 010 (teacher labels), potential experiment 011 (knowledge distillation from teacher ensemble)
