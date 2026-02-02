# Tasks: Supervised ViT Baseline

## 1. Setup
- [x] 1.1 Add dependencies to pyproject.toml (timm, wandb, einops)
- [x] 1.2 Create directory structure (lightning/, models/, preprocessing/)
- [x] 1.3 Install dependencies with `uv sync`
- [x] 1.4 Add python-dotenv for W&B API key management
- [x] 1.5 Create .env.example template
- [x] 1.6 Verify BirdSet HSN dataset is accessible
- [x] 1.7 Create experiment output directories
- [x] 1.8 Setup project shared storage for checkpoints

## 2. Infrastructure (Complete)
- [x] 2.1 Implement `MelSpectrogramTransform` (preprocessing/mel_spectrogram.py)
- [x] 2.2 Implement `ViTAudio` model (models/vit_audio.py)
- [x] 2.3 Implement `BirdSetDataModule` (lightning/birdset_datamodule.py)
- [x] 2.4 Implement `SupervisedViTModule` (lightning/supervised_vit.py)
- [x] 2.5 Create training script (train.py)
- [x] 2.6 Create config file (config.yaml)
- [x] 2.7 Create SLURM script (sbatch/train.sh)

## 3. Testing & Debugging (Complete)
- [x] 3.1 Create pytest test suite (39 tests across 5 modules)
- [x] 3.2 Run initial test suite (27/39 passing)
- [x] 3.3 Debug and fix failing tests:
  - [x] 3.3.1 Fix datamodule tests - VERIFIED 8/8 PASSING
  - [x] 3.3.2 Fix lightning module tests
  - [x] 3.3.3 Add test_step() and test metrics (was missing, caused MisconfigurationException)
- [x] 3.4 Run fast-dev-run to verify full pipeline (train → val → test)

## 4. Execution (Complete)
- [x] 4.1 Run fast-dev-run validation (5 batches) - PASSED
- [x] 4.2 Run full HSN training (50 epochs) - COMPLETE (early stopped at epoch 13)
- [x] 4.3 Monitor W&B and TensorBoard for training curves
- [x] 4.4 Evaluate results against BirdSet baselines (ConvNeXT: 0.38 cmAP)
- [x] 4.5 Document results in results.md
- [ ] 4.6 Update docs/STATUS.md

---

## Final Results Summary

| Metric | ViT (Ours) | ConvNeXT (Baseline) | % of Baseline |
|--------|------------|---------------------|---------------|
| **cmAP** | **0.164** | 0.480 | 34% |
| AUROC | 0.578 | 0.883 | 65% |
| F1 Macro | 0.086 | 0.139 | 62% |
| Top-1 Acc | 0.112 | 0.251 | 45% |
| Top-5 Acc | 0.547 | 0.330 | 166% ⬆️ |

**Key findings:**
- Severe overfitting: train/cmap=0.984 vs val/cmap=0.164
- Early stopped at epoch 13 (best at epoch 3)
- 34% of BirdSet ConvNeXT baseline on cmAP
- Top-5 accuracy exceeds baseline (54.7% vs 33.0%)
- Training time: 37 min on RTX 6000

**See [results.md](results.md) for full analysis and improvement suggestions.**

---

## Recent Work Summary (2026-02-02, Session 3)

**Training completed successfully!**

- Job ID: 3692914
- Wall time: 37 min 16 sec
- Best epoch: 3 (val/cmap=0.164)
- Early stopped: epoch 13 (patience=10)

**Artifacts:**
- Best checkpoint: `/storage/project/ps-dsgt_clef2026-0/shared/.../checkpoints/best-epochepoch=03.ckpt`
- W&B: https://wandb.ai/dsgt-arc/birdset-vit-baseline/runs/dvoxy50z
- Results: [results.md](results.md)

---

## Recent Work Summary (2026-02-02, Session 2)

**Commits**:
- `f5a3238` - Fix checkpoint config: reduce save_top_k, simplify filename
- `afeffdc` - Save checkpoints to project shared storage instead of home dir

**Issues Fixed**:
1. **Disk quota exceeded** - First training run crashed after epoch 0 due to home dir quota (20GB limit)
   - Cleaned up 11GB of old checkpoints from `outputs/checkpoints/`
   - Reduced `save_top_k: 1` (was 3) to save ~2GB per run
   - Simplified checkpoint filename pattern to avoid `/` directory issues
2. **Storage location** - Moved checkpoints to project shared storage
   - Old: `~/experiments/.../outputs/checkpoints/` (home dir, 20GB quota)
   - New: `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/002-supervised-vit-baseline/checkpoints/`
   - Added `PROJECT_NAME` variable for easy switching between projects

**Storage Setup**:
| Storage | Path | Quota | Usage |
|---------|------|-------|-------|
| Home | `/storage/home/.../` | 20GB | Code, configs |
| Project | `/storage/project/ps-dsgt_clef2026-0/shared/` | 1TB | Checkpoints, results |
| Scratch | `/storage/scratch1/.../` | 15TB | Temp files, W&B logs |

---

## Previous Work Summary (2026-02-02, Session 1)

**Commit**: `af0bf77` - Add test_step, AUROC metrics, and TensorBoard logging to exp002

**Issues Fixed**:
1. **MisconfigurationException** - Added missing `test_step()` and `on_test_epoch_end()` to SupervisedViTModule
2. **Missing AUROC metric** - Added AUROC to train/val/test phases for parity with run_inference.py
3. **No local logging** - Added TensorBoard logger alongside W&B
4. **Noisy logs** - Suppressed verbose httpx and huggingface_hub loggers
5. **SLURM integration** - Added `srun` to sbatch script for proper signal handling

---

## Previous Work Summary (2026-02-01)

**Commits**: 3 commits (57902cd, 9c4f7f7, a018ce7)
- Refactored src/ structure (models.py → classifiers.py)
- Created 39 tests across 5 modules
- Fixed 10/12 failing tests (datamodule + lightning module)

**Key Fixes**:
1. BirdSet uses `test_5s` split (not `val` or `test`)
2. Audio files need stereo→mono conversion
3. Lightning module: access hparams.lr, return loss from validation_step

---

## Next Steps

1. **Address overfitting** - Add data augmentation (SpecAugment, mixup)
2. **Try different architectures** - Audio Spectrogram Transformer (AST), ConvNeXT
3. **Hyperparameter tuning** - Lower learning rate, increase regularization
4. **Scale to larger dataset** - Try XCM (larger BirdSet dataset)
