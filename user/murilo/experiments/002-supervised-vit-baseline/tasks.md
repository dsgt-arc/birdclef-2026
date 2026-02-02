# Tasks: Supervised ViT Baseline

## 1. Setup
- [x] 1.1 Add dependencies to pyproject.toml (timm, wandb, einops)
- [x] 1.2 Create directory structure (lightning/, models/, preprocessing/)
- [x] 1.3 Install dependencies with `uv sync`
- [x] 1.4 Add python-dotenv for W&B API key management
- [x] 1.5 Create .env.example template
- [x] 1.6 Verify BirdSet HSN dataset is accessible
- [x] 1.7 Create experiment output directories

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

## 4. Execution
- [x] 4.1 Run fast-dev-run validation (5 batches) - PASSED
- [ ] 4.2 Run full HSN training (50 epochs)
  ```bash
  cd experiments/002-supervised-vit-baseline
  bash sbatch/run.sh
  ```
- [ ] 4.3 Monitor W&B and TensorBoard for training curves
- [ ] 4.4 Evaluate results against BirdSet baselines (ConvNeXT: 0.38 cmAP)
- [ ] 4.5 Document results in results.md
- [ ] 4.6 Update docs/STATUS.md

---

## Recent Work Summary (2026-02-02)

**Commit**: `af0bf77` - Add test_step, AUROC metrics, and TensorBoard logging to exp002

**Issues Fixed**:
1. **MisconfigurationException** - Added missing `test_step()` and `on_test_epoch_end()` to SupervisedViTModule
2. **Missing AUROC metric** - Added AUROC to train/val/test phases for parity with run_inference.py
3. **No local logging** - Added TensorBoard logger alongside W&B
4. **Noisy logs** - Suppressed verbose httpx and huggingface_hub loggers
5. **SLURM integration** - Added `srun` to sbatch script for proper signal handling

**Fast-dev-run Results** (5 batches):
```
val/cmap: 0.033    val/auroc: 0.251    val/f1_macro: 0.000
val/top1_acc: 0.013    val/top5_acc: 0.307
test/cmap: 0.033   test/auroc: 0.251   test/f1_macro: 0.000
```
(Low metrics expected with untrained model and only 5 batches)

**Metrics Tracked**:
- cmAP (class-mean Average Precision) - primary BirdSet metric
- AUROC (macro average)
- F1 macro/micro
- Top-1/Top-5 accuracy
- Loss

**Logging**:
- W&B: `dsgt-arc/birdset-vit-baseline` (full runs only)
- TensorBoard: `outputs/tensorboard/exp002-HSN/`

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

1. Submit full training job:
   ```bash
   cd user/murilo/experiments/002-supervised-vit-baseline
   bash sbatch/run.sh
   ```
2. Monitor progress:
   - W&B: https://wandb.ai/dsgt-arc/birdset-vit-baseline
   - Logs: `tail -f logs/slurm_*.log`
3. After completion, evaluate against BirdSet baselines
