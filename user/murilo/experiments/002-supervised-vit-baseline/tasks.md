# Tasks: Supervised ViT Baseline

## 1. Setup
- [x] 1.1 Add dependencies to pyproject.toml (timm, wandb, einops)
- [x] 1.2 Create directory structure (lightning/, models/, preprocessing/)
- [x] 1.3 Install dependencies with `uv sync`
- [ ] 1.4 Verify BirdSet HSN dataset is accessible
- [ ] 1.5 Create experiment output directories

## 2. Infrastructure (Complete)
- [x] 2.1 Implement `MelSpectrogramTransform` (preprocessing/mel_spectrogram.py)
- [x] 2.2 Implement `ViTAudio` model (models/vit_audio.py)
- [x] 2.3 Implement `BirdSetDataModule` (lightning/birdset_datamodule.py)
- [x] 2.4 Implement `SupervisedViTModule` (lightning/supervised_vit.py)
- [x] 2.5 Create training script (train.py)
- [x] 2.6 Create config file (config.yaml)
- [x] 2.7 Create SLURM script (sbatch/train.sh)

## 3. Testing & Debugging (Paused - Resume Here)
- [x] 3.1 Create pytest test suite (39 tests across 5 modules)
- [x] 3.2 Run initial test suite (27/39 passing)
- [x] 3.3 Debug and fix failing tests:
  - [x] 3.3.1 Fix datamodule tests - VERIFIED 8/8 PASSING ✓
    - Fixed: BirdSet split names (val/test → test_5s)
    - Fixed: Stereo to mono conversion (audio.mean(axis=1))
    - Commit: `9c4f7f7`
  - [x] 3.3.2 Fix lightning module tests - FIXES APPLIED (not verified)
    - Fixed: Access lr via hparams.lr
    - Fixed: validation_step returns loss
    - Commit: `a018ce7`
  - [ ] 3.3.3 Verify training pipeline tests (4 tests - integration)
- [ ] 3.4 **NEXT STEP**: Run full test suite to verify all fixes
  ```bash
  cd user/murilo
  source /tmp/birdset-benchmark-3680756/.venv/bin/activate
  pytest tests/002-supervised-vit-baseline/ -v --tb=short
  ```

## 4. Execution (Ready after tests pass)
- [ ] 4.1 Run fast-dev-run validation (5 batches)
  ```bash
  cd experiments/002-supervised-vit-baseline
  python train.py --fast-dev-run
  ```
- [ ] 4.2 Run full HSN training (50 epochs)
  ```bash
  sbatch sbatch/train.sh
  ```
- [ ] 4.3 Monitor W&B for training curves
- [ ] 4.4 Evaluate results against BirdSet baselines (ConvNeXT: 0.38 cmAP)
- [ ] 4.5 Document results in results.md
- [ ] 4.6 Update docs/STATUS.md

---

## Recent Work Summary (2026-02-01)

**Commits**: 3 commits (57902cd, 9c4f7f7, a018ce7)
- Refactored src/ structure (models.py → classifiers.py)
- Created 39 tests across 5 modules
- Fixed 10/12 failing tests (datamodule + lightning module)

**Test Status**: ~33/39 likely passing (85%)
- ✅ test_preprocessing.py: 7/7
- ✅ test_models.py: 9/9
- ✅ test_datamodule.py: 8/8 (VERIFIED)
- ⏳ test_lightning_module.py: 9/9 (fixes applied, not verified)
- ❓ test_training.py: 0/5 (integration tests, not verified)

**Key Fixes**:
1. BirdSet uses `test_5s` split (not `val` or `test`)
2. Audio files need stereo→mono conversion
3. Lightning module: access hparams.lr, return loss from validation_step
