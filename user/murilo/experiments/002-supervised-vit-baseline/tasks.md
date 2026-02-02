# Tasks: Supervised ViT Baseline

## 1. Setup
- [x] 1.1 Add dependencies to pyproject.toml (timm, wandb, einops)
- [x] 1.2 Create directory structure (lightning/, models/, preprocessing/)
- [ ] 1.3 Install dependencies with `uv sync`
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

## 3. Testing & Debugging (In Progress)
- [x] 3.1 Create pytest test suite (39 tests across 5 modules)
- [x] 3.2 Run initial test suite (27/39 passing)
- [ ] 3.3 Debug failing tests:
  - [ ] 3.3.1 Fix datamodule tests (6 failures)
  - [ ] 3.3.2 Fix lightning module tests (2 failures)
  - [ ] 3.3.3 Fix training pipeline tests (4 failures)
- [ ] 3.4 Verify all tests passing

## 4. Execution
- [ ] 4.1 Run fast-dev-run validation (5 batches)
- [ ] 4.2 Run full HSN training (50 epochs)
- [ ] 4.3 Monitor W&B for training curves
- [ ] 4.4 Evaluate results against BirdSet baselines (ConvNeXT: 0.38 cmAP)
- [ ] 4.5 Document results in results.md
- [ ] 4.6 Update docs/STATUS.md
