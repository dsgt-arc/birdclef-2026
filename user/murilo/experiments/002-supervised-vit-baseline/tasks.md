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

## 3. Execution
- [ ] 3.1 Run fast-dev-run validation (5 batches)
- [ ] 3.2 Run full HSN training (50 epochs)
- [ ] 3.3 Monitor W&B for training curves
- [ ] 3.4 Evaluate results against BirdSet baselines (ConvNeXT: 0.38 cmAP)
- [ ] 3.5 Document results in results.md
- [ ] 3.6 Update docs/STATUS.md
