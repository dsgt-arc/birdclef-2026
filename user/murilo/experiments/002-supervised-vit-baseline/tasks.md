# Tasks: Supervised ViT Baseline

## 1. Setup
- [x] 1.1 Add dependencies to pyproject.toml (timm, wandb, einops)
- [x] 1.2 Create directory structure (lightning/, models/, preprocessing/)
- [ ] 1.3 Install dependencies with `uv sync`
- [ ] 1.4 Verify BirdSet HSN dataset is accessible
- [ ] 1.5 Create experiment output directories (outputs/slurm, outputs/checkpoints, outputs/wandb)

## 2. Implementation
- [x] 2.1 Implement `MelSpectrogramTransform` (preprocessing/mel_spectrogram.py)
- [x] 2.2 Implement `ViTAudio` model (models/vit_audio.py)
- [x] 2.3 Implement `BirdSetDataModule` (lightning/birdset_datamodule.py)
- [x] 2.4 Implement `SupervisedViTModule` (lightning/supervised_vit.py)
- [x] 2.5 Create training script (experiments/002-supervised-vit-baseline/train.py)
- [x] 2.6 Create config file (experiments/002-supervised-vit-baseline/config.yaml)
- [x] 2.7 Create SLURM batch script (experiments/002-supervised-vit-baseline/sbatch/train.sbatch)
- [ ] 2.8 Write unit tests for mel-spectrogram transform
- [ ] 2.9 Write unit tests for ViTAudio forward pass

## 3. Execution - Quick Validation (HSN)
- [ ] 3.1 Run fast_dev_run with 5 batches (`python train.py --fast-dev-run`)
- [ ] 3.2 Verify mel-spectrogram shapes (batch, 1, 128, 500)
- [ ] 3.3 Verify labels are multi-hot (batch, 21 for HSN)
- [ ] 3.4 Verify no NaN/Inf in mel-spectrograms or loss
- [ ] 3.5 Check GPU memory usage (should fit in 24GB with batch_size=32)

## 4. Execution - HSN Training (Infrastructure Validation)
- [ ] 4.1 Start training on HSN with default config
- [ ] 4.2 Monitor first epoch: GPU utilization >80%, data loading <10% overhead
- [ ] 4.3 Verify W&B logging: loss curves, metrics, learning rate
- [ ] 4.4 Verify model checkpointing saves top-3 and last checkpoint
- [ ] 4.5 Let training run to completion (50 epochs or early stopping)
- [ ] 4.6 Check final validation cmAP > 0.30

## 5. Execution - Scale to XCM
- [ ] 5.1 Update config.yaml to use XCM dataset
- [ ] 5.2 Verify XCM dataset downloads successfully (~90GB)
- [ ] 5.3 Run fast_dev_run to verify XCM loading works
- [ ] 5.4 Submit SLURM job for full XCM training
- [ ] 5.5 Monitor training: loss convergence, cmAP improvement
- [ ] 5.6 Wait for completion or early stopping

## 6. Analysis
- [ ] 6.1 Extract final metrics from W&B (train/val cmAP, F1, Top-K accuracy)
- [ ] 6.2 Analyze GPU utilization logs (check for bottlenecks)
- [ ] 6.3 Compare HSN vs XCM convergence patterns
- [ ] 6.4 Profile data loading overhead (time per batch)
- [ ] 6.5 Generate training curves (loss, cmAP over epochs)
- [ ] 6.6 Evaluate on test set with best checkpoint
- [ ] 6.7 Document findings in results.md
- [ ] 6.8 Update Status in proposal.md to "completed"

## 7. Optional Improvements (if time permits)
- [ ] 7.1 Tune batch size for maximum GPU utilization
- [ ] 7.2 Experiment with different num_workers for optimal throughput
- [ ] 7.3 Try different learning rates (grid search)
- [ ] 7.4 Visualize attention maps from ViT
- [ ] 7.5 Error analysis: which species are hardest to classify?
