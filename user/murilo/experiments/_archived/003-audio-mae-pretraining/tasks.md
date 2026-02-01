# Tasks: Audio-MAE Pre-training

## 1. Setup
- [ ] 1.1 Verify Experiment 002 completed successfully (infrastructure validated)
- [ ] 1.2 Verify dependencies installed (einops, wandb, timm)
- [ ] 1.3 Create experiment output directories (outputs/slurm, outputs/checkpoints, outputs/wandb)
- [ ] 1.4 Verify BirdSet XCM dataset is accessible

## 2. Implementation - Core Components
- [ ] 2.1 Implement `MAEDecoder` (models/mae_decoder.py)
  - [ ] 2.1.1 Decoder embedding projection (768 → 512)
  - [ ] 2.1.2 Learnable mask token
  - [ ] 2.1.3 Decoder positional embeddings
  - [ ] 2.1.4 Transformer blocks (4 layers, 512 dim, 8 heads)
  - [ ] 2.1.5 Reconstruction head (512 → 256 patch pixels)
- [ ] 2.2 Implement masking logic (lightning/audio_mae.py)
  - [ ] 2.2.1 `random_masking()`: Random shuffle + keep 25% visible
  - [ ] 2.2.2 Return (x_masked, mask, ids_restore)
  - [ ] 2.2.3 Verify mask ratio is exactly 75%
- [ ] 2.3 Implement patch processing (lightning/audio_mae.py)
  - [ ] 2.3.1 `patchify_and_normalize()`: Split into 16x16 patches + normalize
  - [ ] 2.3.2 `unpatchify()`: Reshape patches back to image
  - [ ] 2.3.3 Verify patch shapes (batch, 248, 256)

## 3. Implementation - Lightning Module
- [ ] 3.1 Implement `AudioMAEModule` (lightning/audio_mae.py)
  - [ ] 3.1.1 Initialize encoder (ViT-Base from Experiment 002)
  - [ ] 3.1.2 Initialize decoder (MAEDecoder)
  - [ ] 3.1.3 Forward pass: encode visible → decode all → reconstruct
  - [ ] 3.1.4 Loss computation: MSE on masked patches only
  - [ ] 3.1.5 Training step with loss logging
  - [ ] 3.1.6 Validation step with loss logging
  - [ ] 3.1.7 Configure optimizers: AdamW with cosine + warmup
- [ ] 3.2 Implement `MAEImageLogger` callback (lightning/callbacks.py)
  - [ ] 3.2.1 Get validation batch
  - [ ] 3.2.2 Forward pass to get reconstructions
  - [ ] 3.2.3 Create masked version (fill masked patches with gray)
  - [ ] 3.2.4 Denormalize reconstructions
  - [ ] 3.2.5 Generate figure: [Original | Masked | Reconstructed]
  - [ ] 3.2.6 Log to W&B with wandb.Image()

## 4. Implementation - Training Script & Config
- [ ] 4.1 Create config.yaml with MAE hyperparameters
  - [ ] 4.1.1 Model: mask_ratio=0.75, decoder_depth=4, decoder_dim=512
  - [ ] 4.1.2 Data: batch_size=64, no labels needed
  - [ ] 4.1.3 Training: lr=1.5e-4, weight_decay=0.05, warmup=10, max_epochs=100
  - [ ] 4.1.4 Callbacks: MAEImageLogger every 5 epochs
- [ ] 4.2 Create train.py script
  - [ ] 4.2.1 Load config
  - [ ] 4.2.2 Initialize AudioMAEModule
  - [ ] 4.2.3 Initialize BirdSetDataModule (reuse from Exp 002, ignore labels)
  - [ ] 4.2.4 Setup callbacks: MAEImageLogger, ModelCheckpoint, LRMonitor
  - [ ] 4.2.5 Initialize W&B logger
  - [ ] 4.2.6 Initialize PyTorch Lightning Trainer
  - [ ] 4.2.7 Add --fast-dev-run flag for quick validation
- [ ] 4.3 Create SLURM batch script (sbatch/train.sbatch)
  - [ ] 4.3.1 Request GPU, CPUs, memory (same as Exp 002)
  - [ ] 4.3.2 Set time limit (72 hours for 100 epochs)
  - [ ] 4.3.3 Load modules and activate environment
  - [ ] 4.3.4 Run train.py

## 5. Testing - Unit Tests
- [ ] 5.1 Test `random_masking()`
  - [ ] 5.1.1 Verify mask ratio is exactly 75%
  - [ ] 5.1.2 Verify ids_restore correctly restores order
  - [ ] 5.1.3 Verify visible patches are correct subset
- [ ] 5.2 Test `patchify_and_normalize()`
  - [ ] 5.2.1 Verify patch count: (128/16) * (500/16) = 248
  - [ ] 5.2.2 Verify patch dim: 16*16*1 = 256
  - [ ] 5.2.3 Verify normalization: each patch has mean~0, std~1
- [ ] 5.3 Test MAEDecoder forward pass
  - [ ] 5.3.1 Input: (batch, 62, 768) visible patches
  - [ ] 5.3.2 Output: (batch, 248, 256) reconstructed patches
  - [ ] 5.3.3 No NaN or Inf in output

## 6. Execution - Quick Validation (HSN, fast_dev_run)
- [ ] 6.1 Run fast_dev_run with 5 batches (`python train.py --fast-dev-run`)
- [ ] 6.2 Verify masking: check mask ratio is 75% (186 of 248 patches)
- [ ] 6.3 Verify forward pass completes without errors
- [ ] 6.4 Verify loss computation: MSE computed only on masked patches
- [ ] 6.5 Verify W&B logging: reconstruction images appear
- [ ] 6.6 Check GPU memory usage (should fit in 24GB with batch_size=64)
- [ ] 6.7 Inspect reconstructions: may be blur (untrained), but no errors

## 7. Execution - HSN Pre-training (Quick Test, 20 epochs)
- [ ] 7.1 Train on HSN for 20 epochs (quick validation of full pipeline)
- [ ] 7.2 Monitor loss: should decrease from ~1.5 to <1.0
- [ ] 7.3 Monitor W&B reconstructions every 5 epochs
  - [ ] 7.3.1 Epoch 5: still mostly blur, but hints of structure
  - [ ] 7.3.2 Epoch 10: clear structure emerging
  - [ ] 7.3.3 Epoch 15: good reconstruction quality
  - [ ] 7.3.4 Epoch 20: high-quality reconstructions
- [ ] 7.4 Verify no training instabilities (loss spikes, NaN)
- [ ] 7.5 Check GPU utilization >70%

## 8. Execution - XCM Pre-training (Production, 100 epochs)
- [ ] 8.1 Update config.yaml to use XCM dataset
- [ ] 8.2 Run fast_dev_run on XCM to verify loading works
- [ ] 8.3 Submit SLURM job for full XCM pre-training
- [ ] 8.4 Monitor training progress (48-72 hours)
  - [ ] 8.4.1 Loss decreases monotonically
  - [ ] 8.4.2 Reconstructions improve visually over epochs
  - [ ] 8.4.3 No OOM or training failures
- [ ] 8.5 Save encoder checkpoints every 10 epochs
- [ ] 8.6 Wait for completion (100 epochs or early stopping if loss plateaus)

## 9. Analysis - Reconstruction Quality
- [ ] 9.1 Extract reconstruction images from W&B across epochs (1, 20, 50, 100)
- [ ] 9.2 Qualitative assessment:
  - [ ] 9.2.1 Do reconstructions show frequency structure?
  - [ ] 9.2.2 Do reconstructions show temporal patterns?
  - [ ] 9.2.3 Are masked regions filled plausibly?
  - [ ] 9.2.4 Compare early (epoch 1) vs late (epoch 100) quality
- [ ] 9.3 Quantitative analysis:
  - [ ] 9.3.1 Plot MSE loss over epochs (train + val)
  - [ ] 9.3.2 Verify final MSE loss <0.5 (target)
  - [ ] 9.3.3 Check for convergence (loss plateau indicates training complete)
- [ ] 9.4 Error analysis:
  - [ ] 9.4.1 Which samples reconstruct poorly? (e.g., silent backgrounds)
  - [ ] 9.4.2 Which samples reconstruct well? (e.g., clear vocalizations)

## 10. Analysis - Encoder Quality (Qualitative)
- [ ] 10.1 Extract encoder embeddings for validation samples
- [ ] 10.2 Visualize embeddings with t-SNE or UMAP
  - [ ] 10.2.1 Do samples cluster by species?
  - [ ] 10.2.2 Do similar vocalizations group together?
- [ ] 10.3 Compare to random encoder embeddings (sanity check)
- [ ] 10.4 Discuss: Does pre-training learn meaningful representations?

## 11. Documentation
- [ ] 11.1 Document findings in results.md
  - [ ] 11.1.1 Final MSE loss (train + val)
  - [ ] 11.1.2 Reconstruction quality assessment (qualitative)
  - [ ] 11.1.3 Training curves (loss over epochs)
  - [ ] 11.1.4 Sample reconstructions (best, worst, typical)
  - [ ] 11.1.5 GPU utilization and training time
- [ ] 11.2 Save encoder checkpoints to artifacts/
  - [ ] 11.2.1 Best checkpoint (lowest val loss)
  - [ ] 11.2.2 Last checkpoint (epoch 100)
  - [ ] 11.2.3 Intermediate checkpoints (epoch 20, 50)
- [ ] 11.3 Update Status in proposal.md to "completed"
- [ ] 11.4 Document lessons learned and next steps

## 12. Optional - Ablation Studies (if time permits)
- [ ] 12.1 Mask ratio ablation: 50%, 60%, 75%, 80%
- [ ] 12.2 Decoder depth ablation: 2, 4, 6, 8 layers
- [ ] 12.3 Without patch normalization: does it hurt training?
- [ ] 12.4 Block masking vs random masking
- [ ] 12.5 Different learning rates: 1e-4, 1.5e-4, 2e-4

## 13. Next Steps - Downstream Evaluation
- [ ] 13.1 Fine-tune encoder on supervised classification (compare to Exp 002)
- [ ] 13.2 Zero-shot retrieval: query-by-example
- [ ] 13.3 Linear probe: freeze encoder, train linear head
- [ ] 13.4 Compare MAE pre-trained vs ImageNet pre-trained vs random init
