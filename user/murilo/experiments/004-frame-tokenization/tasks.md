# Experiment 004: Frame-Based Tokenization - Tasks

## 1. Setup

- [x] 1.1 Create experiment directory structure and `pyproject.toml`
- [x] 1.2 Pipeline validation — fast-dev-run completed successfully (baseline_16x16)
- [ ] 1.3 Verify exp 003 scratch baseline is reproducible (re-run ViT-Small 16×16 scratch with fixed seed)

## 2. Implementation

- [x] 2.1 Create `FrameTokenizedViT` model with configurable `patch_size` tuple
  - Created `src/birdset_benchmark/models/frame_vit.py` (standalone, not extending AudioClassifier)
  - Passes `patch_size=(freq, time)` and `img_size=(128, 500)` to `timm.create_model`
  - 1-channel input adaptation works with non-square patch embedding kernel
  - timm supports non-square `patch_size` — confirmed via fast-dev-run
- [x] 2.2 Create config files for each configuration:
  - `config/baseline_16x16.yaml` — ViT-Small scratch, patch_size: [16, 16]
  - `config/vertical_128x2.yaml` — ViT-Small scratch, patch_size: [128, 2]
  - `config/vertical_128x4.yaml` — ViT-Small scratch, patch_size: [128, 4]
  - `config/vertical_128x1.yaml` — ViT-Small scratch, patch_size: [128, 1]
- [x] 2.3 Create `train.py` using FrameTokenizedViT with patch_size config support
- [x] 2.4 Create SLURM scripts for batch training all configurations
  - `sbatch/_job.sh` — venv in $TMPDIR (safe), uv install, PyTorch CUDA 12.1
  - `sbatch/run.sh` — launcher with `--all` flag to submit all 4 configs
- [x] 2.5 Create tokenization visualization notebook (`notebooks/visualize_tokenization.ipynb`)
- [x] 2.6 Add GPU-accelerated mel spectrogram via nnAudio (`NNAudioMelSpectrogram`)
  - Drop-in replacement for `MelSpectrogramTransform` that runs on GPU
  - 10/10 tests passing (shapes, GPU, torchaudio consistency, edge cases)

## 3. Integration

- [x] 3.1 Integrate `NNAudioMelSpectrogram` into the training pipeline
  - `FrameTokenizedViT` now always creates `NNAudioMelSpectrogram` internally (GPU mel is default)
  - `train.py` passes mel params (sample_rate, n_mels, etc.) to model and sets `waveform_only=True` on DataModule
  - DataModule pre-loads all waveforms into RAM via `_CachedWaveformDataset`; model computes mel on GPU
  - Removed `cache_mel_specs` from all config files (no longer needed)
  - Fixed GPU starvation: disk I/O (sf.read + librosa.resample) was the bottleneck, not mel computation
  - Bumped num_workers 6→8, prefetch_factor 4→8 (matching Anthony's exp 008)
- [x] 3.2 Add pytest tests for DataModule + nnAudio GPU integration
  - `tests/004-frame-tokenization/test_pipeline_integration.py`
  - `TestWaveformDataset` (8 tests): shapes, dtypes, multi-hot labels, DataLoader batching
  - `TestFrameTokenizedViT` (7 tests): forward pass, all 4 patch configs, token counts, NaN/Inf, channel check
  - `TestGPUIntegration` (4 tests): GPU forward, mel-on-GPU hook, all configs on GPU, DataLoader→GPU pipeline
- [x] 3.3 Fast-dev-run validated with nnAudio GPU mel (baseline_16x16)
  - Logs confirm `mode: waveform (GPU mel via nnAudio)` — no more CPU pre-compute
  - 21.5M params, 248 tokens, train/val/test all completed successfully
- [x] 3.4 Verify positional embeddings are correctly sized for each patch configuration
  - Added `test_pos_embed_shape` — parametrized across all 4 configs, 8/8 tests passing
  - 16×16 → (1, 249, 384), 128×2 → (1, 251, 384), 128×4 → (1, 126, 384), 128×1 → (1, 501, 384)
- [x] 3.5 Run fast-dev-run for remaining 3 configs (128×2, 128×4, 128×1)
  - All 4 configs validated end-to-end with nnAudio GPU mel

## 4. Execution

- [x] 4.1 Submit all 4 training runs via `sbatch/run.sh --all`
  - Baseline (16×16), Config A (128×2), Config B (128×4), Config C (128×1)
  - All submitted — check SLURM logs and W&B for results
- [ ] 4.2 Check SLURM job completion — verify all 4 runs finished successfully
  - Review logs in `logs/slurm_*.log` for errors or OOM
  - Confirm checkpoints saved in project shared storage
  - Verify test metrics JSON saved in `outputs/test_results/`
- [ ] 4.3 Check GPU utilization in W&B DeviceStatsMonitor
  - Verify cached waveform loading improved GPU utilization vs. the earlier spiky pattern
- [ ] 4.4 Monitor training via W&B, check for convergence issues

## 5. Analysis

- [ ] 5.1 Compare cmAP across all 4 configurations
- [ ] 5.2 Analyze training curves (convergence speed, overfitting patterns)
- [ ] 5.3 Compare walltime per epoch across configs (token count affects attention cost)
- [ ] 5.4 Document results in `results.md`
- [ ] 5.5 Update concept doc `temporal-resolution-sequence-length.md` with findings

## 6. Optimization (optional)

- [ ] 6.1 Cache pre-loaded waveforms to `.pt` files on disk
  - Pre-load takes ~25 min (train: 1025s, val: 528s) due to sf.read + librosa.resample
  - Save `HSN_train_waveforms.pt` / `HSN_val_waveforms.pt` after first pre-load
  - Subsequent runs load from `.pt` in ~5s instead of re-reading all audio files
  - nnAudio handles mel on GPU correctly; the bottleneck is disk I/O, not mel computation
