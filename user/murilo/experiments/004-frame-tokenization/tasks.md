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

## 3. Next Steps — Integration

- [x] 3.1 Integrate `NNAudioMelSpectrogram` into the training pipeline
  - `FrameTokenizedViT` now always creates `NNAudioMelSpectrogram` internally (GPU mel is default)
  - `train.py` passes mel params (sample_rate, n_mels, etc.) to model and sets `waveform_only=True` on DataModule
  - DataModule returns raw waveforms via `_WaveformDataset`; model computes mel on GPU
  - Removed `cache_mel_specs` from all config files (no longer needed)
- [ ] 3.2 Add pytest tests for DataModule + nnAudio GPU integration
  - Verify `_WaveformDataset` returns correct shapes (raw waveforms + labels)
  - Verify `FrameTokenizedViT` end-to-end: waveform in → logits out
  - Verify GPU utilization: mel computation happens on GPU (not CPU)
  - Test all 4 patch configurations produce correct token counts
- [ ] 3.3 Verify positional embeddings are correctly sized for each patch configuration
  - 16×16 → 248 positions, 128×2 → 250 positions, 128×4 → 125 positions, 128×1 → 500 positions
  - Run fast-dev-run for each config to confirm shapes
- [ ] 3.4 Run fast-dev-run for all 4 configs to validate pipeline end-to-end

## 4. Execution

- [ ] 4.1 Run baseline (16×16) with same seed — controlled comparison
- [ ] 4.2 Run Config A (128×2) — primary hypothesis test
- [ ] 4.3 Run Config B (128×4) — coarser temporal resolution comparison
- [ ] 4.4 Run Config C (128×1) — finest temporal resolution comparison
- [ ] 4.5 Monitor training via W&B, check for convergence issues

## 5. Analysis

- [ ] 5.1 Compare cmAP across all 4 configurations
- [ ] 5.2 Analyze training curves (convergence speed, overfitting patterns)
- [ ] 5.3 Compare walltime per epoch across configs (token count affects attention cost)
- [ ] 5.4 Document results in `results.md`
- [ ] 5.5 Update concept doc `temporal-resolution-sequence-length.md` with findings
