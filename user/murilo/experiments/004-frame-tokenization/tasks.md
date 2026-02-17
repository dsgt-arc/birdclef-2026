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
- [x] 4.2 Check SLURM job completion — verify all 4 runs finished successfully
  - All 4 trained and tested successfully; 128×4 and 128×1 crashed post-training in metric saving (epoch parsing bug)
  - Fixed `train.py:294` — checkpoint `-v1` suffix from PL naming collision
  - Recovered metrics from SLURM logs, wrote corrected `artifacts/training/` metrics.json
- [x] 4.3 Check GPU utilization in W&B DeviceStatsMonitor
  - Cached waveform loading confirmed via `artifacts/gpu-utilization.png`
- [x] 4.4 Monitor training via W&B, check for convergence issues
  - All vertical frame configs peaked at epoch 2; baseline at epoch 7; all early-stopped at epoch 17–18

## 5. Analysis

- [x] 5.1 Compare cmAP across all 4 configurations
  - B: 128×4 (0.063) > A: 128×2 (0.054) > C: 128×1 (0.051) > Baseline (0.029)
  - All vertical configs outperform 16×16 by 1.8–2.2×
- [x] 5.2 Analyze training curves (convergence speed, overfitting patterns)
  - Vertical frames converge 3.5× faster (epoch 2 vs 7)
  - All configs show overfitting (train cmAP >> test cmAP)
- [x] 5.3 Compare walltime per epoch across configs (token count affects attention cost)
  - 128×4: 5.95 it/s, 128×2: 4.15 it/s, 16×16: 4.16 it/s, 128×1: 2.34 it/s
  - Scales with O(n²) attention cost; job walltimes similar (~37–43 min) due to data loading dominance
- [x] 5.4 Document results in `results.md`
- [ ] 5.5 Update concept doc `temporal-resolution-sequence-length.md` with findings

## 6. Analysis — Verdict Update

- [x] 6.1 Identified mel backend confound: nnAudio (exp 004) vs torchaudio (exp 003) caused 16×16 baseline to drop from cmAP=0.073 → 0.029
  - Within-experiment ranking is valid (all 4 configs used nnAudio)
  - Cross-experiment comparison to exp 003 is unreliable
  - Updated `results.md` verdict to "inconclusive (pending torchaudio rerun)"

## 7. Torchaudio Mel Rerun

Re-run all 4 configs with the original torchaudio mel backend (`cache_mel_specs: true`)
to isolate the tokenization variable from the mel backend change.

- [x] 7.1 Add mel bypass in `FrameTokenizedViT.forward()` — dispatch on `x.dim()` (2D=waveform, 4D=pre-computed mel)
- [x] 7.2 Update `train.py` — read `cache_mel_specs` from config, set `waveform_only = not cache_mel_specs`
- [x] 7.3 Create 4 torchaudio config files (suffix `-torchaudio`, add `data.cache_mel_specs: true`)
  - `config/baseline_16x16_torchaudio.yaml`
  - `config/vertical_128x2_torchaudio.yaml`
  - `config/vertical_128x4_torchaudio.yaml`
  - `config/vertical_128x1_torchaudio.yaml`
- [x] 7.4 Validate with fast-dev-run: `bash sbatch/run.sh config/baseline_16x16_torchaudio.yaml --fast-dev-run`
  - Confirmed logs show `mode: cached` and `cache_mel_specs=True, waveform_only=False`
- [x] 7.5 Submit all 4 torchaudio runs
  - All 4 completed successfully (SLURM jobs 4017565, 4017567, 4017569, 4017568)
- [x] 7.6 Compare torchaudio baseline to exp 003 baseline
  - Torchaudio baseline cmAP=0.066 vs exp 003 cmAP=0.073 (within 10%) — mel backend confirmed as primary confound
- [x] 7.7 Compare torchaudio vertical configs to torchaudio baseline — final verdict
  - A: 128×2 (0.098) > B: 128×4 (0.087) ≈ C: 128×1 (0.086) > Baseline (0.066)
  - All vertical configs beat baseline by +30–49%; all beat exp 003's 0.073
  - Ranking changed: 128×2 is now best (was 128×4 with nnAudio)
  - **Verdict: confirmed** — vertical frame tokenization improves cmAP by 30–49%
- [x] 7.8 Update `results.md` with torchaudio rerun results

## 8. Fix nnAudio Mel Filterbank

Root cause identified: `NNAudioMelSpectrogram` doesn't override nnAudio defaults
that differ from torchaudio's. Two mel filterbank parameters are mismatched:

| Parameter | nnAudio default | torchaudio default | Fix |
|-----------|----------------|-------------------|-----|
| Mel scale | `htk=False` (Slaney) | `mel_scale="htk"` (HTK) | Pass `htk=True` |
| Mel norm | `norm=1` (area norm) | `norm=None` (no norm) | Pass `norm=None` |

STFT, power (2.0), window (Hann), fmin/fmax are all identical — the filterbank
construction is the only difference.

- [ ] 8.1 Fix `NNAudioMelSpectrogram`: pass `htk=True, norm=None` to `nnAudio.features.MelSpectrogram(...)`
  - File: `src/birdset_benchmark/preprocessing/nnaudio_mel.py`
  - Also align log epsilon: `torch.log(mel + 1e-6)` instead of `torch.log(mel.clamp(min=1e-6))` to match torchaudio
- [ ] 8.2 Add pytest: numerical comparison of nnAudio vs torchaudio output
  - Generate mel from a reference waveform using both `NNAudioMelSpectrogram` and `MelSpectrogramTransform`
  - Assert `torch.allclose(nnaudio_mel, torchaudio_mel, atol=1e-3)` — outputs should be nearly identical after fix
  - Test with real audio (from HSN) and synthetic waveforms (sine, silence, noise)
  - File: `tests/004-frame-tokenization/test_nnaudio_torchaudio_parity.py`
- [ ] 8.3 Update existing nnAudio tests in `test_pipeline_integration.py` if any assertions change
- [ ] 8.4 Fast-dev-run with fixed nnAudio (`waveform_only=True`, no `cache_mel_specs`)
  - Confirm logs show `mode: waveform` and training completes
  - Compare mel output shape/range to torchaudio baseline
- [x] 8.5 Re-run all 4 nnAudio configs with fixed filterbank
  - Use original configs (no `cache_mel_specs`) — model computes mel on GPU via fixed nnAudio
  - Expect results close to torchaudio rerun (baseline ~0.066, 128×2 ~0.098)
  - Results unchanged — confirming filterbank wasn't the issue; AMP precision is the likely cause
- [x] 8.6 Update `results.md` with fixed nnAudio rerun results

## 9. Optimization (optional)

See `docs/concepts/audio-pipeline-bottlenecks.md` for full analysis of the
disk I/O vs GPU mel computation problem discovered during this experiment.

- [ ] 9.1 Cache pre-loaded waveforms to `.pt` files on disk
  - Pre-load takes ~25 min (train: 1025s, val: 528s) due to sf.read + librosa.resample
  - Save `HSN_train_waveforms.pt` / `HSN_val_waveforms.pt` after first pre-load
  - Subsequent runs load from `.pt` in ~5s instead of re-reading all audio files
  - nnAudio handles mel on GPU correctly; the bottleneck is disk I/O, not mel computation

## 10. Streaming Data Pipeline + AMP Precision Fix

- [x] 10.1 Create streaming_datamodule.py (StreamingBirdSetDataset, RecordingShuffleSampler, StreamingBirdSetDataModule)
- [x] 10.2 Fix AMP precision in FrameTokenizedViT.forward() — autocast(enabled=False) around mel
- [x] 10.3 Update train.py — streaming mode dispatch on data.streaming config key
- [x] 10.4 Create 4 streaming config files
- [x] 10.5 Update run.sh with --all-streaming flag
- [x] 10.6 Export from lightning/__init__.py
- [ ] 10.7 Fast-dev-run validation
- [ ] 10.8 Submit all 4 streaming runs
- [ ] 10.9 Monitor GPU utilization during training
- [ ] 10.10 Analyze results and update results.md
