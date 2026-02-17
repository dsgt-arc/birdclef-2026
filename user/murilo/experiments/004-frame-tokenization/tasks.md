# Experiment 004: Frame-Based Tokenization - Tasks

## 1. Setup

- [x] 1.1 Create experiment directory structure and `pyproject.toml`
- [x] 1.2 Pipeline validation — fast-dev-run completed successfully (baseline_16x16)

## 2. Implementation

- [x] 2.1 Create `FrameTokenizedViT` model with configurable `patch_size` tuple
- [x] 2.2 Create config files for all 4 configurations (baseline_16x16, vertical_128x2, 128x4, 128x1)
- [x] 2.3 Create `train.py` using FrameTokenizedViT with patch_size config support
- [x] 2.4 Create SLURM scripts for batch training all configurations
- [x] 2.5 Create tokenization visualization notebook
- [x] 2.6 Add GPU-accelerated mel spectrogram via nnAudio (`NNAudioMelSpectrogram`)

## 3. Integration

- [x] 3.1 Integrate `NNAudioMelSpectrogram` into the training pipeline
- [x] 3.2 Add pytest tests for DataModule + nnAudio GPU integration
- [x] 3.3 Fast-dev-run validated with nnAudio GPU mel
- [x] 3.4 Verify positional embeddings are correctly sized for each patch configuration
- [x] 3.5 Run fast-dev-run for remaining 3 configs

## 4. Execution (nnAudio runs)

- [x] 4.1 Submit all 4 training runs via `sbatch/run.sh --all`
- [x] 4.2 All 4 runs finished successfully
- [x] 4.3 GPU utilization confirmed via W&B DeviceStatsMonitor
- [x] 4.4 All vertical configs peaked at epoch 2; baseline at epoch 7; early-stopped at epoch 17–18

## 5. Analysis (nnAudio runs)

- [x] 5.1 Compare cmAP: B: 128×4 (0.063) > A: 128×2 (0.054) > C: 128×1 (0.051) > Baseline (0.029)
- [x] 5.2 Vertical frames converge 3.5× faster (epoch 2 vs 7), all configs show overfitting
- [x] 5.3 Walltime scales with O(n²) attention cost; ~37–43 min per job
- [x] 5.4 Document results in `results.md`
- [ ] 5.5 Update concept doc `temporal-resolution-sequence-length.md` with findings

## 6. Verdict Update

- [x] 6.1 Identified mel backend confound: nnAudio vs torchaudio caused baseline drop from 0.073 → 0.029
  - Updated verdict to "inconclusive (pending torchaudio rerun)"

## 7. Torchaudio Mel Rerun

- [x] 7.1 Add mel bypass in `FrameTokenizedViT.forward()` — dispatch on `x.dim()`
- [x] 7.2 Update `train.py` — read `cache_mel_specs` from config
- [x] 7.3 Create 4 torchaudio config files
- [x] 7.4 Validate with fast-dev-run
- [x] 7.5 Submit all 4 torchaudio runs
- [x] 7.6 Torchaudio baseline cmAP=0.066 vs exp 003 cmAP=0.073 — mel backend confirmed as confound
- [x] 7.7 Final verdict: A: 128×2 (0.098) > B: 128×4 (0.087) ≈ C: 128×1 (0.086) > Baseline (0.066)
  - **Verdict: confirmed** — vertical frame tokenization improves cmAP by 30–49%
- [x] 7.8 Update `results.md` with torchaudio rerun results

## 8. Fix nnAudio Mel Filterbank

Fixed `htk=True, norm=None` in NNAudioMelSpectrogram. Rerun showed results unchanged —
filterbank wasn't the cause of the nnAudio vs torchaudio gap. AMP precision identified as likely cause.

- [x] 8.1 Fix NNAudioMelSpectrogram filterbank params
- [ ] 8.2 Add pytest: nnAudio vs torchaudio numerical parity
- [ ] 8.3 Update existing nnAudio tests if assertions change
- [ ] 8.4 Fast-dev-run with fixed nnAudio
- [x] 8.5 Re-run all 4 nnAudio configs — results unchanged
- [x] 8.6 Update results.md

## 9. Optimization (optional, superseded by streaming)

- [ ] 9.1 Cache pre-loaded waveforms to `.pt` files on disk
  - Largely superseded by streaming pipeline (section 10) which avoids pre-loading entirely

## 10. Streaming Data Pipeline + AMP Precision Fix

Two changes: (1) streaming pipeline splits full recordings into all non-overlapping 5s
chunks (~6x more training data per epoch, no RAM pre-load, near-instant startup),
(2) AMP precision fix prevents nnAudio mel from running in float16.

- [x] 10.1 Create streaming_datamodule.py (StreamingBirdSetDataset, RecordingShuffleSampler, StreamingBirdSetDataModule)
- [x] 10.2 Fix AMP precision — autocast(enabled=False) around mel in FrameTokenizedViT
- [x] 10.3 Update train.py — streaming mode dispatch on `data.streaming` config key
- [x] 10.4 Create 4 streaming config files
- [x] 10.5 Update run.sh with --all-streaming flag
- [x] 10.6 Export from lightning/__init__.py
- [ ] 10.7 Update `docs/concepts/audio-pipeline-bottlenecks.md` with streaming findings
- [x] 10.8 Fast-dev-run validation
- [x] 10.9 Submit all 4 streaming runs — training in progress
- [ ] 10.10 Monitor GPU utilization during training
- [ ] 10.11 Analyze results and update results.md
