# Experiment 005: Pretrained Audio Model Fine-Tuning - Tasks

## 1. Data Acquisition

- [x] 1.1 Write download script (`scripts/download_birdclef_data.py`)
- [x] 1.2 Download BirdCLEF 2025 dataset to shared project storage (`~/ps-clef2026_birdclef-0/data/birdclef-2025/`)
  - 206 species, 28,564 audio files, 11.5 GB
- [x] 1.3 Verify download integrity (file counts, total size, spot-check audio files)

## 2. Exploratory Data Analysis

- [x] 2.1 Create EDA notebook (`notebooks/eda_birdclef2025.ipynb`)
- [x] 2.2 Analyze `train.csv` — species distribution, recording durations, geographic coverage
  - 206 species, 28,564 recordings, 3 collections (XC, CSA, iNaturalist)
  - 97.2% have coordinates, concentrated in South America
- [x] 2.3 Analyze audio properties — sample rates, duration histogram, silence detection
  - Uniform 32kHz sample rate, duration 1.2s–382.3s (median 30s)
- [x] 2.4 Identify class imbalance and long-tail species
  - Severe long-tail: max 990 (grekis), median 80, min 2 per species
- [ ] 2.5 Compare BirdCLEF 2025 species overlap with BirdSet HSN/XCM datasets
- [x] 2.6 Summarize findings in EDA notebook summary section

## 3. Setup

- [x] 3.1 Create `pyproject.toml` with dependencies (torch, torchaudio, BEATs/EAT, nnAudio)
- [x] 3.2 Create BirdCLEF 2025 PyTorch Dataset class in shared `src/birdclef/data/birdclef2025.py`
  - `BirdCLEF2025Dataset`: 5s non-overlapping chunks at 16kHz, multi-hot labels, sf.read partial reads
  - `BirdCLEF2025DataModule`: stratified train/val/test split by primary_label (80/10/10)
  - `build_species_list()`: sorted unique primary_label values
- [x] 3.3 Create DataModule for BirdCLEF 2025 (adapted from Anthony's `BirdCLEFDataModule` pattern)
- [x] 3.4 Write `proposal.md` with hypothesis, method, and evaluation plan
- [ ] 3.5 Download BEATs/EAT pretrained checkpoints to shared storage (EAT auto-downloads via HuggingFace)

## 4. Model Implementation

- [x] 4.1 Create EAT encoder wrapper (`src/birdclef/encoders/eat.py`) — HuggingFace, Kaldi fbank, mean pool, freeze/unfreeze
- [x] 4.2 Create BEATs encoder wrapper (`src/birdclef/encoders/beats.py`) — vendor checkpoint, freeze/unfreeze
- [x] 4.3 Create `PretrainedFinetuneModule` (`src/birdclef/lightning/pretrained_finetune.py`)
  - Staged unfreezing, dual LR (head + encoder), AdamW + cosine annealing + warmup
  - Full metrics: cmAP, AUROC, F1 macro/micro, top-1/top-5
- [x] 4.4 Create config files: `config/eat_frozen.yaml` (linear probe), `config/eat_finetune.yaml` (staged fine-tune)
- [x] 4.5 Validate with fast-dev-run
  - 194,489 train chunks, 21,776 val chunks (206 classes)
  - 90.1M total params (158K trainable when frozen, 90M non-trainable)
  - Pipeline verified: data loading, encoder forward, loss, metrics all working

## 5. Evaluation Setup

- [x] 5.1 Update DataModule to support 80/10/10 stratified train/val/test split
  - Added `test_fraction` param, `_split_dataframe()` three-way split, `test_dataloader()`
  - Handles edge cases for small groups (min 1 per split, at least 1 train sample)
- [x] 5.2 Update YAML configs with new split fractions (`val_fraction: 0.1`, `test_fraction: 0.1`)
- [x] 5.3 Add test-set evaluation to `train.py` — run `trainer.test()` on best checkpoint after training
  - Added `test_step` / `on_test_epoch_end` to `PretrainedFinetuneModule`
  - `train.py` runs `trainer.test()` with best checkpoint, saves test metrics to results JSON
- [x] 5.4 Re-validate fast-dev-run with updated 80/10/10 split

## 6. Training

- [x] 6.1 Create SLURM scripts (`sbatch/_job.sh`, `sbatch/run.sh`) with HF_HOME caching
- [x] 6.2 Create training script (`train.py`) with YAML config loading, W&B/TensorBoard logging
- [x] 6.2b Add `save_metrics.py` to write structured metrics to `artifacts/training/<variant>/metrics.json`
  - Auto-runs from `_job.sh` after training (skips gpu-check and fast-dev-run)
  - Frozen baseline saved: cmAP 0.267, AUROC 0.879 on held-out test set
- [x] 6.3 Run frozen-encoder baseline (linear probe) — completed (job 4105049, 9h15m)
- [x] 6.4 Run full fine-tuning (staged unfreezing) — timed out (job 4105050, 12hr limit hit)
- [x] 6.5 Check if finetune run completed or was killed by SLURM time limit
  - Confirmed TIMEOUT via `sacct`: job 4105050 hit 12:00:24 walltime
- [~] 6.6 Re-run frozen + finetune with 12 workers after GPU optimization
  - Frozen: running with 12 workers
  - Finetune: OOM at epoch 5 when encoder unfroze (job 4114047) — batch_size=256
    fills 23+ GB VRAM once 90M encoder params need gradients + Adam states
  - Fix: reduce batch_size to 32 with `accumulate_grad_batches=8` (effective 256)
    to keep same training dynamics in smaller VRAM footprint
- [ ] 6.7 Re-submit finetune with gradient accumulation (batch_size=32, accumulate=8)
- [ ] 6.8 Monitor training via W&B and collect results

## 6b. GPU Utilization Optimization

GPU profiling revealed a data pipeline bottleneck: 8 CPU workers computing
fbanks (~1056ms/batch) couldn't keep the GPU fed (~477ms/batch compute).
**Resolved** by increasing to 12 workers (max for RTX 6000 batch jobs),
which reduced data time from 61ms to 5ms avg and eliminated the bottleneck.
See `gpu-utilization-analysis.md` for full analysis, diagrams, and future options.

- [x] 6b.1 Move fbank conversion from EAT forward pass to dataloader workers
  - Extracted `waveform_to_fbank()` as module-level function in `eat.py`
  - Added `precomputed_features` flag to `EATEncoder`, `transform` param to dataset/datamodule
- [x] 6b.2 Profile GPU utilization with `--gpu-check` flag and document findings
  - GPU: 84.8% avg, 3% min — spiky due to data starvation
  - VRAM: 2.6% used (0.66 GB / 25.2 GB) — massive headroom
  - Data time spikes to 4.7s (10x compute time)
- [x] 6b.3 Add `--num-workers` passthrough to `run.sh`/`_job.sh` with auto `--cpus-per-task` override
  - PACE CPU:GPU limits: 6:1 interactive (salloc), 12:1 batch (sbatch) on gpu-rtx6000/gpu-v100
- [x] 6b.4 Re-run `--gpu-check` with 8 and 12 workers to compare
  - 8 workers (job 4113823): GPU 87.3%, data time 61ms avg, throughput 502 sam/s
  - 12 workers (job 4113831): GPU 85.1%, data time 5ms avg, throughput 523 sam/s
  - 12 workers virtually eliminated data bottleneck (0.8% data loading vs 7.5%)
- [x] 6b.5 Add GPU utilization plot (nvtop-style) to `--gpu-check` output
  - Two-panel chart: GPU util% + memory% on top, data vs compute time on bottom
  - Added run metadata (GPU name, job ID, workers, batch size) and timestamp to summary
- [ ] 6b.6 (Optional) Pre-compute EAT fbanks to disk for maximum throughput
  - One-time ~5 min preprocessing, eliminates CPU bottleneck entirely
  - EAT-specific — other models (BEATs) would not benefit from cached fbanks
  - Only needed if 12 workers is insufficient

## 7. Analysis

- [ ] 7.1 Compare cmAP between frozen and fine-tuned variants on held-out test set
- [ ] 7.2 Analyze per-species performance and failure modes
- [ ] 7.3 Document results in `results.md`
- [ ] 7.4 Identify follow-up directions (soundscape evaluation, Kaggle submission, class imbalance handling)
