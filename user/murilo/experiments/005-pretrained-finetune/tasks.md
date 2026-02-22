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
  - `BirdCLEF2025DataModule`: stratified train/val split by primary_label (90/10)
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

## 5. Training

- [x] 5.1 Create SLURM scripts (`sbatch/_job.sh`, `sbatch/run.sh`) with HF_HOME caching
- [x] 5.2 Create training script (`train.py`) with YAML config loading, W&B/TensorBoard logging
- [ ] 5.3 Run frozen-encoder baseline (linear probe)
- [ ] 5.4 Run full fine-tuning with lower learning rate
- [ ] 5.5 Monitor training via W&B

## 6. Analysis

- [ ] 6.1 Compare cmAP between frozen and fine-tuned variants
- [ ] 6.2 Compare with exp 002/003/004 baselines (trained from scratch on BirdSet HSN)
- [ ] 6.3 Analyze per-species performance and failure modes
- [ ] 6.4 Document results in `results.md`
