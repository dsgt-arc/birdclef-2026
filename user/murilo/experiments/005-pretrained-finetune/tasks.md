# Experiment 005: Pretrained Audio Model Fine-Tuning - Tasks

## 1. Data Acquisition

- [ ] 1.1 Write download script (`scripts/download_birdclef2025.sh`)
- [ ] 1.2 Download BirdCLEF 2025 dataset to shared project storage (`~/ps-clef2026_birdclef-0/data/birdclef-2025/`)
- [ ] 1.3 Verify download integrity (file counts, total size, spot-check audio files)

## 2. Exploratory Data Analysis

- [ ] 2.1 Create EDA notebook (`notebooks/eda_birdclef2025.ipynb`)
- [ ] 2.2 Analyze `train_metadata.csv` — species distribution, recording durations, geographic coverage
- [ ] 2.3 Analyze audio properties — sample rates, duration histogram, silence detection
- [ ] 2.4 Identify class imbalance and long-tail species
- [ ] 2.5 Compare BirdCLEF 2025 species overlap with BirdSet HSN/XCM datasets
- [ ] 2.6 Summarize findings in EDA section of `proposal.md`

## 3. Setup

- [ ] 3.1 Create `pyproject.toml` with dependencies (torch, torchaudio, BEATs/EAT, nnAudio)
- [ ] 3.2 Create BirdCLEF 2025 PyTorch Dataset class in shared `src/birdclef/`
- [ ] 3.3 Create streaming DataModule for BirdCLEF 2025 (reuse `StreamingBirdSetDataModule` pattern or adapt)
- [ ] 3.4 Write `proposal.md` with hypothesis, method, and evaluation plan
- [ ] 3.5 Download BEATs/EAT pretrained checkpoints to shared storage

## 4. Model Implementation

- [ ] 4.1 Create feature extractor wrapper for chosen model (BEATs or EAT)
- [ ] 4.2 Create fine-tuning Lightning module (frozen encoder + trainable classification head)
- [ ] 4.3 Create config files (frozen vs full fine-tune, learning rate sweeps)
- [ ] 4.4 Validate with fast-dev-run

## 5. Training

- [ ] 5.1 Create SLURM scripts (`sbatch/_job.sh`, `sbatch/run.sh`)
- [ ] 5.2 Run frozen-encoder baseline (linear probe)
- [ ] 5.3 Run full fine-tuning with lower learning rate
- [ ] 5.4 Monitor training via W&B

## 6. Analysis

- [ ] 6.1 Compare cmAP between frozen and fine-tuned variants
- [ ] 6.2 Compare with exp 002/003/004 baselines (trained from scratch on BirdSet HSN)
- [ ] 6.3 Analyze per-species performance and failure modes
- [ ] 6.4 Document results in `results.md`
