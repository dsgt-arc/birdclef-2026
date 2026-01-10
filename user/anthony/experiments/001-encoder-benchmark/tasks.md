# Tasks

## 1. Environment Setup
- [ ] 1.1 Create scratch directory structure (`~/scratch/birdclef-2026/`)
- [ ] 1.2 Create virtual environment in scratch
- [ ] 1.3 Create pyproject.toml with dependencies
- [ ] 1.4 Install project in editable mode

## 2. Data & Checkpoint Downloads
- [ ] 2.1 Download BEATs checkpoint (BEATs_iter3_plus_AS2M.pt) via sbatch
- [ ] 2.2 Download EAT checkpoint via sbatch
- [ ] 2.3 Verify ESC-50 data available (vendor/ESC-50 or HuggingFace)
- [ ] 2.4 (Stretch) Download AudioSet bird clips

## 3. Implementation
- [ ] 3.1 Create `src/birdclef/` package structure
- [ ] 3.2 Implement audio loading utilities (resampling to 16kHz)
- [ ] 3.3 Implement wav2vec2 feature extractor (via transformers)
- [ ] 3.4 Implement BEATs feature extractor wrapper
- [ ] 3.5 Implement EAT feature extractor wrapper
- [ ] 3.6 Implement linear probe training loop
- [ ] 3.7 Implement evaluation metrics (accuracy, mAP)
- [ ] 3.8 Create unified benchmark runner script

## 4. Part A: ESC-50 Benchmark
- [ ] 4.1 Prepare ESC-50 dataloader with 5-fold CV splits
- [ ] 4.2 Run wav2vec2 + linear probe on ESC-50
- [ ] 4.3 Run BEATs + linear probe on ESC-50
- [ ] 4.4 Run EAT + linear probe on ESC-50
- [ ] 4.5 Compare results to published (BEATs 98.1%, EAT 95.9%)

## 5. Part B: AudioSet Birds (Stretch Goal)
- [ ] 5.1 Acquire AudioSet bird clips (eval split)
- [ ] 5.2 Prepare AudioSet bird dataloader
- [ ] 5.3 Run encoder comparison on AudioSet birds
- [ ] 5.4 Compute mAP for each encoder

## 6. Analysis & Documentation
- [ ] 6.1 Compile results table with accuracy/mAP/inference time
- [ ] 6.2 Analyze which encoder performs best
- [ ] 6.3 Document findings in results.md
- [ ] 6.4 Update STATUS.md with conclusions
