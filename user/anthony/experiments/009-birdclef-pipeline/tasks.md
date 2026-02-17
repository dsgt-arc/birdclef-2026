# Tasks

## 1. Setup
- [x] 1.1 Download BirdCLEF 2025 data via Kaggle API to `~/scratch/birdclef-2026/data/birdclef-2025/`
- [x] 1.2 Explore data: count species, recordings, soundscape files, durations, file formats
- [x] 1.3 Document data layout in a README or data manifest

## 2. Pretraining
- [x] 2.1 Build soundscape data loader: 10s chunks, nnAudio mel, 128x1000 spectrograms
- [x] 2.2 Adapt SSAMBA MSPM pretraining for 128x1 patches (port from vendor code to HF model)
- [x] 2.3 Implement MPG loss with ~78% span masking (MPC removed — degenerate with 128x1)
- [x] 2.4 Fast-dev-run to verify pretraining loop runs
- [x] 2.5 Overfit test: verify pretraining can memorize batches (loss → 0.02)
- [x] 2.6 Full pretraining: pretrain-full (64 epochs, from scratch), pretrain-warm (62 epochs, AudioSet init)
- [x] 2.7 Document pretraining benchmarks — see results.md

## 3. Fine-tuning
- [x] 3.1 Build labeled data loader: focal recordings → 10s chunks, species labels (206 classes)
- [x] 3.2 Add classifier head (mean pooling → Linear) to SSAMBA backbone
- [x] 3.3a Fast-dev-run to verify fine-tuning loop runs
- [x] 3.3b Overfit test: 200 epochs from scratch, confirmed model learns (train loss → 0.003)
- [x] 3.4 Fine-tune with AudioSet backbone: val/cmap 0.425 @ epoch 11 (Job 3960497, timed out)
- [ ] 3.5 Fine-tune with SSL backbone: Job 4011637 (running)
- [x] 3.6 Document fine-tuning metrics — see results.md

## 4. Inference + Submission
- [x] 4.1 Implement inference: 10s chunks → mean-pool 5s token windows → per-window predictions
- [x] 4.2 ONNX export with mamba-onnx custom ops (CPU inference, no CUDA required)
- [x] 4.3 Format predictions as submission.csv in Kaggle format
- [x] 4.4 Benchmark inference throughput: 13s total on Kaggle (1.5s ONNX session init)
- [x] 4.5 Package for Kaggle: deps dataset (157.8 MB), weights dataset (24.8 MB)
- [x] 4.6 Submit to Kaggle: 0.484 public / 0.476 private (kernel v6)

## 5. Analysis
- [x] 5.1 Document wall-clock benchmarks — see results.md
- [x] 5.2 Write results.md with findings and next steps
- [ ] 5.3 Re-submit with SSL-finetuned model (pending Job 4011637)

## Completed Milestones
- **42/42 tests passing** (CPU + CUDA)
- **Architecture change**: SED attention pooling → clip-level mean pooling (BirdCLEF has no temporal annotations)
- **Data**: 206 species, 103,738 train chunks, 58,356 soundscape chunks
- **Kaggle pipeline**: end-to-end verified (kernel v6, 13s runtime)
- **Pretraining**: both conditions converged (~0.408 L1 loss, ~0.320 MSE)
- **Best finetune**: AudioSet init → 0.425 val/cmap (12 epochs, still climbing)
- **Kaggle score**: 0.484 public / 0.476 private
