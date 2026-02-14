# Tasks

## 1. Setup
- [x] 1.1 Download BirdCLEF 2025 data via Kaggle API to `~/scratch/birdclef-2026/data/birdclef-2025/`
- [x] 1.2 Explore data: count species, recordings, soundscape files, durations, file formats
- [x] 1.3 Document data layout in a README or data manifest

## 2. Pretraining
- [x] 2.1 Build soundscape data loader: 10s chunks, nnAudio mel, 128x1000 spectrograms
- [x] 2.2 Adapt SSAMBA MSPM pretraining for 128x1 patches (port from vendor code to HF model)
- [x] 2.3 Implement joint MPC + MPG loss with ~78% token masking
- [x] 2.4 Fast-dev-run to verify pretraining loop runs
- [ ] 2.5 Overfit test: verify pretraining can memorize 4 batches (MPC acc >> 0.001) — jobs 3951553
- [ ] 2.6 Full pretraining on soundscapes, log wall-clock per epoch
- [ ] 2.7 Document pretraining benchmarks (time, loss curves)

## 3. Fine-tuning
- [x] 3.1 Build labeled data loader: focal recordings → 10s chunks, species labels (206 classes)
- [x] 3.2 Add classifier head (mean pooling → Linear) to SSAMBA backbone
- [x] 3.3a Fast-dev-run to verify fine-tuning loop runs
- [ ] 3.3b Overfit test: verify finetune can memorize 4 batches (cmAP → 1.0) — job 3951554
- [ ] 3.4 Fine-tune on labeled train split, validate on held-out portion
- [ ] 3.5 Document fine-tuning metrics (cmAP, loss, time)

## 4. Inference + Submission
- [x] 4.1 Implement inference: 10s chunks → mean-pool 5s token windows → per-window predictions
- [ ] 4.2 Handle CPU inference (pure PyTorch fallback or ONNX export — no CUDA kernels)
- [x] 4.3 Format predictions as submission.csv in Kaggle format
- [ ] 4.4 Benchmark inference throughput (GPU and CPU)
- [ ] 4.5 Package for Kaggle using Adrian's offline submission template (wheels + weights)
- [ ] 4.6 Submit to Kaggle, record score

## 5. Analysis
- [ ] 5.1 Document wall-clock benchmarks: pretrain, fine-tune, inference
- [ ] 5.2 Write results.md with findings and next steps

## Completed Milestones
- **42/42 tests passing** (CPU + CUDA) — verified after mean-pooling refactoring
- **Architecture change**: SED attention pooling → clip-level mean pooling (BirdCLEF has no temporal annotations)
- **Data**: 206 species, 103,738 train chunks, 58,356 soundscape chunks
