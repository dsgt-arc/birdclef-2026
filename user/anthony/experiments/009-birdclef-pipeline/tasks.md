# Tasks

## 1. Setup
- [ ] 1.1 Download BirdCLEF 2025 data via Kaggle API to `~/scratch/birdclef-2026/data/birdclef-2025/`
- [ ] 1.2 Explore data: count species, recordings, soundscape files, durations, file formats
- [ ] 1.3 Document data layout in a README or data manifest

## 2. Pretraining
- [ ] 2.1 Build soundscape data loader: 10s chunks, nnAudio mel, 128x1000 spectrograms
- [ ] 2.2 Adapt SSAMBA MSPM pretraining for 128x1 patches (port from vendor code to HF model)
- [ ] 2.3 Implement joint MPC + MPG loss with ~78% token masking
- [ ] 2.4 Fast-dev-run to verify pretraining loop runs
- [ ] 2.5 Full pretraining on soundscapes, log wall-clock per epoch
- [ ] 2.6 Document pretraining benchmarks (time, loss curves)

## 3. Fine-tuning
- [ ] 3.1 Build labeled data loader: focal recordings → 10s chunks, species labels (hundreds of classes)
- [ ] 3.2 Add classifier head to pretrained SSAMBA (Linear → num_species)
- [ ] 3.3 Fine-tune on labeled train split, validate on held-out portion
- [ ] 3.4 Document fine-tuning metrics (cmAP, loss, time)

## 4. Inference + Submission
- [ ] 4.1 Implement inference: 10s chunks → per-token SED predictions → aggregate within each 5s half for Kaggle submission slots
- [ ] 4.2 Handle CPU inference (pure PyTorch fallback or ONNX export — no CUDA kernels)
- [ ] 4.3 Format predictions as submission.csv in Kaggle format
- [ ] 4.4 Benchmark inference throughput (GPU and CPU)
- [ ] 4.5 Package for Kaggle using Adrian's offline submission template (wheels + weights)
- [ ] 4.6 Submit to Kaggle, record score

## 5. Analysis
- [ ] 5.1 Document wall-clock benchmarks: pretrain, fine-tune, inference
- [ ] 5.2 Write results.md with findings and next steps
