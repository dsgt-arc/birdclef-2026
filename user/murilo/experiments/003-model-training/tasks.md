# Experiment 003: Model Training - Tasks

## Setup

- [x] Create `AudioClassifier` class (`models/audio_classifier.py`)
- [x] Update `models/__init__.py` with new export
- [x] Create experiment directory structure
- [x] Create 4 config files (convnext scratch/pretrained, vit scratch/pretrained)
- [x] Create `train.py` adapted from exp002
- [x] Create SLURM scripts (`_job.sh`, `run.sh`)

## GPU Utilization Optimization

- [x] Vectorize mel-spectrogram transform, increase batch size to 128
- [x] Move mel-spectrogram from main-process collate into worker `__getitem__`
- [x] Reduce `num_workers` from 8 to 6 (PACE CPU:GPU ratio limit)
- [x] Add opt-in `cache_mel_specs` to pre-compute all mels in `setup()`
- [x] Verify GPU utilization is stable with caching (see `images/`)

## Training Runs

- [~] ConvNeXT-Tiny scratch (`config/convnext_tiny_scratch.yaml`) — running
- [~] ViT-Small scratch (`config/vit_small_scratch.yaml`) — running
- [~] ConvNeXT-Tiny pretrained (`config/convnext_tiny_pretrained.yaml`) — running
- [~] ViT-Small pretrained (`config/vit_small_pretrained.yaml`) — running

## Analysis

- [x] Compare cmAP across all 4 runs
- [x] Analyze training curves and convergence
- [x] Document results in experiment result file
