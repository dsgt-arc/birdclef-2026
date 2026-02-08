# Experiment 003: Model Training - Tasks

## Setup

- [x] Create `AudioClassifier` class (`models/audio_classifier.py`)
- [x] Update `models/__init__.py` with new export
- [x] Create experiment directory structure
- [x] Create 4 config files (convnext scratch/pretrained, vit scratch/pretrained)
- [x] Create `train.py` adapted from exp002
- [x] Create SLURM scripts (`_job.sh`, `run.sh`)

## Training Runs

- [ ] ConvNeXT-Tiny scratch (`config/convnext_tiny_scratch.yaml`)
- [ ] ViT-Small scratch (`config/vit_small_scratch.yaml`)
- [ ] ConvNeXT-Tiny pretrained (`config/convnext_tiny_pretrained.yaml`)
- [ ] ViT-Small pretrained (`config/vit_small_pretrained.yaml`)

## Analysis

- [ ] Compare cmAP across all 4 runs
- [ ] Analyze training curves and convergence
- [ ] Document results in experiment result file
