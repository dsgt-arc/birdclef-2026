---
name: "dinosr"
source: "https://github.com/Alexander-H-Liu/dinosr"
license: "Unspecified (fairseq-based)"
category: "self-supervised"
pipeline_stage:
  - "representation-learning"
relevance: "Self-distillation with online clustering for speech representation learning"
status: "reference"
---

# DinoSR

## Overview

DinoSR (DINO for Speech Representations) applies self-distillation and online clustering to self-supervised speech representation learning. Unlike HuBERT which requires offline k-means clustering, DinoSR performs clustering online during training, eliminating the need for a separate clustering step. This simplifies the training pipeline while achieving competitive performance.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `DinoSR` model | Main model with online clustering | `models/dinosr.py` |
| Config | Hydra configuration | `config/base.yaml` |
| Training | fairseq-based training loop | via `fairseq_cli/hydra_train.py` |

## Relevance to Pipeline

- **Representation Learning**: Phase 2 reference for self-supervised transformer training with online clustering targets
- **Tokenization**: Online clustering creates discrete targets without separate k-means step
- **Embedding**: Produces contextualized representations for downstream tasks

## Usage

```python
# Training
python fairseq_cli/hydra_train.py -m \
    --config-dir examples/dinosr/config/ \
    --config-name base \
    task.data=/path/to/librispeech/ \
    common.user_dir=examples/dinosr

# Loading pre-trained model
import fairseq
import argparse

code_path = "examples/dinosr"
fairseq.utils.import_user_module(argparse.Namespace(user_dir=code_path))
ckpt_path = "/path/to/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]

# Fine-tuning for ASR
python fairseq_cli/hydra_train.py -m \
    --config-dir examples/wav2vec/config/finetuning \
    --config-name base_100h \
    common.user_dir=examples/dinosr \
    model.w2v_path=/path/to/dinosr.ckpt
```

## Dependencies

- fairseq (commit 47e27984)
- PyTorch
- Hydra

## Notes

- Based on DINO (self-distillation with no labels) applied to speech
- Uses teacher-student framework with exponential moving average teacher
- Online clustering eliminates HuBERT's separate k-means preprocessing
- Pre-trained checkpoint available at CSAIL
- Data preparation follows wav2vec 2.0 instructions
- Paper: "DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning"
