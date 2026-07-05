---
name: "BirdSet"
source: "https://github.com/DBD-research-group/BirdSet"
license: "BSD-3-Clause"
category: "dataset"
pipeline_stage:
  - "data-processing"
  - "representation-learning"
relevance: "Large-scale benchmark dataset with 6800+ hours of bird audio for training and evaluation"
status: "reference"
---

# BirdSet

## Overview

BirdSet is a large-scale benchmark dataset for audio classification in avian bioacoustics, published at NeurIPS 2024. It provides over 6,800 hours of recordings from nearly 10,000 species for training, plus 400+ hours across eight evaluation datasets. The dataset is designed for multi-label classification, covariate shift analysis, and self-supervised learning research.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `BirdSetDataModule` | PyTorch Lightning data module for training | `birdset/datamodule/birdset_datamodule.py` |
| `DatasetConfig` | Dataset configuration and preprocessing | `birdset/configs/datamodule_configs.py` |
| `BirdSetTransformsWrapper` | Audio-to-spectrogram transforms | `birdset/datamodule/components/transforms.py` |
| `MultilabelModule` | Training module for multi-label classification | `birdset/modules/multilabel_module.py` |
| Experiment configs | Hydra configs for reproducible experiments | `birdset/configs/experiment/` |

## Relevance to Pipeline

- **Data Processing**: Primary data source for training and evaluation; provides standardized splits and preprocessing
- **Representation Learning**: Large-scale XCL/XCM datasets ideal for self-supervised pre-training
- **Inference**: Evaluation datasets (HSN, SSW, PER, etc.) for measuring classification performance

**When to use BirdSet vs AudioSet:**
- Use [AudioSet](audioset.md) to compare pretrained encoders fairly (zero domain shift)
- Use BirdSet to evaluate fine-grained species classification and soundscape transfer

## Usage

```python
from birdset.configs.datamodule_configs import DatasetConfig, LoadersConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

# Initialize data module for a specific dataset
dm = BirdSetDataModule(
    dataset=DatasetConfig(
        data_dir='data_birdset/HSN',
        hf_path='DBD-research-group/BirdSet',
        hf_name='HSN',
        sampling_rate=32000,
    ),
    loaders=LoadersConfig(),
)

dm.prepare_data()
dm.setup(stage="fit")
train_loader = dm.train_dataloader()
```

## Dependencies

- Python 3.10
- PyTorch, PyTorch Lightning
- Hydra for configuration
- HuggingFace Datasets
- torchaudio, librosa

## Evaluation Datasets

Eight soundscape test datasets for benchmarking:

| Dataset | Region | Species | Test Segments | Notes |
|---------|--------|---------|---------------|-------|
| **HSN** | High Sierra Nevada | 21 | 12,000 | Easy (few species) |
| **SSW** | Sapsucker Woods | 81 | 205,200 | Medium, largest |
| **PER** | Amazon Basin | 132 | 15,120 | Hard (most species) |
| **NES** | Costa Rica | 89 | 24,480 | Medium |
| **UHH** | Hawaiian Islands | 25 | 36,637 | Easy |
| **NBP** | NIPS4BPlus | 51 | 563 | Small test set |
| **SNE** | Sierra Nevada | 56 | 23,756 | Medium |
| **POW** | Powdermill Nature | 48 | 4,560 | Medium |

**Recommendation**: Start with HSN (easiest, fast iteration) and PER (hardest, stress test).

## Notes

- Datasets hosted on HuggingFace at `DBD-research-group/BirdSet`
- XCL (full Xeno-Canto) is ~480GB; XCM (medium subset) is ~90GB
- Pre-trained model checkpoints available for ConvNeXT, EfficientNet, EAT, and others
- Supports multi-label classification with one-hot encoded labels
- Primary metric: **cmAP** (class-mean average precision)
