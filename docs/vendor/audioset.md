---
name: "AudioSet"
source: "https://research.google.com/audioset/"
license: "CC BY 4.0"
category: "dataset"
pipeline_stage:
  - "data-processing"
  - "representation-learning"
relevance: "Pretraining dataset for BEATs/EAT; zero domain shift for evaluating pretrained encoders"
status: "active"
---

# AudioSet

## Overview

AudioSet is a large-scale dataset of 10-second YouTube clips annotated with 632 audio event classes. It's the pretraining dataset for BEATs, EAT, and most audio foundation models, making it ideal for evaluating pretrained encoders without domain shift.

**Why AudioSet for benchmarking?**
- BEATs and EAT were pretrained on AudioSet
- Zero domain shift = isolates encoder quality from transfer learning
- Bird classes have ~27k annotations

## Bird Classes

| ID | Name | Annotations |
|----|------|-------------|
| `/m/015p6` | Bird | 26,894 |
| `/m/020bb7` | Bird vocalization, bird call, bird song | 6,372 |
| `/m/07pggtn` | Chirp, tweet | - |
| `/m/07sx8x_` | Squawk | - |
| `/m/0h0rv` | Pigeon, dove | 8,523 |
| `/m/07r_25d` | Coo | - |
| `/m/04s8yn` | Crow | 850 |
| `/m/07r5c2p` | Caw | - |
| `/m/09d5_` | Owl | 634 |
| `/m/07r_80w` | Hoot | - |
| `/m/01dwxx` | Gull, seagull | - |
| `/m/05_wcq` | Bird flight, flapping wings | 347 |

## Dataset Splits

| Split | Clips | Hours |
|-------|-------|-------|
| Balanced train | 22,176 | ~62 |
| Unbalanced train | 2,042,985 | ~5,800 |
| Eval | 20,383 | ~57 |

## Usage

```python
# Download via tensorflow_datasets
import tensorflow_datasets as tfds

# Or use the CSV manifests directly
# https://research.google.com/audioset/download.html
```

## Limitations for Bird Classification

AudioSet has **coarse-grained** bird labels (12 classes) vs BirdSet's **fine-grained** species labels (~10k species). Use AudioSet to:
- Validate pretrained encoder quality (zero domain shift)
- Compare encoder architectures fairly

Use BirdSet/BirdCLEF to:
- Evaluate fine-grained species classification
- Test domain transfer (focal → soundscape)

## Relevance to Pipeline

| Use Case | AudioSet | BirdSet |
|----------|----------|---------|
| Encoder comparison (fair) | Yes | No (domain shift) |
| Species classification | No (coarse) | Yes |
| Soundscape evaluation | No | Yes |

## References

- [AudioSet Homepage](https://research.google.com/audioset/)
- [Ontology JSON](https://github.com/audioset/ontology)
- [Download Page](https://research.google.com/audioset/download.html)
