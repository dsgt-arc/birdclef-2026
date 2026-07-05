---
name: "ESC-50"
source: "https://github.com/karolpiczak/ESC-50"
license: "CC BY-NC 3.0"
category: "dataset"
pipeline_stage:
  - "evaluation"
relevance: "Standard audio classification benchmark; validates encoder quality on general sounds"
status: "legacy-active"
---

# ESC-50 (Environmental Sound Classification)

## Overview

ESC-50 is the standard benchmark for environmental sound classification. It's small, well-established, and all major audio models report results on it. Use this to validate that your encoder works correctly before domain-specific evaluation.

## Dataset Details

| Property | Value |
|----------|-------|
| Classes | 50 |
| Clips | 2,000 (40 per class) |
| Duration | 5 seconds per clip |
| Total | ~2.78 hours |
| Format | WAV, 44.1kHz mono |
| Evaluation | 5-fold cross-validation |

## Categories

| Category | Example Classes |
|----------|-----------------|
| Animals | Dog, cat, bird, frog |
| Natural | Rain, wind, water |
| Human (non-speech) | Cough, sneeze, breathing |
| Domestic | Clock, door knock, keyboard |
| Urban | Siren, car horn, engine |

## Published Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| BEATs | 98.1% | SOTA |
| EAT | 95.9% | 15x faster training |
| AST | 95.6% | |
| Human | 81.3% | Baseline |

## Usage

```python
# Available on HuggingFace
from datasets import load_dataset

dataset = load_dataset("ashraq/esc50")

# Or download directly
# https://github.com/karolpiczak/ESC-50/archive/master.zip
```

## 5-Fold Cross-Validation

Files are pre-assigned to folds (1-5) in filename: `{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`

Clips from the same source stay in the same fold to prevent leakage.

## Relevance to Pipeline

- **Sanity check**: Verify encoder produces reasonable features
- **Comparison**: Match published numbers before trusting domain-specific results
- **Fast iteration**: Small dataset, runs in minutes

## References

- [ESC-50 GitHub](https://github.com/karolpiczak/ESC-50)
- [ESC-50 on HuggingFace](https://huggingface.co/datasets/ashraq/esc50)
- [Original Paper](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf)
