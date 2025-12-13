---
name: "model2vec"
source: "https://github.com/MinishLab/model2vec"
license: "MIT"
category: "self-supervised"
pipeline_stage:
  - "embedding"
relevance: "Distill transformers into fast static embeddings; core technique for Phase 3"
status: "active"
---

# Model2Vec

## Overview

Model2Vec is a technique to distill any sentence transformer into a small, fast static embedding model. It reduces model size by up to 50x and makes inference up to 500x faster with minimal performance degradation. The core idea is to forward pass a vocabulary through a transformer to create static token embeddings, followed by post-processing steps including PCA and learned weighting.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `StaticModel` | Main inference class for static embeddings | `model2vec/` |
| `distill()` | Distillation function | `model2vec/distill/` |
| `StaticModelForClassification` | Classification fine-tuning | `model2vec/train/` |
| Pre-trained models | potion-base-2M to potion-base-32M | HuggingFace: `minishlab/` |

## Relevance to Pipeline

- **Embedding**: Core technique for Phase 3; distill transformer representations into static token embeddings
- **Inference**: Enables extremely fast inference (500x speedup) crucial for Kaggle CPU constraints
- **Representation Learning**: Static embeddings can be used directly without running transformer at inference

## Usage

```python
from model2vec import StaticModel

# Load pre-trained model
model = StaticModel.from_pretrained("minishlab/potion-base-32M")

# Generate embeddings
embeddings = model.encode(["audio segment 1", "audio segment 2"])

# Token-level embeddings
token_embeddings = model.encode_as_sequence(["audio segment"])

# Distill your own model
from model2vec.distill import distill
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5")
m2v_model.save_pretrained("custom_m2v")

# Fine-tune classifier
from model2vec.train import StaticModelForClassification
classifier = StaticModelForClassification.from_pretrained("minishlab/potion-base-32M")
classifier.fit(texts, labels)
```

## Dependencies

- Base: numpy only
- Distillation: `pip install model2vec[distill]`
- Training: `pip install model2vec[train]`

## Notes

- Distillation takes ~30 seconds on CPU, no dataset required
- Key insight: static token embeddings + mean pooling approximate transformer behavior
- Supports BPE and Unigram tokenizers
- Quantization to int8 reduces size by 50% without performance loss
- Integrated with Sentence Transformers and LangChain
- For audio: would adapt by creating token vocabulary from audio codebook
