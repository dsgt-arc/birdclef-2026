---
name: "model2vec"
source: "https://github.com/MinishLab/model2vec"
license: "MIT"
category: "self-supervised"
pipeline_stage:
  - "embedding"
relevance: "Distill transformers into fast static embeddings; core technique for Phase 3"
status: "reference"
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

## Bioacoustics Adaptation (BEATs)

In our implementation (`user/anthony/src/birdclef/static_embeddings.py`), we adapted Model2Vec for the BEATs transformer by:
1. **Vocabulary Generation**: Using the BEATs discrete tokenizer to define a 1,024-token "audio vocabulary."
2. **Codebook Construction**: Averaging patch embeddings (pre-transformer) from the ESC-50 dataset for each token ID.
3. **Distillation**: Passing each centroid through the BEATs transformer as a single-patch sequence to create context-independent embeddings.

### Key Findings & Limitations

Our Experiment 002 revealed a significant limitation when applying this to bioacoustics:

- **Performance**: Achieved **32x speedup** on CPU using FAISS for quantization.
- **Accuracy Collapse**: Accuracy dropped from **84% to 36%** on ESC-50.
- **Context Collapse**: Unlike natural language, where words carry significant independent meaning, bioacoustic "tokens" are highly context-dependent. Passing single patches through a transformer destroys the self-attention mechanism's ability to model temporal shifts (e.g., pitch modulation).

### Mitigation: Context Recovery

The standalone context-recovery concept note was removed during documentation cleanup; see experiment results for the durable findings.

## Dependencies
