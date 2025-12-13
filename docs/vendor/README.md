# Vendored Repository Documentation

This directory contains documentation for the 14 external repositories vendored in this project. These repositories cover audio codecs, self-supervised learning, bioacoustics models, and utilities relevant to the BirdCLEF audio classification task.

## Repository Summary

| Repository | Category | Pipeline Stage | License | Status |
|------------|----------|----------------|---------|--------|
| [BirdNET-Analyzer](birdnet-analyzer.md) | pretrained-model | inference, embedding | MIT / CC BY-NC-SA 4.0 | reference |
| [BirdSet](birdset.md) | dataset | data-processing, representation-learning | BSD-3-Clause | active |
| [ProtoCLR](protoclr.md) | self-supervised | representation-learning | Unspecified | reference |
| [Wav2vec2.0](wav2vec2.md) | self-supervised | tokenization, representation-learning | Apache-2.0 | reference |
| [WavTokenizer](wavtokenizer.md) | tokenization | tokenization | MIT | active |
| [best-rq-pytorch](best-rq-pytorch.md) | self-supervised | tokenization, representation-learning | MIT | reference |
| [bioacoustics-model-zoo](bioacoustics-model-zoo.md) | utility | inference, embedding | MIT | active |
| [birdclef-2025](birdclef-2025.md) | utility | tokenization, embedding, inference | Unspecified | active |
| [descript-audio-codec](descript-audio-codec.md) | tokenization | tokenization | MIT | reference |
| [dinosr](dinosr.md) | self-supervised | representation-learning | Unspecified | reference |
| [hubert](hubert.md) | self-supervised | tokenization, representation-learning | MIT | reference |
| [model2vec](model2vec.md) | self-supervised | embedding | MIT | active |
| [opensoundscape](opensoundscape.md) | utility | data-processing, inference | MIT | active |
| [perch](perch.md) | pretrained-model | embedding, inference | Apache-2.0 | reference |

## Pipeline Mapping

### Phase 1: Tokenization

Repositories relevant to converting audio into discrete tokens:

- **[WavTokenizer](wavtokenizer.md)**: SOTA neural codec with 40-75 tokens/second (primary candidate)
- **[descript-audio-codec](descript-audio-codec.md)**: High-fidelity RVQGAN codec at 8kbps
- **[birdclef-2025](birdclef-2025.md)**: Baseline k-means on MFCCs approach
- **[best-rq-pytorch](best-rq-pytorch.md)**: Random projection quantization (no learned codebook)
- **[hubert](hubert.md)**: Iterative k-means clustering for discrete units

### Phase 2: Self-Supervised Transformer

Repositories relevant to training transformers on discrete tokens:

- **[hubert](hubert.md)**: Masked prediction with iterative clustering targets
- **[dinosr](dinosr.md)**: Online clustering eliminates separate k-means step
- **[Wav2vec2.0](wav2vec2.md)**: Contrastive learning with Gumbel-softmax quantization
- **[ProtoCLR](protoclr.md)**: Contrastive learning for domain-invariant representations

### Phase 3: Static Embeddings

Repositories relevant to distilling transformers into efficient static embeddings:

- **[model2vec](model2vec.md)**: Core technique for transformer-to-static distillation (primary reference)
- **[birdclef-2025](birdclef-2025.md)**: Baseline Word2Vec approach on tokens

### Phase 4: Inference

Repositories providing production models and utilities:

- **[BirdNET-Analyzer](birdnet-analyzer.md)**: 6500+ species classifier (benchmark target)
- **[perch](perch.md)**: 10k+ species classifier with high-quality embeddings
- **[bioacoustics-model-zoo](bioacoustics-model-zoo.md)**: Unified API for multiple models
- **[opensoundscape](opensoundscape.md)**: Audio processing and CNN training

### Data & Evaluation

- **[BirdSet](birdset.md)**: Large-scale training data (XCL/XCM) and evaluation datasets

## Category Definitions

- **tokenization**: Converts continuous audio into discrete token sequences
- **self-supervised**: Learns representations without labeled data
- **pretrained-model**: Production-ready classification/embedding model
- **dataset**: Provides training or evaluation data
- **utility**: General-purpose tools and libraries

## Status Definitions

- **active**: Actively used or planned for use in our pipeline
- **reference**: Provides concepts or techniques we're adapting
- **experimental**: Under evaluation, not yet committed to
