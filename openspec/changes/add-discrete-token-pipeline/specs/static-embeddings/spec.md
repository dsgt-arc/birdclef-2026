# Static Embeddings

## ADDED Requirements

### Requirement: Token Embedding Distillation
The system SHALL distill transformer representations into static token embeddings using model2vec methodology.

#### Scenario: Corpus processing
- **WHEN** a corpus of audio clips is processed through the trained transformer
- **THEN** token-level embeddings are extracted and aggregated per-token across the corpus

#### Scenario: Static embedding computation
- **WHEN** aggregated token embeddings are available
- **THEN** each token's static embedding is computed as the mean of its contextualized representations

### Requirement: Dimensionality Reduction
The system SHALL apply PCA to reduce static embedding dimensionality.

#### Scenario: PCA fitting
- **WHEN** static embeddings for all tokens are available
- **THEN** PCA is fitted to retain 95% of variance (or target dimensionality)

#### Scenario: Embedding compression
- **WHEN** PCA is applied
- **THEN** token embeddings are projected to lower dimension (e.g., 256 from 768)

### Requirement: Clip-Level Embedding
The system SHALL produce fixed-size embeddings for variable-length audio clips.

#### Scenario: Token sequence to clip embedding
- **WHEN** an audio clip is tokenized
- **THEN** static embeddings are looked up and mean-pooled to produce clip embedding

#### Scenario: Weighted pooling (optional)
- **WHEN** token importance weights are learned
- **THEN** weighted mean pooling is applied instead of uniform mean

### Requirement: Embedding Export
The system SHALL export static embeddings in efficient formats for inference.

#### Scenario: Numpy export
- **WHEN** embeddings are finalized
- **THEN** the token-to-embedding lookup table is saved as numpy array

#### Scenario: ONNX export (optional)
- **WHEN** the full pipeline (tokenize → embed → pool) is traced
- **THEN** an ONNX model is exported for optimized CPU inference
