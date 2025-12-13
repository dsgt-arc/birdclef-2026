# Self-Supervised Learning

## ADDED Requirements

### Requirement: Transformer Encoder
The system SHALL implement a Transformer encoder that processes discrete token sequences.

#### Scenario: Model architecture
- **WHEN** a model is initialized with embedding_dim, num_layers, num_heads
- **THEN** a Transformer encoder with learnable token embeddings and positional encodings is created

#### Scenario: Forward pass
- **WHEN** a sequence of token IDs is provided
- **THEN** the model outputs contextualized embeddings for each position

### Requirement: Masked Prediction Training
The system SHALL train the encoder using masked prediction on unlabeled audio tokens.

#### Scenario: Masking strategy
- **WHEN** a token sequence is prepared for training
- **THEN** 15% of tokens are masked in contiguous spans of 10 frames

#### Scenario: Prediction objective
- **WHEN** masked tokens are predicted
- **THEN** cross-entropy loss is computed between predictions and original token IDs

#### Scenario: Training loop
- **WHEN** training is executed on soundscape dataset
- **THEN** the model is trained with AdamW optimizer, cosine learning rate schedule, and gradient accumulation

### Requirement: Iterative Clustering Target (Optional)
The system SHALL optionally support HuBERT-style iterative clustering for improved targets.

#### Scenario: Cluster refinement
- **WHEN** a trained model's latent representations are available
- **THEN** k-means clustering produces refined pseudo-labels for the next training iteration

### Requirement: Linear Probing Evaluation
The system SHALL evaluate representation quality via linear probing on labeled data.

#### Scenario: Feature extraction
- **WHEN** labeled audio clips are processed
- **THEN** the encoder outputs are mean-pooled to produce fixed-size embeddings

#### Scenario: Linear classifier training
- **WHEN** embeddings and labels are provided
- **THEN** a logistic regression classifier is trained and evaluated with macro F1 score
