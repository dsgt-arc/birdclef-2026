# Inference

## ADDED Requirements

### Requirement: Linear Classifier
The system SHALL use a simple linear classifier for species prediction.

#### Scenario: Classifier training
- **WHEN** static embeddings and species labels are provided
- **THEN** a logistic regression or linear SVM classifier is trained

#### Scenario: Multi-label prediction
- **WHEN** an audio clip embedding is classified
- **THEN** probabilities are returned for all species (sigmoid for multi-label, softmax for single-label)

### Requirement: CPU-Optimized Inference
The system SHALL be optimized for CPU inference within Kaggle time constraints.

#### Scenario: Batch processing
- **WHEN** multiple audio files are queued for inference
- **THEN** processing is batched to maximize throughput

#### Scenario: Memory efficiency
- **WHEN** large test sets are processed
- **THEN** memory usage stays within Kaggle limits (16GB RAM typical)

#### Scenario: Inference timing
- **WHEN** the full test set is processed
- **THEN** total wall-clock time is under the Kaggle CPU budget (typically 9 hours)

### Requirement: Kaggle Submission Format
The system SHALL produce output in the required Kaggle submission format.

#### Scenario: Submission file generation
- **WHEN** predictions are complete
- **THEN** a CSV file with row_id and species probability columns is generated

### Requirement: Ensemble Support
The system SHALL optionally support ensembling multiple models within the CPU budget.

#### Scenario: Model averaging
- **WHEN** multiple classifiers are available
- **THEN** predictions are averaged (or weighted) to produce final output

#### Scenario: Budget allocation
- **WHEN** ensemble is configured
- **THEN** inference time is allocated proportionally across models
