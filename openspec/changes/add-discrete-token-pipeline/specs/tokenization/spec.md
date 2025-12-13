# Tokenization

## ADDED Requirements

### Requirement: Audio Preprocessing
The system SHALL convert raw audio to mel-spectrogram representation suitable for tokenization.

#### Scenario: Standard audio preprocessing
- **WHEN** raw audio at any sample rate is provided
- **THEN** the system resamples to 32kHz, extracts mel-spectrograms at 10Hz frame rate with 128 mel bins

#### Scenario: Chunked processing for long recordings
- **WHEN** audio longer than 10 seconds is provided
- **THEN** the system processes in overlapping 5-second chunks with 2.5-second hop

### Requirement: Random Projection Tokenizer
The system SHALL implement a BEST-RQ style random projection tokenizer with a frozen codebook.

#### Scenario: Tokenizer initialization
- **WHEN** a tokenizer is created with vocabulary size N and projection dimension D
- **THEN** a random projection matrix (input_dim → D) and random codebook (N × D) are initialized and frozen

#### Scenario: Audio frame tokenization
- **WHEN** a mel-spectrogram frame is provided
- **THEN** the frame is projected, L2-normalized, and matched to nearest codebook entry

#### Scenario: Batch tokenization
- **WHEN** multiple audio files are provided
- **THEN** tokenization is parallelized across CPU cores using faiss for efficient nearest neighbor

### Requirement: K-means Tokenizer (Fallback)
The system SHALL support k-means clustering tokenization as a fallback approach.

#### Scenario: Codebook training
- **WHEN** a corpus of mel-spectrogram frames is provided
- **THEN** k-means clustering produces a codebook of N centroids

#### Scenario: Tokenization with trained codebook
- **WHEN** new audio is processed
- **THEN** each frame is assigned to nearest centroid using faiss index

### Requirement: Tokenizer Evaluation
The system SHALL provide metrics to evaluate tokenizer quality.

#### Scenario: Codebook utilization measurement
- **WHEN** a tokenized corpus is analyzed
- **THEN** the system reports token frequency distribution and effective vocabulary size

#### Scenario: Reconstruction quality measurement
- **WHEN** tokens are mapped back to codebook vectors
- **THEN** the system reports MSE between original and reconstructed frames
