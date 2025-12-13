# Tasks

## 1. Project Setup
- [ ] 1.1 Define project.md with tech stack, conventions, and constraints
- [ ] 1.2 Set up Python package structure with pyproject.toml
- [ ] 1.3 Configure development environment (uv, pre-commit, pytest)
- [ ] 1.4 Document Kaggle submission constraints (CPU time, memory limits)

## 2. Data Pipeline
- [ ] 2.1 Review BirdCLEF 2026 dataset structure when available
- [ ] 2.2 Implement audio preprocessing (resampling, chunking, mel-spectrogram extraction)
- [ ] 2.3 Create data loaders compatible with both training and Kaggle inference
- [ ] 2.4 Benchmark I/O performance and identify bottlenecks

## 3. Tokenization Research
- [ ] 3.1 Baseline: Port k-means MFCC tokenizer from birdclef-2025
- [ ] 3.2 Experiment: k-means on mel-spectrograms (different frequency resolution)
- [ ] 3.3 Experiment: Random projection tokenizer (BEST-RQ style)
- [ ] 3.4 Experiment: Learned codebook (VQ-VAE or neural codec approach)
- [ ] 3.5 Evaluate: Codebook utilization, reconstruction quality, downstream task performance
- [ ] 3.6 Document findings and select best tokenization approach

## 4. Self-Supervised Transformer
- [ ] 4.1 Implement base transformer encoder for discrete token sequences
- [ ] 4.2 Implement masked prediction objective (cross-entropy on token IDs)
- [ ] 4.3 Experiment: Reconstruction target (predict original tokens)
- [ ] 4.4 Experiment: Iterative clustering target (HuBERT-style k-means refinement)
- [ ] 4.5 Experiment: Online clustering target (DinoSR-style teacher-student)
- [ ] 4.6 Train on unlabeled soundscape data (self-supervised)
- [ ] 4.7 Evaluate via linear probing on labeled train set
- [ ] 4.8 Document findings and select best training strategy

## 5. Static Embeddings (model2vec)
- [ ] 5.1 Research model2vec methodology (sentence-transformers distillation)
- [ ] 5.2 Implement mean pooling over transformer token embeddings
- [ ] 5.3 Apply PCA for dimensionality reduction
- [ ] 5.4 Learn importance weights for tokens (optional)
- [ ] 5.5 Evaluate static embeddings vs full transformer on classification
- [ ] 5.6 Measure inference speedup

## 6. Inference Pipeline
- [ ] 6.1 Implement linear classifier on static embeddings
- [ ] 6.2 Benchmark CPU inference time per audio clip
- [ ] 6.3 Optimize tokenization for CPU (quantized faiss, batch processing)
- [ ] 6.4 Profile memory usage and optimize if needed
- [ ] 6.5 Test ensemble strategies within CPU budget
- [ ] 6.6 Package for Kaggle submission

## 7. Evaluation & Documentation
- [ ] 7.1 Establish evaluation protocol (train/val split, metrics)
- [ ] 7.2 Compare against birdclef-2025 baseline
- [ ] 7.3 Compare against public pretrained models (BirdNET, Perch, etc.)
- [ ] 7.4 Write research notes documenting experiments and findings
- [ ] 7.5 Prepare final submission strategy
