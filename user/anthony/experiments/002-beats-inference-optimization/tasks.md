# Tasks

## 1. Setup
- [x] 1.1 Verify BEATs Encoder checkpoint is accessible (BEATs_iter3.pt)
- [x] 1.2 Verify BEATs Tokenizer checkpoint is accessible (Tokenizers_iter3.pt)
- [x] 1.3 Set up ESC-50 data loader from experiment 001
- [x] 1.4 Install onnxruntime dependency
- [x] 1.5 Record CPU specs (model, cores) for benchmarks

## 2. Instrumentation (Part A)
- [x] 2.1 Create profiling wrapper for BEATs with stage-level timing
- [x] 2.2 Implement timing for: fbank, patch_embed, layer_norm, proj, transformer, pool
- [x] 2.3 Add memory tracking (peak GPU/RAM)
- [x] 2.4 Implement warmup + N iterations benchmark protocol
- [x] 2.5 Validate profiling overhead is <5% (GPU: 2.15%, CPU: 0.33%)

## 3. PyTorch Benchmarks (Part B)
- [x] 3.1 GPU benchmarks: batch sizes 1, 4, 16, 32, 64
- [x] 3.2 CPU benchmarks: batch sizes 1, 4, 16, 32
- [x] 3.3 Record split times per stage for each config
- [x] 3.4 Record peak memory for each config

## 4. ONNX Export (Part C)
- [x] 4.1 Export encoder frontend (fbank + patch_embed + layer_norm + proj) to ONNX
- [x] 4.2 Export full BEATs encoder to ONNX
- [x] 4.3 Handle dynamic axes (batch size, sequence length)
- [x] 4.4 Verify ONNX output matches PyTorch output (frontend: max_diff=5e-6, encoder: max_diff=3e-6)
- [x] 4.5 CPU benchmarks with ONNX Runtime: batch sizes 1, 4, 16, 32
- [x] 4.6 Record latency and throughput

## 5. Static Embeddings - Distillation (Part D)
- [x] 5.1 Load BEATs Tokenizer (Tokenizers_iter3.pt)
- [x] 5.2 For each audio in ESC-50 corpus:
  - [x] 5.2a Extract 768-dim patch embeddings (Encoder frontend, before transformer)
  - [x] 5.2b Extract discrete token IDs (Tokenizer full pipeline)
  - [x] 5.2c Store (patch_embedding, token_id) pairs
- [x] 5.3 Analyze token distribution (932/1024 tokens used, highly skewed)
- [x] 5.4 Build codebook: average patch embeddings per token ID → (1024 × 768)
- [x] 5.5 Compute static embeddings: pass each centroid through Encoder transformer (single-patch)
- [x] 5.6 Apply model2vec enhancements:
  - [x] PCA dimensionality reduction (768 → 256, 98.5% variance)
  - [x] SIF weighting (downweight common tokens)
- [x] 5.7 Save artifacts:
  - [x] codebook.npy (1024 × 768)
  - [x] static_embeddings.npy (1024 × 768)
  - [x] static_embeddings_pca.npy (1024 × 256)
  - [x] sif_weights.npy (1024,)
  - [x] token_stats.json

## 6. Static Embeddings - Inference (Part E)
- [x] 6.1 Implement quantization: patch → nearest codebook centroid (L2 distance)
- [x] 6.2 Implement fast inference pipeline: ONNX frontend → quantize → lookup → pool
- [x] 6.3 Benchmark static pipeline: batch sizes 1, 4, 16, 32
  - **FIXED**: Replaced naive L2 with scipy cdist (1.8x speedup)
  - **FIXED**: Replaced cdist with FAISS IndexFlatL2 (32x speedup)
  - FAISS: 4.6 ms (217 samples/sec) vs Full BEATs: 150 ms (6.7 samples/sec)
- [x] 6.4 Run ESC-50 5-fold CV with static embeddings
- [x] 6.5 Compare accuracy vs full BEATs
  - **RESULT**: 35.9% accuracy vs 84.0% (48% drop - REJECTED)

## 7. Contextual Pooling (Part G) - Recovery from Context Collapse

### 7.1 Implementation
- [x] 7.1.1 Implement `AttentionPooler`: Linear(768→1) → Softmax → Weighted Sum
- [x] 7.1.2 Implement `CNN1DPooler`: Conv1d layers with global pooling
- [x] 7.1.3 Implement `LightCNN1DPooler`: 768→256→768 bottleneck variant
- [x] 7.1.4 Implement `CNNAttentionPooler`: CNN + Attention hybrid
- [x] 7.1.5 Add pooler support to `StaticEmbeddingsPipelineFAISS`

### 7.2 Training
- [x] 7.2.1 Create training script for contextual poolers on ESC-50
- [x] 7.2.2 Train AttentionPooler with 5-fold CV (38.9% accuracy)
- [x] 7.2.3 Train CNN1DPooler with 5-fold CV (~62.6% accuracy, 2/5 folds)
- [x] 7.2.4 Train LightCNN1DPooler with 5-fold CV (58.1% accuracy)
- [x] 7.2.5 Train CNNAttentionPooler (hybrid) with 5-fold CV (43.4% accuracy)

### 7.3 Evaluation
- [x] 7.3.1 Compare accuracy: mean pool vs attention vs CNN
- [x] 7.3.2 Document findings in results.md

### Results
| Pooler | Accuracy | vs Baseline |
|--------|----------|-------------|
| MeanPooler | 35.9% | — |
| AttentionPooler | 38.9% | +3.0% |
| CNNAttentionPooler | 43.4% | +7.5% |
| LightCNN1DPooler | 58.1% | +22.2% |
| **CNN1DPooler** | **~62.6%** | **+26.7%** |

### Outcome
- **Minimum (70%)**: NOT MET - best is 62.6%
- **Target (80%)**: NOT MET
- Context collapse cannot be fully recovered by pooling alone
- → Proceed to Experiment 003 (distillation)

## 8. Analysis (Part H)
- [ ] 8.1 Plot throughput vs batch size curves (all configs)
- [ ] 8.2 Create split time breakdown charts per stage
- [ ] 8.3 Plot token distribution histogram
- [ ] 8.4 Plot accuracy vs speedup Pareto frontier
- [ ] 8.5 Identify optimal operating points
- [ ] 8.6 Calculate final speedup ratios

## 9. Deliverables
- [ ] 9.1 Save all benchmark results as JSON in artifacts/benchmarks/
- [ ] 9.2 Save ONNX models in artifacts/models/
- [ ] 9.3 Save codebook + embeddings in artifacts/models/static_embeddings/
- [ ] 9.4 Save trained pooler checkpoints in artifacts/models/poolers/
- [ ] 9.5 Save distillation data in artifacts/distillation/
- [ ] 9.6 Generate visualization plots in artifacts/figures/
- [ ] 9.7 Document findings in results.md
- [ ] 9.8 Write recommendations for optimal settings
