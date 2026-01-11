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
- [ ] 5.1 Load BEATs Tokenizer (Tokenizers_iter3.pt)
- [ ] 5.2 For each audio in ESC-50 corpus:
  - [ ] 5.2a Extract 768-dim patch embeddings (Encoder frontend, before transformer)
  - [ ] 5.2b Extract discrete token IDs (Tokenizer full pipeline)
  - [ ] 5.2c Store (patch_embedding, token_id) pairs
- [ ] 5.3 Analyze token distribution (how many patches per token ID?)
- [ ] 5.4 Build codebook: average patch embeddings per token ID → (1024 × 768)
- [ ] 5.5 Compute static embeddings: pass each centroid through Encoder transformer (single-patch)
- [ ] 5.6 Save artifacts:
  - [ ] codebook.npy (1024 × 768)
  - [ ] embeddings.npy (1024 × 768)
  - [ ] token_counts.json

## 6. Static Embeddings - Inference (Part E)
- [ ] 6.1 Implement quantization: patch → nearest codebook centroid (L2 distance)
- [ ] 6.2 Implement fast inference pipeline: ONNX frontend → quantize → lookup → pool
- [ ] 6.3 Benchmark static pipeline: batch sizes 1, 4, 16, 32
- [ ] 6.4 Run ESC-50 5-fold CV with static embeddings
- [ ] 6.5 Compare accuracy vs full BEATs

## 7. Analysis (Part F)
- [ ] 7.1 Plot throughput vs batch size curves (all configs)
- [ ] 7.2 Create split time breakdown charts per stage
- [ ] 7.3 Plot token distribution histogram
- [ ] 7.4 Plot accuracy vs speedup Pareto frontier
- [ ] 7.5 Identify optimal operating points
- [ ] 7.6 Calculate final speedup ratios

## 8. Deliverables
- [ ] 8.1 Save all benchmark results as JSON in artifacts/benchmarks/
- [ ] 8.2 Save ONNX models in artifacts/models/
- [ ] 8.3 Save codebook + embeddings in artifacts/models/static_embeddings/
- [ ] 8.4 Save distillation data in artifacts/distillation/
- [ ] 8.5 Generate visualization plots in artifacts/figures/
- [ ] 8.6 Document findings in results.md
- [ ] 8.7 Write recommendations for optimal settings
