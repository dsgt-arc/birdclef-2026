# Tasks

## 0. Infrastructure
- [x] 0.1 Create sbatch/logs/ directory structure
- [x] 0.2 Create sbatch/download-weights.sbatch
- [x] 0.3 Create sbatch/setup-original-ssamba.sbatch
- [x] 0.4 Create sbatch/run-verification.sbatch
- [x] 0.5 Create scripts/extract_original_embeddings.py
- [x] 0.6 Create scripts/extract_hf_embeddings.py
- [x] 0.7 Create scripts/verify_numerical_equivalence.py

## 1. Setup
- [x] 1.1 Initialize `ssamba_hf` subpackage with `uv init`
- [x] 1.2 Download SSAMBA pretrained checkpoint (ssamba_small_400.pth)
  - Checkpoint at: `~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth`
- [x] 1.3 Inspect checkpoint structure and document all weight keys
  - **Finding**: Uses bidirectional Mamba with `_b` suffix weights (see section 2B)
- [x] 1.4 Verify HuggingFace transformers version has MambaMixer
- [ ] 1.5 Create test audio file for verification

## 2. Implementation
- [x] 2.1 Create `ssamba_hf/config.py` with `SSAMBAConfig`
- [x] 2.2 Create `ssamba_hf/layers.py` with `RMSNorm`, `AudioPatchEmbed`
- [x] 2.3 Create `ssamba_hf/mamba.py` with custom `BidirectionalMamba` (pure PyTorch SSM)
  - Implemented custom bidirectional Mamba with both forward and backward weights
  - Uses pure PyTorch selective scan (no CUDA kernels required)
- [x] 2.4 Create `ssamba_hf/model.py` with `SSAMBAModel`
  - Added middle CLS token positioning
  - Fixed residual handling to match VisionMamba
- [x] 2.5 Implement `convert_ssamba_weights()` in `ssamba_hf/convert.py`
  - 413 weights loaded successfully
  - norm.bias values ignored (original triton RMSNorm has no bias)
- [x] 2.6 Add `get_embeddings()` method for feature extraction
- [x] 2.7 Export public API in `ssamba_hf/__init__.py`

## 2B. Bidirectional Mamba Support (COMPLETED)
The original SSAMBA uses a custom bidirectional Mamba from Vision Mamba (Vim), not standard Mamba:
- Each layer has separate forward and backward parameters (`_b` suffix)
- HuggingFace's MambaMixer only supports unidirectional Mamba

**Solution: Pure PyTorch BidirectionalMamba**

Implemented a custom `BidirectionalMamba` class that:
- Has both forward (A_log, D, conv1d, x_proj, dt_proj) and backward (_b suffix) weights
- Uses pure PyTorch SSM (no CUDA kernels required)
- Processes both directions within each layer call

Tasks:
- [x] 2B.1 Analyze SSAMBA bidirectional Mamba architecture from Vim/vim/models_mamba.py
- [x] 2B.2 Implement custom BidirectionalMamba class (not using HF MambaMixer)
- [x] 2B.3 Update weight conversion with detailed analysis and diagnostics
- [x] 2B.4 Verify forward pass produces same shape outputs
- [x] 2B.5 Run verification to confirm cosine similarity > 0.999 ✓ (achieved 1.000000!)

## 3. Verification (COMPLETED)
- [x] 3.1 Set up original SSAMBA environment
  - Successfully set up with mamba_ssm and triton on GPU
- [x] 3.2 Load original SSAMBA on GPU and extract embeddings for test audio
- [x] 3.3 Load ported SSAMBA on CPU and extract embeddings for same audio
- [x] 3.4 Compare outputs: cosine similarity, max diff, mean diff
  - **Result: Cosine similarity = 1.000000, Max diff = 5.53e-05**
- [x] 3.5 Debug any discrepancies in layer-by-layer comparison
  - Fixed: RMSNorm bias (ignored in original), middle CLS token, residual handling

## 4. Benchmarking
- [ ] 4.1 Measure CPU inference time (single sample)
- [ ] 4.2 Measure CPU inference time (batch of 8)
- [ ] 4.3 Compare memory usage

## 5. Analysis
- [ ] 5.1 Document numerical equivalence results
- [ ] 5.2 Document performance characteristics
- [ ] 5.3 Write results.md with findings
- [ ] 5.4 Decide if ready for integration into main codebase

## Current Status (2026-01-25)

**VERIFIED: HF port is numerically equivalent to original SSAMBA!**

Final Test Results:
```
Cosine Similarity: 1.000000 (PASS!)
Max Absolute Diff: 5.53e-05 (floating point precision)
Mean Absolute Diff: 4.88e-06
HF embeddings norm: 20.7406
Original embeddings norm: 20.7406
```

**Key Fixes Applied:**
1. Custom `BidirectionalMamba` implementation with pure PyTorch SSM
2. Middle CLS token positioning (`use_middle_cls_token=True`)
3. Correct residual handling matching VisionMamba's non-fused path
4. RMSNorm without bias (checkpoint has bias values, but original ignores them)
5. Fixed original extraction script to use correct forward path (`if_bidirectional=False`)

**Previously Identified Issues:**
- Original extraction script used wrong forward path (paired-layer instead of single-layer)
- RMSNorm bias was being loaded but should be ignored (original triton RMSNorm has no bias)
- CLS token was being added at beginning instead of middle

**Weight Loading Summary:**
- Matched: 413 weights (all necessary weights)
- Unexpected: 27 (norm.bias values + head.weight/bias - correctly ignored)

## Verification Workflow

1. Download weights:
   ```bash
   cd user/anthony/experiments/004-ssamba-hf-port/sbatch
   sbatch download-weights.sbatch
   ```

2. Set up original SSAMBA environment:
   ```bash
   sbatch setup-original-ssamba.sbatch
   ```

3. Run verification:
   ```bash
   sbatch run-verification.sbatch
   ```

4. Check results:
   ```bash
   cat ~/scratch/birdclef-2026/exp004-ssamba-hf/verification/comparison_report.json
   ```
