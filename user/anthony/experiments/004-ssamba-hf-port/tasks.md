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
- [ ] ~~2.3 Create `ssamba_hf/mamba.py` with `BidirectionalMambaBlock`~~ **NEEDS UPDATE**
  - HF MambaMixer is unidirectional only
  - Original SSAMBA has bidirectional weights: `A_b_log`, `D_b`, `conv1d_b`, `x_proj_b`, `dt_proj_b`
- [x] 2.4 Create `ssamba_hf/model.py` with `SSAMBAModel`
- [ ] ~~2.5 Implement `convert_ssamba_weights()` in `ssamba_hf/convert.py`~~ **NEEDS UPDATE**
  - Current: 245 weights loaded
  - Missing: `in_proj.bias`, `out_proj.bias` (48 weights)
  - Unexpected: `_b` weights (bidirectional), `norm.bias` (168+ weights)
- [x] 2.6 Add `get_embeddings()` method for feature extraction
- [x] 2.7 Export public API in `ssamba_hf/__init__.py`

## 2B. Bidirectional Mamba Support (IN PROGRESS)
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
- [ ] 2B.4 Verify forward pass produces same shape outputs
- [ ] 2B.5 Run verification to confirm cosine similarity > 0.999

## 3. Verification
- [ ] 3.1 Set up original SSAMBA environment
  - **BLOCKED**: numpy/scipy/triton compatibility issues with mamba_ssm
  - Challenges: mamba_ssm triton ops require specific setup
  - Run: `cd sbatch && sbatch setup-original-ssamba.sbatch`
- [ ] 3.2 Load original SSAMBA on GPU and extract embeddings for test audio
- [ ] 3.3 Load ported SSAMBA on CPU and extract embeddings for same audio
- [ ] 3.4 Compare outputs: cosine similarity, max diff, mean diff
  - Run: `cd sbatch && sbatch run-verification.sbatch`
- [ ] 3.5 Debug any discrepancies in layer-by-layer comparison

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

**Completed:**
- Verification infrastructure (sbatch scripts, extraction scripts)
- Initial HF port with HuggingFace MambaMixer (90.3% cosine similarity)
- Custom `BidirectionalMamba` implementation with both forward and backward weights
- Enhanced weight conversion with checkpoint analysis

**Next Steps:**
1. Run verification with new bidirectional implementation
2. Debug any remaining weight mapping issues
3. Confirm cosine similarity > 0.999

**Previous Test Results (with HF MambaMixer):**
```
Cosine Similarity: 0.903 (FAIL - needed >0.999)
Only 245/413 weights loaded
Missing: all _b suffix weights
```

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
