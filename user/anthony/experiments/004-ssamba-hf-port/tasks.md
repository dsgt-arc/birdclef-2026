# Tasks

## 1. Setup
- [x] 1.1 Initialize `ssamba_hf` subpackage with `uv init`
- [ ] 1.2 Download SSAMBA pretrained checkpoint (amba_small_400.pth)
- [ ] 1.3 Inspect checkpoint structure and document all weight keys
- [ ] 1.4 Verify HuggingFace transformers version has MambaMixer
- [ ] 1.5 Create test audio file for verification

## 2. Implementation
- [ ] 2.1 Create `ssamba_hf/config.py` with `SSAMBAConfig`
- [ ] 2.2 Create `ssamba_hf/layers.py` with `RMSNorm`, `AudioPatchEmbed`
- [ ] 2.3 Create `ssamba_hf/mamba.py` with `BidirectionalMambaBlock`
- [ ] 2.4 Create `ssamba_hf/model.py` with `SSAMBAModel`
- [ ] 2.5 Implement `convert_ssamba_weights()` in `ssamba_hf/convert.py`
- [ ] 2.6 Add `get_embeddings()` method for feature extraction
- [ ] 2.7 Export public API in `ssamba_hf/__init__.py`

## 3. Verification
- [ ] 3.1 Load original SSAMBA on GPU and extract embeddings for test audio
- [ ] 3.2 Load ported SSAMBA on CPU and extract embeddings for same audio
- [ ] 3.3 Compare outputs: cosine similarity, max diff, mean diff
- [ ] 3.4 Debug any discrepancies in layer-by-layer comparison

## 4. Benchmarking
- [ ] 4.1 Measure CPU inference time (single sample)
- [ ] 4.2 Measure CPU inference time (batch of 8)
- [ ] 4.3 Compare memory usage

## 5. Analysis
- [ ] 5.1 Document numerical equivalence results
- [ ] 5.2 Document performance characteristics
- [ ] 5.3 Write results.md with findings
- [ ] 5.4 Decide if ready for integration into main codebase
