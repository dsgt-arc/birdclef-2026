# Tasks

## 1. Setup
- [x] 1.1 Install WavTokenizer and verify model loads (pretrained large from HF)
- [x] 1.2 Select species subset: ~10 species across taxa (birds, frogs, mammals), ~20 focal recordings each
- [x] 1.3 Verify audio resampling pipeline (native SR → 24kHz mono for WavTokenizer)

## 2. Phase 1: Round-Trip Quality — FAIL (cos=0.313, top1=9%)
- [x] 2.1 Batch encode subset focal recordings to WavTokenizer tokens
- [x] 2.2 Decode tokens back to audio, save as wav
- [x] 2.3 Run Perch v2 ONNX on original and reconstructed audio
- [x] 2.4 Compare embeddings (cosine similarity) and logits (top-k agreement) per species
- [ ] 2.5 Listen to a few reconstructions (spot-check quality by ear)
- [x] 2.6 **Gate decision**: FAIL — cos_sim=0.313 (<0.90), top1=9% (<80%)

## 3. Phase 2: Focal Retrieval — FAIL (P@10=0.195)
- [x] 3.1 Extract WavTokenizer codebook embeddings (4096 x 512)
- [x] 3.2 K-means cluster to super-tokens (64, 128, 256)
- [x] 3.3 Map all focal token sequences to super-token sequences
- [x] 3.4 Build k-mer index (k=4, 5, 6)
- [x] 3.5 Leave-one-out retrieval: for each recording, query index, measure precision@10
- [x] 3.6 Ablation: best config K=64, k=4 → P@10=0.195
- [x] 3.7 **Gate decision**: FAIL — P@10=0.195 (<0.80)

## 4. Phase 3: Soundscape Search (bonus) — no signal
- [x] 4.1 Encode 739 labeled soundscape windows to super-token sequences
- [x] 4.2 Extract species motifs from focal recordings (most frequent k-mer patterns per species)
- [x] 4.3 Search soundscape windows for motif matches
- [x] 4.4 Score matches against ground truth labels
- [x] 4.5 Analyze: zero separation, precision=0.019 at threshold=0.01

## 5. Analysis
- [x] 5.1 Document results in results.md
- [x] 5.2 ~~If positive~~: N/A — experiment rejected
- [x] 5.3 Document failure mode in analysis.md (codec domain mismatch, frequency truncation, codebook wrong for bioacoustics)
