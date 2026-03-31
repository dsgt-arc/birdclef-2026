# Tasks

## 1. Setup
- [ ] 1.1 Install WavTokenizer and verify model loads (pretrained large from HF)
- [ ] 1.2 Select species subset: ~10 species across taxa (birds, frogs, insects), ~20 focal recordings each
- [ ] 1.3 Verify audio resampling pipeline (native SR → 24kHz mono for WavTokenizer)

## 2. Phase 1: Round-Trip Quality
- [ ] 2.1 Batch encode subset focal recordings to WavTokenizer tokens
- [ ] 2.2 Decode tokens back to audio, save as wav
- [ ] 2.3 Run Perch v2 ONNX on original and reconstructed audio
- [ ] 2.4 Compare embeddings (cosine similarity) and logits (top-k agreement) per species
- [ ] 2.5 Listen to a few reconstructions (spot-check quality by ear)
- [ ] 2.6 **Gate decision**: proceed if cosine similarity > 0.90 or top-1 match > 80%

## 3. Phase 2: Focal Retrieval
- [ ] 3.1 Extract WavTokenizer codebook embeddings (4096 x D)
- [ ] 3.2 K-means cluster to super-tokens (try 64, 128, 256)
- [ ] 3.3 Map all focal token sequences to super-token sequences
- [ ] 3.4 Build k-mer index (try k=4, 5, 6)
- [ ] 3.5 Leave-one-out retrieval: for each recording, query index, measure precision@10
- [ ] 3.6 Ablation: which super-token vocab size and k-mer length work best?
- [ ] 3.7 **Gate decision**: proceed if precision@10 > 0.80

## 4. Phase 3: Soundscape Search (bonus)
- [ ] 4.1 Encode 739 labeled soundscape windows to super-token sequences
- [ ] 4.2 Extract species motifs from focal recordings (most frequent k-mer patterns per species)
- [ ] 4.3 Search soundscape windows for motif matches
- [ ] 4.4 Score matches against ground truth labels
- [ ] 4.5 Analyze: which species/taxa are easiest/hardest to find?

## 5. Analysis
- [ ] 5.1 Document results in results.md
- [ ] 5.2 If positive: outline next steps (scale up, DTW extension, competition integration)
- [ ] 5.3 If negative: document failure mode (what the codec loses, what doesn't match)
