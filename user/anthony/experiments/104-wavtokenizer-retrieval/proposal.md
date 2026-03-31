# Proposal: WavTokenizer Codec Retrieval for Bird Audio

**Status**: proposed
**Created**: 2026-03-31
**Author**: Anthony

## Hypothesis

WavTokenizer's single-codebook discrete tokens (75 tokens/sec, 4096 vocabulary) preserve enough spectral structure in bird vocalizations to enable species-level retrieval via sequence matching on focal recordings.

## Background

Neural audio codecs compress waveforms into discrete token sequences that can be searched with techniques from bioinformatics (k-mer indexing, seed-and-extend alignment). WavTokenizer achieves extreme compression (75 tokens/sec, single codebook) while outperforming multi-codebook codecs on reconstruction quality — but it was trained on speech and music, not bird audio. This experiment tests whether the representation transfers to bioacoustics.

Motivated by: `docs/concepts/codec-motif-search.md`, `docs/concepts/discrete-audio-token-sequence-model.md`

## Method

### Approach

Three-phase feasibility spike, each gating the next:

1. **Round-trip quality**: Tokenize focal recordings through WavTokenizer, decode back, measure whether BirdNET can still classify the reconstructed audio. If the codec destroys species-discriminative features, stop here.

2. **Focal retrieval**: Quantize the 4096-code vocabulary to ~64-256 super-tokens (k-means on codebook embeddings), extract k-mer fingerprints from focal recordings, test whether same-species recordings cluster / retrieve correctly.

3. **Soundscape search**: Search for focal-derived motifs in the 739 labeled soundscape windows from exp 101. Measure whether motif hits correlate with ground-truth species presence.

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 focal recordings (35K files), 739 labeled soundscape windows (from exp 101) |
| Compute | Single GPU for tokenization (batch encode), CPU for search |
| Dependencies | WavTokenizer (`novateur/WavTokenizer` on HF), Perch v2 ONNX (`justinchuby/Perch-onnx`), scikit-learn (k-means), onnxruntime |
| Code | `user/anthony/experiments/104-wavtokenizer-retrieval/` |

### Procedure

1. Install WavTokenizer, load pretrained large model
2. Select a diverse subset: ~10 species, ~20 focal recordings each (mix of taxa: birds, frogs, insects if available)
3. **Phase 1**: Encode → decode round-trip on the subset. Run BirdNET on original and reconstructed. Compare confidence scores.
4. **Phase 2**: Encode all subset focal recordings to token sequences. Extract codebook embeddings, k-means to 64 super-tokens. Map sequences to super-token space. Build k-mer index (k=4-6). For each recording, query the index and measure retrieval precision@10.
5. **Phase 3** (if Phase 2 passes): Encode labeled soundscape windows. Search for focal motifs using seed-and-extend. Score per-window species presence. Compare against ground truth labels.

### Variables

- **Independent**: codec (WavTokenizer large), super-token vocabulary size (64, 128, 256), k-mer length (4, 5, 6)
- **Dependent**: BirdNET confidence delta, retrieval precision@10, soundscape match recall
- **Controlled**: sample rate (24kHz — WavTokenizer native), species subset, BirdNET model version

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Perch embedding cosine similarity | Cosine similarity of Perch v2 embeddings between original and round-tripped audio |
| Perch top-k species agreement | Fraction of top-k species logits that agree between original and round-tripped |
| Retrieval precision@10 | Fraction of top-10 k-mer retrieval results that are same-species |
| Soundscape recall@0.5 | Fraction of true-positive species in labeled windows detected by motif search (at 0.5 score threshold) |

### Baseline

- Perch v2 ONNX (justinchuby/Perch-onnx) embeddings and logits on original recordings (no codec round-trip)
- Random retrieval baseline (precision@10 = proportion of same-species in corpus)

### Success Criteria

- **Confirm if**: Perch embedding cosine similarity > 0.90 or top-1 species match > 80%, AND focal retrieval precision@10 > 0.80
- **Reject if**: Perch embedding cosine similarity < 0.70 (codec destroys features) OR focal retrieval precision@10 < 0.50 (tokens not species-discriminative)
- **Phase 3 is bonus**: any positive signal on soundscape search is encouraging, no hard gate

## Limitations

- WavTokenizer operates at 24kHz (Nyquist 12kHz). Species with significant energy above 12kHz will be clipped — most bird calls are below this, but some passerines and insects may be affected.
- Perch v2 as round-trip evaluator introduces its own biases — it may be insensitive to features the codec destroys, or sensitive to artifacts the codec introduces. Embedding cosine similarity + logit agreement provide complementary views.
- Small species subset (10 species) may not generalize. Deliberately pick diverse taxa to mitigate.
- k-mer search is exact on super-tokens. Tempo variation within species may require DTW extension (Phase 2b, not scoped here).
