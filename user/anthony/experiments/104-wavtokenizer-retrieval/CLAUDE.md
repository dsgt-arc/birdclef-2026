# Experiment 104: WavTokenizer Codec Retrieval

## What this is

Feasibility spike testing whether WavTokenizer's discrete audio tokens preserve
species-discriminative features for bird audio, enabling sequence-based retrieval
via k-mer matching (borrowed from bioinformatics).

## Status: PROPOSED (not yet run)

Code is implemented, needs GPU job submission.

## Architecture

- **Codec**: WavTokenizer large (80K hrs, speech+audio+music), 75 tokens/sec, single codebook, 4096 vocabulary, 24kHz
- **Evaluator**: Perch v2 ONNX (`justinchuby/Perch-onnx`) — 1536-dim embeddings + 14,795-class logits
- **Retrieval**: k-means super-tokens (codebook → 64-256 clusters) + k-mer inverted index

## Three gated phases

1. **Round-trip quality** (GPU tokenize → CPU eval): encode focal recordings → decode → compare Perch embeddings (cosine sim) and logits (top-k agreement). Gate: cos_sim > 0.90 or top-1 match > 80%.
2. **Focal retrieval** (CPU): quantize codebook to super-tokens, build k-mer index, leave-one-out precision@10. Gate: P@10 > 0.80.
3. **Soundscape search** (CPU): extract species motifs, search 739 labeled windows. Bonus — no hard gate.

## Key files

| File | Purpose |
|------|---------|
| `src/birdclef_104/select_subset.py` | Pick ~10 species × 20 recordings across taxa |
| `src/birdclef_104/tokenize.py` | WavTokenizer encode/decode + codebook extraction (GPU) |
| `src/birdclef_104/roundtrip_eval.py` | Phase 1: Perch v2 ONNX embedding/logit comparison |
| `src/birdclef_104/codebook.py` | k-means quantization of 4096-code codebook |
| `src/birdclef_104/kmer_index.py` | Phase 2: k-mer inverted index + retrieval scoring |
| `src/birdclef_104/soundscape_search.py` | Phase 3: motif search in labeled soundscape windows |
| `configs/exp104.yaml` | All paths, model config, retrieval hyperparameter grid |

## Running

```bash
cd user/anthony/experiments/104-wavtokenizer-retrieval

# Job 1: GPU — subset selection + WavTokenizer encode/decode (~30 min)
sbatch sbatch/tokenize.sbatch

# Job 2: CPU — Perch eval + codebook + k-mer retrieval (~30 min)
# Run after Job 1 completes
sbatch sbatch/eval.sbatch
```

## Dependencies

- **WavTokenizer**: vendor submodule at `vendor/WavTokenizer/`, loaded via PYTHONPATH
- **Perch v2 ONNX**: `justinchuby/Perch-onnx` on HuggingFace, auto-downloaded
- **HF model**: `novateur/WavTokenizer-large-speech-75token`, auto-downloaded
- Persistent venv at `~/scratch/birdclef-2026/.venv-104`

## Artifacts (after running)

```
artifacts/
├── subset.json                    # Selected species + files
├── codebook_embeddings.npy        # 4096 × D codebook vectors
├── tokens/{species_id}/*.npy      # Per-file token sequences (int32)
├── reconstructed/{species_id}/*.wav  # Decoded audio (24kHz)
├── soundscape_tokens/*.npy        # Per-soundscape token sequences
├── supertoken_map_{K}.json        # Code → super-token mappings
├── roundtrip_scores.parquet       # Perch cosine sim + logit agreement
├── retrieval_results.parquet      # Precision@10 per (K, k) config
└── soundscape_results.parquet     # Motif search scores vs ground truth
```

## Known issues / watch for

- WavTokenizer operates at 24kHz (Nyquist 12kHz) — species with energy above 12kHz will be clipped
- HF model filenames are auto-detected (lists repo files for .yaml/.ckpt) — if the HF repo structure changes, may need to set `config_filename` and `model_filename` in exp104.yaml
- WavTokenizer vendor code uses non-standard imports (`from decoder.pretrained import ...`) — requires PYTHONPATH set by sbatch/_setup-venv.sh
- The `_setup-venv.sh` loads cuda/12.6.1 even for CPU eval jobs (harmless but wasteful)
