# Concept: Neural Audio Codec Motif Search for Species Detection

**Status**: nascent
**Created**: 2026-03-24

## Idea

Use neural audio codecs (DAC, EnCodec, WavTokenizer) to convert audio into
discrete token sequences, then apply sequence matching algorithms (SAX, DNA
alignment, suffix arrays) to find species-specific call patterns across
soundscapes. This reframes species detection as **subsequence search** rather
than classification.

## Motivation

Current species detection runs a classifier on fixed 5s windows and outputs
per-window probabilities. This has limitations:

- A 0.3s call gets diluted by 4.7s of silence in the window
- Embedding-based similarity loses temporal structure
- No way to find *where* in a soundscape a species calls without SED

Codec tokenization + sequence search would:
- Find a short call (50-900 tokens) inside a long soundscape (54K-540K tokens)
- Preserve the temporal shape of the call
- Enable approximate matching (same species, slightly different call)
- Use battle-tested sequence search algorithms from bioinformatics

## Audio Codec Token Rates

| Codec | Sample rate | Frame rate | Codebooks | Tokens/sec | 5s clip |
|-------|-----------|-----------|-----------|-----------|---------|
| DAC 44kHz | 44,100 | 86 Hz | 9 | 775 | 3,875 |
| DAC 16kHz | 16,000 | 50 Hz | 12 | 600 | 3,000 |
| EnCodec 48kHz | 48,000 | 150 Hz | varies | up to 2,400 | 12,000 |
| WavTokenizer | 24,000 | 75 Hz | 1 | 75 | 375 |

For search, two options:
- **WavTokenizer** (75 tokens/sec, single codebook): cleanest for search, one token per frame
- **DAC codebook 1** (86 tokens/sec, coarse structure only): search on the first RVQ level for speed, verify with deeper levels

## How It Would Work

### Step 1: Tokenize everything

```
Focal recordings (35K files) → codec → token sequences per species
Soundscapes (10K files × 60s) → codec → token database (540M tokens at 900/sec)
```

### Step 2: Extract species motifs from focal recordings

For each species, find the characteristic call pattern:
- Tokenize all focal recordings of that species
- Find recurring subsequences across recordings (matrix profile or
  frequent n-gram extraction)
- The most frequent motif IS the species' call signature

### Step 3: Search soundscapes for motifs

Find each species' motif in the soundscape token database.

**DNA-style alignment tools** are designed for exactly this:

| Tool | What it does | Why it fits |
|------|-------------|------------|
| BLAST | Fast approximate subsequence search | Find a 450-token call in 540M-token database |
| minimap2 | Seed-and-extend alignment | Handle timing variation between calls |
| Suffix arrays | Exact substring matching | Find identical motif occurrences |
| k-mer indexing | Index all k-length subsequences | Pre-index soundscapes for instant query |

### Step 4: Score matches

For each soundscape window:
- Count how many species motifs match (and at what quality)
- High-quality matches → species present
- Multiple matches of the same motif → repeated calls → high confidence

## Multi-Codebook Strategy

DAC's 9 codebook levels encode progressively finer detail:
- **Level 1**: coarse structure (overall call shape, loudness envelope)
- **Levels 2-4**: medium detail (harmonic structure, frequency modulation)
- **Levels 5-9**: fine detail (timbre, noise texture)

**Hierarchical search**:
1. Search codebook 1 tokens only (86 tokens/sec) — fast, coarse matching
2. Verify matches using codebooks 2-4 — confirm the harmonic structure matches
3. Skip codebooks 5-9 — fine detail varies too much between individuals

This is analogous to BLAST's seed-and-extend: fast coarse search, then
careful alignment on candidates.

## Scale Analysis

| Item | Count |
|------|-------|
| Focal species | 206 |
| Focal recordings | 35,549 |
| Soundscape files | 10,658 × 60s |
| Soundscape tokens (DAC 86Hz, codebook 1) | ~55M |
| Soundscape tokens (DAC 86Hz, all 9 codebooks) | ~495M |
| Soundscape tokens (WavTokenizer 75Hz) | ~48M |
| Typical call motif | 50-500 tokens |

55M tokens is small by bioinformatics standards — a human genome is 3B base
pairs. k-mer indexing would make queries near-instant.

## What This Gives Us

- **Species presence without classification**: find which species occur in
  soundscapes purely by pattern matching, no trained classifier needed
- **Temporal localization for free**: know exactly when a species calls
  (the match position in the soundscape)
- **Works for zero-shot species**: any species with a focal recording can
  be searched for, even without training data
- **Confidence from repetition**: a species that matches 5 times in a
  soundscape is more confidently present than one that matches once

## Open Questions

- **Do codec tokens preserve species-discriminative features?** Codecs are
  trained on speech/music. Bird calls have different characteristics
  (higher frequencies, FM sweeps, harmonics). Need to test round-trip quality.
- **How robust is matching across noise levels?** Clean focal recording
  tokens vs noisy soundscape tokens — does the same call produce similar
  enough tokens for matching?
- **What k-mer size works for calls?** Too short = too many false matches.
  Too long = misses slight variations. Needs empirical tuning.
- **Does the first codebook alone capture enough?** If coarse structure
  isn't species-discriminative, hierarchical search won't work.
- **Computational cost of tokenization?** Encoding 600 hours of audio
  through DAC takes GPU time. One-time cost, but needs benchmarking.

## Relationship to Other Work

- **Complements exp 102**: if the codec search identifies which species
  are in soundscapes, that information feeds into pseudo-labeling
- **Complements exp 101**: could replace or supplement the teacher model
  ensemble for soundscape species detection
- **Extends the discrete token concept**: `docs/concepts/discrete-audio-token-sequence-model.md`
  proposes sequence modeling over tokens; this proposes search over tokens
- **Alternative to embedding retrieval**: embedding similarity operates
  on 5s-clip-level vectors; codec search operates on sub-second patterns

## Testable Hypotheses

1. **Codec tokens are species-discriminative**: for a held-out focal
   recording, searching the remaining focal recordings with k-mer matching
   will retrieve same-species recordings with >80% precision@10.

2. **Cross-domain matching works**: a species motif extracted from focal
   recordings will match in soundscape windows where that species is
   annotated (test on the 739 labeled windows) at >70% recall.

3. **Coarse codebook is sufficient**: searching on DAC codebook 1 alone
   achieves >90% of the match quality of searching all 9 codebooks.
