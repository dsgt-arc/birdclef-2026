# Analysis: Why WavTokenizer Codec Retrieval Failed

**Date**: 2026-03-31

This document provides a detailed breakdown of each failure mode in experiment 104,
examining what went wrong at each phase and what the results tell us about the
viability of neural codec approaches for bioacoustic retrieval.

## Experiment Setup Recap

- **Codec**: WavTokenizer-large (80K hours speech+audio+music), 75 tokens/sec, single codebook of 4096 codes, 24kHz
- **Evaluator**: Perch v2 ONNX (Google bird classifier, 1536-dim embeddings, 14,795 species)
- **Subset**: 10 species × 20 recordings = 200 focal files, plus 66 soundscape files (739 labeled windows)
- **Species mix**: 7 birds, 2 frogs, 1 mammal (Domestic Dog as control)

---

## Phase 1: Round-Trip Quality — Complete Failure

### What happened

Encoding bird audio through WavTokenizer and decoding it back produces audio that
Perch v2 largely cannot recognize. The mean cosine similarity between original and
reconstructed Perch embeddings was **0.313** (gate: >0.90), and top-1 species
agreement was **9%** (gate: >80%).

For context, random audio pairs would have cosine similarity near 0.0 (Perch
embeddings are high-dimensional and sparse). A score of 0.31 means the codec
preserves *some* generic audio structure but loses the spectro-temporal detail
that distinguishes species.

### Per-species patterns

The results show a clear ordering by vocalization type:

| Rank | Species | Cos Sim | Vocalization Character |
|------|---------|---------|----------------------|
| 1 | Lesser Snouted Tree Frog | 0.539 | Low-frequency tonal calls, narrow bandwidth |
| 2 | Domestic Dog | 0.423 | Barking — structurally similar to human speech |
| 3 | Pale-legged Weeping Frog | 0.374 | Repetitive tonal pulses |
| 4 | Purplish Jay | 0.357 | Harsh calls, mid-frequency |
| 5 | Common Potoo | 0.340 | Low-frequency, slow modulation |
| 6 | Bare-faced Curassow | 0.337 | Low-frequency booming |
| 7 | Rufous Hornero | 0.214 | Complex duets, wide bandwidth |
| 8 | Great Kiskadee | 0.201 | Sharp, high-frequency calls |
| 9 | Chestnut-vented Conebill | 0.200 | Thin, high-pitched songs |
| 10 | White-faced Whistling Duck | 0.145 | High-pitched whistles |

**Pattern**: Species with low-frequency, tonal, speech-like vocalizations survive
the codec better. Species with high-frequency, broadband, or rapidly modulated
calls are destroyed. This directly reflects WavTokenizer's training distribution
(speech and music at ≤12kHz effective bandwidth).

### Why 24kHz Nyquist (12kHz) matters

WavTokenizer operates at 24kHz sample rate, giving a Nyquist frequency of 12kHz.
Many bird species have significant energy above 12kHz:

- Passerine contact calls: often 6-14kHz
- High-pitched alarm calls: can exceed 10kHz
- Harmonic overtones: fundamental at 4kHz → harmonics at 8, 12, 16kHz

The Chestnut-vented Conebill (cos=0.200) and Great Kiskadee (cos=0.201) are both
passerines with calls extending into the upper register. The White-faced Whistling
Duck's namesake whistle (cos=0.145) also suffers from bandwidth truncation.

However, frequency truncation alone doesn't explain the full failure. Even the
Lesser Snouted Tree Frog (mostly <4kHz) only achieved cos=0.539 — well below the
0.90 gate. The codec's learned quantization is the primary issue.

### The codebook is wrong for bird audio

WavTokenizer's 4096-code codebook was trained to reconstruct speech and music.
Each code represents a ~13ms audio "phoneme" optimized for:
- Human formant transitions (F1-F4, roughly 300-3500Hz)
- Musical harmonics and timbral structure
- Temporal patterns at speech rate (~4-6 syllables/sec)

Bird vocalizations have fundamentally different structure:
- **Frequency modulation**: rapid sweeps (50-100ms) covering 2-10kHz — no speech analog
- **Temporal resolution**: some calls have pulse rates >100Hz — faster than any speech phoneme
- **Spectral shape**: many species produce pure tones or narrow harmonics, unlike the broadband formants of speech
- **Non-stationarity**: bird calls change rapidly within 13ms windows, while speech formants are quasi-stationary at that scale

The codebook simply doesn't have entries for these patterns. When forced to quantize
a bird call, it maps to the "nearest" speech phoneme, which may share gross energy
distribution but loses the fine structure that distinguishes species.

### Domestic Dog as control

The Domestic Dog (cos=0.423, top-1=10%) is the closest to speech in the dataset.
Dog barking has:
- Similar fundamental frequency range to male speech (80-200Hz)
- Formant-like resonances
- Temporal rhythm comparable to syllables

Its relatively higher score (still far below the gate) confirms the hypothesis:
the codec works best for sounds that resemble its training distribution.

---

## Phase 2: k-mer Retrieval — Slightly Above Random

### What happened

Best configuration (K=64 super-tokens, k=4 k-mer length) achieved P@10 = **0.195**.
With 10 uniformly-sized species, random retrieval gives P@10 ≈ 0.10. So the best
result is ~2x random — statistically above chance but far from the 0.80 gate.

### Retrieval results across all 9 configurations

| K (super-tokens) | k=4 | k=5 | k=6 |
|-------------------|------|------|------|
| 64 | **0.195** | 0.186 | 0.182 |
| 128 | 0.176 | 0.180 | 0.178 |
| 256 | 0.165 | 0.159 | 0.142 |

**Trends**:
- Smaller vocabulary (K=64) works slightly better — coarser super-tokens average out codec noise
- Shorter k-mers (k=4) work slightly better — longer k-mers are too specific to match across codec distortion
- All configs are in the 0.14-0.20 range — no configuration rescues the approach

### Per-species retrieval (best config K=64, k=4)

| Species | P@10 | Notes |
|---------|------|-------|
| Purplish Jay | 0.360 | Best — harsh, distinctive calls |
| Bare-faced Curassow | 0.295 | Low-frequency booming is relatively preserved |
| Chestnut-vented Conebill | 0.275 | Surprising given low round-trip cos (0.200) |
| Domestic Dog | 0.270 | Consistent with speech-like vocalizations |
| Pale-legged Weeping Frog | 0.200 | At random baseline |
| Great Kiskadee | 0.185 | Below random for a distinctive call |
| Rufous Hornero | 0.120 | Complex duets hard to fingerprint |
| Lesser Snouted Tree Frog | 0.105 | Near random despite best round-trip score |
| Common Potoo | 0.070 | Below random |
| White-faced Whistling Duck | 0.065 | Near-zero — whistles not tokenized distinctively |

**Interesting finding**: Round-trip quality and retrieval precision are only weakly
correlated. The Lesser Snouted Tree Frog had the best cosine similarity (0.539) but
one of the worst P@10 (0.105). This means preserving "general" audio structure
(which Perch sees) doesn't imply preserving the *sequential patterns* that k-mer
matching needs. Conversely, Chestnut-vented Conebill had poor round-trip (0.200)
but decent retrieval (0.275) — suggesting some species have distinctive token
*sequences* even when individual tokens are distorted.

### Why k-mer matching fails here

The k-mer approach requires that the same acoustic event maps to the same token
sequence reliably. With a speech-trained codec:

1. **Non-deterministic mapping**: slight amplitude or background noise differences
   cause the same bird call to map to different codebook entries
2. **Context collapse**: different species' calls can map to the same token
   sequences because the codebook doesn't distinguish them
3. **Temporal aliasing**: at 75 tokens/sec (~13ms/token), rapid frequency sweeps
   are undersampled and collapse to the same token

The k-mer index then becomes a bag of largely random n-grams that happen to
co-occur by chance rather than by acoustic similarity.

---

## Phase 3: Soundscape Search — No Signal

### What happened

Per-species motif search produced essentially zero separation between present and
absent species in labeled soundscape windows:

| Species | Present Score | Absent Score | Separation |
|---------|--------------|--------------|------------|
| Bare-faced Curassow | 0.0022 | 0.0000 | +0.0022 |
| White-faced Whistling Duck | 0.0000 | 0.0000 | 0.0000 |
| All others | ~0.0000 | ~0.0000-0.0006 | ~0 or negative |

At threshold=0.01: precision=0.019, recall=0.002 (2 true positives out of 844).

**This is indistinguishable from noise.** The one mildly positive species
(Bare-faced Curassow, separation=0.002) is not meaningful given the near-zero
absolute scores.

### Why soundscape search was hopeless

Even if focal retrieval had partially worked, soundscape search faces additional
challenges:
- **Polyphonic mixing**: soundscapes contain multiple overlapping species, background
  noise, wind, rain — the codec can't separate sources
- **Amplitude variation**: focal recordings are clean and close-mic'd; soundscapes
  have distant, faint vocalizations that fall below the codec's effective quantization
  threshold
- **Duration mismatch**: 5-second soundscape windows may contain only a fragment of
  a call, while focal motifs are extracted from longer, complete vocalizations

---

## What This Tells Us

### The approach is sound, the tokenizer is wrong

The k-mer retrieval framework (codebook → super-tokens → k-mer index → sequence
matching) is a valid approach borrowed from genomics. The failure is entirely at
the tokenization layer:

1. **WavTokenizer was never designed for bird audio** — its codebook learned
   speech/music features
2. **24kHz sample rate clips relevant frequencies** — many bird calls extend above 12kHz
3. **75 tokens/sec may be too coarse** — bird call features change faster than
   speech phonemes

### Would a different codec work?

Possibly, but it would need:
- **Training on bioacoustic data** (or at minimum, fine-tuning)
- **Higher sample rate** (32-48kHz to capture full bird call bandwidth)
- **Larger codebook** or **multi-codebook** to handle the diversity of bioacoustic signals
- **Faster token rate** to capture rapid frequency modulations

No such codec exists off-the-shelf as of March 2026. Building one would be a
substantial research effort (fine-tuning WavTokenizer on bird audio, or training
a new codec from scratch on bioacoustic corpora).

### Comparison to supervised approaches

For reference, experiment 102's EfficientNet baseline (trained directly on
spectrograms) achieves **0.794 Kaggle LB** with a few hours of training. This
experiment achieved P@10=0.195 for *focal-only* retrieval — not even comparable
to the competition task. The gap confirms that learned spectral features
(from supervised or self-supervised training on relevant data) massively
outperform generic audio compression for species discrimination.

---

## Compute Cost

| Job | Type | Wall Time | Memory |
|-----|------|-----------|--------|
| tokenize (6109361) | GPU RTX 6000 | 4m 21s | 5.3 GB |
| eval (6109751) | CPU | 22m 27s | 2.4 GB |
| **Total** | | **~27 min** | |

Very lightweight — the experiment was cheap to run and provided a clear negative result.

---

## Conclusion

WavTokenizer codec retrieval is **not viable** for bird species identification.
The speech-trained codec fundamentally cannot preserve the spectro-temporal features
that distinguish bird species. This is a domain mismatch problem, not an
algorithmic one — the k-mer retrieval framework would need a bioacoustic-native
tokenizer to have any chance of working.

This experiment validates the decision to focus on supervised learning (exp 102)
and domain-specific SSL pretraining (exp 103) rather than repurposing generic
audio tools for bioacoustics.
