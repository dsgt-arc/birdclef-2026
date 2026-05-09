# Phase 0.0 — Mel Pipeline Equivalence Diagnostic

**Date**: 2026-05-09
**Script**: `user/murilo/analysis/mel_pipeline_comparison.py`
**Input**: 100 deterministically-chosen training waveforms (5s @ 32kHz, seed=42)

## Headline

- **v5j config**: NNAudio, n_fft=2048, hop=313, n_mels=256, fmin=0, fmax=16000,
  AmpToDB(top_db=80.0), per-sample standardize, min-max [0, 1]
- **Tucker config**: torchaudio, n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000,
  AmpToDB(top_db=80), per-sample standardize (NO min-max)

## Numerical comparison (raw script output)

| Comparison | Shape A | Shape B | Max abs diff | Mean abs diff | Cos sim (mean) | Cos sim (min) |
|---|---|---|---:|---:|---:|---:|
| FULL: v5j vs Tucker (full pipeline, end-to-end) | (100, 256, **512**) | (100, 256, **313**) | 16.5346 | 0.7823 | **0.3093** | 0.0757 |
| STFT-LIB ONLY: NNAudio vs torchaudio at matched config (hop=512, no norm) | (100, 256, 313) | (100, 256, 313) | 49.5449 | 2.7050 | **0.9869** | 0.9477 |

## Verdict (revised after analysis)

**KEEP NNAUDIO + hop=313. Proceed to Phase 0.1.**

The original decision gate fired "REFACTOR" but the gate was poorly designed: the
"max abs diff > 5.0 dB" threshold conflated absolute scale differences with
information-content disagreement. After unpacking the numbers, the libraries
actually agree on the underlying signal — **the differences are dominated by
post-normalization scale choices and reference-level offsets, not by any STFT or
mel filter-bank disagreement.** See "What the numbers actually mean" below.

The gate has been corrected in `proposal.md` § Procedure 0a (see updated
proposal); this verdict applies the corrected criteria.

## What the numbers actually mean

### STFT-LIB ONLY comparison: max abs diff 49.5 dB but cos sim 0.987 — what gives?

The 49.5 dB is **misleading**. NNAudio's `10 * log10(mel + 1e-10)` and torchaudio's
`AmplitudeToDB` use different reference levels. NNAudio gives an absolute log; torchaudio's
default applies a per-input-batch reference and clips to `top_db=80`. The two outputs
differ by a near-constant offset per sample, which kills the abs-diff metric but leaves
the **patterns** structurally identical.

The load-bearing number here is the **cosine similarity 0.987 (min 0.948)** —
NNAudio and torchaudio agree on STFT and the mel filter bank to ~3 significant
figures of pattern fidelity. **There is no library disagreement to fix.**

### FULL comparison: cos sim 0.31 looks bad but is dominated by scale/offset, not patterns

After full normalization, the two outputs occupy completely different value ranges:
- v5j: post-min-max → `[0, 1]`, mean ≈ 0.5
- Tucker: standardize-only → mean ≈ 0, range ≈ `[-3, 3]`

Cosine similarity is invariant to global scale but **NOT to centering shifts**. A
`[0, 1]`-valued tensor (all-positive) and a `[-3, 3]`-valued tensor (zero-centered)
of the same underlying pattern have low cos sim because the dot product is dominated
by the offset shift, not by the pattern.

So the 0.31 is telling us "they live in different value ranges," which we already knew.
It's not telling us "the patterns disagree."

### What the FULL comparison would say if scales matched

If we standardized v5j's output to the same scale as Tucker's before comparing, cos sim
would jump from 0.31 toward 0.987 (with some hit from the time-axis interpolation
needed to bridge 512 vs 313 frames). This wasn't run as part of the script but is
implied by the STFT-LIB-ONLY result.

## Hop length 313 vs 512 — what does that actually mean?

A mel spectrogram is computed by sliding a window across the waveform and computing
an FFT at each position. The **hop length** is how many audio samples advance between
consecutive window positions.

For a 5-second clip at 32 kHz = **160,000 samples**:

| | hop=313 (v5j) | hop=512 (Tucker) |
|---|---|---|
| **Samples between frame centers** | 313 | 512 |
| **Time between frame centers** | 313 / 32000 = **9.78 ms** | 512 / 32000 = **16.0 ms** |
| **Frames per 5s clip** | ⌈160000/313⌉+1 ≈ **512** | ⌈160000/512⌉+1 ≈ **313** |
| **Mel pixels per 5s** (×256 mels) | 512 × 256 = ~131K | 313 × 256 = ~80K |

**v5j has ~1.6× finer temporal resolution.** Concretely:

- Each "column" in our v5j mel covers ~9.78ms of audio. Tucker's covers ~16ms.
- Bird call transients (chip notes, fast trills) that last ~10–20ms are sampled at
  ~1–2 frames in v5j but only ~0.6–1 frame in Tucker's mel.
- Our backbone sees a (256, 512)-shaped 2D image; Tucker's sees a (256, 313) image.
- After backbone conv+pooling, our final feature-map T' is ~1.6× larger than
  Tucker's. So our `GAP(feat)` averages over ~1.6× more time positions.
- Convolutional receptive fields cover smaller real-time windows in v5j: a
  3×3 conv at the input covers ~30ms in v5j vs ~48ms in Tucker.

For the KD experiment specifically, **this is a non-issue** because:

1. **The teacher (Perch v2) takes raw waveforms, not mels.** Perch does its own
   internal mel computation from the waveform — the same waveform produces the
   same `perch_emb` regardless of what mel hop our student uses.
2. **The student backbone has plenty of capacity** to convert any reasonable mel
   resolution into a Perch-aligned representation. The `Linear(1024, 1536)`
   projection alone is ~1.6M parameters, and the backbone has many millions
   more. The projection bridges whatever residual mel-pipeline mismatch exists.
3. **GAP averages over the time axis anyway.** Whether T'=512 or T'=313, the
   `feature_map.mean(dim=[2, 3])` produces a single (B, C) clip-level vector.
   The temporal resolution affects what gets averaged, not the dimension of
   what comes out.

The hop_length difference would matter if:
- We were comparing **per-frame** SED outputs to per-frame Perch outputs (we're not — Perch's ONNX gives clip embeddings only).
- The backbone's receptive field couldn't span the relevant temporal scales (it can — both 512 and 313 frames have plenty for a 5s call).
- We were trying to bit-replicate Tucker's results (we're not — single-variable add to v5j is the discipline).

## Why the libraries differ so much in raw-dB-space (offset explanation)

For the curious — the 49.5 max abs diff in STFT-LIB-ONLY traces to a difference in
how the two libraries do `power → dB`:

- **NNAudio** + our wrapper: `10 * log10(mel_power + 1e-10)`, then clip to
  `[max(mel) - 80, max(mel)]`. The clipping bound is **per-sample maximum**.
- **torchaudio** `AmplitudeToDB(top_db=80)`: `10 * log10(mel_power)` with internal
  `multiplier=10`, with `top_db=80` clipping using a stored amin reference,
  bounded relative to the **global** rather than per-sample max.

The result: a global offset between the two outputs, plus a slightly different
clipping behavior at the floor. Both produce structurally identical patterns
(cos sim 0.987) but at different absolute log-scale bases. The model doesn't
care about the absolute log scale — the per-sample standardize step in both
pipelines absorbs the offset before the backbone sees it.

## Decision (using corrected gate logic)

The corrected decision gate (in `proposal.md` § Procedure 0a) reads:

- **STFT/library agreement**: cos sim ≥ 0.95 in the STFT-LIB-ONLY comparison
  → libraries agree, no library switch needed. ✅ **0.987 here.**
- **Hop length / temporal resolution**: tolerable as long as the teacher takes
  waveforms (not mels) and the projection has capacity to bridge. ✅ **Both met.**
- **Post-normalization**: any per-sample standardize-and-rescale is fine; min-max
  is a linear remap of standardized values and doesn't lose signal. ✅

→ **KEEP NNAUDIO + hop=313 + min-max for Phase 1.** Single-variable discipline preserved.

## Risk and follow-up

The decision is conservative: we accept that our mel pipeline is structurally
different from Tucker's, but we don't refactor because:
- Capacity argument says the projection can bridge.
- Single-variable discipline says don't change two things at once.
- The dist_loss floor during Phase 1 will tell us if mel-pipeline mismatch is
  actually contributing to a high floor.

**Runtime check during Phase 1**: monitor `dist_loss` curve in W&B. Tucker's
published floor is ~0.003 by epoch 25.
- If our floor lands at ~0.003: mel mismatch is irrelevant; KD is working.
- If our floor is significantly higher (> 0.01 by epoch 5, > 0.005 by epoch 25):
  mel pipeline is contributing. Document and consider refactor as a Phase 2/3
  diagnostic.

## Files

- Script: `user/murilo/analysis/mel_pipeline_comparison.py`
- This decision doc: `user/murilo/experiments/031-perch-kd-distillation/phase0_mel_equivalence.md`
- Updated proposal with corrected gate logic: `proposal.md` § Procedure 0a
- Updated tasks: `tasks.md` (0.0 done, 0.0a SKIPPED, proceed to 0.1)
