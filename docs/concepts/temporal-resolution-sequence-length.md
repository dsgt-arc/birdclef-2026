# Concept: Temporal Resolution and Sequence Length in Audio Models

**Status**: developing
**Created**: 2026-01-30
**Updated**: 2026-02-18

## Idea

Token-based audio models (BEATs, AST, SSAMBA) and CNN-based models (ConvNeXt, EfficientNet, BirdNET) handle time very differently. Transformers/Mamba models patchify spectrograms into discrete tokens with explicit sequence positions, while CNNs tile receptive fields across a continuous feature map. This distinction matters for BirdCLEF because soundscape recordings are minutes long, bird vocalizations span variable durations (50ms chip notes to multi-second songs), and the competition requires CPU inference on long audio. Understanding the frame resolution, token budget, and scaling behavior of each architecture is essential for choosing (or designing) the right student model for distillation.

## Motivation

All our transformer models are trained on ~10s windows (1024 mel frames at 100fps). BirdCLEF soundscapes are 1-5 minutes. There are several intertwined problems:

1. **Token count scales quadratically** for attention-based models. A 60s clip at 16×16 patches produces ~24k tokens -- infeasible for standard attention but fine for Mamba's linear scaling.
2. **Temporal granularity is fixed at the patch level** (~160ms per 16-frame patch). Fine temporal structure within a patch is compressed into a single embedding. Bird calls with rapid frequency modulation may lose structure.
3. **CNN receptive fields grow with depth** but are bounded. A dilated ConvNeXt-1D student can tile across arbitrarily long inputs but only "sees" a fixed context window per output position.
4. **The chunking strategy** (5s chunks, 2.5s hop) is a workaround, not a solution. Chunk boundaries can split vocalizations and ignore long-range temporal context (e.g., repeated call patterns that aid identification).

The question is: what temporal resolution and sequence length should a distilled student operate at, and does the answer differ between architectures?

## Background

### Model specs (from Exp 001, 004)

| Model | Patch | Mel Bins | Hop | Frame Rate | Seq Len (10s) | Tokens (10s) | Complexity |
|-------|-------|----------|-----|-----------|---------------|-------------|------------|
| BEATs | 16×16 | 128 | 10ms | 100 fps | 1024 | ~512 | O(n^2) |
| AST | 16×16 (stride 10) | 128 | 10ms | 100 fps | 1024 | ~1200 (overlap) | O(n^2) |
| SSAMBA | 16×16 | 128 | 10ms | 100 fps | 1024 | ~512 | O(n) |
| EAT | 16×16 | 128 | 10ms | 100 fps | 1024 | ~512 | O(n^2) |

All transformer/Mamba models share the same frontend: 128 mel bins, 10ms hop, 16×16 patches. The differences are in how they process the resulting token sequence.

**Critical caveat**: These models all operate at **16kHz**, giving a Nyquist limit of 8kHz. See the "Sample rate and Nyquist" section below for why this is a problem for bioacoustics.

### CNN models

| Model | Input | Receptive Field | Notes |
|-------|-------|----------------|-------|
| BirdNET | 48kHz, 3s chunks | Fixed, architecture-dependent | Production, 6500 species |
| Perch | PCEN mel, EfficientNet | Bounded by network depth | 10k+ species |
| ConvNeXt-1D (student) | 1D token sequence | Grows with dilated convolutions | Exp 003 candidate |

CNNs don't have a hard sequence length limit -- they tile across the input. But each output position only attends to a local receptive field. This is fundamentally different from self-attention where every token sees every other token.

### Scaling to long audio

| Duration | Frames (100fps) | Patches (16×16, no overlap) | Attention cost (relative) |
|----------|-----------------|----------------------------|--------------------------|
| 5s | 500 | ~256 | 1× |
| 10s | 1000 | ~512 | 4× |
| 30s | 3000 | ~1536 | 36× |
| 60s | 6000 | ~3072 | 144× |
| 5min | 30000 | ~15360 | 3600× |

Mamba scales linearly in all these cases. Attention-based models need chunking, sparse attention, or a fundamentally different approach.

## Open Questions

- **What is the minimum temporal resolution needed for bird species ID?** Many species can be identified from a single spectrogram frame of ~160ms, but temporal patterns (trill rate, call spacing) might require finer resolution or longer context.
- **Does patch shape need to change for bioacoustics?** 16×16 patches at 10ms/frame cover 160ms × full frequency. Bird calls often have narrow frequency bands -- would tall-narrow patches (e.g., 128×2 or 64×8) capture frequency modulation better? The AST paper already showed 128×2 outperforms 16×16 from scratch (see notes below).
- **Can Mamba's linear scaling actually be exploited?** SSAMBA is 3-4x slower than BEATs on GPU despite O(n) complexity (Exp 001). Is this a constant-factor problem that improves at longer sequences, or a fundamental implementation issue?
- **What's the right chunking strategy for long audio?** 5s/2.5s-hop is arbitrary. Should chunk length match the model's training window? Should overlap be adaptive?
- **Can a CNN student with dilated convolutions match a transformer's global context?** The context-recovery gap in Exp 002 (62.6% vs 84%) suggests local context isn't enough. But dilated convolutions with sufficient depth can reach very large receptive fields -- is the issue receptive field size or the quality of long-range interactions?
- **Is there value in multi-scale tokenization?** E.g., fine-grained patches (8×8) for local structure and coarse patches (32×32) for long-range context, combined in a single sequence.

## Testable Hypotheses

1. **Mamba outperforms attention at long sequences**: For audio clips >30s, a Mamba-based student will achieve higher accuracy per FLOP than an attention-based model of similar size, because it doesn't need chunking and can process the full context.
2. **Patch shape affects bioacoustic performance**: ~~Non-square patches (e.g., 32×8 frequency-tall) will improve species classification accuracy on BirdCLEF data compared to standard 16×16, because bird calls have narrow frequency extent but rapid temporal modulation.~~ **CONFIRMED in Exp 004** — see experimental results below.
3. **Dilated CNN receptive field has a saturation point**: There exists a receptive field size beyond which adding more dilation layers gives diminishing returns on ESC-50/BirdCLEF, revealing the minimum context window for audio classification.
4. **Longer training windows improve distillation**: A student trained on 30s teacher windows will outperform one trained on 5s windows when evaluated on full soundscapes, because it learns temporal patterns across calls.
5. **32kHz with full-band tokens outperforms 16kHz with square tokens on BirdCLEF**: A model using 128×2 patches at 32kHz (same token count as BEATs' 16×16 at 16kHz) will achieve higher species classification accuracy on BirdCLEF data, because it preserves high-frequency content and captures temporal modulation that 16kHz 16×16 patches lose.

## Connections

- **Based on**: [Exp 001 results](../../user/anthony/experiments/001-encoder-benchmark/results.md), [Exp 002 context collapse](../../user/anthony/experiments/002-beats-inference-optimization/results.md), [Exp 004 SSAMBA port](../../user/anthony/experiments/004-ssamba-hf-port/results.md)
- **Validated by**: [Exp 004 Frame-Based Tokenization](../../user/murilo/experiments/004-frame-tokenization/results.md) — hypothesis #2 confirmed, 128×4 achieves cmAP=0.118 (+37% over 16×16 baseline)
- **Related to**: [context-recovery](context-recovery.md), [distilled-sequence-model](distilled-sequence-model.md), [ssamba-audio-mamba](ssamba-audio-mamba.md), [audio-pipeline-bottlenecks](audio-pipeline-bottlenecks.md)
- **Could lead to**: Vertical frame tokenization + pretrained ViT; data augmentation for overfitting; horizontal frame tokenization; Mamba vs attention long-sequence benchmark

## Experimental Results (Exp 004: Frame-Based Tokenization)

[Exp 004](../../user/murilo/experiments/004-frame-tokenization/results.md) directly tested hypothesis #2 by training ViT-Small from scratch on HSN with vertical frame patches (128×1, 128×2, 128×4) vs the standard 16×16 baseline. Three pipeline iterations were run.

### Key findings (streaming pipeline — canonical results)

**Vertical frames outperform 16×16 square patches** by 15–37% in cmAP:

| Config | Patch | Tokens | cmAP | Lift vs Baseline |
|--------|-------|--------|------|-----------------|
| Baseline | 16×16 | 248 | 0.086 | — |
| C: 128×1 | 128×1 | 500 | 0.099 | **+15%** |
| A: 128×2 | 128×2 | 250 | 0.115 | **+34%** |
| **B: 128×4** | **128×4** | **125** | **0.118** | **+37%** |

**128×4 (40ms, 125 tokens) is the best config**: The coarser 40ms temporal resolution with the shortest sequence (125 tokens) produces the highest cmAP. The shorter sequence allows more effective self-attention with ViT-Small's 6 heads on HSN's ~66k training chunks.

**128×1 (10ms, 500 tokens) is the weakest vertical config**: 500 tokens is too many for effective attention — it converges earliest (epoch 6) and overfits quickly. This suggests a practical upper bound on sequence length for ViT-Small.

**Best from-scratch result nearly matches pretrained**: 128×4 (0.118) is within 2.5% of exp 003's pretrained ViT-Small (0.121), showing that better tokenization can largely compensate for the lack of ImageNet pretraining.

### Implications for this concept

- **Hypothesis #2 is confirmed**: patch shape matters for bioacoustics. Full-band vertical frames (128×W) are a better inductive bias than 16×16 squares.
- **The token count tradeoff is real**: 125 tokens (128×4) vs 250 (128×2) vs 500 (128×1) shows that shorter sequences can outperform longer ones, because self-attention is more effective over shorter sequences.
- **20–40ms is the sweet spot for temporal resolution**: 128×1 (10ms) consistently underperforms, while 128×4 (40ms) and 128×2 (20ms) are close. Bird call temporal modulation on HSN is well-captured at 40ms resolution.
- **Scaling implications**: with larger datasets (BirdCLEF competition data), finer resolution configs (128×2, 128×1) might become more competitive as the model has more data to learn from longer sequences.

## Notes

### Sample rate and Nyquist: 16kHz is not enough for birds

Most passerine songbirds vocalize up to **8-10 kHz**, with some species reaching **12 kHz** and exceptional species (hummingbirds) producing harmonics up to 28 kHz. The Nyquist criterion requires sampling at 2× the highest frequency of interest.

| Sample Rate | Nyquist Limit | Covers Bird Calls? | Used By |
|-------------|--------------|-------------------|---------|
| 16kHz | 8kHz | **Marginal** -- cuts off upper harmonics | BEATs, AST, SSAMBA, EAT, AudioSet |
| 24kHz | 12kHz | Adequate for most species | WavTokenizer |
| 32kHz | 16kHz | **Good** -- covers all bird vocalizations | Perch, BirdSet, BirdCLEF competition |
| 48kHz | 24kHz | Excellent -- full headroom | BirdNET |

**BirdCLEF competition data is 32kHz.** BirdNET operates at 48kHz. Perch operates at 32kHz. The 16kHz models from the AudioSet/speech world are actively **losing discriminative information** for birds.

This has implications for all our ESC-50 benchmarks (Exp 001): ESC-50 environmental sounds (rain, sirens, dog barks) don't have critical content above 8kHz, but bird calls do. **Results at 16kHz on ESC-50 may not predict performance on BirdCLEF** because the frequency truncation doesn't hurt ESC-50 but will hurt bird species ID.

For the spectrogram frontend, the sample rate changes what the mel bins cover:

| Sample Rate | 128 Mel Bins Cover | Per-Bin Resolution (approx) |
|-------------|-------------------|----------------------------|
| 16kHz | 0-8 kHz | ~63 Hz/bin (mel-scaled) |
| 32kHz | 0-16 kHz | ~125 Hz/bin (mel-scaled) |
| 48kHz | 0-24 kHz | ~188 Hz/bin (mel-scaled) |

At 32kHz, the same 128 mel bins now span twice the frequency range. If bird calls occupy 1-10 kHz, that's ~80 of 128 bins at 32kHz vs nearly all 128 bins at 16kHz. There's a tradeoff: higher sample rate gives more frequency coverage but each bin is coarser. We could also increase to 256 mel bins at 32kHz to maintain per-bin resolution while covering the full range.

### Token sketches at 32kHz (Perch-like analysis)

To design tokens that match the analysis window of production bioacoustic models (Perch at 32kHz, BirdNET at 48kHz), we need to think about what a token "sees" in time-frequency space. Here we sketch several patch configurations at **32kHz, 128 mel bins, 10ms hop** and compare them to the BEATs baseline.

#### What BEATs 16×16 actually sees (16kHz baseline)

```
Frequency (mel)
  8 kHz ┌────────────────────────────────────────────┐
        │  ░░░░░░░░░░░░░░░░                          │
        │  ░░ 16×16 patch ░░  ← 1/8 of freq range    │
        │  ░░  160ms wide ░░    (~1 kHz band)         │
        │  ░░░░░░░░░░░░░░░░                          │
        │                                             │
        │         (7 more patches fill the column)    │
        │                                             │
  0 kHz └────────────────────────────────────────────┘
        0ms                                      10s

  Each patch: 160ms × ~1kHz mel band
  Problem: A bird call at 6-8kHz spanning 50ms occupies parts of
  2 patches temporally and 2 patches in frequency. The call is
  fragmented across 4 tokens.
```

#### Tall-narrow tokens at 32kHz: what they see

**Configuration A: 128×2 "frame tokens" (full-band, 20ms)**

```
Frequency (mel)
 16 kHz ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬──┐
        │█│█│█│█│█│ │ │ │ │ │ │ │ │ │ │ │ │ │ │  │
        │█│█│█│█│█│ │ │ │ │ │ │ │ │ │ │ │ │ │ │  │
        │█│█│█│█│█│ │ │ │ │ │ │ │ │ │ │ │ │ │ │  │
        │█│█│█│█│█│ │ │ │ │ │ │ │ │ │ │ │ │ │ │  │
  0 kHz └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴──┘
        0    100ms                               10s

  Each token: 20ms × full 0-16kHz band
  - A bird call FM sweep is captured as a SEQUENCE of full-band snapshots
  - Natural for species ID: the frequency contour over time IS the call
  - Same token count as 16×16 (512 per 10s) but 8× finer temporal resolution
  - All frequency info in each token -- no fragmentation across freq patches
```

**Configuration B: 128×4 "wide frame tokens" (full-band, 40ms)**

```
Frequency (mel)
 16 kHz ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─┐
        │██│██│██│██│  │  │  │  │  │  │  │  │  │  │ │
        │██│██│██│██│  │  │  │  │  │  │  │  │  │  │ │
        │██│██│██│██│  │  │  │  │  │  │  │  │  │  │ │
        │██│██│██│██│  │  │  │  │  │  │  │  │  │  │ │
  0 kHz └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─┘
        0    160ms                               10s

  Each token: 40ms × full 0-16kHz band
  - 256 tokens per 10s (half the token count of 128×2)
  - 40ms is still fine enough for most bird call modulation
  - Cheaper to process (fewer tokens) while keeping full frequency
```

**Configuration C: 64×4 "half-band tokens" (split freq, 40ms)**

```
Frequency (mel)
 16 kHz ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─┐
        │▓▓│▓▓│▓▓│▓▓│  │  │  │  │  │  │  │  │  │  │ │ ← high band
        ├──┼──┼──┼──┤  │  │  │  │  │  │  │  │  │  │ │
        │██│██│██│██│  │  │  │  │  │  │  │  │  │  │ │ ← low band
  0 kHz └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─┘
        0    160ms                               10s

  Each token: 40ms × half frequency range (0-8kHz or 8-16kHz)
  - 512 tokens per 10s (2 freq bands × 256 time steps)
  - Preserves some frequency locality (low vs high band separation)
  - A bird call at 4-8kHz mostly falls in one freq band per time step
```

#### Full comparison at 32kHz, 128 mel bins, 10ms hop

| Config | Patch (freq×time) | Token Size | Temporal Res | Freq Res | Tokens/sec | Tokens/10s | Tokens/60s |
|--------|-------------------|-----------|-------------|----------|-----------|-----------|-----------|
| BEATs baseline (16kHz) | 16×16 | 160ms × 1kHz | 160ms | 1/8 band | ~50 | 496 | 3,000 |
| **A: Full-band frame** | 128×2 | 20ms × 16kHz | **20ms** | full band | **50** | 500 | 3,000 |
| **B: Wide frame** | 128×4 | 40ms × 16kHz | 40ms | full band | **25** | 250 | 1,500 |
| **C: Half-band** | 64×4 | 40ms × 8kHz | 40ms | 1/2 band | 50 | 500 | 3,000 |
| **D: Quarter-band** | 32×4 | 40ms × 4kHz | 40ms | 1/4 band | 100 | 1,000 | 6,000 |
| **E: Frame-level** | 128×1 | 10ms × 16kHz | **10ms** | full band | 100 | 1,000 | 6,000 |

Token counts computed as: `(mel_bins // patch_freq) × (num_frames // patch_time)`. At 100 fps (10ms hop), 10s = 1000 frames, 60s = 6000 frames. BEATs gets 496 at 10s (not 512) because `1000 // 16 = 62` with 8 frames truncated; at 60s it divides evenly. Configs A and B divide evenly at all durations.

#### What a bird call looks like in each tokenization

Consider a typical warbler trill: a rapid FM sweep from 4kHz to 8kHz repeated every ~100ms for 2 seconds.

```
In BEATs 16×16 (16kHz):
  - Call occupies ~2 freq patches (bins 64-128) × ~1 time patch per sweep
  - Each 160ms patch contains ~1.6 sweeps -- temporal structure is smeared
  - Upper harmonics above 8kHz are LOST entirely
  - 2s call = ~12 tokens (fragmented across freq and time)

In Config A 128×2 (32kHz):
  - Each 20ms token captures a full-band snapshot
  - One FM sweep (~100ms) = 5 sequential tokens showing freq peak moving up
  - The SEQUENCE of tokens encodes the sweep shape -- this is the species ID cue
  - Full frequency range preserved (harmonics visible to 16kHz)
  - 2s call = ~100 tokens (clean temporal sequence)

In Config B 128×4 (32kHz):
  - Each 40ms token captures a full-band snapshot
  - One FM sweep (~100ms) = 2-3 tokens -- still captures sweep direction
  - 2s call = ~50 tokens
  - Good balance of resolution and token budget
```

Config A (128×2) and B (128×4) at 32kHz are the most natural match for bioacoustic analysis: they treat the spectrogram the way a birder reads a sonogram -- scanning left to right, seeing the full frequency band at each moment. This is exactly how Perch and BirdNET conceptually process audio, except those models use CNN receptive fields instead of explicit tokens.

### The CNN vs Transformer framing

With CNNs, the question was always "is the receptive field big enough?" -- you stack more layers or increase dilation to see more context. The model processes the whole spectrogram as a 2D image and each output position implicitly aggregates a fixed spatial neighborhood.

With token-based models, the question shifts to "how many tokens can you afford?" Every token explicitly interacts with every other token (attention) or processes a full sequential scan (Mamba). The context is global by construction, but the computational cost is per-token.

This is why the distillation problem is interesting: we want the CNN's ability to tile efficiently across long audio, but the transformer's ability to capture global context. The student architectures in Exp 003 (ConvNeXt-1D, Mamba2) sit on different points of this tradeoff.

### Practical implications for BirdCLEF

- BirdCLEF inference must run on CPU in <2 hours for the test set
- Soundscapes are typically 1-5 minutes
- A model that can process 60s at once (Mamba) vs one that chunks into 5s windows (attention/CNN) may have a significant advantage for species that are identified by call timing patterns
- The competition scoring is per-soundscape, not per-chunk, so aggregation strategy across chunks matters

### Non-square patches: prior art and opportunity

The AST paper (Gong et al., Interspeech 2021) explicitly tested 128×2 rectangular "frame" patches vs 16×16 square patches. **Rectangular patches performed better when training from scratch** (same patch area of 256). The reason everyone still uses 16×16 is that AST/BEATs/EAT all initialize from ImageNet ViT checkpoints, which only have 16×16 patch embeddings. No pretrained weights exist for non-square configurations.

SSAST removed this constraint by using self-supervised pretraining instead of ImageNet init. The SSAST codebase already supports arbitrary `fshape × tshape` with separate `fstride × tstride` (see `vendor/ssast/src/models/ast_models.py`). It ships with two modes:
- **Patch mode**: 16×16, `cluster=True` masking (standard square patches)
- **Frame mode**: 128×2, `cluster=False` masking (full-frequency-band frames)

The pretraining constraint is that stride must equal shape (no overlap to avoid shortcuts). At finetuning time, stride can be smaller for overlapping patches.

**Why this matters for distillation**: If we're training a student from scratch (distilling from a BEATs teacher), we are not constrained by pretrained patch sizes at all. We can freely choose the student's patch shape to match the structure of the data. For bioacoustics specifically:
- Bird calls have **narrow frequency extent** but **rapid temporal modulation** (FM sweeps, trills)
- Tall-narrow patches (e.g., 64×4 or 128×2) would give finer temporal resolution (~20-40ms per token) while capturing the full frequency band in one patch
- This increases the token count (more temporal slices) but each token is a more natural unit for bird audio than a 160ms × partial-frequency square

The token count tradeoff with different patch shapes for 10s audio (128 mel bins, 1024 frames):

| Patch Shape | Freq Patches | Time Patches (10s) | Tokens (10s) | Temporal Resolution | Frequency Resolution |
|-------------|-------------|-------------------|-------------|--------------------|--------------------|
| 16×16 | 8 | 62 | 496 | 160ms | 1/8 of mel range |
| 128×2 | 1 | 500 | 500 | 20ms | full mel range |
| 64×8 | 2 | 125 | 250 | 80ms | 1/2 of mel range |
| 32×4 | 4 | 250 | 1,000 | 40ms | 1/4 of mel range |
| 128×1 | 1 | 1,000 | 1,000 | 10ms (frame-level) | full mel range |

Note that 128×2 and 16×16 produce **nearly the same number of tokens** (~500) but with very different resolution tradeoffs. The 128×2 configuration gives 8× finer temporal resolution at the cost of collapsing all frequency information into each token. For bird calls where the frequency contour over time is the identifying feature, this seems like the right tradeoff.

References:
- [AST paper / GitHub](https://github.com/YuanGongND/ast) - rectangular patch ablation
- [SSAST paper (AAAI 2022)](https://sls.csail.mit.edu/publications/2022/YGong_AAAI-2022.pdf) - flexible fshape/tshape
- [ElasticAST (Interspeech 2024)](https://www.isca-archive.org/interspeech_2024/feng24c_interspeech.pdf) - variable patch sizes during training

### Sequence length vs batch size tradeoff

On CPU, processing one long sequence may be slower than batching many short chunks due to memory access patterns and SIMD utilization. A 60s Mamba forward pass processes 3072 tokens sequentially, while 12 × 5s chunks of 256 tokens each can be batched. Need to profile both strategies on CPU.
