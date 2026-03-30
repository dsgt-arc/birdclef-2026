# Design: SSAMBA SSL Pretraining

**Created**: 2026-03-30
**Status**: decisions pending — captured from scoping conversation

This doc captures the design decisions and open questions from the initial scoping
conversation. Update as decisions are made before implementation begins.

---

## Decided

### Mel configuration
- **n_mels**: 224 (validated by Murilo exp 017: +0.030 LB over 128)
- **n_fft**: 4096 (narrow frequency bins, better harmonic separation)
- **hop_length**: 320 samples = 10ms per frame at 32kHz
- **fmin**: 50 Hz (captures caiman/frog fundamentals)
- **fmax**: 16,000 Hz (32kHz Nyquist — can't go higher, source audio is 32kHz)
- **sample_rate**: 32kHz (all BirdCLEF audio is already 32kHz, no resampling)

### Patch size: 224×1 (10ms tokens, full-band)
- Each token = full 224-bin frequency profile at one 10ms frame
- 1D temporal sequence — Mamba's natural input structure
- **Why not 224×2 (20ms)**: 224×1 is strictly better — same frequency coverage, double
  temporal resolution, and Mamba is O(n) so 2× tokens = 2× compute (not quadratic)
- **Why not 250ms**: too coarse — whole bird syllables compressed into one token,
  SSL reconstruction task becomes trivial, loses fine temporal structure

### Training window: 30s
- Soundscapes are all exactly 60s → split each into two clean non-overlapping 30s halves
- Focal recordings (median 21s, mean 35s) mostly fit in one 30s window with light padding
- 30s → 3,000 tokens per window — well within SSAMBA O(n) linear regime
- Doubles training examples vs 60s (56K vs 28K windows)

### Data scope (run 103-A)
- BirdCLEF 2026 soundscapes (10,658 files) + focal recordings (35,549 files)
- BirdCLEF 2025 soundscapes deferred — adds complexity, validate 2026-only first
- Hold out 5% of soundscape files for SSL validation (stratified by site)

### Model: SSAMBA-small only (26M, embed_dim=384)
- Gate SSAMBA-base (99M) on small results — don't run both in parallel
- SSAMBA-tiny (7M) is the exp 007 baseline; small is the next step up

### Backbone code: vendor/ssamba (original CUDA implementation)
- NOT the HF port from exp 004 — pure PyTorch SSM is too slow for training
- vendor/ssamba requires mamba-ssm CUDA + causal-conv1d

---

## Open questions (must resolve before implementation)

### 1. Positional embedding at inference time (UNRESOLVED — needs decision)

SSL trains on 30s windows (3,000 tokens, pos_embed shape (1, 3001, D)).
Kaggle evaluation requires 60s soundscape inference (6,000 tokens).
Positional embeddings are tied to training sequence length — mismatch at eval.

**Option A — Bilinear interpolation** (`adapt_for_finetuning` in exp 004 HF port):
- Already implemented in `user/anthony/experiments/004-ssamba-hf-port/ssamba_hf/model.py:260`
- Bilinearly interpolates pos_embed from (1, 1, 3000) → (1, 1, 6000)
- The upper half (tokens 3000–6000) is extrapolated from training distribution
- Standard approach used by SSAMBA/SSAST/AST for different-length finetuning
- **Risk**: bilinear interp in 1D (f_dim=1 for full-band patches) is just linear
  interpolation, which may not capture whatever positional patterns were learned

**Option B — No positional embeddings** (speculative, not validated):
- Initialize pos_embed to zeros and freeze — effectively disabled
- Mamba recurrence encodes temporal order implicitly (token n has seen tokens 0..n-1)
- Pos embeddings primarily provide *absolute* position info, which may not matter for audio
- Free length generalization — 30s training → 60s inference with no adaptation
- **Risk**: this is unconventional; no evidence it works for SSAMBA specifically;
  the original SSAMBA always uses positional embeddings

**Suggested approach**: start with Option A (interpolation) since it's implemented and
validated in the original codebase. Revisit Option B only if interpolation hurts quality.

Check: does vendor/ssamba's `both_models.py` also have the interpolation logic?
It should — the HF port in exp 004 was modeled on it. Confirm before implementing.

### 2. Linear probe / evaluation protocol (UNRESOLVED)

Kaggle evaluation structure:
- Input: 60s soundscape
- Output: 12 × 5s predictions (one 206-dim probability vector per 5s window)

The linear probe needs to match this structure, not just do clip-level pooling.

**Proposed protocol**:
1. Freeze SSL backbone
2. Add per-token linear head: `Linear(D, 206)` applied to each of the 3,000/6,000 tokens
3. Train on focal recordings:
   - Run backbone on 30s focal windows → (B, 3000, D) token embeddings
   - Apply linear head → (B, 3000, 206) frame logits
   - Pool over all tokens → (B, 206) clip prediction
   - Loss: BCE against clip species label
4. Evaluate on labeled soundscapes (739 windows from exp 101):
   - Run backbone on 60s soundscapes (with pos_embed interpolation)
   - Apply linear head → (6000, 206) frame logits
   - Pool each 500-token block (5s) → (12, 206) per-window predictions
   - Metric: Kaggle-style macro AUROC on 739 labeled windows

**Open sub-questions**:
- Pre-compute and cache embeddings, or run inline during probe training?
  Pre-computing is faster for iterating on the probe but requires 30s-window inference.
- What pooling for probe training: mean over all tokens, or max?
  Mean is standard (matches SSAMBA's finetuningavgtok).
- Should probe training use the same 30s windows as SSL, or 5s chunks?
  30s stays in-distribution with SSL training. 5s would match Kaggle eval length
  but requires pos_embed adaptation or length-invariant encoding.

### 3. SSL objective: original SSAMBA has two losses (UNRESOLVED)

Original SSAMBA pretraining uses **two objectives simultaneously**:
1. **Discriminative** (D): predict whether a patch is real or swapped from another recording
2. **Generative** (G): reconstruct masked patches via MSE (like MAE)

The proposal currently assumes only the generative (MSE reconstruction) objective.
Decision needed: use both (as in original), or generative only?

- Both: stronger training signal, more stable early in training (D loss is easier)
- Generative only: simpler implementation, sufficient for downstream tasks per MAE literature
- Check vendor/ssamba/src/pretrain/ to see how the original implements both

### 4. Masking strategy (UNRESOLVED)

Original SSAMBA: mask 40% of patches randomly.
For 224×1 patches (1D temporal sequence), masking options:
- **Random frame masking**: mask 40% of time steps uniformly at random
- **Block masking**: mask contiguous blocks of frames (forces model to predict from
  further context — harder task, potentially richer representations)
- **Structured masking**: mask all tokens in a 5s window (matches evaluation granularity)

Random is simplest and matches original. Block masking is used in some MAE variants
and may be better for audio (a masked 10ms frame is trivially predictable from neighbors;
a masked 500ms block requires genuine temporal modeling).

---

## Follow-on ablations (not in scope for 103-A)

Once 103-A (single-stream, raw soundscapes) establishes a baseline:

| Run | Change | Tests |
|-----|--------|-------|
| 103-B | Input: MixIT 4-ch instead of raw mixture | Do cleaner SSL inputs help? |
| 103-C | Multi-stream K=4 with learnable freq gate | End-to-end implicit demixing |
| 103-D | SSAMBA-base (99M) | Does more capacity help with in-domain SSL? |
| 103-E | BirdCLEF 2025 soundscapes added | Does more unlabeled data help? |

See `docs/concepts/multi-stream-ssamba.md` for 103-C architecture design.
Gate each ablation on the previous result — don't run all in parallel.

---

## Token math reference

At 32kHz, hop_length=320 (10ms/frame), n_mels=224, patch=(224,1):

| Duration | Tokens | Notes |
|----------|--------|-------|
| 5s | 500 | Kaggle eval window |
| 10s | 1,000 | Original SSAMBA training length |
| 30s | 3,000 | **SSL training window (decided)** |
| 60s | 6,000 | Full soundscape / Kaggle inference |

All within SSAMBA O(n) linear regime (superlinear starts ~48k tokens, exp 005).
