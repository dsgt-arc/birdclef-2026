# Design: SSAMBA SSL Pretraining

**Created**: 2026-03-30
**Updated**: 2026-03-30
**Status**: architecture finalized for 103-A

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

### Mel normalization: AmplitudeToDB (default), PCEN (deferred ablation)
- NOT AudioSet global stats (`(mel - mean) / (std * 2)`) — those were computed for
  16kHz/128-mel AudioSet; wrong distribution for 32kHz/224-mel BirdCLEF audio
- **AmplitudeToDB** — default for 103-A:
  1. `10 * log10(mel.clamp(1e-10))` — convert to dB
  2. `clamp(mel, min=mel_max - 80)` — 80dB dynamic range, cuts noise floor
  3. Per-clip standardize: `(mel - mean) / std`
  4. Min-max to [0, 1]: `(mel - min) / (max - min)`
  - Proven in exp 009 SSL pipeline, bounded [0,1] output, numerically stable
- **PCEN (Per-Channel Energy Normalization)** — available as `mel_norm: pcen`, deferred:
  - Adaptive per-band AGC, suppresses noise, amplifies quiet calls
  - Learnable parameters (alpha, delta, r, s) per mel band
  - Issues found during implementation: unbounded output breaks raw NCE dot products,
    IIR filter requires sequential loop (3000 steps), autograd workaround needed
  - Evaluate as ablation once AmplitudeToDB baseline converges
- Config flag `mel_norm` switches between them

### Masking strategy: span masking (40%)
- Mask 40% of tokens using contiguous spans, not random
- Span length ~ Uniform[5, 15] tokens (50ms–150ms)
- **Why spans over random**: with 224×1 patches (10ms/token), adjacent frames are
  nearly identical — random masking is trivially solvable from immediate neighbors.
  Contiguous spans force reconstruction from longer-range temporal context.
- Adapted from exp 009 which validated span masking for Mamba SSL

### SSL objective: dual loss (NCE + MSE)
- Use both discriminative (NCE) and generative (MSE) losses, as in original SSAMBA
- Combined: `L = L_nce + 10 × L_mse` (MSE weighted 10× stronger)
- NCE provides contrastive signal (is this the right patch?), MSE provides
  reconstruction signal (what does the patch look like?)
- Per-patch target normalization: subtract patch mean, divide by patch std
  (forces model to predict spectral shape, not loudness)

### Backbone code: vendor/ssamba (original CUDA implementation)
- NOT the HF port from exp 004 — pure PyTorch SSM is too slow for training
- vendor/ssamba requires mamba-ssm CUDA + causal-conv1d
- Exp 009 has existing SSL code to adapt: `birdclef_009/models/pretrain.py`

---

## Full Architecture (103-A)

```
                        30s audio (32kHz, 960K samples)
                                    │
                        MelSpectrogram(n_fft=4096, hop=320,
                          n_mels=224, fmin=50, fmax=16000)
                                    │
                        AmplitudeToDB (80dB clamp → standardize → [0,1])
                        or PCEN (configurable, deferred ablation)
                                    │
                              (B, 1, 224, 3000)
                                    │
                        Conv2d(1, 384, k=(224,1), s=(224,1))
                           patch embed: 1 token/10ms
                                    │
                              (B, 3000, 384)
                                    │
                      ┌─────────────┴─────────────┐
                      │     span mask 40%          │
                      │  spans of 5–15 tokens      │
                      │  (50ms–150ms each)          │
                      │  → 1200 masked tokens       │
                      │  replaced with mask_embed   │
                      └─────────────┬─────────────┘
                                    │
                      + CLS token at position 1500 (middle)
                      + learned pos_embed (1, 3001, 384)
                                    │
                              (B, 3001, 384)
                                    │
                      ┌─────────────┴─────────────┐
                      │   24 BiMamba blocks        │
                      │   (12 fwd/bwd pairs)       │
                      │   embed_dim=384             │
                      │   d_state=16, d_conv=4      │
                      │   expand=2 (d_inner=768)    │
                      │   RMSNorm, residual in fp32 │
                      └─────────────┬─────────────┘
                                    │
                              RMSNorm(384)
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
          masked positions                    masked positions
            (B, 1200, 384)                     (B, 1200, 384)
                  │                                   │
          MLP(384→384→224)                    MLP(384→384→224)
                  │                                   │
          MSE vs target                        NCE vs target
          (generative)                       (discriminative)
                  │                                   │
                  └────────────┬──────────────────────┘
                               │
                      L = L_nce + 10 × L_mse
```

### SSAMBA-small parameters

| Component | Details |
|-----------|---------|
| embed_dim | 384 |
| depth | 24 layers (12 bidirectional pairs) |
| d_state | 16 (SSM state dimension) |
| d_conv | 4 (conv kernel in SSM) |
| expand | 2 (d_inner = 768) |
| CLS token | 1, inserted at middle of sequence |
| Positional embed | learned, (1, 3001, 384), trunc_normal(0.02) |
| Mask embed | learned, (1, 1, 384), xavier_normal |
| Total params | ~26M |

### Training config

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, lr=1.5e-4, weight_decay=0.05, betas=(0.95, 0.999) |
| Schedule | Cosine anneal + 2-epoch linear warmup |
| Max epochs | 15 (early stop patience=5 on val loss) |
| Batch size | 8 (eff 64 via accumulate_grad_batches=8) |
| Precision | fp16 (RTX 6000, sm_75) |
| torch.compile | Yes |
| Windows | 30s — soundscapes split into 2 halves, focal padded |

### Data

| Source | Windows |
|--------|---------|
| BirdCLEF 2026 soundscapes (10,658 × 2 halves) | 21,316 |
| BirdCLEF 2026 focal (35,549, padded to 30s) | 35,549 |
| **Total** | **~56,865** |
| Val holdout (5% soundscapes by site) | ~1,066 |

---

## Open questions (deferred — not blocking 103-A)

### 1. Positional embedding at inference time (Phase 2)

SSL trains on 30s windows (3,000 tokens, pos_embed shape (1, 3001, D)).
Kaggle evaluation requires 60s soundscape inference (6,000 tokens).

**Option A — Bilinear interpolation** (default, implemented in both vendor/ssamba and
exp 004 HF port). Standard approach, resolve when we get to Phase 2.

**Option B — No positional embeddings** (speculative). Mamba recurrence may encode
temporal order implicitly. Test only if interpolation hurts quality.

### 2. Linear probe / evaluation protocol (Phase 2)

Deferred until SSL training converges. Key decisions:
- Per-token linear head pooled over 5s blocks to match Kaggle format
- Pre-compute embeddings vs inline inference
- Mean vs max pooling
- 30s vs 5s probe training windows

---

## Follow-on ablations (not in scope for 103-A)

Once 103-A (single-stream, raw soundscapes) establishes a baseline:

| Run | Change | Tests |
|-----|--------|-------|
| 103-B | mel_norm: pcen (PCEN normalization) | Does adaptive AGC help representations? Needs cosine NCE (temperature=0.1) due to unbounded output. |
| 103-C | Input: MixIT 4-ch instead of raw mixture | Do cleaner SSL inputs help? |
| 103-D | Multi-stream K=4 with learnable freq gate | End-to-end implicit demixing |
| 103-E | SSAMBA-base (99M) | Does more capacity help with in-domain SSL? |
| 103-F | BirdCLEF 2025 soundscapes added | Does more unlabeled data help? |

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
