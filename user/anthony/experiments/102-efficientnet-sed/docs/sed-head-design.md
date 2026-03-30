# Design: SED Head with Dual-Branch Frame Predictions

## Context

The current exp 102 attention pooling produces clip-level predictions. Murilo's
SED head (exp 016-017) added per-species, per-frame predictions and achieved
+0.060 LB over clip-level. We want to port this idea but also connect it to
the teacher labels and activity scores we already have.

The key insight: a frame-level prediction grid (species × time) gives us a
natural place to inject frame-level supervision from teacher models and
activity scores, not just clip-level labels.

## Current architecture (exp 102)

```
backbone features (B, 1280, H, W)
    → mean over frequency → (B, T, 1280)
    → shared attention (tanh + sigmoid) → weights (B, T, 1)
    → weighted sum → (B, 1280) → Linear → clip logits (B, 234)
```

One attention map shared across all species. No per-frame output.

## Proposed architecture

```
backbone features (B, 1280, F, T)
    → GeM pool frequency → (B, 1280, T)
    → Linear(1280, 512) + ReLU + Dropout → (B, 512, T)
    → Conv1d(512, 234) → frame_logits (B, 234, T)
                          ↓
              ┌───────────┴───────────┐
              │                       │
        softmax(dim=T)           max(dim=T)
              ↓                       ↓
    attention weights (B, 234, T)   framewise_max (B, 234)
              ↓
    sum(weights * frame_logits)
              ↓
        clipwise (B, 234)
              ↓
    blend: 0.75 * clipwise + 0.25 * framewise_max
```

### Single branch vs dual branch

Murilo uses two parallel Conv1d layers: one for frame_logits, one for attention
weights. We propose a **single Conv1d** instead — the attention weights are
just `softmax(frame_logits, dim=time)`. Rationale:

- Simpler (one branch, fewer params)
- The attention is derived from the predictions: "the frame where I'm most
  confident is also the frame that matters most"
- The dual branch was never ablated — we don't know if separate attention helps
- If we want different behavior, the 75/25 clip/max blend already handles
  the "brief call" case via framewise max

If this underperforms, adding a second branch is trivial (one more Conv1d).

### Key components

**GeM (Generalized Mean Pooling)**: learnable power-mean over frequency.
At p=1 it's average, at p→∞ it's max. Initialized at p=3. Lets the model
emphasize loud frequency bins (where calls are) without ignoring the rest.
Collapses (B, 1280, F, T) → (B, 1280, T).

**Projection bottleneck**: Linear(1280→512) reduces the per-frame feature
dimension before the species-level Conv1d. Cuts parameters and adds a
nonlinearity + dropout between backbone and classification.

**Frame logits grid**: Conv1d(512→234) produces a (234, T) grid — per-species
confidence at each time frame. This is the core output that enables both
clip-level aggregation and frame-level supervision.

## Training losses

### Loss 1: Clip-level (same as now)

```
clip_pred = softmax_attention_blend + framewise_max_blend
clip_loss = BCE(clip_pred, hard_clip_labels)
```

Works with the existing focal data (clip-level species labels).

### Loss 2: Framewise max (Murilo's dual supervision)

```
frame_max = frame_logits.max(dim=T)
frame_loss = BCE(frame_max, hard_clip_labels)
```

If the species is in the clip, at least one frame should predict it. This
pushes the model toward temporal localization even with clip-level labels.

### Loss 3 (optional): Frame-level teacher supervision

This is where the dual-branch design pays off for our pseudo-labeling plans.

We already have / can generate:
- **Activity scores** (244K chunks): per-chunk spectral contrast indicating
  call presence. Currently species-agnostic but tells us *when* something
  is calling.
- **Teacher frame predictions**: the current attention model produces
  per-frame (250ms), per-species probabilities. These can supervise the
  new frame_logits directly.
- **Pseudo-labeled soundscapes**: when we run teacher inference on the
  127K unlabeled windows, we get frame-level predictions that can train
  the frame_logits branch, not just clip-level.

```
# Teacher soft labels at frame level (250ms resolution, resampled to match T)
frame_teacher_loss = BCE(frame_logits, teacher_frame_probs)

# Combined
total_loss = clip_weight * clip_loss
           + frame_weight * frame_loss
           + teacher_weight * frame_teacher_loss
```

The frame_logits grid is the natural target for teacher distillation — you're
supervising *where* and *what* simultaneously, not just *what's in the clip*.

### Why this helps pseudo-labeling

The pseudo-labeling design doc (pseudo-labeling-design.md) identified a key
question: "Frame-level vs clip-level? Frame-level requires changing the loss."

The SED head answers that question. With frame_logits as a first-class output:

1. **Teacher predictions map directly to frame_logits** — same shape, same
   semantics. No aggregation needed.
2. **Activity scores can weight frame-level loss** — frames with high spectral
   contrast get more supervision weight, frames in silence get less.
3. **Soft labels work better at frame level** — the exp 014 failure ("soft
   labels hurt") was clip-level, where a 0.02 background signal pollutes
   the whole clip. At frame level, a 0.02 signal at one frame is localized
   noise, not a clip-wide error.
4. **Iterative improvement**: train SED → run on soundscapes → get frame-level
   pseudo-labels → retrain with frame supervision → better frame predictions →
   repeat.

## Implementation plan

### Phase 1: SED head (architecture only)

1. Add `GeM` module (5 lines)
2. Add `SEDHead` module: GeM + projection + single Conv1d + clip/frame output
3. Wire into existing model.py as a new `pooling_mode="sed"`
4. Add `framewise_max` loss term to training.py
5. Config: `blend_clip`, `blend_frame`, `clipwise_weight`, `framewise_weight`

No data changes. Train with existing focal clip labels + dual supervision.
This should get us close to Murilo's +0.060 LB gain.

### Phase 2: Mel improvements (config only)

6. Switch to `norm_type="amplitude_to_db"` in mel transform
7. Change `n_fft=4096`, `n_mels=224`
8. Consider `window_duration=10` (2x context, matches Murilo)

### Phase 3: Frame-level teacher supervision

9. Run trained SED model on focal data → save frame predictions
10. Run on unlabeled soundscapes → save frame pseudo-labels
11. Add frame-level teacher loss with activity score weighting
12. Iterative refinement (1-2 rounds)

## What we're NOT doing

- AttBlockV2 (tanh×softmax variant) — Murilo tested it (v4c), no improvement
- Dual Conv1d branches — start with single, add if needed
- LSEHead — Murilo tested it (v5d), no improvement
- MixUp — regressed on LB twice in Murilo's experiments
- Stacking techniques — Murilo showed they don't combine additively

## Connection to prompted SED

This SED head is a stepping stone, not a competitor to the prompted SSAMBA
concept. The frame_logits grid (234, T) is a fixed-head classifier. Prompted
SED outputs (1, T) conditioned by a reference clip. They solve different
problems and can coexist:

- SED head → stronger teacher for pseudo-labeling → better training signal
  for prompted SED (Stage 2/3 needs teacher labels)
- SED head → baseline to compare prompted SED against
- GeM + projection patterns are reusable in either architecture
- Frame-level teacher pipeline built here transfers directly to prompted
  SED episode construction

## Success criteria

- Phase 1: Kaggle LB > 0.794 (current best) from SED head alone
- Phase 2: LB > 0.83 with mel improvements
- Phase 3: LB > 0.85 with frame-level teacher supervision
