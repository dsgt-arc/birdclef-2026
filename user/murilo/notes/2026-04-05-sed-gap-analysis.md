# SED Pipeline Gap Analysis: Why We're Stuck at 0.906

*2026-04-05*

Systematic comparison of our best SED pipeline (v5j, 0.906 LB) against
competitor approaches that achieve 0.920+ with the same data and no Perch.
Goal: identify what's actually different vs what we've been assuming.

## Sources

- **Salman Ahmed** (0.921 solo, 0.923 w/ priors) — Kaggle discussion #683791
- **Nikita Babich** (1st place BirdCLEF 2025) — `kaggle kernels pull nikitababich/birdclef2025-1st-place-inference`
- **EliKal** (0.909, EffNet-B0) — Kaggle discussion #683791
- **BirdCLEF 2024 1st place** (Team Kefir) — `docs/references/kaggle/birdclef-2024/`
- **Our exp 021 results** — `experiments/021-sed-training/`
- **Nikita reference code**: `docs/references/kaggle/birdclef-2025/birdclef-2025-nikita-1st-place-sed.py`

## The Scoreboard (single model, same data, no Perch KD)

| Who | LB | Backbone | Head | Loss | MixUp | Inference |
|-----|-----|----------|------|------|-------|-----------|
| **Boredom** (7th) | **0.922+** | EfficientNet | ? | ? | ? | ? |
| **Salman** (448th) | **0.921** | non-EffNet | AttHead | CE→sigmoid | waveform | framewise max |
| **EliKal** (464th) | **0.909** | EffNet-B0 | ? | KLD distill | spec | time-shift TTA |
| **v5j** (ours) | **0.906** | HGNetV2-B0 | LSEHead | BCE | spec | clipwise only |
| hengck23 (175th) | 0.896 | HGNetV2-B0 | LSEHead | ? | wave? | 3×5s chunks |

---

## GAP 1: Framewise Max Inference (CRITICAL)

### What they do

**Nikita (1st place 2025)** — The smoking gun is in the inference code. His
AttHead has the attention mechanism **completely commented out**:

```python
# In AttHead.forward() — INFERENCE VERSION:
# time_att_logit = torch.tanh(self.attention(feat))     # COMMENTED OUT
# clipwise_prob = torch.sum(                             # COMMENTED OUT
#     segmentwise_prob * softmax(time_att_logit), dim=-1)

output = {"framewise_logit": framewise_logit}  # ONLY framewise returned
```

At inference, he uses ONLY `framewise_logit.sigmoid()` → overlap reconstruction
→ max per 5s segment. No attention weighting. No clipwise aggregation.

But the **training** code (in `forward()`) uses dual supervision:
```python
loss = 0.5 * criterion(logit(clipwise_prob), labels)      # attention-weighted
     + 0.5 * criterion(segmentwise_logit.max(2)[0], labels)  # framewise max
```

**Training with attention, inferring without.** The attention head is a
training-time regularizer that improves per-frame feature quality, but the
actual predictions come from framewise max.

**Salman** explicitly states: "SED simple Attn head's framewise max is better
than LSE head." He uses CE loss on both clipwise and framewise max during
training, sigmoid at inference.

### What we do

Our v5j uses clipwise-only: `blend_clip=1.0, blend_frame=0.0`.
Our AttHead configs (v6e-v6j) used `blend_clip=0.75, blend_frame=0.25`.

**We have never tested `blend_clip=0.0, blend_frame=1.0` (framewise-max only).**

### Why this matters

LSE pooling is a smooth-max over time — it captures the strongest frame but
also averages in nearby frames, potentially diluting short calls. Framewise max
is a HARD max: the single strongest frame per species. For BirdCLEF where many
calls are brief (0.1-0.5s = 1-5 frames), hard max may preserve more signal.

The 1st place approach suggests that:
1. Attention helps learn better **per-frame features** during training
2. But at inference, the best prediction strategy is simply "what's the highest
   frame score?" — no learned aggregation, no smoothing
3. The attention mechanism can even be **removed entirely** at inference time

### Action

Test framewise-max-only inference on our existing v5j checkpoint. This requires
NO retraining — just change the ONNX export to use `framewise_logits.max(dim=-1)`
instead of clipwise. Can test on LB immediately.

---

## GAP 2: 3-Channel Input (SIGNIFICANT)

### What they do

**Nikita (1st place 2025)**:
```python
x = torch.stack([x, x, x], 1)  # mono mel → 3-channel "RGB"
```

Replicates the single mel channel 3 times to match ImageNet-pretrained backbone
stem expectations. The backbone's first conv layer was pretrained on 3-channel
RGB — feeding 3 identical channels preserves the pretrained weights exactly.

### What we do

Our v5j averages the 3-channel stem weights into 1-channel:
```python
# In timm backbone init:
# in_chans=1 → stem weights averaged from pretrained 3-channel
```

This is timm's default `in_chans=1` handling. It averages the R/G/B conv weights
into a single filter, which changes the filter norms and potentially loses
channel-specific patterns learned during ImageNet pretraining.

### Why this matters

ImageNet-pretrained stems learn different filters per channel (edge detectors,
color gradients, etc.). With 3-channel input, all three filter sets process
the same mel spectrogram — tripling the feature extraction capacity of the
first layer. With 1-channel averaged weights, that capacity is reduced.

EliKal's 0.909 with EffNet-B0 (we get 0.855 with EffNet-B0) may partly be
explained by 3-channel input preserving EffNet's pretrained stem better.

### Action

Test 3-channel input with our existing backbones. Simple change in the model
wrapper: `mel.repeat(1, 3, 1, 1)` before feeding to backbone. Requires
reinitializing with `in_chans=3` (default pretrained weights) instead of
`in_chans=1` (averaged weights). Requires retraining.

---

## GAP 3: Overlap Reconstruction + Delta Blending (SIGNIFICANT)

### What they do

**Nikita (1st place 2025)** — Processes 20s windows with 5s step (75% overlap):

```
Window 1: |---20s---|
Window 2:      |---20s---|
Window 3:           |---20s---|
...
```

Each position is covered by up to 4 windows. Then:
1. Average framewise predictions where windows overlap
2. Max within each 5s segment
3. Delta blending: `0.5 * center + 0.25 * left_shifted + 0.25 * right_shifted`

This is test-time augmentation via temporal shifts, without multiple forward passes.
The delta blending (±2 frames) smooths predictions at segment boundaries.

### What we do

Our v5j processes twelve independent 5s segments, no overlap, no blending.

### Why this matters

Each 5s prediction benefits from seeing the exact same audio through slightly
different temporal alignments. The overlap averaging reduces noise, and the
delta blending provides temporal TTA. EliKal also mentions "time-shift TTA"
as his only inference trick.

### Action

This is an inference-only change. Can be added to the Kaggle kernel without
retraining. Requires loading 15-20s windows with 5s step instead of
independent 5s segments.

---

## GAP 4: CE Loss + Sigmoid Inference (MEDIUM)

### What they do

**Salman**: CE during training (softmax over 234 species), sigmoid at inference.
**BirdCLEF 2024 1st place**: Same approach — "CE was significantly better" than BCE.

The logic: with 234 classes and usually 1 present per focal recording, CE
provides sharper gradients. BCE independently pushes down 233 negatives; CE
competes them against each other via softmax. At inference, switch to sigmoid
for multi-label capability.

### What we tested

- v6a (CE + LSEHead): −0.020 offline
- v6f (CE + our non-standard AttHead): −0.024 offline

### Why our tests may have been invalid

1. **CE + LSEHead is untested by competitors.** All successful CE users pair
   it with AttHead. LSE pooling may not produce the right gradient signal for CE.

2. **Our AttHead has bugs** (see Gap 5). CE + broken AttHead ≠ CE + proper AttHead.

3. **We never tested CE + framewise max.** Salman uses CE on both clipwise AND
   `framewise.max(dim=time)`. The framewise max term is what matters at inference.

### Action

Retest CE loss ONLY with a properly implemented AttHead (see Gap 5), using
dual supervision (clipwise + framewise max). This is a retraining experiment.

---

## GAP 5: Our AttHead Implementation Is Wrong (MEDIUM)

### What they do

**Nikita (1st place 2025)** — his AttHead is nearly identical to Tattaka's:
- `nn.Dropout(p / 2)` before Linear (pre-dropout)
- GeMFreq frequency pooling (learnable power)
- Conv1d for framewise logits
- Conv1d for attention (commented out at inference but used during training)

**Tattaka (5th place 2023)**:
- Attention: `softmax(tanh(x))` — tanh bounds values before softmax
- Clipwise: `sum(att * sigmoid(logits))` — probability-space aggregation
- Returns BOTH logits (for loss) and probs (for prediction) separately

### What we do

Our AttHead (used in v6e-v6j):
- **No pre-dropout** — missing the `Dropout(p/2)` before Linear
- Attention: `softmax(x)` — **no tanh**, raw values can be unbounded
- Clipwise: `sum(att * logits)` — **logit-space aggregation**, unbounded
- Returns only logits — no separate probability-space output

### Impact

The missing tanh allows attention collapse (one frame dominates). Logit-space
aggregation has different gradient dynamics than probability-space. These are
not minor details — they change how the head learns temporal patterns.

Our Phase C conclusion ("AttHead regresses on LB") was based on testing a
non-standard head. We cannot conclude anything about whether a **proper**
AttHead would help.

### Action

Implement competition-standard AttHead matching Nikita/Tattaka exactly.
Or, simpler: keep LSEHead for aggregation but add the attention-trained
framewise logits for loss supervision (training only).

---

## GAP 6: Per-Sample Mel Normalization (MINOR)

### What they do

**Nikita (1st place 2025)** — per-sample standardize + min-max:
```python
mean = X.mean((1, 2), keepdim=True)
std = X.std((1, 2), keepdim=True)
Xstd = (X - mean) / (std + eps)
norm_min, norm_max = Xstd.amin/amax
return (Xstd - norm_min) / (norm_max - norm_min)
```

Each spectrogram is independently normalized to [0, 1].

### What we do

Global dataset statistics: `AmplitudeToDB → (x - global_mean) / global_std`.
No per-sample rescaling.

### Why this matters

Per-sample normalization ensures consistent scale regardless of recording
volume. Our global normalization depends on dataset statistics being
representative of test data (different recording conditions, sites, equipment).
The +0.014 LB gain we got from adding AmplitudeToDB (over ttahara's min-max)
may have been capturing some of this benefit already.

### Action

Low priority. Our normalization already works well (0.906 LB). But worth an
ablation if other changes plateau.

---

## GAP 7: Training Duration / Fewer Epochs (MINOR)

### What they do

- **Salman**: 25 epochs
- **EliKal**: 10 + 12 = 22 epochs total
- **BirdCLEF 2024 1st**: 7-12 epochs
- **Nikita**: 15-40 epochs depending on config

Salman notes: "more epochs make the model too confident on predictions and
that hurt the AUC." Confidence calibration matters for AUROC.

### What we do

20 epochs (v5j). Tested 25 (v6d): flat AUROC, better cmap.

### Action

Not a major gap. Our 20 epochs is in the right range.

---

## Priority-Ordered Action Plan

### Tier 1: No retraining needed (inference-only)

1. **Framewise-max inference on v5j** — change ONNX export to use
   `sigmoid(framewise_logits.max(dim=-1))` instead of `sigmoid(clipwise_logits)`.
   Test on LB immediately. This is the single most likely win because Nikita
   (1st place) and Salman (0.921) both use it.

2. **Overlap reconstruction** — modify Kaggle kernel to process overlapping
   windows and reconstruct with delta blending. Can stack with #1.

### Tier 2: Requires retraining (one variable at a time)

3. **3-channel input** — `mel.repeat(1, 3, 1, 1)` with `in_chans=3` backbone.
   Preserves pretrained stem weights exactly. Test with v5j recipe otherwise
   unchanged.

4. **Proper AttHead** — implement competition-standard (Nikita/Tattaka) head
   with pre-dropout, tanh attention, probability-space aggregation. Train with
   dual supervision (clipwise + framewise max). Infer with framewise max only.

5. **CE loss + proper AttHead + framewise max** — the full Salman recipe.
   Only try after #4 validates that proper AttHead works.

### Tier 3: Speculative

6. Per-sample mel normalization
7. Knowledge distillation (EliKal's approach — requires a teacher model)

---

## Key Lesson

We've been ablating single variables from v5j, but our ablations tested
**broken implementations** of competitor techniques:
- CE was tested with LSEHead (wrong head) and our non-standard AttHead (buggy)
- AttHead was missing pre-dropout, tanh, and probability-space aggregation
- Framewise max inference was never tested
- 3-channel input was never tested

The correct approach is to implement the **full competitor recipe** faithfully
before concluding that specific ingredients don't work.
