# AttHead Implementation Comparison

Comparison of our SED head implementations against the competition-standard
reference from BirdCLEF 2023 5th place (Tattaka/evgeniimaslov2).

**Reference code**: `kaggle kernels pull evgeniimaslov2/birdclef-5th-place-code`

---

## Reference implementation (Tattaka, BirdCLEF 2023 5th place)

```python
class AttHead(nn.Module):
    def __init__(self, in_chans, p=0.5, num_class=264, train_period=15.0, infer_period=5.0):
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()                           # GeM on freq axis
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),                             # pre-dropout
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),                                 # post-dropout
        )
        self.attention = nn.Conv1d(512, num_class, 1)      # attention branch
        self.fix_scale = nn.Conv1d(512, num_class, 1)      # frame logits branch

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)  # (B, T, C)
        feat = self.dense_layers(feat).permute(0, 2, 1)         # (B, 512, T)

        time_att = torch.tanh(self.attention(feat))              # tanh-bounded attention

        if self.training:
            # TRAIN: use full temporal extent
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # Σ(sigmoid(logits) × softmax(tanh(att)))  →  probability space
            logits = torch.sum(
                self.fix_scale(feat) * torch.softmax(time_att, dim=-1),
                dim=-1,
            )  # Σ(logits × softmax(tanh(att)))  →  logit space
        else:
            # EVAL: crop to center infer_period
            feat_time = feat.size(-1)
            start = feat_time / 2 - feat_time * (self.infer_period / self.train_period) / 2
            end = start + feat_time * (self.infer_period / self.train_period)
            start, end = int(start), int(end)

            feat = feat[:, :, start:end]
            att = torch.softmax(time_att[:, :, start:end], dim=-1)

            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat)) * att, dim=-1,
            )
            logits = torch.sum(self.fix_scale(feat) * att, dim=-1)

        return (logits, clipwise_pred, self.fix_scale(feat).permute(0, 2, 1), time_att.permute(0, 2, 1))
```

Key design choices:
- **Returns both logits AND probabilities** — `logits` for loss, `clipwise_pred` for prediction
- **Model forward returns `output_clip`** (probability-space) as the final prediction
- **Loss**: `BCEWithLogitsLoss` applied to `logits` (logit-space weighted sum)
- **train_period=15.0, infer_period=5.0** — the multi-context mechanism

---

## Our implementations

### AttHead (exps 016–021 Phase C)

```python
# Attention: bare softmax (no tanh)
att_weights = torch.softmax(self.attention_conv(x), dim=-1)
# Clipwise: weighted sum of RAW LOGITS (not sigmoid probabilities)
clipwise_logits = (att_weights * frame_logits).sum(dim=-1)
# No train/infer period cropping
```

### AttBlockV2 (exp 017 v4c)

```python
# Attention: tanh(x) * softmax(x) — PANNs style (different from Tattaka)
att_weights = torch.tanh(att_raw) * torch.softmax(att_raw, dim=-1)
# Clipwise: weighted sum of sigmoid probabilities (matches Tattaka)
frame_probs = torch.sigmoid(frame_logits)
clipwise_probs = (att_weights * frame_probs).sum(dim=-1)
# Convert back to logits for loss
clipwise_logits = torch.logit(clipwise_probs.clamp(1e-7, 1.0 - 1e-7))
# No train/infer period cropping
```

### AttHeadMultiContext (exp 021 Phase D — NEW)

```python
# Attention: softmax(tanh(x)) — matches Tattaka
att_raw = torch.tanh(self.attention_conv(x))
att_weights = torch.softmax(att_raw, dim=-1)
# Clipwise: weighted sum of sigmoid probabilities — matches Tattaka
frame_probs = torch.sigmoid(frame_logits)
clipwise_probs = (att_weights * frame_probs).sum(dim=-1)
# Convert back to logits for loss
clipwise_logits = torch.logit(clipwise_probs.clamp(1e-7, 1.0 - 1e-7))
# Train/infer cropping: full temporal extent during training, center crop at eval
```

---

## Feature comparison matrix

| Feature | Tattaka (ref) | Our AttHead | Our AttBlockV2 | Our AttHeadMultiContext |
|---------|--------------|-------------|----------------|----------------------|
| **Attention fn** | softmax(tanh(x)) | softmax(x) | tanh(x) × softmax(x) | softmax(tanh(x)) ✓ |
| **Framewise** | sigmoid(logits) | raw logits | sigmoid(logits) | sigmoid(logits) ✓ |
| **Clipwise** | Σ(att × probs) | Σ(att × logits) | Σ(att × probs) | Σ(att × probs) ✓ |
| **Pre-dropout** | Dropout(p/2) | None | None | Dropout(p/2) ✓ |
| **Train/infer crop** | 15s→5s center | No | No | configurable ✓ |
| **Returns** | (logits, probs, frame, att) | (logits, frame) | (logits, frame) | (logits, frame) |
| **Loss target** | logits (Σ raw×att) | logits (Σ raw×att) | logit(probs) | logit(probs) |

---

## Key differences and their impact

### 1. Attention mechanism: softmax(tanh(x)) vs softmax(x) vs tanh(x)×softmax(x)

- **Tattaka / AttHeadMultiContext**: `softmax(tanh(x))` — tanh squashes raw attention
  values to [-1, 1] before softmax normalization. This prevents extreme attention
  weights from dominating when raw values have large magnitude. The output still
  sums to 1 (proper probability distribution over time).

- **Our AttHead**: `softmax(x)` — direct softmax on raw attention values. Large values
  can dominate, potentially causing "attention collapse" where one frame gets nearly
  all weight. Less stable training.

- **Our AttBlockV2 / PANNs**: `tanh(x) × softmax(x)` — element-wise product of two
  functions. This does NOT sum to 1 and can produce negative values. Different
  mathematical properties from both other approaches.

### 2. Clipwise aggregation: probability space vs logit space

- **Tattaka / AttBlockV2 / AttHeadMultiContext**: `Σ(att × sigmoid(logits))` —
  weighted average of probabilities, bounded [0, 1]. Species detected in some frames
  but absent in others get moderate probability.

- **Our AttHead**: `Σ(att × logits)` — weighted average of raw logits, unbounded
  (-∞, +∞). A high-confidence detection in one frame can dominate the sum.
  Different gradient dynamics through sigmoid in the loss.

### 3. Loss target: logits vs logit(probs)

- **Tattaka**: Returns BOTH `logits` (Σ raw×att, for loss) and `clipwise_pred`
  (Σ sigmoid×att, for prediction). Uses `BCEWithLogitsLoss` on the logit-space sum,
  but evaluates using the probability-space sum. This decouples loss computation
  from prediction — the loss benefits from logit-space numerics while prediction
  benefits from probability-space bounded aggregation.

- **Our AttHeadMultiContext / AttBlockV2**: Returns `logit(Σ sigmoid×att)` —
  computes in probability space, then converts back to logits for the loss.
  Mathematically: `logit(Σ(att × sigmoid(x)))` ≠ `Σ(att × x)`. The logit-of-sum
  is not the same as sum-of-logits. This means our loss gradient flows through
  the `logit()` → `sigmoid()` round-trip, which may introduce numerical issues
  near 0 and 1 (hence the `clamp(1e-7, 1-1e-7)`).

  **Potential concern**: If this causes training instability or suboptimal gradients,
  we could follow Tattaka's approach and return two separate outputs. However, this
  would require changes to the SEDFinetuneModule loss computation. For now, the
  logit(probs) approach is simpler and matches the existing head contract.

### 4. Multi-context train/infer period (THE KEY MECHANISM)

- **Tattaka**: `train_period=15.0, infer_period=5.0`. During training, the backbone
  processes a 15s spectrogram and the head aggregates over all temporal frames.
  During inference, the head crops to the center 1/3 of frames (corresponding
  to the center 5s) before aggregation. This means:

  - The backbone sees 3× more temporal context during both train and inference
  - The attention mechanism learns from the full 15s context
  - But predictions are anchored to the center 5s evaluation window
  - Neighboring context helps disambiguate the center segment

- **Our AttHead / AttBlockV2**: No multi-context. Single 5s windows, isolated.

- **Our AttHeadMultiContext**: Implements the same mechanism with configurable
  `train_period` and `infer_period`.

  **Important**: The center-crop at inference means the model MUST receive the
  full `train_period` duration as input at inference time. For 15s train / 5s infer:
  feed 15s mel → backbone → 15s features → head crops to center 5s → prediction.
  The Kaggle kernel must load 15s of audio centered on each 5s evaluation segment.

---

## Experimental results on our codebase

### AttHead (logit-space, no tanh, no multi-context)

Used in exp 021 Phase C (v6e-v6j). All 5s windows, dual supervision (clip 0.5 + frame 0.5).

| Config | Backbone | val/auroc | Delta vs v5j (LSEHead, 0.874) | LB |
|--------|----------|-----------|-------------------------------|-----|
| v6e | HGNetV2-B0 | 0.870 | −0.004 | pending |
| v6h | HGNetV2-B2 | 0.876 | +0.002 | pending |
| v6j | EffNet-B0 NS | 0.874 | ±0 | pending |
| v6f | HGNetV2-B0 (CE) | 0.850 | −0.024 | skip |

**AttHead regresses on B0** (−0.004). Gain on B2 (+0.002) may be from backbone scaling, not head.

### AttBlockV2 (PANNs-style, sigmoid clipwise, no multi-context)

Used in exp 017 (v4c, v4g, v4h). All on EfficientNet-B0.

| Config | Loss | val/auroc | LB |
|--------|------|-----------|-----|
| v4c | soft_ce | 0.815 | skipped |
| v4g | bce | 0.868 | 0.819 |
| v4h | asl | 0.858 | 0.800 |

**AttBlockV2 underperformed** — PANNs-style tanh×softmax didn't work well in our setup.
Note: these were on EffNet-B0 with the old recipe (before v5j improvements), so not
directly comparable.

### LSEHead (our current best)

| Config | Backbone | val/auroc | LB |
|--------|----------|-----------|-----|
| v5j | HGNetV2-B0 | 0.874 | **0.906** |
| v5j-4fold | HGNetV2-B0 ×4 | — | **0.913** |

**LSEHead outperforms AttHead** on LB in our setup. The head type alone (without
multi-context) has not been the differentiator.

---

## Competitor context (Kaggle discussion #683791)

| Person | Rank | LB | Head | Multi-context | Notes |
|--------|------|-----|------|---------------|-------|
| **hengck23** | 154th | **0.896** | LSE | **3×5s (15s)** | CNN only, no Perch, no PP |
| **Tom** | 6th | high | AttHead? | **4×5s (20s)** | "similar to hengck23" |
| **Boredom** | 9th | **0.925+** | ? | unknown | Single EfficientNet |
| **Salman** | 425th | **0.921** | AttHead | unknown | CE+sigmoid, waveform MixUp |
| **Tattaka** | 5th (2023) | — | AttHead | **3×5s (15s)** | Reference implementation |

**Key insight**: hengck23 uses **HGNetV2-B0 + LSE head** (our exact v5j setup) with
multi-context. His 0.896 → our 0.906 gap may be explained by other factors (augmentation,
training recipe), but the multi-context approach is clearly a major lever used by
multiple top competitors.

---

## Our Phase D plan (AttHeadMultiContext)

We implemented `AttHeadMultiContext` matching Tattaka's design (softmax(tanh(att)),
probability-space clipwise, train/infer period cropping, pre-dropout).

**One difference from Tattaka**: We return `logit(clipwise_probs)` for loss compatibility
instead of separate logit and probability outputs. This is a simplification that keeps
the head contract compatible with our existing `SEDFinetuneModule` without changes.
If this causes issues, we can split the returns later.

### Configs

| Config | Backbone | Context | Rationale |
|--------|----------|---------|-----------|
| v7a | HGNetV2-B0 | 3×5s (15s) | hengck23's approach on our best backbone |
| v7b | HGNetV2-B0 | 4×5s (20s) | Tom's approach (6th place) |
| v7c | EffNet-B0 NS | 3×5s (15s) | Test Boredom's claim with multi-context |

All use v5j recipe base (AmplitudeToDB, OneCycleLR, per-sample MixUp, BCE, 20 epochs),
clipwise_weight=1.0 (no framewise loss, matching v5j).

---

## Implementation verification (2026-04-04)

### What matches Tattaka exactly

1. **Attention**: `softmax(tanh(x))` ✓
2. **Probability-space clipwise**: `Σ(att × sigmoid(logits))` ✓
3. **Pre-dropout**: `Dropout(p/2)` before linear ✓
4. **Train/infer crop**: center crop at eval based on period ratio ✓
5. **GeM frequency pooling**: learnable power-mean ✓

### Deliberate simplification: loss gradient path

**Tattaka returns TWO clipwise outputs:**
- `logits` = `Σ(att × raw_frame_logits)` — used with `BCEWithLogitsLoss`
- `clipwise_pred` = `Σ(att × sigmoid(frame_logits))` — used for evaluation

**We return ONE output:**
- `logit(Σ(att × sigmoid(frame_logits)))` — serves both loss and prediction

**Why this differs mathematically:**
- Tattaka's loss: `BCEWithLogitsLoss(Σ(att × x), y)` → applies `sigmoid(Σ(att × x))`
- Our loss: `BCEWithLogitsLoss(logit(Σ(att × σ(x))), y)` → applies `σ(logit(Σ(att × σ(x))))` = `Σ(att × σ(x))`

The predictions are identical: `Σ(att × sigmoid(x))` in both cases.
But the loss gradients differ: `sigmoid(sum_of_logits)` ≠ `sum_of_sigmoids`.
By Jensen's inequality (sigmoid is concave on x>0, convex on x<0), these give
different gradient magnitudes.

**Why we accept this:** Keeping a single clipwise output preserves compatibility
with our existing `SEDFinetuneModule` (which expects `(clipwise_logits, framewise_logits)`).
Splitting into dual outputs would require changes to the training loop. If v7 models
underperform, switching to Tattaka's dual-output approach is a straightforward follow-up.

### Verified: training and inference flow

**Training (15s input):**
1. Backbone receives (B, 1, 256, ~1533) mel (15s at hop=313)
2. AttHeadMultiContext in `self.training=True` → uses full temporal extent
3. clipwise_logits computed from all ~1533 frames
4. Loss = `clipwise_weight × BCE(clipwise_logits, labels)` (framewise_weight=0.0 in v7)
5. Label corresponds to center 5s segment (soundscapes) or full recording (focal)

**Inference (15s input):**
1. Same backbone pass → (B, C, F, ~1533) features
2. AttHeadMultiContext in `self.training=False` → crops to center ~511 frames (5s)
3. clipwise_logits computed from center frames only
4. `_blended_probs` with blend_clip=1.0 → `sigmoid(clipwise_logits)` = center 5s prediction

**ONNX export:** Model in eval mode → crop logic baked into ONNX graph.
Kaggle kernel feeds 15s mel → gets center 5s prediction. Requires kernel update
for multi-context windowing (task D.8).

### Open questions

1. **Loss gradient path**: If v7 underperforms, try Tattaka's dual-output
   (separate logits for loss, probs for prediction). Requires SEDFinetuneModule change.
2. **LSEHead + multi-context**: hengck23 uses LSE head with 3×5s. Could add
   train/infer cropping to LSEHead too (crop temporal features before LSE pooling).
   This would test multi-context independently of the head architecture change.
3. **Framewise loss**: v7 uses clipwise_weight=1.0 (matching v5j). Tattaka uses
   dual supervision — could try 0.5/0.5 as a follow-up ablation.
4. **Batch size**: v7a uses batch_size=32 (halved from 64) for 3× wider spectrograms.
   May need reduction if OOM on RTX 6000 (24GB).
