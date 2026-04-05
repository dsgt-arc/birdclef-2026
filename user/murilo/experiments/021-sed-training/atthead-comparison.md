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

## Two different multi-context approaches

### Tattaka (BirdCLEF 2023 5th) — AttHead + center crop

```
15s audio → mel (256 × 1533) → backbone → features (B, C, F', T')
  → AttHead processes ALL T' frames during training
  → At inference: CROP to center T'/3 frames, then aggregate
  → 1 prediction per 15s window (for the center 5s)
```

- Head aggregates attention over the full (or cropped) temporal extent
- One prediction per forward pass
- The head architecture IS the multi-context mechanism (train_period/infer_period)

### hengck23 (BirdCLEF 2026, 0.896 LB) — LSEHead + temporal split

```
15s audio → mel (256 × 1533) → backbone → features (B, C, F', T')
  → freq pool → (B, C, T')
  → Linear → framewise_logits (B, T', 234)
  → SPLIT T' into 3 chunks of T'/3
  → LSEPool each chunk independently → 3 predictions
  → All 3 supervised during training
```

Source: `notes/figures/hengck23-cnn-birdclef.png` (Kaggle discussion #683791)

- Head operates on **5s chunks** (same as normal LSEHead)
- **3 predictions** per forward pass, all supervised
- The multi-context comes from the **backbone** — convolutional receptive fields
  span across chunk boundaries, so features for the center chunk are informed by
  flanking chunks
- "conv does have location information" — positional encoding is implicit in CNNs
- **No new head architecture needed** — just split temporal features before existing LSEHead

Key numbers from hengck23's image:
- 5sec baseline: LB 0.876 (4-fold, no PP)
- 3×5sec: **LB 0.896** (4-fold, no PP) → **+0.020 gain**
- Same data as us: 35k focal clips + 66 labeled soundscapes

---

## Phase D results: AttHeadMultiContext (Tattaka approach) — FAILED

We implemented Tattaka's approach (AttHeadMultiContext). All models massively regressed:

| Config | Backbone | Context | val/auroc | Delta vs v5j (0.874) |
|--------|----------|---------|-----------|----------------------|
| v7c | EffNet-B0 NS | 3×5s (15s) | 0.851 | −0.023 |
| v7a | HGNetV2-B0 | 3×5s (15s) | 0.820 | −0.054 |
| v7b | HGNetV2-B0 | 4×5s (20s) | 0.799 | −0.075 |

**Root cause**: Changed too many variables from v5j (head type + window length + batch
size + loss gradient path). AttHead already regresses vs LSEHead (Phase C: −0.013 to
−0.017 LB). Adding 15s aggregation on top compounded the regression.

The fundamental issue: Tattaka's approach aggregates attention over ALL 15s into one
prediction. hengck23's approach keeps 5s-level LSE pooling per chunk — the backbone
provides context through its receptive field, not through the head.

---

## Phase E plan: LSEHead + temporal split (hengck23 approach)

The correct implementation based on the image:

### Architecture

```
15s audio → mel → backbone → features (B, C, F', T')
  → mean freq pool → (B, C, T')          [same as v5j LSEHead]
  → Linear+ReLU+Dropout+Linear → (B, T', 234)  [same as v5j LSEHead]
  → Split T' into N chunks
  → LSEPool each chunk → N × (B, 234)    [same pooling as v5j]
  → All N supervised with their respective labels
```

### What changes from v5j

Only two things:
1. **window_duration**: 5s → 15s (dataset already supports this)
2. **Forward pass**: split temporal features into N chunks, apply LSEPool per-chunk,
   return N predictions instead of 1

Everything else stays identical: LSEHead, BCE loss, OneCycleLR, MixUp, batch params.

### What stays the same

- Head architecture (LSEHead with mean freq pool)
- Loss function (BCE)
- Training recipe (OneCycleLR, per-sample MixUp)
- All hyperparameters

### Implementation

Modify the `SEDFinetuneModule` forward + loss to handle N chunks:

```python
# In forward():
features = self.encoder.forward_features(mel)     # (B, C, F', T')
clipwise, framewise = self.sed_head(features)      # normal LSEHead on full T'

# For multi-context: split framewise before pooling
# Option A: modify LSEHead to return pre-pooled framewise + pool per chunk
# Option B: wrapper that calls LSEHead N times on sub-features
```

**Option A** is cleaner — add a `n_chunks` param to LSEHead:
```python
class LSEHead:
    def __init__(self, ..., n_chunks=1):
        self.n_chunks = n_chunks

    def forward(self, features):
        h = freq_pool(features)           # (B, C, T')
        h = h.permute(0, 2, 1)           # (B, T', C)
        framewise = self.cls_fc(h)        # (B, T', 234)

        if self.n_chunks == 1:
            clipwise = self.lse_pool(framewise)
            return clipwise, framewise.permute(0, 2, 1)

        # Split into chunks, pool each
        chunks = framewise.chunk(self.n_chunks, dim=1)  # N × (B, T'/N, 234)
        clipwise_list = [self.lse_pool(c) for c in chunks]  # N × (B, 234)
        clipwise = torch.stack(clipwise_list, dim=1)    # (B, N, 234)
        return clipwise, framewise.permute(0, 2, 1)
```

The loss needs to handle (B, N, 234) clipwise — sum BCE over all N chunks.

### Labels for N chunks

- **Focal recordings**: same label for all N chunks (whole file is one species)
- **Soundscape segments**: each 5s has its own label from the annotations.
  Currently we load one 15s window centered on one labeled 5s segment.
  The flanking 5s may have different labels — we need per-chunk labels.

For the initial implementation, using the **same label for all 3 chunks** is simplest
(correct for focal, approximate for soundscapes). Can refine later with per-chunk
soundscape labels.

### Open questions

1. **Batch size**: 15s spectrograms are 3× wider. May need batch_size=32 or
   gradient accumulation to compensate.
2. **Per-chunk soundscape labels**: initial implementation uses same label for all
   chunks. Could improve by loading per-5s labels from soundscape annotations.
3. **Wave mix augmentation**: hengck23 mentions "specially designed for 3×5sec."
   May mean mixing different species into different chunks within the 15s window.
4. **N=3 vs N=4**: hengck23 uses 3, Tom uses 4. Start with 3, ablate later.
