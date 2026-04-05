# Multi-Context Data Flow (Phase E, v8a)

*2026-04-04*

Step-by-step trace of the full training data flow for v8a (HGNetV2-B0 + LSEHead +
3×5s multi-context, n_chunks=3). Used for implementation verification and reference.

See also:
- [SED architecture overview](../../notes/sed-architecture-overview.md)
- [AttHead comparison and multi-context analysis](atthead-comparison.md)
- [hengck23 image](../../notes/figures/hengck23-cnn-birdclef.png)

---

## 1. Dataset loads 15s audio

**Focal recordings** (96% of training data):
```
SEDFocalDataset(window_duration=15.0):
  → Random crop 15s from a focal recording (480,000 samples at 32kHz)
  → Label: file-level species label (same for entire recording)
  → Returns: (waveform[480000], label[234], primary_idx)
```

70% of focal recordings are ≥15s (median 23s), so most crops are fully filled.
30% are shorter and get zero-padded.

**Soundscape segments** (4% of training data):
```
SEDSoundscapeDataset(context_duration=15.0):
  → Load 15s centered on one labeled 5s segment
    (5s before + 5s target + 5s after, zero-padded at file boundaries)
  → Label: the CENTER 5s segment's multi-species label
  → Returns: (waveform[480000], label[234], primary_idx)
```

Note: flanking 5s chunks may contain different species than the center label.
This is an approximation — only ~4% of training data, and hengck23 has the
same limitation.

---

## 2. Mel spectrogram on GPU

```
waveform (B, 480000)
  → NNAudioMelSpectrogram(n_fft=2048, hop_length=313, n_mels=256, AmplitudeToDB)
  → mel (B, 1, 256, 1533)
              ↑         ↑
          256 mels   1533 time frames for 15s
```

For comparison:
- v5j with 5s: (B, 1, 256, 511) — 511 frames
- v8a with 15s: (B, 1, 256, 1533) — 1533 frames (3× wider)
- v8b with 20s: (B, 1, 256, 2044) — 2044 frames (4× wider)

---

## 3. Spectrogram MixUp (if active)

```
mel_mix = lambda * mel_i + (1 - lambda) * mel_j      (B, 1, 256, 1533)
label_mix = lambda * label_i + (1 - lambda) * label_j  (B, 234)
```

lambda ~ Beta(1.0, 1.0) = Uniform[0,1], per-sample (each sample in batch
gets its own lambda). MixUp mixes the full 15s spectrograms — both samples
contribute across the entire temporal extent. The blended label applies to
all chunks.

MixUp is disabled for the first 5 epochs (warmup) and applied with 80%
probability per sample after that (theta=0.8).

---

## 4. Backbone forward pass

```
mel (B, 1, 256, 1533)
  → HGNetV2-B0 (4 stages of convolution + downsampling)
  → features (B, 1024, 8, 48)
               ↑      ↑   ↑
           channels   F'  T'=48 temporal positions
```

Temporal reduction: 1533 frames → 48 positions (~32× reduction).
Each temporal position corresponds to ~0.3s of audio.
Each position's receptive field spans multiple neighbors.

**This is where the multi-context magic happens**: position 16 (start of
center chunk) is influenced by positions ~10-22 through the convolutional
receptive field. Features for the center 5s are informed by the flanking
5s on each side, even though they'll be split into separate chunks next.

For comparison:
- v5j with 5s input: T'=16 temporal positions
- v8a with 15s input: T'=48 temporal positions (3×16)
- v8b with 20s input: T'=64 temporal positions (4×16)

---

## 5. LSEHead forward (n_chunks=3)

### Step 5a: Frequency pooling
```python
h = features.mean(dim=2)     # (B, 1024, 8, 48) → (B, 1024, 48)
h = h.permute(0, 2, 1)       # (B, 48, 1024)
```

### Step 5b: Per-frame classification
```python
framewise_logits = cls_fc(h)  # (B, 48, 234)
# Each of the 48 temporal positions gets a 234-species score vector.
# These scores are ALREADY enriched by backbone context — the Linear layer
# sees features that encode information from neighboring chunks.
```

### Step 5c: Split into 3 temporal chunks
```python
chunks = framewise_logits.chunk(3, dim=1)
# chunk[0]: (B, 16, 234) — frames  0-15 → first 5s
# chunk[1]: (B, 16, 234) — frames 16-31 → center 5s
# chunk[2]: (B, 16, 234) — frames 32-47 → last 5s
```

Each chunk has 16 frames — the exact same count as v5j's 5s input.

### Step 5d: LSE pool each chunk independently
```python
clip_0 = lse_pool(chunk[0])   # (B, 234) — "what's in first 5s?"
clip_1 = lse_pool(chunk[1])   # (B, 234) — "what's in center 5s?"
clip_2 = lse_pool(chunk[2])   # (B, 234) — "what's in last 5s?"

clipwise_logits = stack([clip_0, clip_1, clip_2], dim=1)  # (B, 3, 234)
```

The LSE pool finds the peak activation in each 16-frame chunk — same as v5j.
But the per-frame scores it pools over are richer because the backbone had
15s of context.

**framewise_logits** is returned at full temporal extent: (B, 234, 48). This is
the full 15s frame-level predictions (used for framewise loss if enabled, and
for the blended_probs calculation).

---

## 6. Loss computation

### _compute_chunked_loss (clipwise_logits dim == 3)

```python
# clipwise_logits: (B, 3, 234)
# labels: (B, 234) — same label for all 3 chunks

# Compute BCE loss for each chunk
loss_0 = BCE_with_logits(clip_0, labels)  # first 5s vs file label
loss_1 = BCE_with_logits(clip_1, labels)  # center 5s vs file label
loss_2 = BCE_with_logits(clip_2, labels)  # last 5s vs file label

# Average across chunks
clipwise_loss = (loss_0 + loss_1 + loss_2) / 3

# With v8a config (clipwise_weight=1.0, framewise_weight=0.0):
total_loss = 1.0 * clipwise_loss + 0.0 * framewise_loss
           = clipwise_loss
```

**Why same label for all chunks works for focal recordings**: the entire file is
labeled with one species. Whether the bird calls in the first 5s, center 5s, or
last 5s, the label is correct. If a chunk is silence (bird not calling), the LSE
pool will output low scores, and the BCE loss will have a small gradient (since
sigmoid(low_score) ≈ 0, and the label says 1 — this teaches "low confidence when
no call detected"). This is the same noisy-label situation as v5j but with 3×
the training signal per sample.

**Approximation for soundscapes**: the label is from the center 5s only. Flanking
chunks may have different species. This means chunks 0 and 2 may receive incorrect
labels for ~4% of the training data. Acceptable for now.

---

## 7. Metrics

### _blended_probs (clipwise_logits dim == 3)

```python
# clipwise_logits: (B, 3, 234)
clip_probs = sigmoid(clipwise_logits).mean(dim=1)  # (B, 3, 234) → (B, 234)
# Average probability across chunks for val/auroc and other metrics

# With blend_clip=1.0, blend_frame=0.0:
probs = clip_probs  # (B, 234)
```

For validation metrics (AUROC, cmap, etc.), the model's prediction for each
sample is the average probability across all 3 chunks. This is reasonable
since the label represents the overall presence of the species.

---

## 8. Comparison: v5j (5s) vs v8a (3×5s)

| Aspect | v5j | v8a | Change |
|--------|-----|-----|--------|
| Audio loaded | 5s (160K samples) | 15s (480K samples) | 3× more context |
| Mel shape | (B, 1, 256, 511) | (B, 1, 256, 1533) | 3× wider |
| Backbone output T' | 16 | 48 | 3× more temporal positions |
| LSE pool per-chunk | 16 frames | 16 frames | **Same resolution** |
| Predictions per sample | 1 | 3 | 3× training signal |
| Batch size | 64 | 32 | Halved (memory) |
| Loss | BCE on 1 prediction | Avg BCE on 3 predictions | |
| Head architecture | LSEHead | LSEHead (same) | **No change** |
| Head parameters | Same | Same | **No change** |
| Backbone parameters | Same | Same | **No change** |

The ONLY differences are: longer input window and temporal splitting before pooling.
The model architecture (weights, layers, head) is 100% identical to v5j.

---

## 9. Inference considerations (task E.8, not yet implemented)

### ONNX export

The SEDWrapper in the export script currently does:
```python
probs = blend_clip * sigmoid(clipwise_logits) + blend_frame * sigmoid(framewise_max)
```

With n_chunks=3, `clipwise_logits` is (B, 3, 234). With blend_clip=1.0,
blend_frame=0.0, the output is `sigmoid(clipwise_logits)` = (B, 3, 234).

This means the ONNX model outputs **3 predictions per 15s window**, which is
what we want — each prediction corresponds to one 5s chunk.

### Kaggle kernel changes needed

Current kernel processes 5s segments independently. For v8a:

**Option A: 4 non-overlapping 15s windows**
- Process 0-15s, 15-30s, 30-45s, 45-60s → 4 ONNX calls
- Each returns (1, 3, 234) → 3 predictions per window
- 4 windows × 3 chunks = 12 predictions for twelve 5s segments
- Edge segments (0-5s and 55-60s) have no flanking context on one side
- Simple, efficient (4 ONNX calls vs 12 for v5j)

**Option B: Overlapping 15s windows (with center-chunk assignment)**
- For each 5s segment, load a 15s window centered on it
- Use only the center chunk prediction → 12 ONNX calls, each on 15s
- Every segment gets maximum context (flanking on both sides)
- More computation but each prediction has symmetric context

Option A is simpler and more efficient. Option B gives better context for
every segment. Decision deferred to task E.8.

---

## 10. What could go wrong

1. **Uneven chunk sizes**: if T' is not exactly divisible by n_chunks,
   `torch.chunk` returns chunks of unequal size. For v8a: T'=48, n_chunks=3
   → 16+16+16 (exact). For v8b: T'=64, n_chunks=4 → 16+16+16+16 (exact).
   But if backbone output varies slightly with input length, this could break.

2. **Zero-padded focal recordings**: 30% of focal recordings are <15s.
   Zero-padded regions produce near-zero mel values. The backbone features
   for padded regions will be low-information. LSE pool should handle this
   (it finds the peak, ignoring low-scoring frames), but if an entire chunk
   is padding, its prediction will be weak/random.

3. **Soundscape label approximation**: flanking chunks get the center label.
   Only 4% of data, but the incorrect labels could confuse the model for
   species that are truly absent in flanking segments.

4. **Batch size halved**: 32 vs 64. Changes MixUp pairing diversity and
   gradient noise. Could affect training dynamics — monitor val/auroc curve
   for instability.

5. **Memory**: 15s spectrograms are 3× wider → ~3× feature map memory.
   batch_size=32 should fit on RTX 6000 (24GB) with 16-mixed precision,
   but may be tight.
