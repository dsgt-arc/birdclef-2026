# SED Training: End-to-End Architecture Overview

*Last updated: 2026-04-04*

Full picture of our Sound Event Detection pipeline — what we build, how it
trains, how it infers, and what we've learned about each component. Written
as a reference for future experiment planning.

For competition/data context see:
- [BirdCLEF 2026 competition analysis](../../../docs/references/birdclef-2026-competition.md)
- [BirdCLEF 2026 EDA](../../../docs/references/birdclef-2026-eda-analysis.md)
- [AttHead comparison (exp 021)](../experiments/021-sed-training/atthead-comparison.md)
- [Mix2 augmentation reference](../../../docs/vendor/mix2-mixture-of-mixups.md)

---

## 1. The Task

BirdCLEF 2026: identify which of 234 species (birds, frogs, insects, mammals,
reptiles) are calling in 1-minute soundscape recordings from the Brazilian
Pantanal. Each 1-minute file is split into **12 × 5-second segments**. For each
segment, predict the probability of presence for each species. Scored by
**row-wise AUROC** across ~600 hidden test soundscapes.

Key challenge: the model must perform well on 5s segments of soundscapes
(multi-species, noisy, variable SNR) despite training mostly on focal recordings
(single-species, clean, targeted).

---

## 2. Training Data

### Focal recordings (35,549 files, 206 species)

Targeted recordings from iNaturalist and Xeno-Canto where someone recorded a
specific animal. Each file has **one label for the entire recording**
(`primary_label`) plus optional `secondary_labels` (background species, XC only).

- Variable length: median 23s, range 0-600s
- 97.9% are Aves; Amphibia 1.3%, Insecta 0.6%, Mammalia 0.3%, Reptilia <0.01%
- Heavy long-tail: 82/206 species have <100 recordings
- 12.3% have secondary labels (XC only; 0% of iNat recordings)

**How we use them**: random-crop a `window_duration` segment each training step.
The file-level label applies to the crop. Over many epochs, different crops from
the same file cover different parts of the recording. If a crop contains no
actual call (just silence/background), the label is still applied — this is noisy
but the LSE pooling head handles it by learning to output low scores when no call
is present.

### Labeled soundscapes (66 files, 1,478 segments, 75 species)

Expert-annotated 5s segments from Pantanal soundscape recordings. **This is the
target domain** — same recording conditions as the hidden test set.

- Per-5s multi-species labels (semicolon-separated): avg 4.2 species per segment
- Include all 28 soundscape-only species (25 insect sonotypes + 3 amphibians)
- Dense co-occurrence: top 15 co-occurring pairs are all frog-frog (nocturnal choruses)
- Only 4% of training samples but critical for domain alignment

**How we use them**: load a context window centered on each labeled 5s segment.
With `window_duration=5.0` (v5j), we load exactly 5s — the labeled segment itself.

### Unlabeled soundscapes (10,592 files)

Same recording sites as labeled soundscapes but no annotations. Not used in
current SED training. Could be used with pseudo-labeling (explored in exp 014-015,
regressed on LB).

---

## 3. Audio Preprocessing: Mel Spectrogram Pipeline

### AmplitudeToDB + standardize + min-max (v5j recipe)

This normalization was one of our biggest single improvements: **+0.014 LB** over
ttahara's min-max normalization (0.892 → 0.906).

```
Raw audio (5s waveform at 32kHz = 160,000 samples)
  → STFT (n_fft=2048, hop_length=313, win_length=2048)
  → Mel filterbank (256 bands, fmax=None → 16kHz Nyquist)
  → Power spectrogram → AmplitudeToDB (10 * log10, top_db=80dB)
  → Standardize (subtract dataset mean, divide dataset std)
  → Min-max scale to [0, 1]
  → Output: (1, 256, 511) — 1 channel, 256 mel bins, 511 time frames
```

Why each step matters:

1. **AmplitudeToDB**: Compresses the enormous dynamic range of power spectrograms
   (quiet sounds can be 1000× weaker than loud ones). Log-scale makes faint calls
   visible to the CNN. Without this, the model mostly sees loud events.

2. **Global standardization**: Centers data around zero with unit variance using
   precomputed dataset statistics. Helps ImageNet-pretrained backbones (which
   expect standardized input) work effectively on mel spectrograms.

3. **No resize**: ttahara's recipe resized spectrograms to 256×256, distorting
   temporal resolution. We keep the natural 256×511 shape — the backbone sees
   actual temporal structure without warping.

4. **NNAudio on GPU**: Mel computation runs on GPU (NNAudio library) instead of
   CPU DataLoader workers. This was necessary for n_fft=4096 (too slow on CPU)
   and is also faster for n_fft=2048.

### Spectrogram MixUp (current augmentation)

Applied after mel computation, before backbone:
```
mel_mix = lambda * mel_i + (1 - lambda) * mel_j
label_mix = lambda * label_i + (1 - lambda) * label_j
```

Parameters: `alpha=1.0` (uniform lambda), `theta=0.8` (probability of mixing),
per-sample (each sample in batch independently decides whether to mix),
5-epoch warmup (no mixing first 5 epochs).

We tested waveform MixUp (v6b: −0.003 offline) and waveform MixUp + pink noise +
time stretch (v6c: −0.013 offline). Both regressed vs spectrogram MixUp in offline
metrics, but **we never submitted v6b to LB** — and offline metrics have been
unreliable predictors of LB performance.

Competitors (Salman, yukiZ) report waveform MixUp and waveform augmentations
(pink noise, time stretch) as effective. The discrepancy may be due to our
AmplitudeToDB normalization already providing some of the robustness benefits.

---

## 4. Model Architecture

### Backbone: HGNetV2-B0 (6M params)

From `timm` library (`hgnetv2_b0.ssld_stage2_ft_in1k`). ImageNet-pretrained,
adapted to 1-channel input by averaging the 3-channel stem weights.

The backbone operates as a standard CNN feature extractor:
```
Input: (B, 1, 256, 511) — mel spectrogram
  → 4 stages of convolution + downsampling
  → Output: (B, 1024, 8, 16) — feature map
       C=1024 channels, F'=8 frequency bins, T'=16 temporal positions
```

The temporal reduction is ~32× (511 frames → 16 positions), so each temporal
position in the feature map corresponds to ~0.3s of audio.

**Why HGNetV2-B0**: Outperformed EfficientNet-B0 significantly on LB in exp 019
(v5g 0.892 vs v5f 0.855, same recipe). Also outperformed EfficientNet-B0 with
matched embed_dim (v5j 0.906 vs v5m 0.898). The HGNet architecture uses a
different convolution design than EfficientNet's MBConv blocks — Salman (0.921 LB)
also noted "MBConvs from EfficientNets are not learning that well" for mel
spectrograms. One exception: Boredom (9th place) claims EffNet can "easily reach
925+" — the recipe may matter more than the backbone.

### SED Head: LSEHead (our best)

The head converts the 4D backbone feature map into species predictions:

```
Backbone features: (B, 1024, 8, 16)
  → Mean freq pool: average over F'=8 → (B, 1024, 16)
  → Permute → (B, 16, 1024)
  → Linear(1024, 1024) + ReLU + Dropout(0.5) + Linear(1024, 234)
      → framewise_logits: (B, 16, 234) — per-frame species scores
  → LSE temporal pool: smooth-max over T'=16 → (B, 234)
      → clipwise_logits: one score per species
```

**LSE (Log-Sum-Exp) pooling** is a smooth approximation to max:
```
LSE(x) = T * (logsumexp(x / T) - log(N))
```
At temperature T→0 it's hard max; at T→∞ it's mean. T=1.0 is our default.

This is the key design for SED: the model doesn't need the species to be active
in every frame. If a bird calls for 0.5s out of 5s, that creates one high-scoring
frame — LSE pool captures that peak while ignoring the 15 silent frames. It
answers: "what's the strongest evidence for species X anywhere in this window?"

**Why LSEHead beats AttHead**: LSE has no learned temporal aggregation parameters —
just a deterministic smooth-max. It can't learn bad attention patterns, can't
collapse, can't focus on wrong frames. For BirdCLEF where the question is "find
the peak call in this window," the simplest approach works best. AttHead adds
learned per-species attention weights that can overfit with limited data.

### AttHead comparison

We tested several attention-based heads (see `atthead-comparison.md`):

| Head | Mechanism | val/auroc | LB | Delta vs v5j |
|------|-----------|-----------|-----|-------------|
| **LSEHead** | smooth-max (no learned temporal params) | **0.874** | **0.906** | — |
| AttHead | softmax(att) × raw_logits | 0.870-0.876 | 0.889-0.893 | −0.013 to −0.017 |
| AttBlockV2 | tanh×softmax × sigmoid(logits), PANNs-style | 0.815-0.868 | 0.800-0.819 | worse |
| AttHeadMultiContext | softmax(tanh(att)) × sigmoid(logits), Tattaka-style | 0.799-0.851 | — | much worse |

Our AttHead implementation also differs from the competition-standard (Tattaka/
BirdCLEF 2023 5th): we use `softmax(att)` instead of `softmax(tanh(att))`, and
average raw logits instead of sigmoid probabilities. See `atthead-comparison.md`
for the full analysis. These differences may explain part of the underperformance,
but LSEHead's simplicity is likely the main advantage for our data regime.

---

## 5. Training Recipe (v5j)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | HGNetV2-B0 (ImageNet init) | 6M params, 1024 embed_dim |
| Head | LSEHead (mean freq, T=1.0) | Clipwise-only loss |
| Loss | BCEWithLogitsLoss | Multi-label binary CE |
| Optimizer | AdamW | Single param group, uniform LR |
| LR | 5e-4 | OneCycleLR schedule |
| Weight decay | 1e-4 | |
| Scheduler | OneCycleLR | div_factor=25, final_div=4 |
| Warmup | 5 epochs | pct_start = 5/20 = 0.25 |
| Epochs | 20 | Best at epoch 19 typically |
| Batch size | 64 | |
| Window | 5s | Random crop per step |
| MixUp | Spectrogram, alpha=1.0, theta=0.8 | Per-sample, 5-epoch warmup |
| Precision | 16-mixed | AMP for memory efficiency |
| Split | 90/10 train/val | No test holdout, validate on LB |
| Data | Focal (35.5K) + labeled soundscapes (653 train segments) | |
| clipwise_weight | 1.0 | |
| framewise_weight | 0.0 | Only clipwise loss (no framewise) |
| blend_clip | 1.0 | Only clipwise at inference |
| blend_frame | 0.0 | |

Key design choices:
- **Clipwise-only loss** (no framewise supervision): simpler, works better on LB
- **OneCycleLR**: ttahara's recipe, better than CosineAnnealing for this setup
- **Per-sample MixUp**: each sample independently decides whether to mix (theta=0.8
  means 80% get mixed). More diversity than batch-level MixUp.
- **90/10 split, no test holdout**: maximize training data, validate on LB directly.
  Offline val/auroc has been unreliable — improvements often don't translate to LB.

---

## 6. Inference Pipeline

### ONNX export

The trained model (encoder + head) is wrapped and exported to ONNX:
```
SEDWrapper.forward(mel):
    features = encoder.forward_features(mel)       # backbone
    clipwise_logits, framewise_logits = sed_head(features)  # head
    framewise_max = framewise_logits.max(dim=-1).values
    probs = blend_clip * sigmoid(clipwise_logits)
          + blend_frame * sigmoid(framewise_max)
    return probs  # (B, 234) probabilities
```

With v5j (blend_clip=1.0, blend_frame=0.0), this simplifies to
`probs = sigmoid(clipwise_logits)`.

### Kaggle kernel inference

Runs on Kaggle CPU (no GPU). For each test soundscape file (60s):

1. Load full 60s audio at 32kHz
2. Compute mel spectrogram for the entire file at once (full-signal mel)
3. Slice the mel into twelve 5s segments (511 frames each)
4. Run ONNX inference on each segment → (234,) probability vector
5. Map model output indices to competition species columns via `species_mapping.json`
6. Write to `submission.csv`

Runtime: ~20 minutes for single model on ~600 test files (well within 90-min limit).

### Post-processing (optional)

Site/hour priors (Phase A): multiply predictions by prior probabilities based on
recording site and time of day. Gives +0.003 single / +0.004 4-fold on LB.

---

## 7. What We've Tried (Exp 021 Phases A-D)

### Phase A: Site/hour priors — small win (+0.003)

Multiplicative priors from labeled soundscapes (9 sites, 13 hours):
- v5j → v5j-pp2: 0.906 → **0.909** (+0.003)
- v5j-4fold → v5j-4fold-pp2: 0.913 → **0.917** (+0.004)

### Phase B: Recipe ablations on LSEHead — all regressed

| Change | val/auroc | Delta |
|--------|-----------|-------|
| CE loss (primary_label) | 0.854 | −0.020 |
| Waveform MixUp | 0.871 | −0.003 |
| Waveform + noise + stretch | 0.861 | −0.013 |
| 25 epochs (vs 20) | 0.873 | −0.001 |

CE loss is incompatible with our multi-label setup. Waveform augmentations
regressed offline (but v6b was never submitted to LB). Recipe changes on
LSEHead don't help.

### Phase C: AttHead + backbone scaling — regressed on LB

| Model | val/auroc | LB | Delta vs v5j LB |
|-------|-----------|-----|----------------|
| v6h (HGNetV2-B2 + AttHead) | **0.876** | 0.889 | **−0.017** |
| v6j (EffNet-B0 NS + AttHead) | 0.874 | 0.893 | **−0.013** |

AttHead with 75/25 clip/frame blend and dual supervision hurts LB despite
matching or beating v5j offline. Offline metrics are anti-correlated with LB
for this experiment. Our AttHead implementation also differs from competition
standard (see `atthead-comparison.md`).

### Phase D: AttHeadMultiContext (Tattaka approach) — massive regression

| Model | Context | val/auroc | Delta |
|-------|---------|-----------|-------|
| v7a (HGNet-B0) | 3×5s (15s) | 0.820 | −0.054 |
| v7b (HGNet-B0) | 4×5s (20s) | 0.799 | −0.075 |
| v7c (EffNet-B0 NS) | 3×5s (15s) | 0.851 | −0.023 |

Changed too many variables at once (head + window + batch size). Implemented
Tattaka's approach (aggregate attention over all 15s → 1 prediction) when
hengck23's actual approach is different (keep LSEHead, split features into
chunks). See next section.

---

## 8. Multi-Context: The Correct Approach (Phase E)

### The insight from hengck23 (0.896 LB, CNN-only)

Source: Kaggle discussion #683791, image at `notes/figures/hengck23-cnn-birdclef.png`

hengck23 uses our exact architecture (HGNetV2-B0 + LSEHead, 4-fold) with one
key change: **15s input windows (3×5s) instead of 5s**. His results:
- 5s baseline: LB 0.876 (4-fold, no PP)
- 3×5s: **LB 0.896** (4-fold, no PP) → **+0.020 gain**

The architecture:
```
15s audio → mel (256 × 1533 frames)
  → HGNetV2-B0 backbone → features (B, 1024, 8, 48)
       ↑ backbone sees full 15s context via convolutional receptive field
  → Mean freq pool → (B, 1024, 48)
  → Linear → framewise_logits (B, 48, 234)
  → SPLIT into 3 chunks: (B, 16, 234) × 3
  → LSEPool each chunk independently → 3 × (B, 234)
  → All 3 supervised during training
```

**Why this works**: The backbone's convolutional receptive field spans across
chunk boundaries. Features for frame 20 (center chunk) are influenced by
frames 10-30 (reaching into flanking chunks). So each 5s prediction benefits
from backbone-level context, but the LSE pooling still operates at 5s resolution
(16 frames per chunk — same as v5j).

**Why Phase D failed**: We replaced LSEHead with AttHeadMultiContext, which
aggregated attention over ALL 48 frames into one prediction. This diluted the
signal and changed the loss gradient path. hengck23 keeps LSEHead and just
splits the features — the multi-context comes from the backbone, not the head.

### Comparison: v5j vs v5i (10s) vs hengck23 (3×5s)

| Approach | Backbone sees | LSEPool operates on | Effect |
|----------|--------------|---------------------|--------|
| v5j (5s) | 5s (T=16) | 16 frames | 0.906 LB |
| v5i (10s single) | 10s (T=32) | **32 frames** | 0.880 LB (−0.026, diluted) |
| hengck23 (3×5s) | 15s (T=48) | **16 frames/chunk** | 0.896 LB (+0.020 over his baseline) |

v5i pools over the full long window (diluting peak signal). hengck23 splits
first, then pools per-chunk (preserving 5s resolution). The backbone gets more
context in both cases, but only hengck23's approach keeps LSE pooling tight.

### Phase E implementation plan

Only TWO things change from v5j:

1. **window_duration**: 5s → 15s (dataset already supports this)
2. **LSEHead**: add `n_chunks` param — split framewise logits into N chunks before
   LSE pooling, return N predictions

Everything else stays identical: same head architecture, same loss, same recipe,
same hyperparameters.

**Labels for multi-chunk training:**
- Focal recordings (96% of data): same file-level label for all 3 chunks.
  Correct — the whole file is labeled with one species.
- Soundscapes (4% of data): each 5s chunk has its own per-5s label from the CSV.
  This is actually better supervision than focal because we have true per-chunk
  ground truth. Requires loading flanking segment labels from the CSV.

**Inference with multi-context:**
- Process 60s test soundscape as four non-overlapping 15s windows
- Each 15s window → backbone → split into 3 chunks → 3 predictions
- 4 windows × 3 chunks = 12 predictions covering all twelve 5s segments
- Each segment gets a prediction that had backbone context from its neighbors

---

## 9. Augmentation: Current and Future

### Current: spectrogram MixUp (v5j)

Standard linear interpolation of mel spectrograms and labels. Works well —
our best LB results all use this.

### Tested and regressed offline (not submitted to LB)

- **Waveform MixUp** (v6b): −0.003 offline. Mixes raw audio before mel computation.
  Never submitted to LB — offline metrics have been unreliable.
- **Waveform + pink noise + time stretch** (v6c): −0.013 offline.

### Future: Mix2 (Mixture of Mixups)

Reference: `docs/vendor/mix2-mixture-of-mixups.md`

Paper (arxiv 2403.09598) shows that randomly alternating between Mixup (input
space), Manifold Mixup (embedding space), and MultiMix (batch-level Dirichlet)
gives +4.2% macro F-score overall and **+6.2% on rare classes** for anuran
classification.

Key insight: individual methods underperform, but combination is robust. Manifold
Mixup (mixing backbone features before the head) is the highest-value addition
for us. Could stack with multi-context — mix 15s feature maps before splitting
into chunks.

### Future: hengck23's "wave mix for 3×5sec"

hengck23 mentions "wave mix augmentation specially designed for 3×5sec input."
This may mean mixing different species into different 5s chunks within the 15s
window — teaching the model to discriminate per-chunk content even with shared
backbone context. Details unknown; worth experimenting if Phase E succeeds.

---

## 10. Key Lessons from Exp 021

1. **Offline val/auroc does not predict LB**. v6h had the best offline (+0.002)
   but the worst LB (−0.017). We should submit more models to LB even when
   offline metrics look flat or slightly worse (e.g., v6b waveform MixUp).

2. **LSEHead > AttHead for our setup**. The simpler smooth-max beats learned
   attention on LB, likely because it can't overfit and our data is limited.

3. **Don't change multiple variables at once**. Phase D changed head + window +
   batch size simultaneously, making it impossible to isolate the multi-context
   effect. Phase E changes only window_duration.

4. **The competition standard isn't always right for us**. AttHead with dual
   supervision is used by winning solutions, but our implementation underperforms
   LSEHead. The specific implementation details matter (tanh, sigmoid, loss path).

5. **Context comes from the backbone, not the head**. hengck23's approach feeds
   longer audio to the backbone (so convolutions see more context) but keeps the
   head operating at 5s resolution. This is fundamentally different from making
   the head aggregate over longer periods.

6. **AmplitudeToDB normalization is a major lever** (+0.014 LB). Don't touch it.

7. **The v5j recipe is very strong**. OneCycleLR, per-sample MixUp, BCE, HGNetV2-B0,
   clipwise-only loss — every ablation from this recipe has regressed.
