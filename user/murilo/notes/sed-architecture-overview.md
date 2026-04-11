# SED Training: End-to-End Architecture Overview

*Last updated: 2026-04-11*

Full picture of our Sound Event Detection pipeline — from raw `.ogg` files on
disk to Kaggle leaderboard score. Written so someone unfamiliar with BirdCLEF
or audio ML can follow every step.

For competition/data context see:
- [BirdCLEF 2026 competition analysis](../../../docs/references/birdclef-2026-competition.md)
- [BirdCLEF 2026 EDA](../../../docs/references/birdclef-2026-eda-analysis.md)
- [AttHead comparison (exp 021)](../experiments/021-sed-training/atthead-comparison.md)

---

## 1. The Task

BirdCLEF 2026: identify which of 234 species (birds, frogs, insects, mammals,
reptiles) are calling in 1-minute soundscape recordings from the Brazilian
Pantanal. Each 1-minute file is split into **12 × 5-second segments**. For each
segment, predict the probability of presence for each species. Scored by
**row-wise AUROC** across ~600 hidden test soundscapes.

Key challenge: the model must perform well on 5s segments of soundscapes
(multi-species, noisy, variable SNR) despite training mostly on focal recordings
(single-species, clean, targeted). This is called **domain shift** — the training
data and test data come from very different acoustic environments.

---

## 2. Training Data

All audio files are `.ogg` format (Ogg Vorbis compressed audio). Everything is
stored under `shared/data/birdclef-2026/`.

### Focal recordings (`train_audio/`) — 35,549 files, 206 species

```
train_audio/
├── 1161364/          # species folder (by taxon ID)
│   ├── iNat1216197.ogg
│   └── ...
├── 209233/
│   ├── iNat1545859.ogg
│   └── ...
└── ... (206 species folders)
```

Targeted recordings from iNaturalist and Xeno-Canto where someone deliberately
recorded a specific animal. Think: a birdwatcher holding a microphone toward a
bird in a tree. Each file has **one label for the entire recording**
(`primary_label`) plus optional `secondary_labels` (background species picked up
by the microphone).

- Variable length: median 23s, range 0–600s
- 97.9% are Aves; Amphibia 1.3%, Insecta 0.6%, Mammalia 0.3%
- Heavy long-tail: 82/206 species have <100 recordings
- Clean, high SNR — one dominant species per recording

**Metadata** in `train.csv`:
```csv
primary_label,secondary_labels,type,latitude,longitude,scientific_name,common_name,class_name,inat_taxon_id,author,license,rating,url,filename,collection
1161364,[],[],-22.7562,-46.8666,Guyalna cuta,Guyalna cuta,Insecta,1161364,Lucas Barbosa,cc-by-nc,0.0,...,1161364/iNat1216197.ogg,iNat
```

**How we use them**: each training step, we random-crop a `window_duration`
segment from the full recording. With 5s windows, a 60s recording yields many
different 5s views across epochs. The file-level label applies to every crop —
if a crop happens to contain silence (no bird calling), the label is still
positive. This is noisy, but the LSE pooling head (section 5) handles it by
learning to output low scores when no call is present in the audio.

### Labeled soundscapes (`train_soundscapes/`) — 66 of 10,658 files

```
train_soundscapes/
├── BC2026_Train_0001_S08_20250606_030007.ogg   # 1-minute recording
├── BC2026_Train_0002_S08_20250607_030007.ogg
└── ... (10,658 files total)
```

Filename encodes: `BC2026_Train_{fileID}_{site}_{date}_{time}.ogg`
Example: `S08` = Site 08, `20250606` = June 6 2025, `030007` = 3:00 AM.

These are autonomous recorder deployments in the Pantanal — a microphone left
in the field recording continuously. **This is the same domain as the test set.**
Multi-species, noisy, low SNR — multiple birds, insects, and frogs calling
simultaneously, with wind, rain, and other background noise.

Only **66 files have expert labels** (in `train_soundscapes_labels.csv`):
```csv
filename,start,end,primary_label
BC2026_Train_0039_S22_20211231_201500.ogg,00:00:00,00:00:05,22961;23158;24321;517063;65380
BC2026_Train_0039_S22_20211231_201500.ogg,00:00:05,00:00:10,22961;23158;24321;517063;65380
```

Labels are per 5-second segment, with semicolon-separated species IDs (multi-label:
avg 4.2 species per segment). 75 species appear in labeled soundscapes — including
28 species that have **zero focal recordings** (soundscape-only species, mostly
insect sonotypes).

The remaining **10,592 files are unlabeled** — same domain, no annotations. These
are what we tried to pseudo-label in exp 025.

### Species breakdown: 234 total

| Category | N | Description |
|----------|---|-------------|
| Focal + soundscape | 47 | Have both focal recordings and soundscape labels |
| Focal only | 159 | Only focal recordings, zero soundscape data |
| Soundscape only | 28 | Only in labeled soundscapes (25 insect sonotypes + 3 amphibians) |
| **Total** | **234** | |

The 159 focal-only species are the core problem — the model trains on clean
single-species recordings but must predict in noisy multi-species soundscapes
for these species. This domain shift is what limits our LB score.

---

## 3. Audio Processing: From .ogg to Model Input

### Why 32kHz sample rate?

All audio is resampled to 32kHz regardless of the source file's native sample rate
(many `.ogg` files are 48kHz on disk). The Nyquist theorem says you can capture
frequencies up to half the sample rate: 32kHz → max 16kHz. Bird vocalizations
mostly fall below 12kHz, so 16kHz captures everything we need.

- **16kHz** (used by some pipelines): captures up to 8kHz — misses high-pitched
  insect calls and some bird harmonics
- **32kHz** (our choice): captures up to 16kHz — covers all species in this
  competition. Also matches the Perch model's sample rate.
- **48kHz**: captures up to 24kHz — wastes compute on ultrasonic frequencies

We use 32kHz for both training and inference so the mel spectrograms are identical.
If we trained at 32kHz but inferred at 16kHz, the spectrograms would have different
frequency resolution and the model would produce wrong predictions.

### Why spectrograms, not raw waveforms?

A raw audio waveform is just a sequence of air pressure measurements over time.
A 5-second clip at 32kHz is 160,000 numbers — the amplitude of the sound wave at
each sample point. It contains all the information, but frequency patterns (what
makes a bird call distinct) are hidden inside the oscillation rates. A neural
network would need to learn to extract frequency information from scratch.

A **mel spectrogram** converts this 1D waveform into a 2D image where:
- The Y-axis is **frequency** (256 mel-scaled bands from low to high pitch)
- The X-axis is **time** (511 frames spanning 5 seconds)
- The pixel intensity is **loudness** at that frequency and time

Bird calls appear as visible patterns — horizontal lines (pure tones), ascending
curves (song phrases), harmonic stacks (multiple frequencies at once). This
converts "what species is calling?" into a visual pattern recognition problem.

This is why we can use **ImageNet-pretrained backbones** — a CNN trained to
recognize shapes in photographs can recognize shapes in spectrograms. The
frequency-time patterns of a bird call are analogous to edges and textures in
images.

**The tradeoff**: spectrograms throw away phase information (the exact timing of
wave oscillation cycles), but keep the frequency content that distinguishes
species. For bird identification, frequency content matters far more than phase.

### The mel spectrogram pipeline (v5j recipe)

```
.ogg file on disk
  → Load audio (soundfile/torchaudio)
  → Raw waveform: 160,000 samples (5s × 32kHz)
  → STFT (Short-Time Fourier Transform)
      n_fft=2048 (frequency resolution: ~15 Hz per bin)
      hop_length=313 (time resolution: ~10ms per frame)
      → Complex spectrogram: (1025 freq bins × 511 time frames)
  → Mel filterbank (256 bands)
      Maps linear frequency to mel scale (perceptual pitch scale)
      → Power mel spectrogram: (256 × 511)
  → AmplitudeToDB
      10 × log10(power), clamped to 80dB dynamic range
      Compresses enormous loudness range — faint calls become visible
  → Standardize (subtract dataset mean, divide dataset std)
  → Min-max scale to [0, 1]
  → Output: (1, 256, 511) tensor — 1 channel, 256 mel bins, 511 time frames
```

Why each step matters:

1. **STFT**: Decomposes the waveform into frequency content at each time step.
   Like a prism splitting white light into colors, but for sound into pitches.

2. **Mel filterbank**: Human hearing is logarithmic — we perceive the difference
   between 100Hz and 200Hz (one octave) the same as 1000Hz and 2000Hz (one octave).
   Mel scale mirrors this, giving more resolution to lower frequencies where bird
   calls are most diverse.

3. **AmplitudeToDB**: A quiet bird 50 meters away might produce 1000× less power
   than a loud bird 2 meters away. Without log-scaling, the CNN only sees the loud
   bird. Log-scale compresses this range so both are visible. This normalization
   was one of our biggest single improvements: **+0.014 LB** over the previous
   min-max normalization.

4. **Global standardization**: Centers data around zero with unit variance using
   precomputed dataset statistics. Helps ImageNet-pretrained backbones (which
   expect standardized input) work effectively on spectrograms.

5. **No resize**: Some pipelines resize spectrograms to a fixed size (e.g., 256×256),
   distorting temporal resolution. We keep the natural 256×511 shape so the backbone
   sees actual temporal structure without warping.

### Where does mel computation happen? The full training_step flow

The datasets return **raw waveforms** — the mel conversion does NOT happen in
the DataLoader. Instead, it runs **on GPU** inside the Lightning module's
`training_step()`. This is faster (GPU STFT outperforms CPU, especially for
large FFT sizes) and allows waveform augmentations before mel conversion.

Here's the complete sequence for one training step:

```
1. DataLoader loads batch from disk:
   (x, labels, primary_idx) where x = raw waveforms (64, 160000)
   → 64 samples × 5s × 32kHz, loaded by SEDFocalDataset/SEDSoundscapeDataset
   → Transferred to GPU automatically by PyTorch Lightning

2. [Optional] Waveform augmentations (on raw audio, before mel):
   → Pink noise injection (adds realistic background noise)
   → Time stretch (speeds up or slows down the audio)
   (v5j doesn't use these — Salman does)

3. Mel spectrogram conversion ON GPU:
   mel = self.mel_transform(x.float())
   → x: (64, 160000) raw waveforms
   → mel: (64, 1, 256, 511) spectrograms
   This is NNAudioMelSpectrogram: STFT + mel filterbank + AmplitudeToDB +
   standardization, all computed on GPU in one call. Runs under torch.no_grad()
   since the mel transform has no learnable parameters.

4. [Default] Spectrogram MixUp (on mel spectrograms):
   mel, labels = self._apply_mixup(mel, labels)
   → Blends pairs of spectrograms and their labels

   [Alternative] Waveform MixUp (between steps 2 and 3):
   x, labels = self._apply_mixup(x, labels)  → then compute mel
   → Mixes raw audio before mel conversion (Salman's approach)

5. Forward pass through the model:
   clipwise_logits, framewise_logits = self(mel)
   → self.encoder(mel) → backbone features (64, 1024, 8, 16)
   → self.sed_head(features) → predictions

6. Loss computation:
   loss = self._compute_chunked_loss(clipwise, framewise, labels)
   → BCEWithLogitsLoss on clipwise logits vs label vectors

7. Backward pass → optimizer step → scheduler step
```

**Why GPU mel matters for waveform MixUp**: If mel runs on CPU (in the DataLoader),
we can only mix spectrograms (step 4). With GPU mel, we can mix raw waveforms
(step 2) then compute mel — this creates acoustic interference patterns between
overlapping species that don't occur with spectrogram mixing, potentially better
simulating real multi-species soundscapes.

### Spectrogram MixUp (data augmentation)

After mel computation, before feeding to the backbone, we blend pairs of
spectrograms to create synthetic training examples:

```
mel_mix   = λ × mel_A   + (1 − λ) × mel_B
label_mix = λ × label_A + (1 − λ) × label_B
```

This teaches the model to handle overlapping species (common in soundscapes where
multiple animals call simultaneously). Parameters: `alpha=1.0` (uniform λ),
`theta=0.8` (80% of samples get mixed), per-sample mixing, 5-epoch warmup.

**Waveform MixUp** (mixing raw audio before mel computation) is an alternative.
Salman (0.922 LB) uses this. We tested it and it regressed offline, but offline
metrics have been unreliable — the waveform version was never submitted to LB.
The key difference: waveform mixing creates acoustic interference patterns that
don't occur with spectrogram mixing, potentially better simulating real
multi-species soundscapes.

---

## 4. Data Loading: How Training Samples Are Constructed

### Focal recordings: random crops

The `SEDFocalDataset` handles focal recordings. For each training step:

1. Pick a recording (e.g., a 45-second bird song)
2. Random-crop a `window_duration` segment (e.g., 5s → samples 32000–192000)
3. If the recording is shorter than `window_duration`, pad with zeros
4. Return `(waveform, label_vector, primary_species_index)`

The label vector is 234-dimensional: 1.0 for the primary species, 0.0 for
everything else (one-hot encoded, expanded to 234 classes). The same recording
yields different crops each epoch, providing data augmentation for free.

**The noisy label problem**: A random 5s crop from a 60s recording might contain
only silence — the bird might call for just 10 seconds of the recording. But the
label still says "this species is present." How often does the crop actually
contain the call?

| Window | Recording | Call duration | P(call in crop) |
|--------|-----------|---------------|-----------------|
| 5s | 45s | 5s | ~11% |
| 5s | 45s | 15s | ~33% |
| 20s | 45s | 5s | ~57% |
| 20s | 45s | 15s | ~100% |

With 5s windows, most crops of short-calling species are **pure noise with a
positive label**. The model survives this because:
- Over 20 epochs, each recording is sampled ~20 times with different random crops.
  Some crops catch the call (strong gradient) while silent crops produce weak
  confused gradients that average out.
- LSE pooling (section 5) learns to output low scores when no call is present,
  so the "silent crop" case produces a small loss.

Longer windows dramatically improve the call-capture probability. **Energy-based
crop selection** (picking the loudest region instead of random) could push this
further — e.g., 95%+ for 20s windows — but we haven't implemented this yet.
Competitors may be doing this (Gemini suggested "RMS energy sampling").

### Soundscape recordings: context windows

The `SEDSoundscapeDataset` handles labeled soundscape segments. Unlike focal
recordings (random crop, no ground truth for which seconds contain the call),
soundscapes have **exact per-5s labels**.

1. Each segment is a labeled 5s chunk (from `train_soundscapes_labels.csv`)
2. Load a `context_duration` window **centered** on the 5s target
   - With `window_duration=5.0` (v5j): load exactly the 5s segment
   - With `window_duration=20.0`: load 20s centered on the 5s segment
     (7.5s before + 5s target + 7.5s after)
3. The label comes from the CSV: multi-hot vector with all species present
4. Return `(waveform, label_vector, primary_species_index)`

There is **no randomness** here — the same segment always loads the same context
window. The label is the ground truth for the center 5s.

**Label mismatch with longer windows**: When loading 20s context around a labeled
5s segment, the flanking 15s may contain **different species** than the center 5s.
A frog calling in seconds 0–5 might not be calling in seconds 10–15. But the
label only covers the center 5s — the model receives 20s of audio with a label
that describes only the middle portion. With AttHead producing a single prediction
over the full 20s, this creates a training conflict if the attention focuses on
flanking species. In practice, this is mitigated by soundscapes being only 2% of
training data, and dense species co-occurrence (many species persist across
adjacent segments).

### Batch shape alignment

Every sample in a batch must have the **same tensor shape** — PyTorch's DataLoader
stacks them into a single tensor. This means focal and soundscape samples must
return the same waveform length.

The `SEDDataModule` ensures this by passing `window_duration` to both datasets:
- `SEDFocalDataset` gets `window_duration` directly → random crop of that length
- `SEDSoundscapeDataset` gets `context_duration=window_duration` → centered window
  of that length

With `window_duration=20.0`: focal samples are 640,000 samples (20s random crop),
soundscape samples are also 640,000 samples (20s centered on 5s target). Both fit
in the same batch.

### Batch composition

A training batch combines both datasets:
- Focal recordings: ~32,000 train samples (96% of data)
- Soundscape segments: ~650 train segments (4% of data)

They're concatenated into one `ConcatDataset` and shuffled together. Each batch
of 64 samples is mostly focal recordings with a few soundscape segments mixed in.

### The 90/10 split

We hold out 10% of focal recordings and 8 soundscape files for validation:
- Train: ~32,000 focal + ~650 soundscape segments
- Val: ~3,500 focal + ~86 soundscape segments

The validation set is used for monitoring training progress (loss, AUC), but
**offline val metrics have been unreliable predictors of the Kaggle LB score**.
This is why we added **dual validation** in exp 026 — tracking focal and
soundscape metrics separately to see which better predicts LB performance.

---

## 5. Model Architecture & Forward Pass

The `forward()` method of our Lightning module is just two lines:

```python
def forward(self, mel):
    features = self.encoder.forward_features(mel)   # backbone
    return self.sed_head(features)                   # head
```

But a lot happens inside each. Here's the complete forward pass with concrete
tensor shapes for v5j (batch_size=64, 5s window):

### Step 1: Backbone — compress the spectrogram into features

The backbone is HGNetV2-B0 from the `timm` library
(`hgnetv2_b0.ssld_stage2_ft_in1k`). Originally trained on ImageNet photographs,
adapted to 1-channel input by averaging the 3-channel RGB stem weights.

It's a standard CNN with 4 stages of convolution + downsampling. Each stage
roughly halves the spatial dimensions:

```
Input (mel):     (64, 1,    256, 511)    # mel spectrogram
After stage 1:   (64, 32,   128, 256)    # 2× downsample
After stage 2:   (64, 64,    64, 128)    # 4×
After stage 3:   (64, 512,   16,  32)    # 16×
After stage 4:   (64, 1024,   8,  16)    # 32× ← we use this
```

We take only the last stage output: `features = (64, 1024, 8, 16)`.

What each dimension means:
- **1024 channels**: each channel is a learned "detector" — one might fire on
  rising frequency sweeps, another on harmonic stacks, another on broadband noise.
  These are abstract learned features, not interpretable individually.
- **8 frequency positions**: the 256 mel bins compressed to 8 (each covers ~32
  mel bins ≈ a wide frequency band)
- **16 time positions**: the 511 frames compressed to 16 (each covers ~32 frames
  ≈ 0.31 seconds of audio)

Each "pixel" in this 8×16 feature map represents about a 0.3-second ×
wide-frequency-band patch of the spectrogram, described by 1024 features. The
backbone's convolutional receptive field means each pixel is influenced by a
wider area — nearby pixels share information through overlapping convolutions.

**Why HGNetV2-B0 over EfficientNet-B0**: In exp 019, HGNetV2-B0 outperformed
EfficientNet-B0 by +0.037 LB (0.892 vs 0.855) with the exact same recipe. The
HGNet architecture uses a different convolution block design. Salman (0.921 LB)
also noted "MBConvs from EfficientNets are not learning that well" for mel
spectrograms. Exception: Boredom (#1) claims EfficientNet can "easily reach
925+" — the full recipe may matter more than the backbone alone.

### Step 2: Frequency pooling — collapse the frequency axis

The head first eliminates the frequency dimension. We only need to know **when**
a species is calling, not at which exact frequency — that information is already
encoded in the 1024 feature channels by the backbone:

```python
h = features.mean(dim=2)   # (64, 1024, 8, 16) → (64, 1024, 16)
                            #  average over F=8 frequency positions
```

Now we have a **temporal sequence**: 16 time positions, each described by 1024
features.

### Step 3: Per-frame classification — what species are at each time position?

Two linear layers map each time position independently from 1024 features to
234 species scores:

```python
h = h.permute(0, 2, 1)             # (64, 1024, 16) → (64, 16, 1024)
framewise_logits = self.cls_fc(h)   # (64, 16, 1024) → (64, 16, 234)
```

`cls_fc` is: `Linear(1024→1024) → ReLU → Dropout(0.5) → Linear(1024→234)`

Now `framewise_logits[b, t, s]` is: "in batch sample b, at time position t
(~0.3s window), the raw score (logit) for species s."

These are **logits**, not probabilities — they're raw unnormalized scores ranging
from -∞ to +∞. A logit of +5.0 means "very likely present", -5.0 means "very
likely absent", and 0.0 means "50/50". Converting to probability requires the
sigmoid function: `prob = 1 / (1 + exp(-logit))`.

The linear layers are applied **independently to each time position** (same
weights shared across all 16 positions). This means the classifier doesn't know
about temporal order — it just asks "what species evidence is at this position?"

### Step 4: LSE pooling — find the peak evidence across time

```python
clipwise_logits = self.lse_pool(framewise_logits)  # (64, 16, 234) → (64, 234)
```

For each of the 234 species independently, LSE reads the 16 frame-level scores
and produces one summary score. The formula:

```
LSE(x₁...x₁₆) = T × (log(exp(x₁/T) + ... + exp(x₁₆/T)) − log(16))
```

With T=1.0, this behaves like a **soft max** — dominated by the highest-scoring
frame but slightly influenced by others.

**Concrete example**: A bird calls at frame 7 (logit = 5.0), all other frames are
silent (logits ≈ -2.0). The LSE output is approximately 5.0 — the peak is
preserved. With mean pooling it would be: `(5.0 + 15×(-2.0)) / 16 = -1.56` —
the call is completely lost.

**Why LSE works for bird audio**: A bird might call for just 0.5 seconds out of
a 5-second clip. That creates one high-scoring frame out of 16 — LSE captures
that peak while mostly ignoring the 15 silent frames. It answers: "what's the
strongest evidence for species X anywhere in this window?"

### Step 5: Return both outputs

```python
return clipwise_logits, framewise_logits.permute(0, 2, 1)
# clipwise:  (64, 234)        — one prediction per species per clip
# framewise: (64, 234, 16)    — per-frame predictions (kept for inference blend)
```

Both are returned because at inference we blend them (section 6). During training,
only `clipwise_logits` is used for the loss.

### Complete forward pass diagram

```
mel spectrogram (64, 1, 256, 511)
         │
         ▼
┌─── HGNetV2-B0 backbone (4 conv stages) ───┐
│  Input:  (64, 1, 256, 511)                │
│  Output: (64, 1024, 8, 16)                │
│  32× spatial reduction in each dim        │
└───────────────────────────────────────────┘
         │
         ▼  mean over frequency dim
temporal sequence (64, 1024, 16)
    "16 time positions × 1024 features each"
         │
         ▼  Linear(1024→1024) → ReLU → Dropout → Linear(1024→234)
framewise_logits (64, 16, 234)
    "at each of 16 time positions, how likely is each species?"
         │
         ├──────────────────────── kept for inference blend
         │
         ▼  LSE pool over 16 time positions
clipwise_logits (64, 234)
    "what's the strongest evidence for each species in this clip?"
         │
         ▼  used for training loss (BCE)
```

The entire forward pass is differentiable — gradients flow backward from the loss
through LSE pooling, through the linear layers, through the backbone convolutions,
all the way back to the spectrogram input. The backbone learns what frequency-time
patterns to detect, and the head learns how to map those patterns to species.

### LSEHead vs AttHead — why it matters

**LSEHead** has **no learned temporal aggregation parameters** — LSE pooling is a
fixed mathematical operation (smooth max). It can't learn bad attention patterns,
can't collapse, can't focus on wrong frames. For BirdCLEF where the question is
"find the peak call," the simplest approach works best. We tested AttHead 7 times
across exp 021 and LSEHead won every time on LB, even when AttHead had better
offline metrics.

**AttHead** replaces LSE pooling with learned attention:

```python
# AttHead forward (simplified):
attention_weights = softmax(attention_network(framewise), dim=time)   # (B, 234, 16)
clipwise = sum(attention_weights × sigmoid(framewise), dim=time)      # (B, 234)
```

The attention mechanism learns "which frames to focus on" for each species. In
theory, this is more flexible than LSE's fixed smooth-max. In practice, it
underperformed on LB (0.889–0.893 vs 0.906 for LSEHead).

**However**: AttHead may work better with **longer windows** (10–20s). With 5s
windows, the entire clip likely contains the bird call, so attention doesn't add
value over smooth-max — both produce similar results, but LSE is more stable.
With 20s windows, most of the audio is background noise, and attention's ability
to **select the 3–5 seconds where the bird actually calls** becomes critical.
LSE pools uniformly over all time positions, diluting a 3-second call across 20
seconds of silence. This is the key hypothesis for exp 026.

---

## 6. Training Recipe (v5j)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Backbone | HGNetV2-B0 (ImageNet init) | 3.1M trainable params |
| Head | LSEHead (mean freq, T=1.0) | Clipwise-only loss |
| Loss | BCEWithLogitsLoss | Multi-label binary cross-entropy |
| Optimizer | AdamW | Single param group, uniform LR |
| LR | 5e-4 | Same for backbone and head |
| Weight decay | 1e-4 | |
| Scheduler | OneCycleLR | div_factor=25, final_div=4 |
| Warmup | 5 epochs | pct_start = 5/20 = 0.25 |
| Epochs | 20 | Best checkpoint typically at epoch 17–19 |
| Batch size | 64 | |
| Window | 5s | Random crop per step |
| MixUp | Spectrogram, alpha=1.0, theta=0.8 | Per-sample, 5-epoch warmup |
| Precision | 16-mixed | AMP for memory/speed |
| Split | 90/10 train/val | No test holdout |
| Data | 35.5K focal + 653 soundscape segments | |
| clipwise_weight | 1.0 | Only clipwise loss |
| framewise_weight | 0.0 | No framewise supervision |

### Loss function: BCE (Binary Cross-Entropy with Logits)

For each of the 234 species independently: "is this species present?" Binary
yes/no for each species, allowing multiple species to be present simultaneously
(multi-label classification). The loss is:

```
BCE(logit, label) = −[label × log(σ(logit)) + (1 − label) × log(1 − σ(logit))]
```

Where σ is the sigmoid function. Summed across all 234 species, averaged across
the batch. Soft labels from MixUp (e.g., label=0.7) work naturally with BCE.

**Why not Cross-Entropy (CE)?** CE treats species prediction as a single-choice
problem (pick THE species), but soundscapes have multiple species calling
simultaneously. We tested CE loss 6 times and it always regressed on LB (−0.015
to −0.030). However, Salman and Boredom both use CE — they may handle the
multi-label issue differently (CE during training for gradient strength, sigmoid
at inference for independent probabilities). This remains an open question.

### What "clipwise-only loss" means

We only compute loss on the `clipwise_logits` (the single (B, 234) prediction
after LSE pooling). We don't supervise the `framewise_logits` (per-frame
predictions). In theory, framewise supervision could teach the model which frames
contain the call, but in practice the file-level labels from focal recordings
don't tell us WHEN the bird calls — so framewise supervision would push the model
to predict the species in every frame, even silent ones.

At inference, we blend: `probs = 0.75 × sigmoid(clipwise) + 0.25 × sigmoid(max(framewise))`.
The framewise-max term adds a "strongest frame" signal without requiring framewise
training labels.

**Train/inference mismatch**: The v5j config specifies `clipwise_weight=1.0,
framewise_weight=0.0` (clipwise-only training), but the ONNX export used the
script defaults of `blend_clip=0.75, blend_frame=0.25`. So v5j's 0.906 LB
includes framewise signal that the model was **never trained to optimize**. The
framewise-max logits are a byproduct of the forward pass (computed but not
supervised). This works because the framewise head shares weights with the
clipwise path — but it's an accidental win, not a deliberate design choice.

---

## 7. Inference Pipeline

### ONNX export

The trained model is exported to ONNX format for Kaggle CPU inference:

```python
class SEDWrapper(nn.Module):
    def forward(self, mel):
        features = encoder.forward_features(mel)
        clipwise_logits, framewise_logits = sed_head(features)
        framewise_max = framewise_logits.max(dim=-1).values
        probs = 0.75 * sigmoid(clipwise_logits) + 0.25 * sigmoid(framewise_max)
        return probs  # (B, 234) probabilities
```

### Kaggle kernel inference

Runs on Kaggle CPU (no GPU). For each test soundscape file (60 seconds):

1. Load full 60s audio at 32kHz → 1,920,000 samples
2. Compute mel spectrogram for the entire file at once
3. Slice the mel into twelve 5s segments (511 frames each)
4. Run ONNX inference on each segment → (234,) probability vector
5. Map model output indices to competition species columns
6. Write 12 rows to `submission.csv`

Runtime: ~20 minutes for a single model on ~600 test files (well within the
90-minute Kaggle limit).

### Post-processing (optional)

Site/hour priors: multiply predictions by prior probabilities based on recording
site and time of day (from labeled soundscape statistics). Effect: +0.003 single
model / +0.004 with 4-fold ensemble.

---

## 8. The Window Length Question

### Why 5s may not be optimal

Our v5j trains on 5s random crops. But the test set evaluates 5s segments that
are part of continuous 60-second recordings. A bird call might span a segment
boundary (start at 4.5s, end at 5.5s — split across two segments). With 5s
training, each crop is isolated — the model never learns that context from
neighboring seconds can help identify a partial call.

Competitors are training with longer windows:
- **Salman (0.922 LB)**: trains on 10–20s windows
- **hengck23 (0.896 LB)**: trains on 15s (3×5s chunks)
- **Tom (6th place)**: trains on 20s (4×5s)

### Two approaches to longer windows

**Approach 1: Single prediction from longer window**

Train on 20s, have the head produce one prediction for the whole 20s window.
Requires an attention-based head (AttHead) that can learn to focus on the frames
where the bird is calling and ignore the rest.

```
20s audio → backbone features (B, 1024, 8, 64 frames)
  → AttHead: learned attention selects relevant frames
  → Single prediction: (B, 234)
```

This is likely what Salman does. The attention mechanism acts as an implicit
"bird call finder" — it learns which frames contain the actual call and weights
them heavily, ignoring the background noise frames.

**Approach 2: Multi-chunk with per-chunk pooling (hengck23)**

Train on 15–20s, but split the features into N×5s chunks at the head level,
applying LSEHead to each chunk independently:

```
15s audio → backbone features (B, 1024, 8, 48 frames)
  → Split into 3 chunks of 16 frames each
  → LSEHead on each chunk → 3 × (B, 234) predictions
  → All 3 supervised with the same label during training
```

The backbone sees 15s of context (convolutions reach across chunk boundaries),
but pooling stays at 5s resolution. Each 5s prediction benefits from the
backbone knowing what's happening in neighboring chunks.

### What we've tested

| Model | Window | Head | Approach | LB |
|-------|--------|------|----------|-----|
| v5j | 5s | LSEHead | Baseline | **0.906** |
| v5i | 10s | LSEHead (single pool) | Naive: pool over full 10s | 0.880 |
| v8a | 15s | LSEHead (3 chunks) | hengck23's approach | 0.882 |
| v8b | 20s | LSEHead (4 chunks) | hengck23's extended | 0.896 |
| v7a | 15s | AttHeadMultiContext | Tattaka approach | — (offline −0.054) |
| Salman | 20s | AttHead? | Single prediction | **0.922** |

**Key observations**:
- v5i (10s, naive pooling): massive regression. LSE over 32 frames dilutes signal.
- v8a/v8b (multi-chunk LSEHead): better than v5i, but still worse than v5j.
  The multi-chunk approach preserves 5s resolution but doesn't fully close the gap.
- **Untested**: AttHead + 20s (single prediction, attention selects frames).
  This is the key hypothesis for exp 026 — attention may work better than LSE
  for longer windows where most audio is background noise.

---

## 9. Dual Validation (exp 026)

### The problem with our current validation

Our 90/10 split creates a validation set that's 97% focal recordings and 3%
soundscape segments. The `val/auroc` metric is dominated by focal performance —
how well the model recognizes clean single-species recordings. But the LB tests
soundscape performance — noisy multi-species recordings.

This means: improvements in `val/auroc` often don't translate to LB improvements.
We've seen this across 15+ experiments — the offline/LB disconnect is systematic.

### The fix: separate metrics

Dual validation tracks two separate AUC metrics during training:

- **`val/auroc`**: AUC on focal validation recordings (training stability signal)
- **`val_sc/auroc`**: AUC on soundscape validation segments (LB proxy)

The checkpoint is selected based on `val_sc/auroc` — the model version that
performs best on soundscapes (our target domain) gets submitted to Kaggle.

Implementation: the data module returns two separate val dataloaders (focal and
soundscape), and the Lightning module dispatches to separate metric sets based
on `dataloader_idx`.

---

## 10. Key Lessons (exps 016–026)

1. **Only input representation and inference-time changes improve LB.** 15+
   training recipe changes (loss, augmentation, head, regularization) have zero
   successes. Every positive LB delta came from mel processing, backbone choice,
   post-processing, or ensembling.

2. **Offline val/auroc does not predict LB.** Models with better offline metrics
   frequently score worse on the leaderboard. The domain shift between focal
   recordings and soundscapes makes focal-dominated validation unreliable.

3. **LSEHead > AttHead at 5s windows.** Simpler smooth-max beats learned attention
   on LB, likely because it can't overfit. But this may reverse with longer windows.

4. **Don't change multiple variables at once.** Phase D (exp 021) changed head +
   window + batch size simultaneously — impossible to know what caused the
   regression. Phase E isolated window_duration only.

5. **Context comes from the backbone, not the head.** hengck23's approach feeds
   longer audio to the backbone (convolutions see more context) but keeps the head
   at 5s resolution. Fundamentally different from making the head aggregate over
   longer periods.

6. **AmplitudeToDB normalization is a major lever** (+0.014 LB). Don't touch it.

7. **The v5j recipe is very strong but may have hit a ceiling.** Every perturbation
   regresses at 5s. The path to 0.920+ likely requires a structural change (longer
   windows, different head) rather than recipe tuning.

8. **The LB gap is data coverage (exp 024).** 0.9958 AUC on 75 known species,
   but 159 species have zero soundscape data. Pseudo-labeling attempted to fix
   this (exp 025) but regressed — the base model needs to be stronger first.

---

## 11. Identified Gaps and Open Questions

Potential improvements we haven't explored yet, ordered by expected impact:

### 1. Random cropping is wasteful (high priority)

With 5s random crops from focal recordings, we catch the bird call only ~11% of
the time for a species that calls for 5s in a 45s recording. The model wastes most
training steps on silence-with-positive-labels. Longer windows help (57% at 20s)
but still miss calls in long recordings.

**Energy-based crop selection**: Instead of random start position, compute the RMS
energy across the recording and select the window with the highest energy. This
virtually guarantees the vocalization is in the crop. Competitors may be doing this
(Salman trains on 20s with "original labels" — may imply smart cropping).

Implementation: compute RMS in `__getitem__` before loading (or precompute and
cache). Adds CPU cost but dramatically improves label quality.

### 2. Train/inference blend mismatch (medium priority)

v5j trains with `clipwise_weight=1.0, framewise_weight=0.0` but infers with
`blend_clip=0.75, blend_frame=0.25`. The model was never trained to optimize the
framewise predictions that contribute 25% of the inference score.

**Fix**: either train with framewise supervision (add `framewise_weight=0.25`) or
export with `blend_frame=0.0` (clipwise-only inference). The current setup is an
accidental mismatch that happens to work.

### 3. Soundscape label mismatch with longer windows (low priority, but worth noting)

When loading 20s context around a labeled 5s soundscape segment, the flanking
15s may contain different species. The label covers only the center 5s, but the
model sees 20s. With AttHead producing a single prediction over 20s, the
attention might focus on a flanking species not in the label.

Mitigation: soundscapes are only 2% of training. For multi-chunk approaches
(hengck23), we COULD load per-chunk labels from the CSV for soundscape segments
(neighboring 5s segments have their own labels). Not implemented.

### 4. No curriculum or difficulty-aware sampling (speculative)

All focal recordings are sampled uniformly. Species with 5 recordings get the same
sampling rate as species with 500. A curriculum that oversamples rare species or
undersamples easy species might help the long-tail problem.

### 5. CE loss mystery (open question)

We tested CE loss 6 times and it always regressed. But Salman (0.922) and Boredom
(#1) both use CE during training with sigmoid at inference. Either our CE
implementation is wrong (forcing single-label prediction on multi-label data), or
CE interacts with other choices (window length, augmentation, head type) in ways
we haven't tested. CE + AttHead + 20s is an untested combination that might work.

---

## 12. Score Progression

| LB | Model | Key Change | Exp |
|-----|-------|-----------|-----|
| 0.795 | SED v1 | SED architecture | 016 |
| 0.859 | SED v4b | + AmplitudeToDB normalization | 017 |
| 0.892 | v5g | + HGNetV2-B0 backbone | 019 |
| 0.906 | **v5j** | + NNAudio AmplitudeToDB | 019 |
| 0.913 | v5j-4fold | + 4-fold ensemble | 019 |
| 0.917 | v5j-4fold-pp2 | + site/hour priors | 021 |
| 0.925 | Perch V18 TF | Frozen Perch + MLP probes | 020 |
| 0.929 | **Perch+SED** | Score-level ensemble (best) | 023 |

Target: 0.948 (#1 position). Gap: 0.019.
