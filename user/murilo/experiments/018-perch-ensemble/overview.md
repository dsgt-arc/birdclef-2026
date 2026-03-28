# Experiment 018: Pipeline Overview

A plain-language explanation of what exp 018 does and how all the pieces fit together.

## The Idea

The experiment has two phases:

**Phase A**: Reproduce chiranjithdharma's Perch-only pipeline (~0.910 LB) as a
validated baseline. This uses Google's Perch v2 foundation model with lightweight
sklearn classifiers trained on the 66 labeled soundscapes.

**Phase B**: Ensemble the Perch pipeline with our SED model (v4b, 0.859 LB) to
see if combining fundamentally different models pushes past 0.92.

We validate Phase A before attempting Phase B — if the Perch pipeline has a bug,
the ensemble would mask it.

### Model 1: Our SED (v4b, 0.859 LB)

- We *trained* this on BirdCLEF 2026 data (focal recordings + labeled soundscapes)
- Converts audio into a high-resolution mel spectrogram (n_fft=4096, 224 mel bins),
  then feeds it through EfficientNet-B0 with a temporal attention head
- Produces **frame-level** predictions — it knows *when* in a clip a bird calls
- Strong at: species we have lots of training data for, temporal localization
- Weak at: domain shift from clean focal recordings to noisy soundscapes

### Model 2: Google Perch v2 (~0.910 LB with probes)

- Google trained this on a *massive* global bird audio dataset (14,795 species)
- We use it **frozen** — no training at all, just run inference
- For each 5s audio window, it outputs two things:
  - **Logits** (14,795-dim): "how likely is each of 14,795 species"
  - **Embeddings** (1536-dim): a rich feature vector summarizing the audio content
- **Strong at:** species identification from massive pretraining, soundscape robustness
- **Weak at:** 25 insect sonotypes it has never heard, no temporal reasoning across windows

### Why Ensemble?

| | SED | Perch |
|---|---|---|
| Architecture | EfficientNet-B0 (3.6M params) | EfficientNet-B3 (~120M params) |
| Training data | BirdCLEF 2026 only | Global bird audio (10K+ species) |
| Audio frontend | High-res mel (n_fft=4096) | PCEN mel |
| Temporal modeling | Frame-level attention across 10s | Independent 5s windows |
| Species coverage | All 234 species | 209 species (25 unmapped) |

They're fundamentally different models looking at audio in different ways. When SED
is wrong, Perch might be right (and vice versa). That diversity is what makes
ensembles work.

## The Species Mapping Problem

Perch knows 14,795 species, but BirdCLEF 2026 only has 234. We need to bridge that gap:

- **203 species**: direct match — the scientific name exists in both BirdCLEF and
  Perch's label list (e.g., "Crotophaga ani" is in both)
- **6 species**: genus proxy — Perch doesn't know the exact species, but knows
  relatives in the same genus. For example, Perch doesn't know "Caiman yacare"
  (Pantanal caiman), but it knows "Caiman crocodilus" and "Caiman latirostris".
  We take the max score across same-genus species as a proxy.
- **25 species**: unmapped — these are all insect sonotypes (e.g., "Insect son01"
  through "son16") that have no representation in Perch's training data. For these,
  we rely entirely on our SED model.

`scripts/build_species_mapping.py` produces this mapping automatically from
taxonomy.csv and Perch's labels.csv.

## The Probe Pipeline (How Perch Gets from 0.825 to 0.910)

Raw Perch logits mapped to our 234 species score about 0.825 LB — decent but not
great, because Perch was trained to distinguish 14,795 species globally. It doesn't
know which 234 matter for the Pantanal, and it doesn't know that site S14 at 3am
means frogs.

The jump to 0.910 comes from **training lightweight sklearn classifiers** on the 66
labeled soundscapes. This isn't deep learning — it's logistic regression that trains
in ~30 seconds on CPU. But it's real training: we fit ~200 models (one per species),
a PCA transform, and Bayesian prior tables. This is what chiranjithdharma's notebook
does, and we follow the same approach.

### Why embeddings AND logits? Why not just use Perch's predictions directly?

Perch outputs two things per 5s window:
- **Logits** (14,795-dim): direct species predictions — "I think this is species X"
- **Embeddings** (1536-dim): a rich internal representation — "this is what the audio sounds like"

The logits are limited to species Perch was trained on. For the 6 genus-proxy species
and especially for learning Pantanal-specific patterns, the embeddings are more
valuable. They encode acoustic information that can be *repurposed* for any task.
For example, a probe can learn: "when the embedding has this pattern AND the Perch
score for a related Caiman species is high → predict Caiman yacare" — something the
raw logits can't express because Caiman yacare isn't in Perch's training set.

The probes use **both** as features: 32 PCA-compressed embedding dims + the raw Perch
score + temporal context = 37 features per species. Embeddings provide acoustic signal,
logits provide Perch's direct opinion, and the probe learns to combine them.

### Step 1: PCA Compression (1536 dims → 32 dims)

Perch embeddings are 1536-dimensional, but we only have 792 training samples
(66 files × 12 windows). Training classifiers on 1536 features with 792 samples
is a recipe for overfitting. PCA compresses the embeddings to 32 dimensions,
keeping the most informative signal while discarding noise.

### Step 2: Per-Species LogReg Probes

For each species with at least 8 positive samples in the labeled soundscapes, we
train a separate logistic regression model:

- **Input features** (37 dims): the 32 PCA-compressed embedding values + the raw
  Perch score for that species + temporal context (previous window score, next
  window score, file-level mean and max scores)
- **Output**: probability that this species is present in this window
- **Training**: 5-fold cross-validation grouped by recording site (so an entire
  site is held out at once, preventing data leakage)

**Why probes help**: Raw Perch scores are generic — trained on global data. The
probes learn Pantanal-specific patterns. For example: "when Perch gives species X
a score of 0.3 and the embedding looks like this, it's actually species Y in the
Pantanal context."

**Dynamic alpha blending**: Each species gets a trust weight for its probe:
`alpha = 0.25 + 0.25 * min(1.0, (n_positives - 8) / 40)`. Species with more
training samples get higher alpha (more trust in the probe). Species with barely
enough samples (near 8) get low alpha (mostly trust raw Perch). The final score
per species: `(1 - alpha) * raw_perch + alpha * probe_prediction`.

### Step 3: Bayesian Metadata Priors

Wildlife vocalizations follow strict geographic and circadian patterns. A frog
species that only exists at site S14 should have its score downweighted when we're
processing audio from site S22.

We build probability tables from the labeled soundscapes:
- P(species | site) — geographic distribution
- P(species | hour) — circadian activity patterns
- P(species | site, hour) — interaction term

The prior strength varies by taxa:
- **Birds** (lambda=0.4): light prior — birds are somewhat mobile and Perch is
  already good at them
- **Frogs/Insects** (lambda=1.0): strong prior — these species are highly
  location-dependent, so geographic signal is very informative

### Step 4: Temporal Smoothing

Different taxa need different smoothing strategies because they vocalize differently:

- **Birds** (events): brief, transient calls — use "local-max propagation" (alpha=0.15)
  that spreads peak detections to adjacent windows without diluting the peak itself.
  This handles calls that straddle 5s window boundaries.
- **Frogs/Insects** (textures): continuous, sustained sounds — use "neighbor averaging"
  (alpha=0.35) that fills in gaps where the model momentarily drops a continuous chorus.

## How SED Handles the 10s vs 5s Window Mismatch

Our SED model was trained on 10s audio windows, but the competition requires
predictions for 12 × 5s segments per 60s soundscape. Here's how it works:

The kernel loops in **5s steps** (0s, 5s, 10s, ..., 55s), producing one prediction
per segment. But for each 5s segment, the model actually processes a **10s window
centered on that segment** — 2.5s of context on each side.

For example, to predict segment [10s–15s]:
- Center window: model sees [7.5s – 17.5s] (weight: 50%)
- Left-shifted TTA: model sees [5.0s – 15.0s] (weight: 25%)
- Right-shifted TTA: model sees [10.0s – 20.0s] (weight: 25%)

The three overlapping predictions are weighted-averaged into one score for that 5s
segment. This gives us 12 predictions (one per segment) despite using 10s inference
windows — matching Perch's native 12 × 5s output.

The mel spectrogram is computed **once** for the entire 60s file, then sliced for
each window. This avoids redundant FFT computation across overlapping windows and
keeps inference under 30 minutes.

```
60s soundscape timeline:
|  0s  |  5s  | 10s  | 15s  | 20s  | ... | 55s  | 60s  |
|seg 0 |seg 1 |seg 2 |seg 3 |seg 4 | ... |seg 11|

SED inference windows (10s, centered on each 5s segment):
seg 0:  [<-2.5s  ——— 0s ——— 5s ——— 7.5s>]
seg 1:  [<2.5s  ——— 5s ——— 10s ——— 12.5s>]
seg 2:  [<7.5s ——— 10s ——— 15s ——— 17.5s>]
         ↑ 2.5s context   ↑ 5s target   ↑ 2.5s context

Perch inference windows (5s, no overlap):
seg 0:  [0s ——— 5s]
seg 1:  [5s ——— 10s]
seg 2:  [10s ——— 15s]

Both produce (12, 234) → ensemble blend is elementwise
```

## Phase B: The Ensemble Blend

Only attempted after Phase A validates the Perch-only pipeline.

Once both models output `(12, 234)` probability matrices:

```
final = 0.3 * sed_probs + 0.7 * perch_probs
```

Perch gets more weight (0.7) because Perch-based approaches score ~0.910 vs our
SED at 0.859. But SED still contributes:
- Unique signal for the 25 unmapped insect sonotypes (Perch gives 0.0 for these)
- Temporal patterns from frame-level attention that Perch's independent 5s windows miss
- Different errors due to different frontends and training data

For the 25 unmapped species, the kernel uses SED-only predictions (Perch contributes
nothing).

The 0.3/0.7 split is an initial estimate. `scripts/local_validate.py` sweeps from
0.0 to 1.0 on labeled soundscapes to find the actual optimum before Kaggle submission.

## Why a Two-Notebook Chain?

Running Perch v2 on ~600 test soundscapes takes ~80 minutes on Kaggle CPU. That
leaves no room for probes, smoothing, or SED inference within the 90-minute budget.

Every top-scoring notebook (0.910+) solves this with a **two-notebook chain**:

1. **Notebook A (Perch cache)**: Runs Perch v2 on all test soundscapes and saves
   the embeddings (1536-dim) + logits (14,795-dim) as kernel output. Takes ~80 min
   but runs as a separate kernel, not counted against the scoring kernel's budget.

2. **Notebook B (scoring)**: Loads the cached embeddings from Notebook A's output,
   applies pre-trained probes + metadata priors + temporal smoothing, and submits.
   Takes ~5-10 minutes — plenty of room to also run SED for ensemble.

The probe models are trained **offline on PACE** using the 66 labeled soundscapes
(same labeled data, just a different compute environment). At submission time,
Notebook B just loads the serialized probes and applies them to the new test
embeddings — no training happens during submission.

```
PACE (offline, one-time):
  66 labeled soundscapes → Perch v2 → embeddings
  embeddings + ground truth → train PCA + LogReg probes → serialize to Kaggle dataset

Kaggle Notebook A (per submission):
  600 test soundscapes → Perch v2 → embeddings + logits → save as kernel output

Kaggle Notebook B (per submission, reads Notebook A output):
  load cached test embeddings → apply PCA → apply probes → apply priors → smooth
  (optionally) run SED on test audio → blend with Perch scores
  → submit
```

## Execution Order

### Phase A: Offline pipeline (PACE) — DONE

```
build_species_mapping.py → artifacts/perch_species_mapping.json
extract_perch_embeddings.py → shared/embeddings/.../perch-v2/labeled/*.parquet
train_probes.py → shared/artifacts/.../perch-ensemble-v1/ (uploaded to Kaggle)
build_artifacts.py → verify + dataset-metadata.json
local_validate.py → raw 0.7521 → +probes 0.9453 → +priors 0.9604
```

Raw Perch inference validated on Kaggle: **0.825 LB** (matches starter notebook).

### Phase B: Two-notebook chain (NEXT)

1. Create + push Notebook A (Perch cache) — runs Perch, saves embeddings
2. Create + push Notebook B (scoring) — loads cache, applies probes, submits
3. Validate Perch + probes LB score (target ~0.910)

### Phase C: SED + Perch ensemble

Add SED inference to Notebook B. Sweep blend weights. Submit ensemble.

### Phase D: Perch pipeline improvements

Cherry-pick techniques from 0.921-0.926 notebooks (focal loss, file-level
scaling, TTA, per-taxon temperature) if ensemble alone isn't enough.

### Runtime Budget (with two-notebook chain)

| Component | Notebook | Est. Time |
|-----------|----------|-----------|
| Perch inference (600 files) | A (cache) | ~80 min |
| Load cache + apply probes | B (scoring) | ~5 min |
| SED inference (600 files) | B (scoring) | ~30 min |
| Blend + submit | B (scoring) | ~1 min |
| **Total Notebook B** | | **~36 min** (fits easily) |
