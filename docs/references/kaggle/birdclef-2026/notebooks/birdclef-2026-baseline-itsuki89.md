---
slug: "itsuki89/birdclef-2026-baseline"
type: "notebook"
competition: "birdclef-2026"
date_captured: "2026-03-27"
votes: 0
---

# Kaggle Summary: BirdCLEF 2026 Baseline (itsuki89)

**Source**: https://www.kaggle.com/code/itsuki89/birdclef-2026-baseline
**Competition**: BirdCLEF+ 2026
**Score**: 0.902 LB

## Summary

Perch v2 frozen embeddings with per-class logistic regression probes and
hierarchical Bayesian priors fused via taxa-aware lambda weights. This is a
stripped-down variant of the chiranjithdharma 0.910 pipeline (same core ideas)
but without OOF cross-validation for the probes — it trains probes on the full
soundscape set without site-level folds, which likely explains the 0.008 gap.

## Approach

### Architecture

```
Soundscape (60s, 32kHz)
  -> 12 x 5s windows
  -> Perch v2 (frozen TF SavedModel, CPU-only)
    -> per-window: embeddings (1536-dim) + logits (14,795 classes)
  -> Map to BirdCLEF classes (direct scientific name + genus proxies)
  -> Hierarchical Bayesian priors (global -> hour -> site -> site x hour)
  -> Per-class LogisticRegression probes on PCA embeddings + context features
  -> Adaptive alpha blending -> Gaussian temporal smoothing -> sigmoid -> submission
```

### Species Mapping

Three tiers of class handling:

1. **Direct-mapped** (~140+ species): BirdCLEF species whose scientific name
   matches a Perch v2 output class. Logits copied directly.
2. **Genus-proxy mapped** (unmapped Amphibia/Insecta): For unmapped frogs and
   insects, find same-genus species in Perch's 14,795 labels and take the max
   logit across all genus matches. Recovers signal without training.
3. **Prior-only / inactive**: Unmapped species without genus hits get prior-only
   scores (or a hard -8.0 floor if never seen in training soundscapes).

### Inference Strategy

- Batch inference on 16 files at a time (192 windows per batch)
- Pre-computed Perch cache loaded from `jaejohn/perch-meta` dataset (parquet
  metadata + NPZ arrays) for training soundscapes
- Test soundscapes run through Perch live, then fused with the same prior
  tables and probe models trained on the cached training data

## Key Techniques Beyond the Basic Perch Starter (0.825 -> 0.902)

The raw Perch starter maps species and applies sigmoid — nothing else. This
notebook adds four layers of refinement:

1. **Hierarchical Bayesian priors (+~0.04)**: Four levels of location/time
   priors (global, per-hour, per-site, site x hour) with Bayesian smoothing
   (`n/(n+k)` weighting, k=8 for site/hour, k=4 for interaction). This is the
   single biggest lift — many species are strongly location-dependent.

2. **Taxa-aware lambda fusion (+~0.02)**: Different prior strengths for
   different taxa. Birds (event species) get light priors (lambda=0.4) since
   Perch already detects them well. Amphibia/Insecta (texture species) get
   strong priors (lambda=1.0) since their presence is mostly determined by
   location and time of day, not acoustic distinctiveness.

3. **Per-class logistic regression probes (+~0.01)**: For species with >=8
   positive training samples, a LogisticRegression on PCA-compressed Perch
   embeddings + sequential context features learns species-specific decision
   boundaries that improve on raw Perch logits.

4. **Temporal smoothing (+~0.005)**: Gaussian kernel smoothing across the 12
   windows within each 60s file. Uses two passes: first taxa-aware smoothing
   (alpha=0.35 on texture classes during fusion), then a final Gaussian smooth
   with kernel [0.1, 0.2, 0.4, 0.2, 0.1] across all classes.

## Data Pipeline

### Audio Loading

- `soundfile.read()` to float32, mono downmix if stereo
- Pad to exactly 60s (1,920,000 samples at 32kHz) if shorter, truncate if longer
- Reshape into 12 non-overlapping 5s windows (160,000 samples each)
- No augmentation, no resampling

### Cached Training Embeddings

- Pre-computed Perch embeddings and raw scores from `jaejohn/perch-meta`:
  - `full_perch_meta.parquet` — row IDs, filenames, site, hour metadata
  - `full_perch_arrays.npz` — `scores_full_raw` (raw Perch logits mapped to
    BirdCLEF classes) + `emb_full` (1536-dim Perch embeddings)
- Training labels parsed from `train_soundscapes_labels.csv`, deduplicated,
  multi-label union per (filename, start, end)

### Feature Engineering for Probes

Per-class feature vector (~39 dims):
- PCA-compressed Perch embeddings (32-dim via StandardScaler + PCA)
- Raw Perch logit for the class
- Prior logit for the class
- Fused base score for the class
- Sequential context: previous window score, next window score, file-level
  mean, file-level max (4 features from `seq_features_1d`)

## Post-Processing Pipeline

### 1. Bayesian Prior Fusion

Prior logits computed from training soundscape labels:

```
P(species | context) = hierarchical blend of:
  global_p  ->  hour_p (w = n_hour / (n_hour + 8))
              ->  site_p (w = n_site / (n_site + 8))
                ->  site_hour_p (w = n_sh / (n_sh + 4))
```

Converted to logits: `log(p) - log(1-p)`, then added to Perch scores with
class-type-dependent lambda weights.

### 2. Taxa-Aware Score Fusion

| Class type | Fusion formula | Lambda |
|------------|----------------|--------|
| Mapped event (birds) | `perch_logit + 0.4 * prior_logit` | 0.4 |
| Mapped texture (frogs/insects) | `perch_logit + 1.0 * prior_logit` | 1.0 |
| Proxy texture (genus-matched) | `proxy_logit + 0.8 * prior_logit` | 0.8 |
| Prior-only texture (unmapped, active) | `1.0 * prior_logit` (no Perch signal) | 1.0 |
| Prior-only event (unmapped, active) | `0.4 * prior_logit` (no Perch signal) | 0.4 |
| Unmapped inactive | Hard floor: `-8.0` | N/A |

### 3. Texture Temporal Smoothing (During Fusion)

Applied to all active texture classes (Amphibia + Insecta) within each 60s
file before probes:
```
smoothed = (1 - 0.35) * current + 0.5 * 0.35 * (prev_window + next_window)
```
Edge handling: first/last windows replicate their boundary values.

### 4. Probe Prediction Blending

For each species with a trained probe:
```
alpha = 0.25 + 0.25 * min(1.0, (n_pos - 8) / 40)
final = (1 - alpha) * fused_base + alpha * probe_decision_function
```
Alpha ranges from 0.25 (barely 8 samples) to 0.50 (48+ samples).

### 5. Final Gaussian Smoothing

Gaussian kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` convolved across the 12 windows
within each 60s file, applied to all classes. Uses `scipy.ndimage.convolve1d`
with `mode="nearest"` for edge handling.

### 6. Sigmoid Output

Final logits passed through `scipy.special.expit` (sigmoid) to produce
probabilities for submission.

## Training Details

No neural network training occurs in-notebook. The only "training" is:

- **PCA fitting**: StandardScaler + PCA(n_components=32) on full training
  soundscape Perch embeddings
- **Prior table fitting**: Aggregation of multi-label presence by site/hour
  from training soundscape labels
- **Logistic regression probes**: One per species with >=8 positive samples.
  `LogisticRegression(C=0.25, max_iter=400, solver="liblinear",
  class_weight="balanced")`. Trained on the full training set (no OOF folds).

All training happens on CPU, no GPU needed.

## Key Ideas

- **Hierarchical Bayesian priors are the biggest single lift**: Going from raw
  Perch species mapping to location/time-aware priors accounts for most of the
  gain over the 0.825 starter. Many species are geographically restricted, and
  the prior tables capture this.
- **Taxa-aware treatment is critical**: Birds (event vocalisations) and
  frogs/insects (continuous texture) need fundamentally different fusion
  strategies. Texture species are almost entirely location-determined.
- **Probes learn species-specific decision boundaries**: Raw Perch logits use a
  one-size-fits-all threshold. Per-class probes with sequential context features
  (neighboring windows, file-level stats) learn that, e.g., a frog that occurs
  at a site likely persists across multiple windows.
- **Adaptive alpha prevents overfitting on rare species**: The dynamic blending
  weight means probes for species with only 8 samples contribute only 25%
  while probes for common species contribute up to 50%.
- **No GPU needed**: Everything is sklearn + numpy + scipy. Fast to iterate.

## Results

| Metric | Value |
|--------|-------|
| Public LB | **0.902** |

### Comparison to related notebooks

| Notebook | Score | Key difference |
|----------|-------|----------------|
| Perch starter (raw) | 0.825 | No priors, no probes |
| **itsuki89 (this)** | **0.902** | Priors + probes, no OOF |
| chiranjithdharma | 0.910 | + OOF cross-validation (GroupKFold by site) |
| zirach ProtoSSM | 0.913 | + ProtoSSM temporal modeling + MLP probes |
| pantanal-distill | 0.917 | Distilled neural model + Perch ensemble |

## Connections

- **Derived from**: [perch-probes-chiranjithdharma](perch-probes-chiranjithdharma.md)
  (0.910) — nearly identical pipeline but itsuki89 skips OOF GroupKFold, which
  costs ~0.008 LB. The config constants and lambda values are identical.
- **Same family**: [perch-protossm-zirach](perch-protossm-zirach.md) (0.913),
  [perch-blamerx](perch-blamerx.md) (0.907),
  [pantanal-distill](pantanal-distill.md) (0.917)
- **Shared infrastructure**: Same Perch v2 model, same `jaejohn/perch-meta`
  cache dataset, same TF 2.20 wheel installation
- **Key gap vs chiranjithdharma**: The 0.910 notebook uses 5-fold GroupKFold by
  site for OOF probe training and prior estimation. This notebook trains probes
  on the full set, which risks overfitting site-specific patterns into the
  probe models. Fixing this is likely the easiest path to recover the 0.008 gap.
