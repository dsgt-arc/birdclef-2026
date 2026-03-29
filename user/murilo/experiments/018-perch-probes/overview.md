# Experiment 018: Pipeline Overview

A plain-language explanation of what exp 018 does and how the pieces fit together.

## The Idea

Every top-scoring public notebook on the BirdCLEF 2026 leaderboard (0.910–0.925)
uses the same foundation: Google's Perch v2 as a frozen feature extractor, with
progressively more sophisticated classifiers on top. No SED/EfficientNet needed.

This experiment climbs that ladder step by step:

1. **Phase C** (current): Fix probe training with OOF cross-validation → ~0.910
2. **Phase D**: Add ProtoSSM temporal modeling → ~0.917
3. **Phase E**: Add post-processing pipeline → ~0.925

### Google Perch v2 (~0.836 LB raw, ~0.910+ with probes)

- Google trained this on a *massive* global bird audio dataset (14,795 species)
- We use it **frozen** — no training at all, just run inference
- For each 5s audio window, it outputs two things:
  - **Logits** (14,795-dim): "how likely is each of 14,795 species"
  - **Embeddings** (1536-dim): a rich feature vector summarizing the audio content
- **Strong at:** species identification from massive pretraining, soundscape robustness
- **Weak at:** 25 insect sonotypes it has never heard, no temporal reasoning across windows

## The Species Mapping Problem

Perch knows 14,795 species, but BirdCLEF 2026 only has 234. We bridge the gap:

- **203 species**: direct match — scientific name exists in both
- **6 species**: genus proxy — Perch knows relatives in the same genus, we take
  the max logit across same-genus species
- **25 species**: unmapped — all insect sonotypes with no Perch representation,
  scored as -10.0 (effectively zero after sigmoid)

`scripts/build_species_mapping.py` produces this mapping automatically from
taxonomy.csv and Perch's labels.csv.

## The Probe Pipeline (How Perch Gets from 0.836 to 0.910)

Raw Perch logits mapped to 234 species score 0.836 LB. The jump to 0.910 comes
from **training lightweight sklearn classifiers** on the 66 labeled soundscapes.

### Why embeddings AND logits?

Perch outputs two things per 5s window:
- **Logits** (14,795-dim): direct species predictions — "I think this is species X"
- **Embeddings** (1536-dim): a rich internal representation — "this is what the audio sounds like"

The logits are limited to species Perch was trained on. For the 6 genus-proxy species
and especially for learning Pantanal-specific patterns, the embeddings are more
valuable. They encode acoustic information that can be *repurposed* for any task.

### Step 1: PCA Compression (1536 dims → 32-64 dims)

StandardScaler normalizes embedding dimensions, then PCA compresses to keep the
most informative signal. We have 792 training samples (66 files × 12 windows) —
far too few for 1536 features directly.

### Step 2: Per-Species Probes

For each species with at least 8 positive samples, train a separate classifier:

- **Input features** (39-dim in v1, 77-dim target):
  - PCA-compressed embeddings (32 or 64 dims)
  - Raw Perch logit + prior logit + fused base logit
  - Temporal context: prev/next window, file mean/max/std
  - Interaction terms: raw×prior, raw×base, prior×base (v3)
- **Output**: logit via `decision_function()`
- **Blending**: `(1 - alpha) * base_logit + alpha * probe_logit`

**Dynamic alpha**: Species with more training samples get higher alpha (more trust
in probe). `alpha = 0.25 + 0.25 * min(1.0, (n_pos - 8) / 40)`.

### Step 3: Bayesian Metadata Priors

Site/hour/site-hour probability tables from labeled soundscapes, fused additively
in logit space: `logits += lambda * prior_logits`
- Birds: lambda=0.4 (light prior)
- Frogs/Insects: lambda=1.0 (strong — highly location-dependent)

### Step 4: Temporal Smoothing

- **Birds** (events): local-max propagation (alpha=0.15)
- **Frogs/Insects** (textures): neighbor averaging (alpha=0.35)

### CRITICAL: Everything in Logit Space

Sigmoid is applied **once** at the very end. All intermediate operations (priors,
probes, blending, smoothing) happen in logit space. This was the key fix that took
us from 0.757 to 0.893.

## The OOF Gap (0.893 → 0.910)

Our v1 trains probes on all 66 files with no cross-validation. chiranjithdharma
uses 5-fold GroupKFold (by file) with fold-safe priors. Without OOF:
- Probes memorize the 66 labeled files
- Alpha blending weights are too aggressive
- Prior tables see all data, leaking information

Phase C adds proper OOF training to close this gap.

## ProtoSSM (Phase D, ~0.913-0.917)

A lightweight bidirectional state space model that captures temporal patterns
across the 12 windows per soundscape. For example: a dawn chorus pattern where
multiple species call in sequence, or a species that consistently appears in
windows 3-5 of a file.

- Input: PCA-compressed Perch embeddings (12 windows × d_model)
- Architecture: 2-layer bidirectional SSM + prototypical classification head
- Trains in ~10s on 66 labeled files
- Ensembled with MLP probes: W * ProtoSSM + (1-W) * MLP

ResidualSSM then learns systematic errors from the first-pass ensemble and applies
additive corrections (like gradient boosting in logit space).

## Post-Processing Pipeline (Phase E, ~0.917-0.925)

Six sequential stages, each adding small but additive gains:

1. **TTA**: Circular-shift 12-window sequence by [0, +1, -1], average predictions
2. **Per-taxon temperature**: Aves T=1.10 (softer), Texture T=0.95 (sharper)
3. **File-level confidence scaling**: Suppress noise in uncertain files
4. **Rank-aware scaling**: `probs *= file_max^0.5`
5. **Delta shift smoothing**: `(1-0.15)*x + 0.075*(prev + next)`
6. **Per-class threshold sharpening**: OOF-optimized F1 thresholds

## Architecture: Single-Kernel

All reference notebooks (0.910–0.925) use a single kernel:
1. Load pre-extracted labeled embeddings from dataset (~1s)
2. Train probes + ProtoSSM on labeled data (~10-30s)
3. Run Perch v2 live on ~600 test soundscapes (~80 min)
4. Apply probes + post-processing (~5s)

The ~80 min Perch inference is the unavoidable cost. Everything else fits in the
remaining ~10 minutes.

## Execution History

### Phase A: Offline pipeline (PACE) — DONE

```
build_species_mapping.py → artifacts/perch_species_mapping.json
extract_perch_embeddings.py → shared/embeddings/.../perch-v2/labeled/*.parquet
train_probes.py → shared/artifacts/.../perch-ensemble-v1/
local_validate.py → raw 0.7521 → +probes 0.9453 → +priors 0.9604
```

Raw Perch validated on Kaggle: **0.825 LB** (matches starter notebook).
Genus proxy mapping adds +0.011: **0.836 LB**.

### Phase B: Perch + Probes — DONE (0.893 LB)

Three critical fixes to make probes work: logit space throughout, additive priors,
StandardScaler before PCA. Gap to chiranjithdharma (0.910) = 0.017.

### Phase C: OOF + Richer Features — IN PROGRESS

Target: match chiranjithdharma (0.910) by adding OOF cross-validation, upgrading
to 77-dim features, and switching to MLP probes.
