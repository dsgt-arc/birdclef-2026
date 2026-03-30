# Experiment 018: Pipeline Overview

A plain-language explanation of what exp 018 does and how the pieces fit together.

## The Idea

Every top-scoring public notebook on the BirdCLEF 2026 leaderboard (0.910–0.925)
uses the same foundation: Google's Perch v2 as a frozen feature extractor, with
progressively more sophisticated classifiers on top. No SED/EfficientNet needed.

This experiment climbed that ladder:

1. **Phase A-B** (done): Raw Perch → probes → 0.893 LB
2. **Phase C** (done): OOF cross-validation → 0.911 LB (beat chiranjithdharma 0.910)
3. **Phase D** (done): Full ProtoSSM v4 pipeline → submitted, awaiting LB (target ~0.925)

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

### Step 2: Per-Species MLP Probes

For each species with at least 8 positive samples, train a separate MLP classifier:

- **Input features** (77-dim in v3/v4):
  - PCA-compressed embeddings (64 dims)
  - Raw Perch logit + prior logit + fused base logit
  - Temporal context: prev/next window, file mean/max/std
  - Interaction terms: raw×prior, raw×base, prior×base
- **Classifier**: MLPClassifier(128,) with positive oversampling
- **Output**: predict_proba → manual log-odds conversion
- **Blending**: `(1 - 0.40) * base_logit + 0.40 * probe_logit` (fixed alpha)

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

## The OOF Gap (0.893 → 0.911)

Our v1 trained probes on all 66 files with no cross-validation. chiranjithdharma
uses 5-fold GroupKFold with fold-safe priors. Without OOF, probes memorize the
labeled files and prior tables leak information.

v2-fix added proper OOF + 4 bug fixes (K_HOUR=8, sc_clean priors, 6-category
fuse, 59 fully-labeled files) → **0.911 LB** (beat chiranjithdharma 0.910).

## ProtoSSM v4 (Phase D, target ~0.920-0.925)

Ported from Yuriy's ProtoSSM v5 (0.925 LB) into modular codebase under
`src/birdclef/perch/` (8 modules, 140 pytest tests).

### ProtoSSMv2 Architecture

A lightweight bidirectional state space model that captures temporal patterns
across the 12 windows per soundscape (e.g., dawn chorus, counter-singing):

- **Input projection**: Linear(1536 → 128) + LayerNorm + GELU
- **Positional encoding**: learnable (1, 12, 128)
- **Metadata**: site/hour embeddings → projected to d_model
- **BiSSM × 2 layers**: forward SSM + backward SSM → merge + LayerNorm + residual
- **Cross-attention**: 4-head MHA for non-local temporal patterns
- **Prototypical head**: cosine similarity with learned prototypes × temperature + bias
- **Gated fusion**: α * SSM_logits + (1-α) * Perch_logits (per-class learned α)
- **Parameters**: 722K (trains in ~10s on 59 files)

### Training

- **Loss**: focal BCE (γ=2.0) + 0.1 × MSE distillation from Perch + 0.1 × taxonomy aux
- **Augmentation**: file-level mixup (α=0.3, skip first 5 warmup epochs)
- **Optimizer**: AdamW + OneCycleLR (cos anneal, pct_start=0.1)
- **SWA**: average checkpoints from epoch 42+ (70% of 60 epochs)
- **OOF**: 3-fold GroupKFold by site

### ResidualSSM

Second-pass correction model (d=64, 1 BiSSM layer, 176K params). Learns to
predict residuals (labels − sigmoid(first_pass)) with MSE loss. Output head
zero-initialized so corrections start small. Applied as:
`final = first_pass + 0.3 × correction`

### Ensemble

- OOF-optimized blend: `W × ProtoSSM + (1-W) × MLP` (W=0.50 from OOF grid search)
- TTA: circular shifts [0, +1, -1] through ProtoSSM (no extra Perch cost)

## Post-Processing Pipeline (replaces Gaussian smoothing)

Six sequential stages in probability space:

1. **Per-taxon temperature**: Aves T=1.10 (softer), Texture T=0.95 (sharper) → sigmoid
2. **File-level confidence scaling**: multiply by top-2 mean confidence in file
3. **Rank-aware scaling**: `probs *= file_max^0.5`
4. **Delta shift smoothing**: `(1-0.15)*x + 0.075*(prev + next)`
5. **Per-class threshold sharpening**: fixed at 0.5 in submit mode (identity)

## Architecture: Single-Kernel

All reference notebooks (0.910–0.925) use a single kernel:

1. **Phase 1** (~15s): Load data → OOF priors → train MLP probes
2. **Phase 2** (~35s): ProtoSSM OOF → optimize ensemble weight → train final model → ResidualSSM
3. **Phase 3** (~80 min): Run Perch v2 on ~600 test soundscapes (batched, 16 files/TF call)
4. **Phase 4** (~1s): MLP probes → ProtoSSM + TTA → ensemble → ResidualSSM → 6-stage PP

The ~80 min Perch inference is the unavoidable cost. Everything else fits in ~1 min.

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

### Phase C: OOF + Bug Fixes — DONE (0.911 LB)

v2-fix beat chiranjithdharma (0.910) with 4 fixes: K_HOUR=8, sc_clean priors,
6-category fuse_scores, 59 fully-labeled files.

### Phase D: Full ProtoSSM v4 Pipeline — SUBMITTED (awaiting LB)

Ported Yuriy's ProtoSSM v5 (0.925 LB) into modular `src/birdclef/perch/` package.
77-dim MLP probes + BiSSM×2 + cross-attn + ResidualSSM + TTA + 6-stage PP.
140 pytest tests. Kernel: `submission_perch_protossm.py` (2756 lines).
Dry-run: 3.5 min (3 files), OOF AUC=0.9133, ensemble weight 0.50/0.50.
