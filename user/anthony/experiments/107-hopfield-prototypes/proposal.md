# Proposal: Prototype Retrieval for Amphibian-Dominated Soundscapes

**Status**: proposed
**Created**: 2026-04-10 (axiomatic simplification 2026-04-11)
**Author**: Anthony

## Thesis

Exp 106 established that **prototype retrieval in Perch's fixed embedding
space is domain-shift bottlenecked**: ridge regression hit 0.749 LOSO
AUROC and a star-schema cosine classifier got 0.816 on focal-overlap
species but collapsed on the rest. The memory-bank architecture works;
the fixed feature space doesn't bridge focal↔soundscape.

Exp 107 replaces the fixed space with a **learned projection** over the
same Perch v2 embeddings and trains a **factored star-schema Hopfield
retrieval** (species × site prototype banks) end-to-end. The axiomatic
contribution is that **every puzzle piece in our dataset is used through
one unified retrieval mechanism**:

1. **Strong Perch embedding space** (frozen) — input features for all
   streams; known to represent amphibians and insects well in addition
   to birds (exp 105/106 ridge 0.749 on the full taxonomy).
2. **Weak focal labels** — 265K focal windows with recording-level bag
   labels drive the species memory bank via BCE.
3. **Unsupervised soundscape data** — 127K unlabeled Pantanal windows
   with known site metadata drive the site memory bank via **Hopfield
   energy minimization**, the same retrieval equation used at inference.
4. **Structural star schema + prototypes** — species and site prototypes
   compose via a Bayesian gated fusion (per-species Platt → multiplicative
   fusion) to produce the per-window 234-class multi-label score. The
   hierarchical species bank (family + species delta) handles the
   long-tailed data distribution.

The anchoring signal is 739 labeled SC windows (strong stream) that
pull the whole model toward the target domain. Everything else flows
through the Hopfield retrieval machinery.

## Background

### Perch's taxonomy gap

Perch v2 is a 14,795-species bird classifier, but its **embeddings**
are a general bioacoustic feature space: exp 105/106 ridge regression
on the full 234-species BirdCLEF 2026 taxonomy (amphibians + insects
included) hits 0.749 AUROC. The embeddings carry species-discriminative
information for non-birds; only the classifier head is bird-only.

### What exp 106 learned

1. **Ridge on Perch embeddings = 0.749 LOSO AUROC** — the ceiling for
   any linear classifier in Perch's fixed space, bottlenecked by
   domain shift.
2. **Star schema cosine = 0.816 on focal-overlap species, collapses on
   the rest** — prototype retrieval works where focal data exists but
   fails across the focal↔soundscape domain gap in the fixed space.
   The architecture is right; the space is wrong.

## Hypothesis

A factored star-schema memory-bank model, with a **learned projection**
of Perch embeddings and prototypes trained via Hopfield retrieval
(supervised BCE on the species bank + unsupervised energy minimization
on the site bank) will beat exp 106's ridge (0.749) and close a
substantial fraction of the gap on species without focal data or
without Perch coverage.

The mechanistic claim: once the projection is learned end-to-end and
both memory banks live in the same learned space, the factored
composition `species × site` that failed in Perch's fixed logit space
becomes meaningful again. Site prototypes trained by Hopfield
clustering on unlabeled data give the fusion a target-domain anchor
without needing pseudo-labels.

## Method

### Architecture

A learnable memory-bank classifier operating on frozen Perch v2
embeddings. All 4.1M parameters live downstream of the cached
embeddings — Perch itself is not fine-tuned.

```
Perch emb (1536)
       │
       ▼
  proj MLP (1536 → 1024 → 512)
       │
       ▼
    W_Q (512)
       │
       ├───────────────────────┐
       ▼                       ▼
   W_K_species             W_K_site
       │                       │
       ▼                       ▼
  species memory           site memory
  M_f[family_of]           (21, K, 512)
    + Δ_species
  (F,K,512) + (234,K,512)
       │                       │
  logsumexp_k             logsumexp_k
       │                       │
   Platt a,b                sigmoid
       │                       │
    sigmoid                    │
       │                       │
       └────── gate ───────────┘
                │
                ▼
           scores (234,)
```

- **Projection**: `Linear(1536, 1024) → GELU → Linear(1024, 512) → LayerNorm`.
  Nonlinear; shaped by BCE over labeled data + the unsupervised site
  loss via the site bank's gradient path.
- **Hierarchical species memory**: `M_species[s] = M_family[family_of(s)]
  + Δ_species[s]`. `M_family: (F, K, 512)` holds prototypes shared
  across species in each family (~30 families from BirdCLEF taxonomy
  metadata); `Δ_species: (234, K, 512)` holds species-specific
  refinements. K=8 slots per bank. Weight decay on `Δ_species` (1e-2,
  vs 1e-5 elsewhere) keeps low-resource species close to the family
  prior; high-resource species specialize away from it.
- **Site memory bank**: `(21, K, 512)`, K=8 per site. Flat — no
  hierarchy (21 Pantanal sites aren't taxonomically grouped).
- **Retrieval projections** `W_Q`, `W_K_species`, `W_K_site` — standard
  Ramsauer formulation of modern Hopfield networks.
- **Aggregation**: `(1/β)·logsumexp_k(β·q_r·W_K·M_{s,k})` per species,
  then per-species Platt `sigmoid(a_s·score + b_s)`. Platt is
  load-bearing: it makes the species scores proper probabilities so
  the multiplicative `species · site_score` fusion is a valid Bayesian
  composition.
- **Gated fusion**: `score = g·species + (1-g)·species·site_score`,
  with `g = sigmoid(Linear(q))` a learned per-window gate.
- **Output dim**: 234 competition species (no silence class).
- **Total params**: ~4.1M.

### Data streams

Three streams, no filters, no pseudo-labels, no silence class:

| Stream | Source | Count | Labels |
|--------|--------|-------|--------|
| **Strong** | 739 labeled SC | 739 | Window-level GT (234-d multi-hot) |
| **Weak (focal)** | All focal, bird + non-bird, every window, no filter | ~265K | Recording bag label (one-hot on primary species) |
| **Unsupervised** | All 127K unlabeled SC | 127K | No species label — only `site_idx` metadata |

Every focal window keeps its recording's primary-species label,
uniformly across the four regimes (bird/non-bird × has-Perch/no-Perch).
Noise from silent between-call windows is absorbed by the K=8 prototype
slots and the family hierarchy, not by a preprocessing filter. Perch's
role in v1 is **only** the frozen embedding space — it is not a
teacher, a filter, or a pseudo-labeler on any stream.

### Loss: two terms, one novel mechanism

```python
def loss(batch, lambda_site=0.1):
    # Labeled term: strong + weak focal through BCE
    x = concat(batch.strong, batch.weak_focal)
    pred = model(x.emb, x.site_idx)              # site_idx=None for focal
    L_labeled = bce(pred, x.labels)

    # Unsupervised term: Hopfield energy minimization on unlabeled SC.
    # Projection is DETACHED so the site bank clusters within the
    # species-shaped space, not the other way around.
    q_det = model.proj(batch.unsup.emb).detach()
    q_r   = model.W_Q(q_det)
    beta_l = model.log_beta_site.exp()
    M_l    = model.W_K_site(model.site_memory[batch.unsup.site_idx])  # (B,K,D)
    sim    = einsum(q_r, M_l, 'b d, b k d -> b k')
    L_site_unsup = -(torch.logsumexp(beta_l * sim, dim=-1) / beta_l).mean()

    return L_labeled + lambda_site * L_site_unsup
```

**What each term does:**

- **`L_labeled`**: trains the species bank (and everything that routes
  into it — projection, W_Q, W_K_species, Platt, gate, site bank via
  gated fusion on labeled SC windows) on the 739 strong + 265K weak
  focal BCE signal. The species bank is the retrieval target of the
  labeled supervision.
- **`L_site_unsup`**: minimizes the Hopfield retrieval energy for each
  unlabeled SC window against its **known site's** memory bank. This is
  the same equation (`(1/β)·logsumexp_k(β·q_r·W_K·M_{l,k})`) used in
  the forward pass at inference — training and inference share the
  retrieval operator. The loss pulls each window toward SOME prototype
  in its site, which is soft-EM clustering in the projection space.
  Projection is detached so this loss trains only the site memory,
  `W_K_site`, and `log_beta_site` — **not** the projection. Without the
  detach, the loss would push the projection to encode site-identifying
  features (microphone fingerprints, ambient noise signatures), which
  is the exp 106 "site structure entangled with domain shift" failure
  mode we're trying to avoid.

**Why this is principled.** The unsupervised site clustering reuses
the Hopfield retrieval machinery already in the architecture. No
pseudo-labels, no entropy minimization, no contrastive pair construction,
no reconstruction head. The site bank learns from 127K windows of
target-domain acoustic data via the same operator that composes with
the species bank at inference time. The star schema's factored
structure and the Hopfield energy function together give us a natural
way to train each bank from the data source that best fits it:
species ← labeled, site ← unsupervised + gated fusion on labels.

**Single hyperparameter**: `λ_site`. Default 0.1, tune via LOSO.
No ramps, no warm-up, no `γ_weak`.

### Per-species balanced sampling

The weak focal stream uses a `WeightedRandomSampler` with per-species
weights `1 / n_windows_in_species`. This gives every species equal
representation per epoch and prevents common species from dominating
the family prototype gradient — a known issue with the hierarchical
`M_family + Δ_species` parameterization where gradient on a common
species flows into its family prototype and would otherwise make
`M_family[f]` track the common species instead of a family average.

The unsupervised SC stream is sampled uniformly (no species identity
concept at this stream level).

### Initialization — random

- **Projection MLP**: Kaiming
- **Family memory**: small Gaussian
- **Species delta**: zero (species start as pure family fallback)
- **Site memory**: small Gaussian
- **W_Q, W_K_*, gate**: default linear init
- **Platt**: identity (`a_s = 1`, `b_s = 0`)
- **`log β`**: 0

No k-means preprocessing. The soft-EM gradient identity (see
`docs/star-schema-dimensional-analysis.md`) shows that the Hopfield
retrieval loss IS soft clustering — gradient descent on the labeled
BCE + unsupervised Hopfield energy performs the E/M updates
automatically. K-means warm-start is a convenience, not a necessity,
and doing k-means in raw Perch space then projecting through a random
MLP partially destroys the cluster structure anyway.

### Training schedule

- **Optimizer**: AdamW, `lr = 3e-4`, cosine decay, 30–40 epochs
- **Parameter groups**:
  - `species_delta`: weight decay `1e-2` (strong shrinkage to the
    family prior)
  - Everything else: `1e-5`
- **Early stopping**: LOSO AUROC evaluated every 5 epochs
- **Compute**: GPU (`gpu-rtx6000`, RTX 6000 × 1, 12h walltime)

### Taxonomy hierarchy and low-resource species

Species prototypes are parameterized as
`M_species[s] = M_family[f(s)] + Δ_species[s]`. Family memory
receives gradient from every species in the family (plus the balanced
sampler ensures per-species-equal gradient contribution), so `M_family`
trains strongly regardless of any individual species' data volume.
Weight decay on `Δ_species` keeps low-resource species close to their
family prior.

Effect across the four species regimes:

| Regime | Behavior |
|--------|----------|
| Species with lots of focal data | Develops a strong `Δ_species[s]`, specializes away from the family prior |
| Species with little focal data | `Δ_species[s]` stays near 0 under weight decay; effective prototype ≈ family mean — structural regularization |
| Species with zero focal but family siblings have focal | `Δ_species[s]` never gets training gradient; effective prototype = family template. **Partially learnable via siblings.** |
| Species with zero focal and no family sibling with focal | Truly unlearnable. Flagged and excluded from the aggregate AUROC. |

Family level is chosen because whole bird families, frog families, and
insect families typically share acoustic signatures (call morphology,
frequency band, stridulation mechanism). Genus is too narrow; class is
too broad. The species → family mapping is a preflight requirement,
pulled from BirdCLEF metadata or an iNaturalist/eBird fallback CSV.

## Evaluation

LOSO CV on 7 folds (sites with ≥20 labeled SC windows). Each fold
retrains the full model from scratch on all data except the held-out
site's strong-stream positives. Final metric: mean per-fold macro
AUROC, reported with subset breakdowns.

| Metric | Purpose |
|--------|---------|
| Overall LOSO macro AUROC | Primary — directly comparable to exp 106 ridge 0.749 |
| Bird vs non-bird subset | Does the learned projection generalize to non-bird audio? |
| With-focal vs without-focal subset | Prototype quality scales with focal data availability |
| Has-family-sibling-with-focal vs orphaned subset | Does family inheritance rescue zero-focal species? |
| Site × species affinity matrix | Eval-time deliverable; ecological interpretation |

Species with no training signal from any source (no focal, no strong
positives, no family sibling with focal) are flagged **unlearnable**
and excluded from the aggregate AUROC; listed separately in the report.

### Baselines

| Model | LOSO AUROC | Source |
|-------|------------|--------|
| Ridge on Perch embeddings | 0.749 | Exp 106 (overall) |
| Star schema cosine | 0.816 / collapses | Exp 106 (42 focal / 28 no-focal split) |

### Success criteria

- **Primary**: overall LOSO AUROC ≥ 0.76. Beats ridge, justifies the
  learned projection.
- **Go/no-go**: overall LOSO AUROC ≥ 0.749 ± fold variance. If the
  full model can't match a linear classifier in its own feature space,
  the construction isn't earning its complexity.
- **Mechanism**: non-bird subset AUROC should substantially exceed the
  level a naive "don't use unlabeled SC, no site bank" baseline would
  achieve — evidence that the unsupervised site loss and the star
  schema are doing work. (This is not tested as a formal ablation in
  v1; inferred from the subset breakdowns of the primary run.)

**No ablation grid in v1.** If primary LOSO beats ridge and the
subset breakdowns look sensible, ship it. If it misses, ablations
become diagnostic tools in v2.

## Kaggle submission

Exp 107 is a Kaggle deliverable, not just a research experiment.

1. **LOSO validates the architecture** (7 folds + subset report)
2. **Retrain on all data** — focal + 739 labeled SC + unlabeled SC,
   no fold hold-out, LOSO-best hyperparameters, single training run
3. **Package** with Perch v2 ONNX (reused from exp 102/105 Kaggle
   package); exp 107 weights (~4.1M params) + preprocessing
4. **Inference**: Perch v2 → 1536-d embedding → exp 107 model → 234-d
   multi-label probabilities per 5-sec window
5. **Standalone LB submission**
6. **Ensemble with exp 102** — probability-space average, blend
   weight tuned on LOSO against the 739 labels (not on LB). Prior
   hypothesis: exp 107 contributes most to the long tail (non-bird,
   rare, no-focal species); exp 102 dominates on common birds. A
   per-species-group weighted blend may outperform a global weight.

Kaggle's macro AUROC is rank-based, so the per-species Platt
calibration in exp 107 is wasted for standalone LB but is useful for
the ensemble blend step — calibrated probabilities make averaging
across models meaningful.

## Limitations

- **Teacher ceiling on labeled data.** The species bank's upper bound
  is set by how much species-discriminative signal the weak focal
  labels + 739 strong labels actually contain. If focal recording
  metadata is wrong for a species, we can't recover.
- **Domain alignment is hoped, not guaranteed.** The learned projection
  is shaped by labeled data that spans both focal and soundscape.
  Whether it aligns the two domains well enough for retrieval to
  transfer across the gap is the core empirical question.
- **LOSO variance with 7 sites is large.** A fold std of 0.03–0.05 is
  plausible and makes the 0.76 vs 0.749 comparison statistically
  borderline. Fold-level results reported alongside the mean.
- **Single-ecosystem scope.** The projection is trained on Pantanal
  soundscapes + BirdCLEF 2026 focal collection. The model won't
  transfer to other ecosystems without retraining.

## Connection to related experiments

- **Exp 100** (EDA): amphibian-dominated class distribution motivates
  the prototype-based design.
- **Exp 105/106** (Perch pooled + Bayesian): cached Perch embeddings,
  ridge 0.749 baseline, star schema evidence that prototype retrieval
  works in-domain but fails across the focal↔soundscape gap. Direct
  parent of exp 107's thesis.
- **Exp 102** (EfficientNet): current best Kaggle LB 0.794, bird-biased.
  Exp 107 ensembles with it in the submission path — the long-tail
  contribution is the expected value add.
