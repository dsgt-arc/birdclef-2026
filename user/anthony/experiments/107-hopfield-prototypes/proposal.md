# Proposal: Prototype Retrieval for Amphibian-Dominated Soundscapes

**Status**: as-built (v4 regularized).
**Created**: 2026-04-10 (axiomatic simplification 2026-04-11, preflight 2026-04-11, partial-pooling 2026-04-11, v2 cross-attention 2026-04-13, v3 contrastive losses 2026-04-13, v4 regularization 2026-04-13).
**Author**: Anthony

> **As-built note.** This document was originally a *proposal*. It has
> been edited into a record of what was actually built. The thesis
> (factored Hopfield retrieval × three data streams × learned projection
> over frozen Perch) is unchanged. Several v1 design details have been
> overridden in v2/v3/v4 — notably the **scalar-gate fusion** (v1) was
> replaced by **species-site cross-attention** (v2), the **softplus-
> bounded unsupervised term** was replaced by **InfoNCE** (v3), and the
> **training schedule + regularization** was rebuilt in v4. Where v1
> prose still appears below, it is annotated with a pointer to the
> as-built override. The source of truth for current training config is
> `scripts/train.py`, `src/birdclef_107/losses.py`, and
> `sbatch/train_loso.sbatch`. Current shipping results, caveats, and
> the LB gap are in §Results log and §Limitations.

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
   fusion) to produce the per-window 234-class multi-label score.
   Low-resource species are regularized toward a family-level shrinkage
   target (hierarchical-Bayes partial pooling) at training time; the
   inference-time retrieval is species-only.

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
substantial fraction of the gap on species where the Perch embedding
already carries signal but per-species focal data is too thin to
shape a flat prototype.

The mechanistic claim: once the projection is learned end-to-end and
both memory banks live in the same learned space, the factored
composition `species × site` that failed in Perch's fixed logit space
becomes meaningful again. Site prototypes trained by Hopfield
clustering on unlabeled data give the fusion a target-domain anchor
without needing pseudo-labels.

## Method

### Architecture

A learnable memory-bank classifier operating on frozen Perch v2
embeddings. All ~3.4M parameters live downstream of the cached
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
    M_species              site memory
  (234, K=8, 512)         (23, K=8, 512)
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
  Nonlinear; shaped by BCE over labeled data. Detached in the site loss
  (see below) so the site bank clusters within the species-shaped space,
  not the other way around.

- **Species memory bank**: `M_species : (234, K=8, 512)`. The **only**
  prototype bank in the forward pass. Per-species retrieval:
  `s_species(q, s) = (1/β) · logsumexp_k (β · q_r · W_K_species · M_species[s, k])`.
  No family-level bank, no joint retrieval, no concatenation. Decisions
  are made from species prototypes alone.

- **Family-level partial pooling (training-time only).** Family
  information sharing happens via a shrinkage regularizer on
  `M_species`, not via a parameterized family bank. At each step, the
  family centroid is computed as a **stop-grad mean** over siblings,
  and each species is pulled toward its family centroid:

  ```
  M_family_mean[f] = stop_grad(mean_{s' ∈ f} M_species[s'])     # (67, K, 512)
  L_family = λ_fam · mean_s || M_species[s] - M_family_mean[f(s)] ||²
  ```

  Classical hierarchical-Bayes partial pooling, implemented as uniform
  attention over siblings. **Zero new parameters.** The `stop_grad`
  makes the shrinkage asymmetric: low-resource species are pulled
  toward the mean, but data-rich species don't feel backpressure from
  noisy low-resource children distorting their prototypes.

  Dynamics across the data regimes: data-rich birds have BCE ≫
  shrinkage and develop their own patterns; medium-resource amphibians
  feel proportionally stronger shrinkage and settle near the family
  mean (which is itself shaped primarily by data-rich siblings);
  orphans converge entirely to the mean. The family hierarchy is a
  training artifact — the inference-time model contains only
  `M_species`, `W_Q`, `W_K_species`, Platt, the site bank, and the gate.

  The 67 families come from iNat taxonomy + one pseudo-family
  `Insecta_unresolved` that groups the 25 `47158son01..25` morphotypes
  (see "Insect morphotype containment" below). 33 of the 67 families
  are singletons; for a singleton species, the family mean is just
  that species itself, so the shrinkage term becomes a no-op (no
  special casing needed — `(M_species[s] - M_species[s]) = 0`).
  22 families have ≥3 species and cover 177/234 = 76% of the taxonomy.
  See `docs/families.csv` for the mapping.

- **Site memory bank**: `(23, K=8, 512)`, K=8 per site. Flat — 23
  Pantanal sites aren't taxonomically grouped. Sites are `S01..S23`,
  matching training filenames; the Kaggle test row_id
  `BC2026_Test_NNNN_S##_<datetime>_<window>` also carries the site
  code, so `site_idx` is parseable at inference.

- **Retrieval projections** `W_Q`, `W_K_species`, `W_K_site` — standard
  Ramsauer formulation of modern Hopfield networks.

- **Platt calibration**: per-species `sigmoid(a_s·s_species + b_s)`.
  Load-bearing: it makes species scores proper probabilities so the
  multiplicative `species · site_score` fusion is a valid Bayesian
  composition. Per-species `b_s` is also the only mechanism that
  differentiates orphans from their data-rich siblings (see
  Limitations).

- ~~**Gated fusion**: `score = g·species + (1-g)·species·site_score`~~
  **OVERRIDDEN in v2.** The scalar gate was deleted — it hurt by
  −0.045 AUROC on S22 and was inconsistent with the factored-star-
  schema thesis (site contributed only a scalar, not per-species
  information). Shipping architecture uses **species-site cross-
  attention** (`SpeciesSiteCrossAttention` in `model.py`): species
  prototypes are refined per window by attending over the site bank,
  with zero-init `out_proj` so the module starts at identity. See
  `docs/concepts/cross-attention-factored-hopfield.md` for the
  evolution note.

- **Output dim**: 234 competition species (no silence class).

- **Total params**: ~5.0M (v4 as-built — projection ~2.1M including
  dropout modules (zero-param), `W_Q`/`W_K_*` ~786K, species bank
  959K, site bank ~95K, cross-attention MHA ~1.05M, family-bank
  derived not stored). Trainable-parameter count is covered by
  `tests/test_model_shapes.py::test_default_model_param_count_is_about_5m`.

### Data streams

Three streams, no filters, no pseudo-labels, no silence class:

| Stream | Source | Count | Labels |
|--------|--------|-------|--------|
| **Strong** | 739 labeled SC | 739 | Window-level GT (234-d multi-hot) |
| **Weak (focal)** | All focal, bird + non-bird, every window, no filter | ~265K | Recording bag label (one-hot on primary species) |
| **Unsupervised** | All 127K unlabeled SC | 127K | No species label — only `site_idx` metadata |

Every focal window keeps its recording's primary-species label,
uniformly across the four regimes (bird/non-bird × has-Perch/no-Perch).
Noise from silent between-call windows is absorbed by the prototype
slots, not by a preprocessing filter. Perch's role in v1 is **only**
the frozen embedding space — it is not a teacher, a filter, or a
pseudo-labeler on any stream.

**The focal stream is structurally bird-biased.** Aves p50 = 169
recordings/species, Amphibia p50 = 6, Insecta p50 = 0 (25 of 28
insects have zero focal data). Per-species balanced sampling (below)
equalizes the *expected frequency* per epoch, but it cannot create
data that doesn't exist. The supervised signal into `M_species` is
therefore dominated by bird acoustic patterns, and the family-level
shrinkage (above) is the main mechanism by which low-resource
amphibians acquire meaningful prototypes — they are pulled toward a
family mean that is itself shaped by their data-rich siblings.

### Loss: three terms

```python
def loss(batch, lambda_site=0.1, lambda_fam=1e-2):
    # 1. Labeled term: strong + weak focal through BCE
    x = concat(batch.strong, batch.weak_focal)
    pred = model(x.emb, x.site_idx)              # site_idx=None for focal
    L_labeled = bce(pred, x.labels)

    # 2. Unsupervised term: Hopfield energy minimization on unlabeled SC.
    # Projection is DETACHED so the site bank clusters within the
    # species-shaped space, not the other way around.
    q_det = model.proj(batch.unsup.emb).detach()
    q_r   = model.W_Q(q_det)
    beta_l = model.log_beta_site.exp()
    M_l    = model.W_K_site(model.site_memory[batch.unsup.site_idx])  # (B,K,D)
    sim    = einsum(q_r, M_l, 'b d, b k d -> b k')
    L_site_unsup = -(torch.logsumexp(beta_l * sim, dim=-1) / beta_l).mean()

    # 3. Family partial pooling: pull each species toward its family
    # centroid, computed as a stop_grad mean over siblings. Zero new
    # parameters; pure regularizer on M_species.
    #   family_idx: (234,) maps species index to family index in [0, 67)
    M_family_mean = scatter_mean(model.M_species, family_idx, dim=0)   # (67, K, D)
    target = M_family_mean[family_idx].detach()                        # (234, K, D)
    L_family = ((model.M_species - target) ** 2).mean()

    return L_labeled + lambda_site * L_site_unsup + lambda_fam * L_family
```

**What each term does:**

- **`L_labeled`**: trains `M_species`, the projection, `W_Q`,
  `W_K_species`, Platt, the gate, and the site bank (via gated fusion
  on labeled SC windows) on the 739 strong + 265K weak focal BCE
  signal. `M_species` is the only retrieval target of the labeled
  supervision.
- **`L_site_unsup`**: minimizes the Hopfield retrieval energy for each
  unlabeled SC window against its **known site's** memory bank. This is
  the same equation used in the forward pass at inference — training
  and inference share the retrieval operator. The loss pulls each
  window toward SOME prototype in its site, which is soft-EM clustering
  in the projection space. Projection is detached so this loss trains
  only the site memory, `W_K_site`, and `log_beta_site` — **not** the
  projection. Without the detach, the loss would push the projection
  to encode site-identifying features (microphone fingerprints, ambient
  noise signatures), which is the exp 106 "site structure entangled
  with domain shift" failure mode we're trying to avoid.
- **`L_family`**: implements hierarchical-Bayes partial pooling on
  `M_species` as a stop-grad shrinkage toward the family centroid.
  At each step, the family centroid is the uniform mean over siblings
  in `f(s)`; detaching it ensures only the LHS moves. Low-resource
  species receive proportionally stronger effective shrinkage (because
  their BCE gradient is sparse), while data-rich species are essentially
  unaffected. Zero new parameters — the family hierarchy lives entirely
  in this loss term and never appears in the forward pass at inference.

**Why this is principled.** The unsupervised site clustering reuses
the Hopfield retrieval machinery already in the architecture. No
pseudo-labels, no entropy minimization, no contrastive pair construction,
no reconstruction head. The site bank learns from 127K windows of
target-domain acoustic data via the same operator that composes with
the species bank at inference time. Family partial pooling is the
textbook hierarchical-Bayes shrinkage estimator, implemented without
any parameterized family prior.

**Hyperparameters**: `λ_site` (default 0.1) and `λ_fam` (default 1e-2),
both tuned on the S22 LOSO fold. No ramps, no warm-up, no `γ_weak`.

### Per-species balanced sampling

The weak focal stream uses a `WeightedRandomSampler` with per-species
weights `1 / n_windows_in_species`. This gives every species equal
representation per epoch and prevents common (bird) species from
dominating the projection gradient. It also keeps the family centroid
honest: without balanced sampling, the `M_family_mean` for large bird
families (Tyrannidae 22, Thraupidae 13, Furnariidae 13) would be pulled
toward a handful of data-rich species — which is exactly the regime
where low-resource siblings need a broad, representative family prior.

The unsupervised SC stream is sampled uniformly (no species identity
concept at this stream level).

### Insect morphotype containment

25 of the 28 Insecta species in the BirdCLEF 2026 taxonomy are
`47158son01..son25`, all sharing iNat id 47158, which resolves to
the **Insecta class**, not a species. These are unresolved
sound-morphotype labels with no family, no genus, and no focal audio.
No taxonomic hierarchy can help them: there is no real family to
share information with, and `M_species[s]` receives no focal gradient.

v1's containment strategy: one pseudo-family `Insecta_unresolved`
groups all 25 morphotypes. Under partial pooling, all 25 species are
shrunk toward their shared family centroid — which is itself the mean
of the same 25 near-zero prototypes. The fixed point is `M_species[s]`
for all 25 species collapsing toward the shared mean of whatever
little signal the 739 strong SC windows + weak bag labels inject.
At inference, all 25 emit effectively **identical** retrieval scores;
per-species Platt `b_s` is the only differentiator, and with no focal
data it's weakly trained at best.

This is containment, not rescue. The alternatives considered —
flat per-species prototypes outside the hierarchy, or acoustic
pre-clustering into synthetic families — don't change the fundamental
problem: no training signal, no learnable per-species prototype. v1
ships with this hole and accepts near-random AUROC on the insect
subset; they are reported separately and can optionally be excluded
from the aggregate metric.

### Initialization — random

- **Projection MLP**: Kaiming
- **M_species**: small Gaussian
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

### Training schedule (v4 as-built)

The v1 schedule (`lr=3e-4`, `weight_decay=1e-5`, 30–40 epochs, no warmup,
no dropout) was replaced across v2/v3/v4 as failure modes were
diagnosed. Current shipping config:

| Knob | Value | Why |
|---|---|---|
| Optimizer | AdamW | unchanged |
| LR | `5e-4`, cosine decay over all training steps | v3: halved from v1's `3e-4` because contrastive losses need slower convergence. |
| Weight decay | **`1e-2`** | v4: 100× increase from v1's `1e-5`. Direct response to v3's 10× L_lab collapse in 3 epochs. Hits attention weights and prototype matrices. |
| Epochs | 20 | v3/v4: gives cosine enough LR range; v4 peak actually at step 600 but patience=10 absorbs the rest. |
| Early stopping | AUROC every 100 steps, patience=10 | v3: intra-epoch val; v1's "every 5 epochs" missed the peak. |
| Strong-stride | **4** | v4: compute strong BCE only every 4 focal steps. Reduces strong-SC memorization (was ~519×/epoch → ~130×). Implemented via `include_strong` kwarg to `compute_loss`. |
| Proj dropout | `0.1` | v4: `nn.Dropout` inside `PerchProjection` MLP (post-GELU). |
| Cross-attn dropout | `0.1` | v4: passed to `nn.MultiheadAttention(dropout=…)`. |
| Label smoothing | `0.01` | v4: small because multi-hot binary labels. Applied to strong + focal BCE. |
| Unsup λ warmup | 0 for steps 0–200, linear 200→500 | v3: lets labeled BCE place the projection before contrastive terms engage. |
| `λ_site`, `λ_xattn` | `0.01` each (split) | v3: decoupled from v2's shared `λ_site=0.1`; scales match labeled BCE at convergence (~0.02). |
| `λ_fam` | `0.01` | pre-registered sweep `{0, 1e-3, 1e-2, 1e-1, 1}` on S22 still pending. |
| Compute | `gpu-rtx6000` RTX 6000 × 1 | unchanged |

Source of truth: `scripts/train.py`, `sbatch/train_loso.sbatch`
env var defaults, `src/birdclef_107/losses.py::compute_loss` signature.

### Taxonomy hierarchy: what it actually buys

The `L_family` shrinkage term is motivated by **one specific regime:
low-resource amphibians**.

- Aves p50 = 169 focal recordings/species — plenty of data; BCE
  gradient ≫ shrinkage gradient, species prototypes train to their
  own patterns, and the family mean is essentially irrelevant to
  them.
- Amphibia p50 = 6 — 28× less data; BCE gradient is sparse, shrinkage
  proportionally dominates, and the effective prototype settles near
  the family mean with small species-specific deviations. This is
  the textbook shrinkage estimator for this regime.

Dominant families where partial pooling actually does work:
Leptodactylidae(16), Hylidae(12), Thraupidae(13), Furnariidae(13),
Tyrannidae(22), Psittacidae(11). Of these, the amphibian families
(Leptodactylidae, Hylidae) are the ones most likely to benefit —
they have many species, per-species data is thin, and the Perch
embedding already carries amphibian-discriminative signal (exp 106
ridge 0.749 on the full taxonomy is proof).

Partial pooling is **not** primarily an orphan-rescue mechanism:
- 28 species have zero focal data.
- Only **3** are rescuable via family siblings with focal data:
  Leptodactylidae, Microhylidae, Phyllomedusidae (amphibians).
- The other 25 are insect morphotypes (above), all in a single
  pseudo-family that provides no meaningful shrinkage target.

For the 3 rescuable orphans, "rescue" is really **family-level
detection, not species-level discrimination**. The orphan converges
to the family centroid, which is shared with its data-rich siblings.
At inference, the orphan's retrieval score is essentially identical
to any low-data sibling's, and only per-species Platt `b_s` (trained
from strong SC co-occurrence) gives any differentiation. This is a
limit of any hierarchical prototype scheme, not a quirk of partial
pooling; no architecture can manufacture species-specific signal
from zero species-specific data.

The main value of the hierarchy is **variance reduction on the
~40–60 medium-resource amphibians** where Perch embeddings carry
signal but focal data is thin. If that variance reduction is small
in practice, the regularizer is dead weight — see the pre-registered
`λ_fam` sweep under Evaluation.

Effect across the four species regimes:

| Regime | Behavior |
|--------|----------|
| Lots of focal data (most birds) | BCE ≫ shrinkage; `M_species[s]` trains to its own pattern; family mean is near-irrelevant. |
| Little focal data + data-rich family siblings (most amphibians in multi-species families) | Shrinkage is proportionally strong; `M_species[s]` settles near the family mean with small deviations. **The regime partial pooling is designed for.** |
| Zero focal + data-rich family siblings (3 amphibians) | `M_species[s]` converges entirely to the family mean. Retrieval is family-level detection; Platt `b_s` is the only per-species differentiator from siblings. |
| Zero focal + no informative family (25 insect morphotypes, singleton-family orphans) | All morphotypes collapse to the shared `Insecta_unresolved` centroid and emit identical retrieval scores. Unlearnable in v1. Flagged and excluded from aggregate AUROC. |

Family level is chosen because whole bird families, frog families, and
insect families typically share acoustic signatures (call morphology,
frequency band, stridulation mechanism). Genus is too narrow; class is
too broad.

## Evaluation

LOSO CV on up to **9 folds** (sites with any labeled SC windows:
S03, S08, S09, S13, S15, S18, S19, S22, S23). Each fold retrains the
full model from scratch on all data except the held-out site's
strong-stream positives. **Every fold uses the identical training
recipe** — same loss, same `λ_fam` (locked after the S22 sweep below),
same sampler, same schedule. The only variable across folds is which
site's labels are withheld.

### Handling fold imbalance

S22 alone = 954/1478 = 65% of all labeled SC windows; the other 8
folds have 30–120 windows each. Naive macro averaging across folds
is dominated by noise from the tiny folds. The strategy is a
decomposition:

- **Pooled AUROC** — **primary metric.** Concatenate held-out
  predictions across all 9 folds into one prediction set, compute
  one AUROC per species, then macro-average across species. Each
  window contributes once; fold size is irrelevant to the weight;
  this is the closest analog to how Kaggle's hidden test set is
  scored. Directly comparable to exp 106 ridge 0.749.
- **S22 as decision fold.** Because S22 alone has 954 windows, its
  per-fold AUROC is statistically stable and is used for all
  hyperparameter tuning (see `λ_fam` sweep below), subset
  diagnostics, and the family ablation. Tuning on S22 — not on the
  pooled metric — prevents the primary metric from becoming an
  eval-overfitting target.
- **Other 8 folds** → variance / generalization check. Report macro
  mean + per-fold table with positive counts, but understand these
  are noisy estimates of unseen-site generalization, not point
  estimates. A fold AUROC from 30 windows has a confidence interval
  wider than any plausible improvement; treat the distribution
  across folds, not the individual numbers.
- **Weighted fold mean** — sanity check. Should agree with the pooled
  AUROC to within noise. Divergence between pooled and weighted
  indicates cross-fold score calibration drift, which should be
  flagged in the report.

### What to report

| Metric | Purpose |
|--------|---------|
| Pooled AUROC over 9 held-out folds | Primary — directly comparable to exp 106 ridge 0.749 |
| S22 fold AUROC + subset breakdown | Decision metric for tuning and ablation |
| Macro mean + per-fold table over 8 small folds | Generalization variance check |
| Bird vs non-bird subset (pooled) | Does the learned projection generalize to non-bird audio? |
| Amphibian subset (pooled) | The regime partial pooling is designed for; main evidence for/against `L_family` |
| With-focal vs without-focal subset (pooled) | Prototype quality scales with focal data availability |
| Insecta subset (reported separately) | Expected near-random; sanity check that morphotypes aren't polluting other metrics |
| Site × species affinity matrix | Eval-time deliverable; ecological interpretation |

Species with no training signal from any source (no focal, no strong
positives, no family sibling with focal) are flagged **unlearnable**
and excluded from the aggregate AUROC; listed separately in the report.

### Pre-registered λ_fam sweep (also the family ablation)

The family partial pooling is the strongest single argument that this
architecture is more than a flat prototype bank. v1 runs a pre-registered
sweep over `λ_fam` on the S22 fold alone, which also serves as the
family ablation:

`λ_fam ∈ {0, 1e-3, 1e-2, 1e-1, 1}`

- `λ_fam = 0` is the **flat ablation** — no family shrinkage, pure
  per-species prototypes. Tests whether the hierarchy earns its
  complexity at all.
- `λ_fam ∈ {1e-3 .. 1}` maps the shrinkage strength curve.

For each setting, track three subset AUROCs on the S22 held-out set:

| Subset | What to watch for |
|--------|-------------------|
| Data-rich birds (Aves with ≥50 focal recordings) | Should be **flat** across `λ_fam`. A drop at high `λ_fam` = over-shrinkage distorting prototypes that have enough data to speak for themselves. |
| Medium-resource amphibians | Should **rise** from `λ_fam=0`, saturate, possibly drop at very high `λ_fam`. Peak is the target. |
| Overall S22 AUROC | Secondary — used only if the subset curves are ambiguous. |

**Decision rule**: pick the smallest `λ_fam` that saturates the
amphibian curve without measurably reducing the bird curve. If
**no setting of `λ_fam > 0` beats `λ_fam = 0` on the amphibian
subset by ≥ 0.5 AUROC points**, v2 drops `L_family` entirely and
redistributes the complexity budget to a richer site structure
(likely a low-rank site→species projection to turn `site_score`
into a per-species prior).

Once `λ_fam` is chosen from the S22 sweep, **lock it** and run the
full 9-fold LOSO with the locked value for the primary pooled-AUROC
report. Do not re-tune on the full LOSO. If the S22 tuning is wrong,
the pooled metric will reflect that honestly.

### Baselines

| Model | LOSO AUROC | Source |
|-------|------------|--------|
| Ridge on Perch embeddings | 0.749 | Exp 106 (overall) |
| Star schema cosine | 0.816 / collapses | Exp 106 (42 focal / 28 no-focal split) |

### Success criteria

- **Primary**: pooled LOSO AUROC ≥ 0.76. Beats ridge, justifies the
  learned projection.
- **Go/no-go**: pooled LOSO AUROC ≥ 0.749. If the full model can't
  match a linear classifier in its own feature space, the construction
  isn't earning its complexity.
- **Mechanism**: non-bird subset AUROC should substantially exceed the
  level a naive "don't use unlabeled SC, no site bank" baseline would
  achieve — evidence that the unsupervised site loss and the star
  schema are doing work. (This is not tested as a formal ablation in
  v1; inferred from the subset breakdowns of the primary run.)

**No ablation grid in v1 beyond the `λ_fam` sweep above.** If the
pooled LOSO beats ridge and the subset breakdowns look sensible, ship
it. Further ablations become diagnostic tools in v2.

## Kaggle submission

Exp 107 is a Kaggle deliverable, not just a research experiment.

1. **Tune `λ_fam` on S22 fold** (pre-registered sweep over
   {0, 1e-3, 1e-2, 1e-1, 1}). Lock `λ_fam` and `λ_site`.
2. **LOSO validates the architecture** — run all 9 folds with the
   locked hyperparameters. Report pooled AUROC + subset breakdown
   + per-fold table. Same training recipe on every fold.
3. **Retrain on all data** — focal + 739 labeled SC + unlabeled SC,
   no fold hold-out, locked LOSO hyperparameters, single training run.
4. **Package** with Perch v2 ONNX (reused from exp 102/105 Kaggle
   package); exp 107 weights (~3.4M params) + preprocessing. Note:
   `L_family` is a training-time loss only — the shipped model
   contains no family-level parameters.
5. **Inference**: parse `site_idx` from row_id
   (`BC2026_Test_NNNN_S##_..._W`) → Perch v2 → 1536-d embedding →
   exp 107 model → 234-d multi-label probabilities per 5-sec window.
6. **OOD-site fallback.** If the hidden rerun uses sites outside
   `S01..S23`, parsing succeeds but the site bank has no prototype.
   Inference path: if `site_idx` ∉ training sites, set
   `site_score = 1` (no site modulation — fall back to pure species
   retrieval). Alternative: marginalize `max_l site_score(q, l)`
   over the 23 training banks. **Decision**: start with the
   `site_score = 1` fallback for simplicity, revisit if first
   submission shows many unknown sites.
7. **Sample-submission sanity check.** First submission should be
   trivial (exp 102 weights + minimal exp 107 forward) to confirm the
   row_id format matches the sample's single placeholder row. Only
   commit to a full exp 107 run after that.
8. **Standalone LB submission.**
9. **Ensemble with exp 102** — probability-space average, blend weight
   tuned on LOSO against the 739 labels (not on LB). Prior hypothesis:
   exp 107 contributes most on low-resource amphibians; exp 102
   dominates on common birds. A per-species-group weighted blend may
   outperform a global weight. **Caveat**: exp 107's long-tail
   contribution is bounded — 25 of 28 orphans are unresolved insects
   with no focal signal and no taxonomic rescue path. Expect exp 107
   to help amphibians and low-resource birds more than insects.

Kaggle's macro AUROC is rank-based, so the per-species Platt
calibration in exp 107 is wasted for standalone LB but is useful for
the ensemble blend step — calibrated probabilities make averaging
across models meaningful.

## Limitations

> **As-built critical assessment (post v4 smoke, 2026-04-13).** The v1
> limitations below are still mostly valid. What follows are new
> limitations surfaced by actually running the thing. Read these first
> before the v1 list — they are what gate shipping.

### Leaderboard gap (honesty check)

Top BC2026 LB scores are around **0.94**. v4 S22 smoke is **0.7613** —
an 18-point gap. "Beats internal ridge by +0.012" and "LB competitive"
are different claims. Things LB-winning entries typically include that
this architecture does **not**:

- **Pseudo-labeling** on unlabeled soundscapes using a strong teacher
  (BirdNET, Perch, or ensemble). Most BirdCLEF winners use some
  variant. exp 107 has 127K unlabeled SC but only runs InfoNCE on
  them — no classification signal.
- **Full Perch fine-tuning.** We freeze Perch and train a 5M
  projection/head; winners typically fine-tune the backbone
  end-to-end with LoRA or similar.
- **SED / attention temporal head.** 5-sec windows are pooled by
  Perch into a single vector; we do no per-frame modeling. Winners
  tend to use frame-level attention to localize calls within windows.
- **Ensembling.** Single-model, single-seed. Winners typically
  average 3–10 models.
- **Audio-level augmentation.** SpecAugment, mixup in the spectrogram
  domain, noise injection. We operate on frozen 1536-d embeddings,
  so these are unavailable without re-running the Perch extractor,
  which dominates compute.
- **Test-time augmentation.** Multi-window averaging, flip/shift
  aggregation. None implemented.

Bin B (Perch pseudo-labels) is the lowest-cost of the missing pieces:
it touches the three-stream loss and stays inside our frozen-Perch
regime. Bin B is a deferred decision, not a planned feature.

### Single-fold, single-seed variance

The +0.012 over ridge is measured on **one** seed, **one** fold
(S22, 477 test windows). A 0.012 delta on 477×234 per-species pooled
AUROC is well inside typical run-to-run variance for this regime.
Before claiming a real gain:

1. **Full 9-fold LOSO** — still pending. Deferred decision.
2. **Seed replicates** (≥3 seeds on S22) — not done.

The pre-registered primary metric is pooled-AUROC across all 9 LOSO
folds, not the S22 number alone. S22 is a screening gate, not the
shipping criterion.

### `without_focal` regression (v3 → v4)

v3 showed `without_focal` (0.7874, n=28) > `with_focal` (0.7015, n=206).
v4 inverted it: `without_focal` (0.6607, n=28) < `with_focal` (0.7688,
n=206). Net positive because birds dominate 69% of the label space and
v4 improved bird AUROC by +0.142 — but the 28 focal-less species (mostly
amphibians and insects that rely on the cross-attention path alone)
lost 0.127 AUROC. Interpretation:

- The v4 regularization budget (attn dropout 0.1, wd 1e-2) is probably
  calibrated for the focal path, not the cross-attn-only path.
- If the hidden LB test set is less bird-heavy than focal suggests, the
  v4 "win" over v3 may invert.
- A follow-up v4.1 with lighter attention dropout (0.05) should be
  considered before declaring v4 as the final shipping config.

### Peak at 2.9% of planned training

v4 best val AUROC was at step 600 out of 20760 planned. Interpretations
we have not distinguished:

1. The model converges very quickly under the current regularization
   and remaining steps are pure noise. (Innocuous.)
2. The regularization is too aggressive and blocks further improvement.
   (Would motivate v4.1 with lighter knobs.)
3. The LR schedule wastes most of its range; a much shorter cosine
   would give the same result in 3 epochs.

Until a seed replicate or a lighter-reg ablation is run, any of these
could be true. The 0.7613 number is an upper bound on what v4 achieves
under the current knobs, not a claim about the architecture's ceiling.

### Original v1 limitations (still applicable)

- **Teacher ceiling on labeled data.** The species bank's upper bound
  is set by how much species-discriminative signal the weak focal
  labels + 739 strong labels actually contain. If focal recording
  metadata is wrong for a species, we can't recover.
- **Domain alignment is hoped, not guaranteed.** The learned projection
  is shaped by labeled data that spans both focal and soundscape.
  Whether it aligns the two domains well enough for retrieval to
  transfer across the gap is the core empirical question. With the
  projection detached from `L_site_unsup`, the only target-domain
  signal shaping the projection is the 739 strong SC windows — thin
  compared to 265K focal windows on the same loss.
- **LOSO is statistically fragile at the per-fold level.** 9 folds,
  but S22 alone is 65% of labeled windows. Small folds have 30–120
  windows each and have AUROC confidence intervals wider than any
  plausible improvement over ridge. The pooled-AUROC primary metric
  mitigates this (each window contributes once regardless of fold
  size), but the per-fold macro table remains a noisy estimate of
  unseen-site generalization rather than a precise number.
- ~~**Site bank is a scalar gate, not a per-species site prior.**~~
  **RESOLVED in v2.** The scalar gate was replaced by species-site
  cross-attention (per-window refined species prototypes). Site bank
  is read as K/V by a 4-head MHA with zero-init `out_proj`. See
  §Architecture above and the `without_focal` regression caveat —
  cross-attention helps with-focal species more than without-focal
  under current regularization.
- **Family partial pooling is a regularizer for a narrow regime.**
  `L_family` helps medium-resource amphibians where per-species focal
  data is too thin (~40–60 species). It provides zero benefit on 33
  singleton-family species (the shrinkage term is identically zero
  for them) and is near-redundant on data-rich birds. If the `λ_fam`
  sweep shows no setting with ≥ 0.5 AUROC gap on the amphibian subset
  vs. `λ_fam = 0`, v2 drops the term entirely.
- **Orphan "rescue" is family-level detection, not species-level
  discrimination.** For the 3 amphibian orphans (Leptodactylidae,
  Microhylidae, Phyllomedusidae), `M_species[s]` converges to the
  family centroid, which is shared with data-rich siblings. At
  inference, the orphan's retrieval score is essentially identical
  to any low-data sibling's, and only the per-species Platt bias
  `b_s` — trained from strong SC co-occurrence on 739 windows —
  differentiates them. This is a limit of hierarchical prototype
  schemes in general, not an architectural bug; no design can
  manufacture per-species discrimination from zero per-species
  data. v2 could add a per-species learned bias vector to give BCE
  a more direct differentiation path; v1 relies on Platt.
- **Insect morphotypes are unreachable in v1.** 25 of 28 Insecta
  species are unresolved sound morphotypes (`47158son01..25`) with
  no focal data, no family, no genus. Under partial pooling, all 25
  collapse to their shared `Insecta_unresolved` centroid and emit
  identical retrieval scores. This is containment, not a solution —
  the model has no supervised signal for these species beyond
  whatever the 739 strong SC windows provide. Expect near-random
  AUROC on this subset.
- **OOD-site risk at Kaggle.** If the hidden rerun uses sites outside
  `S01..S23`, the site bank has no prototype and the inference path
  falls back to species-only. The magnitude of this risk depends on
  information we can't see until submission.
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
  contribution on amphibians is the expected value add.

## Results log

Single-fold (S22) smoke results across architecture iterations. Full
9-fold LOSO and LB numbers not yet obtained; see §Limitations for the
caveats that gate interpreting these.

| Date | Iter | Config | S22 AUROC | Δ vs ridge (0.749) | Notes |
|---|---|---|---|---|---|
| 2026-04-12 | v1 no-gate | ReZero α=0 (species only) | 0.716 | −0.033 | v1 architecture ceiling without the scalar gate |
| 2026-04-13 | v2 | labeled-only cross-attn | 0.618 | −0.131 | cross-attn with 262 SC windows/epoch — overfits hard |
| 2026-04-13 | v2 | + softplus unsup aux | 0.5921 | −0.157 | softplus-bounded `L_site` + `L_xattn_softplus` both fail: saturate/collapse |
| 2026-04-13 | v3 | InfoNCE, 3 ep | 0.6958 | −0.053 | contrastive losses fix saturation; LR-bounded |
| 2026-04-13 | v3 | InfoNCE, 20 ep | 0.7074 | −0.042 | peak ep 2, regressed — labeled overfit diagnosed |
| 2026-04-13 | **v4** | **+ reg (drop+wd+stride+smoothing)** | **0.7613** | **+0.012** | peak step 600 of 20760; `bird` +0.142, `without_focal` **−0.127** |

**Key per-subset breakdown at v4** (S22, 477 test windows):

| Subset | n species | AUROC |
|---|---|---|
| bird | 162 | **0.8683** |
| amphibian | 35 | 0.6529 |
| insect | 28 | 0.0 (no positives in fold) |
| with_focal | 206 | 0.7688 |
| without_focal | 28 | 0.6607 |

The bird subset carries the +0.012 overall gain. Non-bird subsets are
at or below the v3 baseline. See §Limitations "`without_focal` regression"
for interpretation.
