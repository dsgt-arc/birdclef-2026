# Proposal: Prototype Retrieval for Amphibian-Dominated Soundscapes

**Status**: proposed
**Created**: 2026-04-10 (reframed 2026-04-11)
**Author**: Anthony

## Thesis

BirdCLEF 2026 is amphibian-dominated (exp 100). Perch v2 — the strongest
available acoustic teacher for this project — is a 14,795-species **bird**
classifier and has no output dimension for most of the non-bird species
in the competition taxonomy. Any design that relies on Perch as a
window-level pseudo-labeler silently fails on a large fraction of the
taxonomy.

For species Perch doesn't know, the only mechanism that works is
**species-specific prototypes built from focal recordings**. A prototype
is a small set of acoustic modes in feature space that characterizes
"what species X sounds like" — learned from focal audio where the species
label is known, applied at test time by retrieving soundscape windows
that match the prototype.

Exp 106 established the ceiling of fixed-space prototype retrieval: the
star schema got 0.816 AUROC on the 42 species with focal data but
flatlined on the 28 without focal data, and ridge regression beat it
overall at 0.749 — which is "Perch embeddings + a linear classifier"
and the domain-shift-bottlenecked ceiling for any linear model on Perch's
fixed feature space.

**Exp 107's thesis**: learn a projection that aligns focal and soundscape
audio for the species where we have cross-domain supervision, so that
prototypes in the projection's output space transfer across domains for
**all** species — including those without cross-domain supervision.

Bird species with focal data AND Perch coverage provide the cross-domain
supervision: they appear as focal (Perch-verified per-window labels)
and as soundscape (Perch pseudo-labels + 739 ground-truth labels), and
the shared label in a BCE loss pulls the two sides together in the
projection's output space. Amphibians, insects, and other non-bird
species with focal data contribute prototypes in that same projection's
output space and inherit the domain invariance the bird supervision
funded. Species without focal data rely on the 739 ground-truth labels
alone and are the hardest subset.

**Birds pay for the projection; non-birds ride on it.** That is the
mechanism. The experiment tests whether it works.

## Background

### Perch's taxonomy gap

Perch v2 is trained on Xeno-Canto + Macaulay Library — 14,795 bird
species. BirdCLEF 2026 includes amphibians, insects, and other non-avian
vocalizers outside Perch's output space. Exp 101 (calibration) evaluated
teachers against the 739 labeled soundscape windows and quantified the
coverage gap; Perch-as-pseudo-labeler works only for the bird subset.

### What exp 106 learned

1. **Ridge on Perch embeddings**: 0.749 LOSO AUROC — best linear model,
   domain-shift-bottlenecked.
2. **Star schema cosine**: 0.816 on 42 focal-overlap species, collapses
   on 28 without focal data. Prototype retrieval works where focal data
   exists, fails across the focal↔soundscape domain gap in Perch's fixed
   space.
3. **Site priors**: zero effect — sites are different (Jaccard 0.00–0.25,
   residual cosine -0.68 to +0.89), but site structure is entangled with
   domain shift in Perch's logit dimensions and can't be disentangled
   post-hoc.
4. **127K unlabeled soundscapes**: untapped resource. Usable only with a
   mechanism that extracts species-aware signal from them.

### Four species regimes

The competition taxonomy splits on two axes: does the species have focal
data, and does Perch have the species in its output space?

| Regime | Focal | In Perch | Signal per window |
|--------|-------|----------|-------------------|
| **Bird with focal** | ✓ | ✓ | Perch-verified focal + Perch-pseudo SC + 739 GT |
| **Non-bird with focal** | ✓ | ✗ | Bag-level focal + 739 GT |
| **Bird without focal** | ✗ | ✓ | Perch-pseudo SC + 739 GT |
| **Non-bird without focal** | ✗ | ✗ | 739 GT only (hardest) |

The design must function across all four. The memory-bank architecture
is the choice that makes this possible: a prototype-based classifier
can represent any species that has focal data, whether Perch knows about
it or not, and can be supervised by any combination of label sources
window-by-window.

## Hypothesis

A learnable prototype retrieval model, trained with per-window BCE on
whichever labels exist for each (window, species) pair, where bird
species with focal + Perch coverage provide the dense cross-domain
supervision and non-bird species inherit the resulting domain-aligned
projection, will:

1. Beat exp 106's ridge (0.749 LOSO AUROC) overall, and
2. Close a substantial fraction of the gap on the non-bird subset, where
   prototype retrieval is the only mechanism that can work.

The key mechanistic claim: **once the projection is shaped by bird
cross-domain pairs, non-bird prototypes learned from focal audio alone
retrieve soundscape windows through the same projection.** Birds fund
domain alignment; non-birds benefit structurally.

## Method

### Architecture

A learnable memory-bank classifier operating on cached Perch v2
embeddings. Same prototype-based shape as exp 106's star schema, but
with a **learned projection** instead of identity and **learned
prototypes** instead of fixed k-means centroids.

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
  species memory           site bank
  M_f[family_of]          (21, K, 512)
    + Δ_species
  (F,K,512) + (235,K,512)
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
           scores (235,)
```

- **Projection**: `Linear(1536, 1024) → GELU → Linear(1024, 512) → LayerNorm`.
  Nonlinear because the data regime supports it — the projection is
  shaped by BCE across ~400K+ labeled windows (strong + weak streams),
  not by 739 labels alone.
- **Species memory bank (hierarchical)**: effective species prototype is
  `M_species[s] = M_family[family_of(s)] + Δ_species[s]`, where
  `M_family: (F, K, 512)` holds prototypes shared across all species in
  each family (~30 families from BirdCLEF taxonomy metadata) and
  `Δ_species: (235, K, 512)` holds species-specific refinements. K=8
  slots per bank, captures intra-species acoustic modes (song/call,
  dawn/night, solo/chorus). Weight decay on `Δ_species` keeps
  low-resource species close to their family prior; high-resource
  species specialize away from it. See the "Taxonomy and low-resource
  species" subsection below for the mechanism and rationale.
- **Output dimension = 235**: 234 competition species + 1 explicit
  **silence/background** class. Silence is treated as a species in its
  own single-species family; its prototypes capture acoustic modes of
  silent/background windows (wind, rain, equipment hum, quiet ambient).
  Gives the model a legitimate target for between-call and
  low-energy windows instead of hoping all 234 species outputs go to 0.
- **Site memory bank**: `(21, K, 512)`, K=8 per site. Captures site-level
  acoustic states. K-means initialized on raw Perch unlabeled SC
  embeddings per site.
- **Retrieval projections** `W_Q`, `W_K_species`, `W_K_site` — standard
  Ramsauer et al. formulation of modern Hopfield networks. Lets the
  retrieval space differ from the projection's output space.
- **Aggregation**: `(1/β)·logsumexp_k(β·q·W_K·M_{s,k})` per species
  (with `M_{s,k}` computed as family + delta), then per-species Platt
  `sigmoid(a_s·score + b_s)`. Learnable β per head.
- **Gated fusion**: `score = g·species + (1-g)·species·site_score`,
  with `g = sigmoid(Linear(q))` a learned per-window gate.
- **Total params**: ~4.1M (family bank adds ~120K; species bank is now
  deltas of the same size; breakdown in `design.md`).

### Data and labels: one-time preparation pass

All training data comes from cached Perch v2 embeddings (exp 106). Before
training, a one-time label-preparation pass converts raw focal recording
labels and cached Perch logits into per-window pseudo-labels, stored as
a parquet alongside the embeddings.

| Stream | Source | Labels |
|--------|--------|--------|
| **Strong** | 739 labeled SC | Window-level ground truth (as-is, over 234 species + silence if applicable) |
| **Weak (bird focal)** | Bird species focal recordings | Perch-verified: keep windows where Perch's top-k in the 234 taxonomy contains the recording label above a confidence threshold; multi-hot using Perch's confident predictions. Windows below the RMS silence threshold get the **silence** label instead. |
| **Weak (non-bird focal)** | Non-bird focal recordings | Bag-level: recording label as single positive on all windows. Windows below the RMS silence threshold get the **silence** label *instead of* the recording label — this is where the silence class pays off most, since it removes the "silence = species s" noise from non-bird focal. |
| **Weak (unlabeled SC)** | 127K unlabeled SC | Perch's top-k in taxonomy above confidence threshold. Windows below RMS threshold → silence-positive. Windows above RMS threshold but below Perch confidence → entropy-min stream (might be amphibian/insect that Perch can't label). |
| **Silence (derived)** | All sources, RMS-based | Added in the label-prep pass. A window's RMS energy below a fixed threshold (e.g., −60 dB) produces a positive silence label and zero species labels, regardless of source. |

**RMS-based silence detection (not Perch-based).** Don't use "Perch max
confidence < threshold" as a silence proxy — Perch has low confidence
on amphibian and insect windows too, not just silent ones. Use acoustic
energy directly (windowed RMS or spectral flatness) so silence labeling
doesn't conflate "nothing is calling" with "something Perch can't
identify." The threshold is a small calibration sweep during label prep.

The **Perch-verified filter on bird focal** is the mechanistic fix for
exp 102's focal overfitting pathology: per-window one-hot labels that
include silent windows teach the model "silence = recording label." The
filter drops windows where Perch doesn't confirm the primary species
calling, and the silence class gives the removed-but-low-RMS windows a
legitimate target instead of dropping them entirely.

**Non-bird focal stays bag-level** for calling windows (no teacher to
verify), but silent non-bird focal windows now escape to the silence
class rather than propagating the "silence = species s" pathology. This
is the largest single win from adding the silence label.

### Loss

One loss, three label-weighted terms:

```python
def loss(batch, gamma_weak, lam_ent):
    # Strong stream — 739 labeled SC ground truth
    pred_strong = model(batch.strong.emb, batch.strong.site)
    L_strong = bce(pred_strong, batch.strong.labels)

    # Weak stream — Perch-verified focal + Perch-pseudo unlabeled SC
    pred_weak = model(batch.weak.emb, batch.weak.site)  # site=None for focal
    L_weak = bce(pred_weak, batch.weak.labels)

    # Entropy min on unlabeled SC below Perch's confidence threshold
    pred_ent = model(batch.unconfident.emb, batch.unconfident.site)
    p = pred_ent.clamp(1e-7, 1 - 1e-7)
    L_ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

    return L_strong + gamma_weak * L_weak + lam_ent * L_ent
```

Both `gamma_weak` and `lam_ent` ramp from 0 over the first 5 epochs so
the model trains on clean 739 GT before trusting the weak streams.

**What `L_ent` does.** Binary entropy `H(p) = -p·log(p) - (1-p)·log(1-p)`
is zero when `p` is 0 or 1 and maximal at `p = 0.5`. Minimizing it
pushes each output toward whichever extreme the model is already leaning
toward — a **confidence sharpener**, not a label. For an unlabeled
window where the model has no clear belief about any species, `L_ent`
pushes all outputs toward 0 (the closer extreme for small values); for
a window where the model weakly believes species s is present, it
amplifies that weak belief. Entropy minimization is only applied to the
"unconfident unlabeled SC" stream — windows where Perch couldn't give
us a pseudo-label *and* that don't qualify as silence. Those might
contain amphibians or insects Perch can't identify; entropy min lets
the model commit to its own best guess. It is deliberately ramped from
0 because entropy minimization amplifies whatever bias the model has,
and we want real beliefs to develop before sharpening them.

### Training schedule

**One phase.** AdamW, lr=3e-4, cosine decay, 30–40 epochs.

Memory bank init: k-means on raw Perch focal embeddings per species
(species bank); k-means on raw Perch unlabeled SC embeddings per site
(site bank). Runs once before training starts, uses exp 106's
prototype computation code.

Weak-stream weight `gamma_weak` and entropy weight `lam_ent` ramp from 0
to target over epochs 0–5. Early stopping on LOSO AUROC evaluated every
5 epochs.

No distillation. No separate pretraining phase. One optimizer, one
sampler, one loss.

### Taxonomy and low-resource species

The competition taxonomy is long-tailed. Some species have hundreds of
focal recordings; others have a handful; some have none. K=8 prototype
slots per species would overparameterize low-resource species and leave
zero-focal species entirely unlearnable. The standard "use all windows
+ pad with noise" hack treats the symptom, not the cause.

The architectural fix is to parameterize species prototypes
**hierarchically** as family prototype plus species-specific delta:

```
M_species[s] = M_family[family_of(s)] + Δ_species[s]
```

where `M_family` is shared across all species in a family (~30
families) and `Δ_species` is zero-initialized. Family prototypes
receive gradient from every species in that family, so they train
strongly regardless of any individual species' data count. Weight
decay on `Δ_species` keeps low-resource species close to their family
prior; high-resource species develop strong deltas and specialize.

**What this buys us across the four regimes:**

| Regime | Effect of family parameterization |
|--------|-----------------------------------|
| High-data species | Develops a meaningful `Δ_species[s]`, family prior becomes a background |
| Low-data species | `Δ_species[s]` stays near 0 under weight decay; effective prototype ≈ family mean — structural regularization, not overfitting |
| Zero-focal species with family siblings | `Δ_species[s]` never gets training signal; effective prototype = family prototype. **Partially learnable via siblings' data.** |
| Zero-focal species with no family siblings either | Truly unlearnable; flagged in eval report |

This is a structural fix, not a heuristic. A species with 3 focal
windows doesn't need a special code path — its delta naturally stays
small, and it inherits the family prior. A species with 500 windows
doesn't need to care about the family — its delta dominates.

**Why family level specifically**: family is the tightest taxonomic
grouping with consistent acoustic signatures. Whole frog families share
call morphology; whole warbler families share song structure; insect
families share stridulation mechanisms. Genus is narrower → fewer
per-genus examples → less structural regularization. Class
(bird/amphibian/insect) is too broad to be acoustically meaningful.
Two-level (family + species) is the sweet spot.

**Prerequisite**: taxonomy metadata mapping species → family. BirdCLEF
species lists usually include this via eBird / iNaturalist taxonomy;
the label prep pass verifies it exists before training and pulls a
fallback from an external taxonomy CSV if not.

**Per-regime evaluation reporting.** The evaluation splits results by
data availability rather than just reporting a single overall AUROC.
See the metrics table in the Evaluation section — "with-focal vs
without-focal" and "has family sibling with focal vs truly orphaned"
subset breakdowns localize where the architecture earns its wins and
where it bottoms out.

### LOSO evaluation protocol

7 folds (sites with ≥20 labeled SC windows). Each fold re-runs the full
training from scratch on:
- All focal data (weak stream)
- All unlabeled SC (weak + entropy streams)
- 739 labeled SC **except** the held-out site's windows (strong stream)

Evaluation: held-out site's labeled SC windows.

Final metric: mean of per-fold macro AUROC across the 7 folds, reported
with four subset breakdowns.

## Evaluation

### Metrics

| Metric | Purpose |
|--------|---------|
| Overall LOSO macro AUROC | Primary, directly comparable to exp 106 ridge 0.749 |
| Bird subset AUROC | Did the bird stream work end-to-end? |
| **Non-bird subset AUROC** | **Did domain alignment transfer to species Perch can't label?** |
| With-focal vs without-focal AUROC | Prototype quality scales with focal data availability |
| Has-family-sibling vs orphaned AUROC | Does the family parameterization rescue zero-focal species that have training relatives? |
| Silence class AUROC | Does the silence head actually discriminate quiet windows? Sanity check for the extra output dim |
| Bird focal↔SC projection distance | Diagnostic: shrinking distance between focal(s) and SC-where-Perch-says-s during training = domain alignment is actually working |
| Site × species affinity matrix | Eval-time deliverable for ecological interpretation |

**Per-regime reporting.** Overall AUROC is the headline, but the
subset breakdowns are where the thesis is actually tested. Any species
that has zero training windows (0 focal + 0 labeled SC positives + no
family sibling with focal) is flagged as **unlearnable** and excluded
from the aggregate AUROC — reporting it would inflate or deflate the
headline number with noise. Unlearnable species are listed separately.

### Baselines

| Model | AUROC (LOSO) | Source |
|-------|--------------|--------|
| Ridge (focal emb + EB prior) | 0.749 | Exp 106 (overall) |
| Star schema cosine | 0.816 / collapses | Exp 106 (42 focal / 28 no-focal) |
| Perch top-1 direct | — | Compute as sanity-check baseline in label prep |
| 739 GT only (`γ_weak = 0`) | — | Computed as exp 107 ablation, upper bound on "how much do the 400K weak windows actually help" |

### Success criteria

- **Primary**: overall LOSO AUROC ≥ 0.76. Beats ridge, justifies the
  learned projection.
- **Go/no-go** (cheapest test): overall LOSO AUROC ≥ 0.749 ± 0.02. If
  the full model can't match a linear classifier, the construction
  isn't earning its complexity.
- **Mechanism test**: non-bird subset AUROC substantially above the
  "739 GT only" baseline (quantified after we compute that baseline).
  Evidence that the bird stream's domain alignment actually transferred
  to species Perch can't label.
- **Reject if**: non-bird subset AUROC ≤ bird subset AUROC − 0.15. The
  domain alignment failed to generalize outside the bird class, and the
  memory bank is working only where the teacher could have worked anyway.

### Ablations (post-primary, if primary passes)

| Ablation | Config | Tests |
|----------|--------|-------|
| **Architecture** | Linear classifier on the same learned projection | Does the memory bank beat a linear head, specifically on non-bird species? |
| **Label cleaning** | Raw focal one-hot (skip Perch-verified filter) | Does the Perch filter matter? Connects back to exp 102 focal overfitting — positive result justifies porting to Murilo's pipeline |
| **Weak stream** | `gamma_weak = 0` (train on 739 GT only) | How much do the 400K weak windows actually contribute? |
| **Site bank** | Species-only, no site bank or gate | Does site composition work in the learned space? |
| **Entropy min** | `lam_ent = 0` | Is entropy min on unconfident SC helpful? |
| **Silence class** | Drop silence head (234 classes, no RMS labeling) | Does the explicit silence class help, or is "all species → 0" sufficient? |
| **Family parameterization** | Flat species bank (no `M_family`, `Δ_species` replaced by direct species memory) | Does hierarchical prototyping beat flat prototypes, specifically on low-resource species? |
| **Site classifier aux** | Add small CE loss predicting `site_idx` from the site bank | Does explicit site supervision help or hurt? (Prediction: hurts — encodes artifacts) |
| **Self-training r2** | Use trained bank to pseudo-label non-bird species in unlabeled SC, retrain | Does a second round lift the non-bird subset? (risk: feedback loops) |

## Limitations

- **Bird Perch-verified labels inherit Perch's errors.** Windows where
  Perch misses the primary species are dropped, reducing the effective
  focal dataset for rare birds. Trade-off is explicit: label quality
  up, quantity down.
- **Non-bird focal stays bag-level.** No teacher to clean those labels.
  The architecture absorbs some noise via multi-prototype slots, but
  per-window label noise on non-bird focal is the fundamental cost of
  not having a non-bird teacher.
- **Domain alignment is hoped, not guaranteed,** for non-bird species.
  The projection learned from bird cross-domain pairs has to generalize
  to acoustic features of amphibians and insects. If birds and non-birds
  are too different, the projection may not transfer. Non-bird subset
  AUROC is the empirical test.
- **739 ground-truth windows is small** for training the site bank and
  gate. The strong stream anchors the rest of the model but can only
  directly teach so much; most of the learning on rare species happens
  through the shared projection.
- **LOSO with 7 sites** → high variance in per-fold estimates.
- **The model learns a shared space for this dataset** (Pantanal
  soundscapes + BirdCLEF 2026 focal collection). Won't transfer to
  other ecosystems without retraining.
- **Taxonomy metadata is a prerequisite**. Family-level parameterization
  depends on a clean species → family mapping. If BirdCLEF doesn't ship
  one, the label prep has to pull it from iNaturalist / eBird. Truly
  orphaned species (no focal + no labeled SC + no family sibling with
  training data) are unlearnable regardless of architecture.
- **Silence class is a single bucket**. "Wind," "rain," and "equipment
  hum" all share it. If the silence prototype slots can't cover all
  these modes with K=8, some acoustic noise conditions may leak into
  species predictions. Observable if silence AUROC is low.

## Connection to related experiments

- **Exp 100** (EDA): established the amphibian-dominated class distribution
  that motivates prototype-based design.
- **Exp 101** (calibration): established which teachers work on which
  species. Directly informs the "in Perch" column of the regime table.
- **Exp 102** (EfficientNet): hit focal overfitting from per-window
  one-hot labels. The Perch-verified filter is a mechanistic fix that
  could apply there too — positive exp 107 ablation result would
  justify porting to Murilo's pipeline.
- **Exp 105** (Perch pooled): 0.684 focal cmAP linear probe; infrastructure
  source for cached Perch v2 embeddings.
- **Exp 106** (Bayesian): 0.749 ridge baseline, star schema evidence that
  prototype retrieval works where it has data but fails across domains.
  Direct parent of exp 107's thesis.
