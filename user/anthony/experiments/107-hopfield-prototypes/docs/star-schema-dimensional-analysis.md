# Star Schema: Tensor Framework

A formal description of the tensor structure behind exp 107's prototype
retrieval model. The model is a factored classifier over **species ×
site**, with prototype memory banks as learned entity embeddings and
bilinear compositions as scoring functions. Family-level sharing is
handled via training-time partial pooling, not a parameterized bank.
This document specifies the framework; see `../design.md` for the
training procedure and implementation and `../proposal.md` for the
experimental thesis.

## Why the star schema

Exp 107's thesis (see `../proposal.md`) is that BirdCLEF 2026 is
amphibian-dominated and Perch — the strongest available acoustic teacher —
is a bird-only classifier. For species outside Perch's output space,
prototype retrieval from focal recordings is the only mechanism that
works. The star schema formalizes this:

- Each **species** (234 BirdCLEF targets) gets a prototype set. This
  is the only memory bank in the species forward pass.
- Each **site** (23 Pantanal monitoring locations, `S01..S23`) gets a
  prototype set.
- Per-window scoring is a bilinear retrieval over a shared
  D-dimensional space where both memory banks live.
- Family-level information sharing (**67 families**, including a
  pseudo-family `Insecta_unresolved` for the 25 unresolved insect
  morphotypes) is implemented as a **training-time shrinkage
  regularizer** on `M_species`, not as a second memory bank.

**Family is first-class in the loss, not in the forward pass.**
Species prototypes at inference are `M_species[s]` — one tensor per
species — and retrieval is flat. At training, each species is pulled
toward its family centroid (computed as a stop-grad uniform mean over
siblings) by an additional `L_family` loss term. This is classical
hierarchical-Bayes partial pooling, adds zero parameters, and is
dropped from the shipped Kaggle model entirely.

The decomposition matters because it's the structural fix for
low-resource species: data-rich siblings shape the family centroid
through BCE, and then the shrinkage pulls low-resource siblings toward
that centroid without them ever needing their own training data.
High-resource species have BCE gradient ≫ shrinkage gradient and
develop their own prototypes; low-resource species have sparse BCE
and settle near the centroid with small species-specific deviations.
Orphans (zero focal data) converge entirely to the centroid.

The star schema's factored structure lets the architecture represent
any species that has focal data whether or not Perch knows the species,
gives species without focal data a fallback path via their family's
siblings, and exposes a natural composition (species × site) for
eval-time ecological interpretation.

## Entities

| Symbol | Name | Size | Description |
|--------|------|------|-------------|
| S | Species | 234 | BirdCLEF 2026 targets |
| F | Families | 67 | Taxonomic families (iNat taxonomy + `Insecta_unresolved` pseudo-family for 25 morphotypes) — used only at training time by `L_family` |
| L | Sites | 23 | Pantanal monitoring locations `S01..S23` |
| K_s | Species prototype slots | 8 | Soft-EM components per species |
| K_l | Site prototype slots | 8 | Soft-EM components per site |
| D | Shared space | 512 | Learned projection dimension (where all memory banks and queries live) |
| H | MLP hidden | 1024 | Nonlinear projection hidden layer |
| E | Perch embeddings | 1,536 | Pretrained input space (frozen) |
| W_sup | Labeled windows | ~266K | Strong (739) + weak focal (~265K) for supervised BCE |
| W_unsup | Unlabeled SC windows | 127K | For unsupervised site Hopfield energy loss |

Every entity lives in a single shared D-dimensional space. Relationships
between entities are bilinear forms through D, with explicit `W_Q` and
`W_K` projections on the retrieval side (standard Ramsauer formulation).

**Species memory is flat** at inference — one tensor per species, no
family-level concatenation. Family-level information sharing is
implemented at training time via the `L_family` partial-pooling loss
(stop-grad shrinkage toward the uniform mean over siblings). Family
is the entity that makes the shrinkage meaningful; genus and class
are deliberately absent (genus has too few examples per group, class
is too broad to be acoustically meaningful).

## Tensor representations

**Species memory** — `M_species ∈ ℝ^{S × K_s × D}`

Each species s has K_s prototype vectors in the shared space. This is
the only species-side memory bank; retrieval runs directly over
`M_species` with no family-level decomposition at inference. Trained
by the labeled BCE loss and regularized at training time by the
`L_family` partial-pooling shrinkage term (below).

The slots are unlabeled soft-EM components — training assigns each
window to a softmax mixture over the K prototypes, and the slots
specialize to whatever modes the data demands (song vs call, dawn vs
nocturnal, solo vs chorus). K_s = 8 is enough capacity without
over-parameterizing the low-resource tail.

**Derived family centroid** (training-time only) — `M_family_mean ∈ ℝ^{F × K_s × D}`

Not a parameter. Computed at each training step as the uniform mean
over siblings:

    M_family_mean[f, k, :] = (1 / |f|) · Σ_{s: family_of(s)=f} M_species[s, k, :]

where `family_of : ℕ_S → ℕ_F` is a static species → family lookup
from `docs/families.csv` (iNat taxonomy + the `Insecta_unresolved`
pseudo-family). This centroid is **detached** from the computation
graph (`stop_grad`) so it serves only as a fixed shrinkage target
for `M_species`; it does not appear in the forward pass and is not
saved in the model checkpoint.

**Site memory** — `M_l ∈ ℝ^{L × K_l × D}`

Each site l has K_l prototype vectors capturing acoustic states.
Unlabeled — training discovers site states (dawn chorus, midday quiet,
rain event, insect wall, etc.) from the unlabeled soundscape data plus
the 739 labeled SC gradients, rather than inheriting named modes from
upstream clustering. The site bank is **flat**, not hierarchical: there
is no "family of sites" abstraction that would make sense for 23
Pantanal monitoring locations.

**Window query** — `q_w ∈ ℝ^D`

A window projected into the shared space: `q_w = proj(emb_w)` where
`proj : ℝ^E → ℝ^D` is a nonlinear 2-layer MLP with GELU + LayerNorm.

## Bilinear compositions

All compositions are bilinear forms through the shared D dimension.

**Species scoring** (what it sounds like):

    score_species(w, s) = σ(a_s · (1/β_s) · logsumexp_{k=1..K_s}(β_s · q_w · W_K_s · M_species[s, k]) + b_s)

Retrieval is flat — the species bank is read directly, no family-level
composition at inference.

Shape: `(W, D) @ (S·K_s, D)^T → (W, S·K_s) → reshape + lse over K → (W, S)`.
The logsumexp is the modern Hopfield retrieval energy — a differentiable
soft-max that aggregates across K slots. The affine `(a_s, b_s)` is
per-species Platt calibration. `β_s` is a learnable global temperature
that controls how sharply one slot dominates.

Per-species Platt `b_s` is also the only mechanism that differentiates
orphans from their data-rich siblings once partial pooling has pulled
them onto the same family centroid (see "Orphan collision limit"
below).

**Site scoring** (where it is):

    score_site(w, l_w) = σ((1/β_l) · logsumexp_{k=1..K_l}(β_l · q_w · W_K_l · M_l[l_w, k]))

Where `l_w` is the known site of window w. This scores how typical the
window is for its site — high = expected acoustic state, low = unusual
(rare event, equipment anomaly, hard window).

**Species × Site affinity** (what's expected where):

    A(s, l) = (1/β) · logsumexp_{k_s, k_l}(β · (W_K_s · M_species[s, k_s]) · (W_K_l · M_l[l, k_l]))

Shape: `(S·K_s, D) @ (L·K_l, D)^T → (S·K_s, L·K_l) → reshape + lse → (S, L)`.

A static compatibility matrix computed directly from the learned memory
banks — no window input needed. It encodes which species the model has
learned to expect at which sites and is the eval-time deliverable for
ecological interpretation.

The key property of the learned space: this bilinear form is meaningful
because both `M_species` and `M_l` live in the same D, shaped by a
projection that was jointly supervised by labeled BCE (species side) and
the unsupervised Hopfield energy loss on unlabeled soundscapes (site side).
Exp 106's analogous composition failed because species and site
prototypes lived in incompatible regions of Perch's fixed logit space.

**Composed window score** (posterior):

    score(w, s) = g · score_species(w, s)
                + (1 - g) · score_species(w, s) · score_site(w, l_w)

where `g = σ(gate(q_w)) ∈ [0, 1]` is a learned per-window gate. When site
context isn't informative, g → 1 and the model falls back to species-only
scoring. The gate is conditional on the query, not global — different
windows can use site context to different degrees.

**Unsupervised site clustering** (the novel training loss):

For each unlabeled soundscape window `w_u` with known site `l(w_u)`, the
**Hopfield retrieval energy** is minimized against the window's own site
memory bank:

    L_site_unsup(w_u) = -(1/β_l) · logsumexp_{k=1..K_l}(β_l · q_r · W_K_l · M_l[l(w_u), k])

where `q_r = W_Q · proj(emb_{w_u}).detach()`. The projection is detached
from the gradient — this term trains **only** `M_l`, `W_K_site`, and
`log_β_l`, not the projection itself.

This uses the same retrieval equation (`logsumexp_k(β·q·W_K·M)`) as the
forward-pass site scoring. Minimizing the negative logsumexp is the
Hopfield energy function, which pulls each window toward **some**
prototype in its site's memory bank via the same soft-EM gradient
identity derived below. In recommendation-system terms, this is
**unsupervised clustering of target-domain windows into per-site
prototype sets**, supervised by the known `site_idx` metadata.

Why projection detachment matters: if the projection were trained by
this loss, it would encode site-identifying features (microphone
fingerprints, ambient noise signatures, recording equipment biases).
That's the exp 106 failure mode — site structure entangled with domain
shift. Detaching keeps the projection shaped by species supervision
(via `L_labeled`), while letting the site bank adapt its prototypes to
cluster within that species-shaped space.

**Family partial pooling** (hierarchical-Bayes shrinkage, training-time only):

    L_family = λ_fam · (1 / (S · K_s · D)) · Σ_{s, k, d} ( M_species[s, k, d] - stop_grad(M_family_mean[family_of(s), k, d]) )²

where `M_family_mean` is the uniform mean over siblings defined above.
The stop_grad makes the shrinkage asymmetric: only `M_species[s]` moves
toward the target; the target itself (being a function of `M_species`)
is treated as fixed from the gradient's point of view. At the next step
the target is recomputed from the updated `M_species`, and the process
iterates. The fixed point is a partial-pooling equilibrium where
data-rich species sit far from the centroid (BCE dominates shrinkage),
low-resource species sit near the centroid, and zero-data species sit
exactly at the centroid.

The shrinkage is a pure regularizer on `M_species` — it adds no
parameters and does not touch any other tensor. There is no
"family bank" in the computation graph.

Training signal routing:

| Parameter | Labeled BCE (`L_labeled`) | Unsupervised site energy (`L_site_unsup`) | Family partial pooling (`L_family`) |
|-----------|---------------------------|-------------------------------------------|-------------------------------------|
| `proj` MLP | ✓ | ✗ (detached) | ✗ |
| `W_Q` | ✓ | ✓ | ✗ |
| `W_K_species`, Platt | ✓ | ✗ | ✗ |
| `M_species` | ✓ | ✗ | ✓ (LHS only; target is stop_grad) |
| `W_K_site`, `M_site`, `log_β_l` | ✓ (via gated fusion on labeled SC) | ✓ | ✗ |
| `gate` | ✓ | ✗ | ✗ |

The site bank therefore gets gradient from two complementary sources:
the labeled stream routes species × site composition error into the
site bank through the gated fusion, and the unsupervised stream routes
per-window Hopfield clustering into the same bank from 127K unlabeled
target-domain windows. These two signals cooperate: labeled data says
"prototype `M_l[l, k]` should activate when species `s` is present at
site `l`" and unsupervised data says "prototype `M_l[l, k]` should fit
the typical acoustic modes observed at site `l`."

`M_species` gets gradient from BCE (pulling each species toward its
own positive windows) and from `L_family` (pulling each species
toward its family centroid). The two forces balance per-species based
on data volume — data-rich species feel mostly BCE, low-resource
species feel proportionally more shrinkage.

### Orphan collision limit

Partial pooling does not solve — and cannot solve — the zero-data
species problem. Two species `s`, `s'` in the same family with no
focal gradient on either converge to `M_species[s] = M_species[s'] =
M_family_mean[f]`, emitting **identical** retrieval scores. Only
per-species Platt `b_s` (trained from strong SC co-occurrence on
739 windows) can differentiate them. For the 3 rescuable amphibian
orphans (Leptodactylidae, Microhylidae, Phyllomedusidae) this means
"rescue" = family-level detection shared with data-rich siblings,
not species-level discrimination. For the 25 Insecta morphotypes in
`Insecta_unresolved`, all 25 collapse to one centroid and the
subset is effectively unreachable in v1.

This is not an architectural bug — it's a mathematical consequence
of "no per-species signal, no per-species prototype." No hierarchical
prototype scheme can manufacture discrimination from zero data.

## Dimensional summary

| Operation | Input shapes | Contracted dim | Output shape |
|-----------|--------------|----------------|--------------|
| Projection (MLP) | `(W, E) → (W, H) → (W, D)` | E, H | `(W, D)` |
| Query transform | `(W, D) @ W_Q` | D | `(W, D)` |
| Species key transform | `(S, K_s, D) @ W_K_s` | D | `(S, K_s, D)` |
| Site key transform | `(L, K_l, D) @ W_K_l` | D | `(L, K_l, D)` |
| Species scoring | `(W, D) × (S, K_s, D)` | D | `(W, S)` |
| Site scoring | `(W, D) × (L, K_l, D)` | D | `(W,)` |
| Species × Site affinity | `(S, K_s, D) × (L, K_l, D)` (post-W_K) | D | `(S, L)` |
| Gated composition | `(W, S) × (W,) × (W,)` | — | `(W, S)` |
| Family centroid (training only) | scatter-mean `(S, K_s, D) → (F, K_s, D)` | S | `(F, K_s, D)` |
| Family shrinkage loss | `(S, K_s, D) - (S, K_s, D)` (gathered + detached) | — | scalar |

Total parameters:

| Component | Shape | Parameters |
|-----------|-------|-----------|
| Projection MLP | `(E, H) + (H, D) + LN` | ~2.1M |
| `W_Q` | `(D, D) = (512, 512)` | 262K |
| `W_K_species` | `(D, D) = (512, 512)` | 262K |
| `W_K_site` | `(D, D) = (512, 512)` | 262K |
| Species memory `M_species` | `(S, K_s, D) = (234, 8, 512)` | 959K |
| Site memory | `(L, K_l, D) = (23, 8, 512)` | 94K |
| Platt calibration | `2 · S = 468` | 468 |
| Hopfield temperatures | `β_s, β_l` | 2 |
| Gate | `(D, 1) + bias = 513` | 513 |
| **Total** | | **~3.4M** |

No family-level parameters. `M_family_mean` is derived on the fly at
each training step from `M_species` and is used only to compute the
`L_family` shrinkage term. The shipped Kaggle checkpoint contains
only the tensors listed above.

Weight decay: uniform `1e-5` on all parameter groups. The `L_family`
term (see above) is the mechanism by which low-resource species get
pulled toward a prior; it replaces the earlier design's weight-decay
asymmetry on a "species delta" tensor.

## Logsumexp over K as soft EM (Hopfield retrieval)

The full mixture marginal over species modes is:

    P(x | s) = Σ_k π_k P(x | s, k)

The model approximates this with the modern Hopfield retrieval energy:

    score_s(x) = (1/β) · logsumexp_k(β · q_r · W_K · M_s[s, k])

This is the Ramsauer et al. 2020 energy function and also the
temperature-β soft-max approximation to the hard maximum. It interpolates
between two limits:

- **β → ∞**: logsumexp → max_k (hard EM, winner-take-all). Each window
  routes to one prototype per species; other slots receive zero gradient.
- **β → 0**: logsumexp → mean_k + log(K)/β. All slots contribute equally;
  the mixture has no sharpness.

At finite β, all K prototypes receive gradient proportional to their
softmax attention weight `softmax_k(β · q_r · W_K · M_{s,k})`. This is
the key property that lets us set K abstractly (8 slots): unused
prototypes don't cause dead-slot pathologies, they just receive small
gradients and drift toward the nearest active attractor. Training
discovers how many modes each species actually needs.

β itself is learnable per head (`log_beta_species`, `log_beta_site`). The
model picks its own sharpness — a species with a single dominant mode
drives β large, a species with many subtly different calls keeps β small.

### The soft-EM gradient identity

SGD on the logsumexp retrieval loss is *exactly* a stochastic soft EM
update on the memory bank. Take the gradient of the aggregated species
score with respect to a prototype:

    s(q) = (1/β) · logsumexp_k(β · q · M_{s,k})

    ∂s/∂M_{s,k} = softmax_k(β · q · M_{s,k}) · q
                = r_{s,k}(q) · q

where `r_{s,k}(q) = softmax_k(β · q · M_{s,k})` is the soft responsibility
of prototype k for window q under species s. This is the E-step and the
M-step rolled into one:

- **E-step** (implicit): `r_{s,k}(q)` is computed during the forward pass
  by the softmax inside the logsumexp gradient.
- **M-step** (implicit): SGD moves `M_{s,k}` toward `q`, scaled by
  `r_{s,k}(q)` — exactly the weighted-centroid update of soft EM.

The sigmoid + BCE on top (on labeled data) adds a logistic reweighting
per (window, species) pair: the effective gradient on `M_{s,k}` is
scaled by `(y_{s} − σ(a_s · s(q) + b_s))`, so well-predicted windows
contribute less and misclassified windows contribute more. BCE shaping
focuses the EM procedure on hard cases without breaking the EM structure.

**The unsupervised site loss uses the same identity.** Minimizing
`-score_l(q) = -(1/β_l) · logsumexp_k(β_l · q · W_K_l · M_l[l_w, k])`
on an unlabeled window with known site `l_w` pulls the site prototypes
toward the window's projected query `q` with soft responsibilities
`r_{l_w, k}(q)`. This is unsupervised soft-EM clustering of target-domain
windows per site — the same E/M update that labeled BCE performs for
species, applied to the site bank with unlabeled data. Unlike labeled
BCE, there is no logistic reweighting term because there is no label
to compare against; the loss unconditionally pulls prototypes toward
observed windows of that site.

**Consequence for initialization.** Because gradient descent is already
doing soft EM on both the labeled BCE gradient and the unsupervised
Hopfield energy loss, **k-means warm-start is not necessary**. Exp 107
initializes both `M_species` and the site bank from random Gaussian
noise, and gradient descent performs the soft-EM clustering itself
over the course of training. K-means init in raw Perch space, followed
by a projection through an untrained MLP, would partially destroy the
cluster structure anyway — random init is strictly cleaner.

Consequence: the species bank operates as a soft mixture model over K
learned components; the site bank operates as a soft mixture model
trained jointly by labeled BCE (via gated fusion) and the unsupervised
Hopfield energy loss. Mode discovery emerges as a side effect of
training. No separate clustering step, no fixed mode count, no need to
name the modes up front.

## Recommendation-system framing

The star schema is a factored recommendation model over **species and
sites**, with categorical (family) shrinkage on item embeddings and
contextual gating over user (site) state:

| Recsys concept | Exp 107 analog |
|----------------|----------------|
| Items | Species (S = 234) |
| Item categories | Families (F = 67) — items share category-level priors via training-time shrinkage, not a second embedding table |
| Users | Sites (L = 23) |
| Item embeddings | Species memory `M_species` (multi-prototype, flat) |
| Category prior | Derived family centroid `M_family_mean[f]` = mean of sibling `M_species` rows; stop_grad; no parameters |
| User embeddings | Site memory `M_l` (multi-prototype) |
| Query | Window embedding `q_w` |
| Bilinear score | `q · W_K · M_species` through shared D |
| Strong feedback | 739 labeled SC windows (window-level ground truth) |
| Weak feedback | Focal recording bag labels (per-recording, noisy per-window) |
| Implicit feedback | 127K unlabeled SC windows — "impressions" with no species label, trained via unsupervised Hopfield clustering against their known site's prototype bank |
| Cold-start handling | Zero-focal species converge to their family centroid under `L_family` shrinkage (classical hierarchical-Bayes partial pooling); Platt `b_s` provides weak per-species differentiation from 739 strong SC co-occurrences |

The species × site affinity matrix `A(s, l)` is the recovered analog of
a classical user-item matrix factorization. Rather than fitting it
directly against observed counts (sparse and noisy at 739 labeled
windows), it is recovered from the learned per-entity embeddings —
shaped by labeled BCE on the species side and by the unsupervised
Hopfield energy loss on the site side.

## Modern Hopfield framing

Modern Hopfield networks (Ramsauer et al. 2020) generalize classical
associative memory by replacing discrete patterns with continuous vectors
and the local update rule with scaled dot-product attention. The update

    q_{t+1} = softmax(β · q_t · M^T) · M

converges to the stored pattern closest to the initial query, with
exponential storage capacity in D.

Each memory bank in exp 107 is a Hopfield store: the species bank holds
flat patterns `M_species[s]` (one per species, 8 slots each), and the
site bank holds flat per-site prototypes `M_l[l_w]` for the current
window's site. Family-level sharing is implemented at training time
via the `L_family` partial-pooling shrinkage — a regularizer on the
species store that pulls low-resource rows toward the mean of their
siblings without altering the Hopfield retrieval operator itself. The
shrinkage is analogous to **random-effects pooling in a hierarchical
linear model**: the prior (family centroid) is informed by data-rich
siblings and shapes the posterior (individual species prototypes)
with strength inversely proportional to per-species data volume.
Retrieval dynamics and storage structure are unchanged — only the
training objective is augmented.

Exp 107 differs from canonical associative-memory Hopfield in three
ways: (1) **one-shot retrieval, no iteration** — multi-label scoring
doesn't converge to a single attractor; (2) **two-stage aggregation** —
softmax within a species' K slots, sigmoid across species, so species
don't compete for attention mass; (3) **per-species Platt calibration**
— learnable affine before the sigmoid makes species scores proper
probabilities, load-bearing for the Bayesian `species · site`
multiplicative fusion.

The Hopfield framing is both a mechanism and a training objective: the
same energy function that scores species retrieval at inference is
minimized on unlabeled soundscape windows against their sites' memory
banks (see "Unsupervised site clustering" above), turning self-supervised
site clustering into a direct application of Ramsauer et al.'s energy
formulation. The experiment's core hypothesis is that a **learned**
projection, trained by this dual supervised/unsupervised Hopfield
system, dissolves the domain-shift entanglement that made exp 106's
fixed-space star schema fail.
