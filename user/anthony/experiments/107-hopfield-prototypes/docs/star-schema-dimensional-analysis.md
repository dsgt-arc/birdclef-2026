# Star Schema: Tensor Framework

A formal description of the tensor structure behind exp 107's prototype
retrieval model. The model is a factored classifier over **family × species
× site**, with prototype memory banks as learned entity embeddings and
bilinear compositions as scoring functions. This document specifies the
framework; see `../design.md` for the training procedure and implementation
and `../proposal.md` for the experimental thesis.

## Why the star schema

Exp 107's thesis (see `../proposal.md`) is that BirdCLEF 2026 is
amphibian-dominated and Perch — the strongest available acoustic teacher —
is a bird-only classifier. For species outside Perch's output space,
prototype retrieval from focal recordings is the only mechanism that
works. The star schema formalizes this:

- Each **family** (~30 taxonomic families) gets a prototype set, shared
  across all species in that family
- Each **species** (235, including silence) gets a prototype set
  parameterized as `family prototype + species-specific delta`
- Each **site** (21 Pantanal monitoring locations) gets a prototype set
- Per-window scoring is a bilinear retrieval over a shared
  D-dimensional space where all three memory banks live

**Family is a first-class entity** in this schema, not a supplementary
concept. Species prototypes are not independent — they are hierarchically
parameterized as `M_species[s] = M_family[family_of(s)] + Δ_species[s]`.
This matters because it's the structural fix for low-resource species:
the family memory pools gradient from every species in the family, so
even a species with three focal recordings benefits from hundreds of
sibling windows via the shared family prototype. High-resource species
develop strong `Δ_species[s]` and specialize; low-resource species stay
near `Δ = 0` under weight decay and inherit the family prior.

The star schema's factored structure lets the architecture represent
any species that has focal data whether or not Perch knows the species,
gives species without focal data a fallback path via their family's
siblings, and exposes a natural composition (species × site, recoverable
per-family as well) for eval-time ecological interpretation.

## Entities

| Symbol | Name | Size | Description |
|--------|------|------|-------------|
| S | Species (incl. silence) | 235 | 234 BirdCLEF 2026 targets + 1 silence/background class |
| F | Families | ~30 | Taxonomic families (from BirdCLEF species metadata) + 1 silence "family" containing only the silence class |
| L | Sites | 21 | Pantanal monitoring locations |
| K_s | Species / family prototype slots | 8 | Soft-EM components per species / per family |
| K_l | Site prototype slots | 8 | Soft-EM components per site |
| D | Shared space | 512 | Learned projection dimension (where all memory banks and queries live) |
| H | MLP hidden | 1024 | Nonlinear projection hidden layer |
| E | Perch embeddings | 1,536 | Pretrained input space |
| W | Windows | ~330K | Strong + weak + unconfident training windows |

Every entity lives in a single shared D-dimensional space. Relationships
between entities are bilinear forms through D, with explicit `W_Q` and
`W_K` projections on the retrieval side (standard Ramsauer formulation).

**Species memory is hierarchical** — species prototypes are parameterized
as family prototype plus species-specific delta (see "Tensor
representations" below). This lets low-resource species inherit their
family's prototypes as a prior while allowing high-resource species to
specialize. Family is the entity that makes this decomposition possible;
genus and class are deliberately absent (genus has too few examples per
group, class is too broad to be acoustically meaningful).

## Tensor representations

**Family memory** — `M_f ∈ ℝ^{F × K_s × D}`

Each family f has K_s prototype vectors in the shared space, pooled
across all species in the family. Shared parameters: every species in
family f reads from the same `M_f[f]` row. Family memory gets gradient
from **every species in the family**, so it accumulates training signal
from all sibling species jointly — dense training from all families'
combined data.

**Species delta memory** — `Δ_s ∈ ℝ^{S × K_s × D}`

Each species s has K_s correction vectors, zero-initialized, that add
onto the family prototype to produce the species-specific stored
pattern. Weight decay (1e-2, vs 1e-5 elsewhere) shrinks these toward
zero for low-resource species — they can't generate enough BCE
gradient to counteract the shrinkage, so they stay at Δ ≈ 0 and the
effective species memory collapses onto the family prior. High-resource
species overcome the shrinkage and develop strong deltas that specialize
them away from the family template.

**Effective species memory** — `M_s[s, k] = M_f[family_of(s), k] + Δ_s[s, k]`

Computed on every forward pass. `family_of : ℕ_S → ℕ_F` is a static
species → family lookup from the BirdCLEF taxonomy metadata. The
Hopfield retrieval runs over this effective memory; the family / delta
decomposition is invisible at inference time and only shapes where
gradient flows and how weight decay regularizes.

The slots are unlabeled soft-EM components — training assigns each
window to a softmax mixture over the K prototypes, and the slots
specialize to whatever modes the data demands (song vs call, dawn vs
nocturnal, solo vs chorus). K_s = 8 is enough capacity without
over-parameterizing the low-resource tail.

**Site memory** — `M_l ∈ ℝ^{L × K_l × D}`

Each site l has K_l prototype vectors capturing acoustic states.
Unlabeled — training discovers site states (dawn chorus, midday quiet,
rain event, insect wall, etc.) from the unlabeled soundscape data plus
the 739 labeled SC gradients, rather than inheriting named modes from
upstream clustering. The site bank is **flat**, not hierarchical: there
is no "family of sites" abstraction that would make sense for 21
Pantanal monitoring locations.

**Window query** — `q_w ∈ ℝ^D`

A window projected into the shared space: `q_w = proj(emb_w)` where
`proj : ℝ^E → ℝ^D` is a nonlinear 2-layer MLP with GELU + LayerNorm.

## Bilinear compositions

All compositions are bilinear forms through the shared D dimension.

**Species scoring** (what it sounds like):

    score_species(w, s) = σ(a_s · (1/β_s) · logsumexp_{k=1..K_s}(β_s · q_w · W_K_s · M_s[s, k]) + b_s)

where `M_s[s, k] = M_f[family_of(s), k] + Δ_s[s, k]` (the effective
species memory; the sum is done inside the forward pass, before the
W_K_s projection).

Shape: `(W, D) @ (S·K_s, D)^T → (W, S·K_s) → reshape + lse over K → (W, S)`.
The logsumexp is the modern Hopfield retrieval energy — a differentiable
soft-max that aggregates across K slots. The affine `(a_s, b_s)` is
per-species Platt calibration. `β_s` is a learnable global temperature
that controls how sharply one slot dominates.

The hierarchical decomposition is invisible at this level — the
Hopfield retrieval treats `M_s[s, k]` as a standalone stored pattern.
Its effect is only in gradient flow (family memory gets signal from all
sibling species jointly; species deltas only from their own species)
and in regularization (weight decay collapses low-resource deltas,
making their effective stored patterns equal to the family template).

**Site scoring** (where it is):

    score_site(w, l_w) = σ((1/β_l) · logsumexp_{k=1..K_l}(β_l · q_w · W_K_l · M_l[l_w, k]))

Where `l_w` is the known site of window w. This scores how typical the
window is for its site — high = expected acoustic state, low = unusual
(rare event, equipment anomaly, hard window).

**Species × Site affinity** (what's expected where):

    A(s, l) = (1/β) · logsumexp_{k_s, k_l}(β · (W_K_s · M_s[s, k_s]) · (W_K_l · M_l[l, k_l]))

Shape: `(S·K_s, D) @ (L·K_l, D)^T → (S·K_s, L·K_l) → reshape + lse → (S, L)`.

A static compatibility matrix computed directly from the learned memory
banks — no window input needed. It encodes which species the model has
learned to expect at which sites and is the eval-time deliverable for
ecological interpretation.

The key property of the learned space: this bilinear form is meaningful
because both `M_s` and `M_l` live in the same D, shaped by a projection
that was aligned via bird cross-domain BCE supervision. Exp 106's
analogous composition failed because species and site prototypes lived
in incompatible regions of Perch's fixed logit space.

**Composed window score** (posterior):

    score(w, s) = g · score_species(w, s)
                + (1 - g) · score_species(w, s) · score_site(w, l_w)

where `g = σ(gate(q_w)) ∈ [0, 1]` is a learned per-window gate. When site
context isn't informative, g → 1 and the model falls back to species-only
scoring. The gate is conditional on the query, not global — different
windows can use site context to different degrees.

## Dimensional summary

| Operation | Input shapes | Contracted dim | Output shape |
|-----------|--------------|----------------|--------------|
| Projection (MLP) | `(W, E) → (W, H) → (W, D)` | E, H | `(W, D)` |
| Query transform | `(W, D) @ W_Q` | D | `(W, D)` |
| Effective species memory | `(F, K_s, D)[family_of] + (S, K_s, D)` | — | `(S, K_s, D)` |
| Species key transform | `(S, K_s, D) @ W_K_s` | D | `(S, K_s, D)` |
| Site key transform | `(L, K_l, D) @ W_K_l` | D | `(L, K_l, D)` |
| Species scoring | `(W, D) × (S, K_s, D)` | D | `(W, S)` |
| Site scoring | `(W, D) × (L, K_l, D)` | D | `(W,)` |
| Species × Site affinity | `(S, K_s, D) × (L, K_l, D)` (post-W_K) | D | `(S, L)` |
| Gated composition | `(W, S) × (W,) × (W,)` | — | `(W, S)` |

Total parameters:

| Component | Shape | Parameters |
|-----------|-------|-----------|
| Projection MLP | `(E, H) + (H, D) + LN` | ~2.1M |
| `W_Q` | `(D, D) = (512, 512)` | 262K |
| `W_K_species` | `(D, D) = (512, 512)` | 262K |
| `W_K_site` | `(D, D) = (512, 512)` | 262K |
| Family memory | `(F, K_s, D) = (30, 8, 512)` | 123K |
| Species delta memory | `(S, K_s, D) = (235, 8, 512)` | 963K |
| Site memory | `(L, K_l, D) = (21, 8, 512)` | 86K |
| Platt calibration | `2 · S = 470` | 470 |
| Hopfield temperatures | `β_s, β_l` | 2 |
| Gate | `(D, 1) + bias = 513` | 513 |
| **Total** | | **~4.1M** |

The hierarchical species memory (family + delta) adds ~123K parameters
for the family bank without reducing the species delta bank — the total
is slightly larger than a flat `(235, 8, 512)` species bank by exactly
the family bank size. This is a small cost for a mechanism that converts
low-resource species from "unlearnable" to "learnable via family prior."

Output dimension is now 235 (234 competition species + 1 silence class),
which adds 5 rows to the species delta bank (+10K params) and 2 extra
Platt parameters vs the pre-silence version. The silence class is
treated as a species in its own single-member family for indexing
purposes.

Weight decay asymmetry: `Δ_s` uses 1e-2 weight decay (strong shrinkage);
all other parameters use 1e-5 (essentially none). This shrinkage is
load-bearing — it's what collapses low-resource species deltas toward
zero and lets the family prior dominate for species without enough
data.

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

The sigmoid + BCE on top adds a logistic reweighting per (window, species)
pair: the effective gradient on `M_{s,k}` is scaled by
`(y_{s} − σ(a_s · s(q) + b_s))`, so well-predicted windows contribute less
and misclassified windows contribute more. BCE shaping focuses the EM
procedure on hard cases without breaking the EM structure.

Entropy minimization on unconfident unlabeled data (exp 107's `L_ent` on
windows below Perch's confidence threshold) is also EM-flavored: sharpening
the posteriors is equivalent to sharpening the responsibilities `r_{s,k}`,
tightening the M-step update toward each prototype's high-confidence queries.

**Consequence for initialization.** Because gradient descent is already
doing soft EM, k-means init is a warm start, not a necessity. Exp 107
k-means-initializes the **family memory and site memory** on raw Perch
embeddings (pooled per family and per site respectively), then projects
the centers through the untrained MLP once and writes them into the
parameter tensors. The species delta memory is **zero-initialized** —
species start as pure family fallback and develop their own specialization
only if they have enough training gradient to overcome weight decay.
The soft-EM gradient identity carries the family-level clustering
structure forward as the projection evolves, and the delta parameters
form a species-specific residual layer on top of the family template.

Consequence: the species bank operates as a soft mixture model over K
learned components, with mode discovery emerging as a side effect of BCE
training plus the unconfident-SC entropy regularizer. No separate
clustering step during training, no fixed mode count, no need to name
the modes up front.

## Recommendation-system framing

The star schema is a factored recommendation model over **families,
species, and sites**, with hierarchical item embeddings (species
inheriting from families) and contextual gating over user (site) state:

| Recsys concept | Exp 107 analog |
|----------------|----------------|
| Items | Species (S = 235, incl. silence) |
| Item categories | Families (F ≈ 30) — items share category-level priors |
| Users | Sites (L = 21) |
| Item embeddings | Effective species memory `M_s = M_f[family_of(s)] + Δ_s` (multi-prototype, hierarchical) |
| Category embeddings | Family memory `M_f` (multi-prototype, shared across category members) |
| User embeddings | Site memory `M_l` (multi-prototype) |
| Query | Window embedding `q_w` |
| Bilinear score | `q · W_K · M_s` through shared D |
| Content features (strong) | 739 labeled SC windows (ground truth) |
| Content features (weak) | Perch-verified focal + Perch-pseudo unlabeled SC |
| Implicit feedback | Unconfident unlabeled SC ("impressions" with no explicit label) |
| Cold-start handling | Zero-focal species inherit their family's prototype via `M_f`, weight decay pushes `Δ_s` toward zero |

Concretely, exp 107 combines three retrieval paradigms:

1. **Two-tower retrieval** — query tower `q_w = proj(emb)` and item tower
   `W_K_s · M_s[s, k]` scored by dot product. Standard dense retrieval.
2. **Multi-prototype items** — items (species) have K prototypes instead
   of a single vector, supporting multi-modal distributions. Equivalent
   to item-side memory networks.
3. **Contextual gating** — the gate `g(q_w)` is a per-query mixing weight
   over the species-only score and the species × site composition.
   Context (site) modulates item scores conditionally on the query.

The species × site affinity matrix `A(s, l)` is the recovered analog of
a classical user-item matrix factorization. Instead of fitting it
directly against observed counts (sparse and noisy at 739 labeled
windows), it is recovered from the learned per-entity embeddings —
regularized by the weak stream that shapes both sides independently.

## Modern Hopfield framing

Modern Hopfield networks (Ramsauer et al. 2020) generalize classical
associative memory by replacing discrete patterns with continuous vectors
and the local update rule with scaled dot-product attention. The update

    q_{t+1} = softmax(β · q_t · M^T) · M

converges to the stored pattern closest to the initial query, with
exponential storage capacity in D.

Each memory bank in exp 107 is a Hopfield store:

- Species bank: effective `M_s = M_f[family_of] + Δ_s` flattened to
  `(S · K_s, D)` — stored patterns for every species × mode pair. The
  family / delta decomposition is a **structured parameterization** of
  the stored patterns: low-rank dense (the family templates) plus
  sparse per-item corrections (the species deltas). At retrieval time
  the Hopfield network sees a single flat `M_s`; at training time the
  decomposition controls which patterns gradient flows into and how
  weight decay regularizes them.
- Site bank: `M_l[l_w]` — stored patterns for the current window's site.

The hierarchical parameterization is analogous to a **low-rank + sparse
decomposition of the Hopfield memory matrix**: `M_s = (family template
matrix, replicated across sibling species) + (sparse species corrections)`.
Canonical Hopfield networks have a flat `M`; exp 107 gains sample
efficiency for low-resource items by sharing stored patterns across
taxonomic relatives. Retrieval dynamics are unchanged — only storage is
structured.

Differences from the canonical Hopfield setup:

1. **One-shot retrieval, no iteration.** Multi-label classification doesn't
   need convergence to a single attractor — we want independent scoring
   per species. Iteration would collapse the signal.
2. **Two-stage aggregation.** Canonical Hopfield applies a single softmax
   over all stored patterns. Exp 107 splits this into two stages: softmax
   (via logsumexp) *within* a species to aggregate over its K modes, and
   sigmoid *across* species for multi-label independence. A single global
   softmax would force species to compete for attention mass, which is
   wrong for multi-label outputs.
3. **Per-species Platt calibration.** Before the final sigmoid, the
   logsumexp score is transformed by a learnable affine `(a_s, b_s)`,
   giving each species its own inverse temperature on top of the shared
   Hopfield β and absorbing systematic per-species offsets that would
   otherwise bias the BCE gradient.

The Hopfield framing is a useful lens for intuition — the memory banks
are an associative store, the logsumexp is the Hopfield energy, β is
inverse temperature — and the implementation is plain einsum + logsumexp
+ sigmoid. No library dependency, no iteration. If the experiment fails,
it won't be because "Hopfield" was wrong; the load-bearing hypothesis is
the learned projection's domain-alignment story from `../proposal.md`.
