# Star Schema: Tensor Framework

A formal description of the tensor structure behind exp 107. The model is a
factored recommendation system over species × site, with prototype memory
banks as learned entity embeddings and bilinear compositions as scoring
functions. This document specifies the framework; see `../design.md` for the
training procedure and implementation.

## Entities

| Symbol | Name | Size | Description |
|--------|------|------|-------------|
| S | Species | 234 | BirdCLEF targets |
| L | Sites | 21 | Pantanal monitoring locations |
| K_s | Species prototype slots | 16 | Soft-EM components per species (mode count is learned, not hard-coded) |
| K_l | Site prototype slots | 16 | Soft-EM components per site |
| D | Shared space | 512 | Learned projection dimension (where memory banks and queries live) |
| H | MLP hidden | 1024 | Nonlinear projection hidden layer |
| E | Perch embeddings | 1,536 | Pretrained input space |
| P | Perch logits | 14,795 | Phase 0 distillation target vocabulary |
| W | Windows | 400K+ | Individual observations (focal + labeled SC + unlabeled SC) |

Every entity lives in a single shared D-dimensional space. Relationships
between entities are bilinear forms through D, with explicit `W_Q` and
`W_K` projections on the retrieval side (standard Ramsauer formulation).

## Tensor representations

**Species memory** — `M_s ∈ ℝ^{S × K_s × D}`

Each species s has K_s prototype vectors in the shared space. The slots
are unlabeled soft-EM components — training assigns each window to a
softmax mixture over the K prototypes, and the slots specialize to
whatever modes the data demands (song vs call, dawn vs nocturnal, solo
vs chorus). K_s = 16 is abstract capacity; not all slots are expected to
be active for every species, and with soft-max (logsumexp) retrieval,
unused slots drift toward nearest attractors rather than dying.

**Site memory** — `M_l ∈ ℝ^{L × K_l × D}`

Each site l has K_l prototype vectors capturing acoustic states. As with
the species bank, the slots are unlabeled — training discovers site
states (dawn chorus, midday quiet, rain event, etc.) from the data
rather than inheriting them from upstream clustering.

**Window query** — `q_w ∈ ℝ^D`

A test window projected into the shared space: `q_w = proj(emb_w)` where
`proj : ℝ^E → ℝ^D` is a learned linear projection.

## Bilinear compositions

All compositions are bilinear forms through the shared D dimension.

**Species scoring** (what it sounds like):

    score_species(w, s) = σ(a_s · (1/β_s) · logsumexp_{k=1..K_s}(β_s · q_w · M_s[s, k]) + b_s)

Shape: `(W, D) @ (S·K_s, D)^T → (W, S·K_s) → reshape + lse over K → (W, S)`.
The logsumexp is the modern Hopfield retrieval energy — a differentiable
soft-max that aggregates across K slots. The affine `(a_s, b_s)` is
per-species Platt calibration. `β_s` is a learnable global temperature
that controls how sharply one slot dominates.

**Site scoring** (where it is):

    score_site(w, l_w) = σ((1/β_l) · logsumexp_{k=1..K_l}(β_l · q_w · M_l[l_w, k]))

Where `l_w` is the known site of window w. This scores how typical the
window is for its site — high = expected acoustic state, low = unusual
(rare event, equipment anomaly, hard window).

**Species × Site affinity** (what's expected where):

    A(s, l) = (1/β) · logsumexp_{k_s, k_l}(β · M_s[s, k_s] · M_l[l, k_l])

Shape: `(S·K_s, D) @ (L·K_l, D)^T → (S·K_s, L·K_l) → reshape + lse → (S, L)`.

A static compatibility matrix computed directly from the learned memory
banks — no window input needed. It encodes which species the model has
learned to expect at which sites. The key property of the learned space:
this bilinear form is meaningful because both M_s and M_l live in the same
D, unlike exp 106 where species and site prototypes lived in incompatible
regions of Perch's logit space.

**Composed window score** (posterior):

    score(w, s) = g · score_species(w, s)
                + (1 - g) · score_species(w, s) · score_site(w, l_w)

where `g = σ(gate(q_w)) ∈ [0, 1]` is a learned per-window gate. When site
context isn't informative, g → 1 and the model falls back to species-only
scoring. The gate is conditional on the query, not global — different
windows can use site context to different degrees.

## Dimensional summary

| Operation | Input shapes | Contracted dim | Output shape |
|-----------|-------------|------------|--------------|
| Projection (MLP) | `(W, E) → (W, H) → (W, D)` | E, H | `(W, D)` |
| Query transform | `(W, D) @ W_Q` | D | `(W, D)` |
| Species key transform | `(S, K_s, D) @ W_K_s` | D | `(S, K_s, D)` |
| Site key transform | `(L, K_l, D) @ W_K_l` | D | `(L, K_l, D)` |
| Species scoring | `(W, D) × (S, K_s, D)` | D | `(W, S)` |
| Site scoring | `(W, D) × (L, K_l, D)` | D | `(W,)` |
| Species × Site affinity | `(S, K_s, D) × (L, K_l, D)` (post-W_K) | D | `(S, L)` |
| Gated composition | `(W, S) × (W,) × (W,)` | — | `(W, S)` |
| Distill head (phase 0 only) | `(W, D) → (W, P)` | D | `(W, 14795)` |

Total parameters:

| Component | Shape | Parameters |
|-----------|-------|-----------|
| Projection MLP | `(E, H) + (H, D) + LN` | ~2.1M |
| `W_Q` | `(D, D) = (512, 512)` | 262K |
| `W_K_species` | `(D, D) = (512, 512)` | 262K |
| `W_K_site` | `(D, D) = (512, 512)` | 262K |
| Species memory | `(S, K_s, D) = (234, 16, 512)` | 1.92M |
| Site memory | `(L, K_l, D) = (21, 16, 512)` | 172K |
| Platt calibration | `2 · S = 468` | 468 |
| Hopfield temperatures | `β_s, β_l` | 2 |
| Gate | `(D, 1) + bias = 513` | 513 |
| Distill head (phase 0 only) | `(D, P) = (512, 14795)` | ~7.6M |
| **Total (phases 1-3)** | | **~5.0M** |
| **Total (phase 0)** | | **~12.6M** |

The distill head is large because `P = 14795` — Perch's full species
vocabulary. It's used only in phase 0 and discarded (or frozen) afterwards;
the persistent model is ~5M parameters. Phase 0 is the "temporary
teacher-reproducer" head that exists to carry the distillation gradient
through the projection.

## Logsumexp over K as soft EM (Hopfield retrieval)

The full mixture marginal over species modes is:

    P(x | s) = Σ_k π_k P(x | s, k)

The model approximates this with the modern Hopfield retrieval energy:

    score_s(x) = (1/β) · logsumexp_k(β · q · M_s[s, k])

This is the Ramsauer et al. 2020 energy function and also the temperature-β
soft-max approximation to the hard maximum. It interpolates between two
limits:

- **β → ∞**: logsumexp → max_k (hard EM, winner-take-all). Each window
  routes to one prototype per species; other slots receive zero gradient.
- **β → 0**: logsumexp → mean_k + log(K)/β. All slots contribute equally;
  the mixture has no sharpness.

At finite β, all K prototypes receive gradient proportional to their
softmax attention weight `softmax_k(β · q · M_{s,k})`. This is the key
property that lets us set K abstractly (16 slots rather than 4
biologically motivated modes): unused prototypes don't cause
dead-slot pathologies, they just receive small gradients and drift toward
the nearest active attractor. Training discovers how many modes each
species actually needs.

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

The sigmoid+BCE on top adds a logistic reweighting per (window, species)
pair: the effective gradient on `M_{s,k}` is scaled by
`(y_{s} − σ(a_s · s(q) + b_s))`, so well-predicted windows contribute less
and misclassified windows contribute more. This is just BCE shaping — it
doesn't break the EM structure, it focuses it on the hard cases.

Entropy minimization on unlabeled data is also EM-flavored: sharpening the
posteriors is equivalent to sharpening the responsibilities `r_{s,k}`,
tightening the M-step update toward each prototype's high-confidence
queries.

**Consequence for initialization.** Random init of the species bank is
safe specifically because gradient descent is already doing clustering.
An explicit k-means pass before training would just warm-start the same
EM procedure — and k-means in the raw Perch space is a stale warm start
because the target clustering space (the learned 512-d projection)
doesn't exist yet at init time. For the species bank, letting soft EM
discover modes in the current projection space on every step is strictly
more appropriate than pinning slots to a pre-projection k-means solution.

The site bank is different: 6K+ windows per site makes k-means
well-posed, the projection-drift problem is mitigated by PCA init of the
projection, and the warm start on the large-data head saves real
training effort. Asymmetric init (random species, k-means site) reflects
the asymmetric data regimes.

Consequence: the species bank operates as a soft mixture model over K
learned components, with mode discovery emerging as a side effect of BCE
training plus the unlabeled entropy regularizer. No separate clustering
step, no fixed mode count, no need to name the modes up front.

## Recommendation-system framing

The star schema is a factored recommendation model over species and sites:

| Recsys concept | Exp 107 analog |
|----------------|----------------|
| Items | Species (S = 234) |
| Users | Sites (L = 21) |
| Item embeddings | Species memory `M_s` (multi-prototype) |
| User embeddings | Site memory `M_l` (multi-prototype) |
| Query | Window embedding `q_w` |
| Bilinear score | `q · M` through shared D |
| Content features | Focal recordings (species identity prior) |
| Explicit feedback | Labeled soundscapes (739 windows) |
| Implicit feedback | Unlabeled soundscapes (127K "impressions") |

Concretely, exp 107 combines three retrieval paradigms:

1. **Two-tower retrieval** — query tower `q_w = proj(emb)` and item tower
   `M_s[s, k]` scored by dot product. Standard dense retrieval.
2. **Multi-prototype items** — items (species) have K prototypes instead of
   a single vector, supporting multi-modal distributions. Equivalent to
   item-side memory networks, or a nearest-neighbor retrieval with K stored
   vectors per class.
3. **Contextual gating** — the gate `g(q_w)` is a per-query mixing weight
   over the species-only score and the species × site composition. Matches
   contextual bandits where the context (site) modulates item scores
   conditionally on the query.

The species × site affinity matrix `A(s, l)` is the recovered analog of a
classical user-item matrix factorization. Instead of fitting it directly
against observed counts (sparse and noisy at 739 labeled windows), it is
recovered from the learned per-entity embeddings — regularized by the focal
and unlabeled losses that shape each side independently.

## Modern Hopfield framing

Modern Hopfield networks (Ramsauer et al. 2020) generalize classical
associative memory by replacing discrete patterns with continuous vectors
and the local update rule with scaled dot-product attention. The update

    q_{t+1} = softmax(β · q_t · M^T) · M

converges to the stored pattern closest to the initial query, with
exponential storage capacity in D.

Each memory bank in exp 107 is a Hopfield store:

- Species bank: `M_s` flattened to `(S · K_s, D)` — stored patterns for
  every species × mode pair.
- Site bank: `M_l[l_w]` — stored patterns for the current window's site.

Differences from the canonical Hopfield setup:

1. **One-shot retrieval, no iteration.** Multi-label classification doesn't
   need convergence to a single attractor — we want independent scoring per
   species. Iteration would collapse the signal.
2. **Two-stage aggregation.** Canonical Hopfield applies a single softmax
   over all stored patterns. Exp 107 splits this into two stages: softmax
   (via logsumexp) *within* a species to aggregate over its K modes, and
   sigmoid *across* species for multi-label independence. Otherwise a
   single global softmax would force species to compete for attention
   mass, which is wrong for multi-label outputs.
3. **Per-species Platt calibration.** Before the final sigmoid, the
   logsumexp score is transformed by a learnable affine `(a_s, b_s)`.
   This lets each species have its own inverse temperature on top of the
   shared Hopfield β, and it absorbs systematic per-species offsets that
   would otherwise bias the BCE gradient.

The Hopfield framing is a useful lens for intuition — the memory banks
are an associative store, the logsumexp is the Hopfield energy, β is
inverse temperature — and the implementation is plain einsum + logsumexp
+ sigmoid. No library dependency, no iteration.
