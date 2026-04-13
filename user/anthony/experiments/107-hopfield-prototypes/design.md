# Design: Prototype Retrieval for Amphibian-Dominated Soundscapes

## Overview

A PyTorch memory-bank classifier over frozen Perch v2 embeddings.
Species and site prototypes live in a learned 512-d projection space;
species retrieval is trained via BCE on strong + weak focal labels;
`M_site` and the cross-attention composition operator are trained via
**Hopfield energy minimization** on 127K unlabeled soundscape windows
(two unsupervised terms — direct site energy and site-conditioned
species retrieval energy). At inference the species bank is
cross-attention-refined per window from the site bank, then scored with
the same Hopfield logsumexp used at training.

This document is the **interface contract and mathematical specification**
for the exp 107 implementation. It deliberately does not embed Python
source — see `src/birdclef_107/` for the implementation, and
`docs/star-schema-dimensional-analysis.md` for the full tensor framework,
soft-EM gradient identity, and Hopfield energy derivation. See
`proposal.md` for the thesis and success criteria.

**Reading order.** Start here for the architectural decisions and
training recipe, then cross-reference the star-schema doc for any
mathematical step whose justification isn't obvious, then read the code
to see how the interface is realized. If the code and this document
disagree on anything load-bearing, the code is authoritative and this
document should be updated.

## Locked architecture decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input features | Perch v2 pooled embeddings (1536-d, frozen) | Cached from exp 106; a strong general bioacoustic feature space |
| Output dim | 234 (no silence class) | Axiomatic simplification — bag labels + K=8 prototype slots absorb silent focal noise |
| Memory bank count | Two (species + site), star schema factored | Eval-time affinity matrix is a deliverable |
| Species bank parameterization | **Flat** `M_species : (234, K=8, 512)`. Family hierarchy is a training-time shrinkage regularizer (`L_family`), not a parameterized bank. | Partial pooling via stop-grad mean over siblings; zero inference-time cost |
| Shared space dim | D = 512 | Enough capacity to disentangle focal ↔ soundscape manifolds |
| Projection architecture | `Linear(1536, 1024) → GELU → Linear(1024, 512) → LayerNorm` | Nonlinear; shaped by labeled BCE gradient; frozen under the unsupervised site loss |
| Retrieval projections | Explicit `W_Q`, `W_K_species`, `W_K_site` (all `(512, 512)`) | Ramsauer formulation of modern Hopfield networks |
| Prototype slots | K_s = K_l = 8 | Enough for intra-species acoustic modes (song/call, dawn/night, chorus) without overparameterizing |
| Retrieval aggregation | `(1/β)·logsumexp_k(β · q_r · W_K · M_{*,k})`, learnable `log_β` per head | Modern Hopfield energy — all prototypes receive softmax-weighted gradient |
| Per-species calibration | Built-in Platt `sigmoid(a_s · score + b_s)` | Cleaner BCE gradients; the only per-species differentiator for zero-data orphans that otherwise collide with their data-rich family siblings |
| Species × site composition | **Multi-head cross-attention.** `Q = M_species`, `K, V = M_site`; produces per-window site-conditioned prototypes `M_cond[b, s, k]` consumed by the unchanged Hopfield retrieval | True factored star schema per-(species, site), replacing the v1 scalar fusion gate (`g · sp + (1−g) · sp · site_score`) that collapsed the site bank to one scalar per window. Gate ablation cost −0.045 AUROC on S22. |
| Cross-attention init | Zero-init ``out_proj`` weight + bias. At step 0 ``attended = 0`` → ``refined = Q``, so cross-attention is numerically identity; gradient still flows into ``out_proj`` because ``dloss/d(out_proj) = dloss/d(attended) · (softmax · V · W_V) ≠ 0``. | Safer cold start than ReZero. ReZero deadlocked with α stuck at ≈0 — ``dloss/d(attended) = α = 0`` is the ReZero failure mode. |
| Loss composition | `L_labeled + λ_site · (L_site + L_site_xattn) + λ_fam · L_family` | `L_site` is the v1 direct Hopfield energy on `M_site`; `L_site_xattn` is the v2 species-retrieval energy *through* cross-attention — the path by which the 127K unlabeled SC windows shape the composition operator itself. |
| Projection gradient on unsupervised loss | **Frozen** (`torch.no_grad()` through both MLP and `W_Q`) | Prevents the shared backbone from encoding site-identifying artifacts — stronger than the "detach only proj output" sketch, which would still train `W_Q.weight` via its forward matmul |
| Species-side retrieval operator under the unsupervised cross-attn term | **Detached** — `M_species`, `species_head.W_K.weight`, `species_head.log_β` all explicitly `.detach()`-ed inside `unsupervised_cross_attn_contrastive` | v2/v3 analog of the v1 projection-detach invariant: site-clustering signal must not reach the species retrieval operator, or domain shift entangles with site structure (the exp 106 failure mode). Only cross-attention weights are shaped by this term (v3: `M_site` is also detached inside this loss — see invariant 3 below). |
| Initialization | Random (no k-means) | Soft-EM gradient identity justifies BCE+Hopfield convergence without warm start |
| Tensor operations | einops `einsum`, `reduce`, `rearrange` | Project convention; self-documenting contractions |

**Total parameters: ≈ 4.99M** at default shapes
(projection 2.10M + `W_Q`/`W_K_species`/`W_K_site` 0.79M + `M_species` 0.96M +
`M_site` 0.09M + per-species Platt 468 + two scalar `log_β` + cross-attention 1.05M).
The v1 scalar gate (513 params) has been removed.

## Built-in Platt scaling

The species head applies per-species affine calibration before the
sigmoid:

$$
\text{score}_s = \sigma\!\left(a_s \cdot \frac{1}{\beta_s}\,\text{lse}_k\!\left(\beta_s \cdot q_r \cdot W_{K,s} \cdot M_{s,k}\right) + b_s\right)
$$

where `a_s` (scale) and `b_s` (bias) are learnable per-species parameters
initialized to identity.

Why calibration belongs in the model:

1. **Cleaner BCE gradients.** Uncalibrated scores produce biased BCE
   gradients; per-class scale + bias absorbs systematic per-species
   offsets that would otherwise be learned into the projection.
2. **Orphan differentiation.** `b_s` is the **only** mechanism that
   differentiates zero-data orphans from their data-rich family
   siblings once partial pooling has pulled them onto the same family
   centroid (see `star-schema-dimensional-analysis.md` "Orphan
   collision limit"). Without it, orphans emit identical retrieval
   scores.

The site bank does not need its own Platt. In v2 it is no longer queried
directly at inference — it feeds cross-attention as K/V, and the
site-conditioned species prototypes are scored through the existing
per-species Platt.

## Model interface

The implementation lives in `src/birdclef_107/model.py`. Four
`nn.Module` classes, all in one file:

### `PerchProjection(d_input, d_hidden, d_proj)`

Shared frozen-input → projected-query backbone plus the query rotation
`W_Q`. Every retrieval path routes through this module. Supervised
losses shape it; unsupervised losses call a dedicated "frozen-pass"
entry point that wraps the entire backbone in `torch.no_grad()` so
neither the MLP nor `W_Q.weight` receives gradient.

| Method | Signature | Contract |
|---|---|---|
| `forward(emb)` | `(B, d_input) → (B, d_proj)` | Returns the retrieval query `q_r = W_Q(LayerNorm(·))`. Previously returned a `(q_base, q_query)` tuple where `q_base` fed the v1 fusion gate; the gate has been removed. |
| `query_from_detached(emb)` | `(B, d_input) → (B, d_proj)` | Runs the full backbone inside `torch.no_grad()` and returns `q_r` with `requires_grad=False`. Used by both unsupervised losses. |

**Design invariant.** `model.projection.parameters()` receives **zero
gradient** from any loss that only flows through `query_from_detached`.
The submodule boundary *is* the detach boundary — new shared-backbone
parameters must live here, not on sibling modules, or this invariant
leaks.

This is a stricter guarantee than "detach `proj(emb)`". A literal
`W_Q(proj(emb).detach())` still routes gradient into `W_Q.weight` via
the forward matmul (∂L/∂W_Q.weight = ∂L/∂q_query · detached_inputᵀ ≠ 0),
so the backbone would be partially trained by the site-clustering loss.
`torch.no_grad()` over the whole pass closes that leak.

### `HopfieldSpeciesHead(n_species, K, d_proj)`

Flat species memory bank with modern-Hopfield retrieval and per-species
Platt calibration. Owns `M : (n_species, K, d_proj)`, `W_K`, `log_β`,
`cal_scale`, `cal_bias`. Also owns `family_shrinkage_loss` — that loss
operates on `self.M` directly, so keeping it colocated with the species
bank avoids reaching across module boundaries from the top-level model.

| Method | Signature | Contract |
|---|---|---|
| `forward(q_r)` | `(B, d_proj) → (B, n_species)` | Returns calibrated probabilities in `[0, 1]`. Thin wrapper over `forward_with_prototypes(q_r, self.M)` — used only on focal recordings (no `site_idx`). |
| `forward_with_prototypes(q_r, M_species)` | `(B, D), (S,K,D) \| (B,S,K,D) → (B, n_species)` | Same retrieval with caller-supplied prototypes. The `(B, S, K, D)` shape is the v2 cross-attention-refined bank used on soundscape windows. |
| `family_shrinkage_loss(family_idx)` | `(n_species,) long → scalar` | `L_family` — squared Euclidean to the stop-grad uniform family mean. |

### `HopfieldSiteHead(n_sites, K, d_proj)`

Flat site memory bank. In v2 it is no longer queried directly at
inference — it feeds `SpeciesSiteCrossAttention` as K/V. Its parameters
(`M`, `W_K`, `log_beta`) are shaped exclusively by
`PrototypeRetrievalModel.unsupervised_site_contrastive` (InfoNCE over
23 sites).

No methods. `HopfieldSiteHead` is a parameter container only — the v1
`forward` (per-window sigmoid typicality) and the v2 `retrieval_energy`
(softplus-bounded Hopfield energy) were both removed. Shaping logic
lives at the `PrototypeRetrievalModel` level so the contrastive loss
can access `q_r`, `site_idx`, and `M_site` together.

### `SpeciesSiteCrossAttention(d_proj, n_heads=4)`

Per-window site-conditioning of species prototypes via multi-head
cross-attention.

| Method | Signature | Contract |
|---|---|---|
| `refine(M_species, M_site)` | `(S,K_sp,D), (L,K_site,D) → (L,S,K_sp,D)` | Attention across all `L=23` sites at once: `Q = M_species` (flattened to `(L, S·K_sp, D)` via broadcast); `K, V = M_site`. Used by `unsupervised_cross_attn_contrastive` to score retrieval at every site for InfoNCE. |
| `forward(M_species, M_site, site_idx)` | `(S,K_sp,D), (L,K_site,D), (B,) long → (B,S,K_sp,D)` | Thin wrapper: `return self.refine(M_species, M_site)[site_idx]`. Gathers per-window refined prototypes. Memory is `L × S × K_sp × D ≈ 88 MB` at default shapes, not `B × S × K_sp × D`. |

**Zero-init `out_proj`.** `attended = softmax(·) · V · W_V · 0 = 0` at
init, so `refined = Q` (numerically identity). Gradient into `out_proj`
is non-zero from step 1 — `dloss/d(out_proj) = dloss/d(attended) ·
(softmax · V · W_V)` doesn't involve `out_proj` itself. ReZero
(`α · attended` with `α = 0`) was tried first and deadlocked because
there `dloss/d(attended) = α = 0`.

### `PrototypeRetrievalModel(n_species=234, n_sites=23, K_species=8, K_site=8, d_input=1536, d_hidden=1024, d_proj=512, cross_attention_heads=4)`

Top-level composition of the four sub-modules.

| Method | Signature | Contract |
|---|---|---|
| `forward(emb, site_idx=None)` | `(B, 1536), (B,) long\|None → (B, 234)` | With `site_idx` (strong SC / unsup SC / test windows): `cross_attn` produces per-window `M_cond`, then `species_head.forward_with_prototypes(q_r, M_cond)`. With `site_idx=None` (focal / OOD-site fallback): uses the shared `M_species` directly. |
| `unsupervised_site_contrastive(emb, site_idx)` | `(B, 1536), (B,) long → scalar` | **v3 InfoNCE over 23 sites on direct `M_site` retrieval.** Trains `site_head.{M, W_K, log_β}`. At random init ≈ `log(23) ≈ 3.14` — never saturates. Projection frozen via `query_from_detached`. Replaces v2's `softplus(-agg)` which saturated at init and gave zero gradient to `M_site`. |
| `unsupervised_cross_attn_contrastive(emb, site_idx)` | `(B, 1536), (B,) long → scalar` | **v3 site-InfoNCE on cross-attn-refined species retrieval profiles.** `cross_attn.refine()` produces refined species prototypes at every site; per-site scores are computed via logsumexp over K then over species; InfoNCE forces the correct site's score to exceed wrong sites'. Trains `cross_attn.*` only — `M_species`, `species.W_K`, `species.log_β`, and (v3 new) `M_site` are all explicitly `.detach()`-ed. Replaces v2's `softplus(-logsumexp_s)` which collapsed to "match anything". |
| `family_shrinkage_loss(family_idx)` | `(n_species,) long → scalar` | Delegates to `species_head.family_shrinkage_loss`. |

**Kaggle checkpoint surface.** v2 requires `projection + species_head +
site_head + cross_attn` to reproduce the training-time inference path.
The OOD-site fallback (`site_idx=None`) bypasses both `site_head` and
`cross_attn`, so for sites outside `S01..S23` one passes `site_idx=None`
and the forward pass uses the shared `M_species` — no per-site
prototype needed at inference.

### `affinity_matrix(model)` (module-level helper)

Computes the *pre-attention* species × site affinity as a
`(n_species, n_sites)` tensor by folding every `(k_s, k_l)` prototype
pair into a single logsumexp per `(s, l)` cell. Used for interpretation
and ecological reporting; the v2 inference-time score additionally
applies the learned cross-attention offset per window, so this affinity
is a static view of the banks, not the full inference computation.
Enforced by `tests/test_affinity.py` to match a brute-force
quadruple-loop reference.

## Mathematical specification

Notation follows `docs/star-schema-dimensional-analysis.md`. `B` is
batch, `S = 234` species, `L = 23` sites, `F = 67` families, `K_s = K_l
= 8` prototype slots, `D = 512`, `E = 1536`, `H = 1024`.

**Backbone.** For an input embedding `emb ∈ ℝ^E`:

$$
q = \text{LayerNorm}\big(W_2\,\text{GELU}(W_1\,\text{emb} + b_1) + b_2\big), \qquad q_r = W_Q\,q
$$

`q_r` is the query-space projection, used by species retrieval (both
the focal direct path and the cross-attention-refined soundscape path).
The v1 gate that consumed `q` has been removed.

**Species retrieval + Platt calibration (focal path).** For windows
without a `site_idx` (focal recordings, OOD-site Kaggle fallback):

$$
\text{score}_s(q_r) = \sigma\!\left( a_s \cdot \frac{1}{\beta_s}\,\text{lse}_{k=1..K_s}\!\big(\beta_s \cdot q_r^\top W_{K,s}\,M^{(s)}_k\big) + b_s \right)
$$

where `a_s, b_s` are the per-species Platt parameters, `β_s =
exp(log_β_species)`, and `M^{(s)}_k ∈ ℝ^D` is the k-th prototype of
species `s`.

**Cross-attention refinement (soundscape path).** For a window with
known site `l_w`, species prototypes are updated by a site-dependent
residual:

$$
\Delta^{(l)} = \text{MultiHeadAttention}\big(Q = M_{\text{species}},\ K = V = M^{(l)}_{\text{site}}\big), \qquad
M^{(l, s)}_{\text{cond}, k} = M^{(s)}_k + \Delta^{(l)}_{s, k}
$$

Retrieval runs against the conditioned prototypes:

$$
\text{score}_s(q_r, l_w) = \sigma\!\left( a_s \cdot \frac{1}{\beta_s}\,\text{lse}_{k=1..K_s}\!\big(\beta_s \cdot q_r^\top W_{K,s}\,M^{(l_w, s)}_{\text{cond}, k}\big) + b_s \right)
$$

The retrieval equation is unchanged — only the prototypes carrying the
site-specific residual differ between paths. This is the sense in
which "one retrieval mechanism" is preserved despite the cross-bank
composition.

**Zero-init** of the attention `out_proj` makes `Δ = 0` at step 0, so
the model starts *behaviorally* as the focal path would on the same
`(q_r, M_species)` — no architectural shock. Gradient still flows into
`out_proj` via `dloss/d(attended) · (softmax · V · W_V)`.

**Unsupervised terms on unlabeled SC.**

Two retrieval-energy terms on windows with known `site_idx` but no
species label. Both detach `q_r` via `query_from_detached` so the
projection receives zero gradient:

$$
\mathcal{L}_{\text{site}} = \mathbb{E}_{(q_r, l_w) \sim \text{unsup}}\,\text{softplus}\!\big(-\,\tfrac{1}{\beta_l}\,\text{lse}_k(\beta_l \cdot q_r^\top W_{K,l} M^{(l_w)}_k)\big)
$$

$$
\mathcal{L}_{\text{site\_xattn}} = \mathbb{E}_{(q_r, l_w)}\,\text{softplus}\!\big(-\,\text{lse}_{s=1..S}\!\big[\tfrac{1}{\beta_s}\,\text{lse}_{k}(\beta_s \cdot q_r^\top \overline{W_{K,s}}\, M^{(l_w, s)}_{\text{cond}, k})\big]\big)
$$

`L_site` is the same bounded site Hopfield energy from v1 — "every
unlabeled window should match some prototype in its site's bank".
`L_site_xattn` is the v2 aux — "at least one site-conditioned species
prototype should retrieve this window"; the outer `lse_s` over species
keeps the gradient softer than a hard `max_s`. The overlines on
`W_{K,s}` and (implicitly) `β_s`, `M_{\text{species}}` mark explicit
`.detach()` — the species retrieval operator is never shaped by this
loss. Only `cross_attn.*` and `M_{\text{site}}` (via attention K/V)
move.

Both terms use `softplus(-·) = BCE(σ(·), 1)` to bound the loss to
`[0, ∞)` with natural saturation near zero. The unbounded
`-mean(agg)` formulation diverged in training and dominated
`L_labeled`.

**Family partial pooling (training-time only).** Let `f(s) ∈ {0, ..., F-1}`
be the family-of-species lookup. The family centroid at the current
step is the uniform mean over siblings:

$$
\bar M_f = \frac{1}{|f(·) = f|}\,\sum_{s:\, f(s) = f}\,M^{(s)}, \qquad
\mathcal{L}_{\text{family}} = \frac{1}{S \cdot K_s \cdot D}\,\sum_{s, k, d}\!\left( M^{(s)}_{k,d} - \text{sg}\big(\bar M_{f(s), k, d}\big) \right)^2
$$

where `sg(·)` is the stop-gradient operator. The shrinkage is
asymmetric — only `M^{(s)}` moves toward the target; the target itself
is a fixed view of the current `M_species`, recomputed each step. At
equilibrium, data-rich species sit far from the centroid (BCE dominates
shrinkage), low-resource species sit near it, and zero-data species
sit exactly at it. See `star-schema-dimensional-analysis.md` §"Family
partial pooling" for the hierarchical-Bayes framing.

**Total training loss.**

$$
\mathcal{L} = \underbrace{\text{BCE}(\text{sp}_{\text{strong}}, y_{\text{strong}}) + \text{BCE}(\text{sp}_{\text{weak}}, y_{\text{weak}})}_{\mathcal{L}_{\text{labeled}}} + \lambda_{\text{site}}\,(\mathcal{L}_{\text{site}} + \mathcal{L}_{\text{site\_xattn}}) + \lambda_{\text{fam}}\,\mathcal{L}_{\text{family}}
$$

Defaults (v3/v4 as-built): `λ_site = λ_xattn = 0.01` (split into two
scales after v3 — the shared-weight v1 design was superseded when the
two contrastive terms diverged in magnitude), `λ_fam = 1e-2`. **Unsup
λ warmup**: `λ_site = λ_xattn = 0` for steps 0–200, linear ramp to
target by step 500, so labeled BCE shapes the projection alone before
the contrastive losses engage. `λ_fam` is a pre-registered S22 sweep
`{0, 1e-3, 1e-2, 1e-1, 1}` — default is the middle value until the
sweep locks the optimum.

**Gradient routing.** Which parameters does each loss touch?

| Parameter | `L_labeled` | `L_site` | `L_site_xattn` | `L_family` |
|---|:-:|:-:|:-:|:-:|
| `projection.{proj, W_Q}` | ✓ | ✗ (frozen via `no_grad`) | ✗ (frozen via `no_grad`) | ✗ |
| `species_head.M` (= `M_species`) | ✓ | ✗ | ✗ (`.detach()`) | ✓ (LHS only; target is stop-grad) |
| `species_head.W_K, log_β` | ✓ | ✗ | ✗ (`.detach()`) | ✗ |
| `species_head.cal_scale, cal_bias` | ✓ | ✗ | ✗ | ✗ |
| `site_head.M` (= `M_site`) | ✓ (via cross-attn K/V on strong SC) | ✓ | ✗ (v3: `.detach()` inside xattn path) | ✗ |
| `site_head.W_K, log_β` | ✗ | ✓ | ✗ | ✗ |
| `cross_attn.*` (MHA weights) | ✓ (via strong SC, ~262 wins/epoch) | ✗ | ✓ (via unsup SC, 127K wins/epoch) | ✗ |

Four invariants worth stating explicitly:

1. **Projection is frozen under every unsupervised loss.** Both
   `unsupervised_site_contrastive` and `unsupervised_cross_attn_contrastive`
   call `query_from_detached`, which wraps the full backbone in
   `torch.no_grad()`. The submodule boundary is the detach boundary.
2. **Species retrieval operator is frozen under `L_xattn`.**
   `M_species`, `species_head.W_K.weight`, and `species_head.log_β` are
   explicitly `.detach()`-ed inside `unsupervised_cross_attn_contrastive`.
   This is the v2 analog of invariant 1 for the species side: the
   species bank must not be shaped by site-clustering signal, or domain
   shift entangles with site structure (exp 106 failure mode).
3. **`M_site` is exclusively shaped by `L_site`** (v3 invariant). The
   v2 routing path `L_xattn → M_site` (via cross-attn K/V) was closed
   by `.detach()`-ing `M_site` inside `unsupervised_cross_attn_contrastive`.
   Clean one-objective-per-parameter: `L_site` shapes `M_site`,
   `L_xattn` shapes `cross_attn.*`.
4. **`cross_attn.*` is trained by *both* labeled and unsupervised
   signals.** Under v1 the composition operator (the gate) was trained
   by labeled strong SC alone — ~262 windows/epoch, too thin. The v3
   `L_xattn_contrastive` term extends this to 127K unlabeled windows;
   this is the mechanism that closes the "cross-attn signal is
   starved" gap.

**Zero-init timing.** At step 0, `out_proj.weight = 0` so
`d_loss / d_out_proj.weight = d_loss / d_attended · (softmax · V)` is
non-zero (out_proj trains from step 1), but
`d_loss / d_V = softmax^T · out_proj.weight^T · d_loss / d_attended` is
zero (V/K projections and hence `in_proj_weight` and `M_site` don't
train via the cross-attn path at step 0). From step 2 onward, out_proj
has non-zero weight and the full gradient chain unlocks. `M_site` still
trains from step 1 via the direct `L_site` term — only its
cross-attention path is gated on the out_proj unlock.

These invariants are enforced by `tests/test_gradient_routing.py`:

- `test_unsup_site_contrastive_does_not_update_projection`
- `test_unsup_cross_attn_contrastive_does_not_update_projection`
- `test_unsup_cross_attn_contrastive_does_not_update_species_retrieval_ops`
- `test_unsup_cross_attn_contrastive_does_not_update_M_site` (v3 invariant 3)
- `test_unsup_cross_attn_contrastive_trains_out_proj_at_init`
- `test_unsup_cross_attn_contrastive_trains_in_proj_after_unlock`
- `test_site_contrastive_nonzero_at_init`
- `test_xattn_contrastive_nonzero_at_init`
- `test_labeled_bce_does_update_projection` (negative control)

## Label preparation

One-time preprocessing pass producing three on-disk streams keyed on
the same `(window_id, embedding)` schema used throughout exp 106.

| Stream | Source | Labels | Site | Size |
|---|---|---|---|---|
| `strong` | `~/scratch/.../exp106/soundscape_labeled.parquet` | 234-d window-level multi-hot, parsed from `ground_truth` semicolon-lists | present (`site_idx`) | 739 windows |
| `weak_focal` | `~/scratch/.../exp106/focal/by_species/*.parquet` | 234-d one-hot on `primary_label` | absent (`site_idx = None`) | ~265K windows |
| `unsupervised` | `~/scratch/.../exp106/soundscape_unlabeled/by_site/*.parquet` | — | present (`site_idx`) | 127K windows |

No Perch filtering, no silence detection, no tri-state, no RMS
thresholds. Focal bag labels and the 739 strong SC labels are the
complete labeled supervision. The 127K unsupervised windows carry only
site metadata and flow through the two unsupervised terms (direct site
Hopfield energy + cross-attention species retrieval energy).

**Preflight statistics computed in the same pass:**

| Stat | Use |
|---|---|
| Per-species focal window count | Weights for per-species-balanced focal sampler |
| Per-family species count | Sanity check for partial-pooling centroid stability; singletons (~33/67) have a no-op shrinkage term |
| Species without focal + without 739 positives + without family sibling with focal | Unlearnable species list (excluded from aggregate AUROC) |
| Perch top-1 direct AUROC on labeled SC | Reference baseline for the sanity-check section |

**Family index construction.** `family_utils.build_family_index` reads
the package-bundled `src/birdclef_107/assets/families.csv` (derived
out-of-band from the iNat taxonomy by `scripts/fetch_families.py`),
aligns the mapping to the canonical species ordering, and returns a
`(234,)` long tensor with contiguous family ids. The 25 Insecta
morphotype rows have no iNat family resolution and are routed to a
shared `Insecta_unresolved` pseudo-family (67 families total).

The CSV ships as a package asset so editable and wheel installs both
find it without hardcoded repo-layout paths. The `.gitignore` has an
explicit negate rule (`!.../assets/families.csv`) so the asset enters
git despite the repo-wide `*.csv` ignore.

## Training

### Batch construction

Three streams, independent per-step samplers:

| Stream | Sampler | Per-step size |
|---|---|---|
| `strong` | uniform (739 windows cycled) | 32 |
| `weak_focal` | per-species balanced `WeightedRandomSampler` with weight `1 / count(species)` | 256 |
| `unsup` | uniform over 127K windows | 256 |

The per-species-balanced focal sampler is load-bearing — without it,
long-tail species see near-zero gradient on the weak path, and the
family shrinkage has to rescue them alone. With balancing, every
species's weak path contributes on the order of its K_s prototype slots
per epoch.

### Schedule (v4 as-built)

The v1 schedule (`lr=3e-4`, `weight_decay=1e-5`, 30–40 epochs, no warmup
or asymmetry, early-stopping every 5 epochs with patience 2) was
replaced iteratively. Current shipping config:

| Knob | Value | v1 → v4 delta & rationale |
|---|---|---|
| Optimizer | AdamW | unchanged |
| LR | `5e-4`, cosine decay over all 20 epochs, `eta_min=0` | v3: `3e-4 → 5e-4` for stronger initial signal; but 3-epoch v3 run decayed to `2.5e-8` by end (LR-bounded). 20-epoch extension gives meaningful LR throughout. |
| Weight decay | `1e-2` | v4: `1e-5 → 1e-2` (1000× increase). Attacks labeled overfitting; hits attention weights and prototype matrices. |
| Epochs | 20 | v3/v4: cosine needs headroom; v4 peak typically within the first epoch regardless. |
| Early stopping | every 100 steps, patience 10 val checks | v3: intra-epoch — v1's "every 5 epochs" missed the peak. |
| Proj dropout | `0.1` | v4: `nn.Dropout` inside `PerchProjection` MLP (post-GELU). |
| Attn dropout | `0.1` | v4: passed to `nn.MultiheadAttention(dropout=…)` inside `SpeciesSiteCrossAttention`. |
| Label smoothing | `0.01` | v4: small ε for multi-hot BCE, applied to strong + focal targets before `binary_cross_entropy`. |
| Strong stride | `4` | v4: strong BCE only every 4th focal step (strong loader has ~2 batches cycled every step → each strong window seen ~519×/epoch without stride; ~130× with stride 4). Implemented via `include_strong` kwarg on `compute_loss`. |
| Unsup λ warmup | 0 for 0–200 steps, linear 200→500 | v3: labeled BCE places projection alone before contrastive losses engage. |

Three streams compose into a single per-step loss via `compute_loss` in
`src/birdclef_107/losses.py::compute_loss(label_smoothing, include_strong, ...)`.
The training loop in `scripts/train.py` owns the stride gate and unsup
warmup logic.

### Regularization budget rationale

Per-knob allocation of regularization pressure:

- **strong_stride=4** targets the *data regime* (oversampling of 262
  strong windows), not the model capacity. It is a training-loop
  adjustment, not a model change.
- **attn_dropout=0.1** targets the cross-attention path that only
  strong SC traverses (focal uses `site_idx=None` and skips
  cross-attn entirely). It does NOT hit the focal path.
- **proj_dropout=0.1** targets the shared trunk — affects all three
  streams (strong + focal + unsup). This is the only knob that
  regularizes the unsupervised contrastive paths (through
  `query_from_detached` → `self.proj(emb)`).
- **weight_decay=1e-2** targets attention weights and prototype
  matrices via AdamW's decoupled formulation. Affects all parameter
  groups uniformly.
- **label_smoothing=0.01** targets the BCE logits. Deliberately small
  (not the 0.1 typical for cross-entropy) because multi-hot binary
  labels are already sparse — aggressive smoothing would distort the
  positive-class probability.

Empirically (v4 S22 smoke), this budget peaks at step 600 of 20760
planned. Two interpretations remain open: (a) 5M-param model with
frozen features genuinely converges in one epoch of focal coverage;
(b) budget is too aggressive and blocks further gains. Seed replicate
or a v4.1 ablation with dropout 0.05 is the diagnostic.

## LOSO evaluation

**9 folds** — sites with any labeled SC windows: `S03, S08, S09, S13,
S15, S18, S19, S22, S23`. Each fold:

1. Split `strong` on `site_idx`: train on 8 sites, evaluate on the
   held-out site's labeled windows.
2. Reset the full model (random init).
3. Train with the **locked** `λ_fam` and `λ_site` from the S22 sweep
   below, same sampler, same schedule.
4. Evaluate with per-species AUROC on the held-out set.

Every fold uses an identical recipe — only the held-out site varies.

### Reporting strategy

S22 alone = 954/1478 = 65% of all labeled SC windows. Other folds have
30–120 windows each. Naive cross-fold macro averaging is dominated by
noise from tiny folds. Reporting:

| Metric | Role |
|---|---|
| **Pooled AUROC** (primary) | Concat held-out predictions across all 9 folds, compute per-species AUROC, macro across species. Each window contributes once; fold size is irrelevant. Directly comparable to exp 106's 0.749. |
| **S22 fold AUROC** (decision metric) | 954-window fold, the only one with stable per-fold variance. Used for hyperparameter tuning and ablations. |
| **Other 8 folds** | Noisy unseen-site estimators, reported as a variance check (per-fold table + macro). Not point estimates. |
| **Weighted fold mean** | Sanity check that pooled and weighted agree. |

Subset breakdowns (all computed on the pooled predictions): overall,
bird vs non-bird, amphibians, with-focal vs without-focal, Insecta
(reported separately, expected near-random), has-family-sibling-with-focal
vs orphaned. Unlearnable species (no focal + no 739 positives + no
family sibling with focal) excluded from aggregate, listed separately.

**Leakage audit.** Unsupervised SC is shared across folds (no labels to
leak); focal data is site-less. Standard semi-supervised LOSO.

### Pre-registered `λ_fam` sweep on S22 (doubles as family ablation)

Before the full 9-fold LOSO, sweep `λ_fam ∈ {0, 1e-3, 1e-2, 1e-1, 1}`
on the S22 fold alone. `λ_fam = 0` is the flat ablation (no family
shrinkage). Track three subset AUROCs on S22's held-out set:

| Subset | Expected behavior |
|---|---|
| Data-rich birds (Aves with ≥50 focal recordings) | **Flat** across `λ_fam`. A drop at high `λ_fam` = over-shrinkage distorting prototypes that have enough data to speak for themselves. |
| Medium-resource amphibians | **Rise** from `λ_fam = 0`, saturate, possibly drop at very high `λ_fam`. Peak is the target. |
| Overall S22 AUROC | Secondary — used only if the subset curves are ambiguous. |

**Decision rule.** Pick the smallest `λ_fam` that saturates the
amphibian curve without measurably reducing the bird curve. If no
`λ_fam > 0` beats `λ_fam = 0` on the amphibian subset by ≥ 0.5 AUROC,
v2 drops `L_family` entirely.

Lock the chosen `λ_fam` and run the full 9-fold LOSO without re-tuning.

## Kaggle submission path

1. **Tune `λ_fam` on S22** (pre-registered sweep above). Lock `λ_fam`
   and `λ_site`.
2. **LOSO validation** on 9 folds with locked hyperparameters →
   architecture gate is pooled AUROC ≥ 0.749; target ≥ 0.76.
3. **Final training on all data** (focal + 739 labeled SC + 127K
   unlabeled SC, no fold hold-out), locked hyperparameters, single-GPU
   run ~1 h.
4. **Package**: exp 107 checkpoint (~5.0M params) + Perch v2 ONNX
   (reused from exp 102/105 Kaggle package) + preprocessing. `L_family`
   is training-time only — the shipped checkpoint contains no
   family-level parameters.
5. **Inference**: parse `site_idx` from the row_id
   (`BC2026_Test_NNNN_S##_<datetime>_<window>`) → Perch v2 → 1536-d
   embedding → exp 107 model → 234-d multi-label probs per 5-sec window.
6. **OOD-site fallback**: if `site_idx ∉ {S01..S23}` (hidden rerun uses
   new sites), set `site_score = 1` to fall back to pure species
   retrieval, or equivalently omit the site head from the deploy
   bundle and always pass `site_idx=None`.
7. **Sample-submission sanity check**: first submission is a trivial
   forward with exp 102 weights + a minimal exp 107 stub, to confirm
   the row_id format matches the sample.
8. **Standalone LB submission.**
9. **Ensemble with exp 102** best checkpoint: probability-space
   average, blend weight tuned on LOSO (not on LB). Prior hypothesis:
   exp 107 helps amphibians and low-resource birds; exp 102 dominates
   on common birds. Insect morphotypes are unreachable in v1 for
   either model.

### Submission checklist

- [ ] `λ_fam` sweep on S22 fold
- [ ] LOSO primary validates architecture (pooled AUROC ≥ 0.749)
- [ ] Final retraining on all data
- [ ] Model export (PyTorch checkpoint)
- [ ] Kaggle package assembly
- [ ] Sample-submission sanity check
- [ ] Standalone LB submission
- [ ] Ensemble LB submission

## Directory structure

```
user/anthony/experiments/107-hopfield-prototypes/
├── proposal.md                       # thesis, method, success criteria
├── design.md                         # ← this file (interface + math spec)
├── docs/
│   └── star-schema-dimensional-analysis.md    # tensor framework + Hopfield formalism
├── pyproject.toml                    # editable dep on birdclef_106
├── src/birdclef_107/
│   ├── __init__.py
│   ├── model.py                      # 4 nn.Module classes + affinity_matrix helper
│   ├── losses.py                     # compute_loss(model, batch, family_idx, ...)
│   ├── family_utils.py               # build_family_index + DEFAULT_FAMILIES_CSV
│   ├── assets/
│   │   └── families.csv              # iNat species → family mapping (234 rows)
│   ├── label_prep.py                 # Phase B: materializes the 3 streams
│   ├── data.py                       # Phase B: datasets + 3-stream sampler
│   ├── train.py                      # Phase B: AdamW cosine loop
│   ├── evaluate.py                   # Phase C: LOSO + subset metrics
│   ├── kaggle_export.py              # Phase C: final retrain + submission packaging
│   └── cli.py                        # Phase B+: prepare-labels, train, evaluate, submit
├── tests/
│   ├── test_model_shapes.py
│   ├── test_gradient_routing.py
│   ├── test_affinity.py
│   ├── test_overfit_one_batch.py
│   └── test_family_index.py
├── configs/
│   └── base.yaml                     # Phase B: locked hyperparameters
├── scripts/
│   ├── fetch_families.py             # one-time iNat API call; writes assets/families.csv
│   └── analyze_coverage.py           # preflight long-tail + LOSO fold inventory
├── sbatch/
│   ├── _setup-venv.sh
│   ├── test.sbatch                   # Phase A smoke tests (cpu-small, ~30s warm)
│   ├── analyze_coverage.sbatch       # preflight (cpu-small)
│   ├── prepare_labels.sbatch         # Phase B: label prep (CPU)
│   ├── train_loso.sbatch             # Phase B: LOSO training (GPU)
│   └── train_final.sbatch            # Phase C: full-data retrain for Kaggle (GPU)
└── artifacts/                        # gitignored
    ├── labels/                       # per-window label parquets
    ├── checkpoints/
    │   ├── fold_{i}/best.pt
    │   └── final_full.pt             # for Kaggle submission
    ├── metrics/
    │   └── loso_results.json         # overall + subset breakdowns
    └── affinity/
        └── species_site_matrix.npz   # (234, 23) eval-time affinity
```

## Dependencies

**External**

| Package | Used for |
|---|---|
| `torch` | Model, training loop, GPU (RTX 6000 recommended) |
| `einops` | `einsum`, `rearrange` — memory-bank contractions |
| `polars` | Cached parquet IO (inherited from exp 106) |
| `numpy`, `scikit-learn` | Metrics (`roc_auc_score`) |
| `pyyaml` | Config loading |

**Cached artifacts consumed from exp 106** (via editable `birdclef_106` dep):

| Artifact | Used for |
|---|---|
| Perch v2 embeddings (265K focal + 739 labeled + 127K unlabeled) | Model input; read via `birdclef_106.data_loading` partition-aware loaders |

Perch logits are **not** consumed in v1 (no distillation, no
pseudo-labeling, no Perch-based filtering). Pooled embeddings only.

**Internal**

- `birdclef_106.data_loading` — cached parquet loaders, LOSO utilities,
  canonical 234-species ordering.

No prototype transfer from exp 106 — memory banks are randomly
initialized and trained end-to-end. Exp 106's `multi_prototypes.npz`
and `site_prototypes.npz` live in Perch's fixed logit space, not in
exp 107's learned projected space, and aren't dimensionally compatible.

## Testing plan

Four invariants enforced before any LOSO job, plus an integration test
on the real families.csv. All run on CPU via `sbatch/test.sbatch` in
under a minute on a warm venv.

| Test file | Invariant |
|---|---|
| `tests/test_model_shapes.py` | Default-shape construction succeeds with ~5.0M trainable params (v1 3.94M core + 1.05M cross-attention); both forward signatures produce `(B, 234)` probabilities in `[0, 1]`; both unsupervised energies return finite scalars. |
| `tests/test_gradient_routing.py` | Enforces the four gradient-routing invariants: (1) projection frozen under `L_site` and `L_site_xattn`; (2) species retrieval operator (`M_species`, `species.W_K`, `species.log_β`) frozen under `L_site_xattn` via explicit `.detach()`; (3) `cross_attn.*` and `M_site` both receive unsupervised gradient under `L_site_xattn`; (4) negative control — labeled BCE shapes the projection. |
| `tests/test_cross_attention.py` | Cross-attention module: output shape `(B, S, K_sp, D)`; zero-init identity at step 0; site-dependence emerges as `out_proj` trains; labeled BCE gradient reaches `out_proj` at init and `M_site` once attention is active; default-size outputs are not saturated (regression against the LayerNorm-std-blowup bug). |
| `tests/test_affinity.py` | `affinity_matrix` einsum matches a brute-force quadruple-loop reference on a tiny synthetic model (max abs err < 1e-5); default-shape output is `(234, 23)` and finite. |
| `tests/test_overfit_one_batch.py` | Tiny model (n_species=5, n_sites=2, d_proj=8) drives BCE on a single 16-window fake batch below 1e-2 in 200 AdamW steps. Catches wiring errors in Platt head, logsumexp aggregation, or einsum contractions. |
| `tests/test_family_index.py` | `build_family_index` on the package-bundled CSV returns a `(234,)` long tensor with 60–80 contiguous family ids; raises cleanly on missing species; is accepted by `model.family_shrinkage_loss` without shape errors. |
| `tests/test_data.py` | Data pipeline: LOSO folds are non-overlapping and complete; datasets emit correct tensor shapes; focal balancer upsamples rare species; end-to-end training step runs and emits all four loss components (`labeled`, `site`, `site_xattn`, `family`). |

## SLURM setup

| Job | Partition | Resources | Wall |
|---|---|---|---|
| Phase A smoke tests (`sbatch/test.sbatch`) | `cpu-small` | 2 CPU, 6 GB | ~30 s warm |
| Preflight coverage (`sbatch/analyze_coverage.sbatch`) | `cpu-small` | 2 CPU, 4 GB | ~10 min |
| Label prep (`sbatch/prepare_labels.sbatch`) | `cpu-small` | 4 CPU, 8 GB | ~30 min |
| LOSO training (`sbatch/train_loso.sbatch`) | `gpu-rtx6000` | 1× RTX 6000, 16 GB RAM | 12 h (S22 sweep + 9 folds sequential) |
| Final retraining (`sbatch/train_final.sbatch`) | `gpu-rtx6000` | 1× RTX 6000, 16 GB RAM | ~1 h |

**Venv.** Reuses `~/scratch/birdclef-2026/.venv-106` (torch + polars
already installed) and layers `einops` + `pytest` + the editable exp
107 package on top via `sbatch/_setup-venv.sh`. Per `CLAUDE.md`, never
install a venv under `$HOME`.

## Connection to exp 106

Reused infrastructure:
- Cached Perch v2 embeddings (739 + 127K + 265K)
- Partitioned parquets (`by_species/`, `by_site/`)
- LOSO fold utilities
- Canonical species ordering (`build_species_list`)

**Not inherited**: exp 106's `multi_prototypes.npz` and
`site_prototypes.npz`. Exp 107's memory banks are initialized from
random Gaussian noise and trained end-to-end in the learned projection
space; exp 106's prototypes live in Perch's fixed logit space and
aren't dimensionally compatible.

Exp 107 adds:
- Label preparation pass (bag-label focal + unsupervised SC preparation)
- PyTorch memory-bank model with flat species bank, cross-attention
  site conditioning, two unsupervised Hopfield energy terms, and
  `L_family` partial pooling (~400 LOC in `src/birdclef_107/model.py`,
  factored into 5 `nn.Module` classes)
- One-phase training loop with per-species-balanced focal sampling
- LOSO cross-validation wrapper with pooled AUROC + subset metrics
- Kaggle packaging + ensemble blend

## Open decisions

1. ~~**`λ_site` default.** `0.1` is a guess.~~ **RESOLVED in v3.**
   `λ_site = λ_xattn = 0.01` (split from v2's shared `λ_site=0.1`
   because the two contrastive terms have different magnitudes).
   Locked empirically by v3 smoke showing the `unsup/labeled` ratio
   stays in [0.5, 2.0] at this scale.
2. **`λ_fam` default.** `1e-2` is an order-of-magnitude guess based on
   the effective-sample-size interpretation (shrinkage equivalent to
   ~a handful of virtual samples, matching the amphibian median of 6
   recordings). The pre-registered S22 sweep
   `{0, 1e-3, 1e-2, 1e-1, 1}` locks the actual value used in LOSO.
   **Status: sweep still pending (deferred until v4 LOSO).**
3. **`W_Q` under unsupervised losses.** Currently frozen via `no_grad`
   in `query_from_detached`, applied by both
   `unsupervised_site_contrastive` and
   `unsupervised_cross_attn_contrastive`. If S22 LOSO shows site
   retrieval undertrained, revisit by swapping the `no_grad` for a
   plain `.detach()` on the backbone output and letting `W_Q.weight`
   receive gradient again. **Status: still open — have not tested the
   alternative.**
4. ~~**Shared vs split `λ`.**~~ **RESOLVED in v3.** Split into
   `λ_site` and `λ_xattn`. The v2 shared-weight design was discarded
   when contrastive reformulation changed the relative magnitudes.
5. ~~**Softness of the cross-attn aux.**~~ **RESOLVED in v3.** The
   `softplus(-logsumexp_s)` formulation collapsed to "match anything"
   in v2 — `L_xattn` dropped 17× in 300 steps and corrupted labeled
   retrieval. Replaced with site-InfoNCE (`L_xattn_contrastive`)
   which forces *peakier retrieval at the correct site than at wrong
   sites*; constant-across-sites profiles give zero gradient in that
   direction. Saturation/collapse diagnostic tests cover this.
6. **v4 regularization calibration (NEW).** v4 S22 smoke showed
   `without_focal` subset regressed from v3's 0.787 to v4's 0.661
   (n=28) while `with_focal` improved from 0.702 to 0.769 (n=206).
   The current regularization budget (attn_dropout=0.1, wd=1e-2,
   label_smoothing=0.01, strong_stride=4) may be miscalibrated for
   species that rely on cross-attention alone. A v4.1 ablation with
   `attn_dropout=0.05` (halved) is the natural next diagnostic.
   **Status: open.**
7. **Bin B — Perch pseudo-labels on unlabeled SC (NEW).** Under
   consideration as the next lift if v4 LOSO lands near but below
   v4 S22 smoke. Exp 106 has a species mapping
   (`perch_species_mapping.json`, 209/234 species) and cached Perch
   logits (14,795-d). Pseudo-label loss would trains projection +
   species head via soft targets, orthogonal to the contrastive
   parameter-path for `M_site` + `cross_attn`. **Status: designed,
   not implemented.**
