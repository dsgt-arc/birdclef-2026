# Design: Prototype Retrieval for Amphibian-Dominated Soundscapes

## Overview

A PyTorch memory-bank classifier over frozen Perch v2 embeddings. Species
and site prototypes live in a learned 512-d projection space; species
retrieval is trained via BCE on strong + weak focal labels; site
retrieval is trained via **Hopfield energy minimization** on 127K
unlabeled soundscape windows. Both banks compose at inference via gated
fusion on per-species Platt-calibrated probabilities.

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
| Per-species calibration | Built-in Platt `sigmoid(a_s · score + b_s)` | Load-bearing for Bayesian gated fusion (`species · site_score` needs calibrated probabilities); also the only per-species differentiator for zero-data orphans |
| Gated fusion | `g · species + (1 − g) · species · site_score`, `g = σ(Linear(q))` | Per-window mixing between species-only and species × site |
| Loss composition | `L_labeled + λ_site · L_site_unsup + λ_fam · L_family` | Three terms; `L_family` is hierarchical-Bayes partial pooling on `M_species` |
| Projection gradient on unsupervised loss | **Frozen** (`torch.no_grad()` through both MLP and `W_Q`) | Prevents the shared backbone from encoding site-identifying artifacts — stronger than the "detach only proj output" sketch, which would still train `W_Q.weight` via its forward matmul |
| Initialization | Random (no k-means) | Soft-EM gradient identity justifies BCE+Hopfield convergence without warm start |
| Tensor operations | einops `einsum`, `reduce`, `rearrange` | Project convention; self-documenting contractions |

**Total parameters: ≈ 3.94M** at default shapes
(projection 2.10M + `W_Q`/`W_K_species`/`W_K_site` 0.79M + `M_species` 0.96M +
`site_memory` 0.09M + per-species Platt 468 + two scalar `log_β` + gate 513).

## Built-in Platt scaling

The species head applies per-species affine calibration before the
sigmoid:

$$
\text{score}_s = \sigma\!\left(a_s \cdot \frac{1}{\beta_s}\,\text{lse}_k\!\left(\beta_s \cdot q_r \cdot W_{K,s} \cdot M_{s,k}\right) + b_s\right)
$$

where `a_s` (scale) and `b_s` (bias) are learnable per-species parameters
initialized to identity.

Two reasons calibration belongs in the model:

1. **Multi-label composition needs calibrated probabilities.** The gated
   fusion multiplies `species · site_score`. This is only a valid
   Bayesian composition ("probability species s is present and
   compatible with the site") when both factors are proper
   probabilities. Without per-species Platt, `species · site_score` is a
   product of arbitrary-scale sigmoid outputs; systematic per-species
   offsets break the composition.
2. **Cleaner BCE gradients.** Uncalibrated scores produce biased BCE
   gradients; per-class scale + bias absorbs systematic offsets that
   would otherwise be learned into the projection.

The site head does not need per-site Platt — it emits a single scalar
per window, and the fusion gate handles systematic bias.

Beyond fusion hygiene, `b_s` is the **only** mechanism that
differentiates zero-data orphans from their data-rich family siblings
once partial pooling has pulled them onto the same family centroid (see
`star-schema-dimensional-analysis.md` "Orphan collision limit").

## Model interface

The implementation lives in `src/birdclef_107/model.py`. Four
`nn.Module` classes, all in one file:

### `PerchProjection(d_input, d_hidden, d_proj)`

Shared frozen-input → projected-query backbone plus the shared query
rotation `W_Q`. Every retrieval path routes through this module.
Supervised losses shape it; unsupervised losses call a dedicated
"frozen-pass" entry point that wraps the entire backbone in
`torch.no_grad()` so neither the MLP nor `W_Q.weight` receives gradient.

| Method | Signature | Contract |
|---|---|---|
| `forward(emb)` | `(B, d_input) → ((B, d_proj), (B, d_proj))` | Returns `(q_base, q_query)`. `q_base` feeds the fusion gate (pre-`W_Q`); `q_query` feeds both retrieval heads (post-`W_Q`). |
| `query_from_detached(emb)` | `(B, d_input) → (B, d_proj)` | Runs the full backbone inside `torch.no_grad()` and returns `q_query` with `requires_grad=False`. Used by `L_site_unsup`. |

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
| `forward(q_r)` | `(B, d_proj) → (B, n_species)` | Returns calibrated probabilities in `[0, 1]`. Computes `lse_k(β · q_r · W_K · M_{s,k}) / β`, then `σ(a_s · · + b_s)`. |
| `family_shrinkage_loss(family_idx)` | `(n_species,) long → scalar` | `L_family` per star-schema §"Family partial pooling" — squared Euclidean to the stop-grad uniform family mean. See math below. |

### `HopfieldSiteHead(n_sites, K, d_proj)`

Flat site memory bank with two retrieval paths that share the same
prototype store, key projection, and inverse temperature:

| Method | Signature | Contract |
|---|---|---|
| `forward(q_r, site_idx)` | `(B, d_proj), (B,) long → (B,)` | Per-window sigmoid typicality. Gathers `M[site_idx]` and returns `σ(lse_k(β · q_r · W_K · M_{l_w,k}) / β)`. |
| `retrieval_energy(q_r, site_idx)` | `(B, d_proj), (B,) long → scalar` | `L_site_unsup` path: `mean(-lse_k(β · q_r · W_K · M_{l_w,k}) / β)`. Does **not** itself detach — the caller must pass a detached `q_r`. |

The two methods share a private `_aggregate(q_r, site_idx)` helper that
computes the `(B,)` logsumexp aggregate. The inference-time sigmoid
produces a typicality scalar in `[0, 1]`; the training-time
energy minimization uses the raw aggregate directly.

### `PrototypeRetrievalModel(n_species=234, n_sites=23, K_species=8, K_site=8, d_input=1536, d_hidden=1024, d_proj=512)`

Top-level composition of the three sub-modules with a per-window fusion
gate. Delegates everything that isn't fusion to the sub-modules.

| Method | Signature | Contract |
|---|---|---|
| `forward(emb, site_idx=None)` | `(B, 1536), (B,) long\|None → (B, 234)` | Calibrated multi-label probs. With `site_idx=None` (focal path) returns species-only scores; otherwise gated fusion `g · sp + (1 − g) · sp · site`. |
| `unsupervised_site_energy(emb, site_idx)` | `(B, 1536), (B,) long → scalar` | Calls `self.projection.query_from_detached(emb)` then `self.site_head.retrieval_energy(...)`. |
| `family_shrinkage_loss(family_idx)` | `(n_species,) long → scalar` | Delegates to `self.species_head.family_shrinkage_loss`. |

The shipped Kaggle checkpoint needs `projection + species_head + gate`
only. The site head can be omitted entirely for the OOD-site fallback
by declining to load it, and the forward pass with `site_idx=None`
works unchanged.

### `affinity_matrix(model)` (module-level helper)

Computes the eval-time species × site affinity as a
`(n_species, n_sites)` tensor by folding every `(k_s, k_l)` prototype
pair into a single logsumexp per `(s, l)` cell. Used for interpretation
and ecological reporting — not part of any training forward pass. See
`star-schema-dimensional-analysis.md` §"Species × Site affinity" for
the derivation. The implementation must agree on a tiny synthetic
fixture with a brute-force quadruple-loop reference — enforced by
`tests/test_affinity.py`.

## Mathematical specification

Notation follows `docs/star-schema-dimensional-analysis.md`. `B` is
batch, `S = 234` species, `L = 23` sites, `F = 67` families, `K_s = K_l
= 8` prototype slots, `D = 512`, `E = 1536`, `H = 1024`.

**Backbone.** For an input embedding `emb ∈ ℝ^E`:

$$
q = \text{LayerNorm}\big(W_2\,\text{GELU}(W_1\,\text{emb} + b_1) + b_2\big), \qquad q_r = W_Q\,q
$$

`q` is the base projection (pre-`W_Q`), used only by the fusion gate.
`q_r` is the query-space projection, used by both retrieval heads.

**Species retrieval + Platt calibration.**

$$
\text{score}_s(q_r) = \sigma\!\left( a_s \cdot \frac{1}{\beta_s}\,\text{lse}_{k=1..K_s}\!\big(\beta_s \cdot q_r^\top W_{K,s}\,M^{(s)}_k\big) + b_s \right)
$$

where `a_s, b_s` are the per-species Platt parameters, `β_s = exp(log_β_species)`,
and `M^{(s)}_k ∈ ℝ^D` is the k-th prototype of species `s`. The
logsumexp is the modern-Hopfield retrieval energy — see
star-schema §"Logsumexp over K as soft EM".

**Site retrieval (two paths with a shared aggregate).** For a window
with known site `l_w`:

$$
\text{agg}_{\text{site}}(q_r, l_w) = \frac{1}{\beta_l}\,\text{lse}_{k=1..K_l}\!\big(\beta_l \cdot q_r^\top W_{K,l}\,M^{(l_w)}_k\big)
$$

$$
\text{score}_{\text{site}}(q_r, l_w) = \sigma(\text{agg}_{\text{site}}), \qquad
\mathcal{L}_{\text{site\_unsup}} = -\,\mathbb{E}_{(q_r, l_w) \sim \text{unsup}}\,\text{agg}_{\text{site}}(q_r, l_w)
$$

The inference-time sigmoid produces a typicality score in `[0, 1]`
consumed by the fusion gate; the training-time loss is a direct
Hopfield energy minimization (mean negative aggregate). Both use the
same underlying `_aggregate` function — that's a correctness invariant,
not an optimization.

**Gated fusion.**

$$
g(q) = \sigma(w_g^\top q + b_g) \in [0, 1], \qquad
\text{score}(w, s) = g(q) \cdot \text{score}_s + (1 - g(q)) \cdot \text{score}_s \cdot \text{score}_{\text{site}}
$$

Broadcasting: `g` is `(B, 1)` so it broadcasts against the species
scores `(B, S)` cleanly. The `rearrange(site_score, "b -> b 1")`
reshape puts the site typicality on the same footing. A past iteration
collapsed `g` to `(B,)` with a `.squeeze(-1)` and the broadcast
silently broke — `(B,) × (B, S)` is ill-defined for `B ≠ S`, caught by
`tests/test_model_shapes.py::test_forward_with_site_returns_batch_species_shape`.
Keep `g` two-dimensional.

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
\mathcal{L} = \underbrace{\text{BCE}(\text{sp}_{\text{strong}}, y_{\text{strong}}) + \text{BCE}(\text{sp}_{\text{weak}}, y_{\text{weak}})}_{\mathcal{L}_{\text{labeled}}} + \lambda_{\text{site}}\,\mathcal{L}_{\text{site\_unsup}} + \lambda_{\text{fam}}\,\mathcal{L}_{\text{family}}
$$

Defaults: `λ_site = 0.1`, `λ_fam = 1e-2`. Locked after the S22
pre-registered sweep (below). No per-term ramps, no warmup, no
asymmetric weight decay.

**Gradient routing.** Which parameters does each loss touch?

| Parameter | `L_labeled` | `L_site_unsup` | `L_family` |
|---|:-:|:-:|:-:|
| `projection.proj` (MLP) | ✓ | ✗ (frozen via `no_grad`) | ✗ |
| `projection.W_Q` | ✓ | ✗ (frozen via `no_grad`) | ✗ |
| `species_head.W_K, log_β, cal_scale, cal_bias` | ✓ | ✗ | ✗ |
| `species_head.M` (= `M_species`) | ✓ | ✗ | ✓ (LHS only; target is stop-grad) |
| `site_head.W_K, M, log_β` | ✓ (via gated fusion on strong SC) | ✓ | ✗ |
| `gate` | ✓ | ✗ | ✗ |

Two things worth flagging against the star-schema routing table:

1. **`W_Q` under `L_site_unsup`.** We freeze `W_Q` (no_grad), whereas
   the star-schema table currently lists it as trained. Rationale:
   `W_Q` is a shared query transformation across species and site
   retrieval, and the exp 106 lesson ("site entangled with domain
   shift") argues for conservative treatment of shared parameters
   under the site-clustering loss. If site-driven rotation of `W_Q`
   turns out to be load-bearing, we can revisit by loosening the
   `query_from_detached` implementation. Until then, the stronger
   invariant holds.
2. **`species_head.W_K` under `L_site_unsup`.** The routing table
   correctly has this as ✗: `L_site_unsup` only flows through
   `site_head`, so species-side retrieval parameters are untouched.

The `projection` detach invariant is verified by
`tests/test_gradient_routing.py::test_unsup_site_energy_does_not_update_projection`,
which asserts zero gradient across `model.projection.parameters()`
after a pure `L_site_unsup` backward pass, and non-zero gradient on the
site head's three trainable parameter groups.

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
site metadata and flow exclusively through `L_site_unsup`.

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

### Schedule

- **Optimizer**: AdamW, uniform weight decay `1e-5`.
- **LR**: `3e-4` cosine-decayed across 30–40 epochs.
- **Early stopping**: LOSO AUROC checked every 5 epochs on the held-out
  fold; stop on no-improvement patience of 2 checks.
- **No ramps or warmup**, no per-term loss schedule, no asymmetric
  weight decay. `L_family` replaces the earlier design's weight-decay
  asymmetry on a "species delta" tensor.

Three streams compose into a single per-step loss via `compute_loss` in
`src/birdclef_107/losses.py`; the training loop (Phase B) owns the
sampler plumbing.

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
4. **Package**: exp 107 checkpoint (~3.94M params) + Perch v2 ONNX
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
| `tests/test_model_shapes.py` | Default-shape construction succeeds with ~3.94M trainable params; both forward signatures produce `(B, 234)` probabilities in `[0, 1]`; `unsupervised_site_energy` returns a finite scalar. |
| `tests/test_gradient_routing.py` | Under a pure `L_site_unsup` backward, every parameter under `model.projection` has zero gradient, while `site_head.M`, `site_head.W_K.weight`, and `site_head.log_beta` all have non-zero gradient. Negative control: labeled BCE gives the backbone non-zero gradient. |
| `tests/test_affinity.py` | `affinity_matrix` einsum matches a brute-force quadruple-loop reference on a tiny synthetic model (max abs err < 1e-5); default-shape output is `(234, 23)` and finite. |
| `tests/test_overfit_one_batch.py` | Tiny model (n_species=5, n_sites=2, d_proj=8) drives BCE on a single 16-window fake batch below 1e-2 in 200 AdamW steps. Catches wiring errors in Platt head, logsumexp aggregation, or einsum contractions. |
| `tests/test_family_index.py` | `build_family_index` on the package-bundled CSV returns a `(234,)` long tensor with 60–80 contiguous family ids; raises cleanly on missing species; is accepted by `model.family_shrinkage_loss` without shape errors. |

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
- PyTorch memory-bank model with flat species bank, unsupervised site
  Hopfield energy loss, and `L_family` partial pooling (~300 LOC in
  `src/birdclef_107/model.py`, factored into 4 `nn.Module` classes)
- One-phase training loop with per-species-balanced focal sampling
- LOSO cross-validation wrapper with pooled AUROC + subset metrics
- Kaggle packaging + ensemble blend

## Open decisions

1. **`λ_site` default.** `0.1` is a guess. Tune on LOSO if the
   unsupervised site loss either dominates training (too high) or
   produces no visible effect on the site bank's prototype diversity
   (too low).
2. **`λ_fam` default.** `1e-2` is an order-of-magnitude guess based on
   the effective-sample-size interpretation (shrinkage equivalent to
   ~a handful of virtual samples, matching the amphibian median of 6
   recordings). The pre-registered S22 sweep
   `{0, 1e-3, 1e-2, 1e-1, 1}` locks the actual value used in LOSO.
3. **`W_Q` under `L_site_unsup`.** Currently frozen via `no_grad` in
   `query_from_detached`. Star-schema §"Training signal routing"
   currently lists `W_Q` as trained — the prose, formula, and routing
   table disagree there. The code enforces the stronger invariant (see
   "Gradient routing" table above). If the S22 sweep shows site
   retrieval undertrained, revisit by swapping the no_grad for a
   plain `.detach()` on the backbone output and letting `W_Q.weight`
   receive gradient again.
