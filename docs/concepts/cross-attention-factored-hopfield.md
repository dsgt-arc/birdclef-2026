# Concept: Cross-Attention Factored Hopfield Retrieval

**Status**: exploratory/future-work seed
**Created**: 2026-04-12
**Updated**: 2026-04-12

## Idea

Replace exp 107 v1's scalar site gate with **attention-refined prototypes
retrieved via unchanged Hopfield energy**. Species and site prototypes
are refined by self-attention among themselves, then species prototypes
are cross-conditioned on the current window's site prototypes before
retrieval. The final scoring operator stays identical to v1 —
`lse_k(β · q · W_K · M)` — so "one retrieval mechanism" still holds,
but the prototypes `M` are now context-dependent.

The three attention blocks:

1. **Species self-attention** (once per forward): species prototypes
   attend over each other, learning family/acoustic similarity in
   parameter space. *Candidate replacement for `L_family`.*
2. **Site self-attention** (once per forward): site prototypes attend
   over each other, learning ecological/diel similarity across S01–S23.
3. **Species-site cross-attention** (per window): each window's species
   prototypes attend over its site's prototypes to produce a site-conditioned
   bank `M_species_conditioned[b, s, k]`. This is the factored
   interaction the proposal v1 explicitly flagged as the v2 candidate.

Hopfield retrieval then runs against `M_species_conditioned`:

$$
\text{sp\_score}_{b,s} = \sigma\!\left(a_s \cdot \frac{1}{\beta} \log\!\sum_k \exp(\beta \cdot q_r[b] \cdot W_K \cdot M_{\text{cond}}[b, s, k]) + b_s\right)
$$

## Motivation

Exp 107 v1.2's smoke test on S22 plateaus at **0.65 AUROC**, below the
exp 106 ridge baseline of **0.749** on the same 262 LOSO windows. Two
specific design weaknesses became visible during debugging:

1. **Scalar site gate under-shares.** `g · species + (1-g) · species ·
   site_score` multiplies one scalar site score across all 234 species.
   Species and site prototypes never interact directly in the forward
   pass. Proposal v1.2 explicitly flagged this: *"This is a confidence
   gate, not a factored star schema in the strict sense. Proposal v1
   keeps this construction to avoid a parameter explosion; if it
   underperforms, a low-rank site→species projection becomes the first
   thing to try in v2."*
2. **Site bank has no labeled gradient path.** Under v1, `M_site` is
   trained *only* by `L_site_unsup`. The softplus bounding of that loss
   (forced by unbounded-energy divergence in the raw Hopfield formulation)
   saturates at random init and produces zero gradient — so in practice
   the 127K unlabeled windows contribute nothing to species performance.

Cross-attention directly addresses both: species prototypes are per-window
site-conditioned (solving #1), and labeled BCE flows backprop through
the cross-attention into `M_site` (solving #2). The unsupervised loss
becomes an auxiliary signal, not the sole signal.

A further benefit: species self-attention may **subsume `L_family`**.
If similar species' prototypes cross-reference each other via learned
attention, the stop-grad family-mean shrinkage is redundant. This is a
clean ablation (`λ_fam = 0` vs v1's `λ_fam = 1e-2`).

## Background

- **Exp 106** (`docs/concepts/../references/` — ridge 0.749 baseline on
  LOSO CV): established the empirical ceiling we're trying to beat. The
  star schema `species_centroid @ site_centroid` in Perch logit space
  hit 0.816 on focal-overlap species; v2 is the attempt to reproduce
  that cross-bank composition in a learned projection space.
- **Exp 107 v1.2** (`user/anthony/experiments/107-hopfield-prototypes/proposal.md`):
  factored memory-bank architecture, three-term loss. Phase B training
  smoke-tested at 0.65 AUROC on S22.
- **Ramsauer et al. 2020, "Hopfield Networks is All You Need"**: proves
  Hopfield energy `lse_k(β · q · M)` is equivalent to a single attention
  head. Composing attention blocks around Hopfield retrieval is
  architecturally natural — we're stacking attention heads, not bolting
  on a foreign mechanism.
- **Acoustic atom basis** (`docs/concepts/acoustic-atom-basis.md`):
  sibling concept that also addresses v1's "unlabeled windows don't
  help species" problem, but via a third shared bank rather than
  cross-attention. The two concepts are complementary — atoms could
  additionally feed through cross-attention.

## Open Questions

1. **Does cross-attention alone close the gap to ridge 0.749?** The
   minimal v2 (just block 3, species-site cross-attention) is the
   cleanest test. If it plateaus below 0.749, the bottleneck is deeper
   than information sharing between banks.
2. **Does species self-attention replace `L_family`?** If the ablation
   `λ_fam = 0` with self-attention matches `λ_fam > 0` without, we drop
   the shrinkage regularizer. If it helps specifically on the amphibian
   subset, the family-like structure is being learned implicitly.
3. **How do we train `M_site` once it has a labeled gradient path?**
   Keep `L_site_unsup` as an auxiliary regularizer, or drop it? The
   answer depends on whether unlabeled data adds signal beyond what
   labeled BCE can extract through the cross-attention.
4. **What is the right rank/dimension for the attention blocks?**
   Multi-head (4 × 128) vs single-head (1 × 512)? More heads =
   more parallel interaction patterns (family, habitat, time-of-day)
   but more parameters.
5. **Are species prototypes the right attention tokens?** Currently
   `(S · K, D) = (1872, 512)`. Alternative: pool each species to one
   token `(S, D)`, then attend, then broadcast back. Reduces attention
   matrix from 1872² to 234² ≈ 8× cheaper. Loses per-K specialization
   during attention.
6. **Does this break the OOD-site fallback?** At Kaggle inference, a
   held-out site (S24+) has no `M_site` entry. Cross-attention needs a
   key/value source for unseen sites. Fallbacks: (a) use global mean
   of training `M_site` as prior, (b) skip cross-attention path and
   fall back to v1-style species-only retrieval. Either is a small
   code branch at inference.

## Testable Hypotheses

1. **H1 — Cross-attention closes the gap to ridge.** v2-min (just
   block 3) on S22 achieves pooled AUROC ≥ 0.73, substantially
   reducing the v1 deficit. If achieved, the scalar-gate bottleneck is
   confirmed as the main v1 failure mode.
2. **H2 — Self-attention subsumes `L_family`.** v2-full with
   `λ_fam = 0` matches v2-full with `λ_fam = 1e-2` on amphibian subset
   (within 0.5 AUROC). If true, `L_family` is redundant in v2.
3. **H3 — Unsupervised site loss becomes optional.** v2-full with
   `λ_site = 0` matches v2-full with `λ_site = 0.1` on pooled AUROC.
   If true, the 127K unlabeled windows are best used as a data-augmentation
   source for future versions, not as a dedicated training loss.
4. **H4 — Orphan differentiation emerges.** The 3 rescuable amphibian
   orphans (Leptodactylidae/Microhylidae/Phyllomedusidae) achieve
   non-random AUROC in v2, whereas under v1 they collapse to the
   family centroid. Cross-attention should give them species-specific
   site responses that Platt-`b_s` alone cannot.

## Architecture Sketch (for reference)

```
Perch emb → projection → q_r : (B, D)

M_species : (234, K, D) ──┐
                          ├─→ Species self-attention ──┐
M_site : (23, K, D) ──────┤     (once per forward)      │
                          │                             │
                          └─→ Site self-attention ──────┤
                              (once per forward)         │
                                                         ▼
                                  Species-site cross-attention
                                  (Q=M_species_refined,
                                   K,V=M_site_refined[site_w])
                                  per window
                                                         │
                                                         ▼
                                  M_species_conditioned[b, s, k] : (B, S, K, D)
                                                         │
                                                         ▼
                                  Hopfield retrieval (unchanged)
                                  lse_k(β · q_r · W_K · M_species_conditioned)
                                                         │
                                                         ▼
                                  Platt → sp_score : (B, 234)
```

Approximate parameter budget:

| Block | Params |
|-------|--------|
| Species self-attention (1 block, 4 heads, D=512) | ~1M |
| Site self-attention (1 block) | ~1M |
| Species-site cross-attention (1 block) | ~1M |
| v1 total (for comparison) | ~3.9M |
| v2 total | ~6.9M |

## Training Signal Map Under v2

| Stream | Trains |
|--------|--------|
| `L_labeled` (739 strong + 265K weak focal BCE) | projection, W_Q, W_K_species, M_species, Platt, **all 3 attention blocks**, **M_site (via cross-attention backprop)** |
| `L_site_unsup` (127K unlabeled Hopfield energy) | M_site, W_K_site, log_beta_site (projection detached, same as v1) |
| `L_family` (optional — to ablate) | M_species only |

The critical change: `M_site` now receives gradient from BOTH losses,
with `L_labeled` providing the direct task-relevant signal the site
bank was missing in v1.

## Connections

- **Based on**: exp 107 v1.2 architecture (`user/anthony/experiments/107-hopfield-prototypes/proposal.md`),
  specifically the flagged v2 candidate *"low-rank site→species projection"*.
- **Related to**: `docs/concepts/acoustic-atom-basis.md` (complementary
  v2 direction — atoms as a third bank, orthogonal to cross-attention).
  Could be composed: atoms feed the species bank, cross-attention
  composes species with site.
- **Could lead to**: Exp 108 proposal — Attention-Refined Hopfield
  Retrieval. Implementation path: new branch off
  `exp107-partial-pooling-v1.2`, copy the `birdclef_107` package to
  `birdclef_108`, add attention blocks to `model.py`, keep all data
  loading and loss code.

## Notes

**Why not just use a transformer?** A transformer's attention output
IS the class score directly. We want a retrieval-based score because
(a) it exposes the prototype structure for interpretation (the
`affinity_matrix` visualization), (b) it preserves the unsupervised
Hopfield-energy pathway (the 127K unlabeled windows have a native
retrieval loss), (c) it keeps the Platt per-species calibration that
v1 designed for Bayesian fusion. A pure transformer loses all three.

**Why not just promote `affinity_matrix` to the forward pass?** That
function already computes a joint logsumexp over species × site
prototype pairs. Using it directly gives us Option 3 (joint Hopfield
over union of banks) from the design discussion — zero new params.
But it's the weakest form of sharing: species and site prototypes
compete in a shared softmax without conditioning each other. Cross-
attention is strictly more expressive: each species prototype gets a
site-specific adjustment, not just a competitive ranking against
site prototypes.

**Risk: overcapacity.** v2 has ~6.9M params trained by 739 strong SC
windows + 265K weak focal. The v1 model at 3.9M already overfits the
262 train-split SC windows (training loss → 0.017 while val AUROC
drops). Doubling parameters could amplify this. Mitigations: start
with v2-min (one block, ~1M extra params), heavier weight decay,
dropout in attention blocks, or multi-task auxiliary losses.

**Alternative: explicit CP factorization.** Instead of learned QKV
attention, parameterize the cross-bank interaction as:

$$
\Delta[s, l, k] = \sum_{r=1}^R \alpha_{\text{species}}[s, r] \cdot \alpha_{\text{site}}[l, r] \cdot B[r, k]
$$

with rank `R` as an explicit knob (~70K params for R=16). Less flexible
than attention but structurally transparent — the interaction tensor
is literally a sum of rank-1 outer products. Good fallback if attention
overfits.
