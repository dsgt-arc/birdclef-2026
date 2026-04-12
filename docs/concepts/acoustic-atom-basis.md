# Concept: Acoustic Atom Basis for Species Prototypes

**Status**: nascent
**Created**: 2026-04-12
**Updated**: 2026-04-12

## Idea

Add a third memory bank `A : (N, D)` — a shared set of learned
"acoustic atoms" in the same 512-d projection space as the species and
site banks — trained by Hopfield energy minimization on all 127K
unlabeled soundscape windows. Then parameterize species prototypes as
sparse combinations of these atoms:

$$
M^{(s)}_k = \sum_n C_{s,k,n} \cdot A_n
$$

where `C : (S, K_s, N)` are per-species coefficients trained by labeled
BCE. The atom basis is shaped by target-domain acoustic structure; the
coefficients route that structure into species-specific prototypes.

This changes the species bank from `S·K_s·D` free parameters (flat
prototypes) to `S·K_s·N + N·D` (coefficients + atoms). When
`N ≪ S·K_s = 1872`, the species bank's effective capacity shrinks, and
low-resource species are forced to compose their prototypes from atoms
that are themselves populated by the full 127K unlabeled target-domain
windows. The atoms are the unsupervised contribution to species
retrieval — currently absent in exp 107 v1, where the 127K windows
only train the site head and produce a scalar modulator.

## Motivation

Exp 107 v1 uses 127K unlabeled soundscape windows exclusively for site
prototype clustering. The site head produces a per-window scalar
typicality that modulates species confidence via gated fusion — it can
**scale** species probability but cannot **redistribute** it among
species. That's a fundamental asymmetry: unlabeled data gets no seat
at the species table.

Meanwhile, low-resource species (amphibians with ~6 focal recordings,
insect morphotypes with zero) must learn 8 independent 512-d prototype
vectors from their own sparse data or collapse entirely onto the family
centroid via `L_family` shrinkage. The family centroid is shaped by
labeled BCE on siblings — helpful but not informed by target-domain
acoustic structure.

An atom basis addresses both:
1. Unlabeled soundscape windows get a **species-level** contribution
   path (atoms → species prototypes via coefficients) instead of just a
   scalar modulator.
2. Low-resource species inherit target-domain structure (dawn chorus,
   rain events, amphibian chorus archetypes) directly through the atoms
   their coefficients select — structural regularization beyond what
   `L_family` provides.

## Background

The atom-basis idea sits at the intersection of several established
lines:

- **Sparse coding / dictionary learning.** Species prototypes as sparse
  combinations of learned dictionary atoms. The dictionary (atoms) is
  trained unsupervised; the coefficients are task-supervised. Classical
  references: Olshausen & Field 1996 (sparse coding), Mairal et al.
  2009 (task-driven dictionaries).

- **SwAV-style clustering with Sinkhorn.** The anti-collapse mechanism
  for atom training — Sinkhorn-normalized assignments ensure each atom
  gets approximately equal assignment mass per batch, preventing
  degenerate solutions where all atoms converge to the data mean.
  Caron et al. 2020 (SwAV).

- **Modern Hopfield networks (Ramsauer et al. 2020).** The atom bank is
  a Hopfield store with `N` slots, trained by energy minimization on
  unlabeled data. The species bank becomes a parameterized query into
  the atom store: species prototypes are Hopfield-retrieved patterns
  from `A`, weighted by `C`. This is two-level retrieval — consistent
  with the framework already used for species and site heads.

- **Exp 107 v1 architecture** (this experiment). Species and site
  Hopfield retrieval over a learned 512-d projection of Perch v2
  embeddings. `L_site_unsup` trains the site bank on 127K unlabeled
  windows; the backbone is frozen under that loss. The atom bank would
  follow the same frozen-backbone pattern: `L_atom` uses
  `query_from_detached`, so atoms adapt to whatever feature space
  labeled BCE has shaped.

- **Exp 106 lesson.** Domain shift entangles with site structure in
  Perch's fixed logit space. The atom basis must NOT train the backbone
  (same invariant as `L_site_unsup`), or target-domain clustering signal
  contaminates the projection.

## Open Questions

- **What N?** Too few atoms and species expressiveness collapses (all
  species look alike). Too many and the basis is equivalent to the flat
  species bank with extra indirection. The sweet spot is probably
  `32 ≤ N ≤ 128` — enough for acoustic diversity but fewer than
  `S·K_s = 1872`. Needs a sweep on S22.

- **How to prevent atom collapse?** Minimizing `−lse_n` with no
  counter-force pulls all atoms toward the data mean. The site bank
  doesn't collapse because site indexing partitions the data into 23
  disjoint groups — atoms don't have that safety net. Options:
  (a) Sinkhorn-balanced assignment (SwAV), (b) entropy regularization
  on atom usage, (c) small atom-loss weight so BCE dominates and
  species-level signal keeps atoms apart. All add hyperparameters.

- **Sparse or dense coefficients?** If `C` ends up dense, the species
  bank is just a learned linear combination of atoms — over-parameterized
  relative to the flat bank. Sparsity regularization (L1 on C, or
  top-K masking) would realize the "sparse basis" benefit and reduce
  effective parameters for low-resource species. But sparsity introduces
  its own tuning knob.

- **Interaction with `L_family`.** `L_family` pulls `M_species` toward
  sibling centroids. With `M_species[s,k] = Σ_n C_{s,k,n}·A_n`, the
  shrinkage operates on the composed output — it pulls coefficients so
  the resulting prototypes are near the family mean. Does this compose
  cleanly with the atom basis, or do the two regularizers fight?

- **Do atoms discover non-bird acoustic modes?** The backbone is shaped
  by (bird-rich) weak focal BCE. If the projected unlabeled windows
  cluster in bird-centric acoustic axes, the atoms are bird-adjacent
  regardless of the unsupervised clustering. The "birds pay for the
  projection, non-birds ride on it" thesis extends to atoms — question
  is whether the shared projection space has enough non-bird structure
  for the atoms to pick up.

- **Inference cost.** Species retrieval with the atom basis requires
  computing `M_species[s,k] = C[s,k,:] @ A` at forward time, then the
  usual `lse_k(β · q_r · W_K · M_{s,k})`. If `N` is small and `C` is
  sparse, this is cheap. If `N` is large and `C` is dense, it's slower
  than the flat bank. Pre-computing `M_species = C @ A` once per forward
  amortizes the cost.

## Testable Hypotheses

1. **Atom basis improves LOSO AUROC for species without focal data.**
   If the atom basis helps, its lift should concentrate on the
   "non-bird without focal" regime where `L_family` is currently the
   only rescue mechanism. Metric: pooled AUROC on the without-focal
   subset, atom-basis model vs flat-species-bank model, on the same
   S22 fold.

2. **Atom basis acts as a stronger regularizer than `L_family` alone
   for low-resource species.** If both `L_family` (λ_fam > 0) and the
   atom basis are active, per-species AUROC for species with < 10
   focal recordings should be higher than either alone. Metric:
   per-species AUROC on the < 10 focal subset.

3. **Atoms discover interpretable acoustic modes.** After training, the
   N atoms should cluster into recognizable acoustic archetypes (dawn
   chorus, rain, insect wall, amphibian chorus, etc.) when listened to
   via nearest-neighbor soundscape windows. Metric: qualitative
   inspection of nearest-neighbor audio for each atom.

4. **N = 64 is a reasonable default.** If the atom basis is insensitive
   to N in the range 32–128, the exact value doesn't matter and the
   concept is robust. If it's sharply sensitive, the basis is fragile.
   Metric: S22 AUROC as a function of N.

## Connections

- **Based on**: exp 107 v1 (Hopfield prototype retrieval), exp 106
  (star-schema, domain shift analysis), SwAV (Sinkhorn-balanced
  unsupervised clustering), sparse coding / dictionary learning
- **Related to**: `docs/concepts/soundscape-domain-shift.md` (the
  focal↔SC gap that motivates target-domain representation learning),
  exp 107 design.md §"Open decisions" item 3 (W_Q under L_site_unsup —
  same question of how much unsupervised signal to let into shared
  parameters)
- **Could lead to**: exp 107 v2 with a fourth loss term `L_atom`, or a
  standalone ablation experiment comparing flat vs atom-basis species
  banks on the S22 fold

## Notes

**Alternative directions considered and deferred:**

- **Self-train species bank via Sinkhorn-balanced Hopfield energy
  (Direction B).** Simpler: no new bank, just add a loss that pulls
  `M_species` toward target-domain cluster centroids with load-balanced
  assignment. Weaker than the atom basis because species prototypes
  stay independent — no shared vocabulary. Good for a cheap ablation
  before committing to the atom basis.

- **EMA self-distillation on unlabeled SC (Direction C).** Teacher-student
  with stop-grad EMA. Conservative, well-studied, but subject to
  confirmation bias. Exp 102 was burned by per-window pseudo-label
  lock-in. Not recommended for v1 given recent history.

- **SSL head on the backbone (Direction D).** Contrastive / masked
  prediction on unlabeled SC windows. The most aggressive option —
  breaks the backbone-freeze invariant. Biggest potential reward (trains
  the projection on 127K target-domain windows), biggest risk (site
  entanglement reappears). Reserve for Phase C if Phase B LOSO shows the
  backbone can't bridge focal↔SC.

**Timing.** This concept is parked for after exp 107 Phase B delivers
LOSO results with the plain three-term loss. The per-subset breakdown
(bird vs non-bird, with-focal vs without-focal) will determine whether
the atom basis addresses a real bottleneck or a hypothetical one. If
the flat species bank + `L_family` already handles low-resource species
adequately, the atom basis is unnecessary complexity.
