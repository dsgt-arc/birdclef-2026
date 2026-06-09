# Concept: Contrastive matched-filter exemplar mining

**Status**: developing — the **constructive successor** to the
`decouple-frequency-localization-from-focal-selection` falsification chain.
Strand 5 (D1) proved the at-chance species fail because the pipeline's *only*
template-selection criterion is self-recurrence, which generic broadband
energy satisfies. This concept fixes the **selection rule at mining time**.
Design fully specified, literature-reviewed, and stress-tested by two
adversarial PI reviews (the first killed the matrix-profile *algorithm* as
the vehicle; the surviving idea is re-grounded on the detector the pipeline
already ships). Not yet experimentally validated. Tracking: GitHub issue #49.
**Created**: 2026-05-19
**Updated**: 2026-05-19

## Idea

Select the canonical per-species template by a **discriminative contrast**,
not by recurrence alone. A template should be the focal subsequence that

1. **recurs within the species** — small nearest-neighbor distance to the
   same species' *other distinct* focal recordings (the "self-join"; this is
   the recurrence the pipeline already has), **and**
2. **is species-distinctive** — large nearest-neighbor distance to a
   cross-species focal corpus (the "AB-join"; the negative is exactly D1's
   `focal_specificity.build_pool`).

Rank candidates by the shapelet contrast `P_diff = P_AB − P_self` and take
the **exemplar** (the actual min-`P_self` focal crop) as the canonical
template — *not* the synthetic median patch the EM stage currently produces
(`mining.py:941,1032`). This replaces the recurrence-breadth gate
(`min_repetitions`, `mining.py:166,1127`) and the median-EM loop with one
selection rule, slotted at the existing pre-EM seam (`mining.py:1542→1550`).

The distance is the pipeline's **existing production detector**:
`_score_track`'s *jointly* (whole-patch) z-normalized 2-D spectrogram
cross-correlation (`mining.py:891`) — the `monitoR`/`ohun` matched filter.
The gallery and soundscape annotation path
(`_score_track`→NMS→`consolidate`) are unchanged: the detector already *is*
this distance. This adheres to the core thesis — a visually apparent
recurring motif in focal recordings, canonicalized as a real exemplar, used
as a gallery to annotate soundscapes by template matching.

## Motivation

Strand-5/D1 quantified the defect label-free: for 4/7 strand-3 at-chance
species the committed template is a generic broadband-energy detector
(`auc_vs_class` ≤ 0.47; a fixed cross-taxon attractor set shared across a
frog, a tinamou and a monkey), and it reaches the **validated consensus
route** with healthy `member_bg` d′ — so d′ actively misleads and the bug is
in *selection*, not calibration (strand 4) or domain shift. Self-recurrence
cannot exclude generic energy because generic energy recurs. The fix has to
add a discriminative term *at the point templates are chosen*. The cross-
species focal corpus is a free negative (the focal folder is curation
truth), and D1 already runs this exact distance against it.

## Background

**Antecedents (inspiration only, not the mechanism).** The self-vs-contrast
selection idea is the time-series *shapelet* (Ye & Keogh) under the
matrix-profile "unifying view" (Yeh/Zhu/Keogh): motifs, discords and
shapelets are one object under z-normalized subsequence distance, and
STUMPY's shapelet tutorial gives the exact contrast `P_diff = P_AB − P_AA`.
These are cited as intellectual antecedents for the *self-join vs AB-join*
idea **only**.

**Why the matrix-profile *algorithm* is rejected (not deferred).** Classical
matrix profile (STAMP/STOMP) is univariate. Its multivariate extension
mSTAMP (*Matrix Profile VI*) z-normalizes **each frequency bin
independently** and assumes a motif synchronous on a chosen k-bin subspace —
which destroys the **cross-band energy profile**. But that profile is
precisely what separates a structured species call from broadband energy,
i.e. the strand-5 discriminant. mSTAMP is therefore not merely ill-suited to
PCEN; it is *specifically destructive to this problem*. The correct vehicle
is the pipeline's existing *jointly* z-normed whole-patch 2-D matched filter,
which preserves the cross-band structure mSTAMP would erase, and which D1
already ran cross-species with clean separation (controls 0.74–0.87,
attractors 0.37–0.47). "Matrix profile" is demoted from mechanism to cited
antecedent; this is **not** a STUMPY/mSTAMP implementation.

- Diagnosis: `concepts/decouple-frequency-localization-from-focal-selection.md`
  § "v4 validation, strand 5"; instrument `src/data_analysis/auto_motif/focal_specificity.py`.
- Extrinsic validation strands (the adjudicating ground truth):
  `docs/auto-motif-soundscape-efficacy.md` (strands 3/4/5).
- Front-end / matched-filter precedent: PCEN long-distance detection
  (Lostanlen et al.); `monitoR`, `ohun` spectrogram cross-correlation.

## Open Questions

- **Nuisance recurrence (the strongest objection).** A per-site,
  per-microphone, per-channel, insect or rain artifact that recurs within a
  species' focal folder but is absent cross-species *maximizes* `P_diff` for
  the wrong reason — structurally the same trap as strand-5 (D1's pool is
  cross-species but still all *focal* audio). Carrying the distinct-
  recordings breadth requirement into the contrast only **partially**
  mitigates it; a systematic per-species artifact survives both. The
  label-free contrast cannot settle this — only strand-3 soundscape ground
  truth can.
- Is the 2-D NCC, well-behaved as D1's *screen*, equally well-behaved as an
  *optimization target*? It is frequency-translation-variant (harmonics /
  pitch shift), residually background-level sensitive despite z-norm, and
  patch-resize dependent (`normalize_resize_patch`).
- Compute plan for the full self-join vs AB-join over the focal corpus — MP's
  anytime/scalability guarantees are **not** inherited and must be paid for
  explicitly.
- When does a single-distance nearest-neighbor contrast lack power for
  pitch-variable / heterogeneous repertoires, triggering the Perch fallback?

## Testable Hypotheses

1. **Attractor kill (hypothesis, not theorem).** Re-mining the 4 mining-
   failure species (`326272, 25092, undtin1, 43435`) by `argmax(P_AB −
   P_self)` selects a *different* exemplar than the cross-taxon attractor —
   or correctly returns "no discriminative motif" — and D1 `auc_vs_class`
   on the re-mined template clears 0.55. (Downgraded from "generic energy
   can never win": D1 showed generic templates *can* score above chance vs
   class, so small `P_self` is not guaranteed.)
2. **Non-regression as falsification.** `22967` (hand-curated; by D1 the
   *most* focally-specific species, `auc_vs_class` 0.87) emerges with
   (near-)maximal `P_diff` and retains strand-3 soundscape ROC ≥ 0.90.
   Failure here **falsifies the method** — it is not a tunable.
3. **Ground-truth adjudication.** strand-3 `soundscape_eval` on the re-mined
   attractor species: ≥1 moves off chance, else recorded as "not detectable
   by template matching," not patched. This is the only instrument that can
   adjudicate the nuisance-recurrence risk; the label-free contrast is
   necessary, not sufficient.

## Connections

- **Based on**: `concepts/decouple-frequency-localization-from-focal-selection.md`
  § "v4 validation, strand 5" (D1); `src/data_analysis/auto_motif/focal_specificity.py`
  (`build_pool`, the cross-species negative + the `_score_track` reuse);
  `docs/auto-motif-pipeline.md`.
- **Related to**: the strand-4 supervised-calibration track (orthogonal:
  the 2/7 genuine recoverable domain-shift species `23158`, `chacha1`,
  which have high `P_diff` and fail only in-field); Perch embeddings kept as
  the **fallback adjudication representation** if the 2-D NCC contrast fails
  the attractor re-mine — *not* the default replacement.
- **Could lead to**: GitHub issue #49 (to be rewritten around this contrast,
  with the MP framing dropped and the nuisance-recurrence safeguard +
  falsification plan as the gate); a prototype on the 4+1 species; an A/B
  experiment proposal vs the production recurrence-breadth path behind an
  opt-in `AutoConfig` toggle (validated frog path stays default until
  parity is shown).

## Notes

**Simplification ledger.** Deletes the 26-D Gabor shape features
(`mining.py:679`), HDBSCAN clustering (`min_cluster_size`,
`allow_single_cluster`), the recurrence-breadth gate (`min_repetitions`),
and the median-EM loop (`max_em_iter`, `template_change_eps`,
`em_threshold_quantile`, and the **0.45 clamp floor** strand-4 showed pins
67/87 families). The canonical template stops being a synthetic median over
a recurrence cluster and becomes a single clean exemplar crop (the
`monitoR`/`ohun` and PI-review recommendation). Net: ~10 hardcoded constants
removed; one principled, constant-free selection rule on a primitive the
detector already ships — a simplification, not a bolt-on filter.

**Anti-artifact safeguards (carry, don't delete, the breadth premise).**
Self-neighbors must come from *distinct recordings* (the recurrence-breadth
requirement folded into the contrast, not removed); inspect the top selected
exemplars; report confusers + support recordings per selected template.
These reduce — but do not eliminate — the nuisance-recurrence risk; strand-3
remains the adjudicator.

**Provenance.** Two adversarial PI reviews: (a) triaged six candidate
recovery directions and demanded candidate-level (not post-EM) selection
with `auc_vs_class` as a veto-and-ranking metric; (b) adjudicated the
"matrix profile is univariate / ill-suited to PCEN" objection — verdict:
keep the selection-contrast principle, re-ground it on the existing 2-D
matched filter, drop the matrix-profile branding, and flag nuisance
recurrence as the residual the label-free contrast cannot close.
