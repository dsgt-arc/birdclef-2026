# Concept: Semi-supervised box annotation loop (Perch-retrieval + DSP pre-boxing + margin-only human review)

**Status**: developing
**Created**: 2026-06-01
**Updated**: 2026-06-01

## Idea

Turn weak per-5s-window presence labels into time-frequency **bounding boxes** with
the least possible human effort, by composing three assets the project already
has into a closed active-learning loop:

1. **Weak window labels** seed a discriminator for free (no cold start).
2. **Perch embeddings** (one vector per 5s window) *route* the human — they rank
   which windows to look at — but never draw a box.
3. **DSP boxing** (discriminative band × in-band active-time) *pre-draws* the box
   inside each routed window, so the human **confirms/rejects/nudges** rather than
   draws.

The loop: build a **contrastive Perch prototype** for a sonotype from its present
vs absent windows (mean(present) − mean(absent) in embedding space); rank all
windows by similarity; let the DSP pre-box the top candidates; the human
adjudicates only the **uncertain margin** (scores near threshold, Perch/DSP
disagreements); confirms and **hard negatives** refine the prototype and threshold;
re-rank. Each round needs fewer corrections than the last — the virtuous part.

The division of labor is the crux: **Perch decides *which* windows to annotate;
DSP decides *where* the box goes; the human only adjudicates the boundary.**

## Motivation

Experiment 201 established that *fully automatic* propagation of a band+in-band-shape
signature does not transfer (cross-site precision 0.06, recall 0.17; fires on up to
~50% of the insect-free dominant site). Swapping the signature's spectral basis to
the present-minus-absent contrast did not move those numbers — confirming the
discriminator, not the operating point, is the ceiling. Meanwhile Perch retrieval —
the obvious stronger discriminator — operates at **window granularity with frequency
collapsed**, so it *cannot* produce boxes; pivoting to it would silently abandon the
experiment's actual goal (boxes) for the easier one (window-level presence).

The resolution is to stop trying to fully automate and instead **minimize human
boxing effort**. A human-in-the-loop *relaxes* the transfer ceiling (the human is the
discriminator of last resort) — but does not remove it: the human only sees positives
the router surfaces, so the loop must keep a **random-audit / exploration arm** or it
silently misses positives the prototype ranks low. Within that caveat, Perch +
retrieval + DSP pre-boxing collapse the per-box cost to a confirm/reject click on the
most informative few. For **stationary insects** the cost collapses further where the
box is near-constant (fixed band, often spanning the window): confirming the band
*once* can transfer geometry to a sonotype's other present-windows — on the order of
one confirm per **gate-clearing, geometry-stable** sonotype. This is *not* all 25:
5 are below the `n_present>=8` gate, 16 are single-site (no cross-site variability to
confirm from one window), and the co-occurring pairs son15/16 & son22/23 are
inseparable from weak labels (see Open Questions).

The plumbing already exists and is one seam short of a loop:

| Piece | Status (exp 201) |
|---|---|
| DSP candidate boxes | ✅ `birdclef_201.boxes` |
| Pre-drawn boxes imported as Label Studio *predictions* the human corrects | ✅ `birdclef_201.label_studio` (export/push) |
| Corrected boxes parsed back to gold | ✅ `label_studio.parse_export` |
| Perch contrastive prototype + window ranking | ❌ missing |
| Active-batch selection (uncertain + hard-neg) + feedback re-rank | ❌ missing — export is unordered, one-shot, no retrain |

The current export dumps *all* boxes in *all* present-windows in arbitrary order:
maximum effort, no learning. Closing the loop is one new module (`retrieve.py`)
reusing everything else.

## Background

- **Experiment 201** (`user/anthony/experiments/201-soundscape-event-boxing/`):
  `proposal.md`, `results.md` (the negative transfer result + the boxing front-end
  that works), and the `birdclef_201` package (boxes, signature, propagate,
  label_studio, geom).
- **Perch v2 soundscape embeddings**:
  `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/embeddings.parquet`
  — 127,896 rows = 10,658 files × 12 chunks; one 1-D embedding per **5s window**
  (`start_sec`/`end_sec` = chunk bounds). No frequency axis, no sub-window time.
- **Weak labels**: `train_soundscapes_labels.csv` — 739 real windows (deduped), 66
  files, 75 labels, per-window multi-label presence. Insect presence confined to 4
  sites (S08/S15/S19/S23); S22 (the dominant site, 477 windows) has zero insects —
  a free pure-specificity probe.
- Related concepts:
  - [docs/concepts/weak-supervision-label-fusion.md](weak-supervision-label-fusion.md) — using weak/window labels as a supervision signal.
  - [docs/concepts/unsupervised-event-detection-labeling.md](unsupervised-event-detection-labeling.md) — event detection without strong labels.
  - [docs/concepts/soundscape-domain-shift.md](soundscape-domain-shift.md) — the cross-site transfer risk this loop must survive.
  - [docs/concepts/prompted-sound-event-detection.md](prompted-sound-event-detection.md) — prototype/prompt-driven detection.

## The loop (sketch)

```
weak labels ──► Perch contrastive prototype ──► rank ALL windows by cosine
   (free)            (free, no cold start)              │
                                                        ▼
                          DSP pre-draws a box in each top-ranked candidate
                                       (free, already built)
                                                        │
                                                        ▼
                    HUMAN confirms / rejects / nudges only the uncertain margin
                                       (the only cost)
                                                        │
                       confirmed boxes + hard negatives ◄┘
                                       │
                                       ▼
              refine prototype + threshold ──► next batch is cleaner ──► repeat
```

Effort-minimizing levers, in order of leverage:

1. **No cold start** — prototype seeded from the 739 weak labels, zero human input.
2. **Confirm, never draw** — DSP pre-boxes (existing predictions pipeline).
3. **Geometry transfer (insects)** — one band confirm covers all present-windows of a
   sonotype; ~25 confirms seeds everything.
4. **Retrieval routing** — review where positives are dense, not a 127k haystack.
5. **Uncertainty / hard-negative sampling** — each click maximally sharpens the
   boundary; rejections are the most valuable signal.
6. **Feedback** — confirms + hard negatives refine prototype + threshold → fewer
   corrections next round.

## Feasibility gate (must pass before building the full loop)

The whole loop rests on Perch routing being better than trivial. The first successor
experiment is a small **gate**, not the loop: under leave-one-insect-site-out, does the
weak-label contrastive Perch prototype rank a sonotype's present windows above
negatives (same-site, cross-site, and the insect-free S22 probe) **better than** a
site prior, a band-energy ranker, and random-within-site — with **no train/eval label
leakage** (prototype seeded on held-out files/sites or a small human gold subset, never
the windows it is scored on)? Concrete pass bar: cross-site AP and P@20 beat every
trivial baseline, and S22 false-positive rate at the chosen operating point ≤ 5%.
**If it does not clear this, stop or pivot — do not build the annotation UI on a router
that cannot route.**

## Open Questions

- Does the **contrastive Perch prototype** (mean present − mean absent, window-level)
  separate a sonotype from same-site background well enough to rank usefully —
  especially cross-site and on the insect-free S22 specificity probe — where the
  spectral band+shape signature failed?
- How should the **box geometry** be produced inside a routed window — reuse the
  discriminative band × in-band energy directly, or the connected-component ROI? For
  near-stationary insects do these agree, and how often does a single confirmed band
  transfer without correction?
- What **acquisition function** minimizes total clicks: pure top-k by similarity (high
  yield) vs uncertainty band near threshold (high information) vs a mix? How big a
  per-round batch?
- How are **hard negatives** best folded back — subtract from the prototype, train a
  light logistic head on embeddings, or just move the threshold?
- Co-occurring band-sharing pairs (son15≡16, son22≡23) have identical present-sets, so
  a weak-label prototype is identically blind to the distinction — they are
  **inseparable from weak labels**. Decision: **merge or defer** them as unresolved
  complexes until box-level / pulse-rate evidence exists; do not claim the loop
  separates them. (A within-band pulse-rate / AM-spectrum discriminator is the candidate
  separator — a separate concept.)
- **Feedback semantics**: a rejected pre-drawn box ≠ a window-level negative — it may
  mean "wrong geometry," not "sonotype absent." The loop must distinguish four labels:
  window-positive, window-negative, box-positive, box-negative. Which of these refines
  the prototype vs the box geometry vs the threshold?
- **Does feedback actually refine Perch?** The available embeddings are whole-5s-window
  vectors. Confirmed *boxes* refine window labels only — to get a box/event-level
  embedding you must **recompute Perch on cropped / band-passed audio**. Is that worth
  it, or does feedback just train a window-level classifier + sharpen the band geometry?
- **Effort accounting**: how few human actions to reach a usable per-sonotype gold set?
  What's the marginal yield-per-click curve, and where does it flatten (stop rule)?
- Does the loop survive the **site == sonotype** confound (16/25 sonotypes single-site)
  — i.e. does it learn the sonotype or the site? (This is what the feasibility gate tests.)

## Testable Hypotheses

All evaluated with **no train/eval leakage**: the prototype is seeded on held-out
files/sites or a human gold subset, never the windows it is scored on.

1. **Prototype ranking beats spectral signature *and* trivial baselines.** Under
   leave-one-insect-site-out, ranking windows by cosine to the weak-label contrastive
   Perch prototype beats (a) the exp 201 band+shape signature (cross-site P0.06/R0.17,
   S22 fp ≤0.5) and (b) a site prior, a band-energy ranker, and random-within-site, on
   **cross-site AP and P@20**, with **S22 FP@operating-point ≤ 5%**. *Falsified if it
   does not beat all trivial baselines.*
2. **Margin-only review reaches usable gold cheaply.** Define "usable gold" =
   ≥20 confirmed positive boxes/sonotype **or** ≥80% coverage of that sonotype's known
   positive windows. Measure **clicks-per-accepted-gold-box** and
   **minutes-per-accepted-gold-box** (action = confirm / reject / nudge / draw-missed /
   relabel), plus a **missed-positive audit rate** from a random-audit batch. Target:
   ≥5× fewer actions than blind review of all proposed boxes (exp 201: ~21 boxes/window
   × all present-windows), at equal or better gold coverage.
3. **The loop improves per round.** Over ≥3 rounds, **held-out validation AP rises** and
   **actions-per-accepted-positive falls**. (Correction rate is *not* expected to fall
   monotonically — active learning surfaces harder cases later; measure correction rate
   only on comparable **random-audit** batches, not on the margin batch.)
4. **Geometry transfers for stationary insects.** For a gate-clearing sonotype, a single
   confirmed band transfers to its other present-windows with **frequency IoU ≥ 0.5 and
   time IoU ≥ 0.5 in ≥70%** of its gold positive windows — so human cost is dominated by
   routing/adjudication, not drawing. (State whether "single confirmed band" is per
   sonotype globally or per site/recording; handle continuous insects spanning full 5s.)

## Connections

- **Based on**: experiment 201 (`proposal.md`, `results.md`); Perch v2 soundscape
  embeddings; [weak-supervision-label-fusion](weak-supervision-label-fusion.md).
- **Related to**: [unsupervised-event-detection-labeling](unsupervised-event-detection-labeling.md),
  [soundscape-domain-shift](soundscape-domain-shift.md),
  [prompted-sound-event-detection](prompted-sound-event-detection.md); a future
  within-band **pulse-rate / AM-spectrum discriminator** concept (separates co-occurring
  band-sharing sonotype pairs and is itself box-localizing).
- **Could lead to**: an experiment 201 successor (or strand) implementing `retrieve.py`
  — contrastive Perch prototype → ranked, DSP-pre-boxed, capped Label Studio batch →
  `parse_export` → re-rank with confirms+hard-negatives folded in.

## Notes

- **Why not just Perch retrieval (the v1 "next step")**: Perch gives window-level
  presence, frequency collapsed — it cannot output a box. It belongs in the loop as
  the *router/seed*, not the boxer. This concept exists to correct that conflation in
  `results.md`'s next-steps.
- **Why a loop and not more automation**: exp 201 showed the automatic discriminator
  ceiling is the bottleneck; a human-in-the-loop *relaxes* it (subject to the
  routing/exploration caveat in Motivation). The research question shifts from "can we
  detect X automatically" to "how few human actions to gold boxes."
- **Framing, kept honest**: the claim is **not** "Perch solves cross-site insect
  detection" — it's "Perch + DSP can cut human box-labeling cost enough to build the
  gold set that stronger (e.g. box-level / pulse-rate) methods need." If the feasibility
  gate fails, that framing fails with it.
- The negative transfer result does **not** doom this — it's the reason to put a human
  in the loop and measure clicks-to-gold rather than chase a fully-automatic P/R.
- Scope decision (2026-06-01): **insects-first**, to exploit geometry transfer and stay
  on-thesis with v1 before extending to non-stationary (bird/frog) boxes where the
  per-call geometry varies and active learning earns more.
