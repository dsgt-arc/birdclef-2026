# Results: Soundscape event boxing v1 (insects)

## Summary (verdict)

**Mixed, and the negative half is the informative half.** The boxing front end
works: connected-component box proposal + the present-vs-absent **discriminative
band** recovers clean, plausible fixed bands for 19/25 insect sonotypes. But the
simple band+shape **propagation signature does not transfer**: cross-site recall
collapses to **0.17** at precision **0.06** (signatures fire on other insect sites
where the sonotype is absent), and several fire on the insect-free dominant site
(S22) up to **48%** of windows — i.e. they are largely generic-energy detectors,
not species detectors. The single biggest finding is structural: **insect labels
exist in only 4 sites (S08/S15/S19/S23), none in S22, and 16/25 sonotypes are
single-site**, so cross-site transfer is mostly untestable and site is badly
confounded with sonotype.

## Update (2026-06-01): contrast-basis fix tried; signature strand retired

Two follow-ups after the verdict above:

1. **Discriminative-shape fix didn't move the result.** v1's signature keyed its
   cosine on the raw present-consensus in-band shape, which bakes in site
   background. Rebuilding it from the positive-clipped **present-minus-absent
   contrast** (the discriminative residual) and breaking calibration F1-ties toward
   higher specificity changed essentially nothing:

   | split | precision (before → after) | recall (before → after) |
   |---|---|---|
   | in-site (fit ceiling) | 0.526 → 0.526 | 0.614 → 0.589 |
   | cross-site (LOSO) | 0.06 → 0.055 | 0.17 → 0.151 |

   Mean S22 false-fire rate held at ~0.18; per-sonotype deltas were mixed (son06
   0.073→0.011 better, son11 0.436→0.528 worse). Within a narrow band the present
   and contrast shapes are near-collinear, and the query side still compares a raw
   window shape — so the basis swap cannot close a window-level, frequency-collapsed
   discriminator gap. Confirms the verdict: **the discriminator, not the operating
   point, is the ceiling.**

2. **`signature.py` + `propagate.py` removed.** The fully-automatic band+shape
   propagation strand is falsified and no longer load-bearing. The go-forward
   direction is the human-in-the-loop design in
   `docs/concepts/semi-supervised-box-annotation-loop.md` (Perch-prototype routing +
   DSP pre-boxing + margin-only review). The boxing front-end (`boxes.py`) and Label
   Studio curation (`label_studio.py`) are retained; the `artifacts/signatures/` and
   `artifacts/propagate/` outputs referenced below are historical.

## Update (2026-06-02): Perch-prototype feasibility gate — FAIL

Built `retrieve.py gate` — the Stage-A go/no-go for the annotation loop: does a
weak-label contrastive Perch prototype route insect sonotypes cross-site (LOSO over the
**8** multi-site, gate-clearing sonotypes), beating trivial baselines, without firing on
the insect-free probe (S22-dominated)? Within-site AP/P@20 (held-out site only, site
confound removed) + insect-free FP@recall0.8, macro over folds:

| ranker | within-site AP | P@20 | insect-free FP@0.8 |
|---|---|---|---|
| centroid (mean present) | 0.504 | 0.365 | **0.331** |
| band_energy | 0.501 | 0.412 | 0.572 |
| prototype (contrastive) | 0.463 | 0.379 | 0.625 |
| prototype_centered | 0.444 | 0.374 | 0.849 |
| centroid_centered | 0.429 | 0.368 | 0.844 |
| random | 0.424 | 0.373 | 0.788 |
| site_prior (base-rate floor) | 0.370 | 0.371 | 1.0 |

**FAIL** — but not because routing is at chance. Within-site, the best embedding ranker
(centroid, AP 0.504) *does* clear the base-rate floor (0.370) and random (0.424); the
problem is it (a) only **matches a trivial band-energy detector** (0.501, which beats it
on P@20, 0.412 vs 0.365) — Perch adds nothing over cheap DSP energy here — and (b) **fires
on 33% of insect-free windows** (≫ the 5% bar), i.e. it is not specific. Per-sonotype,
**only son03** (the lone 3-site sonotype) routes well (AP 0.81 vs 0.20, FP 0.069 — *near*,
not under, the bar); the other 7 are at/below base rate with FP 0.13–0.82. **Per-recording
centering made it worse** (AP↓, FP↑ to ~0.85), plausibly because stationary insects sit in
the recording mean, so subtracting it erases the signal. Read (hypotheses, not proven by
this gate): the site==sonotype confound dominates, and Perch may under-embed insects (it
is bird-centric). Per the concept's kill criterion, **Stage B was not built.** Artifacts:
`artifacts/retrieve/gate/{gate_table.csv,gate.md,per_sonotype.csv,scores_debug.parquet}`.

## Observations

Data (`labels`):
- The published CSV duplicates every row exactly: **739 real windows**, not 1,478.
- Insect presence is confined to **S08 (60 windows), S15 (48), S23 (36), S19 (24)**;
  **S22 (477 windows) has zero insect labels**. Only **9/25** sonotypes span >=2 sites.

Boxing (`boxes`):
- 15,683 candidate boxes over 739 windows (~21/window).
- 19 of 25 sonotypes yield a discriminative band; 5 are gated `insufficient`
  (n_present < 8: son02/05/09/12/19); son20 clears the gate but its contrast has
  no band. Bands are narrow and plausible (e.g. son07 ~409-559 Hz, son17 ~2440-2590 Hz).
- **son15 ≡ son16** and **son22 ≡ son23** get *identical* bands — perfectly
  co-occurring sonotype pairs the present/absent contrast cannot separate.

Propagation (`propagate`, LOSO over the 4 insect sites):
- in-site fit ceiling: precision **0.53**, recall **0.61** (19 sonotypes).
- cross-site (LOSO, pooled, counting FPs in sites where the sonotype is absent):
  precision **0.06**, recall **0.17** (7 recall-testable). The precision is this
  low because 11 sonotypes fire on other insect sites where they are absent
  (pure cross-site false positives, e.g. son22/23/24 n_present=0 but fp>0).
  Best individual: son04 P0.33/R0.20, son25 P0.28/R0.20.
- S22 specificity (fires here are all false positives): son11 **48%**, son22/23
  **42%**, son25 **37%** false-fire — vs clean son04 **1%**, son18 **2%**, son07 **5%**.

## Analysis

- The discriminative-band idea is sound and cheap: contrasting present vs
  same-site-absent consensus spectra isolates a stationary insect's band without
  any manual boxing, and the bands look right.
- But a band + an in-band spectral-shape cosine is too weak a discriminator: it
  captures *where* energy sits, not *that it is this insect*, so it (a) misses the
  sonotype in a new site whose background shifts the in-band shape (recall 0.17),
  and (b) fires on any window with comparable band energy, including insect-free
  S22 (low specificity). This mirrors exp 200's "generic-energy detector" failure
  mode, now measured directly via the S22 false-fire rate.
- A few sonotypes (son04, son07, son18) are both reasonably specific and
  band-clean — narrow, isolated bands with little competing energy. These are the
  cases where the stationary-insect assumption genuinely holds.

## Confounders / limitations

- **Site == sonotype**: 16/25 sonotypes occur in one site only, so their band may
  encode that site's background. Cross-site numbers exist for just 7 sonotypes.
- **Box->label attribution is a prior, not truth** (multi-label windows, no
  per-box ground truth) — 14,632/15,683 boxes are `unattributed`. The Label Studio
  curation loop exists to upgrade these.
- Pool yield (`--pool`) uses whole-file peak-normalization vs per-window at train
  time; `per_recording_mean_spectrum` is MAD-normalized so scale largely cancels,
  but treat pool yield as indicative.

## Artifacts

- `artifacts/labels/` — counts, site/sonotype breakdowns.
- `artifacts/boxes/` — `labeled_boxes.parquet` (15,683), `discriminative_bands.csv`,
  `attributed_boxes.parquet`, `bands/<son>.png` (present-vs-absent contrast),
  `window_spectra.npz` (per-window spectra cache).
- `artifacts/signatures/` — `signatures_insect.{csv,npz}`.
- `artifacts/propagate/` — `validation.{csv,md}`, `specificity.csv`,
  `thresholds.csv`, `pool_yield.csv` (after `--pool`).
- `artifacts/label_studio/<project>/` — `tasks.json` + `label_config.xml` for the
  2-D RectangleLabels curation projects (`insects_v1`, `all_taxa`).

## Next steps

> Superseded twice (see the dated Updates above): the band+shape signature was
> falsified, then the **Perch-retrieval pivot itself failed its feasibility gate for
> insects** (2026-06-02). Perch routing of insect windows cross-site is not viable
> (site==sonotype confound; Perch is bird-centric). Current live forks:

1. **Birds-first routing.** Perch is in-distribution for Aves (28 labeled species), so
   the same gate may pass there even though DSP boxing is harder for birds (the
   inversion: DSP boxing easy for insects/hard for birds; Perch routing the reverse).
   Re-run `retrieve gate` over bird species before building any loop.
2. **Curate insect windows without routing.** Weak labels already mark which windows
   hold which sonotype — DSP pre-boxing + the existing `label_studio` export curate the
   labeled insect windows with no Perch at all (it just won't expand to the unlabeled
   pool). First for the clean sonotypes (son04/07/18) and ambiguous boxes.
3. **Cheap probe**: does routing quality scale with #training sites? son03 (the lone
   3-site sonotype) was the only insect that routed cleanly — if it's purely "3 sites
   beats 2," the insect failure is data-limited, not method-dead.
4. Merge the co-occurring pairs (son15/16, son22/23) in any curation — identical
   present-sets, inseparable from weak labels.

See `docs/concepts/semi-supervised-box-annotation-loop.md` for the loop design + gate.
