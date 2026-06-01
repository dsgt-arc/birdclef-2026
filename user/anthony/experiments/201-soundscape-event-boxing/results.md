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

1. **Curate** a slice in Label Studio (`push --replace`, annotate, `parse`) to get
   gold boxes — first for the few clean sonotypes (son04/07/18) and the ambiguous
   boxes, to measure how much attribution error inflated the weak propagation.
2. **Stronger discriminator** than band+shape: the obvious lever is **Perch
   embedding retrieval** on the soundscape windows (all 10,658 already embedded) —
   few-shot nearest-neighbor from the gold insect boxes, which should beat a raw
   spectral signature on both transfer and specificity. This is the v2 pivot the
   embedding-route discussion pointed at.
3. Merge the co-occurring pairs (son15/16, son22/23) or separate them with a
   discriminator that uses more than the shared band.
