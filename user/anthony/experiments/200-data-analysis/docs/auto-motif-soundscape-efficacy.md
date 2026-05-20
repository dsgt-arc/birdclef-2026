# Auto-motif soundscape efficacy (label-based extrinsic validation)

The pipeline's quantitative claims were never tested against *truth*: the
Perch-embedding screen and the co-firing guardrail are label-free proxies,
and the recording-sampled manual anchor is scaffolded but unlabeled. This
thread closes part of that gap with the one ground truth available —
`train_soundscapes_labels.csv` (66 files, weakly-labeled 5 s windows, 2
annotator passes; all 41 production species occur in it).

## What it does

`data_analysis.auto_motif.soundscape_eval` drives the **real production
detector** over every labeled soundscape — `_score_track` → `find_peaks`
(same `min_spacing`/prominence) → envelope anchor → per-species
`nms_events`/`consolidate_events` at `AutoConfig` defaults. The only
substitution vs. the in-miner detector: a held-out soundscape has no EM
member/background scores, so each family's **committed persisted threshold**
is used — which *is* the production operating point under test.

Two metrics, kept deliberately apart (conflating them is the trap):

- `roc_auc` / `pr_auc` — **threshold-free**: is the mined motif a real,
  separable acoustic pattern on soundscapes? Trust these (within support).
- `precision` / `recall` / `f1` — at the **committed** threshold: does the
  focal-calibrated cut transfer? Low here with high AUC = good template,
  miscalibrated cut, *not* a template failure.

Positives use the **union** of the two annotator passes. Per-species AUC is
only reported as evidence when `n_pos ≥ 10` (a small-sample floor — a single
positive window makes AUC sampling noise; the same n=1 discipline applied to
the other validation strands).

## Headline (calibrated)

Of 41 species, only **22 are adequately supported** by this label set; the
other 19 are underpowered (`n_pos < 10`) and are not evidence either way.
Among the 22:

- Median threshold-free **ROC-AUC 0.672**.
- **8 genuinely separable** (ROC ≥ 0.70): `hyamac1, 65380, redjun, litnig1,
  22973, bufpar, 22967, chvcon1` — note `22967`, the narrowband species the
  whole pipeline was originally validated on, lands here (instrument sanity
  check).
- **7 at chance** (ROC < 0.55 — templates do *not* transfer): `23158, 25092,
  undtin1, chacha1, 43435, 326272, orwpar`.
- The committed-threshold operating point **fails decisively**: precision ≈
  prevalence, recall ≈ 1.0 (focal thresholds fire on ~everything in
  soundscapes). The lone well-supported high-prevalence species `65380`
  (`n_pos` 333) is the only usable operating point: precision 0.49, ROC
  0.86, PR 0.79.

**Read:** the boundary-resolution yields genuinely separable motifs for a
meaningful fraction of supported species, but **not uniformly**, and the
production thresholds are miscalibrated for soundscape deployment (expected
focal→soundscape domain shift). This *bounds* the claim — it does not
vindicate or refute the pipeline wholesale. The manual anchor remains the
instrument for precision; this label set is too sparse per-species for most.

Full per-species table, headline, and the underpowered list:
`../artifacts/auto_motif/_soundscape_eval/production/_summary.md`. Canonical
scientific record: `concepts/decouple-frequency-localization-from-focal-selection.md`
§ "v4 validation, strand 3".

## Can the operating point be fixed before deployment? — window-level CFAR, falsified

The strand-3 reading ("good template, miscalibrated cut") invites a
label-free fix: re-derive the threshold from the deployment recording's own
background (CFAR) instead of the frozen focal value. `--rescore cfar`
implements this at the **5 s-window** unit with **no focal floor** — per
species, fire iff the window-max score clears `Q99.5` of that species'
corpus-pooled scored-window distribution (`0.995` is the pipeline's own
committed false-alarm design, not a new constant). `max_score` is untouched,
so ROC/PR-AUC stay **byte-identical** — a clean operating-point-only A/B.

It does **not** work, and the failure is informative:

| operating point | caught (recall) | false alarms | precision |
|---|---|---|---|
| committed (focal/floor) | 1260 (71.4 %) | 18 976 | 0.062 |
| window-CFAR `Q0.995` | 22 (1.2 %) | 140 | 0.136 |

A probe first reframed strand 3: the committed cut is **67/87 families
pinned at the 0.45 clamp floor** — a floor + frame-vs-window granularity
artifact, not a learned focal cut that fails to transfer. CFAR then fails for
a deeper reason: every species has 792 corpus windows, so `Q0.995` fires
**exactly 4 windows/species** (the `0.5 %` budget was a *per-frame* rate,
~10³× too tight at the window unit), and — fundamentally — the optimal cut is
**prevalence-dependent** (pool = `(1−π)·bg + π·target`, π 45 %→1 %): high-π
species self-mask (`65380/22973/22967` → P 1.0, R ~0.015), low-π separable
species starve (`litnig1/chvcon1/trsowl` → P 0, R 0). No fixed unsupervised
quantile works across that range — *neither end of the label-free spectrum
knows π, which is label information*.

Constructively, the A/B proves the separable templates are **operable**:
even at the too-tight cut, F1 rose for `hyamac1`(ROC 0.96) 0.09→0.29,
`bufpar`(0.79) 0.08→0.22, `redjun`(0.82) 0.09→0.12. The bottleneck is
operating-point *selection*, which must be **supervised** — a per-species
threshold/isotonic fit on the soundscape labels under grouped leave-site-out
CV, restricted to the ~8 ROC ≥ 0.75 species. The 7 at-chance species stay a
mining problem. A/B artifacts:
`../artifacts/auto_motif/_soundscape_eval/production__cfar/_summary.md`;
record: concept doc § "v4 validation, strand 4".

## Why the at-chance species fail — D1 (strand 5)

Strand 3 left *why* the 7 at-chance species are at chance open, and strand 4
showed it is **not** a calibration knob. `data_analysis.auto_motif.focal_specificity`
closes it label-free: the focal analog of the soundscape ROC — the exact
production matched-filter over focal audio of a species vs. other species
(the focal folder is curation truth, so no labels needed). `auc_vs_class`
(target focal vs. **same-class** other species' focal) is decisive — a clean
self-vs-congener separability is *necessary* for soundscape recovery.

- Instrument validated: the hand-curated frog `22967` is the most
  focally-specific species (`auc_vs_class` 0.87); all 3 controls transfer.
- **4/7 at-chance are a mining failure** (`auc_vs_class` 0.37–0.47, several
  *below* 0.5 — the template prefers other species' clean focal audio):
  `326272, 25092, undtin1, 43435`. The top confusers are a **fixed
  cross-taxon attractor set** (antbird, jay, macaw, a mammal) shared by a
  frog, a tinamou and a monkey — the mined "motif" is a generic
  broadband-energy detector, not a species call. 3 of the 4 ran the
  **validated consensus route** with healthy `member_bg` d′ — so d′ is here
  *actively misleading*, and the boundary reaches beyond the band-free patch.
- **2/7 are genuine recoverable domain shift** (`chacha1` 0.70, `23158`
  0.74 — focal-specific, lost only in-field). `23158` is now coherent across
  all strands.

No threshold/calibration/domain trick recovers the mining-failure majority;
the constructive path is the concept's deferred **v2 discriminative
recurrence** (cross-species/TF-IDF negative or a Perch-embedding focal
prior), now strongly motivated and quantified. Canonical record + the
honest asymmetry (low `auc_vs_class` is decisive, high is necessary-not-
sufficient): concept doc § "v4 validation, strand 5"; artifact
`../artifacts/auto_motif/_focal_specificity/_summary.md`.

## Caveats

- Soundscape labels are weak presence over 5 s and multi-label; templates
  cover *recurring motifs*, not a species' whole vocal repertoire — a species
  calling with a non-mined call type is a legitimate miss, not a template
  failure (recall is upper-bounded by motif coverage of the repertoire).
- Most species have only a handful of positive windows here; this set
  validates discriminability where support exists and decisively shows the
  committed-threshold transfer failure, but cannot adjudicate precision for
  the long tail.

## Reproduce

```bash
python -m data_analysis.auto_motif.soundscape_eval --bank production
python -m data_analysis.auto_motif.soundscape_eval --bank gabor_stack   # the unvalidated selector's alternative
python -m data_analysis.auto_motif.soundscape_eval --rescore cfar       # window-level CFAR A/B (strand 4); -> production__cfar/
python -m data_analysis.auto_motif.soundscape_eval --files <file.ogg>    # targeted re-score; aggregation always spans the full cache
python -m data_analysis.auto_motif.focal_specificity                     # D1 strand-5: why the at-chance species fail (label-free)
python -m data_analysis.auto_motif.focal_specificity --targets all       # every production species
```

Resumable: a file with a status JSON is skipped; delete
`_soundscape_eval/<bank[__rescore]>/status/<file>.json` to force one to re-run. Driver
is a `spawn`-context process pool (bank loaded once per worker) — **not**
Luigi: 66 files is far below the >~1k-file scale where sharded fan-out earns
its weight, and `spawn` makes the fork-after-threadpool BLAS deadlock
impossible by construction.
