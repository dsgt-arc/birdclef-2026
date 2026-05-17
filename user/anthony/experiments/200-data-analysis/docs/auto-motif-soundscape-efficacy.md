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
python -m data_analysis.auto_motif.soundscape_eval --files <file.ogg>    # targeted re-score; aggregation always spans the full cache
```

Resumable: a file with a status JSON is skipped; delete
`_soundscape_eval/<bank>/status/<file>.json` to force one to re-run. Driver
is a `spawn`-context process pool (bank loaded once per worker) — **not**
Luigi: 66 files is far below the >~1k-file scale where sharded fan-out earns
its weight, and `spawn` makes the fork-after-threadpool BLAS deadlock
impossible by construction.
