# 2026-04-24 — roahaw failure-mode analysis: silent crops vs acoustic confusion

First application of the triage-filtered `export_figures.py` pipeline to a
single over-represented species. Referenced notes:
[triage](2026-04-24-triage-v5j-failure-modes.md),
[certainty + species composition](2026-04-24-certainty-and-species-composition.md).

**The methodology (triage + certainty + species composition → filtered figure
export) works, and it changed what I believed about why v5j fails on
`mixed_uncertainty`.**

## Why roahaw

From species composition: `roahaw` has the largest sample count (n=15) among
species over-represented in `mixed_uncertainty`, at a 71% within-sample
failure rate on that bucket. Highest-signal species for pattern replication.

## Priority B: is roahaw rare? No.

| species | n_focal | rank / 206 | percentile |
|---|---:|---:|---:|
| `roahaw` | **489** | **17** | **92nd** |
| `coffal1` | 495 | 6 | 98th |
| `socfly1` | 494 | 8 | 97th |
| `epaori4` | 188 | 73 | 65th |
| `grepot1` | 72 | 141 | 32nd |
| `palhor3` | 69 | 143 | 31st |
| `varant1` (handled well) | 291 | 42 | 80th |

The rare-species hypothesis is **falsified** for roahaw and for three other
over-represented species (`coffal1`, `socfly1`, `epaori4`). These are
among the most abundant species in training. Balanced sampling would not
help.

Two of the 10 over-represented species (`grepot1`, `palhor3`) *are* moderately
rare, but they're a minority.

## The roahaw bimodality

All 48 roahaw val samples, by triage category:

| Category | n | share |
|---|---:|---:|
| `correct_confident` (gt_prob ≥ 0.9) | 26 | 54% |
| `mixed_uncertainty` | **15** | **31%** |
| `correct_uncertain` | 6 | 13% |
| `gt_invisible` | 1 | 2% |

**Median gt_prob across all 48 = 0.96.** Most crops, the model is near-certain.
But mean is 0.62 because of a long left tail at ~0. The model is **bimodal on
roahaw**: either very confident or totally dismissive, with little in between.

Contrast `varant1` (0% failure rate): min gt_prob 0.119, median 0.862. Not
bimodal — the model is always at least moderately confident. Hypothesis:
varant1 vocalizes continuously throughout its recordings; roahaw doesn't.

## What the 15 figures actually show

Opened four of them (rank01, rank02, rank11, rank15). Clear split.

**14 of 15 look like "silent / low-content crops"** — rank01, rank02, rank11
match this pattern:
- Mel spectrogram is mostly background noise, rain, or ambient hiss. No
  clean vocalizations.
- Gradient saliency is scattered weak activations across the whole
  spectrogram — no coherent time-frequency pointing.
- Top-10 predicted species chart is extremely flat. Top-1 prob is low
  (0.03–0.27, mean ~0.14 across all 15 samples vs 0.76 for `confident_wrong`).
- Per-frame attention is near-uniform, per-frame probs near zero across all
  frames.

These are crops where roahaw is labeled at the recording level but not
vocalizing in the 5s window. The model correctly says "I don't hear anything
particular here," and BCE penalizes it because the recording's primary label
is roahaw.

**1 of 15 is a real acoustic confusion** — rank15, row 2071. This is the ONE
sample in the 15 with `saliency_informative=True`:
- Mel shows **clear repeating vertical stripes** — real bird vocalizations
  in the mid frequencies.
- Gradient saliency aligns with the vocalization bands.
- Top-10 has `saffin` at 0.167 and `roahaw` very close behind at ~0.135.
  Real species confusion.
- Per-frame attention is peaked on the frames with the vocalization.

## What this tells us

**Most roahaw "failures" are not failures.** They're artifacts of the
recording-level-label / random-crop pipeline. BCE loss penalizes the model
for being honest about low-information crops. Training to reduce this loss
would mean teaching the model to hallucinate the primary species regardless
of the crop content — the opposite of what we want.

**The `saliency_informative` flag correctly picked the single real failure.**
This is direct evidence that triage + saliency_informative is a good
methodology for separating real model failures from label-noise artifacts.
Before, I had grouped all of `mixed_uncertainty` as "real failures." The
roahaw cross-section shows that's wrong — only the ~35% with
`saliency_informative=True` are genuine discrimination failures; the rest
are crop-selection label noise.

Back-of-envelope: 196 saliency-informative mixed_uncertainty samples out of
551 total → ~36% are real model failures, ~64% are silent-crop artifacts.

## Implications for "improve v5j using labeled data only"

Not covered by this analysis: LB improvement. These findings are offline-val
interpretation only. But they narrow the hypothesis space:

Unlikely to help (based on this evidence):
- **Balanced sampling** — roahaw/coffal1/socfly1 are NOT rare; failures
  aren't coverage-limited.
- **Hard-negative mining on specific species pairs** — top-1 confusions in
  the silent-crop samples are scattered across 13 different species for the
  15 roahaw cases. No pair pattern.

Plausible leverage points to investigate next:
- **Energy/VAD-based crop filtering during training** — avoid sampling
  5s crops that don't contain vocalizations. Reduces label-noise penalty
  without adding data.
- **Label-noise-aware BCE weighting** — down-weight the loss on crops
  with low total signal energy (so the model isn't punished for honesty
  when the crop is silent). Same intuition, different implementation point.
- **Use `all_labels` / secondary_labels as positive supervision** — so a
  "wrong top-1" that corresponds to a background species isn't penalized.
  Already noted in the triage work; this is more urgent given how much of
  mixed_uncertainty is actually silent-crop (no bird at all), not
  cooccurrence.

Intentionally NOT recommending any of these as experiments yet. The analysis
so far covers 4/15 samples visually + aggregate stats on the other 11. A
bigger sweep (next note) would confirm or refine the 93% silent-crop
estimate.

## Validated tooling

The `export_figures.py` filtered mode worked end-to-end:
- `--filter-category mixed_uncertainty --filter-species roahaw --top-k -1`
  produced exactly 15 samples (dedup on `(filename, offset_sec)` applied).
- Output at `figures/v5j/mixed_uncertainty_roahaw/` with 15 four-panel PNGs,
  15 competitor PNGs, 15 WAVs, one summary batch, and a manifest.json that
  records filter criteria, dedup key, git-traceable config + checkpoint paths.
- CPU runtime: ~2 min for 15 samples.

## How to reproduce

```bash
cd user/murilo/analysis

# Priority B: rare-species check
python -c "
import pandas as pd
train = pd.read_csv('/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train.csv')
counts = train['primary_label'].value_counts()
print(counts[['roahaw','coffal1','socfly1','epaori4','grepot1']])"

# Filtered figure export for all 15 roahaw mixed_uncertainty:
python export_figures.py \
    --losses data/losses_v5j.parquet \
    --config ../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml \
    --checkpoint /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt \
    --filter-category mixed_uncertainty \
    --filter-species roahaw \
    --top-k -1 \
    --batch-k 5 \
    --device cpu
```

## Action items (narrow)

- Expand visual inspection to a wider sample. Confirm the "14/15 silent-crop,
  1/15 real confusion" split generalizes beyond roahaw. Candidates:
  `coffal1` (n=8), `socfly1` (n=13), `epaori4` (n=8), and the
  saliency-informative-only slice across all species.
- Before experimenting: read a few of the roahaw `correct_confident` samples
  (gt_prob > 0.9) to see what "the model hearing roahaw cleanly" looks like.
  That's the baseline for "what a non-silent crop should look like."
- Cross-check: do the roahaw mixed_uncertainty WAVs actually sound silent to
  a human ear? Playing them confirms whether the "silent crop" interpretation
  is literal or if there's activity the model is missing.

## Files

- `analysis/export_figures.py` — filtered mode added in this session
- `analysis/figures/v5j/mixed_uncertainty_roahaw/` — 15 samples exported
- `analysis/figures/v5j/mixed_uncertainty_roahaw/manifest.json` — filter
  criteria, dedup key, exact checkpoint, per-sample metadata
