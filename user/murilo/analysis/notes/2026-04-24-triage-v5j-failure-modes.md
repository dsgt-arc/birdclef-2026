# 2026-04-24 — v5j triage: failure modes are mostly not failures

First application of `triage.py` on `losses_v5j.parquet` (3,654 val samples).
Goal: understand what "high loss" actually represents before spending time
on saliency analysis.

## What triage adds to the losses table

For each sample, joins with `train.csv` to pull `secondary_labels` (for focal)
or uses `all_labels` (for soundscape), then flags:

- `known_labels` — primary + all secondary species known to be in the crop
- `top1_is_secondary` — model's top-1 is in the known labels but ≠ primary
- `top1_is_known` — top-1 is anywhere in the ground truth label set
- `saliency_informative` — gt_prob in [0.05, 0.9], the range where sigmoid
  gradient isn't saturated and saliency can be trusted
- `triage_category` — heuristic bucket

Output: `analysis/data/triage_v5j.parquet`.

## Category distribution (3,654 val samples)

| Category | n | % | Mean loss |
|---|---:|---:|---:|
| `correct_confident` (top-1==GT, gt_prob≥0.9) | 1,643 | 45.0 | 0.005 |
| `correct_uncertain` (top-1==GT, gt_prob<0.9) | 986 | 27.0 | 0.008 |
| `mixed_uncertainty` (wrong top-1, low conf) | 551 | 15.1 | 0.021 |
| `gt_invisible` (gt_rank>20 AND gt_prob<0.05) | 225 | 6.2 | 0.036 |
| `cooccurrence_label_ambiguity` (top-1 in secondaries) | 211 | 5.8 | 0.039 |
| `confident_wrong` (wrong top-1, prob≥0.5, rank≥5) | 38 | 1.0 | 0.033 |

**`confident_wrong` — the actual "model is confidently wrong and it's nobody's
fault but the model" bucket — is 1% of the val set.**

## Saliency informativeness (3,654 val samples)

| gt_prob range | n | % | Meaning |
|---|---:|---:|---|
| [0.05, 0.9] | 1,274 | 34.9 | Sigmoid not saturated → saliency reliable |
| > 0.9 | 1,659 | 45.4 | Saturated high → gradient vanishes, map is noise |
| < 0.05 | 721 | 19.7 | Saturated low → same problem, inverted |

**Only ~35% of samples have informative gradient saliency.** A blind
top-K highest-loss scan mostly looks at saturated maps (the two extremes
dominate the loss tails).

## Top-20 highest-loss reclassification

50% of the top-20 are `cooccurrence_label_ambiguity` — the model's top-1 IS a
known secondary species in the recording. Not a failure mode.

| Rank | Sample | Category | Saliency OK? | Note |
|---:|---|---|:---:|---|
| 01 | crbthr1 → rubthr1 | mixed_uncertainty | yes | rubthr1 NOT in 12 secondaries → real confusion |
| 02 | undtin1 → grekis | cooccurrence | no | grekis in secondaries |
| 03 | picpig2 (correct) | correct_confident | no | Loss from 7 secondaries firing |
| 04 | barant1 → chvcon1 | mixed_uncertainty | no | chvcon1 not in secondaries |
| 05 | chbmoc1 → chobla1 | cooccurrence | no | chobla1 in secondaries |
| 06 | barant1 → pvttyr1 | cooccurrence | no | pvttyr1 in secondaries |
| 07–08 | 1491113 → 22967 | cooccurrence | yes | Insect soundscape, duplicate segments |
| 09 | grasal3 (correct) | correct_uncertain | **yes** | Best saliency case in top-20 |
| 10 | yeofly1 → bobfly1 | cooccurrence | no | bobfly1 in secondaries |
| 11 | saffin (correct) | correct_confident | no | Loss from secondaries |
| 12 | strcuc1 → undtin1 | gt_invisible | no | gt rank 43 |
| 13 | soulap1 (correct) | correct_confident | no | — |
| 14–15 | 47158son03 | cooccurrence | no | Insect soundscape, duplicate |
| 16 | crbthr1 (correct) | correct_uncertain | yes | — |
| 17 | epaori4 → roahaw | gt_invisible | no | gt rank 38 |
| 18 | strcuc1 (correct) | correct_confident | no | — |
| 19–20 | 47158son03 | cooccurrence | no | Insect soundscape, duplicate |

## Implications for how we analyze the model

1. **The "top-20 highest loss" heuristic is polluted** by co-occurrence label
   ambiguity. To find real failure modes, filter to `mixed_uncertainty` +
   `confident_wrong` + `gt_invisible`.

2. **Most saliency maps are saturation noise.** Restrict saliency analysis
   to `saliency_informative=True` (~35% of samples). Otherwise the gradient
   is too small to carry signal, and brightness comparisons between panels
   are meaningless.

3. **BCE-over-234-classes inflates loss from label incompleteness.** When a
   focal recording has 7 secondary species, all of them contribute to BCE
   if they fire at all. "High loss with correct top-1" is often the secondary
   species genuinely activating — not model error. Loss ranking is therefore
   a biased view of model quality.

4. **Duplicate soundscape segments in top-20** — `1491113` and `47158son03`
   each appear multiple times as "rank N highest loss" because the dataset
   has duplicate annotation rows per 5s segment. This is a duplication bug
   in how we materialize segments, not 4 distinct failures.

## Prior interpretation errors (from the pre-triage saliency discussion)

- **rank01 crbthr1 rubthr1 "promiscuous detector" claim**: the brightness
  comparison between crbthr1 (prob 0.406) and rubthr1 (prob 0.966) saliency
  was invalid because rubthr1 is saturated. crbthr1 IS a real misprediction
  (rubthr1 not in 12 secondaries), but the visual reasoning used to
  characterize it was unsound. Saliency maps are normalized per-panel to
  their own max (`grad / grad.max()` in `attribution.py`), so absolute
  brightness is not comparable across panels.

- **rank02 undtin1 "diffuse detector has nothing to point at" claim**: wrong.
  grekis, the top-1, IS a known secondary. This sample isn't a detector
  failure at all — the model correctly found a co-occurring species and
  was penalized because it wasn't the arbitrarily-chosen primary.

- **rank03 picpig2 "secondary contamination" claim**: partially wrong. Seven
  known secondaries are present in the recording — those "contaminating"
  activations are probably real detections of co-occurring species, not
  noise. The BCE loss treats all non-primary species as negatives, which
  is mathematically inconsistent with the actual multi-label ground truth.

## Action items

- Restrict saliency-based analysis to samples with `saliency_informative=True`.
- For bulk figure export, add a `--filter-category` flag to only export
  e.g. `mixed_uncertainty + confident_wrong` subsets.
- Cross-check all future "high loss" failure claims against triage category
  before drawing conclusions.
- Investigate the `confident_wrong` bucket (n=38) — small and tractable,
  and these are the only samples that are unambiguously model failures.
- Consider de-duplicating soundscape segments in the loss parquet to avoid
  duplicate rows inflating specific segments in loss rankings.

## Files

- `analysis/triage.py` — script that produces this
- `analysis/data/triage_v5j.parquet` — enriched val table
- `analysis/figures/v5j/all/` — figures referenced above (not checked in)
