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

### Category × source

| Category | focal | soundscape |
|---|---:|---:|
| correct_confident | 1,613 | 30 |
| correct_uncertain | 980 | 6 |
| mixed_uncertainty | 547 | 4 |
| gt_invisible | 223 | 2 |
| cooccurrence_label_ambiguity | 81 | 130 |
| **confident_wrong** | **38** | **0** |

Two things fall out of this split:

- **`confident_wrong` is 100% focal.** The "surgical failure" bucket only
  probes the focal→focal regime — it can't tell us anything about focal→
  soundscape domain shift directly. Any modeling fix derived from these 38
  samples needs its LB effect validated separately.
- **Soundscape is 76% co-occurrence** (130 / 172). Soundscape annotations are
  inherently multi-species, so "top-1 ≠ primary but ∈ labels" is the norm,
  not an error. Raw-loss ranking on soundscape is an even worse quality signal
  than on focal — almost any soundscape row with non-trivial loss is an
  arbitrary-primary artifact, not a model failure.

## Saliency informativeness (3,654 val samples)

| gt_prob range | n | % | Meaning |
|---|---:|---:|---|
| [0.05, 0.9] | 1,274 | 34.9 | Sigmoid not saturated → saliency reliable |
| > 0.9 | 1,659 | 45.4 | Saturated high → gradient vanishes, map is noise |
| < 0.05 | 721 | 19.7 | Saturated low → same problem, inverted |

**Only ~35% of samples have informative gradient saliency.** A blind
top-K highest-loss scan mostly looks at saturated maps (the two extremes
dominate the loss tails).

### Analyzable subset: real failure AND saliency informative

Intersecting the "real failure mode" categories with `saliency_informative=True`
yields the actual set where gradient saliency can be trusted on a genuine
error:

| Filter | n |
|---|---:|
| real failure: `{mixed_uncertainty, confident_wrong, gt_invisible}` | 814 |
| AND `saliency_informative=True` | **197** |
| └── `mixed_uncertainty` | 196 |
| └── `confident_wrong` | 1 |
| └── `gt_invisible` | 0 |

**The surgical `confident_wrong` bucket is almost entirely saturated** — only
1 of 38 is saliency-informative. `gt_invisible` is by definition outside the
informative range (gt_prob < 0.05). So the saliency-actionable failure set is
really **the ~196 saliency-informative `mixed_uncertainty` samples**.

For a "what does good attention look like" baseline, `correct_uncertain`
(n=986, all saliency-informative by construction — gt_prob<0.9, top-1==GT) is
the natural comparison set.

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
   each appear multiple times as "rank N highest loss" because the losses
   parquet has one row per ground-truth annotation, and soundscape segments
   can carry multiple primary-label rows for the same `(filename, offset_sec)`.
   E.g., idx 3484 and 3496 are identical rows (same file, same 10.0s offset,
   same loss 0.110687, same primary 1491113, same top-1 22967). Not a
   materialization bug — a known consequence of how the val set is built from
   the annotation CSV. But it does inflate specific segments in loss rankings,
   so ranking should dedupe on `(filename, offset_sec)` first.

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
- For bulk figure export, add a `--filter-category` flag. Recommended default:
  `{mixed_uncertainty, confident_wrong}` AND `saliency_informative=True` AND
  dedupe on `(filename, offset_sec)`. That restricts output to the
  ~197-sample analyzable set instead of the polluted raw top-K.
- Cross-check all future "high loss" failure claims against triage category
  before drawing conclusions.
- Investigate the `confident_wrong` bucket (n=38, all focal) — these are
  unambiguous misidentifications but almost all have saturated saliency
  (only 1/38 saliency-informative), so the analysis needs another handle
  (top-1 saliency, per-frame attention, or audio inspection) rather than
  GT-side gradient saliency.
- Keep in mind `confident_wrong` is focal-only; any modeling change derived
  from it needs LB validation to confirm the fix generalizes to the
  soundscape domain.
- Dedupe on `(filename, offset_sec)` before ranking to collapse the
  `1491113` / `47158son03` duplicates.
- Use `correct_uncertain` (n=986, all saliency-informative, top-1==GT) as
  the "good attention" baseline — it's the natural counterpart to the
  `mixed_uncertainty` failure set.

## Files

- `analysis/triage.py` — script that produces this
- `analysis/data/triage_v5j.parquet` — enriched val table
- `analysis/figures/v5j/all/` — figures referenced above (not checked in)
