# Marbled White-lipped Frog Auto Motif Mining

## Header

| Field | Value |
|---|---|
| Date | 2026-05-15 |
| Status | Stable; ~0.96 curated recovery vs manual 1.00 |
| BirdCLEF label | `22967` |
| Pipeline | `src/data_analysis/auto_motif/mining.py` (species-agnostic) |
| Band discovery | `src/data_analysis/auto_motif/band.py` |
| Reference | `docs/auto-motif-pipeline.md` |
| Comparison | `scripts/compare_auto_vs_existing.py` |
| Output | `artifacts/auto_motif/marbled_frog/` |

## What this is

A species-agnostic motif-mining pipeline that takes a BirdCLEF label and
discovers the frequency bands, event size, and motif templates directly
from the focal recordings — no per-species hand-tuning. Lives in
`auto_motif/` (not `marbled_frog/`) because Marbled Frog is just the first
species we apply it to. This notebook is the Marbled Frog A/B against the
hand-tuned manual pipeline; `auto-motif-pipeline.md` is the species-agnostic
reference.

## Current result

Band discovery = **robust median core + exemplar-weighted recall
extension** (see reference doc). Discovered bands:

```text
core (median consensus):
  band 0: 1351-1602 Hz   chip / upper primary
  band 1: 1132-1352 Hz   chip / primary
  band 2:  866-1212 Hz   primary low
recall extension (exemplar-weighted consensus, outside core):
  band 3: 1839-1989 Hz   secondary low lobe
  band 4: 2382-2743 Hz   secondary high lobe
```

Per-component curated recovery (`tables/ab_curated_recovery_by_component.csv`):

| component | auto | manual |
|---|---|---|
| chip_1k_1p8k | 4/5 | 5/5 |
| primary | 4/4 | 4/4 |
| primary_independent | 10/10 | 10/10 |
| secondary | 4/4 | 4/4 |
| secondary_independent | 10/10 | 10/10 |
| **overall** | **~0.96** | 1.00 |

Precision vs manual unchanged (~0.61) on *fewer* events (≈187 vs 208) —
the gain is real recovery, not detection flooding. Band discovery is
deterministic; downstream HDBSCAN/EM give minor run-to-run variance (the
chip box is the unstable one).

## The secondary story (the lesson)

This is why band discovery is a core + extension and not a single
consensus.

The secondary is a brief broadband click whose energy peak **jitters a few
hundred Hz between recordings**. Per recording it is strong (PCEN time-p95
z = 3–10 in 6 of 8 recordings, including iNat1698107 at 7.55). But the
**cross-recording median** asks "is this *bin* salient in ≥50% of
recordings"; because the peak lands at a different bin per recording, the
median fragments it down to a low bump that ranks behind the three tall
chip/primary runs and is dropped. PCEN was not the culprit — PCEN
*preserves* the secondary far better than dB-MAD; the median washout was.

The fix weights each recording by its own strength in the species'
undisputed dominant (anchor) band, then re-runs discovery on that
exemplar-weighted consensus and keeps only the bands the median core never
covered. No tuned constant — the weight is the species' own strongest
signal. This recovered `secondary` 0/4 → 4/4 and `secondary_independent`
9/10 → 10/10 while leaving the median core (and the PCEN primary gain)
untouched. Globally *replacing* the median with the weighted consensus was
tried first and regressed the chip/primary 0.78 → 0.60 (the chip/primary
bands merged); keeping the robust median as the core and using exemplar
weighting only for recall is what works. This is the literature's
robust-estimator-core + recall-extension pattern (LTAS-variant under a
multiple-instance framing with confidence weighting).

## History

- **Baseline (dB-MAD, KDE size prior)**: 60.6%. Onset-anchor mitigation →
  75.8%.
- **PCEN + maad/HDBSCAN/RIBBIT swap (2026-05-13)**: +9 pp → ~0.78, but
  introduced the `secondary` 0/4 regression (median washout, surfaced by
  PCEN sharpening the dominant bands).
- **Robust-core + exemplar-weighted recall extension (2026-05-15)**: +18
  pp → ~0.96; secondary fully recovered, primary gain preserved.

Commit history carries the phase-by-phase A/B narrative.

## Remaining gap

One `chip_1k_1p8k` box (4/5 vs manual 5/5), and it is the run-to-run
unstable component. The remaining distance to manual's 1.00 is small and
downstream (clustering/EM), not a band-discovery problem.
