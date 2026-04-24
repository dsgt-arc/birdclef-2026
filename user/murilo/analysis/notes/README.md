# Analysis notes

Findings captured from interactive analysis sessions using the tools in
`analysis/` (model_explorer notebook, export_figures, triage). One file per
self-contained topic, date-prefixed, newest first.

Keep each file readable on its own — don't require a reader to reconstruct
the conversation. Link to data artifacts (parquets, figures) rather than
pasting large tables.

## Index

- [2026-04-24 — roahaw failure-mode analysis: silent crops vs acoustic confusion](2026-04-24-roahaw-failure-mode-analysis.md) — roahaw is NOT rare (92nd percentile in training), so balanced sampling wouldn't help. ~14/15 mixed_uncertainty roahaw samples are silent/low-content crops where BCE penalizes the model for being honest; 1/15 is a real `saffin` vs `roahaw` confusion — and that's the sample `saliency_informative=True` flagged. Methodology validated.
- [2026-04-24 — certainty scalars + species composition of v5j failures](2026-04-24-certainty-and-species-composition.md) — `shannon_norm` and `top1_prob` are ~equivalent (corr −0.86); entropy is confirmation of triage consistency, not a new signal. Species composition shows `mixed_uncertainty` is non-uniform across species: `roahaw` 71% failure rate at n=15, top-10 species drive 21% of the bucket.
- [2026-04-24 — v5j triage: failure modes are mostly not failures](2026-04-24-triage-v5j-failure-modes.md) — first pass of `triage.py` on v5j. 50% of top-20 highest-loss is label co-occurrence, only 35% of samples have informative saliency, and `confident_wrong` is just 1% of the val set.
