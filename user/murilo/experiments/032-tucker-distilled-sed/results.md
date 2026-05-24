---
experiment: "032-tucker-distilled-sed"
status: in-progress
date: 2026-05-23
author: Murilo Gustineli
---

# Results: Tucker distilled-SED full replication

**Date Completed**: TBD
**Author**: Murilo Gustineli

## Status: IN-PROGRESS

Phase 0 (porting) underway on branch `mg/exp032-tucker-distilled-sed-replication` (PR #51).

See [`proposal.md`](proposal.md) for the hypothesis + method, and [`tasks.md`](tasks.md) for the current checklist.

This file will be filled in as each phase completes.

## Summary

[One-line: what happened? Did the hypothesis hold?]

**Verdict**: [TBD — confirmed | partial | weak | rejected]

## Observations

### Quantitative

| Run | Code | Node | Ckpt | LB | Δ vs paired ctrl | Δ vs v5j_control_last 0.897 |
|---|---|---|---|---:|---:|---:|
| distilled_sed_control | HEAD | TBD | best-by-macro | TBD | — | TBD |
| distilled_sed_kd | HEAD | TBD | best-by-macro | TBD | TBD | TBD |

### Qualitative

- TBD

## Analysis

[What do these results mean? Why did we get this outcome?]

### Confounders

- Single fold — ±0.013 seed band on v5j HGNet; EffNet variance unknown
- Multi-variable port — paired Δ isolates KD branch within new stack, but distilled_sed_control vs v5j_control_last 0.897 conflates backbone + head + loss + mixup + SpecAugment + schedule + sampler

## Artifacts

| Type | Location |
|---|---|
| Configs | `configs/distilled_sed_{control,kd}.yaml` |
| Logs | `logs/` |
| W&B | project `birdclef-2026-exp032` (TBD) |
| TensorBoard | `outputs/tensorboard/` |
| Checkpoints | `/storage/project/.../032-tucker-distilled-sed/checkpoints/` (TBD) |
| Metrics JSON | `artifacts/training/{distilled_sed_control,distilled_sed_kd}/metrics.json` |
| Code | branch `mg/exp032-tucker-distilled-sed-replication`, PR #51 |

## Next Steps

See [`tasks.md`](tasks.md) Phases 2–3 + § Closeout for the decision-tree from the Phase 1 verdict.

**Leads to**: TBD (depends on Phase 1 outcome — see proposal § Falsifiable).
