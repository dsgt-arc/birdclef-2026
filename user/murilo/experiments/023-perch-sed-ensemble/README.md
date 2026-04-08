# Experiment 023: Perch + SED Ensemble

Combine Perch ProtoSSM v18 (0.925 LB) with SED v5j-4fold-pp2 (0.917 LB) via
score-level fusion in a single Kaggle kernel.

See [proposal.md](proposal.md) for hypothesis, motivation, and design rationale.
See [tasks.md](tasks.md) for the full task checklist and phase breakdown.

**Branch**: `mg/023-ensemble`
**Target**: 0.930+ LB

## Directory Structure

```
023-perch-sed-ensemble/
├── README.md
├── proposal.md
├── tasks.md
├── results.md
└── kaggle_kernel/
    ├── submission_ensemble.py          # Combined Perch ONNX + SED kernel
    └── kernel-metadata-ensemble.json   # Kaggle push metadata
```

## Key References

- [Exp 020 Perch ONNX](../020-perch-onnx/) — Perch ProtoSSM pipeline source
- [Exp 010 Submission Pipeline](../010-submission-pipeline/) — SED submission source
- [Pattern Analysis](../../notes/2026-04-06-experiment-pattern-analysis.md) — evidence base
