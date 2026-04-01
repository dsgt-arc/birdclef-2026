# Experiment 020: Perch ONNX + ProtoSSM Improvements

Fix the v4.2 regression, swap TF→ONNX Perch for 9x inference speedup,
scale ProtoSSM toward 0.925-0.930, then ensemble Perch + SED.

See [proposal.md](proposal.md) for hypothesis, motivation, and design rationale.
See [tasks.md](tasks.md) for the full task checklist and phase breakdown.

**Branch**: `mg/020-perch-onnx`
**Current Best**: 0.918 LB (Perch ProtoSSM v4, exp 018)
**Target**: 0.925-0.930 Perch alone, 0.930+ ensemble

## Phases

| Phase | Goal | LB Target | Dependencies |
|-------|------|-----------|-------------|
| A | Recover 0.918 + bisect v4.2 regression | 0.918 | None |
| B | Swap TF→ONNX Perch (~9x speedup) | 0.918 | Phase A |
| C | Scale ProtoSSM (pantanal-distill improvements) | 0.925+ | Phase B |
| D | Perch + SED ensemble | 0.930+ | Phase B+C |

## Key References

- [Perch ONNX model](https://huggingface.co/justinchuby/Perch-onnx) — 9.2x faster, Apache 2.0
- [pantanal-distill V18 annotated](../../notes/pantanal-distill-0930-annotated.md) — 0.930 LB reference
- [Exp 018 Perch Probes](../018-perch-probes/tasks.md) — v4/v4.2 history
- [Exp 019 SED Ablations](../019-sed-ablations/tasks.md) — v5j-4fold (0.913)

## Directory Structure

```
020-perch-onnx/
├── README.md
├── proposal.md
├── tasks.md
├── kaggle_kernel/
│   ├── submission_perch_protossm.py    # Phase B+C: ONNX Perch kernel
│   ├── submission_ensemble.py          # Phase D: Perch + SED ensemble
│   ├── bisect/                         # Phase A: regression bisection
│   └── kernel-metadata-*.json
├── scripts/
│   ├── validate_perch_onnx.py          # B.1: ONNX validation
│   └── upload_perch_onnx.py            # B.2: Kaggle upload
└── sbatch/
    └── validate-onnx.sbatch
```
