# Exp 033 — Own-teacher pseudo-labels on `distilled-sed-kd`

**Status**: scaffolding in progress (2026-05-26).

## Summary

Test whether 1:1 own-teacher pseudo-labeling (Antoine/OverfitOracle/Boredom-style) on the `distilled-sed-kd` recipe lifts single-fold LB from **0.906** (exp 032 Phase D) to **≥ 0.920** (CONFIRM).

10-ablation grid: 5 protocols × 2 teachers.

See [`proposal.md`](proposal.md) for the full hypothesis, falsifiable success matrix, and method. See [`tasks.md`](tasks.md) for the implementation checklist.

## Key references

- Tucker recipe (exp 032 base): `user/murilo/experiments/032-tucker-distilled-sed/`
- PL machinery (exp 025 source): `user/murilo/src/birdclef/data/sed.py:444-580`
- Pre-existing PL parquet T1: `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet`
- To-be-generated PL parquet T2: `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`
- Phase D checkpoint (warm-start source): `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/last-v2.ckpt`

## Quickstart

### 0. Pre-flight checks (already passed; re-run if anything upstream changed)

```bash
~/clef/birdclef-2026/.venv/bin/python user/murilo/experiments/033-own-teacher-pl/scripts/preflight_checks.py
```

### 1. Generate T2 PLs (~3 hr; only needed before T2 ablations)

```bash
cd user/murilo/experiments/033-own-teacher-pl

# Step 0.1: cache distilled-sed-kd-5fold inference on all soundscapes
sbatch sbatch/cache_inference.sh

# After cache completes, Step 0.2: generate PLs
sbatch sbatch/generate_pseudo_labels.sh

# Verify the new parquet
~/clef/birdclef-2026/.venv/bin/python scripts/preflight_checks.py \
  --parquet-path /storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet
```

### 2. Submit ablations

```bash
cd user/murilo/experiments/033-own-teacher-pl
for AB in a1 a2 a3 a4 a5; do
  for T in v5j kd; do
    bash sbatch/run.sh configs/${AB}_${T}.yaml --fold 0
  done
done
```

### 3. D.5 end-to-end validation (mandatory pre-flight before LB)

For each ckpt, run exp 032's validate_end_to_end.sh:

```bash
cd user/murilo/experiments/032-tucker-distilled-sed
CKPT=/storage/.../2026-exp033-aN-{v5j,kd}/last.ckpt \
CONFIG=../033-own-teacher-pl/configs/aN_{v5j,kd}.yaml \
TRAIN_AUROC=<from metrics.json> \
FOLD=0 \
sbatch sbatch/validate_end_to_end.sh
```

### 4. Export + Kaggle upload + LB submit

Pattern matches exp 032's `010-submission-pipeline/`. New `MODELS` entries and ONNX export wrappers will be added when ckpts land.

## Ablation grid (concise)

| ID | Init | Sampler | Focal/SC ratio | LR | Epochs | Teacher |
|---|---|---|---|---|---|---|
| **A1.v5j / A1.kd** | random | MixSamp | 0.5 / 0.5 | 5e-4 | 25 | T1 / T2 |
| **A2.v5j / A2.kd** | random | Uniform | natural ~17/83 | 5e-4 | 25 | T1 / T2 |
| **A3.v5j / A3.kd** | warm-start | MixSamp | 0.2 / 0.8 | 5e-5 | 10 | T1 / T2 |
| **A4.v5j / A4.kd** | warm-start | Uniform | natural | 5e-5 | 10 | T1 / T2 |
| **A5.v5j / A5.kd** | random | MixSamp | 0.7 / 0.3 | 5e-4 | 25 | T1 / T2 |

- **T1** = v5j-4fold (LB 0.913), via existing `v1/sed_pseudo_labels.parquet`
- **T2** = distilled-sed-kd-5fold (LB 0.915), via `v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` (to be generated)
- **Warm-start** = continue training from `distilled-sed-kd` Phase D `last-v2.ckpt` (LB 0.906)

See [`proposal.md`](proposal.md) § Method for the full rationale per ablation.

## Decision matrix

| Bucket | LB lift over 0.906 | Reading |
|---|---|---|
| **CONFIRM** | ≥ +0.014 | Round 2 + Sydorskyi-yoink |
| **PARTIAL** | +0.007 to +0.013 | Ensemble with exp 023 (0.929) |
| **WEAK** | +0.001 to +0.006 | Extend to 5-fold |
| **REJECT** | ≤ 0 | Pivot to Sydorskyi-yoink directly |

## Test suite

```bash
~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/033-own-teacher-pl/ -v
```

Backward-compat gate (exp 032 must still pass after exp 033 changes):

```bash
~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/032-tucker-distilled-sed/ -q
```
