# Experiment 105: Perch pooled-embedding head

## What this is

Train a small head (linear probe; MLP optional) on top of frozen Perch v2
pooled embeddings (1536-d from `embedding` output of the stock ONNX). Tests
whether Perch's bird-domain encoder beats exp 102 Run N (frozen EfficientNet +
SED head) on BirdCLEF 2026 window-level cmAP.

Frame-level features (`spatial_embedding`, shape `[B, 16, 4, 1536]`) are
**deferred** to a follow-up. Phase 1 is the cheapest testable path.

## Status

Phase 1 — package skeleton in progress. See `proposal.md`, `tasks.md`, and
the plan at `~/.claude/plans/expressive-watching-axolotl.md`.

ONNX inspection done (job 6275860) — `spatial_embedding` is a stock output,
no graph surgery needed. See `artifacts/onnx_inspection/`.

## Running

```bash
cd user/anthony/experiments/105-perch-frame-sed

# Stage 1: pilot feature extraction (10 soundscape files, smoke test)
sbatch sbatch/extract-features.sbatch --limit 10

# Stage 1: full extraction
sbatch sbatch/extract-features.sbatch

# Stage 2: train linear probe
sbatch sbatch/train-head.sbatch --head linear
```

## Key design

- **Two stages** decoupled by a feature cache. Stage 1 (CPU, cpu-small) runs
  Perch ONNX and writes pooled embeddings as memmapped npy. Stage 2 (GPU,
  gpu-rtx6000, short) trains heads on the cache with no audio I/O.
- **Split reuse**: copies `_split_soundscape_files(seed=42)` from exp 102 so
  train/val matches exp 102 exactly (tested).
- **Loss reuse**: `FocalBCELoss` copied from exp 102, clipwise only.
- **Metric reuse**: `torchmetrics.AveragePrecision(average="macro")` matches
  exp 102's `val/cmap`.
