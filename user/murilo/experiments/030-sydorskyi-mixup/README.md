# Experiment 030: Sydorskyi-faithful MixUp on v5j

Status: **proposed** (configs scaffolded, training pending)
Date: 2026-04-26

A/B comparison testing whether replacing v5j's Zhang-style interpolated
spectrogram MixUp with Sydorskyi 2025 2nd-place's faithful waveform recipe
moves LB.

## TL;DR

| Variant | Backbone | Mixup recipe | Phase |
|---|---|---|---|
| `v5j_control` | HGNetV2-B0 | spec, interp, theta=0.8 (current v5j) | A |
| `v5j_sydorskyi` | HGNetV2-B0 | wave, `(w1+w2)/2` + peak-norm + max(y1,y2), p=0.5 | A |
| `effnetv2b0_control` | EfficientNetV2-B0 | spec, interp, theta=0.8 | B (extension) |
| `effnetv2b0_sydorskyi` | EfficientNetV2-B0 | wave + Sydorskyi mixup | B (extension) |

Phase A is the load-bearing test (single-variable mixup change on v5j).
Phase B adds a 2x2 ablation with EfficientNetV2-B0 (Salman's choice for
his single-model 0.937 LB) to attribute backbone vs mixup main effects.

## How to run

```bash
cd user/murilo/experiments/030-sydorskyi-mixup

# Phase A — committed test:
bash sbatch/run.sh configs/v5j_control.yaml
bash sbatch/run.sh configs/v5j_sydorskyi.yaml

# Phase B — 2x2 extension:
bash sbatch/run.sh configs/effnetv2b0_control.yaml
bash sbatch/run.sh configs/effnetv2b0_sydorskyi.yaml

# Sanity check first (5 batches, no W&B):
bash sbatch/run.sh configs/v5j_sydorskyi.yaml --fast-dev-run
```

W&B project: `birdclef-2026-exp030`. All four runs go to the same project
for easy curve comparison.

## What this tests vs what it doesn't

**Tests**: whether the MixUp target distribution mismatch (audit finding:
80% of v5j's training target values are sub-1) is a single-variable LB
lever, isolated from all other Sydorskyi pipeline differences.

**Does not test**: full Sydorskyi pipeline replication (no in-domain
pretrain, no BG noise, no SpecAugment, no balanced sampler, no focal
loss, no label smoothing, no pseudo-distillation). Each of those is a
separate axis for separate experiments.

## Decision matrix (single-fold LB on v5j arm)

| Control LB | Sydorskyi LB | Verdict |
|---|---|---|
| ≥ 0.901 | ≥ 0.911 | Strong confirm — endorse, queue 4-fold |
| ≥ 0.901 | 0.901-0.911 | Within seed noise — repeat with different seed |
| ≥ 0.901 | ≤ 0.895 | Reject — pivot to pseudo-distillation |
| <0.895 | (any) | Reproducibility issue — halt and investigate |

## Files

- `proposal.md` — hypothesis, method, scoping decisions
- `tasks.md` — execution checklist
- `configs/` — 4 YAML configs (one per variant)
- `train.py` — training script (copied from exp 029, retains transfer-learning
  logic which is unused for exp 030 since no `init` block in configs)
- `save_metrics.py` — post-training W&B metric extraction
- `sbatch/_job.sh` + `sbatch/run.sh` — SLURM job submission
- `pyproject.toml` — Python deps (matches exp 029)

## References

- Audit: `user/murilo/analysis/notes/2026-04-26-mixup-implementation-audit.md`
- Simulation: `user/murilo/analysis/notes/2026-04-26-mixup-target-simulation-results.md`
- D1 verdict: `user/murilo/analysis/notes/2026-04-26-d1-train-side-attention-verdict.md`
- Sydorskyi reference: `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md`
- Implementation: `mixup_mode="average_max"` added to
  `user/murilo/src/birdclef/lightning/sed_finetune.py:_apply_mixup`
