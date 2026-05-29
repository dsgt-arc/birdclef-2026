# Results — Exp 036: Unlock the XC backbone (faithful fine-tune recipe)

**Status**: implemented — all local gates green; **PACE training + LB pending (Murilo runs)**.
**Date**: 2026-05-29
**Author**: Murilo Gustineli

## Implementation (code complete, verified locally)

Built additively on exp 034 (shared `src/`, defaults preserve exp 034 behavior). Parts:
- **Part 1 — metric suite**: `XCPretrainFinetuneModule` now logs the full exp-032 suite on
  train+val (cmap/auroc/f1_macro/f1_micro/top1/top5 + `val/n_evaluable`) + `train/logit_std`
  (overfit proxy), W&B + TB. Clip-only aggregation preserved.
- **Part 2 — flexible optimizer**: `optimizer_mode` (uniform/discriminative), `backbone_lr_mult`,
  `warmup_epochs`, `unfreeze_backbone_epoch` (+ `on_train_epoch_start` unfreeze hook). id-set
  backbone/head param split.
- **Part 3 — Phase B**: `ClassBalancedMixSamp` (focal draws class-balanced, sc uniform; batch
  composition + 4-tuple collate preserved) + `use_class_balanced_sampler` datamodule toggle.
- **Part 4 — SWA**: `scripts/swa_average.py` (post-hoc avg of last-K `model.*`; float averaged,
  int buffers from last; export-consumable).
- **Part 5 — configs/harness**: `train.py` (036), `_common.yaml` (`save_top_k=-1` +
  `save_weights_only`), cells `a0-fullft-repro` / `a1-lower-lr` / `a2-discriminative-lr` /
  `a3-gradual-unfreeze` / `b1-class-balanced`; `sbatch/{_job.sh,train.sh}`.

## Verification gates (local — all green)

| Gate | Check | Result |
|---|---|---|
| 0 | baseline suite (032/033/034) | ✅ 258 passed |
| 1 | metric suite unit + fast-dev-run + back-compat | ✅ |
| 2 | optimizer modes unit + A1/A2/A3 fast-dev-run | ✅ |
| 3 | class-balanced sampler unit + b1 fast-dev-run | ✅ |
| 4 | SWA unit (mean / int-from-last / export-consumable) | ✅ 7/7 |
| 5 | configs load + inherit chain + a0==exp034-fullft | ✅ |
| 5+7 | full exp036 suite + 032/033/034 regression | ✅ **290 passed** (258 + 32 new) |
| 6 | **A0 parity on PACE** (val/auroc ≈ exp 034 fullft 0.883) | ⬜ Murilo runs |

## Run commands (PACE — Murilo) — LB-arbitrated, no trustworthy CV

```bash
cd user/murilo/experiments/036-xc-backbone-finetune
# Gate 6 first: A0 must track exp 034 fullft (W&B gvx4, 0.883). Material divergence → STOP.
bash sbatch/train.sh                                            # a0-fullft-repro (parity)
# Phase A (parallel) — PACE not the bottleneck:
CONFIG=configs/a1-lower-lr.yaml          bash sbatch/train.sh
CONFIG=configs/a2-discriminative-lr.yaml bash sbatch/train.sh
CONFIG=configs/a3-gradual-unfreeze.yaml  bash sbatch/train.sh
# → export+LB the top 1–2 vs exp 034's 0.879 (mirror exp 034: export_xcpretrain.py + upload_kaggle).
# Phase B: re-point b1's `inherit:` to the winning A cell, then:
CONFIG=configs/b1-class-balanced.yaml    bash sbatch/train.sh

# Checkpoint candidates per run (save_top_k=-1):
#   SWA (lead): python scripts/swa_average.py --ckpt-dir <PROJECT>/experiments/036-xc-backbone-finetune/checkpoints/<run>/fold0 --last-k 5 --output .../swa-last5.ckpt
#   last = highest epochNN.ckpt ; best = max val/auroc epoch (tb:metrics)
# Export each candidate via export_xcpretrain.py (SWA ckpt exports through the SAME path).
```

**Success**: LB > 0.906 (beat our ImageNet ceiling on the XC backbone) — real progress; > 0.93 = goal.
**Reject**: nothing clears 0.879 after LR fix + faithful data → fall back to securing the 0.929 + PP/TTA picks.
**Caveat**: offline (incl. cv_trust honest OOF) is a *rough screen* only (us + Antoine) — LB decides.
Before submission: run the FULL `user/murilo/tests/` suite (Gate 7).
