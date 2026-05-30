# Results — Exp 036: Unlock the XC backbone (faithful fine-tune recipe)

**Status**: **REJECT.** Phase A trained (4 cells, fold 0); Gate 6 PASS; a1-lower-lr won
offline (0.921) but **LB 0.881 (SWA) / 0.882 (last) ≈ the 0.879 baseline = flat**. Phase B/C
not pursued. Pivot to final-submission picks (0.929 ensemble + PP/TTA) before 2026-06-03.
**Date**: 2026-05-30
**Author**: Murilo Gustineli

## Phase A results (2026-05-30) — LR-sanity, fold 0, 50 ep each

All 4 cells finished clean (50 epoch ckpts + `last.ckpt` each, `save_top_k=-1`). The
**best-by-val/auroc** trajectory (the metric exp 034 reported) is what matters — last-epoch
`metrics.json` is misleading (cosine LR decays the late epochs).

| Cell | best val/auroc | best ep | last ep | shape over 50 ep | logit_std |
|---|---:|---:|---:|---|---:|
| a0-fullft (lr 1e-4) | 0.906 | ~3 | 0.875 | peaks ep3 → **decays −0.031** (overfits) | 1.65 |
| **a1-lower-lr (5e-5)** | **0.921** | ~34 | 0.918 | climbs → **flat 0.91–0.92 plateau (ep19–50)** | 1.62 |
| a2-discriminative | 0.888 | ~15 | 0.882 | flat-low | 1.56 |
| a3-gradual-unfreeze | 0.914 | ~9 | 0.900 | peaks ep9 → settles ~0.90 | 1.64 |

**Gate 6 — PASS (harness faithful).** a0 reproduces exp 034 full-FT *essentially
bit-identically* epoch-by-epoch (ep1 0.8918/0.8919, ep2 0.8890/0.8889, ep3 0.9063/0.9059,
ep4 0.8928/0.8923 — diffs ~1e-4 = dataloader-shuffle noise). exp 034 full-FT's *true* offline
peak was also ~0.906 @ep3; the "0.883" in prior handoffs was a late/representative epoch, not
the best. The added metric suite did **not** perturb training. **LB anchor:** a0 ≡ exp034-fullft,
offline-best 0.906 → **LB 0.865 raw / 0.879 +PP**.

**Sub-claim 1 CONFIRMED (offline).** Full-FT @1e-4 overfits the in-domain backbone (peaks ep3
then collapses 0.906→0.875). Lower LR (a1) *prevents the collapse* → stable 0.91–0.92 plateau
across the whole back half. a1 is the clear pick: best offline (+0.015 over the 0.906 anchor),
cleanest mechanism, and the **ideal SWA shape** (flat tail). a3 is a viable second mechanism;
a2 is dead.

**Caveat (must weight):** offline anti-correlates with LB here ([[feedback_offline_metrics_unreliable]],
[[project_cv_measurement_failure]]). exp 034 **KD-joint scored offline 0.9217 → LB 0.878 (zero
lift)** — and a1 lands at the *same* 0.92 offline number. The reason to still bet on a1: KD's
0.92 came from a *different objective*; a1's is the **same objective as a0, LR-only, fixing a
visible pathology**. LB-or-bust.

### LB verdict (2026-05-30) — REJECT: gentler LR does NOT unlock the backbone

Submitted **a1-SWA** (avg of ep45–49, `swa-last5.ckpt`) + **a1-last** (`last.ckpt`), both +PP,
vs the 0.879 baseline. Wiring reused `export_xcpretrain.py` (identical graph; smoke-tested: both
ckpts loaded 786 `model.*`, 0 missing/unexpected, 234-identity mapping).

| Submission | offline val/auroc | LB | Δ vs exp034 fullft+PP (0.879) |
|---|---:|---:|---:|
| a1-SWA + PP (`xcfinetune-effv2s-lowlr-swa`) | 0.917 (SWA) | **0.881** | +0.002 |
| a1-last + PP (`xcfinetune-effv2s-lowlr-last`) | 0.921 | **0.882** | +0.003 |

**Read — three things, all converging:**
1. **Flat on LB.** +0.002/+0.003 = within noise. The LR fix did **not** transfer. The
   "overfitting" a1 cured offline (0.875→0.921) was overfitting to the **noisy labeled-SS val
   metric**, not improved generalization to the LB test distribution.
2. **Hard single-fold ceiling, confirmed 3×.** Vanilla full-FT (0.879), +KD (0.878), +lower-LR
   (0.881/0.882) — three different fine-tune recipes on this XC backbone+head all land in
   **0.878–0.882**, regardless of schedule.
3. **Offline metric proven useless above ~0.88 LB.** 2nd offline→LB anti-correlation at the
   *same* ~0.92→~0.88 mapping (1st: exp034 KD-joint 0.9217→0.878). **SWA added nothing**
   (0.881 ≤ 0.882 last). No way to screen further variants offline → any next step is a blind shot.

**Sub-claim 1 (Phase A): REJECTED on LB.** **Phase B (class-balanced Sydorskyi data — the
proposal's "main lever") and Phase C (PL+PP): NOT PURSUED.** Rationale (decision 2026-05-30):
(a) base rate ≈ 0 — our history shows training-recipe changes don't move our LB (exp 027
"16+ changes, 0 wins"); (b) offline can't screen it (blind LB shot); (c) deadline 2026-06-03
EV favors locking in the **0.929 ensemble + PP/TTA** final picks; (d) even an optimistic Phase B
(~0.90) stays below our 0.906 single best and the 0.929 ensemble. **The XC model stays available
as a backbone-diverse ENSEMBLE member** (the genuinely-promising path exp 034 flagged) — that is
a separate, cheap test, not part of this experiment's fine-tune thesis.

**Net:** exp 036 confirms exp 034's "necessary-not-sufficient" verdict from the fine-tune angle —
the XC backbone's ~0.88 single-fold ceiling is set by the **omitted stack** (PL + 5-fold +
ensemble), not by the fine-tune schedule. Reusable assets: the metric suite, flexible optimizer,
`ClassBalancedMixSamp`, `swa_average.py`, and the exp 010 export/upload/kernel wiring all remain.

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
