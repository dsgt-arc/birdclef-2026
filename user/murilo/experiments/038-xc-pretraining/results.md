---
experiment: "038-xc-pretraining"
status: Stage 1 COMPLETE — all 3 cells PASS the XC-val sanity gate (val/auroc 0.984–0.986 ≫ 0.95 threshold); Stage 2 fine-tune + LB PENDING
date: 2026-05-31
author: Murilo Gustineli
---

# Results — Exp 038: Broad Neotropical XC Pretraining (EffNetV2-S)

**Status (2026-05-31): Stage 1 (pretraining) COMPLETE — sanity gate PASS, decisively.** All 3 cells
(A dual-BCE+LS / B focal-BCE / C no-mixup) trained the full 25 epochs cleanly on rtx6000 and reach
**held-out XC macro-AUROC 0.984–0.986** vs the proposal's **≳0.95 "pretrain works" threshold** — the
broad Neotropical bird-ID loop demonstrably learned (top-5 ~89% across **2,683 species**). The
deep-research **abandon-from-scratch tripwire was NOT triggered**. As predicted, **AUROC ranks the 3
cells within 0.0013** (un-rankable — XC-val is a *sanity gate*, not a 2026-LB proxy:
[[project_cv_measurement_failure]], [[feedback_offline_metrics_unreliable]]). **Stage 2 (re-head to
234 → KD-SED fine-tune → LB) is the arbiter and is NOT YET RUN.** This experiment's honest ceiling is
~0.88–0.90 single-fold (exp 034/036) — value = backbone-diverse ensemble member + working-note, not a
plateau-break. **Date**: 2026-05-31 · Murilo

## Sanity-gate verdict (Stage 1, the only signal we have so far)

| Gate (proposal §5) | Bar | Result | Verdict |
|---|---|---|---|
| **Pretrain works** (held-out XC macro-AUROC, in-distribution) | ≳0.95 | **0.984–0.986** (all 3 cells) | **PASS** (decisive) |
| **Abandon-from-scratch tripwire** (pretrain must beat ImageNet-init on XC by smoke ckpt) | — | smooth monotonic climb, no divergence | **NOT triggered** |

This says only that the multi-label loop learned broad XC bird ID — it does **not** predict the
downstream 2026 LB (focal→soundscape domain shift is unchanged by pretraining). LB decides at Stage 2.

## Stage 1 — XC-val metrics at the val/auroc-peak epoch (the backbone we'd export)

EffNetV2-S (`tf_efficientnetv2_s.in21k`, embed_dim 1280), SED-mirror head (`AttHeadLogitSpace`,
num_classes **2,683**, KD off), Tucker mel (n_mels=256, sr=32000), 5 s random crop, dual-supervision
`0.5·BCE(clip)+0.5·BCE(frame_max)`. **148,427 train / 2,834 val** (stratified per-species 2% val;
`require_audio` drops the ~5.8% missing OGGs). Eff batch 128, AdamW lr 5e-4 → cosine→1e-6, 25 ep,
`save_top_k=-1`. rtx6000, ~10 min/epoch, ~5h50m walltime/cell.

| Cell | recipe (lever) | loss | export ckpt | **best val/auroc** | val/cmap | val/f1_macro | val/f1_micro | val/top1 | val/top5 | job |
|---|---|---|:--:|---:|---:|---:|---:|---:|---:|---|
| **A** | dual-BCE + **LS 0.05** + full aug (PRIMARY) | bce | epoch23 | 0.9843 | 0.6753 | 0.2324 | 0.4858 | 0.7445 | 0.8892 | 9351785 |
| **B** | **focal-BCE** (`FocalLossBCE` α0.25/γ2) | focalbce | epoch23 | **0.9847** | 0.6746 | 0.2113 | 0.4558 | 0.7322 | 0.8913 | 9351788 |
| **C** | dual-BCE, **mixup OFF** | bce | epoch22 | **0.9856** | 0.6737 | **0.2678** | **0.5164** | **0.7555** | 0.8867 | 9351789 |

(**best val/auroc** = the monitor peak = the `save_top_k=-1` checkpoint Stage 2 loads. **cmap / f1 /
top-k are the final-epoch (ep24) values** logged to W&B — they match `artifacts/training/*/metrics.json`
exactly; the final-epoch auroc 0.9842 / 0.9846 / 0.9854 sits within 0.0001 of the peak. `val/loss`
~0.0020–0.0024 is tiny by construction — BCE over 2,683 mostly-zero targets — not meaningful in
absolute terms. Verified against `outputs/wandb/wandb/<run>/files/wandb-summary.json`.)

## What happened (analysis)

1. **All 3 learned strongly, monotonically, no overfit.** val/auroc climbs 0.51 (random, ep0) →
   ~0.97 by ep6 → plateau ~0.984–0.986 by ep17–24; cmap still creeping at the end (0.674–0.676,
   plateauing under the cosine→1e-6 tail). No divergence, no NaN, logit scale healthy. **All 3 real
   runs completed 25 ep on the first try** (each preceded by a one-off `fast_dev_run` smoke, job
   9351455, + a `gpu-check` throughput probe, job 9351577 — pre-flight checks, *not* restarts; both
   wrote into cellA's dir, hence the extra logs / the 33 KB stray TB event file).

2. **AUROC cannot rank the cells (Δ ≤ 0.0013) — as the methodology predicted.** This is the expected
   "similar models are un-rankable by our offline metric" pattern ([[project_cv_measurement_failure]]).
   So we do **not** pick the Stage-2 backbone by XC-val.

3. **But the sharpness metrics DO separate, and the train-vs-val gap reveals the mixup trade-off
   cleanly.** Cell **C (no-mixup) is sharpest in-distribution** — final-epoch val f1_macro **0.268** /
   f1_micro **0.516** / top-1 **0.755**, vs A (0.232/0.486/0.745) and B (0.211/0.456/0.732). The effect
   is starkest on **train**: train/f1_micro **C 0.532 ≫ A 0.323 / B 0.300** and train/auroc **C 0.994
   vs A 0.899 / B 0.898**. This is exactly mixup as a regularizer: with mixup on (A, B) the *training*
   targets are convex-mixed, so the model can never fit them sharply (low train f1/auroc); with mixup
   off (C) it fits train hard (train/auroc 0.994). Crucially, **on held-out XC the gap mostly closes**
   (val/auroc C 0.985 vs A 0.984 — Δ0.001; val/f1_micro 0.516 vs 0.486) — i.e. mixup's train-time
   handicap is a generalization tax that largely pays itself back on val, which is the whole point of
   the regularizer. **Whether it *also* pays back on the focal→soundscape shift is exactly what XC-val
   cannot tell us — only Stage-2 LB can.** → keeps the plan honest: **fine-tune Cell A first** (it
   carries the mixup + LS regularization closest to our proven 0.906 downstream recipe), with **C as
   the explicit "sharper / less-regularized" contrast** and **B as the focal contrast** — ready
   backups (no second wait), screened on LB only if A is promising.

4. **Absolute f1_macro (0.22–0.26) and cmap (~0.675) are modest — and that's normal here.** 2,683-class
   *extreme* multi-label with **random 5 s crops** (a crop can miss the labeled vocalization → label
   noise) and macro-averaging over a long tail of rare species depresses F1/cMAP. **AUROC (ranking) and
   top-k are the trustworthy signals**, and they are strong (top-5 ~0.89 across 2,683 classes).

## Caveat (load-bearing) — Stage 1 is NOT a verdict on the experiment

The proposal's falsifiable test is **downstream single-fold LB vs 0.906** (CONFIRM if it beats 0.906 —
which would refute the "pretraining caps at 0.88" read; else it is a diverse ensemble member, kept iff
the prob-space blend into 0.929 beats 0.929). Stage 1's strong XC-val says the **backbone is well-formed
and worth fine-tuning** — nothing more. Exp 034/036 already showed a broad-XC backbone fine-tunes to
~0.88 < our 0.906 KD; the open question Stage 2 answers is whether *our own* supervised XC pretrain
(under the exact SED regime) does better than warm-starting someone else's bird checkpoint.

## Next steps

**A) exp 038 Stage 2 (NOT YET BUILT):** load Cell A `epoch23.ckpt` backbone → re-init a 234-class
`AttHeadLogitSpace` → run the exp 032 KD-SED fine-tune recipe (KD-from-Perch + labeled-SS + dual-BCE +
mixup; reuse `DistilledSEDFinetuneModule` with KD **on**) → exp 010 ONNX export → **LB** (single-fold
vs 0.906; 5-fold vs 0.915; prob-space blend into 0.929). B/C backbones are backups.

**B) Deadline END-GAME (higher deadline-EV, still pending):** secure the 0.929 exp-023 ensemble as
final pick #1 (reproducible + submittable; `Species mapping: 234 mapped` canary —
[[feedback_species_mapping_canary]]); PP/TTA A/B for pick #2. **Deadline 2026-06-03 (~3 days), LB
rate-limited (~5/day, ~14 pulls left)** — weigh Stage-2's LB pull against locking the final picks.

## Artifacts

| Type | Location |
|---|---|
| Checkpoints (all 25 ep + last) | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/038-xc-pretraining/checkpoints/2026-exp038-pretrain-cell{A,B,C}-*/` |
| Best ckpt (Stage-2 input) | cellA `epoch23.ckpt` · cellB `epoch23.ckpt` · cellC `epoch22.ckpt` (= `best_checkpoint_path` in each `pretrain_results.json`) |
| Structured metrics | `artifacts/training/pretrain_cell{A,B,C}_*/metrics.json` (regenerated post-hoc via `save_metrics.py`; `best_val_auroc` + full provenance; per-epoch val curve in TB/W&B) |
| Pretrain results | `checkpoints/<run>/pretrain_results.json` (`best_val_auroc`, best/last ckpt paths, stopped_epoch=25) |
| Logs | `logs/slurm_pretrain_cell{A,B,C}_*_935178{5,8,9}.log` (real runs) + `_9351455` (fast_dev_run smoke) / `_9351577` (gpu-check probe) |
| TensorBoard / W&B | `outputs/tensorboard/exp038-*/` · W&B project `birdclef-2026-exp038` (runs c27deyi9 / ip1hn7zj / zlr0itcd) |
| Species mapping | `checkpoints/<run>/species_to_idx.json` (2,683 entries) |
| Corpus | `shared/pretrain/birdset-neotropical/` (`manifest.parquet`, `species_list.json`, `audio/<ebird_code>/<XCid>.ogg`) |

## Implementation (Stage 1)

New code only in the data layer + 2 gated SED-module hparams: `src/birdclef/data/xc_pretrain_dataset.py`
(`XCPretrainDataset` multi-hot incl. in-vocab secondaries / random 5 s crop / `apply_aug`+focal×focal
mixup ported from `FocalDS`; stratified per-species `XCPretrainDataModule`); gated additive
`loss_type="focalbce"` (lazy torchvision import) + `label_smoothing` in `DistilledSEDFinetuneModule`
(the `bce`/`softauc` paths are **bit-identical**, regression-tested; LS applied in-loss to a local
target copy so `labels.int()` metrics stay hard). 23 new unit tests + 238 regression (032/033/037) PASS;
fast-dev-run (a100) + gpu-check (rtx6000) PASS. `set_float32_matmul_precision('high')` for TF32.
**Note:** the 3 cells launched *before* the metric-parity commit → `outputs/test_results/` + auto-
`save_metrics` were absent; `metrics.json` regenerated post-hoc — `save_metrics.py` was extended with a
fallback that reads `best_val_auroc`/checkpoint paths from `pretrain_results.json` and reconstructs the
**final-epoch `val/*` vector from the W&B run summary** (`wandb-summary.json`, matched by `run_name` via
the run's `config.yaml`) = the same callback-metrics semantics train.py writes for normal runs.
