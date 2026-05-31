---
experiment: "038-xc-pretraining"
status: Stage 1 COMPLETE (XC-val sanity gate PASS, 0.984–0.986); Stage 2 fine-tune TRAINED (A & C × 5 folds; offline val/auroc A 0.910 / C 0.909 — tied, ≈ the 0.906 B0 baseline offline); **LB the arbiter, PENDING**
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

## Stage 2 — fine-tune on 2026 (TRAINED; offline metrics; LB the arbiter, pending)

Both Cell A (mixup-regularized backbone) and Cell C (no-mixup backbone) fine-tuned on the 234 targets
× **5 folds each** (10 runs, all 25 ep clean, rtx6000, ~5–6 h). Recipe = exp 032's 0.906 KD-SED
(`DistilledSEDFinetuneModule`, KD-from-Perch-v2, dual-BCE, mixup, Tucker mel 256, AttHeadLogitSpace)
with the encoder swapped to EffNetV2-S warm-started from each Stage-1 backbone (encoder-only; head
re-init). Final-epoch val on the held-out labeled-SS fold:

| Backbone | fold0 | fold1 | fold2 | fold3 | fold4 | **mean val/auroc** | mean val/cmap |
|---|--:|--:|--:|--:|--:|--:|--:|
| **A** (mixup) | 0.8873 | 0.9147 | 0.9192 | 0.9072 | 0.9237 | **0.9104** (±0.013) | 0.6934 |
| **C** (no-mixup) | 0.8944 | 0.9214 | 0.9069 | 0.9080 | 0.9116 | **0.9085** (±0.009) | 0.6887 |

**Read (offline only — NOT an LB verdict):**
- **A vs C is a statistical tie** (Δmean auroc 0.0019, well inside the ±0.01 fold spread; cmap
  0.693 vs 0.689). As at every prior altitude, offline labeled-SS **cannot rank** two similar SED
  models ([[project_cv_measurement_failure]]) — and it **anti-correlates with LB**
  ([[feedback_offline_metrics_unreliable]]). So this does **not** tell us which backbone transfers
  better; **LB does.** The mixup-transfer question (A vs C) is decided on the leaderboard.
- The means (A 0.910 / C 0.909) sit **right on the exp 032 B0 baseline's offline** (fold-0 val/auroc
  0.908) — i.e. our XC-pretrained V2-S backbone is *offline-competitive* with ImageNet+KD. Whether
  that converts to LB (vs the 0.906 single / 0.929 ensemble) is the open question, against the honest
  ~0.88–0.90 single-fold ceiling (exp 034/036).
- `f1_macro` (0.05–0.09) is low by construction on the sparse 234-class labeled-SS val (matches the
  exp 032 pattern); auroc/cmap are the comparable signals.

Checkpoints: `experiments/038-xc-pretraining/checkpoints/2026-exp038-finetune-cell{A,C}/fold{0-4}/`
(`save_top_k=-1`; export uses `last*.ckpt` = final epoch, the 0.906 lineage convention). Submission
wiring built (exp 010, commit baecafb): models `finetune-cell-{a,c}` (fold0) + `-5fold`.

## Next steps — LB (the arbiter)

1. **Single-fold A vs C vs 0.906** (the decisive read): export + submit `finetune-cell-a` and
   `finetune-cell-c` (fold 0). CONFIRM if either beats 0.906 (would refute the "XC-pretrain caps 0.88"
   read); else they're diverse members.
2. **5-fold ensembles** `finetune-cell-{a,c}-5fold` (logit-space fold mean) vs 0.915; then the
   **prob-space blend into the 0.929** ensemble — keep iff it beats 0.929 on LB.
3. **Canary** every submission: kernel log must print `Species mapping: 234 mapped`
   ([[feedback_species_mapping_canary]]).

**Deadline END-GAME:** the 0.929 exp-023 ensemble is already submitted + scored — just **select it as
a final submission** (+ pick the 2nd slot from the best LB result here). LB ~5/day.

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
