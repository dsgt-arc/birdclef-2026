# Tasks — Exp 033 Own-teacher pseudo-labels on distilled-sed-kd

## Tests

Tests live in `user/murilo/tests/033-own-teacher-pl/` (mirroring the experiment dir name).

Run them with the interactive venv:

```bash
~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/033-own-teacher-pl/ -v
```

| Test file | Coverage |
|---|---|
| `test_pseudo_label_integration.py` | PLDS + GTDatasetWrapper + ConcatDataset + collate fn |
| `test_loss_weighting.py` | `_dual_supervision_loss` backward-compat + per-sample weighting + training_step 3/4-tuple |
| `test_mixsamp_3source.py` | MixSamp with 3 sources (focal+sc+pl); empirical share ratios |
| `test_pl_filter.py` | PL parquet filter for val-fold leakage + labeled SS train file exclusion |

**Backward-compat gate**: `pytest user/murilo/tests/032-tucker-distilled-sed/ -q` must still show 163/163 PASS after Phase C.

---

## Reference materials

Read these BEFORE starting any task. Implementation tasks reference them by short name.

| Short name | File |
|---|---|
| **Proposal** | [`proposal.md`](proposal.md) |
| **Antoine recipe synthesis** | [`notes/kaggle-discussion-non-perch-limit.md`](../../notes/kaggle-discussion-non-perch-limit.md) |
| **Pre-flight findings (PASS)** | [`scripts/preflight_checks.py`](scripts/preflight_checks.py) (7/7) |
| **PL machinery to port (exp 025)** | `user/murilo/src/birdclef/data/sed.py:444-580` (PseudoLabelSoundscapeDataset, GTDatasetWrapper, StratifiedBatchSampler) |
| **PL loss handling reference (exp 025)** | `user/murilo/src/birdclef/lightning/sed_finetune.py:360-403` (loss_weight-aware BCE) |
| **Tucker recipe module (exp 032)** | `user/murilo/src/birdclef/lightning/distilled_sed_finetune.py` |
| **Tucker recipe data module (exp 032)** | `user/murilo/src/birdclef/data/distilled_sed_datamodule.py` |
| **Tucker recipe datasets + MixSamp (exp 032)** | `user/murilo/src/birdclef/data/distilled_sed_dataset.py` |
| **T1 PL parquet (v5j-4fold teacher)** | `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet` |
| **T2 ONNX (distilled-sed-kd-5fold)** | `experiments/010-submission-pipeline/artifacts/weights/distilled-sed-kd-5fold/model_fold{0..4}.onnx` |

---

## Phase 0 — Prerequisites: T2 inference cache + PL generation

### 0.1 Write inference caching script — DONE (commit `0729e21`)
- [x] 0.1.1 `scripts/cache_inference.py` — load 5 ONNX sessions, iterate 10,658 SS files × 12 windows, compute per-fold logits (no TTA, no smoothing). Write `logits.parquet` (schema: filename, fold, chunk_idx, start_sec, end_sec, *234_species_cols) to `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/`.
- [x] 0.1.2 `sbatch/cache_inference.sh` — wrapper; rtx6000 partition; ~30-60 min walltime.
- [ ] 0.1.3 Local sanity: run on 5 files; verify parquet schema matches `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/logits.parquet`. **(smoke-test path: `LIMIT=5 SUFFIX=_smoke sbatch sbatch/cache_inference.sh`)**

### 0.2 Write PL generation config + sbatch — DONE (commit `0729e21`)
- [x] 0.2.1 `configs/pl_gen_v2.yaml` — same temperature/power as v1 (`temperature: 0.6`, `power_transform: 1.5`, `fold_strategy: max`); points at the new T2 source; output to `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/`. Perch tier C commented out (pure-SED only).
- [x] 0.2.2 `sbatch/generate_pseudo_labels.sh` — invokes `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py --config <our v2 config>`.

### 0.3 Extend preflight + post-gen verification
- [x] 0.3.1 Extend `scripts/preflight_checks.py` with `--parquet-path` arg so it can validate both v1 and v2 parquets. (commit `5aa5558`, v1 preflight 7/7 PASS)
- [ ] 0.3.2 After T2 PL gen lands: re-run preflight against v2 parquet. Expected: same 7/7 PASS as v1.

---

## Phase 1 — Core code integration (no SLURM yet) — DONE (commit `4687b1f`)

### 1.1 Extend `DistilledSEDFinetuneModule` for per-sample loss weighting — backward-compat
- [x] 1.1.1 In `lightning/distilled_sed_finetune.py:215`: add `loss_weight: torch.Tensor | None = None` arg to `_dual_supervision_loss`. When None: existing behavior. When provided: BCE with `reduction='none'`, mean over class dim, multiply by per-sample weight, take mean.
- [x] 1.1.2 In `lightning/distilled_sed_finetune.py:265` (`training_step`): conditionally unpack 4-tuple (`loss_weight = batch[3] if len(batch) > 3 else 1.0`). Convert scalar → batch tensor when needed. Thread to `_dual_supervision_loss`.
- [x] 1.1.3 KD loss (MSE) stays unweighted (feature-space alignment is sample-independent).
- [x] 1.1.4 Verify exp 032 tests still pass: `pytest user/murilo/tests/032-tucker-distilled-sed/test_dual_supervision_loss.py -q`. Must show same 22/22 PASS as before. **(163/163 exp 032 PASS confirmed after Phase C.)**

### 1.2 Add PL dataset classes to `distilled_sed_dataset.py`
- [x] 1.2.1 Port `PseudoLabelSoundscapeDataset` from `sed.py:444-520` as `PLDS` in `distilled_sed_dataset.py`. Match the audio loader pattern of `ScDS` (existing class in same file).
- [x] 1.2.2 Port `GTDatasetWrapper` from `sed.py:523-539`. Converts 3-tuple `(wav, lb, primary_idx)` from FocalDS/ScDS into 4-tuple `(wav, lb, primary_idx, loss_weight=1.0)`.
- [x] 1.2.3 Verify MixSamp at `distilled_sed_dataset.py:473-587` already handles 3 sources (per Explore agent confirmation: yes, N-source capable).

### 1.3 Extend `DistilledSEDDataModule` with PL support
- [x] 1.3.1 Add params: `pseudo_label_path: str | None = None`, `pseudo_label_weight: float = 1.0`, `pseudo_label_sampler: str = "uniform"` (or "mixsamp"), `pseudo_label_focal_share/sc_share/pl_share: float`.
- [x] 1.3.2 In `setup()`: when `pseudo_label_path` is set, load + filter PL parquet (exclude val-fold SS files + labeled-SS train files). Build `PLDS`. Wrap focal+sc with `GTDatasetWrapper`. Combine via `ConcatDataset([gt_focal, gt_sc, pl])`.
- [x] 1.3.3 In `train_dataloader()`: dispatch on `pseudo_label_sampler`: "mixsamp" → 3-source MixSamp with configured shares; "uniform" → `DataLoader(shuffle=True)` over ConcatDataset.
- [x] 1.3.4 Update `_collate_drop_extras` in `distilled_sed_datamodule.py:498` to stack the 4th element (loss_weight) into batch tensor; preserve drop behavior for 5th `tag` element.

---

## Phase 2 — Tests + backward-compat gate — DONE (commit `1a241dd`)

### 2.1 New tests for PL integration — 32 tests, all PASS
- [x] 2.1.1 `test_pseudo_label_integration.py` — PLDS shape/types, GTDatasetWrapper attaches 1.0, ConcatDataset indexing, collate stacks loss_weight, end-to-end DataLoader yields correct schema. **(11 tests PASS)**
- [x] 2.1.2 `test_loss_weighting.py` — `_dual_supervision_loss(loss_weight=None)` matches existing; `loss_weight=ones` mathematical identity; `loss_weight=[1,0]` zeros 2nd sample; KD loss path unaffected; `training_step` handles 3-tuple AND 4-tuple. **(9 tests PASS)**
- [x] 2.1.3 `test_mixsamp_3source.py` — 3 sources, batch composition matches shares empirically, index translation correct. **(8 tests PASS)**
- [x] 2.1.4 `test_pl_filter.py` — load v1 parquet, filter excludes 14 fold-0 SS files (168 segments) + 52 labeled-SS train files (624 segments), pool size = 127,104. **(4 tests PASS)**

### 2.2 Backward-compat gate
- [x] 2.2.1 Run: `pytest user/murilo/tests/032-tucker-distilled-sed/ -q`. Must show 163/163 PASS (unchanged). **(confirmed)**
- [x] 2.2.2 Run: `pytest user/murilo/tests/033-own-teacher-pl/ -q`. Must show new tests PASS (target ~25 tests). **(32 PASS, exceeds target; combined 195/195.)**

---

## Phase 3 — train.py + fast-dev-run sanity — DONE (commit `34e3314` code; `6ab8312` verify)

### 3.1 Write exp 033 train.py
- [x] 3.1.1 `train.py` — extend exp 032's train.py (`experiments/032-tucker-distilled-sed/train.py:163-177`) with `--pseudo-label-path` arg. All other args inherit from exp 032. **(also added `_deep_merge` for `_common.yaml` inheritance.)**
- [x] 3.1.2 Pass `pseudo_label_path` + sampler config knobs to DataModule constructor.
- [x] 3.1.3 Handle warm-start: if `training.warm_start_ckpt` is set in config, load Phase D ckpt via `module.load_state_dict(strict=False)` before Trainer.fit.

### 3.2 sbatch wrappers
- [x] 3.2.1 `sbatch/_job.sh` — copy from exp 032, change run_name pattern + log dir.
- [x] 3.2.2 `sbatch/run.sh` — copy from exp 032, simplify (exp 033 just runs configs/<name>.yaml).

### 3.3 Fast-dev-run sanity — DONE (2026-05-26)
- [x] 3.3.1 Run on `a2_v5j.yaml` (simplest: no warm-start, uses existing v1 PL parquet):
  ```bash
  cd user/murilo/experiments/033-own-teacher-pl
  bash sbatch/run.sh configs/a2_v5j.yaml --fast-dev-run
  ```
  SLURM job 9185603 on `gpu-rtx6000` (atl1-1-03-007-1-0), walltime 1m28s, exit 0.
  Log: `logs/slurm_a2_v5j_9185603.log`.
- [x] 3.3.2 Verify: PLDS loaded, DataLoader yields 5-tuple batches, 5 train steps + val sanity complete, loss_weight correctly applied.
  - ✅ `Exp 033 PL mode enabled: .../v1/sed_pseudo_labels.parquet` (weight=1.0, sampler=uniform, exclude_labeled_sc=True)
  - ✅ `DistilledSEDDataModule PL filter: 127,896 → 127,728 (excl val-fold) → 127,104 (excl labeled-sc)` — matches design exactly
  - ✅ `train_dataloader (PL/uniform): focal=28906, sc=624, pl=127104, total=156,634`
  - ✅ `steps/epoch=2447` (fold 0; 28,906 focal + 624 sc → ⌈ /12 ⌉ batches, consistent with Tucker `nst` math)
  - ✅ KD active: `train/dist_loss_step: 0.021` (Perch v2 ONNX teacher loaded; alpha_distill=1.0)
  - ✅ Losses sane: `train/cls_loss_step 0.983, train/loss_step 1.004, val/loss 0.705`; no NaN/Inf
  - ✅ `Trainer.fit stopped: max_steps=5 reached` + `Job completed successfully!`

---

## Phase 4 — Run ablations + LB

### 4.1 Submit all 10 ablation training jobs in parallel
- [x] 4.1.0 **Bug fix: `num_workers` regression** (`ecc3ccb` + `87d668d`, 2026-05-26). Exp 032 inherited `num_workers=0` from Tucker's Jupyter notebook; the value propagated to exp 033 via copy-paste. With workers=0, single-threaded data loading on exp 033's 127K-file PL pool gave **~40 min/epoch** (4× slower than expected). All exps 025-031 used 8 workers; bumped exp 032 + exp 033 `_common.yaml` to match. Real per-epoch time at workers=8: **~20 min** on rtx6000 (note: still 2× slower than predicted 10 min/epoch — likely worker-startup overhead per epoch and audio I/O dominance from the large unique-file pool; acceptable).
- [x] 4.1.1 Submit T1 ablations (relaunched 2026-05-26 PM after num_workers fix):
  ```bash
  cd user/murilo/experiments/033-own-teacher-pl
  for AB in a1 a2 a3 a4 a5; do bash sbatch/run.sh configs/${AB}_v5j.yaml --fold 0; done
  ```
  Jobs: 9190539 / 9190540 / 9190542 / 9190544 / 9190545 (gpu-rtx6000).
- [x] 4.1.2 Submit T2 ablations (overnight, parallel with T1):
  ```bash
  for AB in a1 a2 a3 a4 a5; do bash sbatch/run.sh configs/${AB}_kd.yaml --fold 0; done
  ```
- [x] 4.1.3 Monitor via `squeue`. All 10 jobs completed overnight 2026-05-26 → 2026-05-27. 25-ep runs finished in 6h29m–7h25m, 10-ep runs in 2h38m–2h41m. (Real per-epoch time at `num_workers=8` was ~20 min on rtx6000, 2× the proposal's predicted 10 min — attributed to per-epoch worker startup + I/O dominance from 127K-file PL pool; acceptable.)
- [x] 4.1.4 **All 10 runs trained to full `max_epochs`** with no NaN, no instability. Each log has `Training complete!` + `outputs/test_results/2026-exp033-<slug>-fold0.json` + `last.ckpt` (or `last-v1.ckpt` for T1 relaunches) + W&B upload. **Slurm FAIL emails were a false alarm**: `_job.sh:127` calls `save_metrics.py` which was missing from exp 033 (Phase E omission — copied `_job.sh`/`run.sh` from exp 032 but not `save_metrics.py`). `set -eu` propagated the missing-file error → `--mail-type=END,FAIL` fired despite training succeeding. Fix landed 2026-05-27 (commit pending): new `experiments/033-own-teacher-pl/save_metrics.py` mirrors `train.py:load_config`'s `_common.yaml` deep-merge so per-ablation diff-only configs serialize correctly. Backfilled `artifacts/training/a{1..5}_{v5j,kd}/fold0/metrics.json` for all 10 runs on the head node. Future runs will exit 0.

### 4.2 D.5 end-to-end validation (mandatory pre-flight)
- [ ] 4.2.1 For each of 10 ckpts, run `experiments/032-tucker-distilled-sed/sbatch/validate_end_to_end.sh` with the appropriate config + CKPT env vars + FOLD=0 (val set is fold 0).
- [ ] 4.2.2 All must PASS (gap ≤ 0.02 vs training-side val/auroc).
- [ ] 4.2.3 Rank by e2e val/auroc; record into `results.md`.

### 4.3 LB submission (under 5/day cap)
- [ ] 4.3.1 Export ONNX for the top 5 by e2e rank. Use exp 032's `010-submission-pipeline/export.sbatch` patterns; may need to add wrappers for `distilled-sed-pl-a{N}-{T}` slugs.
- [ ] 4.3.2 Day 1: upload + push kernel + LB submit for top 5.
- [ ] 4.3.3 Day 2: same for remaining 5.

### 4.4 Apply decision matrix
- [ ] 4.4.1 Per proposal.md § Falsifiable: CONFIRM ≥ +0.014, PARTIAL +0.007-0.013, WEAK +0.001-0.006, REJECT ≤ 0.
- [ ] 4.4.2 Record per-ablation LB into `results.md`. Identify best protocol × teacher cell.
- [ ] 4.4.3 If CONFIRM: plan Round 2 (teacher iteration with Round 1 student as teacher) and/or Sydorskyi-yoink layering.
- [ ] 4.4.4 If PARTIAL: explore ensemble with exp 023 (0.929) to compound.
- [ ] 4.4.5 If WEAK: extend to 5-fold to disambiguate.
- [ ] 4.4.6 If REJECT: pivot to Sydorskyi-yoink directly.

---

## Closeout

- [x] C.1 (offline metrics portion) `results.md` written with full per-ablation table, paired Δ across teacher/sampler/init/PL-exposure axes, training dynamics, compute, artifacts, next-steps (2026-05-27). LB table + verdict still pending — re-update after Phase 4.2 + 4.3.
- [x] C.2 (offline metrics portion) `MODEL-RUNS.md` updated with new `## Experiment 033` section (Phase G offline metrics table, read-outs, compute, LB predictions, next steps). LB column + decision-matrix entry pending Phase 4.3.
- [ ] C.3 Update `user/murilo/CONTEXT.md` with next-session pickup notes (after D.5 + LB land).
- [ ] C.4 PR — ready for merge with verdict. Squash-merge per `feedback_squash_merge.md`.
