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

### 4.2 D.5 end-to-end validation (mandatory pre-flight) — DONE (2026-05-27)
- [x] 4.2.1 D.5 launcher `sbatch/launch_d5_validation.sh` written (auto-picks `last-v1.ckpt` for T1, `last.ckpt` for T2). Used exp 032's `validate_end_to_end.sh` with the `_common.yaml` deep-merge patch landed in commit `49d20d9`. Jobs 9196512–9196521 submitted 2026-05-27 10:13.
- [x] 4.2.2 **All 10 PASS.** Max gap 0.0079 (well under 0.02 tolerance). E2E generally higher than training-side on warm-start runs (a3, a4 × both teachers: +0.004 to +0.007 — Tucker's "smoothing-trick" pattern); mixed on random-init runs.
- [x] 4.2.3 Rank by e2e val/auroc → top 5 = a4-kd, a3-kd, a2-kd ⭐, a4-v5j, a3-v5j. Recorded in results.md + MODEL-RUNS.md.

### 4.3 LB submission (under 5/day cap) — STOPPED at 3/5 Day 1 after REJECT trajectory confirmed
- [x] 4.3.1 ONNX export wired into exp 010 submission pipeline (commit `4280e72`):
  - patched `scripts/export_distilled_sed.py` with `load_config_with_common` to handle `_common.yaml` inheritance;
  - added `EXP033_DIR` + generic `export_distilled_sed_pl()` function in `sbatch/export.sbatch`;
  - added 10 MODELS entries to `scripts/upload_kaggle.py`;
  - generated 10 kernel-metadata JSONs with verified title-to-id slug match;
  - added 10 weight slugs to `submission.py` MODEL_SLUGS.
  Day-1 batch export (job 9198770): all 5 cleanly exported (walltime 2m11s; ONNX verification max diff 3.81e-6 to 4.77e-6 across the 5).
- [x] 4.3.2 Day 1: 3 of 5 submitted to LB; all REJECT.
  - **a4-kd**: LB **0.899** (Δ −0.007). On the edge of cross-hardware noise.
  - **a2-kd** ⭐: LB **0.886** (Δ −0.020). Firmly inside REJECT band; canonical Antoine doesn't work.
  - **a4-v5j**: LB **0.897** (Δ −0.009). Paired with a4-kd → teacher-axis Δ = +0.002 LB, refutes the diversity hypothesis.
  - Remaining 2 Day-1 cells (a3-kd, a3-v5j) NOT submitted — trajectory said don't burn slots.
- [ ] 4.3.3 ~~Day 2: same for remaining 5.~~ NOT executed (decision matrix triggered REJECT pivot; saving Day-2 slots for Sydorskyi-yoink).

### 4.4 Apply decision matrix — DONE: REJECT
- [x] 4.4.1 Per proposal.md § Falsifiable: all 3 LB Δ in REJECT band (≤ 0).
- [x] 4.4.2 Per-ablation LB recorded into results.md + MODEL-RUNS.md. Best LB this experiment: a4-kd 0.899 (still below the 0.906 baseline).
- [ ] ~~4.4.3 If CONFIRM: plan Round 2 + Sydorskyi-yoink layering.~~ Not triggered.
- [ ] ~~4.4.4 If PARTIAL: ensemble with exp 023 (0.929).~~ Not triggered.
- [ ] ~~4.4.5 If WEAK: 5-fold disambiguation.~~ Not triggered.
- [x] 4.4.6 **REJECT → pivot to Sydorskyi-yoink directly.** Active path; scoping next session.

---

## Closeout

- [x] C.1 `results.md` updated 2026-05-27 PM with LB verdict (REJECT), 3-cell LB table, paired Δ readings (teacher axis +0.002 LB refuting offline +0.011 auroc; protocol-cell axis −0.013 LB; offline-LB anti-correlation data point #24), next-steps pointing at Sydorskyi-yoink pivot.
- [x] C.2 `MODEL-RUNS.md` exp 033 header updated to "REJECT"; offline-metrics table re-titled with LB column; LB-expectations section replaced with actual D.5 + LB results table + verdict + paired Δ readings.
- [ ] C.3 Update `user/murilo/CONTEXT.md` with next-session pickup notes (Sydorskyi-yoink pivot scope + remaining Day-2 LB cap budget).
- [ ] C.4 PR — ready for merge with REJECT verdict. Squash-merge per `feedback_squash_merge.md`.

---

## Phase 5 — Sydorskyi-divergence diagnostic tests (2026-05-27)

REJECT verdict was suspicious — exp 033 implementation diverges from the only published 0.93+ PL recipe (Sydorskyi 2025 2nd place) on 16 of 20 axes. Full audit at `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md`. Phase 5 reissues the experiment with the cheapest divergence fixes before pivoting.

**Detailed workflow + commands + decision criteria**: `experiments/033-own-teacher-pl/diagnostic-tests.md` (resume-after-disconnect anchor; read this first if returning to this work).

### Test 1 — chunk discard + trim probability (LANDED CODE, READY TO LAUNCH)

- [x] 5.1.1 Modify `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` to support optional `filter:` section (`min_max_prob`, `trim_min_prob`); backward-compat (no filter section = no-op).
- [x] 5.1.2 Create `configs/pl_gen_v3.yaml` — T2 source + Sydorskyi-canonical 0.5/0.1 filters. Output: `shared/pseudo-labels/birdclef-2026/v3-distilled-sed-kd-5fold-trim01-discard05/`.
- [x] 5.1.3 Create `configs/a2_kd_v3.yaml` — a2-kd canonical Antoine cell pointed at v3 parquet.
- [x] 5.1.4 Re-issue v3 PL parquet (cpu-small sbatch, job 9227102, walltime 46s). 33% chunk-discard rate (lower than Sydorskyi's ~83% — cause: our T=0.6 + P^1.5 sharpen + max-across-folds pre-bias). Trim filter zeroed 98% of cells.
- [x] 5.1.5 Pre-flight v3 parquet — 5/7 PASS; 2 FAILs in Check 5 are stale assertions written for unfiltered parquets (chunks dropped by discard filter; no actual leakage risk). Deferred preflight patch noted in `diagnostic-tests.md § Deferred follow-ups`.
- [ ] 5.1.6 Train a2-kd-v3 (sbatch, ~6.5h on rtx6000).
- [ ] 5.1.7 D.5 e2e validation (~5 min).
- [ ] 5.1.8 Export ONNX + Kaggle upload + LB submit.
- [ ] 5.1.9 Read LB. Decision matrix:
  - LB > 0.913 → trim+discard explain REJECT; scale to other cells (a3-kd-v3, a4-kd-v3).
  - LB 0.906-0.913 → modest lift; run Test 2 (focal × PL fusion).
  - LB 0.886-0.906 → filters alone don't help; run v4 (Sydorskyi-faithful PL gen) AND scaffold Test 2 in parallel.
  - LB < 0.886 → filters hurt; investigate parquet directly before continuing.

### Test 1 fallback — v4 (Sydorskyi-faithful PL generation; LANDED CODE, conditional)

- [x] 5.1.10 Extend `generate_pseudo_labels.py` to honor `fold_strategy: "mean"` in addition to `"max"` (was hardcoded). `temperature_scale` and `power_transform` are already no-ops at neutral values 1.0.
- [x] 5.1.11 Create `configs/pl_gen_v4.yaml` — fold_strategy=mean + T=1.0 + P^1.0 + same 0.5/0.1 filters. Output: `shared/pseudo-labels/birdclef-2026/v4-distilled-sed-kd-5fold-meanfolds-nosharpen-trim01-discard05/`.
- [x] 5.1.12 Create `configs/a2_kd_v4.yaml` — a2-kd canonical Antoine cell pointed at v4 parquet.
- [ ] 5.1.13 Run only if v3 (a2-kd-v3) lands LB ≤ 0.906. Re-issue v4 parquet; train + D.5 + LB.
- [ ] 5.1.14 Expected v4 discard rate ~80% (closer to Sydorskyi's 83%). Surviving pool ~20-25K chunks.

### Test 2 — focal × PL MixUp fusion (CONDITIONAL on Test 1)

Implements Sydorskyi's "cleverest piece" — focal (hard) × PL (soft) audio mix at 50/50 with sum-clip targets. Adds Path D to `FocalDS.__getitem__`.

- [ ] 5.2.1 Add `PLDataPool` + `focal_pl_mixup_prob` to FocalDS.
- [ ] 5.2.2 Wire into DataModule under new `sampler="focal_fusion"` option.
- [ ] 5.2.3 Unit tests + smoke test (fast-dev-run).
- [ ] 5.2.4 Create `configs/a2_kd_v3_fusion.yaml`.
- [ ] 5.2.5 Train + D.5 + LB.
- [ ] 5.2.6 Decision matrix per `diagnostic-tests.md` § Test 2.

### Test 3 — add focal loss (CONDITIONAL on Tests 1, 2 inconclusive)

Adds `sigmoid_focal_loss(α=0.25, γ=2.0)` to BCE branch.

- [ ] 5.3.1 Modify `_dual_supervision_loss` with optional `focal_loss_weight`.
- [ ] 5.3.2 Train `a2_kd_v3_fusion_focal.yaml`.
- [ ] 5.3.3 LB read; pivot decision.

### Pivot trigger

If all 3 tests fail (LB < 0.906): pivot to Sydorskyi-yoink (Path A). Document negative result. Don't run remaining exp 033 cells (a3-kd, a3-v5j Day 1, all Day 2).

---

## Phase 5 closure (2026-05-27 EOD)

**Decision (2026-05-27 EOD)**: stop running PL experiments after a2_kd_v3. v3 is the final data point on exp 033. Whatever its LB result, **the next major effort goes to Sydorskyi-yoink (exp 034)** — not Tests 2 or 3.

Rationale (captured in conversation with Claude on 2026-05-27 PM):
- Even my best-case prior on v3 (CONFIRM > 0.915) lands at ~0.92, not 0.94. PL is a finishing move on a strong base, not a way to leapfrog the 0.03 LB gap to 0.94.
- The disclosed 0.94+ paths all rely on a DATA move (XC pretrain or Sydorskyi backbone), not a recipe move. PL diagnostics are recipe moves.
- We have real code (`github.com/VSydorskyy/BirdCLEF_2025_2nd_place`) + 2 independent disclosures at LB 0.941 (Ali Ozan, Jack) for the Sydorskyi-yoink path.
- Continuing to fix all 5 PL divergences would take 3-4 days, the same as a yoink pivot, with lower ceiling.

**Phase 5 sub-status**:
- [x] 5.1.1-5.1.5 Test 1 v3 scaffold + parquet generation + pre-flight: DONE
- [ ] 5.1.6 a2-kd-v3 training: launched 2026-05-27 EOD; expected complete ~05:30 2026-05-28
- [ ] 5.1.7 D.5 e2e validation on a2-kd-v3: TBD after training
- [ ] 5.1.8 LB submit for a2-kd-v3: TBD after D.5
- [ ] 5.1.9 Read LB → record in results.md (NOT trigger Test 2/3)
- [ ] 5.1.10-14 v4 scaffold: COMPLETE but DEFERRED. Not running. Available if deep research surprises us.
- [ ] 5.2 Test 2 (focal × PL fusion): CANCELLED (effort goes to exp 034 instead)
- [ ] 5.3 Test 3 (focal loss): CANCELLED

---

## Phase 6 — Sydorskyi-yoink pivot (planned for 2026-05-28+, NOT scaffolded yet)

Will live in a new experiment directory: `experiments/034-sydorskyi-yoink/` (or similar — to be created tomorrow).

**Trigger**: combination of (a) a2_kd_v3 LB result and (b) deep research output at `docs/references/deep-research/2026-05-27-pivot-assessment.md`. Decision matrix in `CONTEXT.md § 2`.

**Default plan** (assuming research confirms yoink): download Sydorskyi 2025 backbone, attach own head (attention dim 1028, melspec match), fine-tune on 2026 data. Estimated 3-4 days.

**Closeout for exp 033** (after a2_kd_v3 LB lands tomorrow):
- C.5 Update `results.md` with v3 row
- C.6 Update `MODEL-RUNS.md` with v3 entry
- C.7 PR-ready state — exp 033 done regardless of v3 outcome; open PR with REJECT (+ v3 follow-up note) verdict
