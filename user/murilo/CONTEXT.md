# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-24 (Phase 1 LB submitted + REJECTED — catastrophic
collapse to LB 0.522 / 0.513 on control + KD vs target 0.910 and baseline
0.897. Strong leading hypothesis: training/inference mel-pipeline mismatch.
Next session = Phase D comprehensive audit. Phase 2 and Phase 3 BLOCKED.)

**Competition deadline**: 2026-06-03 (~10 days from this update).

---

## Pick up here (next session) — UPDATED 2026-05-24 EOD

Phase 1 closed REJECTED. The verdict is on the implementation, NOT the
hypothesis — Tucker's recipe is not falsified. Our port collapsed both
control + KD to LB ~0.52 (vs target 0.910, vs baseline 0.897).

**Phase D comprehensive audit is the only path forward.** Phase 2 (5-fold)
and Phase 3 (KD diagnosis) are blocked until Phase D + retrain CONFIRM.

### Phase 1 LB summary

| Run | LB | Δ paired ctrl | Δ vs v5j_control_last 0.897 | Δ vs Tucker target 0.910 |
|---|---:|---:|---:|---:|
| distilled_sed_control (last.ckpt, ep 24) | **0.522** | — | **−0.375** | **−0.388** |
| distilled_sed_kd (last.ckpt, ep 24) | **0.513** | **−0.009** | **−0.384** | **−0.397** |

Tucker macro AUC on val fold 0 at control ep 6 = 0.889 (matches Tucker's
~0.91 to 4 dp). Model IS learning on the training distribution. LB collapse
means training distribution ≠ inference distribution.

### Strong leading hypothesis: mel-pipeline mismatch

- **Tucker** (verified against vendored `bc2026-distilled-sed.py` lines 400–407, 947–949, 1240–1246): `power_mel → AmplitudeToDB(top_db=80) → per-sample (mean, std) normalize`. **One** normalization pass. Identical at training and inference.
- **Our training**: `NNAudio(norm_type="amplitude_to_db")` does `power_to_db → z-score → min-max [0,1]`, THEN module's `per_sample_mel_norm` does another z-score. **Three** normalization passes.
- **Our inference** (new `compute_mel_tucker` in submission.py): single z-score after AmplitudeToDB — **matches Tucker, but NOT our training distribution**.

Model trained on triple-normalized features sees single-normalized features
at inference → near-random output. Single bug sufficient to explain the LB
collapse.

### Phase D — comprehensive audit checklist

Full breakdown in `experiments/032-tucker-distilled-sed/tasks.md` § Phase D.
Short form:

1. **D.1**: `kaggle kernels pull tuckerarrants/bc2026-distilled-sed`; diff against vendored copy.
2. **D.2**: Side-by-side numerical test of our `_compute_mel` vs Tucker's `MelSpecTransform` + per-sample norm. Same waveform → same mel output? Write `scripts/diag_mel_pipeline.py`.
3. **D.3**: If mismatch confirmed: add `norm_type: "tucker_amp2db"` to NNAudioMelSpectrogram that does ONLY `10*log10 + top_db_clamp`, letting `_compute_mel`'s existing per-sample z-score finish the pipeline. OR bypass NNAudio at training and use torchaudio MelSpec + AmplitudeToDB directly (matches Tucker byte-for-byte). Add unit test.
4. **D.4**: Audit the lower-likelihood candidates only if D.3 doesn't recover LB: head shape, NNAudio vs torchaudio drift, audio loading, ONNX wrapper, species mapping.
5. **D.5**: Add `scripts/validate_end_to_end.py` — load ckpt, run kernel inference path on `train_soundscapes/`, compute macro AUROC, compare to offline val. Make this pre-flight before EVERY LB submission going forward.
6. **D.6**: Retrain paired control + KD with the fixed mel. Re-export, re-submit. Phase 2 / Phase 3 unblocked only when D.5 passes AND retrained control LB ≥ 0.900.

### Methodology lesson for the codebase

**Offline val/cmap/auroc cannot detect a training-inference distribution
mismatch** because the offline val uses the training pipeline. End-to-end
validation against soundscape labels using the inference kernel's full path
must be a pre-flight check, not an after-the-fact diagnostic.

This is the single biggest methodology gap exposed by exp 032.

### Open items snapshot (carried forward)

- KD ranker (`sbatch/rank_kd_only.sh`) — submit if the per-epoch KD breakdown is needed for results.md. Not blocking Phase D.
- The Kaggle kernels for distilled-sed-{control,kd} are already pushed; keep them for the post-Phase-D re-submission so we can compare same-kernel results against the LB 0.522 / 0.513 baseline.
- 19+ commits on `mg/exp032-tucker-distilled-sed-replication` branch. PR #51 stays draft until Phase D + retrain CONFIRMs/PARTIALs.

---

## Earlier session content (pre-LB submission, preserved for reference)

The original "Pick up here" content for the LB submission is now obsolete
(LB submitted, REJECT verdict). What follows below is preserved because
the val-metric bug context + Phase 0 inventory is still load-bearing for
the Phase D audit.

---

## Critical mid-session finding: val-metric bug + fix

**Symptom (caught while runs were active)**: W&B showed `val/cmap` ~0.07–0.11
across all epochs of both runs. Tucker's published trajectory at the same
point is ~0.62 (ep 1) → 0.93 (ep 17). Looked like a major regression.

**Diagnostic** (`scripts/diag_val_auroc.py`, both synthetic + real-data
verification on epoch 6 of the live control checkpoint):

```
Val fold class coverage (real, control epoch 6):
  Classes with 0 positives:     205   ← contributes 0.0 to torchmetrics macro
  Classes evaluable for AUROC:   29

Tucker's compute_macro_auc (skips 0-positive classes):  0.8891   ← TRUE model
torchmetrics MultilabelAUROC(average='macro'):          0.1102   ← what W&B shows

Arithmetic check: (29 × 0.8891 + 205 × 0.0) / 234 = 0.110  ✓
```

**Cause**: `torchmetrics 1.9 MultilabelAUROC(average='macro')` returns 0.0
(not NaN) for classes with no positives and **includes them in the macro
denominator**. Tucker's `compute_macro_auc` (notebook lines 369–384)
explicitly skips classes with `col.sum()==0 OR col.sum()==len(col)`.
`MultilabelAveragePrecision` has the same flaw, so `val/cmap` (the
ModelCheckpoint monitor) was equally broken — best-by-`val/cmap` selected
the wrong epoch.

**Fix** (commit `8d64be8`): New `src/birdclef/metrics/tucker_macro.py` with
`TuckerMacroAUC` + `TuckerMacroAP` (drop-in torchmetrics.Metric subclasses
that port Tucker's skip-empty logic). Plugged into `DistilledSEDFinetuneModule`
for train/val/test cmap+auroc. Added `val/n_evaluable` log for transparency.
11 new tests, **151/151 total tests pass**.

**Implication for next session**: The currently-running jobs (9088255, 9088258)
were launched against the OLD broken metric. Their in-training ModelCheckpoint
"best" symlink is wrong; the in-W&B `val/auroc` trajectory shows the broken
~0.11 curve. **All 25 epochs ARE saved** because `save_top_k=-1`, so we re-rank
post-hoc with `scripts/score_all_checkpoints.py` and pick the actual best
epoch for LB submission. **No need to cancel + resubmit** — the LB outcome is
identical because the same set of saved weights is being evaluated.

**Confidence the model is actually fine**: at epoch 6 (control), Tucker's
metric reads 0.889 — tracking ~exactly with Tucker's published 0.91 at the
same epoch. Recipe is working.

---

## Phase 1 status — IN PROGRESS

### Task 1.1: fast-dev-run on SLURM — DONE (PASS)

Both fast-dev-runs completed cleanly in jobs `9087357` (control) and
`9087739` (KD). All sanity-check conditions met:
- COMPLETED 0:0 exit code; 5 train + 5 val batches; no NaN/Inf
- All logged config strings matched audit (EffNetEncoder 1280-d post_head,
  AttHeadLogitSpace, SpecAugment 10/10/1/2, MixSamp `per_src=[58, 6]`,
  fold construction `739/792 SS windows matched ... 53 all-zero`)
- Control logs only `train/loss`; KD logs all three (`cls`, `dist`, total)
- KD `dist_loss=0.020` at step 5 (random init); on track for Tucker's
  0.003 floor by ep 25

### Task 1.2: full 25-ep paired training — IN PROGRESS

Jobs `9088255` (control) + `9088258` (KD) launched ~midnight EDT. Both on
`gpu-rtx6000`, distinct nodes (`atl1-1-03-006-17-0` control,
`atl1-1-03-006-21-0` KD) — ~0.007 LB cross-node variance per exp 031.

**Status at session end (~00:50 EDT)**:
- Control: ~6+ epochs in, `val/cmap=0.075` by BROKEN metric (= ~0.89 by
  Tucker's correct metric per the diagnostic)
- KD: ~4+ epochs in, `val/cmap=0.085` by BROKEN metric
- Both checkpointing every epoch (save_top_k=-1)
- ETA: ~03:30 EDT for both to hit ep 25

### Task 1.3: ONNX export + Kaggle pipeline — NOT STARTED

Pending task 1.2 + post-hoc ranking. Audit-flagged bugs to fix in our port:
- Mel-size mismatch: use `N_MELS=256` consistently OR mark mel/time as
  dynamic axes (audit § 7.2)
- Submission inference: logit-space blend per-fold → 5-tap Gaussian
  smooth → single sigmoid at end (audit § 7.3)

### Task 1.4: LB submission + decision matrix — NOT STARTED

---

## Today's session (2026-05-24) — outside Phase 1 progression

### 1. Naming rename: `tucker_*` → `distilled_sed_*`

Decoupled run/artifact identifiers from author name. Tucker stays as
content/provenance marker (exp dir name, audit notes, code comments) but
is dropped from anything that leaves the experiment dir (Kaggle slugs,
W&B run names, LB attribution).

- `configs/tucker_control.yaml` → `configs/distilled_sed_control.yaml`
- `configs/tucker_kd.yaml` → `configs/distilled_sed_kd.yaml`
- `run_name: 2026-exp032-tucker-{control,kd}` → `2026-exp032-distilled-sed-{control,kd}`
- `#SBATCH --job-name=exp032-tucker` → `exp032-dsed`
- PR #51 title: `Exp 032: Tucker distilled-SED full replication` →
  `Exp 032: Distilled-SED full replication (port of Tucker Arrants's notebook)`
- New "Provenance" section at the top of `proposal.md` documenting source
- Commit `3dd71f1`

Kept (provenance markers, intentional):
- Exp directory name `032-tucker-distilled-sed/`
- Feature branch `mg/exp032-tucker-distilled-sed-replication`
- Audit notes `2026-05-09-tucker-distilled-sed-code-audit.md`, `2026-05-23-tucker-dataset-and-inference-audit.md`
- Test names like `test_tucker_mixup`, `test_*_match_tucker_cell_*`
- Code comments quoting Tucker's value choices

### 2. matplotlib added to deps for gpu-check .png artifacts

`gpu_check` `.txt` worked but `.png` failed with `ModuleNotFoundError: No
module named 'matplotlib'`. Added `matplotlib>=3.7.0` to
`pyproject.toml`. Commit `030d59b`.

### 3. gpu-check artifacts (control + KD)

Both runs profiled (1 epoch, no W&B). Key finding from control's `.txt`:

```
Data Loading %            83.4%      86.0%      89.7%
Throughput           35 samples/s 49 samples/s 58 samples/s
GPU Utilization           10.0%      52.2%      85.0%
Memory %                   1.2%       1.5%       1.9%
Recommendations:
  Memory usage is 1.5% of 17 GB → Batch size could be ~64x larger
  Data loading is 86.0% of step time → Consider more num_workers or RAM caching
```

**Severe data-loading bottleneck** at `num_workers=0` (Tucker's default,
inherited as-is). KD's profile is less spiky than control's because the
Perch ONNX teacher forward fills idle data-wait windows with extra GPU
work. **Phase 2 easy win**: bump to `num_workers=4` or `7` for the 5-fold
sweep — significantly faster wall time, recipe semantics unchanged.

Artifacts at `outputs/gpu_check/2026-exp032-distilled-sed-{control,kd}_<jobid>.{txt,png}`.

### 4. Val-metric bug fix (covered above in detail)

Commit `8d64be8`: 1088 insertions, 12 deletions.
- New `src/birdclef/metrics/{__init__.py,tucker_macro.py}` (`TuckerMacroAUC` + `TuckerMacroAP`)
- New `user/murilo/tests/032-tucker-distilled-sed/test_tucker_macro_metric.py` (11 tests)
- New `scripts/diag_val_auroc.py` (synthetic + real-data diagnostic)
- New `scripts/score_all_checkpoints.py` (post-hoc ranker)
- Patched `distilled_sed_finetune.py` (swap broken metrics, add `val/n_evaluable` log)

---

## Phase 0 status — DONE (all 9 tasks, 151/151 tests pass)

Phase 0 unchanged from yesterday; 11 metric tests added today bring the
total from 140 → 151.

| # | Task | Commit | Tests | Output files |
|---|---|---|---:|---|
| 0.1 | AttHeadLogitSpace + GeMFreqPool | `2f624b3` | 14 | `src/birdclef/heads/sed_head.py` |
| 0.2 | EfficientNetEncoder `use_post_head=True` | `5341c58` | 11 | `src/birdclef/encoders/efficientnet.py` |
| 0.3 | DistilledSEDFinetuneModule (KD + dual-supervision + schedule) | `394279e` | 26 | `src/birdclef/lightning/distilled_sed_finetune.py` |
| 0.4 | FocalDS + ScDS + apply_aug (3-path mixup) | `72cfe23` | 18 | `src/birdclef/data/distilled_sed_dataset.py` |
| 0.5 | SpecAugment (1 freq×10 + 2 time×10) | `5bae24d` | 12 | `src/birdclef/preprocessing/spec_augment.py` |
| 0.6 | Schedule + clip-grad-norm verification | `492b245` | (in 0.3 tests) | (verification in `test_dual_supervision_loss.py`) |
| 0.7 | MixSamp batch sampler (90/10) | `00b7edc` | 14 | (added to `distilled_sed_dataset.py`) |
| 0.8 | Fold construction + DistilledSEDDataModule | `74bfc48` | 20 | `src/birdclef/data/distilled_sed_datamodule.py` |
| 0.9 | Paired configs | `d6cb1e9` | 25 | `experiments/032-tucker-distilled-sed/configs/` |
| (today) | TuckerMacroAUC + TuckerMacroAP metric | `8d64be8` | 11 | `src/birdclef/metrics/tucker_macro.py` |

### Tucker-recipe audit findings (load-bearing)

1. **KD = embedding-space MSE distillation with `feat.detach()`** (Perch category #3)
2. **AttHead aggregates in LOGIT space** (no sigmoid roundtrip)
3. **Dual-supervision at training + inference** (0.5·BCE clip + 0.5·BCE frame_max; 0.5·σ(clip) + 0.5·σ(frame_max))
4. **Schedule**: AdamW + LinearLR(2 ep, start_factor=1/25) → Cosine(23 ep, eta_min=1e-6), per-step
5. **MixSamp 90/10 focal/SS batch composition**
6. **5-fold (StratifiedKFold focal + GroupKFold SS); all 66 labeled SS files participate**
7. **Inference**: logit-space blend across folds + 5-tap Gaussian smoothing across 12 windows/file, single sigmoid at end (submission-side, NOT in LightningModule)
8. **Macro AUC must skip 0-positive classes** (TODAY'S FIX — Tucker's `compute_macro_auc`)

---

## Key reference docs (load-bearing for Phase 1)

| Doc | What it contains |
|---|---|
| `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Forward pass, AttHead aggregation, loss formula, mixup, optimizer, mel pipeline |
| `analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md` | Dataset internals, fold construction, MixSamp, all-zero SS labels (53/792), checkpoint selection, inference, 5-tap smoothing, notebook bugs to NOT inherit |
| `notes/2026-05-22-strategic-direction-tracks-A-B.md` | Strategic context: Track A (Tucker primary) → Track B (own-teacher PL) |
| `notes/kaggle-discussion-tucker-distilled-sed-baseline.md` | Tucker's discussion thread + KD lift quantification (+0.02 LB) |
| `experiments/031-perch-kd-distillation/results.md` | Exp 031 REJECT verdict (closed); post-mortem |
| `experiments/032-tucker-distilled-sed/proposal.md` | Hypothesis, paired-control success matrix, full setup table, + Provenance section (today) |
| `experiments/032-tucker-distilled-sed/tasks.md` | Phase 0 done + Phase 1+ checklist |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` | Tucker's verbatim notebook |
| `vendor/.../bc2026-distilled-sed.py` | .py extraction (gitignored, regenerable) |

---

## Best LB (unchanged — no LB submissions today)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| v5j single (exp 019) | **0.906** | Best single SED (real) |
| effnetv2b0_sydorskyi (exp 030) | 0.901 | Best non-v5j single |
| v5j_control_v2 / v5j_control_last (exp 030/031) | 0.896 / 0.897 | HEAD-code retrains |
| v5j_kd_last (exp 031, ep 19) | 0.891 | HEAD-code + KD; −0.006 vs paired control |
| **v5j_kd (exp 031, ep 17)** | **0.890** | REJECTED on v5j+HGNet+LSEHead |

**Exp 032 single-fold target**: 0.910–0.917 (Tucker's published 0.917 is
5-fold + PP; single-fold should land just below).

---

## Key patterns and constraints

### `save_top_k: -1` policy
Both exp 032 configs save every epoch (per `feedback_save_top_k_does_not_preserve_last`).
Critical today because the in-training monitor was broken — post-hoc rank
all 25 saved epochs.

### Paired within-run controls
Only reliable methodology under ~0.007 LB cross-node variance. Submit
distilled_sed_control + distilled_sed_kd back-to-back; read paired Δ.

### Tucker-style macro AUC (TODAY)
TuckerMacroAUC/AP skip 0-positive classes. `MultilabelAUROC(average='macro')`
from torchmetrics 1.9 silently counts them as 0.0 — produces misleading
0.11 readings on Tucker's small SS val pool.

### num_workers=0 → severe data bottleneck (TODAY)
gpu-check showed 86% data time / 52% GPU util / 1.5% memory. Tucker's
default is fine on Colab G4 but suboptimal on PACE. Bump to 4 or 7 in
Phase 2 (recipe-neutral, big throughput win).

### Reproducibility: separate modules
Exp 032 created new modules (`DistilledSEDFinetuneModule`,
`DistilledSEDDataModule`, `distilled_sed_dataset.py`, `tucker_macro.py`)
rather than modifying shared code, keeping exp 030/031 paths bit-identical.

### Pinned environment
`experiments/032-tucker-distilled-sed/pyproject.toml` is source of truth.
Tight pins on torch 2.5.1, torchaudio 2.5.1, pytorch-lightning 2.6.1,
torchmetrics 1.9.0, timm 1.0.26, huggingface-hub==1.9.2, nnAudio 0.3.4,
onnxruntime-gpu==1.26.0, matplotlib>=3.7.0 (added today).

---

## Branch / git state

- **Current branch**: `mg/exp032-tucker-distilled-sed-replication`.
  **19 commits ahead** of main, all pushed.
- **PR**: [#51](https://github.com/dsgt-arc/birdclef-2026/pull/51)
  (draft) — `Exp 032: Distilled-SED full replication (port of Tucker Arrants's notebook)`
- **Most recent commits (this session)**:
  - `8d64be8` — Fix val/auroc + val/cmap dragged to ~0.11 by 0-positive-class bug
  - `030d59b` — Add matplotlib to exp 032 deps so --gpu-check produces .png plots
  - `3dd71f1` — Rename exp 032 run/config identifiers from tucker_* to distilled_sed_*
  - `9be1b00` — Update CONTEXT.md for end of 2026-05-23 session
  - `f463863` — Fix exp 032 sbatch banner

---

## Compressed timeline (today, 2026-05-24)

| Stage | What | Outcome |
|---|---|---|
| Session start | Loaded CONTEXT.md + audit notes + Tucker notebook + proposal | Confirmed Phase 0 done, ready for Phase 1 |
| Pre-flight | Verified 140 tests pass + ONNX file present (`vendor/.../perch_v2_no_dft.onnx`, 395M) | ✅ |
| Task 1.1 | Submitted fast-dev-runs (jobs 9087357 + 9087739); both PASSED all sanity conditions | ✅ moved to 1.2 |
| Rename | tucker_* → distilled_sed_* across configs/runs/sbatch/PR title; provenance kept in exp dir + audits + comments | commit `3dd71f1`; PR title updated via gh api |
| gpu-check round 1 | Submitted gpu-check; `.txt` ok but `.png` failed (matplotlib missing) | Diagnosed |
| matplotlib fix | Added matplotlib to pyproject.toml | commit `030d59b` |
| Task 1.2 | Submitted full 25-ep paired training (jobs 9088255 + 9088258) | Running through ~03:30 EDT |
| gpu-check round 2 | Re-ran; both `.txt` + `.png` artifacts produced. Headline: 86% data time, 52% GPU util, 1.5% memory — severe data bottleneck | num_workers bump as Phase 2 lever |
| Val-metric anomaly | User noticed val/auroc plateauing at 0.11 in W&B vs Tucker's published 0.62+ trajectory | Started investigation |
| Diagnosis | Pulled Tucker's notebook source; found `compute_macro_auc` skips 0-positive classes (lines 369-384) | Hypothesis |
| Synthetic verification | `diag_val_auroc.py` synthetic run: Tucker 0.93 vs torchmetrics 0.65 with 70/234 zero-positive classes | Confirmed mechanism |
| Real-data verification | Same script with epoch-6 control ckpt: Tucker 0.889 vs torchmetrics 0.110, 205 zero-positive classes. Matches W&B 0.11 exactly | Confirmed on real data |
| Fix | New `TuckerMacroAUC` + `TuckerMacroAP` (drop-in torchmetrics subclasses); swapped into module; 11 new tests | 151/151 pass; commit `8d64be8` |
| Ranker | `scripts/score_all_checkpoints.py` for post-hoc ranking — ready for next session | Will pick true best epoch from all 25 |
| Session end | CONTEXT.md updated; runs still going (~14 ep each at 00:50 EDT) | Tomorrow: rank + LB submit |

---

## File index for active work

| File | Purpose |
|---|---|
| **Modules (shared lib)** | |
| `user/murilo/src/birdclef/heads/sed_head.py` | + AttHeadLogitSpace + GeMFreqPool |
| `user/murilo/src/birdclef/encoders/efficientnet.py` | + `use_post_head` flag |
| `user/murilo/src/birdclef/lightning/distilled_sed_finetune.py` | DistilledSEDFinetuneModule (uses TuckerMacroAUC/AP since today) |
| `user/murilo/src/birdclef/data/distilled_sed_dataset.py` | FocalDS + ScDS + apply_aug + MixSamp |
| `user/murilo/src/birdclef/data/distilled_sed_datamodule.py` | DistilledSEDDataModule + fold-construction helpers |
| `user/murilo/src/birdclef/preprocessing/spec_augment.py` | SpecAugment (Tucker's params) |
| `user/murilo/src/birdclef/metrics/tucker_macro.py` | **NEW** — TuckerMacroAUC + TuckerMacroAP (skip 0-positive classes) |
| **Experiment dir** | |
| `user/murilo/experiments/032-tucker-distilled-sed/proposal.md` | + Provenance section (today) |
| `user/murilo/experiments/032-tucker-distilled-sed/tasks.md` | Phase 0 done; Phase 1+ checklist |
| `user/murilo/experiments/032-tucker-distilled-sed/results.md` | Placeholder; fill as Phase 1 lands |
| `user/murilo/experiments/032-tucker-distilled-sed/configs/distilled_sed_control.yaml` | **RENAMED** — Full recipe, KD OFF |
| `user/murilo/experiments/032-tucker-distilled-sed/configs/distilled_sed_kd.yaml` | **RENAMED** — Full recipe, KD ON |
| `user/murilo/experiments/032-tucker-distilled-sed/train.py` | Wired to DistilledSEDDataModule + DistilledSEDFinetuneModule |
| `user/murilo/experiments/032-tucker-distilled-sed/sbatch/run.sh` + `_job.sh` | SLURM launchers |
| `user/murilo/experiments/032-tucker-distilled-sed/pyproject.toml` | + matplotlib (today) |
| `user/murilo/experiments/032-tucker-distilled-sed/scripts/diag_val_auroc.py` | **NEW** — val-metric diagnostic |
| `user/murilo/experiments/032-tucker-distilled-sed/scripts/score_all_checkpoints.py` | **NEW** — post-hoc ckpt ranker |
| `user/murilo/experiments/032-tucker-distilled-sed/outputs/gpu_check/` | **NEW** — gpu-check `.txt` + `.png` artifacts |
| **Tests (151 total — was 140)** | |
| `user/murilo/tests/032-tucker-distilled-sed/test_att_head_logit_space.py` | task 0.1 (14) |
| `user/murilo/tests/032-tucker-distilled-sed/test_efficientnet_encoder_post_head.py` | task 0.2 (11) |
| `user/murilo/tests/032-tucker-distilled-sed/test_dual_supervision_loss.py` | tasks 0.3 + 0.6 (26) |
| `user/murilo/tests/032-tucker-distilled-sed/test_tucker_mixup.py` | task 0.4 (18) |
| `user/murilo/tests/032-tucker-distilled-sed/test_specaugment.py` | task 0.5 (12) |
| `user/murilo/tests/032-tucker-distilled-sed/test_mixsamp_sampler.py` | task 0.7 (14) |
| `user/murilo/tests/032-tucker-distilled-sed/test_fold_construction.py` | task 0.8 (20) |
| `user/murilo/tests/032-tucker-distilled-sed/test_configs.py` | task 0.9 (25) |
| `user/murilo/tests/032-tucker-distilled-sed/test_tucker_macro_metric.py` | **NEW** — TuckerMacroAUC/AP (11) |
| **Audit + reference notes** | |
| `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Forward + loss audit |
| `user/murilo/analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md` | Dataset + sampler + inference audit |
| `user/murilo/notes/2026-05-22-strategic-direction-tracks-A-B.md` | Strategic direction |
| `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` | Tucker's discussion thread |
| **Checkpoints (today)** | |
| `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-control/best-epochepoch=*.ckpt` | All 25 epochs from job 9088255 (in progress) |
| `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/best-epochepoch=*.ckpt` | All 25 epochs from job 9088258 (in progress) |
| **Data + model paths** | |
| `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/` | Competition data |
| `vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx` | Perch v2 teacher ONNX |

---

## Pending decisions / open items for next session

1. **Verify both training jobs completed cleanly** (9088255 + 9088258). If failed, check log tails + diagnose.
2. **Run `score_all_checkpoints.py` for both ckpt dirs** to pick the best epochs by Tucker's correct metric. Update results.md with the per-epoch table.
3. **ONNX export bug fix** (task 1.3.1) — use `N_MELS=256` consistently OR mark mel/time as dynamic axes (audit § 7.2).
4. **Submission inference in logit space** (task 1.3.3) — `submission.py` needs per-window `0.5·clip + 0.5·frame_max` in LOGIT space, then 5-tap Gaussian smooth across 12 windows-per-file, then single sigmoid at end (audit § 7.3).
5. **LB submission + decision matrix** (task 1.4).
6. **Phase 2 num_workers bump** (4 or 7) for the 5-fold sweep if Phase 1 CONFIRMs/PARTIALs.
7. **Site/hour priors** (Phase 2 stretch) — port from EoS.6 stack if needed.
8. **5-fold extension** (Phase 2) — sweep folds {1, 2, 3, 4} after fold 0 lands.
9. **Optional cleanup**: relaunch training with the fixed metric so W&B trajectory shows the real ~0.85+ curve instead of broken ~0.11. Stylistic; LB outcome would be identical. Default = skip.

---

## How to resume after disconnect

1. Read this file top-to-bottom.
2. Skim the two audit notes (`2026-05-09-...` + `2026-05-23-...`).
3. Check git state: `git log --oneline -10` should show `8d64be8` → `030d59b` → `3dd71f1` → `9be1b00` → … most recent first.
4. Verify tests still pass: `~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/032-tucker-distilled-sed/ -q`. Expect **151 pass**.
5. Check `squeue` for running jobs; verify both 9088255 + 9088258 completed.
6. Run `score_all_checkpoints.py` for control + KD ckpt dirs.
7. Submit top-ranked checkpoints to LB (task 1.3).
8. Apply decision matrix (proposal.md § Falsifiable).

**Slack / Kaggle / Wandb credentials**: in `.env` (gitignored). Loaded by
`train.py` from `user/murilo/.env`. If W&B login fails on SLURM,
`WANDB_MODE=offline` or `--fast-dev-run` skips W&B entirely.
