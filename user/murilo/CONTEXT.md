# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-24 late evening (Phase D LB landed **CONFIRM** —
paired Δ +0.050 LB. distilled_sed_kd LB 0.906 matches v5j-original.
Phase 2 launched: 5 jobs running — kd folds 1/2/3/4 + kd_hgnet ablation,
ETA ~04:00 EDT 2026-05-25.)

**Competition deadline**: 2026-06-03 (~9 days from this update).

---

## Pick up here (next session) — UPDATED 2026-05-24 EOD

When you resume, the 5 retrain jobs should be DONE. Walk this checklist:

### 1. Verify all 5 Phase 2 jobs completed cleanly

```bash
squeue -u $USER -o "%.10i %.30j %.8T %.10M %.20R"
# Expect all 5 to be gone. If any still RUNNING, wait.

for JOB in 9158704 9158705 9158706 9159168 9158713; do
    echo "=== $JOB ==="
    LOG=$(ls user/murilo/experiments/032-tucker-distilled-sed/logs/slurm_*${JOB}.log 2>/dev/null)
    if [ -z "$LOG" ]; then echo "no log!"; continue; fi
    tail -5 "$LOG"
done
```

Expect `Job completed successfully!` + `Begin Slurm Epilog: …`. All 25 epochs
saved per `save_top_k: -1` policy.

**Running jobs as of session end (started 2026-05-24 ~23:12 EDT, ETA ~04:00):**

| Job ID | Name | Output ckpt dir |
|---|---|---|
| 9158704 | exp032-distilled_sed_kd-fold1 | `.../2026-exp032-distilled-sed-kd/fold1/` |
| 9158705 | exp032-distilled_sed_kd-fold2 | `.../2026-exp032-distilled-sed-kd/fold2/` |
| 9158706 | exp032-distilled_sed_kd-fold3 | `.../2026-exp032-distilled-sed-kd/fold3/` |
| 9159168 | exp032-distilled_sed_kd-fold4 (resubmit) | `.../2026-exp032-distilled-sed-kd/fold4/` |
| 9158713 | exp032-distilled_sed_kd_hgnet | `.../2026-exp032-distilled-sed-kd-hgnet/` |

Base ckpt dir: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/`.

The original fold 4 job 9158707 died at t=1m45s due to a `--fold` argparse
bug (choices=[0,1,2,3] inherited from a 4-fold legacy; my Phase 2 fix
removed the vestigial fold_assignments.json check but missed the argparse
constraint). Fixed in commit `224e569`; resubmitted as 9159168.

### 2. Run validate_end_to_end on each new checkpoint (pre-flight before LB)

We instituted D.5 end-to-end validation as a mandatory pre-flight after
Phase 1's catastrophic LB collapse (0.522 / 0.513 vs target 0.910).
**Every retrain must PASS validate_end_to_end before LB submission.**

```bash
cd user/murilo/experiments/032-tucker-distilled-sed

# 5-fold KD: validate folds 1-4 (fold 0 already validated in commit 787e43f)
for F in 1 2 3 4; do
  CKPT=/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/fold${F}/last.ckpt
  if [ ! -f "$CKPT" ]; then
    CKPT=$(ls -t /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/fold${F}/last*.ckpt 2>/dev/null | head -1)
  fi
  CKPT="$CKPT" CONFIG=configs/distilled_sed_kd.yaml \
    sbatch sbatch/validate_end_to_end.sh
done

# HGNet+KD: validate (no fold suffix, top-level ckpt dir)
CKPT=$(ls -t /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd-hgnet/last*.ckpt | head -1) \
CONFIG=configs/distilled_sed_kd_hgnet.yaml \
  sbatch sbatch/validate_end_to_end.sh
```

Look for `PASS ✅` in each log. Decision rule: if gap > 0.02 between
training-side `val/auroc` and end-to-end, DO NOT submit to LB until
diagnosed. (Phase D ran 0.011 + 0.004 gaps; pipeline is consistent now.)

### 3. Export ONNX + Kaggle upload + LB submit (after all PASS)

```bash
cd user/murilo/experiments/010-submission-pipeline

# 5-fold KD ensemble (fold 0 reuses Phase D's top-level last-v2.ckpt;
# folds 1-4 picked from fold{N}/last*.ckpt)
bash sbatch/run.sh export --model distilled-sed-kd-5fold

# HGNet+KD single-fold
bash sbatch/run.sh export --model distilled-sed-kd-hgnet

# After both exports complete (~10 min total):
python scripts/upload_kaggle.py --model distilled-sed-kd-5fold
python scripts/upload_kaggle.py --model distilled-sed-kd-hgnet
# Wait 2-3 min for Kaggle to process
python scripts/upload_kaggle.py --model distilled-sed-kd-5fold --push-kernel
python scripts/upload_kaggle.py --model distilled-sed-kd-hgnet --push-kernel
```

Then submit both to LB via Kaggle UI. Suggested descriptions:

**distilled-sed-kd-5fold**:
> distilled_sed_kd 5-fold ensemble (Phase 2.A) — same Tucker recipe + Perch v2 KD; folds 0-4, seed=42, last-v2.ckpt each. Inference: per-fold blended LOGITS → fold-mean in LOGIT space → 5-tap Gauss smooth → single sigmoid. Phase D single-fold = 0.906; Tucker reports 5-fold + smoothing = 0.917. Expected band: 0.910-0.920.

**distilled-sed-kd-hgnet**:
> distilled_sed_kd_hgnet single-fold (Phase 2.D ablation) — same Tucker recipe + KD as distilled-sed-kd, only backbone changed: tf_efficientnet_b0.ns_jft_in1k → hgnetv2_b0.ssld_stage2_ft_in1k (v5j's choice), use_post_head=true → false. Tests whether the -0.041 control-vs-v5j gap was backbone-driven. Phase D EffNet+KD = 0.906; this measures HGNet+KD on the same fold 0.

### 4. Apply decision matrix once LBs land

Phase 2.A 5-fold ensemble LB:
- ≥ 0.920 → Tucker target exceeded; consider site/hour priors for 0.93+
- 0.910–0.919 → matches Tucker's published 0.917; consider ensemble with exp 023 Perch+SED
- 0.900–0.909 → ensemble lift is modest; investigate (probably normal v5j-style 4→5 fold lift)
- < 0.900 → unexpected; single-fold dominated by fold 0, ensemble pulling down

Phase 2.D HGNet+KD LB:
- > 0.910 → backbone was real bottleneck; pivot to HGNet for any further runs
- 0.900–0.910 → marginal backbone effect; EffNet was ~fine
- < 0.900 → EffNet > HGNet for this recipe; stay on EffNet

### 5. Update results.md + MODEL-RUNS.md + this CONTEXT.md with the verdict.

### 6. Branch / git state

- Branch `mg/exp032-tucker-distilled-sed-replication`, **~30 commits ahead of main**, all pushed.
- PR #51 (draft) — title `Exp 032: Distilled-SED full replication (port of Tucker Arrants's notebook)`.
- Most recent commits:
  - `224e569` — Fix --fold argparse to accept fold 4
  - `5cd047f` — Phase 2 scaffolding (5-fold + HGNet)
  - `db8660c` — Phase D LB CONFIRM (+0.050)
  - `787e43f` — D.5/D.6.2 end-to-end validation PASS
  - `cffd725` — export funcs auto-pick newest last*.ckpt
  - `8fa429f` — D.6.1 retrain offline metrics
  - `3806b63` — D.5 validate_end_to_end script
  - `9660862` — export dispatch stub mel on mel.type
  - `b305c5e` — D.2/D.3 confirm + fix mel mismatch
  - `63122db` — Phase 1 REJECT (catastrophic)

---

## What landed today (full session walkthrough)

### Phase 1 — REJECTED (Phase 1 broken-mel LB collapse)

| Run | Phase 1 LB | Δ paired ctrl | Δ vs v5j_control_last 0.897 |
|---|---:|---:|---:|
| distilled_sed_control (last.ckpt ep 24) | **0.522** | — | **−0.375** |
| distilled_sed_kd (last.ckpt ep 24) | **0.513** | **−0.009** | **−0.384** |

Catastrophic. Both runs near-random despite offline Tucker macro AUC at
Tucker's published 0.91 band. Verdict scoped to implementation, NOT
hypothesis. Tucker recipe was NOT falsified.

### Phase D — full audit + fix + LB retest = **CONFIRM**

#### D.2 numerical diagnostic (scripts/diag_mel_pipeline.py)

Ran Tucker reference vs our broken training vs our (Tucker-faithful)
kernel on the same 5s audio window:

| Comparison | max\|d\| | Pearson r | Status |
|---|---:|---:|---|
| Tucker reference vs our kernel | 4.77e-07 | 1.000000 | ✅ kernel was already Tucker-faithful |
| Tucker reference vs our training | **3.20** | 0.9356 | ❌ training diverged 3σ |
| Our training vs our kernel | **3.20** | 0.9356 | ❌ same gap |

Diagnosis: NNAudio's `norm_type="amplitude_to_db"` does
`power_to_db → z-score → min-max [0,1]` internally, then `_compute_mel`'s
`per_sample_mel_norm=True` does another z-score. THREE normalization
passes. Tucker does ONE (z-score after AmplitudeToDB). Model trained on
triple-normalized features sees single-normalized features at inference
→ near-random.

#### D.3 fix — TuckerMelSpectrogram

New `user/murilo/src/birdclef/preprocessing/tucker_mel.py`. Verbatim port
of Tucker's `MelSpecTransform` (notebook lines 400-407): torchaudio
MelSpec + AmplitudeToDB(80), zero internal normalization. The module's
`per_sample_mel_norm=True` (unchanged) applies the single per-sample
z-score on top → matches Tucker's training pipeline to atol=1e-5.

- Configs flipped `mel.type: "nnaudio"` → `"tucker"`.
- 12 new unit tests in `test_tucker_mel.py`; all 163/163 exp 032 tests pass.
- Regression guard `test_diverges_from_nnaudio_amplitude_to_db` locks the
  fix so future refactors can't accidentally re-route through NNAudio.

Critical detail: training mel transform is now torchaudio, inference mel
is torchaudio — **they share the SAME torchaudio code paths**. Cannot
drift apart by construction.

#### D.5 end-to-end validation script

New `scripts/validate_end_to_end.py` + `sbatch/validate_end_to_end.sh`.

Loads a ckpt, runs the KERNEL inference path (kernel_load_audio_chunk →
compute_mel_tucker → encoder → AttHeadLogitSpace → per-window LOGIT-space
blend → 5-tap Gauss smooth → single sigmoid) on the same val SS fold
windows the training validation_step uses. Computes Tucker macro AUC.

If training-side `val/auroc` ≈ end-to-end (within tolerance, default 0.02),
the pipeline is consistent. Otherwise → train↔inference mismatch.

**Now mandatory pre-flight before every LB submission going forward.**

#### D.6.1 retrain (jobs 9136231 + 9136232)

Same configs (just `mel.type: "tucker"`), same data, same seed=42, fold 0:

| Run | val/auroc | val/cmap | best_val_score | walltime |
|---|---:|---:|---:|---|
| distilled_sed_control (ep 24 last-v2.ckpt) | **0.887** | 0.645 | 0.659 | 3h39m |
| distilled_sed_kd (ep 24 last-v2.ckpt) | **0.908** | 0.692 | 0.709 | 4h26m |
| **paired Δ** | **+0.021** | **+0.047** | +0.050 | — |

Both in Tucker's published single-fold band (~0.91).

W&B: `dsgt-arc/birdclef-2026-exp032/g5rblf6o` (control) + `0w29iaut` (KD).

#### D.6.2 end-to-end validation (jobs 9156567 + 9156568)

| Run | Training-side val/auroc | End-to-end val/auroc | Gap | Status |
|---|---:|---:|---:|---|
| control | 0.887 | 0.898 | 0.011 | ✅ PASS |
| kd | 0.908 | 0.912 | 0.004 | ✅ PASS |

End-to-end is slightly HIGHER than training-side because the kernel
path includes the 5-tap Gauss smoothing across windows (acts as a
temporal denoiser; +0.004 lift per Tucker's "MengYe smoothing trick").

#### D.6.3 LB submission — CONFIRM

| Run | LB | Δ paired ctrl | Δ vs v5j_control_last 0.897 | Δ vs Tucker target 0.910 |
|---|---:|---:|---:|---:|
| distilled_sed_control (Phase D last-v2.ckpt) | **0.856** | — | **−0.041** | **−0.054** |
| distilled_sed_kd (Phase D last-v2.ckpt) | **0.906** | **+0.050** | **+0.009** | **−0.004** |

**Verdict: CONFIRM per proposal.md § Falsifiable** (paired Δ ≥ +0.014).

- distilled_sed_kd 0.906 ≡ v5j-original LB 0.906 (exact match) and within
  0.004 of Tucker's published single-fold target.
- Recipe alone (control) underperforms v5j by 0.041 — the **KD branch
  does all the heavy lifting** on this stack.
- KD on AttHeadLogitSpace: +0.050; KD on LSEHead (exp 031): −0.006. Same
  KD branch, +0.056 LB swing from head choice. Validates exp 031
  post-mortem hypothesis: "LSEHead was the wrong downstream consumer
  of Perch features."

### Phase 2 scaffolded + launched (commits `5cd047f` + `224e569`)

Per CONFIRM verdict, Phase 2 extends in two directions:

**Phase 2.A — 5-fold KD ensemble** (Tucker's 0.917 LB = 5-fold + smoothing):
- Skip control's 5-fold (control underperforms, not worth compute).
- Train KD on folds 1-4 (fold 0 already done at 0.906 LB).
- New `export_distilled_sed_kd_5fold` handles asymmetric layout: fold 0
  uses top-level Phase D `last-v2.ckpt`; folds 1-4 use `fold{N}/last*.ckpt`.
- New `predict_soundscape_tucker_ensemble` in submission.py: per-window
  LOGIT-space fold mean → 5-tap Gauss smooth → single sigmoid. Routed by
  `use_tucker AND is_ensemble` (kernel detects ensemble via
  `model_fold*.onnx` glob).

**Phase 2.D — HGNet+KD backbone ablation** (your hypothesis):
- New config `distilled_sed_kd_hgnet.yaml`. Only differences vs
  `distilled_sed_kd.yaml`:
  - `encoder.model_name`: `tf_efficientnet_b0.ns_jft_in1k` → `hgnetv2_b0.ssld_stage2_ft_in1k`
  - `encoder.use_post_head`: `true` → `false` (HGNet features_only → 1024-d)
  - `run_name` suffix for separate ckpt dir
- Tests whether the −0.041 control-vs-v5j gap was backbone-driven.

---

## Tucker's 0.917 vs our 0.906 — the answer

Tucker's published 0.917 LB = **5-fold ensemble + 5-tap Gaussian smoothing**,
NOT single-fold. He mentions a fold-1 example at 0.922 LB; per-fold
variance in his published trajectories is ~±0.015. Our 0.906 single-fold
(fold 0) is at the bottom of his single-fold band — plausible because
fold 0's labeled SS may be harder-to-predict than fold 1.

Expected outcome of Phase 2.A: 5-fold ensemble at LB 0.910–0.920.

Tucker also mentions 0.93+ "with PP" (private code, NOT in the public
notebook). Site/hour priors (EoS.6 style) are the prime candidate for
further lift. Reserved for Phase 2 stretch if 5-fold + HGNet land below 0.92.

---

## Key methodology lessons exposed by this experiment

1. **Offline val/cmap/auroc cannot detect a training-inference distribution
   mismatch** because the offline val uses the training pipeline. End-to-end
   validation (kernel inference path on `train_soundscapes_labels.csv`,
   compute macro AUROC, compare to offline) must be a pre-flight check, not
   an after-the-fact diagnostic. `validate_end_to_end.py` now exists for
   this purpose; **run it before every LB submission**.

2. **Library-style classes can hide pipeline composition bugs**.
   `NNAudioMelSpectrogram(norm_type="amplitude_to_db")` is one line but
   does FOUR internal operations. Whether that matches Tucker's recipe is
   not obvious from the call site. Diagnosed via numerical side-by-side.

3. **torchmetrics MultilabelAUROC(average='macro')** counts 0-positive
   classes as 0.0 AND includes them in the denominator. For a small val
   pool with many sparse classes (our case: 205/234 classes had zero
   positives), this drags the macro to ~0.11 while the "true" Tucker
   macro (skip-empty) reads ~0.89. Fixed via `TuckerMacroAUC` /
   `TuckerMacroAP` (drop-in torchmetrics.Metric subclasses).

4. **Lightning's `save_last: true` versions instead of overwriting**.
   After N runs in the same directory, you get `last.ckpt` + `last-v1.ckpt`
   + ... + `last-v{N-1}.ckpt`. Hardcoded `last.ckpt` paths pick the
   OLDEST. Use `find_last_ckpt` helper (newest mtime) instead.

5. **argparse `choices=[...]` is a hard gate** — runtime validation in the
   function body runs only after argparse passes. Fold 4 died at parse
   time because choices was [0,1,2,3] from a 4-fold-CV legacy.

---

## Best LB

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current overall best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| Tucker published 5-fold+smooth | 0.917 | Reference (replicate target) |
| **distilled-sed-kd (exp 032 Phase D, single-fold)** | **0.906** | Today's CONFIRM |
| v5j single (exp 019) | 0.906 | Best single SED (matches us) |
| v5j_control_last (exp 031) | 0.897 | HEAD-code v5j retrain |
| **distilled-sed-control (exp 032 Phase D)** | **0.856** | Recipe alone w/o KD |
| Phase 1 broken-mel (exp 032) | 0.513-0.522 | Pre-fix; obsolete |

**Exp 032 Phase 2.A 5-fold target**: 0.910–0.920 LB.

---

## Branch / git state

- **Current branch**: `mg/exp032-tucker-distilled-sed-replication`.
  ~30+ commits ahead of main, all pushed.
- **PR**: [#51](https://github.com/dsgt-arc/birdclef-2026/pull/51) (draft)
  — `Exp 032: Distilled-SED full replication (port of Tucker Arrants's notebook)`.
- Plan: PR stays draft until Phase 2 lands (then we can either merge or
  decide what to do with the rejected Phase 1 history).

---

## File index for Phase 2 (active work)

| File | Purpose |
|---|---|
| **New today (Phase D + Phase 2)** | |
| `user/murilo/src/birdclef/preprocessing/tucker_mel.py` | **D.3 FIX** — TuckerMelSpectrogram (torchaudio + AmplitudeToDB, no internal norm) |
| `user/murilo/tests/032-tucker-distilled-sed/test_tucker_mel.py` | 12 tests, all pass; regression guard included |
| `experiments/032-tucker-distilled-sed/scripts/diag_mel_pipeline.py` | **D.2** — side-by-side numerical mel diagnostic |
| `experiments/032-tucker-distilled-sed/scripts/validate_end_to_end.py` | **D.5** — pre-flight before LB |
| `experiments/032-tucker-distilled-sed/sbatch/validate_end_to_end.sh` | sbatch wrapper for D.5 |
| `experiments/032-tucker-distilled-sed/configs/distilled_sed_kd_hgnet.yaml` | **Phase 2.D** — HGNet+KD ablation config |
| `experiments/010-submission-pipeline/scripts/export_distilled_sed.py` | Tucker-faithful ONNX export; dispatches stub mel on `mel.type` |
| `experiments/010-submission-pipeline/kaggle_kernel/kernel-metadata-distilled-sed-{control,kd,kd-5fold,kd-hgnet}.json` | Kernel metadata files |
| **Modified today** | |
| `experiments/032-tucker-distilled-sed/train.py` | Removed vestigial `fold_assignments.json` check; fixed `--fold` argparse to accept 4 |
| `experiments/032-tucker-distilled-sed/configs/distilled_sed_{control,kd}.yaml` | `mel.type: "tucker"` (was nnaudio) |
| `experiments/010-submission-pipeline/kaggle_kernel/submission.py` | + `compute_mel_tucker` + `predict_soundscape_tucker[_ensemble]` + dispatch |
| `experiments/010-submission-pipeline/sbatch/export.sbatch` | + `export_distilled_sed_{control,kd,kd_5fold,kd_hgnet}` + `find_last_ckpt` auto-pick |
| `experiments/010-submission-pipeline/scripts/upload_kaggle.py` | + 4 MODELS entries for distilled-sed-* |
| **Inherited from earlier sessions** | |
| `user/murilo/src/birdclef/heads/sed_head.py` | AttHeadLogitSpace + GeMFreqPool (Phase 0.1) |
| `user/murilo/src/birdclef/encoders/efficientnet.py` | `use_post_head` flag (Phase 0.2) |
| `user/murilo/src/birdclef/lightning/distilled_sed_finetune.py` | DistilledSEDFinetuneModule (Phase 0.3); uses TuckerMacroAUC/AP |
| `user/murilo/src/birdclef/data/distilled_sed_dataset.py` | FocalDS + ScDS + apply_aug + MixSamp (Phase 0.4 + 0.7) |
| `user/murilo/src/birdclef/data/distilled_sed_datamodule.py` | DistilledSEDDataModule + fold helpers (Phase 0.8) |
| `user/murilo/src/birdclef/preprocessing/spec_augment.py` | SpecAugment (Phase 0.5) |
| `user/murilo/src/birdclef/metrics/tucker_macro.py` | TuckerMacroAUC + TuckerMacroAP (skip 0-positive classes) |
| **Checkpoints (post-Phase-D + Phase 2 retrains)** | |
| `.../032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-control/last-v2.ckpt` | Phase D control (LB 0.856) |
| `.../032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/last-v2.ckpt` | Phase D KD fold 0 (LB 0.906) |
| `.../2026-exp032-distilled-sed-kd/fold{1,2,3,4}/last*.ckpt` | Phase 2.A folds 1-4 (PENDING — running now) |
| `.../2026-exp032-distilled-sed-kd-hgnet/last*.ckpt` | Phase 2.D HGNet+KD (PENDING — running now) |

---

## Key reference docs

| Doc | What it contains |
|---|---|
| `experiments/032-tucker-distilled-sed/proposal.md` | Hypothesis, falsifiable success matrix, Provenance section |
| `experiments/032-tucker-distilled-sed/tasks.md` | Phases 0/1/D/2 checklist with all done items + pending |
| `experiments/032-tucker-distilled-sed/results.md` | Phase D LB results (CONFIRM verdict) + Phase 1 (REJECT, historical) |
| `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Forward + loss + head + mel audit |
| `analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md` | Dataset internals + folds + 5-tap smoothing + inference + 2 notebook bugs NOT to inherit |
| `notes/kaggle-discussion-tucker-distilled-sed-baseline.md` | Tucker's discussion + per-fold LB band (0.913-0.934 ns22_auc, ~0.91-0.92 LB) |
| `notes/2026-05-22-strategic-direction-tracks-A-B.md` | Strategic context: Track A (Tucker) → Track B (own-teacher PL) |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` | Tucker's verbatim notebook (re-pullable via `kaggle kernels pull tuckerarrants/bc2026-distilled-sed`) |

---

## Pending decisions / open items for next session

1. **Verify Phase 2 jobs completed** (folds 1/2/3/4 + hgnet — 5 jobs total).
2. **Run validate_end_to_end on each new ckpt** (D.5 pre-flight, MANDATORY).
3. **Export + upload + push kernels** for `distilled-sed-kd-5fold` and `distilled-sed-kd-hgnet`.
4. **Submit both to LB**.
5. **Apply Phase 2 decision matrix** (see § Pick up here above) → drives next move:
   - Site/hour priors (EoS.6 port) if 5-fold lands < 0.92 and we want more lift.
   - Ensemble with exp 023 Perch+SED stack if 5-fold ≥ 0.92.
   - Pivot to HGNet if 2.D shows backbone was a bottleneck.
6. **Update results.md + MODEL-RUNS.md + this CONTEXT.md** with verdicts.
7. **PR #51** — decide on merge plan (squash all 30+ commits? Keep history?).

---

## How to resume after disconnect

1. Read this file top-to-bottom.
2. `git log --oneline -10` should show `224e569` → `5cd047f` → `db8660c` → `787e43f` → ... most recent first.
3. Check `squeue -u $USER` — expect the 5 Phase 2 jobs to be done.
4. Walk the "Pick up here" checklist above.
5. Verify tests still pass: `~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/032-tucker-distilled-sed/ -q`. Expect **163 pass** (151 from Phase 0 + 12 from `test_tucker_mel.py`).

**Slack / Kaggle / W&B credentials**: in `user/murilo/.env` (gitignored).
Loaded by `train.py`. If W&B login fails on SLURM, `WANDB_MODE=offline` or
`--fast-dev-run` skips W&B entirely.
