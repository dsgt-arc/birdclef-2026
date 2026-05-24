# Tasks — Exp 032 Tucker distilled-SED replication

## Tests

Pytests live in `user/murilo/tests/032-tucker-distilled-sed/` (mirroring the experiment dir name under the shared tests/ tree, matching the existing `tests/016-sed-architecture/` convention).

Run them with the interactive venv:

```bash
~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/032-tucker-distilled-sed/ -v
```

| Test file | Task | Coverage |
|---|---|---|
| [`test_att_head_logit_space.py`](../../tests/032-tucker-distilled-sed/test_att_head_logit_space.py) | 0.1 | AttHeadLogitSpace + GeMFreqPool (14 tests) |
| [`test_efficientnet_encoder_post_head.py`](../../tests/032-tucker-distilled-sed/test_efficientnet_encoder_post_head.py) | 0.2 | EfficientNetEncoder `use_post_head` flag (11 tests) |
| [`test_dual_supervision_loss.py`](../../tests/032-tucker-distilled-sed/test_dual_supervision_loss.py) | 0.3 | DistilledSEDFinetuneModule: dual-supervision loss + inference blend + KD branch + audit-validation tests (22 tests) |
| _planned_ `test_tucker_mixup.py` | 0.4 | waveform mixup with `MIXUP_HARD=True` + focal×SS mixup |
| _planned_ `test_specaugment.py` | 0.5 | 1 freq × 10 + 2 time × 10 masking |
| _planned_ `test_warmup_cosine_schedule.py` | 0.6 | LinearLR(2ep) → CosineAnnealingLR(23ep) |
| _planned_ `test_mixsamp_sampler.py` | 0.7 | MixSamp 90/10 batch composition |
| _planned_ `test_fold_construction.py` | 0.8 | StratifiedKFold focal + GroupKFold SS + upsampling |

The shared `user/murilo/tests/pytest.ini` configures the runner for this layout (`testpaths = .`, `python_files = test_*.py`).

---

## Reference materials

Read these BEFORE starting any task. Implementation tasks reference them by short name.

| Short name | File |
|---|---|
| **Forward+loss audit** | [`analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`](../../analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md) |
| **Dataset+inference audit** | [`analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md`](../../analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md) |
| **Tucker discussion** | [`notes/kaggle-discussion-tucker-distilled-sed-baseline.md`](../../notes/kaggle-discussion-tucker-distilled-sed-baseline.md) |
| **Strategic direction** | [`notes/2026-05-22-strategic-direction-tracks-A-B.md`](../../notes/2026-05-22-strategic-direction-tracks-A-B.md) |
| **Architecture figures** | [`notes/figures/tucker-distilled-sed-arch.png`](../../notes/figures/tucker-distilled-sed-arch.png), [`arch2.png`](../../notes/figures/tucker-distilled-sed-arch2.png) |
| **Source notebook** | `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` + `.py` extraction |
| **Exp 031 KD scaffolding (~30% already done)** | [`experiments/031-perch-kd-distillation/`](../031-perch-kd-distillation/) |
| **Proposal** | [`proposal.md`](proposal.md) |

---

## Phase 0 — Port the Tucker recipe into our codebase (no SLURM yet)

Each task lists the **reference** for the verbatim code, and a **test** that gates the next task.

### 0.1 Port `AttHeadLogitSpace` to `src/birdclef/heads/sed_head.py` — DONE
- [x] 0.1.1 `AttHeadLogitSpace` + `GeMFreqPool` added to `src/birdclef/heads/sed_head.py`. Verbatim port of Tucker's notebook cells 10/14: GeMFreqPool with `p.clamp(min=1.0)`, Dropout(0.25)→Linear→ReLU(inplace)→Dropout(0.5), two `Conv1d(hidden, num_classes, k=1)` with xavier_uniform init and zero bias, aggregation in logit space (`sum(softmax(tanh(att)) * cla, dim=time)`).
- [x] 0.1.2 Dispatch wired in `train.py`: `head_type: "att_head_logit_space" → AttHeadLogitSpace`. Constructor signature compatible with existing `head_kwargs` dict (dropout=0.5 default reproduces Tucker's exact Dropout(0.25)→Dropout(0.5) sequence).
- [x] 0.1.3 Unit test at `user/murilo/tests/032-tucker-distilled-sed/test_att_head_logit_space.py` covers: forward shapes, manual-formula equality (no sigmoid roundtrip), gradient flow through both convs, gradient through frame-max supervision (att not in this graph), xavier zero-mean init, zero bias init, GeMFreqPool p-clamp, output shape, p learnable, finite output on edge-case inputs, train/eval mode both work. **14/14 tests pass** (verified pre-existing failures in `tests/016-sed-architecture/` are unrelated — `SEDFinetuneModule.training_step` tuple-unpacking drift, not from this task).
- **Test gate**: ✅ PASSED.

### 0.2 Swap backbone HGNetV2-B0 → `tf_efficientnet_b0.ns_jft_in1k` — DONE
- [x] 0.2.1 `EfficientNetEncoder` extended with a `use_post_head: bool = False` flag. Default behavior unchanged (features_only mode, last-stage output). When True, switches to Tucker's mode: `num_classes=0, global_pool=""` exposing the post-conv_head 1280-d feature map for EffNet-B0. **This was load-bearing**: features_only=True stops BEFORE conv_head and gives only 320-d, NOT the 1280-d Tucker's `BirdSEDModel` operates on. Wired through `train.py` via `config["encoder"]["use_post_head"]`.
- [x] 0.2.2 Verified output shape on (B, 1, 256, 313) input: Tucker mode produces (B, 1280, 8, 10), spatial dims unchanged between modes (only channel dim differs). End-to-end: encoder.forward_features → AttHeadLogitSpace → (B, 234) clip + (B, 234, 10) frame.
- [x] 0.2.3 Pretrained=True load verified for `tf_efficientnet_b0.ns_jft_in1k` in Tucker mode (`drop_path_rate=0.1, in_chans=1, num_classes=0, global_pool=""`). Forward produces finite output. HF Hub download succeeds.
- [x] **Existing tests** (`tests/016-sed-architecture/`, default-mode encoder usage) unchanged — same 19 pre-existing test_sed_v3.py failures (`SEDFinetuneModule.training_step` tuple-unpacking drift, NOT from this task).
- [x] **New unit tests** at `user/murilo/tests/032-tucker-distilled-sed/test_efficientnet_encoder_post_head.py` — 11 tests cover: default mode unchanged for EffNet-B0 (320-d) + HGNet (1024-d), Tucker mode gives 1280-d, forward returns single tensor (not list), output is 4D, pooled forward still works, end-to-end with AttHeadLogitSpace, gradient flow from BCE loss reaches both encoder and head, spatial dims match across modes (only channel dim differs), drop_path_rate propagates. **11/11 pass**.
- **Test gate**: ✅ PASSED.

### 0.3 Add dual-supervision loss + inference blend — DONE
- [x] 0.3.1 **New module** `src/birdclef/lightning/distilled_sed_finetune.py` with class `DistilledSEDFinetuneModule` (per user pushback on reproducibility: separate module rather than modifying `SEDFinetuneModule` in-place). Implements dual-supervision loss `clipwise_weight · BCE(clip_logits) + framewise_weight · BCE(frame_max_logits)` with default weights 0.5/0.5 matching Tucker's notebook cell 14. Verified equivalent to Tucker's exact formula (`bce = 0.5*bce_clip + 0.5*bce_frame; ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8); cls_loss = (ps * sw).mean()`) when `wt=mk=ones, sw=ones` — direct fp comparison gives diff 1.19e-07. Module also imports `DistillHead` and `PerchTeacher` from `sed_finetune.py` (leaf utility classes; safe to share).
- [x] 0.3.2 **Inference blend in module**: `_blended_probs` returns `blend_clip · σ(clip) + blend_frame · σ(frame_max)` in probability space, used by validation_step + predict_step. Matches Tucker's `_predict_from_waveforms` cell 14. **Submission-side logit-space blending across folds** lives in `submission.py` (deferred to task 1.3.3 per dataset+inference audit § 7.3) — not in this module.
- [x] 0.3.3 Unit tests at [`test_dual_supervision_loss.py`](../../tests/032-tucker-distilled-sed/test_dual_supervision_loss.py) — 22 tests covering: training_step no-NaN, loss equals Tucker's formula, weights respected (synthetic logits), blend equals 0.5·σ(clip)+0.5·σ(frame_max), blend weights respected, gradient flow through both head convs, gradient flow to encoder when KD off, KD loss includes distill term, KD backbone gradient ONLY from distill branch (cls detached), KD backbone gradient present with α=1, configure_optimizers builds SequentialLR with per-step interval, LR curve correctness, fall-back when no warmup room, validation_step runs, per-sample mel norm produces ~zero mean unit std, **audit-validation tests**: exact formula match to cell 14, teacher waveform cropping (160000 samples; verifies center-crop on too-long and right-pad on too-short), inference blend bounded in [0,1], cls branch uses detached features when KD on, full loss equals cls + α · dist. **22/22 pass**.
- [x] **Existing tests**: `tests/016-sed-architecture/` unchanged (same 19 pre-existing failures unrelated to this task). `test_att_head_logit_space.py` (task 0.1) + `test_efficientnet_encoder_post_head.py` (task 0.2) still pass.
- **Train.py wiring**: `train.py` imports `DistilledSEDFinetuneModule` and constructs it with the Tucker-specific arg list (no soft_ce/ce/asl, no manifold_mixup, no multi-context).
- **Known deviations** (documented in module docstring):
  - `focal_missing` source weight (sw=0.0) not handled — our pipeline guarantees no missing cache files, so this is a no-op.
  - `clip_grad_norm_(1.0)` deferred to Lightning's `Trainer(gradient_clip_val=1.0)` config (task 0.6 verifies).
- **Test gate**: ✅ PASSED.

### 0.4 Port Tucker mixup recipe
- [ ] 0.4.1 In `src/birdclef/data/sed.py` (or whatever owns the dataset class): replace v5j's spec-mixup with waveform mixup. `lam = Beta(0.4, 0.4)`, `MIXUP_HARD=True → lb = max(lb1, lb2)`, p=0.5 per sample. (dataset+inference audit § 2.3.)
- [ ] 0.4.2 Add focal × labeled-SS mixup: with p=0.5 (only if focal-focal mixup didn't fire), blend with a random labeled SS window from the same training fold. `lb = max(lb_focal, lb_sc)`. (dataset+inference audit § 2.3 step 3.)
- [ ] 0.4.3 Add `apply_aug`: gain jitter ±6 dB (p=0.5) + SNR-controlled Gaussian noise (10–30 dB, p=0.5). Time shift comes from the random crop offset, NOT `apply_aug`. (dataset+inference audit § 2.4.)
- **Test gate**: 100-sample dataset iteration produces expected mixup rate (~50% focal-focal, ~25% focal-SC, ~25% none).

### 0.5 Add SpecAugment
- [ ] 0.5.1 New module wrapping `torchaudio.transforms.FrequencyMasking(freq_mask_param=10)` + `TimeMasking(time_mask_param=10)`. Apply 1× freq mask + 2× time mask in sequence (dataset+inference audit § 1.0 / notebook cell 10).
- [ ] 0.5.2 Apply AFTER per-sample mel normalization (per Tucker's order). Both at training only.
- **Test gate**: visual mel-spec output shows expected masked regions on a sample batch.

### 0.6 Switch schedule OneCycle → 25 ep cosine + 2 ep warmup
- [ ] 0.6.1 Replace OneCycleLR with `SequentialLR([LinearLR(start_factor=1/25, total_iters=warmup_steps), CosineAnnealingLR(T_max=total_steps - warmup_steps, eta_min=1e-6)], milestones=[warmup_steps])`. Verify warmup_steps = nst × 2, total_steps = nst × 25.
- [ ] 0.6.2 Add `clip_grad_norm_(parameters(), 1.0)` between `scaler.unscale_` and `scaler.step` (Lightning's `gradient_clip_val` should do this — verify).
- **Test gate**: fast-dev-run shows LR rises linearly for 2 epochs, decays cosine afterward.

### 0.7 Implement `MixSamp` 90/10 batch sampler
- [ ] 0.7.1 Port `MixSamp` as a custom `torch.utils.data.Sampler` (notebook cell 12; dataset+inference audit § 3). For BATCH=64, SHARES={"focal":0.9, "sc":0.1} → 58 focal + 6 SC per batch, with-replacement, shuffled.
- [ ] 0.7.2 Wire into `SEDDataModule` so train_dataloader uses `batch_sampler=MixSamp(...)`. Set `num_workers=0` initially to match Tucker (can revisit later).
- [ ] 0.7.3 Unit test: 100 batches → confirm ~58 focal + ~6 SC composition; seed shift per epoch.
- **Test gate**: unit test passes; one fast-dev-run epoch completes with expected batch composition logged.

### 0.8 5-fold construction (Phase 1 = fold 0 only)
- [ ] 0.8.1 Focal `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `primary_label` with `drop_duplicates("original_idx")` (dataset+inference audit § 1.1).
- [ ] 0.8.2 Soundscape `GroupKFold(n_splits=5)` on `filename` — all 66 labeled files participate (dataset+inference audit § 1.2).
- [ ] 0.8.3 Rare-species upsampling: `MIN_SAMPLE=20`, applied AFTER fold assignment so duplicates stay in their original's fold (dataset+inference audit § 1.3).
- [ ] 0.8.4 All-zero-label SS handling: **match Tucker (include them as all-negative training samples)** per the dataset+inference audit § 12 default. 53 windows ≈ 0.7% of training samples; sit between annotated windows in otherwise-labeled files.
- [ ] 0.8.5 Phase 1 trains fold 0 only; Phase 2 extends to {0,1,2,3,4} if Phase 1 passes the gate.
- **Test gate**: fold assignment matches a known reference (focal duplicates share original's fold; no SS-window leak across folds).

### 0.9 Write paired configs
- [ ] 0.9.1 `configs/tucker_control.yaml` — full Tucker recipe, `use_perch_distill: false`, `save_top_k: -1`, fold 0 only.
- [ ] 0.9.2 `configs/tucker_kd.yaml` — same as control + `use_perch_distill: true, alpha_distill: 1.0, perch_embed_dim: 1536, perch_onnx_path: vendor/kaggle/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx`.
- [ ] 0.9.3 Both inherit pinned deps via `032-tucker-distilled-sed/pyproject.toml`.

### 0.10 Update sbatch + train.py for new head/sampler dispatch
- [ ] 0.10.1 In `train.py`: dispatch `head_type: "att_head_logit_space"` to instantiate `AttHeadLogitSpace`.
- [ ] 0.10.2 Update sbatch wrappers' default config name (already done — defaults to `tucker_kd.yaml`).
- [ ] 0.10.3 `sbatch/_job.sh` should still work as-is; verify it picks up the renamed config.

---

## Phase 1 — Paired single-fold training + LB

### 1.1 Fast-dev-run gradient check
- [ ] 1.1.1 `bash sbatch/run.sh configs/tucker_kd.yaml --fast-dev-run` → 1 batch forward+backward, confirm gradients flow through both cls head and dist head, dist_loss is finite, no NaN/Inf.
- [ ] 1.1.2 Repeat for `tucker_control.yaml` with `use_perch_distill: false`.
- **Gate**: both fast-dev-runs succeed cleanly.

### 1.2 Submit paired SLURM jobs
- [ ] 1.2.1 `bash sbatch/run.sh configs/tucker_control.yaml` — KD off.
- [ ] 1.2.2 `bash sbatch/run.sh configs/tucker_kd.yaml` — KD on.
- [ ] 1.2.3 Verify both land on `gpu-rtx6000` (control hardware variance). If they land on different nodes, log the disparity — exp 031 showed ~0.007 LB cross-node variance.
- [ ] 1.2.4 Monitor halt conditions per `proposal.md` § Halt-watch: `dist_loss` plateaus >0.01 by ep 5 → halt; either loss NaN → halt.

### 1.3 ONNX export + Kaggle pipeline
- [ ] 1.3.1 Export each fold's best-by-macro checkpoint to ONNX. **Fix the export wrapper bug**: use N_MELS=256 consistently OR mark mel/time axes as dynamic in `dynamic_axes` (dataset+inference audit § 7.2).
- [ ] 1.3.2 Upload weights datasets to Kaggle (`tucker_control` + `tucker_kd` slugs).
- [ ] 1.3.3 Add MODEL_SLUGS entries in `submission.py`. Implement inference in **logit space** (per-window `0.5·clip + 0.5·frame_max`, fold-mean, 5-tap Gaussian smooth, single sigmoid).
- [ ] 1.3.4 Push paired Kaggle kernels.

### 1.4 LB submissions + decision matrix
- [ ] 1.4.1 Submit both kernels to LB.
- [ ] 1.4.2 Compute paired Δ (tucker_kd − tucker_control).
- [ ] 1.4.3 Apply success matrix (`proposal.md` § Falsifiable):
  - **CONFIRM**: Δ ≥ +0.014 → Phase 2 (5-fold)
  - **PARTIAL**: Δ +0.007 to +0.013 → Phase 2 with caveats
  - **WEAK**: Δ +0.001 to +0.006 → 4-fold to disambiguate or accept noise
  - **REJECT**: Δ ≤ 0 → Phase 3 diagnosis
- [ ] 1.4.4 Also read `tucker_control` LB vs v5j_control_last 0.897 to isolate recipe-vs-KD attribution.
- [ ] 1.4.5 Update `results.md` with paired-table + verdict + post-mortem.

---

## Phase 2 — 5-fold ensemble (only if Phase 1 CONFIRM or PARTIAL)

- [ ] 2.1 Extend training to folds {1, 2, 3, 4}.
- [ ] 2.2 Add OOF inference + per-class macro AUC reporting.
- [ ] 2.3 Submit 5-fold ensemble to LB (logit-space mean across folds, then smooth + sigmoid).
- [ ] 2.4 If LB ≥ 0.92 single-recipe → consider porting site/hour priors from EoS.6 stack (`vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/`).

---

## Phase 3 — Diagnosis (only if Phase 1 REJECT)

Ranked by EV per LB submission:

- [ ] 3.1 ALPHA_DISTILL sweep (0.1, 0.5, 2.0) — diagnose if KD has marginal effect on the new stack.
- [ ] 3.2 Mel-pipeline alignment: align student's mel to Perch's exactly to reduce dist_loss floor.
- [ ] 3.3 Longer schedule (30 ep) — Tucker's published trajectory still grinding at ep 25.
- [ ] 3.4 Pivot to Sydorskyi-yoink (Track A stretch) — `notes/2026-05-22-strategic-direction-tracks-A-B.md` § Stretch path.

---

## Closeout

- [ ] C.1 Update `experiments/032-tucker-distilled-sed/results.md` with final table + verdict.
- [ ] C.2 Update `user/murilo/MODEL-RUNS.md` with LB numbers.
- [ ] C.3 Update `user/murilo/CONTEXT.md` "Pick up here" with the next direction.
- [ ] C.4 PR #51 ready for merge.
