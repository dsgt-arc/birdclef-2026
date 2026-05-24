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
| [`test_tucker_mixup.py`](../../tests/032-tucker-distilled-sed/test_tucker_mixup.py) | 0.4 | FocalDS + ScDS + apply_aug: waveform mixup, hard-max labels, focal×SS mixup, augmentation, rate verification (18 tests) |
| [`test_specaugment.py`](../../tests/032-tucker-distilled-sed/test_specaugment.py) | 0.5 | SpecAugment: defaults match Tucker, masking actually zeroes rows/cols, integration with DistilledSEDFinetuneModule (12 tests) |
| (extended in `test_dual_supervision_loss.py`) | 0.6 | LR curve completeness + gradient_clip_val contract (4 new tests added to the task-0.3 file) |
| [`test_mixsamp_sampler.py`](../../tests/032-tucker-distilled-sed/test_mixsamp_sampler.py) | 0.7 | MixSamp batch composition, with-replacement sampling, edge cases (14 tests) |
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

### 0.4 Port Tucker mixup recipe — DONE
- [x] 0.4.1 **New file** `src/birdclef/data/distilled_sed_dataset.py` (per reproducibility argument: separate from `data/sed.py` to keep exp 030/031 paths bit-identical). Contains `FocalDS`, `ScDS`, and `apply_aug`. `FocalDS` implements Tucker's 3-path logic: focal-focal mixup → focal-SS mixup → no-mixup. Waveform mixup with `lam = Beta(mixup_alpha, mixup_alpha)`, `mixup_hard=True → lb = max(lb1, lb2)`, default `mixup_prob=0.5, mixup_alpha=0.4`.
- [x] 0.4.2 Focal × labeled-SS mixup wired into `FocalDS`: only reached when focal-focal didn't fire; `focal_sc_mixup_prob=0.5` default; eligible SS pool filtered to `fold != fold_k` per audit § 2.3 step 3. `MIXUP_HARD=True → lb = max(lb_focal, lb_sc)`.
- [x] 0.4.3 `apply_aug` implements gain jitter (±6 dB, p=0.5) + SNR-controlled Gaussian noise (10–30 dB, p=0.5). **No time-shift** — time variation comes from the random crop offset in `_load_focal_chunk` per audit § 2.4 (cell 12 only has gain + noise; the markdown's "three augmentations" claim is incorrect).
- [x] **Tests** at [`test_tucker_mixup.py`](../../tests/032-tucker-distilled-sed/test_tucker_mixup.py) — 18 tests, all pass:
  - `apply_aug`: p=0 no-op; p=1 modifies; gain range respected (±6 dB tested over 200 samples); SNR range respected (10–30 dB tested with sinusoid signal).
  - `FocalDS`: returns 5-tuple `(wav, lb, wt, mk, src)` matching Tucker's signature; wt/mk all ones per audit § 4; no-mixup label matches primary.
  - Mixup paths: `mixup_prob=1.0 → 100% focal-focal`; both probs=0 → 100% no-mixup; `focal_sc_mixup_prob=1.0, mixup_prob=0 → 100% focal-SC`.
  - `MIXUP_HARD=True → max(lb1, lb2)`: avg positives per label = ~1.95 over 50 samples (theoretical 1.95 with 20 rows in df).
  - `MIXUP_HARD=False`: λ-weighted soft labels produce values in (0, 1).
  - Beta(0.4, 0.4) distribution: λ values empirically have mean ≈ 0.5 and > 30% extremes in [0, 0.1] ∪ [0.9, 1.0] (bimodal as expected).
  - SS pool fold filtering: when all SS rows are in `fold_k`, focal-SS path never fires.
  - `ScDS`: no mixup; label returned as-is from `Y_SC[i]`; shape-mismatch validation raises.
  - **Audit-validation `test_mixup_rates_match_audit_2_3`**: 1000-sample empirical rates over the default config produce **~50% focal-focal, ~25% focal-SC, ~25% no-mixup**, matching audit § 2.3 exactly (±5% tolerance).
- **Test gate**: ✅ PASSED.

### 0.5 Add SpecAugment — DONE
- [x] 0.5.1 New module `src/birdclef/preprocessing/spec_augment.py` with class `SpecAugment` — verbatim port of Tucker's cell 10. Wraps `torchaudio.transforms.FrequencyMasking(freq_mask_param=10)` + `TimeMasking(time_mask_param=10)`. Applies `num_freq_masks=1` freq masks then `num_time_masks=2` time masks in sequence. All four params are configurable; defaults match Tucker exactly.
- [x] 0.5.2 Applied AFTER per-sample mel normalization in `DistilledSEDFinetuneModule._compute_mel`, training only — the LightningModule guards via `if training and self.spec_augment is not None`. (torchaudio's masking modules don't honor `self.training` themselves; the eval-mode skip is the caller's responsibility.)
- [x] **Wired through `train.py`** via `config["spec_augment"]` block (`enabled`, `freq_mask_param`, `time_mask_param`, `num_freq_masks`, `num_time_masks` — defaults match Tucker). Logs the active config on construction.
- [x] **Tests** at [`test_specaugment.py`](../../tests/032-tucker-distilled-sed/test_specaugment.py) — 12 tests, all pass:
  - Defaults match Tucker exactly (freq=10, time=10, n_freq=1, n_time=2)
  - Output shape and dtype preserved
  - Output is finite
  - `num_freq_masks=0 + num_time_masks=0` is identity transform
  - Freq mask zeroes some freq rows (full-row zero detection on constant input)
  - Time mask zeroes some time columns (same approach)
  - Combined 1 freq + 2 time masks zero both axes over 20-batch sample
  - Mask width never exceeds `param` (verified over 50 samples with param=20)
  - End-to-end through `DistilledSEDFinetuneModule._compute_mel`: training=True applies SpecAugment; training=False skips it
  - `spec_augment=None` → identity even at training=True
  - Stochastic (different calls produce different masks)
  - `.to(device)` works (submodules registered as nn.Modules)
- **Visual gate**: replaced the "visual mel-spec" check with programmatic verification that masked rows/cols are exactly zero (more reliable + reproducible than visual inspection). The combined-masking test runs 20 random batches and asserts both axes get masked.
- **Test gate**: ✅ PASSED.

### 0.6 Switch schedule OneCycle → 25 ep cosine + 2 ep warmup — DONE
- [x] 0.6.1 **Schedule was implemented in task 0.3** (`DistilledSEDFinetuneModule.configure_optimizers`) — `SequentialLR([LinearLR(start_factor=1/25, total_iters=warmup_steps), CosineAnnealingLR(T_max=total_steps − warmup_steps, eta_min=1e-6)], milestones=[warmup_steps])`, per-step `interval`. **Verification added in this task**:
  - `test_warmup_total_steps_match_audit_formula`: with `estimated_stepping_batches=2500, max_epochs=25` → `nst=100`, `warmup_steps=200`, `total_steps=2500`. SequentialLR's milestone is exactly 200; CosineAnnealingLR's `T_max=2300=total_steps-warmup_steps`.
  - `test_lr_reaches_eta_min_at_end_of_training`: stepping through all `total_steps` lands LR at `eta_min` within fp tolerance.
  - `test_lr_curve_shape_audit_compliance`: 250-step curve verified — starts at `lr × warmup_start_factor` (2e-5), peaks at `lr` (5e-4) at end of warmup, decays monotonically through the cosine phase, ends at `eta_min` (1e-6). Audit § 5 fully covered.
- [x] 0.6.2 `clip_grad_norm_(parameters(), 1.0)` deferred to Lightning's `Trainer(gradient_clip_val=...)`, which calls `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=v, norm_type=2.0)` internally between `backward()` and `optimizer.step()` — exactly matching Tucker's manual call. Wired through `train.py` line 595: `gradient_clip_val=config["training"]["grad_clip_val"]`. Configs (task 0.9) will set `training.grad_clip_val: 1.0` per Tucker. **Verification**:
  - `test_gradient_clip_handled_by_lightning_trainer`: documents the contract and sanity-checks PyTorch's `clip_grad_norm_(params, 1.0)` produces post-clip total norm ≈ 1.0 on a synthetic case with pre-clip norm > 1.
- **Test gate**: ✅ PASSED (4 new tests in `test_dual_supervision_loss.py`, all pass; cumulative file is now 26 tests).

### 0.7 Implement `MixSamp` 90/10 batch sampler — DONE
- [x] 0.7.1 `MixSamp` added to `src/birdclef/data/distilled_sed_dataset.py` (alongside FocalDS / ScDS — same module). Verbatim port of Tucker's notebook cell 12. For `bs=64, shares={"focal":0.9, "sc":0.1}` → per_src=[58, 6]. With-replacement uniform sampling per source via `rng.integers(0, size, size=n)`, indices translated by per-source offsets so they index correctly into a `ConcatDataset`, then in-batch shuffled with `rng.shuffle`. Tucker's rounding-remainder logic preserved (largest source gets the difference when `bs * share` doesn't round cleanly). `share=0.0` exactly opts a source out (the `max(1, ...)` floor applies only to nonzero shares).
- [x] 0.7.2 **Wiring deferred to task 0.8** — the actual DataModule that uses `MixSamp` needs the fold-construction infrastructure (focal df + sc df + Y_SC + sc_mixup_sources pool) that task 0.8 builds. `MixSamp` itself is ready to be wired with `batch_sampler=MixSamp(...)`, `num_workers=0`.
- [x] 0.7.3 **Tests** at [`test_mixsamp_sampler.py`](../../tests/032-tucker-distilled-sed/test_mixsamp_sampler.py) — 14 tests, all pass:
  - **Audit § 3 verification**: `bs=64, shares={"focal":0.9, "sc":0.1}` → exactly 58 focal + 6 SC per batch over 10 batches
  - Offsets translate correctly (multi-source case with 3 sources)
  - Rounding-remainder distribution: batch size always exactly `bs` regardless of share fractions
  - `share=0.0` explicitly opts source out (zero indices from that source)
  - Small nonzero share gets floor-of-1
  - `__len__` equals `nst`; same seed reproduces; different seeds diverge (Tucker uses `seed=42+ep` per epoch)
  - In-batch shuffle: indices from different sources are interleaved, not contiguous
  - With-replacement: small pool (size=10) + large batch (50) produces duplicates
  - Empirical rate matches share exactly over 100 batches (focal: 58/64, sc: 6/64)
  - Zero-sized source skipped (relevant for empty SS pool when held out)
  - Invalid inputs raise ValueError (mismatched lengths, bs<=0, nst<0)
  - **End-to-end with `DataLoader` + `ConcatDataset`**: integration test confirms the sampler works as `batch_sampler=` argument and produces the expected per-source counts at retrieval time.
- **Test gate**: ✅ PASSED. Fast-dev-run with composition logging is part of task 1.1 (Phase 1 fast-dev-run); the unit-test integration test already exercises the DataLoader path.

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
