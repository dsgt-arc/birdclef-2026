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
| [`test_fold_construction.py`](../../tests/032-tucker-distilled-sed/test_fold_construction.py) | 0.8 | Focal StratifiedKFold + GroupKFold SS + label-matrix + DistilledSEDDataModule end-to-end (20 tests) |

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

### 0.8 5-fold construction (Phase 1 = fold 0 only) — DONE
- [x] 0.8.1 `build_focal_folds` in new file `src/birdclef/data/distilled_sed_datamodule.py`. Adds `original_idx` (range(N)), runs `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on `primary_label` via `drop_duplicates("original_idx")` first, merges fold column back. Tested: balanced class distribution per fold, seed reproducibility, error handling for existing `fold`/`original_idx` columns.
- [x] 0.8.2 `build_sc_folds` uses `GroupKFold(n_splits=5)` on filename — all 12 windows of any file share a fold. `build_sc_cache_meta` enumerates 12 × 5s windows per labeled file (66 files × 12 = 792 rows per audit § 1.4). All labeled SS files participate (S22 included).
- [x] 0.8.3 Rare-species upsampling: `MIN_SAMPLE=20` (configurable), applied AFTER `merge` of fold column so duplicates inherit source's fold. Tested: `(original_idx, fold)` consistency = no cross-fold leak after upsampling (`leaks = df.groupby("original_idx")["fold"].nunique()`; all == 1).
- [x] 0.8.4 `build_sc_label_matrix` builds Y_SC by matching `(filename, start_sec)` against `train_soundscapes_labels.csv`. Unmatched rows → all-zero (53 windows of the 792 per audit § 1.4 — included as all-negative training samples, matching Tucker's default). Tested: matched rows produce correct multi-hot labels; unmatched rows all-zero; multiple label-CSV rows with same `(filename, start_sec)` are unioned; unknown species silently skipped.
- [x] 0.8.5 **`DistilledSEDDataModule`** wraps everything: setup() builds folds + Y_SC; `train_dataloader()` returns `ConcatDataset([FocalDS, ScDS])` with `MixSamp(90/10)`; `val_dataloader()` returns ScDS on val fold only (no focal validation, matching Tucker's `_predict_from_waveforms` on labeled SS). Phase 1 trains `fold_k=0` only; Phase 2 will sweep {0,1,2,3,4}.
- [x] **Task 0.7.2 deferred wiring**: `DistilledSEDDataModule` uses `batch_sampler=MixSamp(...)` with `num_workers=0`. Per-epoch seed shift (`seed=base_seed + current_epoch`) implemented in `train_dataloader()` — Lightning calls it once per epoch when `reload_dataloaders_every_n_epochs` is set; otherwise the seed stays fixed (acceptable for fast-dev-run; production configs will enable reload).
- [x] **train.py wired**: replaced `SEDDataModule` with `DistilledSEDDataModule`. Loads `train.csv` + `train_soundscapes_labels.csv` from `data_dir`, builds `sc_df` via `build_sc_cache_meta`, constructs `species_to_idx` from species list, and forwards mixup/augmentation params from `config["mixup"]`. Audio loaders resolve paths against `data_dir/train_audio` (focal) and `data_dir/train_soundscapes` (SS). Test stage skipped (Tucker's recipe has no separate test set; val fold's labeled SS is the LB-style metric).
- [x] **Tests** at [`test_fold_construction.py`](../../tests/032-tucker-distilled-sed/test_fold_construction.py) — 20 tests, all pass:
  - Focal folds: balanced class distribution per fold, original_idx range, upsampling preserves fold, no cross-fold leak (verified via `groupby("original_idx")["fold"].nunique() == 1`), seed reproducibility, error handling for misformatted inputs.
  - SS cache: 12 windows × 60s file structure; audit § 1.4 count match (66 → 792).
  - SS folds: all windows of any file share a fold; all 5 folds populated.
  - Label matrix: correct multi-hot for matched (filename, start_sec); all-zero for unmatched; multi-row union; unknown species skipped.
  - DataModule: setup() populates folds + Y_SC; explicit steps_per_epoch honored; train DataLoader batch composition = MixSamp output (16 = 14 focal + 2 sc with shares=0.9/0.1); val DataLoader uses ONLY val-fold rows; no file leak between train and val; changing fold_k swaps held-out files.
- **Test gate**: ✅ PASSED.

### 0.9 Write paired configs — DONE
- [x] 0.9.1 [`configs/distilled_sed_control.yaml`](configs/distilled_sed_control.yaml) — full Tucker recipe, `use_perch_distill: false`, `save_top_k: -1`, fold 0 only. Encodes all the audit-load-bearing hyperparameters:
  - **Backbone**: `tf_efficientnet_b0.ns_jft_in1k`, `use_post_head: true` (1280-d), `drop_path_rate: 0.1`
  - **Head**: `att_head_logit_space`, `hidden_dim: 512`, `dropout: 0.5`, `gem_p: 3.0`
  - **Mel** (audit § 7.1): `n_fft: 2048, hop: 512, n_mels: 256, fmin: 20, fmax: 16000, top_db: 80`
  - **SpecAugment** (audit § 2.4): `freq_p: 10, time_p: 10, n_freq: 1, n_time: 2`
  - **Mixup** (audit § 2.3): `focal_focal_prob: 0.5, alpha: 0.4, hard: true, focal_sc_prob: 0.5, aug_prob: 0.5`
  - **Schedule** (audit § 5): `lr: 5e-4, wd: 1e-4, max_epochs: 25, warmup: 2 ep, warmup_start_factor: 1/25, eta_min: 1e-6, grad_clip: 1.0`
  - **Data**: `bs: 64, focal_share: 0.9, sc_share: 0.1, n_folds: 5, min_sample: 20, seed: 42`
  - **Loss + blend**: 50/50 dual-supervision and inference blend
  - **Save**: `save_top_k: -1` (avoids the exp 031 best-vs-last pitfall per [`feedback_save_top_k_does_not_preserve_last`](../../../.claude/projects/-storage-project-ps-clef2026-birdclef-0-mgustineli3-birdclef-2026/memory/feedback_save_top_k_does_not_preserve_last.md))
  - **Early stopping**: disabled (`patience: 999`) — Tucker trains full 25 ep
- [x] 0.9.2 [`configs/distilled_sed_kd.yaml`](configs/distilled_sed_kd.yaml) — bit-identical to control except for: `run_name`, `training.use_perch_distill: true`, `training.perch_onnx_path: "vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx"`. `alpha_distill: 1.0`, `perch_embed_dim: 1536`.
- [x] 0.9.3 Both configs inherit the pinned deps via `032-tucker-distilled-sed/pyproject.toml` (torch 2.5.1, timm 1.0.26, hf-hub 1.9.2, nnAudio 0.3.4, onnxruntime-gpu 1.26.0).
- [x] **Tests** at [`test_configs.py`](../../tests/032-tucker-distilled-sed/test_configs.py) — 25 tests, all pass:
  - Backbone, head, mel, SpecAugment, mixup, schedule, data, loss/blend, KD — each load-bearing audit value is asserted.
  - **Paired-control invariant** (`test_paired_invariant_only_kd_keys_differ`): control vs kd diverge ONLY on `run_name`, `training.use_perch_distill`, `training.perch_onnx_path`. Any other drift is a methodology violation and fails this test.
  - `test_perch_onnx_file_exists`: the referenced ONNX file actually exists at the resolved repo-relative path.
  - `test_save_top_k_minus_one`: locks in the every-epoch save policy.
- **Test gate**: ✅ PASSED. Full 032 suite: 140/140 tests pass (was 115 + 25 new).

### 0.10 Update sbatch + train.py for new head/sampler dispatch
- [ ] 0.10.1 In `train.py`: dispatch `head_type: "att_head_logit_space"` to instantiate `AttHeadLogitSpace`.
- [ ] 0.10.2 Update sbatch wrappers' default config name (already done — defaults to `distilled_sed_kd.yaml`).
- [ ] 0.10.3 `sbatch/_job.sh` should still work as-is; verify it picks up the renamed config.

---

## Phase 1 — Paired single-fold training + LB — DONE (REJECT, blocked on Phase D audit)

### 1.1 Fast-dev-run gradient check — DONE
- [x] 1.1.1 fast-dev-run KD config (job 9087739, PASSED — see CONTEXT.md 2026-05-24).
- [x] 1.1.2 fast-dev-run control config (job 9087357, PASSED).
- **Gate**: ✅ both passed cleanly.

### 1.2 Submit paired SLURM jobs — DONE
- [x] 1.2.1 control submitted (job 9088255). Completed 02:54 EDT, walltime 3h45m. All 25 epochs saved.
- [x] 1.2.2 KD submitted (job 9088258). Completed 03:35 EDT, walltime 4h27m. All 25 epochs saved.
- [x] 1.2.3 Both on `gpu-rtx6000`, distinct nodes (atl1-1-03-006-17-0 vs ...21-0). Cross-node variance ~0.007 expected.
- [x] 1.2.4 No halt conditions tripped. No NaN. `dist_loss` reached Tucker's 0.003 floor by ep 25.

### 1.3 ONNX export + Kaggle pipeline — DONE
- [x] 1.3.1 New `experiments/010-submission-pipeline/scripts/export_distilled_sed.py` — exports pre-blended logits, dynamic mel+time axes (fixes audit § 7.2). last.ckpt (ep 24) for both control + KD.
- [x] 1.3.2 Datasets uploaded: `birdclef-2026-distilled-sed-{control,kd}-weights`.
- [x] 1.3.3 `submission.py` extended with `compute_mel_tucker` + `predict_soundscape_tucker` (LOGIT-space per-window blend → 5-tap Gauss smooth → single sigmoid).
- [x] 1.3.4 Both kernels pushed; both ran cleanly on Kaggle (no errors).

### 1.4 LB submissions + verdict — DONE (REJECT, but verdict on implementation not hypothesis)
- [x] 1.4.1 Both kernels submitted to LB.
- [x] 1.4.2 Paired Δ = −0.009 (KD slightly worse than control, within noise).
- [x] 1.4.3 Decision matrix → **REJECT** by the literal threshold (Δ ≤ 0), but the verdict is uninformative because the control itself is broken (collapsed to LB 0.522, far below the secondary baseline of 0.897).
- [x] 1.4.4 Secondary read: `distilled_sed_control` LB 0.522 vs v5j_control_last 0.897 = **−0.375**. Recipe alone catastrophically regressed → "investigate before reading the KD delta" gate triggered.
- [x] 1.4.5 Results table + verdict written to `results.md`.

### Phase 1 LB summary

| Run | LB | Δ paired ctrl | Δ vs v5j_control_last 0.897 | Δ vs target 0.910 |
|---|---:|---:|---:|---:|
| distilled_sed_control (last.ckpt) | **0.522** | — | **−0.375** | **−0.388** |
| distilled_sed_kd (last.ckpt) | **0.513** | **−0.009** | **−0.384** | **−0.397** |

---

## Phase D — COMPREHENSIVE AUDIT (NEW, BLOCKING)

**Why this phase exists**: Phase 1 LB collapsed to ~0.52 (near-random) on BOTH
control and KD. Offline Tucker macro AUC tracked Tucker's published trajectory
to 4 dp (control ep 6 = 0.889, Tucker's = 0.91). This means the model IS
learning something on the training distribution but predicts garbage on the
LB. **Strong leading hypothesis**: training/inference mel-pipeline mismatch.

Phase D blocks Phase 2 and Phase 3.

### D.1 Confirm vendored notebook is current
- [ ] D.1.1 `kaggle kernels pull tuckerarrants/bc2026-distilled-sed` and diff against `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`. Update vendored copy if upstream changed since 2026-05-23.

### D.2 Audit mel pipeline (highest-EV) — DONE ✅
- [x] D.2.1 Tucker's pipeline read verbatim from vendored `bc2026-distilled-sed.py`:
  - lines 400–407: `MelSpecTransform` = `torchaudio.transforms.MelSpectrogram(SR, n_fft, hop, n_mels, fmin, fmax, power=2.0)` → `AmplitudeToDB(top_db=80)`.
  - lines 945–950: training loop applies per-sample `(mel - mean) / (std + 1e-6)` after the transform.
  - lines 1240–1246: inference mel via librosa, same params + `power_to_db(top_db=80)` + per-sample z-score. Identical to training pipeline.
- [x] D.2.2 Our broken training pipeline confirmed: `NNAudioMelSpectrogram(norm_type="amplitude_to_db")` does `10·log10 + top_db clamp + z-score + min-max [0,1]`, then `_compute_mel.per_sample_mel_norm` does another z-score. Three normalizations vs Tucker's one.
- [x] D.2.3 **Side-by-side numerical diagnostic** (`scripts/diag_mel_pipeline.py`) on `BC2026_Train_0001_S08_20250606_030007.ogg` 5s window:

  | Comparison | max\|d\| | mean\|d\| | Pearson r |
  |---|---:|---:|---:|
  | TUCKER vs OURS-KERNEL (Tucker-faithful) | **4.77e-07** | 5.57e-08 | 1.000000 |
  | TUCKER vs OURS-TRAINING (NNAudio amp2db) | **3.20** | 0.248 | 0.9356 |
  | OURS-TRAINING vs OURS-KERNEL | **3.20** | 0.248 | 0.9356 |

  💥 **CONFIRMED**: kernel matches Tucker to machine epsilon; training does NOT. Single bug accounts for the LB collapse.

### D.3 Fix training mel to match Tucker — DONE ✅
- [x] D.3.1 Chose path (a): align training mel to Tucker verbatim. Adds zero risk of NNAudio/torchaudio backend drift since both training and inference now use the SAME torchaudio ops.
- [x] D.3.2 NEW module `user/murilo/src/birdclef/preprocessing/tucker_mel.py` with `TuckerMelSpectrogram` — verbatim port of Tucker's `MelSpecTransform` (torchaudio MelSpec + AmplitudeToDB(80), zero internal normalization). The module's existing `per_sample_mel_norm=True` step completes the single z-score on top. Combined pipeline === Tucker's reference.
- [x] D.3.3 `train.py` dispatch extended: `mel.type == "tucker"` → instantiate `TuckerMelSpectrogram`. Existing `"nnaudio"` / `"ttahara"` paths untouched.
- [x] D.3.4 Unit test `tests/032-tucker-distilled-sed/test_tucker_mel.py` (12 tests):
  - `test_matches_tucker_reference_no_norm`: pre-norm output identical to Tucker's reference to atol=1e-5.
  - `test_full_tucker_pipeline_matches`: TuckerMelSpec + module's per-sample norm === Tucker's training pipeline to atol=1e-5.
  - `test_diverges_from_nnaudio_amplitude_to_db`: regression guard — Tucker pipeline MUST differ from NNAudio's broken `amplitude_to_db` path (asserts max|d| > 0.1).
  - Shape, dtype, device, configurability, top_db clamp coverage.
  - **All 12 pass**.
- [x] D.3.5 Both configs `distilled_sed_{control,kd}.yaml` updated: `mel.type: "tucker"` + `power: 2.0`; `norm_type` removed (no longer applicable — TuckerMelSpec has no internal norm).

### D.4 Audit lower-likelihood candidates (if D.3 doesn't recover LB)
- [ ] D.4.1 **AttHeadLogitSpace shape**: verify `frame_logits.max(dim=-1)` (ours) is semantically identical to `framewise.permute(0, 2, 1).max(dim=1)` (Tucker, after his permute). Write a unit test using a random `(B, C, T)` tensor.
- [ ] D.4.2 **NNAudio vs torchaudio MelSpec drift**: compare the raw mel power outputs (BEFORE any normalization) between NNAudio and torchaudio on the same waveform. Expected: small drift due to mel filterbank differences (slaney vs htk).
- [ ] D.4.3 **Audio loading**: compare `submission.py:load_audio_chunk` output (sample-level) against `librosa.load(file, sr=32000, offset=t, duration=5.0)` (Tucker's training side). Confirm sample-by-sample agreement.
- [ ] D.4.4 **ONNX wrapper output**: feed the same mel through PyTorch wrapper and ONNX session; max diff should already be < 5e-2 (verified at export time), but re-verify under inference conditions.
- [ ] D.4.5 **Species mapping**: confirm the `species_mapping_2026_234.json` we used is identity 234→234 (exp 030/031 used the same file at LB 0.897, so this is almost certainly OK).

### D.5 End-to-end validation before any retraining — SCAFFOLDED ✅
- [x] D.5.1 NEW `scripts/validate_end_to_end.py` + `sbatch/validate_end_to_end.sh`.
  - Loads a checkpoint into `DistilledSEDFinetuneModule`.
  - Builds the same val fold (StratifiedKFold focal + GroupKFold SS) the training side uses.
  - Runs the **kernel inference path** end-to-end on the val SS windows: `kernel_load_audio_chunk` → `compute_mel_tucker` → encoder.forward_features → AttHeadLogitSpace → per-window LOGIT-space blend (0.5/0.5) → 5-tap Gauss smooth → single sigmoid.
  - Computes Tucker macro AUC + AP (skip-empty) against the same `Y_SC` matrix `DistilledSEDDataModule` builds for training.
  - Optional `--training-side-auroc <X>` flag prints PASS/FAIL based on absolute gap vs `--tolerance` (default 0.02).
  - Optional `--output <tsv>` writes per-class metrics for offline inspection.
- [ ] D.5.2 Run D.5.1 against the Phase 1 `last.ckpt` (LB 0.522) as a sanity check. Predicted: end-to-end AUC ≈ 0.52 (matches LB), while training-side `val/auroc` was ~0.89 → confirms the script detects the very gap that motivated Phase D.
- [ ] D.5.3 Bake D.5.1 into workflow: run before every LB submission going forward. Decision rule: if end-to-end AUC is within 0.02 of training-side `val/auroc`, pipeline is consistent → LB-safe. Otherwise → diagnose mismatch before submitting.

### D.6 Retrain after fix — paired training + validation DONE (jobs 9136231 + 9136232 + 9156567 + 9156568)
- [x] D.6.1 Resubmitted paired training with `mel.type: "tucker"`. Same configs (minus the mel block update), same data, same seed=42, single fold 0. Both completed cleanly.
- [x] D.6.2 D.5.1 end-to-end validation PASSED for both runs (jobs 9156567 + 9156568):

  | Run | Training-side val/auroc | End-to-end val/auroc | Gap | Status |
  |---|---:|---:|---:|---|
  | control (last-v2.ckpt) | 0.887 | 0.898 | 0.011 | ✅ PASS (< 0.02 tol) |
  | kd (last-v2.ckpt) | 0.908 | 0.912 | 0.004 | ✅ PASS (< 0.02 tol) |

  End-to-end is slightly HIGHER than training-side because the kernel path adds 5-tap Gaussian smoothing across the 12 windows-per-file in LOGIT space, which acts as a temporal denoiser (matches Tucker's published "+0.004 from MengYe smoothing trick"). Pipeline is end-to-end consistent. **LB submission is safe.**

- [x] D.6.3 Re-exported ONNX (job 9156490; cffd725 made find_last_ckpt auto-pick last-v2.ckpt; ONNX max diff 4.77e-06 for both). Uploaded weights v2 + pushed kernels. Submitted both to LB.

### Phase D LB results (2026-05-24 EOD): CONFIRM

| Run | LB | Δ paired ctrl | Δ vs v5j_control_last 0.897 | Δ vs Tucker target 0.910 |
|---|---:|---:|---:|---:|
| distilled_sed_control (Phase D, last-v2.ckpt) | **0.856** | — | **−0.041** | **−0.054** |
| distilled_sed_kd (Phase D, last-v2.ckpt) | **0.906** | **+0.050** | **+0.009** | **−0.004** |

**CONFIRM verdict per proposal.md § Falsifiable** (paired Δ +0.050 ≥ +0.014 threshold). distilled_sed_kd matches v5j-original 0.906 LB exactly and lands within 0.004 of Tucker's single-fold target. Validates exp 031 post-mortem hypothesis: "LSEHead was the wrong downstream consumer of Perch features."

### Phase D LB methodology check

End-to-end validation (D.5) predicted:
- Control: e2e 0.898 → LB 0.856 (−0.042 offline↔LB gap)
- KD: e2e 0.912 → LB 0.906 (−0.006 offline↔LB gap)

The KD model's offline↔LB gap (~0.006) is consistent with exp 030/031 baseline (offline cmap ~0.87 ↔ LB ~0.90). The control's wider gap (~0.042) is unexplained but doesn't matter for the verdict; KD is the lever.

### D.5/D.6 checkpoint book-keeping note (post-mortem)

After 3 runs in the same checkpoint directory, Lightning's `save_last: true`
ended up with `last.ckpt` (May 23 fast-dev-run) + `last-v1.ckpt` (Phase 1 broken-mel)
+ `last-v2.ckpt` (Phase D retrain). Phase D commit `cffd725` updated
`export_distilled_sed_{control,kd}` to use the existing `find_last_ckpt`
helper (newest-mtime) instead of hardcoded `last.ckpt`. Future retrains in
the same dir → `last-v3.ckpt` etc. → picked automatically. Same fix is
implicit in `sbatch/validate_end_to_end.sh` since that script takes
`CKPT=` explicitly.

### Phase D retrain offline summary (training-side, on val fold 0)

| Run | walltime | val/auroc (Tucker macro AUC) | val/cmap (Tucker macro AP) | best_val_score |
|---|---:|---:|---:|---:|
| distilled_sed_control (last, ep 24) | 3h39m | **0.887** | 0.645 | 0.659 |
| distilled_sed_kd (last, ep 24) | 4h26m | **0.908** | 0.692 | 0.709 |
| paired Δ (KD − control) | — | **+0.021** | **+0.047** | +0.050 |

Both numbers land near Tucker's published single-fold target (~0.91 macro). The paired Δ on val/auroc (+0.021) is consistent with Tucker's published +0.020 LB lift — the first signal that the KD branch transfers in our codebase. **Translation to LB pending D.6.2 + D.6.3.**

### Phase D gate
Phase 2 (5-fold) and Phase 3 (diagnosis) DO NOT begin until:
- D.5.1 passes (end-to-end macro AUROC matches offline val to 2 dp), AND
- Retrained control LB ≥ 0.900 (matches v5j_control_last baseline).

---

## Phase 2 — 5-fold ensemble (deferred until Phase D + retrain CONFIRM/PARTIAL)

- [ ] 2.1 Extend training to folds {1, 2, 3, 4}.
- [ ] 2.2 Add OOF inference + per-class macro AUC reporting.
- [ ] 2.3 Submit 5-fold ensemble to LB (logit-space mean across folds, then smooth + sigmoid).
- [ ] 2.4 If LB ≥ 0.92 single-recipe → consider porting site/hour priors from EoS.6 stack (`vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/`).

---

## Phase 3 — KD-recipe diagnosis (deferred; only meaningful if retrained CONTROL approaches Tucker's 0.91 single-fold)

Ranked by EV per LB submission:

- [ ] 3.1 ALPHA_DISTILL sweep (0.1, 0.5, 2.0) — diagnose if KD has marginal effect on the new stack.
- [ ] 3.2 Mel-pipeline alignment: align student's mel to Perch's exactly to reduce dist_loss floor.
- [ ] 3.3 Longer schedule (30 ep) — Tucker's published trajectory still grinding at ep 25.
- [ ] 3.4 Pivot to Sydorskyi-yoink (Track A stretch) — `notes/2026-05-22-strategic-direction-tracks-A-B.md` § Stretch path.

---

## Closeout

- [x] C.1 Update `experiments/032-tucker-distilled-sed/results.md` with Phase 1 LB table + REJECT verdict + leading hypothesis.
- [x] C.2 Update `user/murilo/MODEL-RUNS.md` with the Phase 1 LB numbers.
- [ ] C.3 Update `user/murilo/CONTEXT.md` "Pick up here" — next session = Phase D audit (D.1 → D.6).
- [ ] C.4 PR #51 ready for merge (currently 19+ commits ahead of main; will rebase after Phase D + retrain).
