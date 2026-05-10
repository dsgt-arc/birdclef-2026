# Tasks: 031 Perch KD Distillation

## Phase 0 — Setup (no GPU training)

### 0.0 Mel-pipeline equivalence diagnostic (gates the rest of Phase 0)

- [x] 0.0.1 Built numerical-equivalence script
  (`user/murilo/analysis/mel_pipeline_comparison.py`).
- [x] 0.0.2 Ran on 100 fixed training waveforms. Results:
  - **STFT-LIB ONLY** (libraries at matched hop=512, no norm):
    cos sim mean **0.987**, min 0.948 → NNAudio and torchaudio agree on
    STFT + mel filter bank.
  - **FULL** (v5j hop=313 + min-max vs Tucker hop=512 + standardize):
    shape mismatch (512 vs 313 time frames), cos sim 0.31 (dominated by
    scale/offset differences from min-max vs no-min-max, not by pattern
    disagreement).
- [x] 0.0.3 Applied corrected decision gate (see `phase0_mel_equivalence.md`
  § "Verdict (revised)"). Original gate fired "REFACTOR" but was
  poorly designed — abs-diff metric conflated scale offsets with
  signal disagreement. Corrected gate uses cosine similarity on the
  STFT-LIB-ONLY comparison (the meaningful number) → 0.987 ≥ 0.95 →
  **KEEP NNAUDIO + hop=313**. Proposal § Procedure 0a updated with the
  corrected gate logic.
- [x] 0.0.4 Documented in `phase0_mel_equivalence.md` including hop_length
  explanation and runtime-check criteria for Phase 1's dist_loss curve.

### 0.0a Mel pipeline refactor — **SKIPPED**

Skipped because the corrected gate (cosine sim 0.987 in STFT-LIB-ONLY
comparison) confirmed library agreement. Hop length and normalization
differences are tolerable for KD because: (a) Perch teacher takes raw
waveforms not mels, (b) the 1280→1536 projection has capacity to bridge
mel-pipeline differences. Single-variable discipline vs v5j is preserved.

If Phase 1 lands in PARTIAL/WEAK (LB 0.897-0.909) AND `dist_loss` floor
> 0.005 by epoch 25, revisit this decision as a Phase 2 diagnostic.

### 0.1 Vendor Tucker's Perch ONNX teacher (Option A) ✅ DONE 2026-05-09

- [x] 0.1.1 `kaggle datasets download tuckerarrants/perch-v2-no-dft-onnx`
  vendored to `vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx`
  (394 MB).
- [x] 0.1.2 ONNX integrity verified via `verify_perch_teacher.py` on a GPU
  node (SLURM job 8316130, rtx6000). All 6 checks pass:
  - Output shape `(B, 1536)` ✅
  - Non-degenerate on random + real waveforms ✅
  - Cross-sample distinctness (cos sim 0.054–0.281 off-diagonal) ✅
  - **CUDAExecutionProvider works**: 17 ms / batch of 4 (vs 2135 ms CPU).
- [x] 0.1.3 Schema documented at `phase0_perch_teacher_schema.md`. Notable
  bonus discoveries: ONNX exposes 4 outputs — clip embedding (idx 0,
  `[B, 1536]`), spatial embedding (idx 1, `[B, 16, 4, 1536]`), Perch's
  internal mel (idx 2, `[B, 500, 128]`), and Perch classifier logits
  (idx 3, `[B, 14795]`). Phase 1 uses idx 0 only; the others are noted
  for future Phase 3+ experiments (per-region KD, logit-space KD).
- **Operational decision**: training-time teacher forward (ONNX-on-GPU)
  is fast enough — no precompute needed. Per-fold training time expected
  to ~2-2.5× v5j baseline (40min → 80-120min).

### 0.2 Code changes (`sed_finetune.py`) ✅ DONE 2026-05-09

Decisions baked in (per session discussion 2026-05-09):
- **Mixup interaction = Option B**, with the 1× teacher trick. `_apply_mixup`
  uses `indices = torch.randperm(B)` to shuffle within the batch. Perch is
  per-sample (no cross-batch BN/attention), so `perch(x[indices]) ==
  perch(x)[indices]`. Compose target = `lam·perch(x) + (1-lam)·perch(x)[indices]`
  with **one** teacher forward.
- **Phase 1 KD restricted to** `mixup_mode="interpolate"` + `raw_waveform_mixup=False`
  (guards in `__init__`). Other mixup modes raise `NotImplementedError`.
- **Train/val split = Option A** (current), no methodology change.

- [x] 0.2.1 Added `DistillHead(nn.Module)` class. Direct port of Tucker
  notebook cell 10 (`vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`).
  GAP over `dim=[2, 3]` of backbone features → `Linear(backbone_dim, 1536)`.
  See `src/birdclef/lightning/sed_finetune.py:DistillHead`.
- [x] 0.2.2 Added `PerchTeacher` class (plain Python, NOT `nn.Module` — keeps
  Lightning from trying to relocate the ONNX session). Uses `onnxruntime` with
  `CUDAExecutionProvider` fallback to CPU. Returns `(B, 1536)` float32 with
  no_grad. See `src/birdclef/lightning/sed_finetune.py:PerchTeacher`.
- [x] 0.2.3 In `SEDFinetuneModule.__init__`: when `use_perch_distill=True`,
  instantiate `self.distill_head = DistillHead(encoder.embed_dim, 1536)` and
  `self.perch_teacher = PerchTeacher(perch_onnx_path)`. Guards: ValueError on
  missing path, NotImplementedError on non-interpolate mixup or `raw_waveform_mixup=True`.
- [x] 0.2.4 Modified `training_step` standard path: KD branch (when KD on)
  computes `perch_emb = teacher.embed(x)` on raw pre-mixup waveform, captures
  `(indices, lam)` from `_apply_mixup(..., return_params=True)` to compose
  the matching teacher target, then forwards through `encoder.forward_features`
  → `distill_head` (grad to backbone) and `sed_head(features.detach())`
  (no grad to backbone via cls).
- [x] 0.2.5 Modified `training_step` loss: when KD on,
  `loss = cls_loss + alpha_distill * distill_loss`. `train/cls_loss`,
  `train/dist_loss`, and `train/loss` all logged separately.
- [x] 0.2.6 KD is OFF by default (`use_perch_distill: bool = False` in
  `__init__` signature). Back-compat preserved for all other configs.
- [x] 0.2.7 Wrote standalone gradient-flow test
  `test_kd_gradient_flow.py` that monkey-patches a `StubPerchTeacher` so the
  test runs without the real ONNX. Verifies: backbone grad-norm > 0,
  sed_head grad-norm > 0, distill_head grad-norm > 0, `cls_loss`/`dist_loss`
  finite, and the row-permutation invariance `perch(x[perm]) == perch(x)[perm]`
  the 1× trick relies on.

### 0.3 Config ✅ DONE 2026-05-09

- [x] 0.3.1 Created `031-perch-kd-distillation/configs/v5j_kd.yaml`. Base =
  `019-sed-ablations/config/hgnet_b0_sed_v5j.yaml` (the v5j-original config —
  functionally identical to `030/v5j_control.yaml`, but using the original-name
  source for naming fidelity). Added KD keys: `use_perch_distill: true`,
  `perch_onnx_path: vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx`,
  `perch_embed_dim: 1536`, `alpha_distill: 1.0`. Made explicit:
  `mixup_mode: "interpolate"`, `raw_waveform_mixup: false` (required by KD path).
- [x] 0.3.2 Cross-checked: only KD keys + the two explicit defaults differ
  vs `v5j_control.yaml`. No incidental drift.

### 0.4 Fast-dev-run validation ✅ DONE 2026-05-09 / 05-10

- [x] 0.4.1 Wrote `test_kd_gradient_flow.py` (synthetic batch, no GPU
  required). **Ran 2026-05-09** — all 3 sub-tests passed:
  - Test 1 (KD on, mixup OFF): backbone grad 3.60e+00, sed_head 4.53e-01,
    distill_head 1.39e+00 (all > 0; loss=16.25, finite).
  - Test 2 (KD on, mixup ON, per-sample interpolate): all grads > 0
    (loss=12.03, finite).
  - Test 3 (row-permutation invariance): `max |perch(x[perm]) -
    perch(x)[perm]| = 0.00e+00` (exact, not approximate).
- [x] 0.4.2 SLURM fast-dev-run on GPU node. **Job 8318761 FAILED** with
  `NoSuchFile` because `train.py:repo_root` was misnamed (resolved to
  `<repo>/user/murilo/`, not the actual repo root). Fixed in commit
  `5f4b6aa` by adding `ACTUAL_REPO_ROOT = repo_root.parent.parent` and
  using it for ONNX path resolution. Same commit also added
  `intra_op_num_threads=1` / `inter_op_num_threads=1` to the
  `PerchTeacher` ORT `SessionOptions` to suppress 16+ noisy
  `pthread_setaffinity_np` warnings on cluster nodes. **Job 8318887
  PASSED** — 5 batches in ~4s, KD log line confirmed:
  `Perch v2 KD enabled: backbone_dim=1024, perch_embed_dim=1536,
  alpha_distill=1.0, onnx=<correct path>`. DistillHead 1.6M params
  (matches `Linear(1024, 1536)`).
- [x] 0.4.3 Confirmed 3 separate loss keys in fast-dev-run output:
  `train/cls_loss_step=0.699`, `train/dist_loss_step=0.058`,
  `train/loss_step=0.757` (= 0.699 + 1.0·0.058 ✓). Zero affinity warnings.
  No NaN / Inf. cls_loss ≈ ln(2) is the right BCE baseline at random
  init; dist_loss=0.058 will decay (Tucker's published floor ~0.003 by
  ep 25; with 20 epochs we expect ~0.005-0.010).

### 0.5 Live-comparison config (added 2026-05-09)

- [x] 0.5.1 Created `configs/v5j_control.yaml` (commit `fa94043`) — same
  encoder/head/mel/data/training/mixup/seed as v5j_kd.yaml minus the KD
  keys (`use_perch_distill` omitted → default False). Two purposes:
  (a) live W&B side-by-side with v5j_kd in project `birdclef-2026-exp031`,
  (b) implicit back-compat receipt — if this run lands at LB 0.896 ± 0.005
  (matching exp 030's v5j_control_v2), it confirms my KD-branch additions
  to `sed_finetune.py` left the non-KD path bit-identical.

## Phase 1 — KD training run

### 1.1 SLURM training (KD + control in parallel) ✅ DONE 2026-05-10

- [x] 1.1.1 Submitted both SLURM jobs 2026-05-09 PM / 05-10 AM:
  - **Job 8318953**: `bash sbatch/run.sh configs/v5j_kd.yaml`
    (started 2026-05-09 23:37, rtx6000, run name `2026-exp031-v5j-kd`)
  - **Job 8319084**: `bash sbatch/run.sh configs/v5j_control.yaml`
    (started 2026-05-10 00:05, rtx6000, run name `2026-exp031-v5j-control`)
  - Both finished cleanly; v5j_kd 78 min, v5j_control 38 min.
- [x] 1.1.2 W&B halt-watches all PASSED. dist_loss decayed cleanly to
  0.004 (Tucker's published floor ~0.003), cls_loss finite, zero NaN/Inf.
  KD val/auroc lag at ep 0-3 expected and observed (Δ −0.164 → −0.012 ep 14).
- [x] 1.1.3 Captured into `results.md` § Phase 1 training:
  - v5j_kd: val/auroc 0.8629 (ep 17), val/cmap 0.5376, val/f1_macro 0.4303,
    train/dist_loss 0.004, walltime 78m, W&B `xlkwpuv5`
  - v5j_control: val/auroc 0.8747 (ep 15), val/cmap 0.6268, val/f1_macro
    0.5270, walltime 38m, W&B `o16zyft7`
  - All 8 offline metrics for v5j_control bit-identical to 4 dp vs
    exp 030 v5j_control_v2 → **offline-side back-compat receipt sealed**.
- [x] 1.1.4 Checkpoints recorded:
  - `2026-exp031-v5j-kd/best-epochepoch=17.ckpt` + `last.ckpt` (ep 19)
  - `2026-exp031-v5j-control/best-epochepoch=15.ckpt` + `last.ckpt` (ep 19)

### 1.2 ONNX export + Kaggle pipeline ✅ DONE 2026-05-10

Decision baked in: skip v5j_control LB submission. Bit-identical offline
metrics under deterministic training imply bit-identical weights ⇒ LB
inherits exp 030's 0.896 without spending a daily Kaggle slot.

- [x] 1.2.1 Exported v5j_kd best-by-val ONNX via
  `experiments/010-submission-pipeline/scripts/export_efficientnet_sed.py`.
  `strict=False` load handled the `distill_head.*` keys cleanly (filtered
  to `unexpected`). ONNX output `(B, 234)` non-degenerate on the dummy mel
  in the export test. Commit `2241093`.
- [x] 1.2.2 Added `hgnet-b0-sed-v5j-kd` MODELS entry in `upload_kaggle.py`
  (slug `birdclef-2026-hgnet-b0-sed-v5j-kd-weights`).
- [x] 1.2.3 Added MODEL_SLUGS entry in `submission.py`.
- [x] 1.2.4 Created `kernel-metadata-hgnet-b0-sed-v5j-kd.json`
  (title slug-resolves cleanly to id).
- [x] 1.2.5 Uploaded weights dataset + pushed kernel.
- [x] 1.2.6 Kernel ran cleanly on Kaggle: ONNX session in 0.2s, 24 rows
  predicted across 2 train soundscapes, 36.9s total. No errors.

### 1.3 LB submission + verdict ✅ DONE 2026-05-10 — REJECT

- [x] 1.3.1 Submitted v5j_kd kernel. v5j_control submission skipped
  (offline bit-identity sealed back-compat).
- [x] 1.3.2 Recorded LBs:
  - **v5j_kd LB = 0.890**
  - v5j_control LB = 0.896 (inherited from exp 030 v5j_control_v2)
- [x] 1.3.3 **Back-compat check sealed via offline 4-dp metric match**
  (no LB submission needed; bit-identical metrics under deterministic
  training imply bit-identical weights).
- [x] 1.3.4 Applied success matrix vs v5j_control 0.896:
  - **REJECT** (LB 0.890 ≤ 0.896, inside but on wrong side of ±0.013 seed band)
  - KD lever did NOT transfer as a single-variable add on this codebase.
  - Phase 2 (ownership + 4-fold) **halted**.

### 1.4 Best-vs-last checkpoint diagnostic — RETRAIN REQUIRED, PENDING

**Post-hoc finding (2026-05-10)**: neither original run actually saved the
literal final-epoch (ep 19) weights. Inspection of all checkpoints showed:

- v5j_kd `last.ckpt`: epoch=1, global_step=5 (fast-dev-run leftover from job 8318887)
- v5j_kd `last-v1.ckpt`: epoch=17, global_step=9198 — bit-identical to `best-epochepoch=17.ckpt`
- v5j_control `last.ckpt`: epoch=15, global_step=8176 — bit-identical to `best-epochepoch=15.ckpt`

With `save_top_k: 1` + `save_last: true`, Lightning's `ModelCheckpoint`
callback only fires on metric improvement, and `save_last: true` is a
save-action modifier that mirrors whatever just got written. Once val/auroc
plateaus, `last.ckpt` is never updated. **The diagnostic requires a
`save_top_k: -1` retrain.** Saved as feedback memory
`feedback_save_top_k_does_not_preserve_last.md`.

#### 1.4a Retrain with save_top_k=-1

- [x] 1.4a.1 Created `configs/v5j_kd_resave.yaml` and
  `configs/v5j_control_resave.yaml`. Identical to originals except
  `save_top_k: -1` and `_resave` run-name suffix. Determinism preserved
  (`seed_everything(42, workers=True)` + `Trainer(deterministic=True)`),
  so per-epoch weights will be bit-identical to a hypothetical
  save_top_k=-1 version of the original runs.
- [ ] 1.4a.2 Submit two SLURM jobs:
  - `bash sbatch/run.sh configs/v5j_kd_resave.yaml` (~80 min, rtx6000)
  - `bash sbatch/run.sh configs/v5j_control_resave.yaml` (~40 min, rtx6000)
- [ ] 1.4a.3 After both finish, sanity-check: pull
  `best-epochepoch=17.ckpt` (kd_resave) and `best-epochepoch=15.ckpt`
  (control_resave) and verify they're bit-identical (state_dict hash) to
  the originals. Confirms determinism + that `_resave` runs reproduce.
  - Disk: ~1.5 GB (kd) + ~1.1 GB (control) = ~2.6 GB on scratch storage.

#### 1.4b LB submissions (after 1.4a)

- [ ] 1.4b.1 Add `hgnet-b0-sed-v5j-kd-last` MODELS entry in
  `upload_kaggle.py` pointing at
  `2026-exp031-v5j-kd-resave/best-epochepoch=19.ckpt`. Slug
  `birdclef-2026-hgnet-b0-sed-v5j-kd-last-weights` (47 chars, fits Kaggle 50-char limit).
- [ ] 1.4b.2 Add `hgnet-b0-sed-v5j-control-last` MODELS entry pointing at
  `2026-exp031-v5j-control-resave/best-epochepoch=19.ckpt`. Slug
  `birdclef-hgnet-b0-sed-v5j-control-last-weights` (47 chars, drops
  `-2026-` to fit limit per seed43 pattern).
- [ ] 1.4b.3 Update `export.sbatch` with two new export functions and
  dispatch entries; the explicit ckpt path in each function bypasses the
  `_find_best_ckpt` helper (no helper changes needed).
- [ ] 1.4b.4 Create 2 `kernel-metadata-*.json` files + 2 `MODEL_SLUGS`
  entries in `submission.py`.
- [ ] 1.4b.5 Export, upload, push, submit both → record LBs.

#### 1.4c Apply revised verdict (after 1.4b)

- [x] 1.4c.2 v5j_control_last LB result: **0.897** (ep 19, save_top_k=-1 rerun)
  - **Codebase reproduction FAILED.** Ep 19 weights from today's training
    give LB 0.897, NOT v5j-original's 0.906. The 0.009 gap is systematic
    (4 retrains in exp 030 + 031 all land 0.896-0.897), not seed noise.
  - Inference pipeline is clean (v5j-resubmit confirmed 0.906 from exp 019
    ep 19 ckpt). The drift is **training-side**.
  - Right v5j single-fold baseline going forward = **0.897**, not 0.906.
  - See `analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md` for
    root-cause investigation.
- [ ] 1.4c.1 Apply revised verdict using `v5j_kd_last` LB (kernel pushed,
  awaiting score). Read against the corrected baseline 0.897:
  - If `v5j_kd_last` ≥ 0.911 (0.897 + 0.014): KD lever transfers; REJECT
    was wrong. Proceed to Phase 2.
  - If `v5j_kd_last` 0.904–0.910: PARTIAL.
  - If `v5j_kd_last` 0.898–0.903: WEAK.
  - If `v5j_kd_last` ≤ 0.897: REJECT stands.

#### 1.4e Codebase drift root-cause analysis (URGENT)

The training-side drift means we cannot reproduce historical v5j-original
0.906 results from current code, and likely cannot reproduce other historical
single-fold runs either. Critical to identify the root cause before any
future LB-targeting work.

- [ ] 1.4e.1 Identify the exp 019 commit at which v5j-original was trained
  (look at `017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/` mtimes,
  cross-reference with git log)
- [ ] 1.4e.2 Diff key training-path files between that commit and HEAD:
  `src/birdclef/lightning/sed_finetune.py`, `src/birdclef/data/sed.py`,
  `src/birdclef/preprocessing/nnaudio_mel.py`, `src/birdclef/heads/sed_head.py`
- [ ] 1.4e.3 Diff library versions: `pyproject.toml`, `uv.lock` if available
- [ ] 1.4e.4 Diff config files: today's v5j_control.yaml vs the v5j-original
  training config in exp 019
- [ ] 1.4e.5 Pull v5j-original's training metrics.json (if archived) and
  compare per-epoch val/auroc trajectory to v5j_control_resave
- [ ] 1.4e.6 Form ranked hypothesis list, propose a single-variable
  diagnostic experiment to confirm the cause
- [ ] 1.4e.7 Document root cause + remediation path in
  `analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md`

#### 1.4d Documentation cleanup

- [ ] 1.4d.1 Update `feedback_save_top_k_does_not_preserve_last.md` if
  retrain reveals additional nuance (e.g., does `save_top_k=-1` interact
  with `early_stopping`? — currently we have patience=999 so likely not,
  but worth noting).
- [ ] 1.4d.2 Consider updating the project's default training config
  template (if one exists) to use `save_top_k: -1` by default — this
  pitfall has bitten us multiple times per user feedback.

## Phase 2 — Ownership + scaling (HALTED — Phase 1 verdict was REJECT)

### 2.1 Perch teacher ownership (P0.1 Option B or C)

- [ ] 2.1.1 Locate our existing Perch v2 inference code (exp 020). Identify
  the embedding layer (1536-d output before any classifier).
- [ ] 2.1.2 **Decision point**: Option B (export our own no-DFT ONNX) vs
  Option C (PyTorch teacher). Document the choice in
  `phase2_perch_ownership_decision.md`.
- [ ] 2.1.3 Implement the chosen option.
- [ ] 2.1.4 Validate equivalence: compute embeddings for 100 fixed
  validation waveforms with both Tucker's ONNX and our owned teacher.
  Cosine similarity should be ≥ 0.999 per-sample. If < 0.999, diagnose
  before proceeding.

### 2.2 Validate ownership transfer

- [ ] 2.2.1 Re-train `v5j_kd_v2` with our owned Perch teacher (single
  fold, seed=42). Same config as Phase 1 except `perch_onnx_path`
  pointing at our owned teacher.
- [ ] 2.2.2 ONNX export + Kaggle submit. Confirm LB lands within ±0.005
  of Phase 1 LB.
- [ ] 2.2.3 If LB matches → ownership transfer confirmed. If not →
  diagnose teacher difference before continuing.

### 2.3 4-fold scaling

- [ ] 2.3.1 Generate 4-fold splits (reuse v5j-4fold split logic from exp
  019/021). Configs: `v5j_kd_fold0`, `v5j_kd_fold1`, `v5j_kd_fold2`,
  `v5j_kd_fold3`.
- [ ] 2.3.2 Submit 4 SLURM jobs. ~3-4 hours total wall time on rtx6000.
- [ ] 2.3.3 Export 4 ONNX checkpoints, upload as 4 weights datasets,
  build a `v5j-kd-4fold` Kaggle kernel that ensembles all 4.
- [ ] 2.3.4 LB submission. Compare to v5j-4fold-pp2 (0.917) — target
  ≥ 0.93 if Tucker's "with PP" recipe maps cleanly.

## Analysis (after Phase 1, repeat after Phase 2)

- [ ] A.1 Inspect dist_loss trajectory. Floor at expected ~0.003 by epoch
  25? Higher floor suggests teacher/student input distribution mismatch.
- [ ] A.2 Compare cls_loss trajectory to v5j-control_v2. Faster
  convergence? Slower? Same? Tells us how much the "feature is
  Perch-aligned now" affects head learning.
- [ ] A.3 Best-by-val epoch — KD-trained model expected to reach plateau
  earlier or later than v5j? Note for `save_top_k=-1` deferred work.
- [ ] A.4 Per-frame analysis on the trained checkpoint: does Tucker's KD
  change v5j's "honesty" pattern from the 04-25 finding? (Re-run
  `per_frame_analysis.py`.) If attention sharpens or logit_std rises
  on hard crops, the mechanism is doing what we hoped.
- [x] A.5 Documented in `results.md`. MODEL-RUNS.md updated with v5j_kd LB
  0.890. CONTEXT.md update pending.
- [x] A.6 Post-mortem written into `results.md` § Post-mortem with 4 ranked
  candidate causes. See § Phase 3+ candidates below for the experiment
  options.

## Phase 3+ candidates (decision pending)

Phase 2 (ownership + 4-fold) is HALTED per the REJECT verdict. The
post-mortem in `results.md` ranks four candidate causes for the lever
not transferring; each maps to a follow-up experiment. Decide one (or
pivot away from KD entirely) before scaffolding.

Ordered by expected information value per LB submission:

- [ ] P3.1 **`v5j_kd` with bug-fixed AttHead** (logit-space aggregation per
  `feedback_atthead_aggregation_domain.md`). Tests "LSEHead is the wrong
  consumer of Perch features" — highest-likelihood candidate. Cost:
  ~80 min training + 1 LB.
- [ ] P3.2 **25- or 30-epoch retrain** of same v5j_kd recipe. Tests "20
  epochs was too short". Cost: ~110 min training + 1 LB.
- [ ] P3.3 **α_distill sweep** (0.1, 0.5, 2.0). Tests "α=1.0 miscalibrated
  for our codebase". Cost: 3 trainings + 3 LB. Expensive; lowest priority.
- [ ] P3.4 **Full Tucker replication** (multi-variable: EffNet-B0 NS-JFT +
  AttHead + dual-supervision + hard mixup + SpecAugment + focal×SS +
  90/10 batch + 25 ep cosine). Reverses the methodology — start with full
  reproduction, then ablate. Cost: 1 retrain + 1 LB.

Pivot candidates (away from KD):

- [ ] P3.5 Direct AttHead fix on v5j (no KD): test whether logit-space
  aggregation alone fixes the 7 AttHead regressions. Independent of KD;
  cheap test of the audit finding.
- [ ] P3.6 KLD on logits (KD category #2 from `project_kd_categories_2026_05_09.md`,
  the EliKal 0.909 recipe). NEVER TRIED in our project; structurally
  distinct from MSE-on-embedding (this experiment).
- [ ] P3.7 Other "more information" levers: external 2025 BirdCLEF data,
  background noise mixing (paused in earlier work), additional pretraining.
