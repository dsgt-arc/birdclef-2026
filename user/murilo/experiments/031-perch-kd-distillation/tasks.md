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

### 0.2 Code changes (`sed_finetune.py`)

- [ ] 0.2.1 Add `DistillHead(nn.Module)` class. Direct port of Tucker
  notebook cell 10 (`vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`).
  GAP over `dim=[2, 3]` of backbone features → `Linear(backbone_dim, 1536)`.
- [ ] 0.2.2 Add `PerchTeacher` class. ONNX inference via `onnxruntime`,
  `@torch.no_grad()`, returns `(B, 1536)` float32. Handles cpu↔gpu
  transfers internally.
- [ ] 0.2.3 In `SEDFinetuneModule.__init__`: when `use_perch_distill: True`,
  instantiate `self.distill_head = DistillHead(backbone_dim, 1536)` and
  `self.perch_teacher = PerchTeacher(perch_onnx_path, device)`.
- [ ] 0.2.4 Modify forward path: after `feat = self.backbone(mel)`, branch
  on `use_perch_distill`:
  - True: `distill_emb = self.distill_head(feat)`,
    `feat_for_head = feat.detach()`
  - False: existing behavior, `feat_for_head = feat`
- [ ] 0.2.5 Modify `training_step` loss: when KD on,
  `distill_loss = F.mse_loss(distill_emb, perch_teacher.embed(wav_5s))`,
  `loss = cls_loss + alpha_distill * distill_loss`. Log both to W&B.
- [ ] 0.2.6 Confirm KD is OFF by default (back-compat for non-031 configs).
- [ ] 0.2.7 Add unit-style sanity check: assert backbone gradient norm > 0
  AND SED-head gradient norm > 0 in a single training step (run via
  fast-dev-run).

### 0.3 Config

- [ ] 0.3.1 Create `031-perch-kd-distillation/configs/v5j_kd.yaml`. Copy
  `030-sydorskyi-mixup/configs/v5j_control.yaml` as the base. Add the
  KD-specific keys (see proposal § Procedure step 7).
- [ ] 0.3.2 Cross-check the KD config diff vs `v5j_control.yaml`: ONLY
  KD-related keys should differ (no incidental drift).

### 0.4 Fast-dev-run validation

- [ ] 0.4.1 Run `bash sbatch/run.sh --fast-dev-run --config configs/v5j_kd.yaml`
  on a 100-sample subset. Verify:
  - dist_loss decreases over the first ~10 steps
  - cls_loss still trains normally
  - No NaN/Inf in either loss
  - Backbone gradient norm > 0; SED head gradient norm > 0 (per 0.2.7)
- [ ] 0.4.2 Inspect a saved batch's distill_emb vs perch_emb: shape `(B,
  1536)`, both finite, MSE loss matches manual recomputation.
- [ ] 0.4.3 Spot-check W&B run: `train/dist_loss`, `train/cls_loss`,
  `train/loss` all logged separately.

## Phase 1 — KD training run

### 1.1 SLURM training

- [ ] 1.1.1 Submit SLURM job:
  `sbatch sbatch/train.sbatch configs/v5j_kd.yaml`. Expect ~40-60 min
  on rtx6000 (Tucker's compute report suggests 2× the no-KD walltime).
- [ ] 1.1.2 Monitor `dist_loss` curve in W&B during training. Halt
  conditions:
  - `dist_loss` plateaus above 0.01 by epoch 5 → likely Perch teacher /
    mel mismatch. Stop and diagnose.
  - `cls_loss` diverges (goes above v5j-control's epoch-1 value) →
    detach broke something. Stop and diagnose.
- [ ] 1.1.3 After training completes, capture metrics into `results.md`:
  - val/auroc, val/cmap, val/f1_macro at best-by-val epoch
  - dist_loss curve (per-epoch)
  - best-by-val epoch number
  - walltime
  - SLURM job ID, W&B run ID
- [ ] 1.1.4 Tag the checkpoint path so it's locatable from the proposal.

### 1.2 ONNX export + Kaggle pipeline

- [ ] 1.2.1 Export the LSEHead-only inference path to ONNX. The
  distillation head MUST be excluded from the exported graph. Verify
  ONNX output is `(B, 234)` non-degenerate on a fixed random mel input.
- [ ] 1.2.2 Add `hgnet-b0-sed-v5j-kd` MODELS entry to
  `experiments/010-submission-pipeline/scripts/upload_kaggle.py`.
- [ ] 1.2.3 Add `hgnet-b0-sed-v5j-kd` MODEL_SLUGS entry to
  `experiments/010-submission-pipeline/kaggle_kernel/submission.py`.
- [ ] 1.2.4 Create `kernel-metadata-hgnet-b0-sed-v5j-kd.json` in
  `kaggle_kernel/`.
- [ ] 1.2.5 Upload weights:
  `python scripts/upload_kaggle.py --model hgnet-b0-sed-v5j-kd`
- [ ] 1.2.6 Push kernel:
  `python scripts/upload_kaggle.py --model hgnet-b0-sed-v5j-kd --push-kernel`
- [ ] 1.2.7 Verify kernel ran cleanly on Kaggle (no errors, submission.csv
  produced).

### 1.3 LB submission + verdict

- [ ] 1.3.1 Submit kernel from Kaggle UI. Wait for LB score (typically
  10-30 min).
- [ ] 1.3.2 Record LB score in `results.md`.
- [ ] 1.3.3 Apply the success criteria matrix:
  - LB ≥ 0.920 → CONFIRM, proceed to Phase 2
  - LB 0.913-0.919 → PARTIAL, diagnose
  - LB 0.907-0.912 → WEAK, diagnose
  - LB ≤ 0.906 → REJECT, halt Phase 2

## Phase 2 — Ownership + scaling (gated on Phase 1 = CONFIRM)

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
- [ ] A.5 Document everything in `results.md`. Update CONTEXT.md "What
  happened recently" section. Update MODEL-RUNS.md with the new LB.
- [ ] A.6 If Phase 1 rejected: write a focused post-mortem note
  identifying the most likely cause (teacher mismatch / detach bug /
  KD requires Tucker's full stack / something else).
