# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-23 (Phase 0 of exp 032 complete — all 9 ports
implemented + tested + audit-validated. PR #51 open on branch
`mg/exp032-tucker-distilled-sed-replication`. Ready for Phase 1
fast-dev-run on SLURM, which Murilo runs.)

**Competition deadline**: 2026-06-03 (~11 days from this update).

---

## Pick up here (next session) — UPDATED 2026-05-23

**Phase 0 (port) is DONE.** Phase 1 (training + LB) is the next step.
Phase 1 starts with a SLURM fast-dev-run gradient check (task 1.1), then
the full paired 25-ep training (task 1.2), then ONNX export + Kaggle
kernels (task 1.3), then LB submission + decision matrix (task 1.4).

**Strategic direction unchanged** (from `notes/2026-05-22-strategic-direction-tracks-A-B.md`):
- **Track A (PRIMARY)**: full Tucker distilled-SED replication. Known
  ceiling 0.917 single / 0.93+ with 5-fold + PP. Code reference at
  `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`. **This
  is what exp 032 implements; all 9 Phase 0 ports done this session.**
- **Track A stretch**: Sydorskyi-yoink (Ali Ozan path, 0.941 ceiling).
  Methodology-only; start day 4–5 if Track A is stable.
- **Track B (final layer)**: own-teacher PL on unlabeled SS. +0.007 to
  +0.013 on top. Antoine's simple protocol.

### Immediate next step — task 1.1: fast-dev-run on SLURM

Per `feedback_user_runs_commands`, Murilo runs SLURM jobs.

**The exact commands + sanity-check checklist are in the previous
session output and reproduced in `experiments/032-tucker-distilled-sed/tasks.md`
§ Phase 1.** Pre-flight check + dry-run + submission commands:

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026

# 1) Pre-flight: confirm 140 tests still pass + ONNX file present
~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/032-tucker-distilled-sed/ -q | tail -5
ls -lh vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx

# 2) Dry-run inspect (no submission)
cd user/murilo/experiments/032-tucker-distilled-sed
bash sbatch/run.sh configs/tucker_control.yaml --fast-dev-run --dry-run
bash sbatch/run.sh configs/tucker_kd.yaml      --fast-dev-run --dry-run

# 3) Submit paired fast-dev-runs (control first; KD second)
bash sbatch/run.sh configs/tucker_control.yaml --fast-dev-run
bash sbatch/run.sh configs/tucker_kd.yaml      --fast-dev-run

# 4) Monitor
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.20R"
tail -F logs/slurm_tucker-control_*.log
tail -F logs/slurm_tucker-kd_*.log
```

Pass conditions (both jobs):
- COMPLETED 0:0 exit code
- 5 train batches + 5 val batches run cleanly, no NaN/Inf
- Logged config matches audit:
  - `EfficientNetEncoder: tf_efficientnet_b0.ns_jft_in1k, embed_dim=1280, post_head, drop_path=0.1`
  - `Building SED head: att_head_logit_space`
  - `SpecAugment enabled: freq_mask_param=10, time_mask_param=10, num_freq_masks=1, num_time_masks=2`
  - `build_sc_label_matrix: 739/792 SS windows matched ... 53 all-zero windows`
  - MixSamp `per_src=[focal=58, sc=6]` for bs=64
- `tucker_control` logs `train/loss` only (no `train/dist_loss`)
- `tucker_kd` logs **both** `train/cls_loss` and `train/dist_loss`, with
  `dist_loss` finite (Tucker's published floor ≈ 0.003; on random init
  we expect ~0.01 – 0.5 over 5 batches)
- LR at step 0: `2e-5` (= 5e-4 × 1/25, LinearLR start_factor)

If pass: move to task 1.2 (full 25-ep paired training). If fail: paste
the failing log section into the next session for diagnosis.

---

## Phase 0 status — DONE (all 9 tasks, 140/140 tests pass)

| # | Task | Commit | Tests | Output files |
|---|---|---|---:|---|
| 0.1 | AttHeadLogitSpace + GeMFreqPool | `2f624b3` | 14 | `src/birdclef/heads/sed_head.py` |
| 0.2 | EfficientNetEncoder `use_post_head=True` (1280-d for EffNet-B0) | `5341c58` | 11 | `src/birdclef/encoders/efficientnet.py` |
| 0.3 | DistilledSEDFinetuneModule (new module — KD branch + dual-supervision + schedule) | `394279e` | 26 (22+4 from 0.6) | `src/birdclef/lightning/distilled_sed_finetune.py` |
| 0.4 | FocalDS + ScDS + apply_aug (3-path mixup, waveform-domain, hard-max labels) | `72cfe23` | 18 | `src/birdclef/data/distilled_sed_dataset.py` |
| 0.5 | SpecAugment (1 freq×10 + 2 time×10, training-only) | `5bae24d` | 12 | `src/birdclef/preprocessing/spec_augment.py` |
| 0.6 | Schedule + clip-grad-norm verification (already wired in 0.3; verification + LR-curve tests added here) | `492b245` | (extended 0.3) | (tests in `test_dual_supervision_loss.py`) |
| 0.7 | MixSamp batch sampler (90/10 focal/SS, with-replacement, in-batch shuffled) | `00b7edc` | 14 | (added to `distilled_sed_dataset.py`) |
| 0.8 | Fold construction + DistilledSEDDataModule | `74bfc48` | 20 | `src/birdclef/data/distilled_sed_datamodule.py` |
| 0.9 | Paired `tucker_control.yaml` + `tucker_kd.yaml` configs | `d6cb1e9` | 25 | `experiments/032-tucker-distilled-sed/configs/` |

Plus: `22a5a5e` (Tucker dataset+inference gap audit), `3d80317` (audit
amendment with verified SS counts), `df0daf9` (scaffold dir + pyproject),
`23af3af` (proposal/tasks/results/save_metrics), `d6505ea` (tests
section in tasks.md), `f463863` (sbatch banner fix).

### What each module does

- **`AttHeadLogitSpace`**: Tucker's verbatim attention head. GeMFreqPool
  (p=3 with `clamp(min=1.0)`) → Dropout(0.25)→Linear→ReLU→Dropout(0.5)
  → two Conv1d (xavier init) → `clip_logits = sum(softmax(tanh(att)) *
  cla, dim=time)` in **logit space** (no sigmoid roundtrip — fixes the
  exp 020/021 saturation issue documented in
  `feedback_atthead_aggregation_domain`).
- **`EfficientNetEncoder(use_post_head=True)`**: switches timm creation
  from `features_only=True` (320-d last stage) to `num_classes=0,
  global_pool=""` (1280-d post-conv_head). **Load-bearing**: matches
  Tucker's `BirdSEDModel`.
- **`DistilledSEDFinetuneModule`**: replaces `SEDFinetuneModule` for
  exp 032 (kept original for v5j paths bit-identical). Loss is
  `0.5·BCE(clip) + 0.5·BCE(frame_max)` + `α·MSE(distill_emb, perch_emb)`.
  KD branch uses `features.detach()` so backbone is driven only by MSE.
  Schedule = AdamW(5e-4, 1e-4) + LinearLR(2 ep, start_factor=1/25) →
  CosineAnnealingLR(23 ep, eta_min=1e-6) per-step.
- **`FocalDS` / `ScDS`**: dataset classes implementing Tucker's 3-path
  mixup (focal-focal at p=0.5; focal-SS at p=0.5 only if focal-focal
  didn't fire; no-mixup otherwise). `apply_aug` = gain ±6 dB + SNR-
  controlled Gaussian noise (10–30 dB), no time-shift. Empirically
  verified rates at default config: 50% / 25% / 25% over 1000 samples.
- **`SpecAugment`**: wraps `torchaudio.transforms.{FrequencyMasking, TimeMasking}`.
  1 freq×10 + 2 time×10, applied AFTER per-sample mel normalization.
  Training-only — eval-mode skip is enforced by the LightningModule.
- **`MixSamp`**: custom `torch.utils.data.Sampler` for `ConcatDataset`.
  For bs=64, shares={"focal":0.9, "sc":0.1} → `per_src=[58, 6]`.
- **`DistilledSEDDataModule`**: builds focal folds (StratifiedKFold(5)
  on `primary_label` with `drop_duplicates("original_idx")` first;
  rare-species upsampling to MIN_SAMPLE=20 AFTER fold assignment),
  SS folds (GroupKFold(5) on `filename`), Y_SC label matrix (matches
  (filename, start_sec) against `train_soundscapes_labels.csv`).
- **Paired configs**: differ on exactly 3 keys (run_name,
  use_perch_distill, perch_onnx_path). The paired-control invariant
  is asserted by `test_paired_invariant_only_kd_keys_differ`.

---

## Key reference docs (load-bearing for Phase 1)

| Doc | What it contains |
|---|---|
| [`analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`](analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md) | Forward pass, AttHead aggregation domain, loss formula, mixup defaults, optimizer, mel pipeline. |
| [`analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md`](analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md) | Dataset class internals, fold construction, MixSamp 90/10, per-sample weights (inert), all-zero SS labels (53 of 792, 0.7% of training samples — included as all-negative per Tucker default), checkpoint selection, inference in logit space, 5-tap Gaussian smoothing, **two notebook bugs to NOT inherit** (mel-size mismatch in export wrapper, checkpoint filename inversion). |
| [`notes/2026-05-22-strategic-direction-tracks-A-B.md`](notes/2026-05-22-strategic-direction-tracks-A-B.md) | Strategic context: Track A (Tucker primary) → Track B (own-teacher PL). |
| [`notes/kaggle-discussion-tucker-distilled-sed-baseline.md`](notes/kaggle-discussion-tucker-distilled-sed-baseline.md) | Tucker's discussion thread + KD lift quantification (+0.02 LB). |
| [`experiments/031-perch-kd-distillation/results.md`](experiments/031-perch-kd-distillation/results.md) | Exp 031 REJECT verdict (closed); post-mortem (highest-likelihood cause: LSEHead is wrong consumer of Perch features — fixed in exp 032 by AttHeadLogitSpace). |
| [`experiments/032-tucker-distilled-sed/proposal.md`](experiments/032-tucker-distilled-sed/proposal.md) | Hypothesis (single-fold target 0.910–0.917), paired-control success matrix, full setup table. |
| [`experiments/032-tucker-distilled-sed/tasks.md`](experiments/032-tucker-distilled-sed/tasks.md) | Phase 0 (done) + Phase 1–3 checklist. Tests section at top has run commands + file index. |
| [`vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`](../../vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb) | Tucker's notebook (re-pulled 2026-05-23, byte-identical to 2026-05-09 vendored copy). |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.py` | .py extraction (gitignored, regenerable). |

---

## Exp 031 verdict (FINAL, closed 2026-05-11) — historical reference

Phase 1 KD as a single-variable add on **v5j HGNet+LSEHead+20ep+α=1.0** did
not transfer. Within-codebase paired-control delta: **−0.006**. Confined
to v5j HGNet+LSEHead — Tucker's full recipe (the thing exp 032
implements) is NOT falsified.

### Why KD didn't transfer on v5j+LSEHead (ranked) — and what exp 032 fixes

1. **LSEHead is the wrong consumer of Perch features** (highest likelihood). → **Exp 032 ports AttHeadLogitSpace** (logit-space aggregation, dual-supervision).
2. **20-epoch budget too short.** → **Exp 032 uses 25 ep** (Tucker's).
3. **α_distill = 1.0 miscalibrated** for our codebase. → **Exp 032 keeps α=1.0** but with the bug-fixed AttHead + dual-supervision the relative magnitudes work out differently.
4. **KD lift may require Tucker's full stack.** → **Exp 032 ports the full stack** (AttHead + dual-supervision + hard-max mixup + SpecAugment + focal×SS + 90/10 batch + 25-ep cosine + 5-fold + Gaussian smoothing).

### Drift investigation: RESOLVED 2026-05-11

- ~0.007 LB cross-hardware GPU fp non-determinism (fundamental, not fixable)
- ~0.002 LB code drift (pinned via `uv.lock` in exp 031 — exp 032's
  pyproject.toml carries the same direct pins: torch 2.5.1, timm 1.0.26,
  hf-hub 1.9.2, nnAudio 0.3.4, onnxruntime-gpu 1.26.0).

**Implication for exp 032 methodology**: paired within-run controls are
the only reliable signal. tucker_control vs tucker_kd are submitted
back-to-back on the same partition; read deltas vs that paired control,
**NOT** vs historical v5j 0.906.

---

## Best LB (unchanged from 2026-05-23 carry-over)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| v5j single (exp 019) | **0.906** | Best single SED (real). |
| effnetv2b0_sydorskyi (exp 030) | 0.901 | Best non-v5j single |
| v5j-eracode (exp 031 drift diag) | 0.899 | Same code as v5j-original, different node |
| v5j_control_v2 (exp 030) / v5j_control_last (exp 031, ep 19) | 0.896 / 0.897 | HEAD-code single-fold retrains |
| v5j_kd_last (exp 031, ep 19) | 0.891 | HEAD-code + KD; −0.006 vs paired control |
| **v5j_kd (exp 031, ep 17)** | **0.890** | REJECTED on v5j+HGNet+LSEHead |

**Target**: 0.948 (gap 0.019 from 0.929 ensemble, 0.042 from v5j-original).

**Exp 032 single-fold target**: 0.910 – 0.917 (Tucker's published 0.917
is 5-fold + PP; we expect slightly below at single-fold).

---

## Key patterns and constraints (carry-over + new from this session)

### `save_top_k: -1` policy
Both exp 032 configs use `save_top_k: -1` (save every epoch) per
`feedback_save_top_k_does_not_preserve_last`. Avoids the exp 031
best-vs-last ambiguity.

### Paired within-run controls
Only reliable methodology under ~0.007 LB cross-node variance. Submit
tucker_control + tucker_kd back-to-back; read the paired Δ, NOT vs
historical baselines.

### "Match Tucker" on all-zero SS labels — decision recorded 2026-05-23
Per audit § 1.4, **53 of 792 cached SS windows** are all-zero (sit
between annotated windows in otherwise-labeled files; ~0.7% of training
samples at 10% SC batch share). Default: include them as all-negative
training samples (Tucker's behavior). Methodologically clean because
these windows were annotator-reviewed.

### Audit-validation testing pattern
Each task's tests explicitly cite the audit section being validated
(e.g. `test_mixup_rates_match_audit_2_3` verifies empirically that
default mixup probs produce 50%/25%/25% over 1000 samples). Any drift
in implementation vs audit will fail loudly.

### Reproducibility: separate modules over in-place edits
For exp 032, we created NEW modules (`DistilledSEDFinetuneModule`,
`DistilledSEDDataModule`, `distilled_sed_dataset.py`) rather than
modifying shared code, to keep exp 030/031 paths bit-identical. Per
user pushback during task 0.3.

### Pinned environment
`experiments/032-tucker-distilled-sed/pyproject.toml` is the source of
truth (no `uv.lock`). Tight pins on torch 2.5.1, torchaudio 2.5.1,
pytorch-lightning 2.6.1, torchmetrics 1.9.0, timm 1.0.26,
huggingface-hub==1.9.2, nnAudio 0.3.4, onnxruntime-gpu==1.26.0. SLURM
`_job.sh` creates fresh venv in `$TMPDIR` per job.

### Tucker audit findings (load-bearing)
1. **Tucker's KD = embedding-space MSE distillation with `feat.detach()`** (Perch category #3).
2. **Tucker's AttHead aggregates in LOGIT space** (no sigmoid roundtrip).
3. **Dual-supervision at training + inference** (0.5·BCE clip + 0.5·BCE frame_max; 0.5·σ(clip) + 0.5·σ(frame_max)).
4. **Schedule**: AdamW + LinearLR(2 ep, start_factor=1/25) → Cosine(23 ep, eta_min=1e-6), per-step.
5. **MixSamp 90/10 focal/SS batch composition.**
6. **5-fold (StratifiedKFold focal + GroupKFold SS); all 66 labeled SS files participate.**
7. **Inference**: logit-space blend across folds + 5-tap Gaussian smoothing across the 12 windows-per-file, single sigmoid at end (submission-side, NOT in LightningModule).

### "Perch as teacher" — three categories
| # | What | Status |
|---|---|---|
| 1 | PL on UNlabeled data | exp 014/015/025 → dead |
| 2 | KD on labeled (KLD on logits) | NEVER TRIED |
| 3 | KD on labeled (MSE on embedding + stop-grad) | REJECTED on v5j+HGNet+LSEHead+20ep (exp 031, −0.006); **exp 032 retests on Tucker's full stack** |

---

## Branch / git state

- **Current branch**: `mg/exp032-tucker-distilled-sed-replication`. **16
  commits ahead** of main, all pushed to origin.
- **PR**: [#51](https://github.com/dsgt-arc/birdclef-2026/pull/51)
  (draft) — "Exp 032: Tucker distilled-SED full replication". Keep
  growing on this branch as Phase 1 lands.
- **Most recent commits (this session)**:
  - `f463863` — Fix sbatch banner (was 'Experiment 030: Sydorskyi MixUp')
  - `d6cb1e9` — Add paired tucker_control + tucker_kd configs (task 0.9)
  - `74bfc48` — Fold construction + DistilledSEDDataModule (task 0.8)
  - `00b7edc` — MixSamp 90/10 batch sampler (task 0.7)
  - `492b245` — Schedule + clip-grad-norm verification (task 0.6)
  - `5bae24d` — SpecAugment (task 0.5)
  - `72cfe23` — FocalDS + ScDS + apply_aug (task 0.4)
  - `394279e` — DistilledSEDFinetuneModule + dual-supervision (task 0.3)
  - `5341c58` — EfficientNetEncoder use_post_head (task 0.2)
  - `2f624b3` — AttHeadLogitSpace + GeMFreqPool (task 0.1)
  - `d6505ea` — Tests section in tasks.md
  - `23af3af` — proposal/tasks/results skeleton
  - `df0daf9` — Scaffold (dir copy + pyproject)
  - `3d80317` — Amend audit doc with verified SS counts
  - `22a5a5e` — Tucker dataset + inference gap audit
- **Latest commit on main**: `52d83b9` (squash-merge of exp 031 PR #50,
  this session).

---

## Compressed timeline (this session, 2026-05-23)

| Stage | What | Outcome |
|---|---|---|
| Session start | Loaded full context (CONTEXT.md + 2026-05-22 strategic note + Tucker audits + kaggle discussion threads) | Confirmed exp 031 closed; exp 032 scope = full Tucker replication |
| Pre-work 1 | Squash-merged exp 031 PR #50 to main | commit `52d83b9` on main |
| Pre-work 2 | Re-pulled Tucker's notebook; byte-identical to 2026-05-09 vendored copy | Confirmed source unchanged |
| Pre-work 3 | Wrote `2026-05-23-tucker-dataset-and-inference-audit.md` (gap audit for dataset/sampler/inference) | 22a5a5e |
| Pre-work 4 | Verified SS counts (10,658 SS files, 66 labeled, 53 all-zero windows of 792 cached); amended audit doc | 3d80317 |
| Pre-work 5 | Created new branch `mg/exp032-tucker-distilled-sed-replication`; opened PR #51 (draft) | (PR #51) |
| Scaffold | Copied exp 031 → exp 032 dir; pinned pyproject.toml; clean artifacts; restored proposal/tasks/results/save_metrics + artifacts/.gitkeep | df0daf9, 23af3af |
| 0.1 | AttHeadLogitSpace + GeMFreqPool ported; unit tests; train.py dispatch | 2f624b3 (14 tests) |
| 0.2 | EfficientNetEncoder use_post_head flag (1280-d post-conv_head); unit tests | 5341c58 (11 tests) |
| Tests section | Added tests run command + file index to tasks.md | d6505ea |
| 0.3 | DistilledSEDFinetuneModule (new module per user reproducibility pushback); audit-validation tests | 394279e (22 tests) |
| 0.4 | FocalDS + ScDS + apply_aug; rate verification 50/25/25% over 1000 samples | 72cfe23 (18 tests) |
| 0.5 | SpecAugment wrapping torchaudio masking; end-to-end integration with LightningModule | 5bae24d (12 tests) |
| 0.6 | Schedule + clip-grad-norm verification (extends 0.3 tests); flaky SpecAugment test fix | 492b245 (4 new tests) |
| 0.7 | MixSamp batch sampler; 14 tests incl. DataLoader integration | 00b7edc (14 tests) |
| 0.8 | Fold construction + DistilledSEDDataModule; train.py rewired to use it; test stage removed | 74bfc48 (20 tests) |
| 0.9 | Paired tucker_control + tucker_kd configs; paired-invariant test locks methodology | d6cb1e9 (25 tests) |
| Phase 1 prep | Fixed sbatch banner; wrote fast-dev-run commands + sanity-check checklist | f463863, this CONTEXT.md |

**Total: 140 tests, all pass.** Every Tucker-recipe component is ported,
tested against the audits, and wired through `train.py` + configs.

---

## File index for active work

| File | Purpose |
|---|---|
| **Modules (shared lib)** | |
| `user/murilo/src/birdclef/heads/sed_head.py` | + AttHeadLogitSpace + GeMFreqPool |
| `user/murilo/src/birdclef/encoders/efficientnet.py` | + `use_post_head` flag |
| `user/murilo/src/birdclef/lightning/distilled_sed_finetune.py` | DistilledSEDFinetuneModule (NEW; old SEDFinetuneModule untouched) |
| `user/murilo/src/birdclef/data/distilled_sed_dataset.py` | FocalDS + ScDS + apply_aug + MixSamp |
| `user/murilo/src/birdclef/data/distilled_sed_datamodule.py` | DistilledSEDDataModule + fold-construction helpers |
| `user/murilo/src/birdclef/preprocessing/spec_augment.py` | SpecAugment (Tucker's params) |
| **Experiment dir** | |
| `user/murilo/experiments/032-tucker-distilled-sed/proposal.md` | Hypothesis + paired success matrix |
| `user/murilo/experiments/032-tucker-distilled-sed/tasks.md` | Phase 0 done; Phase 1+ checklist |
| `user/murilo/experiments/032-tucker-distilled-sed/results.md` | Placeholder; fill as Phase 1 lands |
| `user/murilo/experiments/032-tucker-distilled-sed/configs/tucker_control.yaml` | Full Tucker recipe, KD OFF |
| `user/murilo/experiments/032-tucker-distilled-sed/configs/tucker_kd.yaml` | Full Tucker recipe, KD ON |
| `user/murilo/experiments/032-tucker-distilled-sed/train.py` | Wired to DistilledSEDDataModule + DistilledSEDFinetuneModule |
| `user/murilo/experiments/032-tucker-distilled-sed/sbatch/run.sh` + `_job.sh` | SLURM launchers |
| `user/murilo/experiments/032-tucker-distilled-sed/pyproject.toml` | Pinned deps (source of truth, no uv.lock) |
| **Tests (140 total)** | |
| `user/murilo/tests/032-tucker-distilled-sed/test_att_head_logit_space.py` | task 0.1 (14) |
| `user/murilo/tests/032-tucker-distilled-sed/test_efficientnet_encoder_post_head.py` | task 0.2 (11) |
| `user/murilo/tests/032-tucker-distilled-sed/test_dual_supervision_loss.py` | tasks 0.3 + 0.6 (26) |
| `user/murilo/tests/032-tucker-distilled-sed/test_tucker_mixup.py` | task 0.4 (18) |
| `user/murilo/tests/032-tucker-distilled-sed/test_specaugment.py` | task 0.5 (12) |
| `user/murilo/tests/032-tucker-distilled-sed/test_mixsamp_sampler.py` | task 0.7 (14) |
| `user/murilo/tests/032-tucker-distilled-sed/test_fold_construction.py` | task 0.8 (20) |
| `user/murilo/tests/032-tucker-distilled-sed/test_configs.py` | task 0.9 (25) |
| **Audit + reference notes** | |
| `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Forward + loss audit |
| `user/murilo/analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md` | Dataset + sampler + inference audit (THIS SESSION) |
| `user/murilo/notes/2026-05-22-strategic-direction-tracks-A-B.md` | Strategic direction |
| `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` | Tucker's discussion thread |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` | Tucker's verbatim notebook |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.py` | .py extraction (gitignored) |
| **Data + model paths** | |
| `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/` | Competition data (train.csv, train_soundscapes_labels.csv, audio dirs) |
| `vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx` | Perch v2 teacher ONNX |
| **Carry-over (exp 031)** | |
| `user/murilo/experiments/031-perch-kd-distillation/results.md` | Closed; REJECT verdict + post-mortem |
| `user/murilo/MODEL-RUNS.md` | All model scores |

---

## Pending decisions / open items for next session

1. **Fast-dev-run results** (task 1.1) — Murilo runs the commands above;
   we diagnose any failures together.
2. **Full 25-ep paired training** (task 1.2) — pending fast-dev-run pass.
   Same partition window; tucker_control + tucker_kd submitted back-to-back.
   Expected wall time ~90 min each on rtx6000.
3. **ONNX export bug fix** (task 1.3.1) — Tucker's notebook has a mel-
   size mismatch in `SEDExportWrapper` (hardcodes 128 mels, but training
   is 256). Our port needs to fix this — use N_MELS=256 consistently OR
   mark mel/time as dynamic axes. Specifics in `analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md` § 7.2.
4. **Submission inference in logit space** (task 1.3.3) — per audit § 7.3,
   submission.py needs per-window `0.5·clip + 0.5·frame_max` in **logit
   space**, then 5-tap Gaussian smooth across 12 windows-per-file, then
   single sigmoid at the end. Currently `DistilledSEDFinetuneModule.predict_step`
   blends in **probability space** (correct for offline metrics) — the
   submission-side LOGIT-space blend is separate work.
5. **Site/hour priors** (Phase 2 if Phase 1 passes) — NOT in Tucker's
   public notebook. Would port from EoS.6 (`vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/`).
6. **5-fold extension** (Phase 2 if Phase 1 CONFIRM or PARTIAL) — sweep
   folds {1, 2, 3, 4} after fold 0 lands.

---

## How to resume after disconnect

1. Read this file top-to-bottom.
2. Skim the two audit notes (`2026-05-09-...` + `2026-05-23-...`) — these
   are the ground-truth references.
3. Check git state: `git log --oneline -8` should show f463863 → d6cb1e9 → … 22a5a5e.
4. Verify tests still pass: `~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/032-tucker-distilled-sed/ -q`. Expect 140 pass.
5. Open `experiments/032-tucker-distilled-sed/tasks.md` — Phase 0 all
   marked DONE; Phase 1 task 1.1 is the next checklist.
6. Run the fast-dev-run commands above (or read § "Immediate next step").
7. After fast-dev-run results land: paste the log section, diagnose, or
   move to task 1.2 (full training).

**Slack / Kaggle / Wandb credentials**: in `.env` (gitignored). The
training script loads from `user/murilo/.env`. If W&B login fails on
SLURM, set `WANDB_MODE=offline` or use `--fast-dev-run` which already
skips W&B.
