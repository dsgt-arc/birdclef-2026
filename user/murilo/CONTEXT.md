# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-10 (Phase 0 done; Phase 1 IN FLIGHT — full
training kicked off late 05-09 / early 05-10. Two SLURM jobs running:
8318953 v5j_kd, 8319084 v5j_control. Resume here when both finish.)

---

## Pick up here (next session)

**Two SLURM jobs are running** (or just finished). Check status first:

```bash
squeue -u $USER
ls -lt user/murilo/experiments/031-perch-kd-distillation/logs/
```

| Job | What | Started | Expected wall time |
|---|---|---|---|
| **8318953** | `v5j_kd` full training (KD on, the test) | 2026-05-09 23:37 | ~80-150 min on rtx6000 |
| **8319084** | `v5j_control` full training (KD off, the baseline + back-compat check) | 2026-05-10 00:05 | ~80-120 min on rtx6000 |

**When both finish** (per session-end exchange 2026-05-10 00:08):

1. **Pull the W&B side-by-side** in project `birdclef-2026-exp031`. Confirm:
   - `v5j_kd` `train/dist_loss` decayed monotonically — Tucker's published
     floor is ~0.003 by ep 25; we're at 20 epochs so expect ~0.005-0.010 floor.
   - `v5j_kd` `train/cls_loss` is finite throughout (no divergence).
   - `v5j_control` `val/auroc` trajectory matches exp 030's v5j_control_v2
     (best-by-val ep ~14-15, val/auroc plateau ~0.85). If it lands close to
     0.851 by ep 14-15, the KD-branch additions to `sed_finetune.py` are
     back-compat for non-KD runs (implicit verification).
   - **Halt-watch retro**: did `train/dist_loss > 0.01` by epoch 5? If yes,
     mel-pipeline mismatch contributed; document as a Phase 2 diagnostic
     question. (If not, the KD path is clean.)
2. **ONNX export both checkpoints** from `outputs/checkpoints/` dirs.
3. **Upload to Kaggle**:
   - `birdclef-2026-hgnet-b0-sed-v5j-kd-weights`
   - `birdclef-2026-hgnet-b0-sed-v5j-control-weights` (back-compat receipt)
4. **Push two kernels**, submit to LB.
5. **Apply success matrix** on KD result vs control's actual LB number:
   - CONFIRM if `v5j_kd` LB ≥ `v5j_control` LB + 0.014
   - PARTIAL if +0.007 to +0.013
   - WEAK if +0.001 to +0.006 (inside seed noise)
   - REJECT if ≤ control LB

If `v5j_control` LB diverges from exp 030's 0.896 by more than ±0.005,
that's a **back-compat failure** and we'd need to diagnose drift in the
KD-branch additions before trusting the KD result.

### Implementation decisions (locked, don't re-litigate)

- **Mixup interaction = Option B with the 1× teacher trick.**
  `_apply_mixup(return_params=True)` returns `(indices, lam)`; per-sample
  Perch invariance gives `perch(x[indices]) == perch(x)[indices]`, so
  the matched target costs 1× teacher forward. Phase 1 KD restricted to
  `mixup_mode="interpolate"` and `raw_waveform_mixup=False`; other modes
  raise `NotImplementedError`. See `proposal.md` § Mixup interaction.
- **Train/val split = Option A (no change).** The 4-single-sample-species
  duplicate-into-train+val is val/auroc reporting bias only (~0.002 macro),
  not train-weight contamination. Methodology audit deferred to a dedicated
  post-031 experiment.
- **Single fold for Phase 1.** ±0.013 seed noise band accepted; +0.014
  CONFIRM threshold sits at ~1.1× the noise floor. Phase 2 = 4-fold if
  Phase 1 confirms.

---

## Where exp 031 is

**Status**: Phase 0 implementation complete and verified; Phase 1 training
in flight (jobs 8318953 + 8319084).

| Phase | What | Status |
|---|---|---|
| 0.0 | Mel pipeline equivalence diagnostic | ✅ KEEP NNAUDIO + hop=313 (cosine sim 0.987 on STFT-only) |
| 0.1 | Vendor + verify Perch v2 ONNX teacher | ✅ All 6 checks pass; CUDA 17 ms/batch=4 |
| 0.2 | `DistillHead` + `PerchTeacher` + KD branch in `sed_finetune.py` | ✅ DONE — Option B with 1× teacher trick (commit `ede03e6`) |
| 0.3 | Config `configs/v5j_kd.yaml` + train.py + sbatch wrappers | ✅ DONE (commit `ede03e6`) |
| 0.4.1 | `test_kd_gradient_flow.py` (synthetic-batch local check) | ✅ ALL CHECKS PASS — backbone/head/distill_head grads > 0 in both mixup-on and mixup-off cases; row-permutation invariance exact (diff=0.0) |
| 0.4.2 | SLURM fast-dev-run (real Perch ONNX on GPU) | ✅ PASSED (job 8318887). 5 batches: cls_loss=0.699, dist_loss=0.058, loss=0.757 (= cls + 1.0·dist ✓). Zero affinity warnings. (Earlier job 8318761 failed on ONNX path; fixed in commit `5f4b6aa`.) |
| 0.4.3 | Add `v5j_control.yaml` for live W&B comparison + back-compat receipt | ✅ DONE (commit `fa94043`) |
| 1.1 | `v5j_kd` SLURM training | ⏳ RUNNING (job 8318953, started 2026-05-09 23:37) |
| 1.2 | `v5j_control` SLURM training (parallel) | ⏳ RUNNING (job 8319084, started 2026-05-10 00:05) |
| 1.3 | ONNX export + Kaggle upload + LB submit | pending 1.1/1.2 |

**Phase 1 success gate** (vs `v5j_control_v2 = 0.896` baseline, ±0.013 noise):
- CONFIRM if LB ≥ 0.910 → KD lever works on our codebase, proceed to Phase 2 (ownership + 4-fold).
- PARTIAL/WEAK 0.897-0.909 → diagnose mel pipeline, dist_loss floor.
- REJECT ≤ 0.896 → halt Phase 2; structural post-mortem.

---

## Read these files first

Top three are read fully; others are reference / skim.

| File | Why |
|---|---|
| `user/murilo/experiments/031-perch-kd-distillation/proposal.md` | Active experiment hypothesis + method + success criteria |
| `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Three structural findings from auditing Tucker's notebook (KD recipe shape, AttHead aggregation domain, dual-supervision blend) |
| `user/murilo/experiments/031-perch-kd-distillation/v5j-vs-tucker-pipeline-comparison.md` | Side-by-side full pipeline comparison, risk-coded; identifies the 7+ axes Tucker differs from v5j |
| `user/murilo/experiments/031-perch-kd-distillation/tasks.md` | Phase-by-phase task list with completion status |
| `user/murilo/experiments/031-perch-kd-distillation/phase0_perch_teacher_schema.md` | Perch ONNX schema: 4 outputs (clip emb, spatial emb, internal mel, classifier logits); Phase 1 uses output 0 only |
| `user/murilo/experiments/031-perch-kd-distillation/phase0_mel_equivalence.md` | Phase 0.0 verdict + hop_length explanation |
| `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` | Tucker's discussion post + architecture figures |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` | Tucker's actual notebook code; cells 4/10/12/14 are load-bearing |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor scores + "Proven 0.93+ recipes" synthesis |

Backlog (skim only if older context needed):
- `analysis/notes/2026-04-28-next-experiment-direction.md` — strategic synthesis post exp 030; PL vs KD distinction
- `analysis/notes/2026-04-26-mixup-implementation-audit.md` — v5j uses interpolated targets; Sydorskyi/Tucker use max
- `analysis/notes/2026-04-26-d1-train-side-attention-verdict.md` — v5j vs v6k attention; data-driven failure
- `analysis/notes/2026-04-25-per-frame-attention-analysis.md` — v5j is structurally honest on hard crops

---

## Best LB (unchanged through exp 030 close)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| v5j single (exp 019) | **0.906** | Best single SED; inference reproducible |
| effnetv2b0_sydorskyi (exp 030) | 0.901 | Best non-v5j single |
| v5j_control_v2 (exp 030) | **0.896** | Same recipe retrained, single fold seed=42 — Phase 1 baseline |

**Target**: 0.948 (gap 0.019 from current best ensemble, 0.014 from best
single SED).

---

## Key patterns and constraints

### Tucker audit findings (load-bearing 05-09)
1. **Tucker's KD = embedding-space MSE distillation with `feat.detach()`
   on SED branch.** Category #3 of "Perch as teacher", distinct from
   our exp 014/015/025 PL-on-unlabeled.
2. **Tucker's AttHead aggregates in logit space**
   (`sum(softmax(tanh(att)) * frame_logits)`); ours aggregates in
   probability space with `sigmoid → average → inverse-sigmoid`
   roundtrip. Likely root cause of our 7 AttHead regressions. Worth a
   fresh AttHead test with the bug fixed *after* exp 031, not as part of it.
3. **Dual-supervision blend at inference** (`0.5 * sigmoid(clip) + 0.5
   * sigmoid(frame_max)`) is the structural fix to v6k's uniform
   attention. Our v6k had `blend_clip=0.0`, removing gradient pressure
   on attention.

### "Perch as teacher" — three categories, only one tried
| # | What | Status |
|---|---|---|
| 1 | PL on UNlabeled data | exp 014/015/025 → "dead" |
| 2 | KD on labeled (KLD on logits) | NEVER TRIED (likely EliKal's 0.909) |
| 3 | KD on labeled (MSE on embedding + stop-grad) | NEVER TRIED (Tucker's 0.917 recipe) — **exp 031 tests this** |

### What moves LB
Backbone change (+0.037), mel normalization (+0.034), PP/TTA/smoothing
(+0.032), SED architecture (+0.028), Perch+SED ensemble fusion (+0.004).
**Common thread**: change *what the model sees* or *how predictions are
combined* — never *how the model trains*.

### What's dead (don't redo)
- CE loss (7 attempts, -0.015 to -0.409)
- Interpolated mixup variants (3 attempts, -0.019 to -0.031)
- PL on unlabeled (exp 014/015/025) — categories #2/#3 above are NOT this
- 2025-only pretrain at our scale (exp 029, -0.021)
- 7 AttHead variants — but see above audit finding; aggregation-domain bug means our test was structurally broken
- Background noise mixing (paused before run; per-frame analysis demoted it)
- Multi-context heads (-0.010 to -0.075)

### Single-fold seed band: ±0.013 on HGNet
v5j_control_v2 (seed=42) = 0.896, v5j_control_seed43 = 0.883. Any
single-fold delta below ±0.013 is inside noise. Future single-model
experiments need 4-fold or ≥3 seeds to escape it.

### v5j is structurally honest on hard crops
Per the 04-25 per-frame finding: GT-class attention entropy rises and
logit std collapses on low-SNR crops. BCE+random-crop training penalizes
this honesty (recording-level label says species is present whether or
not it's in the 5s window). Mechanistically explains 22+ recipe-tweak
failures. Implication: useful levers are *more information* (KD,
pretraining, external data) or *better inference* (PP/TTA), NOT making
the model more confident on hard crops.

### Best-vs-last checkpoint instability (deferred)
v5j-original best=ep 19; v5j_control_v2 best=ep 15. With `save_top_k=1`
we never serialize ep 19 in new runs. A `save_top_k=-1` retrain of
v5j_control would isolate "best plateau too early" from real codebase
drift. Cheap, not blocking exp 031.

### dual_validation lesson
Any flag that affects which checkpoint we serialize needs an LB receipt,
not just an offline-metric receipt. Verify weight divergence via
state_dict hash, not run-name alone.

### Offline metrics anti-correlate with LB
Cross-architecture comparisons via val/auroc are unreliable. cmap and
f1_macro are slightly more directional but still not LB-predictive
above noise.

### `head_lr` quirk
Silently ignored under `scheduler_type: one_cycle` (only used in cosine
branch of `configure_optimizers`). Only matters if we revisit split LR.

---

## Branch / git state

- **Current branch**: `mg/exp031-perch-kd-distillation`. **4 commits ahead**
  of main, all pushed to origin:
  - `fa94043` — Add v5j_control.yaml for live W&B comparison
  - `5f4b6aa` — Fix exp 031 fast-dev-run: ONNX path resolution + onnxruntime warnings
  - `ede03e6` — Implement exp 031 Perch v2 KD branch (Phase 0.2-0.4)
  - `3bb7afc` — Update CONTEXT.md branch/git state after PR #46 merge
- **Latest commit on main**: `24d813a` — "Scaffold exp 031: Perch v2
  knowledge distillation (Phase 0 setup) (#46)" (squash-merge of PR #46).
- **No open PRs.** Next PR opens after Phase 1 LB receipt.
- **Deleted branches** (cleanup history):
  - `mg/exp030-sydorskyi-mixup` (local + remote, fully on main via squash `58dd8c2`)
  - `mg/exp030-wrapup-and-next-direction` (local + remote, fully on main via squash `24d813a`)
  - `mg/exp031-perch-kd-distillation` was the head of PR #46; auto-deleted by GitHub on squash-merge, then recreated locally off updated main as the new working branch.

---

## Compressed timeline

| Date | What | Outcome |
|---|---|---|
| 04-23 | Built model attention explorer | Captured Salman 0.937 + Boredom 0.950+ |
| 04-24 | v5j triage tooling | Foundation for per-frame analysis |
| 04-25 | Audio validation + per-frame analysis | **v5j is structurally honest**; explains recipe failures. exp 029 launched |
| 04-25 PM | exp 029 LB 0.885 | REJECTED — 2025 pretrain at our scale doesn't help |
| 04-26 | Sydorskyi audit + D1 verdict + mixup audit + exp 030 scaffold | PR #42 merged; exp 030 v1 trained |
| 04-27 | exp 030 v2 retrain + v5j-resubmit kernel | v5j_control_v2 LB 0.896 surfaces ckpt-selection issue |
| 04-28 AM | v5j-resubmit LB 0.906 + deterministic-training root cause | seed=43 retrain launched |
| 04-28 PM | seed=43 LB 0.883 → seed band ±0.013 | **exp 030 closed**; PL vs KD distinction surfaced |
| 05-09 | Tucker notebook audit + exp 031 Phase 0 | Phase 0.0/0.1 done; PR #46 merged to main as squash `24d813a`; Phase 0.2 mixup-interaction unresolved |
| 05-09 PM | Mixup-interaction + split methodology resolved; Phase 0.2-0.4 implemented; fast-dev-run passed | Option B with 1× teacher trick locked; Option A (no split change) locked; commits `ede03e6` `5f4b6aa` `fa94043` |
| 05-10 AM | Phase 1 SLURM jobs kicked off | 8318953 v5j_kd + 8319084 v5j_control running in parallel; resume here next session |

---

## File index for active work

| File | Purpose |
|---|---|
| `user/murilo/experiments/031-perch-kd-distillation/` | **Active experiment** |
| `user/murilo/experiments/031-perch-kd-distillation/configs/v5j_kd.yaml` | KD training config (use_perch_distill: true) |
| `user/murilo/experiments/031-perch-kd-distillation/configs/v5j_control.yaml` | No-KD baseline config (parallel run for W&B comparison) |
| `user/murilo/experiments/031-perch-kd-distillation/test_kd_gradient_flow.py` | Synthetic-batch local sanity check (no GPU/ONNX needed) |
| `user/murilo/experiments/031-perch-kd-distillation/train.py` | Training driver with KD param plumbing |
| `user/murilo/experiments/031-perch-kd-distillation/sbatch/run.sh` | Launcher (`bash sbatch/run.sh configs/v5j_kd.yaml`) |
| `user/murilo/experiments/031-perch-kd-distillation/logs/` | SLURM logs — most recent: `slurm_v5j_kd_8318953.log`, `slurm_v5j_control_8319084.log` |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | DistillHead, PerchTeacher, KD branch in training_step |
| `user/murilo/analysis/mel_pipeline_comparison.py` | Phase 0.0 mel equivalence script |
| `user/murilo/analysis/verify_perch_teacher.py` | Phase 0.1 ONNX schema/IO verification |
| `vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx` | Vendored Perch teacher (413 MB, gitignored) |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/` | Tucker's notebook (vendored) |
| `user/murilo/MODEL-RUNS.md` | All model scores |
| `user/murilo/FINDINGS.md` | Cross-experiment history |
