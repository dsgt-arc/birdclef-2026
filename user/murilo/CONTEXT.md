# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-09 (Phase 0.2-0.4 implementation done; on
`mg/exp031-perch-kd-distillation` branch; ready for SLURM fast-dev-run
+ Phase 1 LB).

---

## Pick up here (next session)

**Phase 0 code is in.** Mixup-interaction question from last session
resolved as **Option B with the 1× teacher trick**: `_apply_mixup` uses
`indices = torch.randperm(B)` to shuffle within the batch; Perch is
per-sample (no cross-batch coupling), so `perch(x[indices]) ==
perch(x)[indices]`. Teacher target under mixup composes from one
teacher forward — no 2× cost. See `proposal.md` § Mixup interaction.

Train/val split question resolved as **Option A (no change)** after
discriminator analysis: the current 4-single-sample-species
duplicate-into-train+val is val/auroc reporting bias only (~0.002 macro),
not train-weight contamination. Defer methodology cleanup to a dedicated
post-031 experiment that addresses the actual load-bearing weaknesses
(per-site stratification, fold-balance asymmetry).

**Next concrete steps** (compute, not code):

1. Run `python user/murilo/experiments/031-perch-kd-distillation/test_kd_gradient_flow.py`
   — synthetic-batch gradient-flow check, no GPU needed. Verifies all
   three submodules (encoder, sed_head, distill_head) get gradient when
   KD is on, and that `perch(x[perm]) == perch(x)[perm]` for the stub.
2. `bash sbatch/run.sh configs/v5j_kd.yaml --fast-dev-run` — real ONNX
   path on a GPU node, verify dist_loss/cls_loss logged separately,
   no NaN.
3. `bash sbatch/run.sh configs/v5j_kd.yaml` — full Phase 1 training
   (~80-120 min on rtx6000). Halt watch: `dist_loss > 0.01` by epoch 5.
4. ONNX export → Kaggle weights upload → kernel push → LB submit.
5. Apply success matrix vs v5j_control_v2 = 0.896 (CONFIRM ≥ 0.910).

---

## Where exp 031 is

**Status**: Phase 0 implementation complete; ready for fast-dev-run + Phase 1.
Proposal: `user/murilo/experiments/031-perch-kd-distillation/proposal.md`.

| Phase | What | Status |
|---|---|---|
| 0.0 | Mel pipeline equivalence diagnostic | ✅ KEEP NNAUDIO + hop=313 (cosine sim 0.987 on STFT-only; libraries agree) |
| 0.1 | Vendor + verify Perch v2 ONNX teacher | ✅ All 6 checks pass; CUDA 17 ms/batch=4 on rtx6000 |
| 0.2 | `DistillHead` + `PerchTeacher` + KD branch in `sed_finetune.py` | ✅ DONE — Option B with 1× teacher trick |
| 0.3 | Config `configs/v5j_kd.yaml` + train.py + sbatch wrappers | ✅ DONE |
| 0.4 | Fast-dev-run validation | ⏭️ `test_kd_gradient_flow.py` written; SLURM fast-dev-run pending |
| 1 | Single-fold KD training + LB submission | pending 0.4 |

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

- **Current branch**: `mg/exp031-perch-kd-distillation`. Phase 0.2-0.4
  implementation committed; ahead of main by 1 commit ("Implement Perch v2 KD
  distillation branch (Phase 0.2-0.4)"). Pushed to origin.
- **Latest commit on main**: `24d813a` — "Scaffold exp 031: Perch v2
  knowledge distillation (Phase 0 setup) (#46)" (squash-merge of PR #46).
- **No open PRs.** Next PR opens after Phase 1 LB receipt (or sooner if
  we want a code-only review pass before paying compute).
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

---

## File index for active work

| File | Purpose |
|---|---|
| `user/murilo/experiments/031-perch-kd-distillation/` | **Active experiment** |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | Where Phase 0.2 changes will land |
| `user/murilo/analysis/mel_pipeline_comparison.py` | Phase 0.0 mel equivalence script |
| `user/murilo/analysis/verify_perch_teacher.py` | Phase 0.1 ONNX schema/IO verification |
| `user/murilo/analysis/sbatch/verify_perch_teacher.sh` | sbatch wrapper (`bash` self-submits via sbatch) |
| `vendor/kaggle/datasets/perch-v2-no-dft-onnx/` | Vendored Perch ONNX (gitignored; re-fetch instructions in `vendor/kaggle/datasets/README.md`) |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/` | Tucker's notebook (vendored) |
| `user/murilo/MODEL-RUNS.md` | All model scores |
| `user/murilo/FINDINGS.md` | Cross-experiment history |
