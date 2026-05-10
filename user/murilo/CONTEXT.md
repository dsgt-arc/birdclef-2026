# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-10 PM (Phase 1 LB done; verdict REJECT;
**codebase regression discovered** — can't reproduce v5j-original 0.906
from scratch. Ground-truth worktree training job 8330980 is RUNNING
on node `atl1-1-02-004-35-0`. Resume when it finishes.)

---

## Pick up here (next session)

**SLURM job 8330980 is running** (or just finished). It retrains
v5j_control using the v5j-era source code from a worktree at commit
`e420633`. This is the **ground-truth test** for whether the
codebase regression that landed today's training at LB 0.897 (vs
v5j-original 0.906) is in the codebase itself.

```bash
squeue -u $USER
ls -lt /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-v5j-era/user/murilo/experiments/019-perch-guided-sed/logs/
```

Expected walltime: ~40 min on rtx6000. Output checkpoints at:
`/storage/project/.../experiments/017-mel-overhaul/checkpoints/2026-exp019-v5j-diag-eracode-may10/`

**When it finishes**, run the **hash-compare first** (no LB slot needed):

```bash
source ~/clef/birdclef-2026/.venv/bin/activate
python -c "
import torch, hashlib
def h(p):
    s = torch.load(p, map_location='cpu', weights_only=False)['state_dict']
    keys = sorted(k for k in s if not any(m in k for m in ['cmap','auroc','f1_macro','f1_micro','top1','top5','_val_sc']))
    return hashlib.sha256(b''.join(s[k].cpu().numpy().tobytes() for k in keys)).hexdigest()
print('v5j-original (Mar 31):', h('/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt'))
print('diagnostic (May 10): ', h('/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-exp019-v5j-diag-eracode-may10/best-epochepoch=19.ckpt'))
"
```

Three possible outcomes:

| Hash result | Interpretation | Next step |
|---|---|---|
| **Identical** | Training is fully deterministic across May 10 vs Mar 31 of v5j-era code. **HEAD codebase IS the cause** of the 0.906→0.897 regression. LB will be 0.906. | **Bisect the 8 PRs** (exp 021/022/025/026/027/030/031) to localize the offending commit. No LB submission needed. |
| **Different but val/auroc matches** | Training is non-deterministic across runs. | Submit ep 19 to LB. If 0.906, env-stable; if 0.897, env-shifted; intermediate suggests partial cause. |
| **Different and val/auroc shifts** | Codebase + env both changed. | Deeper investigation: bisect AND environment audit. |

Read the full diagnostic README at
`user/murilo/analysis/diagnostics/v5j-codebase-drift/README.md`.

---

## Exp 031 verdict: REJECT

Phase 1 KD as a single-variable add on v5j HGNet+LSEHead **did not transfer**.

| Submission | Checkpoint | LB | vs same-codebase control |
|---|---|---:|---:|
| v5j_kd (exp 031) | ep 17 best-by-val | **0.890** | −0.006 |
| v5j_control (exp 031) | ep 15 best-by-val | **0.896** (inherited from exp 030) | — |
| v5j_kd_last | ep 19 (save_top_k=-1 rerun) | **0.891** | −0.006 |
| v5j_control_last | ep 19 (save_top_k=-1 rerun) | **0.897** | — |

Both ckpt-selection regimes give the same within-codebase KD delta: **−0.006**.
Inside the ±0.013 seed band but systematic and on the wrong side. KD doesn't help on this stack.

### Why REJECT is unambiguous despite the codebase regression

The KD-vs-control comparisons happen **within the same codebase** (today's HEAD).
Whatever drift broke 0.906 → 0.897 affects BOTH v5j_kd and v5j_control identically.
The within-codebase delta (−0.006 at both best and last ckpts) is uncontaminated.

### Post-mortem (4 ranked candidate causes)

1. **LSEHead is the wrong consumer of Perch features** (highest likelihood) — LSE pools by magnitude; Perch encodes spatial localization that AttHead would consume directly.
2. **20-epoch budget too short** — KD trajectory still narrowing gap to control through ep 17 best.
3. **α_distill = 1.0 miscalibrated** for our codebase's mel + feature dim.
4. **KD lift may require Tucker's full stack** (AttHead + dual-supervision + hard-max mixup + SpecAugment + focal×SS + 90/10 batch composition + 25-ep cosine).

Phase 2 (ownership + 4-fold) HALTED. Phase 3+ candidates documented in
`experiments/031-perch-kd-distillation/results.md` and `tasks.md`.

---

## BIG FINDING: v5j codebase regression (priority #1)

Discovered 2026-05-10 PM via the best-vs-last LB diagnostic. **The training pipeline produces different (worse) weights today than it did on Mar 31, with identical config + seed + data.**

| Run | LB | val/auroc | val/cmap | val/f1_macro |
|---|---:|---:|---:|---:|
| v5j-original (Mar 31, exp 019) | **0.906** | 0.8737 (ep 19) | **0.6429** | **0.5515** |
| v5j-resubmit (exp 019 ep 19 ckpt → today's inference) | **0.906** | — | — | — |
| v5j_control (today, ep 15 best) | 0.896 | 0.8747 | 0.6268 | 0.5270 |
| v5j_control_last (today, ep 19) | 0.897 | 0.8728 | 0.6268 | 0.5270 |

**Inference is verified clean** (v5j-resubmit 0.906 exactly reproduces original).
**Training has drifted**: val/auroc parity, but cmap −0.016 and f1_macro −0.025
relative to v5j-original. Per-epoch val/auroc trajectory **diverges at epoch 0**
(orig 0.50607 vs today 0.50880).

### Diagnostic findings: 8 STATIC CAUSES RULED OUT

All bit-identical between v5j-era (`e420633`) code and HEAD code under same venv:

- ✅ Init-state hash (encoder + sed_head + module)
- ✅ Pretrained backbone weights (across timm 1.0.25/1.0.26/1.0.27 + hf-hub 1.6.0/1.8.0/1.14.0)
- ✅ Forward+backward gradient hash on synthetic input
- ✅ Train/val split filenames + ORDER
- ✅ `configure_optimizers` config
- ✅ All ML/CUDA libs in actual training (torch 2.5.1+cu121, cuDNN 9.1.0.70, cublas 12.1.3.1, nccl 2.21.5, driver 575.57.08)
- ✅ Python 3.11.11 in both training runs
- ✅ Data files (train.csv mtime 2026-03-11, unchanged)

The drift requires **end-to-end GPU training** to manifest. Candidates:
multi-worker DataLoader RNG state, GPU fp non-determinism, GradScaler
state evolution, wheel binary drift at unchanged version numbers.

**Decisive next step**: the worktree training job 8330980 (currently
running) retrains v5j_control using v5j-era source code. Hash-compare
ep 19 vs v5j-original ep 19 → ground-truth answer on whether codebase
is to blame.

Full forensics: `analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md`
and `analysis/notes/2026-05-10-v5j-drift-diagnostics-findings.md`.

### Implications

- **0.906 is the real v5j baseline** (proven by v5j-resubmit). Stay with it; don't lower the bar to 0.897.
- **Within-codebase deltas are valid**; cross-codebase deltas (today vs Mar 31 era) are contaminated by ~0.009 of regression.
- **Reproducibility is the highest-priority fix** before any further training-recipe experiments. Until resolved, we're 0.009 in the hole before starting.
- Several historical "REJECT/WEAK" verdicts comparing today's runs against the 0.906 baseline overstated negative deltas by ~0.009.

---

## Where exp 031 is

**Status**: Phase 1 COMPLETE → REJECT verdict. Codebase regression
discovered. Ground-truth diagnostic in flight.

| Phase | What | Status |
|---|---|---|
| 0.0–0.4 | Phase 0 prerequisites | ✅ DONE |
| 1.1 | v5j_kd + v5j_control SLURM training | ✅ DONE (jobs 8318953, 8319084) |
| 1.2 | ONNX export + Kaggle pipeline scaffolding (v5j_kd) | ✅ DONE (commit `2241093`) |
| 1.3 | v5j_kd LB submission | ✅ DONE — LB 0.890 → **REJECT** |
| 1.4a | save_top_k=-1 retrains (v5j_kd_resave, v5j_control_resave) | ✅ DONE — determinism verified bit-identical |
| 1.4b | v5j_kd_last + v5j_control_last pipeline scaffolding | ✅ DONE (commit `b884fc3`) |
| 1.4c | Best-vs-last LB submissions | ✅ DONE — control 0.897, KD 0.891 (REJECT confirmed) |
| 1.4d | Codebase regression investigation | ✅ 8 static causes ruled out |
| 1.4e | Worktree training (ground-truth test) | ⏳ RUNNING (job 8330980, node `atl1-1-02-004-35-0`) |

---

## Read these files first

Top three are read fully; others are reference / skim.

| File | Why |
|---|---|
| `user/murilo/experiments/031-perch-kd-distillation/results.md` | Full Phase 1 results + REJECT verdict + post-mortem |
| `user/murilo/analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md` | Initial drift analysis + ranked hypotheses |
| `user/murilo/analysis/notes/2026-05-10-v5j-drift-diagnostics-findings.md` | 8 static causes ruled out; worktree-training plan |
| `user/murilo/analysis/diagnostics/v5j-codebase-drift/README.md` | Worktree training experiment: how to run + interpret |
| `user/murilo/experiments/031-perch-kd-distillation/tasks.md` | Phase-by-phase task list with completion status |
| `user/murilo/experiments/031-perch-kd-distillation/proposal.md` | KD experiment proposal (historical, success matrix) |
| `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Tucker recipe code audit (KD shape, AttHead aggregation, dual-supervision blend) |

Backlog (skim only if older context needed):
- `analysis/notes/2026-04-28-next-experiment-direction.md` — pre-exp 031 strategic synthesis
- `analysis/notes/2026-04-26-mixup-implementation-audit.md` — v5j vs Sydorskyi/Tucker
- `analysis/notes/2026-04-25-per-frame-attention-analysis.md` — v5j is structurally honest on hard crops

---

## Best LB (UNCHANGED — 0.906 still real)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| v5j single (exp 019) | **0.906** | Best single SED. Proven achievable via v5j-resubmit reproduction. Today's pipeline gets 0.897 instead — that's a REGRESSION, not a new normal. |
| effnetv2b0_sydorskyi (exp 030) | 0.901 | Best non-v5j single |
| v5j_control_v2 (exp 030) | 0.896 | Single-fold seed=42 same recipe; the same regression that makes today's runs land at 0.896-0.897 |
| **v5j_kd (exp 031)** | **0.890** | REJECTED — KD doesn't transfer as single-variable add |

**Target**: 0.948 (gap 0.019 from current best ensemble, 0.014 from best
single SED). Reproducing 0.906 from scratch is a precondition.

---

## Key patterns and constraints

### Lightning save_top_k pitfall (NEW — feedback memory)
`save_last: true` with `save_top_k: 1` makes `last.ckpt` a name-alias of
the best checkpoint, NOT the literal final epoch. Use `save_top_k: -1` if
best-vs-last analysis may matter (which is essentially always for v5j-family
single-fold runs). Documented in
`feedback_save_top_k_does_not_preserve_last.md`.

### Codebase regression: 0.906 → 0.897 on v5j retraining (NEW)
Today's training pipeline can't reproduce v5j-original 0.906 from scratch.
Same config, same seed, same data, same most-libs → different weights →
worse LB. 8 static diagnostics ruled out all testable causes. Drift
requires end-to-end GPU training to manifest. Worktree experiment (job
8330980) is the decisive test. Until resolved, single-fold experiments
are comparing against a known-broken baseline.

### Tucker audit findings (load-bearing)
1. **Tucker's KD = embedding-space MSE distillation with `feat.detach()` on SED branch** (category #3 of "Perch as teacher")
2. **Tucker's AttHead aggregates in logit space** (`sum(softmax(tanh(att)) * frame_logits)`); ours aggregates in probability space with `sigmoid → average → inverse-sigmoid` roundtrip. Likely root cause of our 7 AttHead regressions.
3. **Dual-supervision blend at inference** (`0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)`) is the structural fix to v6k's uniform attention.

### "Perch as teacher" — three categories
| # | What | Status |
|---|---|---|
| 1 | PL on UNlabeled data | exp 014/015/025 → dead |
| 2 | KD on labeled (KLD on logits) | NEVER TRIED (likely EliKal's 0.909) |
| 3 | KD on labeled (MSE on embedding + stop-grad) | **REJECTED on v5j+LSEHead+20ep (exp 031, LB 0.890 vs 0.896 control)** |

### What moves LB (revised, post-031)
Backbone change (+0.037), mel normalization (+0.034), PP/TTA/smoothing
(+0.032), SED architecture (+0.028), Perch+SED ensemble fusion (+0.004).
**Common thread**: change *what the model sees* or *how predictions are
combined* — never *how the model trains*. Exp 031's KD addition is a
training-side change; consistent with the failure pattern.

### What's dead (don't redo)
- CE loss (7 attempts, -0.015 to -0.409)
- Interpolated mixup variants (3 attempts, -0.019 to -0.031)
- PL on unlabeled (exp 014/015/025) — categories #2/#3 above are NOT this
- 2025-only pretrain at our scale (exp 029, -0.021)
- 7 AttHead variants — but aggregation-domain bug means our test was structurally broken
- Background noise mixing (paused before run)
- Multi-context heads (-0.010 to -0.075)
- **NEW**: Tucker-style embedding KD on v5j+LSEHead (exp 031, -0.006). Still untested: KD on bug-fixed AttHead or KD on EffNet.

### Single-fold seed band: ±0.013 on HGNet
v5j_control_v2 (seed=42) = 0.896, v5j_control_seed43 = 0.883.

### v5j is structurally honest on hard crops
GT-class attention entropy rises and logit std collapses on low-SNR crops.
Mechanistically explains 22+ recipe-tweak failures.

### Phase 3+ candidates (if v5j codebase regression resolved)
1. KD + bug-fixed AttHead (logit-space aggregation) — highest priority
2. KD with 25-30 epoch schedule
3. α_distill sweep (0.1, 0.5, 2.0)
4. Full Tucker replication (EfficientNet-B0 NS-JFT + AttHead + dual-supervision + hard mixup + SpecAugment + focal×SS)
5. KLD-on-logits (Perch category #2, EliKal's recipe — never tried)

### `head_lr` quirk
Silently ignored under `scheduler_type: one_cycle`. Only matters if we revisit split LR.

---

## Branch / git state

- **Current branch**: `mg/exp031-perch-kd-distillation`. **9 commits ahead** of main, all pushed to origin:
  - `c6aaadf` — Scaffold v5j-era worktree training experiment (drift root-cause diagnostic)
  - `65fc467` — v5j drift diagnostics: 8 static causes RULED OUT
  - `dae4abe` — v5j_kd_last LB 0.891; correct codebase-regression framing
  - `3e1cb11` — Comprehensive v5j codebase drift analysis
  - `773280b` — Exp 031 best-vs-last LB: codebase reproduction FAILED
  - `df6a779` — Add v5j_control_resave training metrics
  - `c101906` — Add v5j_kd_resave training metrics
  - `b884fc3` — Wire exp 031 best-vs-last LB diagnostic through submission pipeline
  - `b1aa7e7` — Exp 031 Phase 1 REJECT verdict + best-vs-last retrain scaffolding
- **Latest commit on main**: `24d813a` (squash-merge of PR #46, Phase 0 setup)
- **No open PRs.** Next PR opens when worktree experiment resolves drift.
- **Worktrees**:
  - main: `/storage/project/.../mgustineli3/birdclef-2026/` (branch `mg/exp031-perch-kd-distillation`)
  - **v5j-era diag**: `/storage/project/.../mgustineli3/birdclef-v5j-era/` (detached HEAD at `e420633`). Remove with `git worktree remove` after the experiment.

---

## Compressed timeline

| Date | What | Outcome |
|---|---|---|
| 04-23 | Built model attention explorer | Captured Salman 0.937 + Boredom 0.950+ |
| 04-25 | Audio validation + per-frame analysis | **v5j is structurally honest** |
| 04-26 | Sydorskyi audit + D1 + mixup audit + exp 030 scaffold | PR #42 merged |
| 04-27 | exp 030 v2 + v5j-resubmit | v5j_control_v2 LB 0.896 surfaces ckpt-selection issue |
| 04-28 | v5j-resubmit LB 0.906 + seed=43 retrain (LB 0.883) | exp 030 closed; PL vs KD distinction |
| 05-09 | Tucker audit + exp 031 Phase 0 | PR #46 merged; Phase 0.2-0.4 implemented |
| 05-10 AM | Phase 1 SLURM (v5j_kd + v5j_control) | Both train cleanly; bit-identical to exp 030 v5j_control_v2 |
| 05-10 PM | v5j_kd LB 0.890 = REJECT; best-vs-last diagnostic launched | save_top_k=-1 retrains kicked off |
| 05-10 PM | v5j_kd_resave + v5j_control_resave done; determinism confirmed | save_last pitfall codified as feedback memory |
| 05-10 PM | v5j_control_last LB 0.897, v5j_kd_last LB 0.891 | **CODEBASE REGRESSION DISCOVERED**: can't reproduce 0.906 from scratch |
| 05-10 PM | 8 static diagnostics — ALL ruled out causes | Init, math, libs, CUDA, Python, data, optimizer, splits all bit-identical |
| 05-10 PM | Worktree training scaffolded + submitted (job 8330980) | Ground-truth test running on node `atl1-1-02-004-35-0` (same as v5j-original!) |

---

## File index for active work

| File | Purpose |
|---|---|
| `user/murilo/experiments/031-perch-kd-distillation/results.md` | **Phase 1 results + REJECT verdict** |
| `user/murilo/experiments/031-perch-kd-distillation/tasks.md` | Full task list (1.4e for worktree experiment) |
| `user/murilo/analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md` | Initial drift analysis + 7 ranked hypotheses |
| `user/murilo/analysis/notes/2026-05-10-v5j-drift-diagnostics-findings.md` | **8 static diagnostics — all RULED OUT** |
| `user/murilo/analysis/diagnostics/v5j-codebase-drift/README.md` | Worktree experiment: run + interpret |
| `user/murilo/analysis/diagnostics/v5j-codebase-drift/hgnet_b0_sed_v5j_diag_eracode.yaml` | Diagnostic config (save_top_k=-1) |
| `/storage/project/.../mgustineli3/birdclef-v5j-era/` | **v5j-era worktree** (detached HEAD `e420633`) |
| `/storage/project/.../mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/` | v5j-original ep 19 ckpt (LB 0.906 reference) |
| `/storage/project/.../mgustineli3/experiments/031-perch-kd-distillation/checkpoints/` | All exp 031 ckpts (v5j_kd, v5j_control, both resaves) |
| `user/murilo/experiments/031-perch-kd-distillation/artifacts/training/` | metrics.json for all 4 exp 031 runs |
| `user/murilo/experiments/010-submission-pipeline/` | Kaggle pipeline (kernel-metadata for 4 v5j models: kd, kd-last, control-last; v5j_control inherited via offline match) |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | DistillHead, PerchTeacher, KD branch in training_step |
| `user/murilo/MODEL-RUNS.md` | All model scores |
| `user/murilo/FINDINGS.md` | Cross-experiment history |
