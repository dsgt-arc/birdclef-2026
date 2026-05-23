# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-22 (strategic direction set after processing
3 Kaggle discussion threads + 2 competitor notebooks. Decision: Track A
= full Tucker replication first, Track B = own-teacher PL second.
See `notes/2026-05-22-strategic-direction-tracks-A-B.md` for full
discussion + rationale.)

**Competition deadline**: 2026-06-03 (~12 days from this update).

---

## Pick up here (next session) — UPDATED 2026-05-22

**Strategic direction set.** After processing the May-21 Kaggle discussion
state (3 thread captures + competitor notebook vendoring + extensive
discussion), the next experiment scope has expanded:

- **Track A (PRIMARY, high-confidence)**: full Tucker distilled-SED replication.
  Known ceiling 0.917 single / 0.93+ with 5-fold + PP. Full code reference
  at `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`. Exp 031
  already has 70% (DistillHead, PerchTeacher, KD branch). Remaining work
  is well-scoped (head swap, dual-sup, mixup, SpecAugment, schedule, 5-fold).
- **Track A stretch (parallel/sequential)**: Sydorskyi-yoink + own head + 2026
  fine-tune (Ali Ozan path). Higher ceiling (0.941) but methodology-only
  (no code reference); start day 4-5 if Tucker pipeline is stable.
- **Track B (final layer)**: own-teacher PL on unlabeled SS. +0.007 to +0.013
  LB lift on top of Track A. Antoine's simple protocol (no confidence filter).

**The exp 032 scope expands** from the previously-scoped "single-variable
AttHead+KD-on-HGNet" test to the full Tucker stack replication. See
`notes/2026-05-22-strategic-direction-tracks-A-B.md` § Concrete next-session
starting point for the full implementation order.

**Read these first when resuming:**
1. `notes/2026-05-22-strategic-direction-tracks-A-B.md` — strategic decision + code-reference audit
2. `notes/kaggle-discussion-best-single-model-lb.md` — full threaded competitor discussion (2026-05-21 capture)
3. `notes/kaggle-discussion-non-perch-limit.md` — Antoine's full PL/XC-pretrain trajectory
4. `notes/2026-05-21-nina2025-eos6-protossm-stack-analysis.md` — what we are NOT chasing as lead path
5. `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` — Tucker's code audit (unchanged from before)
6. `experiments/031-perch-kd-distillation/results.md` — exp 031 REJECT post-mortem (still valid)

### Scaffold steps (in order)

> **NOTE 2026-05-22**: scope expanded from single-variable AttHead+KD test
> to full Tucker recipe replication (Track A primary). Steps 1-5 below are
> still the right starting commands; configs/training-loop differ from the
> original scope. See `notes/2026-05-22-strategic-direction-tracks-A-B.md`
> § Concrete next-session starting point for the **full Tucker
> replication implementation order** (10 sub-steps including backbone
> swap, dual-supervision, hard-max waveform mixup, SpecAugment, 25 ep
> cosine, 90/10 batch composition, focal×SS mixup, 5-fold, site/hour
> priors + Gaussian smoothing).

1. **Create new experiment directory**:
   ```bash
   cp -r user/murilo/experiments/031-perch-kd-distillation user/murilo/experiments/032-kd-atthead
   cd user/murilo/experiments/032-kd-atthead
   # clean out exp 031 artifacts (logs, outputs, artifacts/training/, checkpoints config)
   rm -rf logs/ outputs/ artifacts/training/
   # update pyproject.toml: rename project, inherit uv.lock from exp 031 (or regenerate)
   ```

2. **Port Tucker's logit-space AttHead** to
   `user/murilo/src/birdclef/heads/sed_head.py` as a new class
   `AttHeadLogitSpace`. Reference:
   - Tucker code audit: `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`
   - Tucker's notebook (verbatim): `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` (cells 10-14)
   - Our broken version: `feedback_atthead_aggregation_domain.md`

   Key features:
   - `GeMFreqPool(p_init=3.0)` with `p.clamp(min=1.0)`
   - Dense: `Dropout(0.25) → Linear(in_channels, 512) → ReLU(inplace=True) → Dropout(0.5)`
   - Two `Conv1d(512, num_classes, k=1, bias=True)`: one for attention, one for class logits
   - **xavier_uniform_ init on both convs**, bias=0
   - **Aggregation in LOGIT SPACE**:
     `clip_logits = sum(softmax(tanh(att(x)), dim=-1) * cls(x), dim=time)`

3. **Two paired configs** in `032-kd-atthead/configs/`:
   - `v5j_atthead_control.yaml` — HGNet + AttHeadLogitSpace, KD off, save_top_k: -1
   - `v5j_atthead_kd.yaml` — same as above + KD on (use_perch_distill: true)
   - Both inherit pinned deps via `032-kd-atthead/pyproject.toml`

4. **Update train.py + sbatch wrappers** in `032-kd-atthead/` to recognize
   the new head type. Search for `head_type` dispatch in train.py and add
   `att_head_logit_space` branch that instantiates `AttHeadLogitSpace`.

5. **Submit paired SLURM jobs**:
   ```bash
   cd user/murilo/experiments/032-kd-atthead
   bash sbatch/run.sh configs/v5j_atthead_control.yaml
   bash sbatch/run.sh configs/v5j_atthead_kd.yaml
   ```
   Run on the same SLURM partition (control for hardware variance — both
   will land on similar rtx6000 nodes).

6. **Submit both to LB**, compute paired-control delta, apply matrix:

   | Pattern | Conclusion |
   |---|---|
   | atthead_kd > atthead_control > v5j_control (0.897) | Both AttHead fix and KD contribute |
   | atthead_kd > atthead_control ≈ v5j_control | KD needs AttHead, AttHead alone doesn't help |
   | atthead_kd ≈ atthead_control > v5j_control | AttHead fix helps, KD adds nothing |
   | All three close | Neither helps on HGNet → pivot to EffNet+full-Tucker-stack |

### Reference materials (read before scaffolding)

- `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` — Tucker's recipe, KD lift quantification, training logs
- `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` — Tucker's notebook code
- `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` — code audit of Tucker's notebook
- `user/murilo/experiments/031-perch-kd-distillation/results.md` — Phase 1 verdict + post-mortem
- Memory: `feedback_atthead_aggregation_domain.md` (logit-space vs probability-space)

---

## Exp 031 verdict: REJECT (FINAL — closed 2026-05-11)

Phase 1 KD as a single-variable add on **v5j HGNet+LSEHead+20ep+α=1.0** did
not transfer. Within-codebase paired-control delta: **−0.006** at both
best-by-val and ep 19 checkpoints. Inside ±0.013 single-fold seed band but
systematic; KD slightly hurts on this stack.

### Five LB submissions, final table

| Submission | Code | Node | Ckpt | LB | Δ vs 0.906 | Δ vs paired control |
|---|---|---|---|---:|---:|---:|
| v5j-original (exp 019, Mar 31) | e420633 | `02-004-35-0` | ep 19 | **0.906** | — | — |
| v5j-resubmit (Apr 28) | e420633 ckpt → today's inference | n/a | ep 19 | **0.906** | 0.000 ✅ | — |
| v5j_control_last | HEAD | `02-004-25-0` | ep 19 | **0.897** | −0.009 | — |
| v5j_kd | HEAD | `03-003-19-0` | ep 17 best-by-val | **0.890** | −0.016 | **−0.006 vs 0.896** |
| v5j_kd_last | HEAD | `03-003-19-0` | ep 19 | **0.891** | −0.015 | **−0.006 vs 0.897** |
| **v5j-eracode (drift diag)** | **e420633 (worktree)** | `03-003-19-0` | ep 19 | **0.899** | **−0.007** | — |

### Phase 1 REJECT is unambiguous

Within-codebase paired KD vs control delta = −0.006 at both ckpt regimes.
The hardware/code variance issues (below) don't contaminate this comparison
because v5j_kd and v5j_control trained on the SAME codebase at the SAME
time. Confined to v5j HGNet+LSEHead — Tucker's recipe (EffNet+AttHead+25ep
+dual-supervision) is NOT falsified by this experiment.

### Why KD didn't transfer on v5j+LSEHead (ranked)

1. **LSEHead is the wrong consumer of Perch features** (highest likelihood).
   LSE pools by magnitude; Perch encodes spatial localization that AttHead
   would consume directly. Phase 3.1 tests this.
2. **20-epoch budget too short.** KD trajectory still narrowing gap to
   control through ep 17 best.
3. **α_distill = 1.0 miscalibrated** for our codebase's mel + feature dim.
4. **KD lift may require Tucker's full stack** (AttHead + dual-supervision +
   hard-max mixup + SpecAugment + focal×SS + 90/10 batch + 25-ep cosine).

---

## v5j drift investigation: RESOLVED 2026-05-11

After 5 LB submissions + 8 static diagnostics + worktree training, the
0.906 → 0.897 retrain gap is attributed to:

- **~0.007 LB**: cross-hardware GPU fp non-determinism. v5j-original ran
  on node `02-004-35-0`; today's retrains run on different nodes. Same
  code (verified by v5j-eracode) gets LB 0.899 on a different node. Not
  fixable in code; fundamental to GPU training.
- **~0.002 LB**: code change between e420633 and HEAD. Now pinned via
  uv.lock + version pins in pyproject.toml (commit `9f9ec3d`). Future
  retrains will use timm 1.0.26 / hf-hub <1.10 / etc.

**Decisive signal: val/cmap match.**

| Run | Code | Node | val/cmap |
|---|---|---|---:|
| v5j-original | e420633 | A | **0.6429** |
| v5j-eracode | e420633 | B | **0.6429** ← matches to 4 dp |
| v5j_control_resave | HEAD | C | 0.6268 ← drops 0.016 |

Same code different node = 0 cmap drift. Different code = 0.016 cmap drift
but only 0.002 LB drift (offline-LB anti-correlation).

### Implications going forward

1. **0.906 stays as the real v5j baseline** (proven by v5j-resubmit).
2. **Single-fold v5j has ~0.007 cross-node LB variance** — fundamental,
   not fixable in code. Plus ~0.002 code-side (now pinned).
3. **Paired within-run controls** are the reliable methodology.
   Compare new-method LB to its same-run control's LB, NOT to historical 0.906.
4. **Multi-fold averaging** reduces this variance for any future ensemble.

---

## Where exp 031 is

**Status**: CLOSED (2026-05-11). All tasks complete.

| Phase | Status |
|---|---|
| 0.0–0.4 Phase 0 setup | ✅ DONE |
| 1.1 SLURM training | ✅ DONE |
| 1.2 Kaggle pipeline | ✅ DONE |
| 1.3 LB submission → REJECT | ✅ DONE |
| 1.4 Best-vs-last + drift diagnostics | ✅ DONE |
| 1.4e Worktree training (v5j-eracode) | ✅ DONE — LB 0.899 |
| uv.lock pinning + v5j-era worktree cleanup | ✅ DONE |
| Phase 2 (ownership + 4-fold) | HALTED (REJECT verdict) |
| **Phase 3.1 (KD + bug-fixed AttHead)** | **NEXT — scaffold as exp 032** |

---

## Read these files first

| File | Why |
|---|---|
| `user/murilo/experiments/031-perch-kd-distillation/results.md` | Final exp 031 results + REJECT verdict + Phase 3.1 plan |
| `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` | Tucker's recipe, KD lift quantification, training logs |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` | Tucker's notebook code (cells 10-14 are AttHead + KD) |
| `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Code audit of Tucker's notebook |
| `user/murilo/analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md` | Drift investigation (initial hypotheses) |
| `user/murilo/analysis/notes/2026-05-10-v5j-drift-diagnostics-findings.md` | 8 static diagnostics, all ruled out |
| `user/murilo/analysis/diagnostics/v5j-codebase-drift/README.md` | Worktree experiment + interpretation |
| `user/murilo/experiments/031-perch-kd-distillation/tasks.md` | Full exp 031 task list |

Backlog:
- `analysis/notes/2026-04-28-next-experiment-direction.md` — pre-exp 031 strategic synthesis
- `analysis/notes/2026-04-26-mixup-implementation-audit.md`

---

## Best LB (unchanged)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| v5j single (exp 019) | **0.906** | Best single SED. Real (proven by v5j-resubmit). Single-fold retrains land 0.896-0.899 due to ~0.007 hardware variance + ~0.002 code variance (now pinned). |
| effnetv2b0_sydorskyi (exp 030) | 0.901 | Best non-v5j single |
| **v5j-eracode (exp 031 drift diag)** | **0.899** | Same code as v5j-original, different node, today's env |
| v5j_control_v2 (exp 030) | 0.896 | HEAD-code single-fold retrain |
| v5j_control_last (exp 031, ep 19) | 0.897 | HEAD-code with save_top_k=-1 |
| v5j_kd_last (exp 031, ep 19) | 0.891 | HEAD-code + KD; −0.006 vs paired control |
| **v5j_kd (exp 031, ep 17)** | **0.890** | REJECTED — KD doesn't transfer on HGNet+LSEHead |

**Target**: 0.948 (gap 0.019 from current best ensemble, 0.042 from
v5j-original).

---

## Key patterns and constraints

### Lightning save_top_k pitfall
`save_last: true` with `save_top_k: 1` makes `last.ckpt` a name-alias of
the best checkpoint, NOT the literal final epoch. Use `save_top_k: -1`
when best-vs-last matters. Documented in
`feedback_save_top_k_does_not_preserve_last.md`.

### Single-fold LB variance: ~0.007 cross-hardware + ~0.002 code (NEW 2026-05-11)
Even with bit-identical code/config/seed, single-fold v5j has ~0.007 LB
variance from cross-hardware GPU fp behavior. Plus ~0.002 from code drift
(now pinned via uv.lock). For LB-comparable experiments, use **paired
within-run controls** — read deltas vs same-run control, not vs historical.

### Pinned environment via uv.lock (NEW 2026-05-11)
`exp031/pyproject.toml` pins direct ML deps (torch 2.5.1, pytorch-lightning
2.6.1, torchmetrics 1.9.0, timm 1.0.26, huggingface-hub <1.10, nnAudio
0.3.4). `exp031/uv.lock` snapshots all 112 transitive deps. `.gitignore`
no longer excludes `uv.lock`. New experiments should inherit these pins.

### Tucker audit findings (load-bearing for Phase 3.1)
1. **Tucker's KD = embedding-space MSE distillation with `feat.detach()` on SED branch** (Perch category #3)
2. **Tucker's AttHead aggregates in LOGIT space** (`sum(softmax(tanh(att)) * frame_logits, dim=time)`); ours aggregates in probability space with `sigmoid → average → inverse-sigmoid` roundtrip. Root cause of our 7 AttHead regressions.
3. **Dual-supervision blend at inference** (`0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)`) is the structural fix to v6k's uniform attention.

### "Perch as teacher" — three categories
| # | What | Status |
|---|---|---|
| 1 | PL on UNlabeled data | exp 014/015/025 → dead |
| 2 | KD on labeled (KLD on logits) | NEVER TRIED (likely EliKal's 0.909) |
| 3 | KD on labeled (MSE on embedding + stop-grad) | **REJECTED on v5j+HGNet+LSEHead+20ep (exp 031, −0.006)**; not yet tested on EffNet+AttHead+25ep (Tucker's actual recipe) |

### What moves LB
Backbone change (+0.037), mel normalization (+0.034), PP/TTA/smoothing
(+0.032), SED architecture (+0.028), Perch+SED ensemble fusion (+0.004).
**Common thread**: change *what the model sees* or *how predictions are
combined* — never *how the model trains*. Phase 3.1 (AttHead + KD) is a
borderline case — AttHead is "what the model sees" but KD is "how it trains."

### What's dead (don't redo)
- CE loss (7 attempts, -0.015 to -0.409)
- Interpolated mixup variants (3 attempts, -0.019 to -0.031)
- PL on unlabeled (exp 014/015/025) — categories #2/#3 above are NOT this
- 2025-only pretrain at our scale (exp 029, -0.021)
- 7 AttHead variants — but aggregation-domain bug means test was structurally broken; **Phase 3.1 retests with fixed AttHead**
- Background noise mixing (paused before run)
- Multi-context heads (-0.010 to -0.075)
- Tucker-style embedding KD on v5j+HGNet+LSEHead (exp 031, -0.006). Untested: KD on bug-fixed AttHead, KD on EffNet.

### v5j is structurally honest on hard crops
GT-class attention entropy rises and logit std collapses on low-SNR crops.
Mechanistically explains 22+ recipe-tweak failures.

---

## Branch / git state

- **Current branch**: `mg/exp031-perch-kd-distillation`. **14 commits ahead** of main, all pushed to origin. Most recent:
  - `9f9ec3d` — Pin v5j-era lib versions in exp 031 + commit uv.lock
  - `3ccbb13` — Scaffold same-node v5j-original reproduction attempt (later cancelled)
  - `0541fa9` — Wire v5j-eracode through submission pipeline
  - `8b4de5b` — Update CONTEXT.md with Phase 1 REJECT + codebase regression discovery
  - `c6aaadf` — Scaffold v5j-era worktree training experiment
  - `65fc467` — v5j drift diagnostics: 8 static causes RULED OUT
  - `dae4abe` — v5j_kd_last LB 0.891; correct codebase-regression framing
  - `3e1cb11` — Comprehensive v5j codebase drift analysis
- **Latest commit on main**: `24d813a` (squash-merge of PR #46, Phase 0 setup)
- **No open PRs.** Next PR: close exp 031 as a unit (all 14 commits) when ready.
- **Worktrees**: only main (`birdclef-2026/`). v5j-era worktree cleaned up.

---

## Compressed timeline

| Date | What | Outcome |
|---|---|---|
| 04-23–04-25 | Built model attention explorer, audio validation, per-frame analysis | v5j is structurally honest |
| 04-26 | Sydorskyi audit + D1 + mixup audit + exp 030 scaffold | PR #42 merged |
| 04-27 | exp 030 v2 + v5j-resubmit | v5j_control_v2 LB 0.896 surfaces ckpt-selection issue |
| 04-28 | v5j-resubmit LB 0.906 + seed=43 retrain (LB 0.883) | exp 030 closed; PL vs KD distinction surfaced |
| 05-09 | Tucker audit + exp 031 Phase 0 | PR #46 merged; Phase 0.2-0.4 implemented |
| 05-10 AM | Phase 1 SLURM (v5j_kd + v5j_control) | Both train cleanly; bit-identical to exp 030 v5j_control_v2 |
| 05-10 PM | v5j_kd LB 0.890 = REJECT; best-vs-last diagnostic launched | save_top_k=-1 retrains kicked off |
| 05-10 PM | v5j_kd_resave + v5j_control_resave done; determinism confirmed | save_last pitfall codified as feedback memory |
| 05-10 PM | v5j_control_last LB 0.897, v5j_kd_last LB 0.891 | "Codebase regression" hypothesis formed |
| 05-10 PM | 8 static diagnostics — ALL ruled out causes | Init, math, libs, CUDA, Python, data, optimizer, splits all bit-identical |
| 05-10 PM | Worktree training scaffolded + submitted on node `02-004-35-0` | Same node as v5j-original (lucky scheduling); failed wandb auth → retry |
| 05-10 PM | v5j-eracode retry succeeded on node `03-003-19-0` (different from v5j-original) | val/cmap matches v5j-original to 4 dp! |
| 05-10 PM | v5j-eracode LB = 0.899 (not 0.906) | Hypothesis refined: code accounts for 0.002, hardware for 0.007 |
| 05-10 PM | Same-node test cancelled (user pushback); careful code audit | Confirmed: code change has small LB impact; bigger gap is hardware fp variance |
| 05-11 AM | uv.lock pinned, v5j-era worktree cleaned up | Exp 031 closed |

---

## File index for active work

| File | Purpose |
|---|---|
| `user/murilo/experiments/031-perch-kd-distillation/results.md` | **Phase 1 final results** |
| `user/murilo/experiments/031-perch-kd-distillation/tasks.md` | Full task list (closed) |
| `user/murilo/experiments/031-perch-kd-distillation/pyproject.toml` | **Pinned deps (timm==1.0.26, torch==2.5.1, etc.)** |
| `user/murilo/experiments/031-perch-kd-distillation/uv.lock` | **Full transitive dep snapshot (480 KB)** |
| `user/murilo/MODEL-RUNS.md` | All model scores incl. v5j-eracode |
| `user/murilo/CONTEXT.md` | This file |
| `user/murilo/analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md` | Drift investigation start |
| `user/murilo/analysis/notes/2026-05-10-v5j-drift-diagnostics-findings.md` | 8 static diagnostics ruled out |
| `user/murilo/analysis/diagnostics/v5j-codebase-drift/` | Diag scripts + configs (preserved post-worktree-cleanup) |
| `/storage/project/.../mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/` | v5j-original ep 19 ckpt (LB 0.906 reference) |
| `/storage/project/.../mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-exp019-v5j-diag-eracode-may10/` | v5j-eracode 20-ep ckpts (LB 0.899) |
| `/storage/project/.../mgustineli3/experiments/031-perch-kd-distillation/checkpoints/` | All exp 031 ckpts (v5j_kd, v5j_control, both resaves) |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | DistillHead, PerchTeacher, KD branch in training_step |
| `user/murilo/src/birdclef/heads/sed_head.py` | LSEHead, AttHead (broken), AttBlockV2 — **needs AttHeadLogitSpace for exp 032** |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` | Tucker's notebook (reference for exp 032 AttHead port) |
