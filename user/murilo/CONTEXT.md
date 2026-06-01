# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first to resume;
follow the linked files for full detail.

**Last updated**: 2026-05-31 (late). **Active experiment: exp 038 — broad Neotropical XC
pretraining (EffNetV2-S) → Stage-2 fine-tune.** Stages 0/1/2 ALL DONE. **Submission pipeline BUILT,
canary bug FIXED, and 4 models SUBMITTED — LB scores are IN (one 5-fold timed out).** Branch
`mg/038-xc-pretraining` (pushed, open — NOT merged). exp 037 CLOSED/REJECT (merged, PR #57).

**Competition deadline**: 2026-06-03 (~2–3 days). LB rate-limited (~5/day).

**▶ NEXT SESSION = REVIEW RESULTS + DISCUSS NEXT STEPS. Make NO decisions before discussing with Murilo.**
The exp-038 LB results are all in (below). The single next action is a **joint review + decision** on the
open questions in "Discuss next" — do not act/submit/merge/wire anything until agreed.

---

## ✅ This session's outcome — exp 038 Stage-2 LB results (the arbiter)

All 4 models exported + uploaded + pushed + submitted; every scored kernel printed the
`Species mapping: 234 mapped` canary ([[feedback_species_mapping_canary]]) and ran clean.

| Model | LB | read |
|---|---:|---|
| `finetune-cell-a` (single, mixup-reg backbone) | **0.907** | **parity** w/ 0.906 KD (+0.001, within ~±0.007 noise); **+0.025–0.029 over the ~0.88 XC-backbone ceiling** |
| `finetune-cell-c` (single, no-mixup backbone) | **0.901** | A>C by 0.006 → **mixup-reg pretrain transfers better** |
| `finetune-cell-c-5fold` (logit fold-mean) | **0.911** | +0.010 over its single fold; **−0.004 below the 0.915 best ensemble**; below 0.929 |
| `finetune-cell-a-5fold` (logit fold-mean) | **TIMED OUT** | exceeded the 90-min CPU budget — VM variance at the ~73/90-min margin (same arch as C-5fold, NOT a bug) |

**Two headline findings:**
1. **The ~0.88 XC-backbone ceiling (exp 034/036) is REFUTED.** Our *own* broad-Neotropical supervised XC
   pretrain, fine-tuned under the exact 0.906 KD-SED regime, reaches **parity with in-domain ImageNet+KD**
   (A 0.907 ≈ 0.906). The prior "XC-pretrain caps at 0.88" read was about the *transfer recipe*, not XC
   pretraining itself. This is the scientific win + working-note result.
2. **A > C (single-fold) resolves the mixup-transfer question** offline called a tie — mixup-regularized
   pretrain transfers better to the focal→soundscape shift. (Offline *fold0* even anti-ranked it C>A — the
   usual [[feedback_offline_metrics_unreliable]] / [[project_cv_measurement_failure]].)

**Caveat — NOT a plateau-break.** Best exp-038 = C-5fold **0.911 ≤ the 0.906/0.915/0.929 stack** (exactly
the honest ceiling the proposal set). Value = backbone-diverse ensemble member + paper, as scoped.
**A-vs-C at the *ensemble* level is UNRESOLVED** (A-5fold has no score; A led C single-fold, so an
A-5fold ≥ 0.911 is plausible if re-run / 4-fold-trimmed).

Today's LB pulls used: 4 (cell-a, cell-c, cell-c-5fold scored; cell-a-5fold timed out).

---

## ▶ Discuss next session (decide TOGETHER — no decisions made this session)

Deadline 2026-06-03, LB ~5/day. Open questions, in rough priority:

1. **A-5fold timed out — retry vs trim?** Retry as-is is a coin-flip at the ~73/90-min margin; a **4-fold
   mean** (drop weakest A fold, ~58 min) is the safe way to get a Cell-A ensemble score + resolve A-vs-C at
   the ensemble level. (4-fold export/blend NOT wired.) Worth a pull at all, given C-5fold 0.911 < 0.915?
2. **Blend C-5fold (0.911) into the banked 0.929?** It's −0.018 below 0.929 alone but backbone-diverse
   (EffNetV2-S vs the 0.929 stack) → a prob-space blend *might* help; keep iff it beats 0.929 on LB. **Blend
   kernel NOT wired** — only build it if we decide it's worth a pull.
3. **Final 2 submission picks.** 0.929 exp-023 ensemble is banked → **select it as final pick #1** (safe).
   Pick #2 = best exp-038 / blend result, TBD.
4. **Merge + working-note.** `mg/038-xc-pretraining` is open (NOT merged). When to merge (`--squash`,
   [[feedback_squash_merge]]) + whether to write the ceiling-refute working-note now.

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Overall best → **select as final pick #1** (already scored) |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| distilled-sed-kd-5fold (exp 032) | 0.915 | Best Tucker-SED ensemble |
| **exp 038 `finetune-cell-c-5fold`** | **0.911** | best exp-038; −0.004 vs 0.915; backbone-diverse member |
| distilled-sed-kd single-fold (exp 032) | 0.906 | Best single (ImageNet+KD) — exp-038 single-fold target |
| **exp 038 `finetune-cell-a`** (single) | **0.907** | parity w/ 0.906; **refutes the ~0.88 XC ceiling** |
| **exp 038 `finetune-cell-c`** (single) | **0.901** | A>C → mixup-reg pretrain transfers better |
| exp 034 / 036 XC-backbone (+PP) | 0.878–0.882 | the ceiling exp-038 just refuted |
| exp 038 `finetune-cell-a-5fold` | TIMED OUT | 90-min budget; retry/4-fold-trim = a discussion item |

---

## How to resume after disconnect
1. Read this top-to-bottom, then exp 038 `results.md` (§Stage 2 — LB results) + `tasks.md` (§Stage 2).
2. `git log --oneline -10` on `mg/038-xc-pretraining` (pushed, open — NOT merged). This session's commits:
   `481db57` (single-fold LB) + the 5-fold-LB/context commit on top. exp 037/036/034/032 CLOSED on main.
3. **Start with a results review, then discuss the 4 open questions in "Discuss next" — make NO decisions
   or submissions until agreed with Murilo.** (Explicit instruction from the prior session.)
4. **[[feedback_user_runs_commands]]**: give Murilo commands; don't run SLURM/Kaggle yourself. Local
   pytest + `save_metrics.py` + local CPU export smoke-tests + analysis are fine. Interactive venv
   `/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv` ([[feedback_use_uv_venv]]).
5. Venv policy: NEVER create a venv under home/project; SLURM jobs build on `$TMPDIR` (CLAUDE.md).
   Compute NOT the bottleneck ([[feedback_pace_compute_not_bottleneck]]); Kaggle LB IS rate-limited
   (~5/day). `--squash` for PR merges ([[feedback_squash_merge]]). Pre-commit ruff-format may reformat a
   staged file → re-`git add` + re-commit if a commit "fails" on the format hook.
6. **Runtime budget (load-bearing for any 5-fold resubmit):** 90-min CPU limit, ~600 hidden soundscapes.
   EffNetV2-S 5-fold ≈ 7.3 s/file ≈ ~73 min est = thin margin → A-5fold timed out. A **4-fold** mean ≈
   58 min is the safe fallback. Single-fold ≈ 15 min (huge headroom). (Refs: `docs/references/
   birdclef-2026-competition.md`, exp-021 v5j-4fold ~50 min.)
7. **GPU availability**: a100/h200 often full; h100 free nodes often 0 idle CPUs (pends). **rtx6000
   (24 GB)** is the reliable parallel option.

**Credentials**: `user/murilo/.env` (gitignored; HF_TOKEN, W&B). The HF token was pasted in chat in a
prior session — Murilo should rotate `hf-token-pace`.
