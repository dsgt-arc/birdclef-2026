# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first to resume;
follow the linked files for full detail.

**Last updated**: 2026-06-01 (evening session). **Active: exp 038 XC-pretraining endgame — final-pick
selection.** The **`a4fold` blend kernel is SUBMITTED to LB and RUNNING** (check the score FIRST thing).
Branch `mg/038-xc-pretraining` (pushed, open — NOT merged). **Deadline 2026-06-03 (~2 days). LB ~5/day.**

The experiment is essentially DONE pending one score. **No new builds expected** — next session is read the
`a4fold` LB, lock the 2 final picks, merge + working-note. (Only build more if the blend fails AND we want
one last idea — but we're out of cheap levers; see "Closed tonight".)

---

## ▶ NEXT SESSION — START HERE (in order)

1. **Check Kaggle LB for the `a4fold` blend** (`murilogustineli/birdclef-2026-perch-sed-a4fold`, submitted
   tonight, running at wrap-up). It's the **last live shot at beating the banked 0.929** (Perch-mlp +
   backbone-diverse EffNetV2-S A-4fold SED, prob-space w=0.5).
2. **Lock the 2 final Kaggle picks** from the score (pick #1 = **0.929 banked, locked**):
   - blend **> 0.929** → blend becomes pick #1, **0.929 → pick #2** (two >0.916 picks, ideal).
   - **0.916 < blend ≤ 0.929** → blend = **pick #2** (diverse, beats the standalone).
   - blend **≤ 0.916** → keep **A-4fold standalone 0.916** as **pick #2**.
3. **Merge `mg/038-xc-pretraining` (`--squash`, [[feedback_squash_merge]])** + write the **ceiling-refute
   working-note** (our own XC pretrain ≈ ImageNet+KD; ~0.88 ceiling refuted).

[[feedback_user_runs_commands]] — give Murilo Kaggle/SLURM commands; don't run them yourself.

---

## ✅ This session's progress (2026-06-01)

### 1. `finetune-cell-a-4fold` LB = **0.916** (best exp-038) — commit `2889755`
Folds {0,1,3,4} (fold2 dropped via `scripts/analyze_fold_coverage.py`), logit-space mean, reused the 5-fold
ONNX. **+0.005 over C-5fold 0.911 → A>C at the ensemble level too**; ≈ 0.917 best SED (−0.001), +0.001 vs
0.915 Tucker-SED, −0.013 vs 0.929. Scored cleanly → a 4-fold fits the 90-min wall. Lands AT the SED-ensemble
ceiling while being backbone-diverse. (Full read: exp 038 `results.md`.)

### 2. Built 3 endgame variants of the 0.929 kernel — commits `299caf2`, `5f837fe`
All are **self-contained configs of exp 023 `kaggle_kernel/submission_ensemble.py`** (reuse its `push.sh`,
which makes a unique kernel + routes datasets per config). Push from
`user/murilo/experiments/023-perch-sed-ensemble/kaggle_kernel/`: `bash push.sh <config>`.
- **`ensemble_rank_mlp`** → kernel `perch-sed-rank_mlp`: new `blend_space` CFG knob = per-species pct-rank
  then avg (eos8 `_safe_rank_cols`). Default `blend_space="prob"` is **bit-identical to 0.929**.
- **`ensemble_tax_mlp`** → kernel `perch-sed-tax_mlp`: `tax_smoothing` CFG + `build_tax_smoothing_groups`/
  `apply_taxonomy_smoothing` (genus 0.15 → class 0.05 mean-share, 29 genus / 4 class groups). α=0 = no-op.
- **`ensemble_a4fold`** → kernel `perch-sed-a4fold`: swap v5j SED → A-4fold. Ported `load_audio_chunk`/
  `compute_mel_tucker`/`predict_soundscape_tucker_ensemble` from exp 010 → `run_sed_pipeline` routes
  `tucker_mel` to the logit-space path; `push.sh` uses `kernel-metadata-a4fold-ensemble.json` (swaps the
  v5j dataset for `finetune-cell-a-4fold-weights`). **Re-push cmd: `bash push.sh ensemble_a4fold`.**
  - **Runtime fix (this session):** EffNetV2-S SED ran ~8 s/file → blend ~98 min > 90-min wall. **Batched
    the 12 windows into ONE ONNX call/fold** (dynamic batch dim) → Kaggle commit-run confirms **~4.5 s/file**
    (~1.78×) → **~63 min projected, ~27 min margin**. **PARITY: byte-identical** (max abs diff 0.0) to the
    standalone 0.916 — verified end-to-end on the real 4-fold ONNX + 2 soundscapes. So the blend's SED member
    IS the 0.916 signal exactly.

### 3. eos8 PP levers BOTH REGRESSED — commit `0e251f3` (closed the question)
- **`rank_mlp` 0.914 (−0.015 vs 0.929)** — rank-space discarded useful magnitude in our 2-member blend; the
  eos8 rank lift lived in a richer multi-branch stack, not a 2-way Perch+SED. **OUT.**
- **`tax_mlp` 0.886 (−0.043)** — class-level sharing over 162 Aves over-smoothed (range → [0, 0.85]). **OUT.**
- Both eos8 PP levers FAIL to transfer (the project-long "ported PP/recipe tricks rarely survive" pattern,
  [[feedback_offline_metrics_unreliable]] lineage). **rank+tax stack is pointless** (two regressions).

---

## The banked 0.929 (final pick #1, LOCKED) — composition
exp 023 `birdclef-2026-perch-sed-mlp_only` (`submission_ensemble.py`): **Perch ONNX (MLP-only probes,
`proto_weight=0`) + 6-stage PP**, blended **prob-space w=0.5** with **SED v5j-4fold-pp2 (4-fold + TTA)**.
Runtime ~64 min/600 files. The 3 variants above modify THIS kernel via config. Hard constraint: ONE kernel,
90-min CPU, ~600 hidden SS → all members' runtimes ADD UP (Perch ~18 + A-4fold-batched ~45 ≈ 63 min fits).

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only (exp 023) | **0.929** | banked → **final pick #1 (LOCKED)** |
| v5j-4fold-pp2 (exp 021) | 0.917 | best SED ensemble |
| **exp 038 `finetune-cell-a-4fold`** | **0.916** | best exp-038; A>C at ensemble level; **pick-#2 floor** |
| distilled-sed-kd-5fold (exp 032) | 0.915 | best Tucker-SED ensemble |
| `perch-sed-rank_mlp` (eos8 rank-space) | 0.914 | −0.015 vs 0.929 → **REGRESSED, OUT** |
| exp 038 `finetune-cell-c-5fold` | 0.911 | backbone-diverse member |
| exp 038 `finetune-cell-a` (single) | 0.907 | parity w/ 0.906; refutes ~0.88 XC ceiling |
| exp 038 `finetune-cell-c` (single) | 0.901 | A>C → mixup-reg pretrain transfers better |
| `perch-sed-tax_mlp` (eos8 taxonomy) | 0.886 | −0.043 vs 0.929 → **REGRESSED, OUT** |
| exp 034 / 036 XC-backbone (+PP) | 0.878–0.882 | the ceiling exp-038 refuted |
| **`perch-sed-a4fold` blend** | **RUNNING (check first)** | Perch-mlp + A-4fold SED, prob w=0.5, ~63 min; last live shot at >0.929 |

## How to resume after disconnect
1. **Check the `a4fold` blend LB on Kaggle first.** Then read this top-to-bottom + exp 038 `results.md`/`tasks.md`.
2. `git log --oneline -8` on `mg/038-xc-pretraining` (pushed, open — NOT merged). This session's commits:
   `2889755` (A-4fold 0.916), `299caf2` (3 variants built), `5f837fe` (a4fold batching fix), `0e251f3`
   (PP LB results), + this CONTEXT update.
3. **[[feedback_user_runs_commands]]**: give Murilo commands; don't run SLURM/Kaggle yourself. Local
   pytest + CPU export/ONNX smoke-tests + analysis are fine. Interactive venv
   `/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv` ([[feedback_use_uv_venv]]).
4. Venv policy: NEVER create a venv under home/project; SLURM jobs build on `$TMPDIR` (CLAUDE.md).
   Compute NOT the bottleneck ([[feedback_pace_compute_not_bottleneck]]); Kaggle LB IS (~5/day).
   `--squash` for PR merges ([[feedback_squash_merge]]).
5. **Runtime budget (load-bearing):** 90-min CPU wall, ~600 hidden SS. **EffNetV2-S 4-fold SED batched
   ≈ 45 min** (~4.5 s/file; was ~8 s/file unbatched, A-5fold TIMED OUT at 5 folds). Perch-MLP ≈ 18 min;
   SED-v5j-4fold+TTA ≈ 46 min. Kaggle code comps score ONE kernel's CSV → can't blend separate kernels.
   **A-4fold ONNX has a dynamic batch dim → batch windows into one call/fold (1.78× faster, byte-identical).**
6. **GPU**: a100/h200 often full; **rtx6000 (24 GB)** is the reliable parallel option.
7. eos8 notebook (committed `.ipynb`): `_cells.py` is gitignored — regenerate with
   `experiments/038-xc-pretraining` … (see git history) if needed; both eos8 PP levers already tested + OUT,
   so likely no longer needed. Coverage analysis: `experiments/038-xc-pretraining/scripts/analyze_fold_coverage.py`.

**Credentials**: `user/murilo/.env` (gitignored; `KAGGLE_API_TOKEN` (KGAT_… format), `HF_TOKEN`,
`WANDB_API_KEY`). The HF token was pasted in chat earlier — rotate `hf-token-pace`.
