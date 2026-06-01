# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first to resume;
follow the linked files for full detail.

**Last updated**: 2026-06-01 (late 05-31 session). **Active: exp 038 XC-pretraining endgame +
eos8-notebook-derived post-processing.** `finetune-cell-a-4fold` SUBMITTED to LB (RUNNING — check the
score first thing). Branch `mg/038-xc-pretraining` (pushed, open — NOT merged).
**Deadline 2026-06-03 (~2 days). LB rate-limited (~5/day).**

---

## ▶ NEXT SESSION — START HERE (in order)

1. **Check Kaggle LB for `finetune-cell-a-4fold`** (submitted late 05-31, was running at wrap-up).
   Reads to take: does it beat C-5fold **0.911**? approach the banked **0.929**? + note its **real
   hidden-test runtime** (de-risks the Perch+A-4fold blend budget). Resolves **A-vs-C at the ensemble
   level** (A led C single-fold 0.907 vs 0.901).
2. **Build the two eos8-derived post-processing variants** of the 0.929 ensemble kernel (NOT built yet —
   pre-build was offered tonight and deferred): **(a) rank-space Perch+SED blend** and **(b) taxonomy
   smoothing**. Build local (no LB pull), then submit. See "eos8 notebook" below for code refs.
3. **Perch + A-4fold blend** — *gated on #1*. Only if A-4fold is strong enough to plausibly beat 0.929.

Make build decisions with Murilo; [[feedback_user_runs_commands]] — give him commands for SLURM/Kaggle.

---

## ✅ Tonight's progress (2026-05-31 late → 06-01)

### 1. `finetune-cell-a-4fold` built + submitted (RUNNING on LB)
- A-5fold **timed out** last session (90-min CPU wall, ~73/90 margin). A-4fold = the **existing 5-fold
  ONNX with fold2 dropped**, logit-space mean over folds **{0,1,3,4}**. **Reuses the 5-fold ONNX — NO
  re-export.** Projected ~58–72 min CPU (C-5fold finished at 5 folds → 4-fold has strictly more margin).
- **Why drop fold2 (not the offline-weakest fold0):** ran
  `experiments/038-xc-pretraining/scripts/analyze_fold_coverage.py` (reproduces the exact
  `StratifiedKFold(seed=42)` focal split + `GroupKFold` SS split, bit-identical to training). Findings:
  - Folds are **near coverage-equivalent** (identical ~28,439 train recs each). The 28 "missing" species
    are **globally focal-absent** (the 25 `47158son*` focal-less + 3 others) → blind to ALL folds equally
    → irrelevant to the drop choice (learned only via labeled-SS + Perch KD).
  - Only real differentiator = **true singleton blind-spots** (species with focal data confined to one
    fold): **fold2 holds 2** (`23150`,`23724`), folds 0/1 hold 1, folds 3/4 hold 0 → **drop fold2** =
    cleanest 4-fold coverage of the full 234-label space.
  - The offline val/auroc spread (fold0 **0.887** … fold4 0.924) is a **val-composition artifact** — each
    fold validates on a *different* species subset (**29 → 45** distinct species). So we **KEPT fold0**
    (offline-worst but **LB-verified 0.907**), refuting "drop the offline-weakest."
    [[feedback_offline_metrics_unreliable]] · [[project_cv_measurement_failure]]
- Wiring (committed this session, exp 010): `artifacts/weights/finetune-cell-a-4fold/` (gitignored ONNX),
  `kaggle_kernel/kernel-metadata-finetune-cell-a-4fold.json`, `scripts/upload_kaggle.py` MODELS entry,
  `kaggle_kernel/submission.py` MODEL_SLUGS. The `"finetune-cell" in model_name` route copies the 234-map
  ([[feedback_species_mapping_canary]]).
- **Validated:** py_compile, `--dry-run`, ONNX 4-fold load/run/logit-mean smoke (→234-dim), and the Kaggle
  **commit-run log**: `Ensemble mode: 4 fold models` · `Species mapping: 234 mapped` · `Using Tucker
  4-fold ensemble` · 52 s on the 2-file public set, clean. (The 2-file `~10 s/file` is NOT a runtime
  proxy — cold-start dominated; the C-5fold-finished bound says 4-fold ≈ 58–72 min.)
- Commands to re-run if needed (Murilo): `python scripts/upload_kaggle.py --model finetune-cell-a-4fold`
  then (after ~2–3 min) `... --model finetune-cell-a-4fold --push-kernel`, then Submit to Competition.

### 2. Analyzed a 0.95-LB public notebook → the cheap transferable levers (next builds)
Pulled `ryutoyoda/birdclef-2026-exp013-eos8-sidecar` →
**`vendor/kaggle/notebooks/birdclef-2026/eos8-sidecar/`** (`.ipynb` + `_cells.py` = readable 27-cell dump).
**It's NOT a secret model** — same ingredients we have (Perch + Tucker distilled-SED + ProtoSSM)
recombined in **rank space** with heavy post-processing + **taxonomy smoothing** (+ BirdNET / ResidualSSM
we don't have). Structure: `p = 0.0305·yukiZ + 0.9695·PowerOptimization`; PowerOpt (self-reported
**0.949**) = `0.6·rank(proto)+0.4·rank(sed)` → ecological priors → file-confidence/rank-aware scaling →
temporal smoothing → taxonomy smoothing (SSM probes trained in-kernel on the 66 labeled SS), then gated
BirdNET + PCEN/ConvNeXt sidecars.

- **Metric = macro-averaged ROC-AUC** (per-class, rank-based, rare-taxa-weighted, excludes test-absent
  classes) → "relies entirely on sorting order" → rank-space ops + column-mixing (taxonomy) are
  well-matched. Calibration (temperature/sigmoid on a single column) is a no-op for AUC.
- **Transferable techniques (ranked by EV / effort), with `_cells.py` line refs:**
  1. **Rank-space blend** of our Perch+SED (vs prob-space avg): `_safe_rank_cols` =
     `df.rank(axis=0, method="average", pct=True)` (~L6065). Low–med effort; highest *structural* EV
     (metric is pure ranking; Perch-ONNX & SED have different score scales → rank-equalize before
     blending). Untried axis (our exp-023 grid was prob-space, flat 0.926–0.929).
  2. **Taxonomy smoothing** (~30-line final post-process): `f_TAX_SMOOTHING_POSTPROC` (~L5811) — genus
     α=0.15 → class α=0.05 mean-sharing using our `taxonomy.csv`. Cheap, rare-species helper, stackable
     on ANY submission. Expect small (+0.000–0.005).
  3. (Deprioritized — high effort) file-confidence + rank-aware scaling (~L1180 / L1276), ecological
     site/hour prior tables (~L1753), ResidualSSM probe, BirdNET v2.4 gated sidecar (~L6207).
- We already match: temporal smoothing (our 5-tap Gaussian ≈ their `delta_shift_smooth`), site/hour
  priors (pp2), MLP probes on Perch. We **dropped ProtoSSM under ONNX** in exp 023 (it hurt).

---

## The banked 0.929 (final pick #1) — composition, for the PP builds
exp 023 `birdclef-2026-perch-sed-mlp_only` (`submission_ensemble.py`, ~1,977 lines): **Perch ONNX
(MLP-only probes, `proto_weight=0`) + 6-stage PP** blended **prob-space w=0.5** with **SED v5j-4fold-pp2
(4-fold + TTA)**. Runtime ~64 min/600 files (Perch ~18 + SED ~46). The rank-space/taxonomy variants modify
THIS kernel. Hard constraint: one kernel, 90-min CPU, ~600 hidden SS → all members' runtimes ADD UP
(that's why 0.929-stack + C-5fold = ~137 min is infeasible; Perch + A-4fold ≈ 76 min fits).

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only (exp 023) | **0.929** | banked → **final pick #1** |
| v5j-4fold-pp2 (exp 021) | 0.917 | best SED ensemble |
| distilled-sed-kd-5fold (exp 032) | 0.915 | best Tucker-SED ensemble |
| exp 038 `finetune-cell-c-5fold` | 0.911 | best exp-038; backbone-diverse member |
| exp 038 `finetune-cell-a` (single) | 0.907 | parity w/ 0.906; refutes ~0.88 XC ceiling |
| **exp 038 `finetune-cell-a-4fold`** | **PENDING (running)** | folds {0,1,3,4}, fold2 dropped; ~58–72 min |
| exp 038 `finetune-cell-c` (single) | 0.901 | A>C → mixup-reg pretrain transfers better |
| exp 034 / 036 XC-backbone (+PP) | 0.878–0.882 | the ceiling exp-038 refuted |
| exp 038 `finetune-cell-a-5fold` | TIMED OUT | → replaced by the 4-fold |

## How to resume after disconnect
1. **Check the A-4fold LB on Kaggle first.** Then read this top-to-bottom + exp 038 `results.md`/`tasks.md`.
2. `git log --oneline -8` on `mg/038-xc-pretraining` (pushed, open — NOT merged).
3. For the next builds, read the eos8 notebook (committed `.ipynb`). `_cells.py` (taxonomy smoothing
   ~L5811, rank-space `_safe_rank_cols` ~L6065) is **gitignored** (derived) — regenerate deterministically:
   `python -c "import json;nb=json.load(open('vendor/kaggle/notebooks/birdclef-2026/eos8-sidecar/birdclef-2026-exp013-eos8-sidecar.ipynb'));open('vendor/kaggle/notebooks/birdclef-2026/eos8-sidecar/_cells.py','w').write(''.join((f'\n# ===== [CELL {i}] {c[\"cell_type\"].upper()} =====\n'+(''.join(c.get('source',[])) if c['cell_type']=='code' else '# '+''.join(c.get('source',[])).replace(chr(10),chr(10)+'# '))) for i,c in enumerate(nb['cells'])))"`
   (same line numbers). Coverage analysis: `experiments/038-xc-pretraining/scripts/analyze_fold_coverage.py`.
4. **[[feedback_user_runs_commands]]**: give Murilo commands; don't run SLURM/Kaggle yourself. Local
   pytest + CPU export/ONNX smoke-tests + analysis are fine. Interactive venv
   `/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv` ([[feedback_use_uv_venv]]).
5. Venv policy: NEVER create a venv under home/project; SLURM jobs build on `$TMPDIR` (CLAUDE.md).
   Compute NOT the bottleneck ([[feedback_pace_compute_not_bottleneck]]); Kaggle LB IS (~5/day).
   `--squash` for PR merges ([[feedback_squash_merge]]).
6. **Runtime budget (load-bearing):** 90-min CPU wall, ~600 hidden SS. EffNetV2-S single-fold ≈ 15 min,
   4-fold ≈ 58–72 min (safe), 5-fold ≈ 73–90 min (A-5fold timed out). Perch-MLP ≈ 18 min;
   SED-v5j-4fold+TTA ≈ 46 min. Kaggle code comps score ONE kernel's CSV → can't blend separate kernels.
7. **GPU**: a100/h200 often full; **rtx6000 (24 GB)** is the reliable parallel option.

**Credentials**: `user/murilo/.env` (gitignored; `KAGGLE_API_TOKEN` (KGAT_… format, read by Kaggle CLI
2.x), `HF_TOKEN`, `WANDB_API_KEY`). The HF token was pasted in chat earlier — rotate `hf-token-pace`.
