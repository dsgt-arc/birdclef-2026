# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first to resume;
follow the linked files for full detail.

**Last updated**: 2026-05-31 (evening). **Active experiment: exp 038 — broad Neotropical XC
pretraining (EffNetV2-S) → Stage-2 fine-tune.** Stage 0 + Stage 1 + Stage 2 training ALL DONE +
documented. **Submission pipeline BUILT + the species-mapping canary bug FOUND & FIXED.** Branch
`mg/038-xc-pretraining` (pushed, open — NOT merged). exp 037 CLOSED/REJECT (merged, PR #57).

**Competition deadline**: 2026-06-03 (~3 days). LB rate-limited (~5/day).

**▶ NEXT SESSION STARTS HERE — Murilo has NOT submitted anything to Kaggle yet.** Everything is built,
exported-locally-validated, and the canary bug is fixed. The single next action is: **run the 4 Kaggle
upload+push commands (below), verify `Species mapping: 234 mapped` in each kernel log, submit the 4
notebooks, then read the LB.** See "Pick up here".

---

## What's done (exp 038)

**Goal:** our own broad Neotropical EffNetV2-S backbone (ImageNet-21k → supervised XC pretrain) →
fine-tune on the 234 targets → backbone-diverse ensemble member + working-note. Honest ceiling
~0.88–0.90 single-fold (exp 034/036); NOT expected to beat 0.906/0.929. Value = diversity/paper.

- **Stage 0 — DONE.** Curated BirdSet-XCL → 2,683 Neotropical bird sp / 160,542 recs (158/162 targets);
  ~151k 32 kHz OGGs at `/storage/project/ps-clef2026_birdclef-0/shared/pretrain/birdset-neotropical/`.
- **Stage 1 (pretrain, 3 cells) — DONE.** A dual-BCE+LS / B focal-BCE / C no-mixup, EffNetV2-S
  (`tf_efficientnetv2_s.in21k`, use_post_head 1280-d), SED-mirror (`DistilledSEDFinetuneModule`,
  num_classes=2683, KD off), Tucker mel 256. **XC-val sanity gate PASS** (best val/auroc **A 0.9843 /
  B 0.9847 / C 0.9856** ≫ 0.95; tied within 0.0013 — XC-val is a sanity gate, NOT an LB ranker).
  Best ckpts: A `epoch23` / C `epoch22` at `experiments/038-xc-pretraining/checkpoints/2026-exp038-pretrain-cell{A,C}-*/`.
- **Stage 2 (fine-tune, A & C × 5 folds) — DONE (10 folds, 25 ep clean, rtx6000).** Recipe = exp 032's
  **0.906 KD-SED** (`DistilledSEDFinetuneModule`, Perch-v2 KD, dual-BCE, mixup, Tucker mel 256,
  AttHeadLogitSpace) with the encoder swapped to EffNetV2-S **warm-started** from each Stage-1 backbone
  (`init.encoder_only: true` = full-copy backbone, re-init 234 head). Code: `train_finetune.py` +
  `configs/finetune_cell{A,C}.yaml` + `sbatch/{finetune,run_finetune}.sh`. **Offline val (held-out
  labeled-SS, anti-correlates w/ LB → for-the-record only): Cell A mean 0.9104 (±0.013), Cell C mean
  0.9085 (±0.009) — a TIE; both ≈ the exp 032 B0 baseline offline (0.908).** A-vs-C is an **LB-only**
  question. Checkpoints: `checkpoints/2026-exp038-finetune-cell{A,C}/fold{0-4}/` (export uses `last*.ckpt`).

**Submission wiring (exp 010) — BUILT + locally validated.** 4 models: `finetune-cell-{a,c}` (fold0
single) + `finetune-cell-{a,c}-5fold` (logit-space fold-mean). Cloned the distilled-sed-kd pipeline
(V2-S backbone only → `export_distilled_sed.py` + the kernel Tucker path reused UNCHANGED). **Export
proven locally on CPU**: ONNX↔PyTorch max-diff 1e-6, schema `mel→blended_logits[batch,234]`, mel routes
to `tucker`, **canary species_mapping = 234**.

**⚠️ Canary bug FOUND + FIXED this session (3rd recurrence — [[feedback_species_mapping_canary]]):** the
*first* cell-a-5fold kernel-test printed `Species mapping: 41 mapped` (invalid). Root cause =
`upload_kaggle.py::upload_weights` `shutil.copy2`s a species_mapping over the weights dir on every
upload, routed by name token; `finetune-cell-*` matched none → shipped the generic 41-overlap map
(copy2 preserves mtime → the Mar-14 clue). **Fixes (committed `620145a`, `5788863`):** (1) added
`finetune-cell` to the 234-branch in `upload_kaggle.py`; (2) `export_distilled_sed.py` now self-emits
`species_mapping.json` (234 identity); (3) `export.sbatch` `_verify_species_mapping_234` aborts if ≠234.
All 4 on-disk dirs now read 234; the export jobs already ran (artifacts in `artifacts/weights/finetune-cell-*`).

---

## Pick up here (next session) — Kaggle submission (NOTHING submitted yet)

**1. Upload + push the 4 kernels** (from `user/murilo/experiments/010-submission-pipeline/`). `--push-kernel`
re-uploads weights first, so with the fixed routing all 4 ship the 234-map:
```bash
cd user/murilo/experiments/010-submission-pipeline
for m in finetune-cell-a-5fold finetune-cell-c-5fold finetune-cell-a finetune-cell-c; do
  python scripts/upload_kaggle.py --model $m            # create/update dataset
done
# wait ~2-3 min, then:
for m in finetune-cell-a-5fold finetune-cell-c-5fold finetune-cell-a finetune-cell-c; do
  python scripts/upload_kaggle.py --model $m --push-kernel
done
```
Then **submit each notebook from the Kaggle UI** (4 of ~5 daily pulls). **On every kernel run, confirm
the log prints `Species mapping: 234 mapped`** before trusting the score ([[feedback_species_mapping_canary]]).
(If a re-export is ever needed: `bash sbatch/run.sh export --model <name>` — but the artifacts already exist.)

**2. Read the LB** (the arbiter — offline can't rank A vs C, [[project_cv_measurement_failure]]):
- single-fold `finetune-cell-a` vs `finetune-cell-c` vs **0.906** + vs exp-034's **0.879** (does our own
  XC-pretrain backbone beat warm-starting someone else's? does mixup-A transfer better than sharp-C?).
- 5-fold `finetune-cell-{a,c}-5fold` vs **0.915**.
- **CONFIRM** if any beats 0.906; else they're diverse members.

**3. Blend + final picks (END-GAME):** the **0.929 exp-023 ensemble is already submitted+scored** — just
**select it as a final submission** (safe pick). 2nd pick = best exp-038 result *iff* a prob-space blend
of the best 5-fold into 0.929 beats 0.929 on LB (blend kernel NOT yet wired — do it properly if a 5-fold
lands well; it's post-deadline-safe since 0.929 is banked).

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Overall best → **select as a final pick** (already scored) |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| distilled-sed-kd-5fold (exp 032) | 0.915 | Best Tucker-SED ensemble |
| distilled-sed-kd single-fold (exp 032) | 0.906 | Best single model (ImageNet + KD) — Stage-2 target to beat |
| exp 034 / 036 XC-backbone (+PP) | 0.878–0.882 | XC-backbone single-fold ceiling (the ~0.88 evidence) |
| **exp 038 finetune-cell-{a,c}{,-5fold}** | **pending** | trained + wired + canary-fixed; **submit next session** |

---

## How to resume after disconnect
1. Read this top-to-bottom, then exp 038 `results.md` + `tasks.md` (§Stage 2).
2. `git log --oneline -10` on `mg/038-xc-pretraining` (pushed, open — NOT merged). Top commits: `5788863`
   / `620145a` (canary fix), `ef81774` (Stage-2 offline metrics), `baecafb` (submission wiring),
   `1cb73aa` (Stage-2 build). exp 037/036/034/032 are CLOSED on main.
3. Go straight to "Pick up here" — run the 4 upload+push commands, verify the 234 canary, submit, read LB.
4. **[[feedback_user_runs_commands]]**: give Murilo commands; don't run SLURM/Kaggle yourself. Local
   pytest + `save_metrics.py` + local CPU export smoke-tests + analysis are fine. Interactive venv
   `/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv` ([[feedback_use_uv_venv]]).
5. Venv policy: NEVER create a venv under home/project; SLURM jobs build on `$TMPDIR` (CLAUDE.md).
   Compute NOT the bottleneck ([[feedback_pace_compute_not_bottleneck]]); Kaggle LB IS rate-limited
   (~5/day). `--squash` for PR merges ([[feedback_squash_merge]]). Pre-commit ruff-format may reformat a
   staged file → re-`git add` + re-commit if a commit "fails" on the format hook.
6. **GPU availability**: a100/h200 often full; h100 free nodes often 0 idle CPUs (pends). **rtx6000
   (24 GB)** is the reliable parallel option. The finetune ran batch 32 × accum 2 = eff-batch 64 to fit
   rtx6000 alongside the GPU Perch KD teacher.

**Credentials**: `user/murilo/.env` (gitignored; HF_TOKEN, W&B). The HF token was pasted in chat in a
prior session — Murilo should rotate `hf-token-pace`.
