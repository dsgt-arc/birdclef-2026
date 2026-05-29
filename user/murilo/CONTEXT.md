# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-29 EOD. Exp 034 (xcpretrain) first LB landed
(frozen 0.819 / full-FT 0.865, below the 0.906 base — bare model, gap = the
omitted stack). This session shipped the PP lever (kernel pushed) and scaffolded
the KD lever (Phase K, ready to launch). **Two LB reads pending** — see below.

**Competition deadline**: 2026-06-03 (5 days out).

**Branch**: `mg/exp034-pretrained-backbone` (PR **#54**, open). Latest commit `09dea9e`.

---

## Where we are

**Exp 034 — pretrained-backbone reuse (xcpretrain): Sydorskyi 2025 in-domain
EffNetV2-S backbone + our SED attention head, fine-tuned on BirdCLEF 2026.**

First valid LB (after fixing a species_mapping 41-vs-234 bug — see
[[feedback_species_mapping_canary]]):
- frozen-backbone (25 ep): **0.819**
- full-FT (50 ep): **0.865**

Both **below the 0.906 base**. The read is **"necessary-not-sufficient," NOT
"backbone fails"**: Gate 1 parity bit-identical, kernel↔eager 9e-7, 234-mapped.
Decisive control comparison — exp 032's no-KD control (ImageNet backbone, same
recipe, +PP smoothing) = **0.856**; exp 034 full-FT (in-domain backbone, NO PP) =
**0.865** → the in-domain backbone is a small real lever even without PP. The gap to
0.906/0.929 is the **STACK exp 034 omits**, by LB-evidenced value: **KD +0.050**,
PL +0.027, PP +0.012–0.020, 5-fold +0.009, ensemble. 0.94 competitors stack
backbone + PL + PP + ensemble. Full gap audit: `results.md § LB results`.

**Two levers actioned this session (both LB reads PENDING):**

1. **PP smoothing (cheapest, no-retrain) — kernel pushed, awaiting LB.**
   Added logit-space 5-tap Gaussian smoothing to `predict_soundscape_xcpretrain`
   (mirrors our 0.906 Tucker recipe). Pushed as a **distinct kernel**
   `birdclef-2026-xcpretrain-effv2s-fullft-pp` ("BirdCLEF 2026 XCPretrain EffV2S
   FullFT PP") that **reuses the existing `-fullft-weights` dataset** so the with-PP
   LB is a clean A/B vs the no-PP **0.865**. Logs verified (234-mapped + "5-tap
   LOGIT-space Gauss smooth"). NOTE: Sydorskyi himself uses NO smoothing — this is
   OUR lever (see [[reference_sydorskyi_inference_recipe]]).

2. **KD on the in-domain backbone (biggest lever) — scaffolded, NOT launched.**
   Phase K. Ported exp 032's Perch v2 embedding distillation into
   `XCPretrainFinetuneModule._step_kd` (DistillHead GAP+Linear 256→1536, MSE vs
   frozen Perch v2 ONNX teacher; `total = FocalBCE(cls) + alpha_distill·MSE`).
   `kd_detach_cls_features` selects Tucker-faithful (detach) vs joint. 5 unit tests
   + real-teacher integration check PASS. **2 configs ready, awaiting Murilo to
   launch the GPU grid.** DistillHead is discarded at inference → the SAME
   `export_xcpretrain.py` + kernel apply (KD changes weights, not the graph).

**On hold (deliberately):** 5-fold (Phase F) + ensemble (Phase I) — wait until the
PP + KD LB reads come back before fanning out (Murilo's call, 2026-05-29).

---

## Pick up here (next session)

### 0. Read state first
1. This file.
2. `experiments/034-pretrained-backbone/results.md` § LB results + § Gap to our best.
3. `experiments/034-pretrained-backbone/tasks.md` § Phase K (KD), § E.4 (LB verdict).
4. `git log --oneline -6` → top should be `09dea9e`.

### 1. Check the PP LB read (vs no-PP full-FT 0.865)
The `...-fullft-pp` kernel was pushed (maybe submitted EOD 05-29). Get its LB score.
- **PP LB > 0.865** → smoothing stacks on this backbone; keep it on for all future
  exp 034 submissions. Strong green light for KD.
- **PP LB ≈/< 0.865** → logit-space PP doesn't help here; try **prob-space** smoothing
  (Sydorskyi's native space for this prob-output head) as the quick follow-up
  (`mel_cfg["gaussian_smooth_logit_space"]` already gates it; would need a prob-space
  branch added). Don't block KD on this.

### 2. Launch the KD grid (Phase K) — GIVE MURILO THESE COMMANDS (don't run)
```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/experiments/034-pretrained-backbone
CONFIG=configs/xcpretrain-effv2s-kd.yaml       bash sbatch/train.sh   # Tucker-faithful (detach), 50 ep
CONFIG=configs/xcpretrain-effv2s-kd-joint.yaml bash sbatch/train.sh   # joint backbone grad, 50 ep
```
**Epoch-1 watch (W&B):** `train/dist_loss` vs `train/cls_loss`. FocalBCE ≪ exp 032's
plain BCE, so `alpha_distill=1.0` may over-weight the MSE — if `dist_loss` dominates,
kill + lower `alpha_distill` (one-line config edit), relaunch.
**Prior:** expect **joint** > **detach** here — detach forces an already-strong
in-domain backbone to mimic Perch (gave +0.050 on a WEAK ImageNet backbone; risks
regressing a strong one); joint keeps task fine-tuning + adds Perch as an auxiliary.

### 3. After KD trains → export + LB (mirror the full-FT path)
KD models export with the SAME `export_xcpretrain.py` (architecture unchanged).
Add `xcpretrain-effv2s-kd{,-joint}` entries to `upload_kaggle.py` MODELS +
`export.sbatch` + a kernel-metadata JSON (mirror `kernel-metadata-xcpretrain-effv2s-fullft-pp.json`;
include "234 mapped" + "5-tap Gauss smooth" canaries). Export → upload → push kernel → LB.
**Exit (Phase K):** KD LB > 0.865; stretch ~0.91 (approaching the +0.050 paired delta).

### 4. Then the 5-fold / ensemble decision
Once PP + KD reads are in: if a single exp 034 config clears ~0.90, fan out to 5-fold
(Phase F) + rank-ensemble exp 034 ⊕ exp 032 5-fold (0.915) ⊕ exp 023 (0.929) (Phase I).
If KD doesn't transfer to the in-domain backbone, drop it → go straight to ensemble.

---

## Submission descriptions (for Murilo, 500-char Kaggle limit)

**PP kernel (`...-fullft-pp`), if not yet submitted:**
```
XCPretrain EffV2S full fine-tune (Sydorskyi in-domain backbone + our SED head)
on BirdCLEF 2026, WITH post-processing: 5-tap logit-space Gaussian temporal
smoothing across 5s windows (mirrors our 0.906 recipe). Identical weights to the
0.865 no-PP fullft run — isolates the PP lever. Single fold; no KD/PL/ensemble.
Raw audio → ONNX (nnAudio mel in-graph), 234-class identity mapping.
```

---

## Key decisions / learnings this session (2026-05-29)

1. **Backbone is necessary-not-sufficient.** In-domain backbone alone = 0.865 (a
   small +0.009 over the ImageNet control even without PP); the gap is the omitted
   KD/PP/PL/ensemble stack, not the backbone.
2. **Sydorskyi's inference uses NO smoothing/TTA** (audited `vendor/sydorskyi-2025/`).
   His lift = PL + **prob-space** ensemble (mean/vlom/gaus) + SWA + Google-model blend.
   → For exp 034 Phase F, use **prob-space fold averaging**, not the logit-space
   fold-mean our Tucker ensemble uses. ([[reference_sydorskyi_inference_recipe]])
3. **PP space is a genuine fork.** This head outputs a probability (`clipwise_pred_long`),
   so prob-space smoothing is more native; we shipped **logit-space** (our 0.906 prior)
   first by Murilo's call. Prob-space is the documented fallback.
4. **KD detach vs joint is the Phase K experiment.** Tucker's +0.050 used detach on a
   weak backbone; joint is likely safer on a strong one. Grid settles it.
5. **species_mapping canary is mandatory.** Always eyeball "Species mapping: 234 mapped"
   in the kernel log before trusting an LB. ([[feedback_species_mapping_canary]])

---

## Files touched this session

| File | Change |
|---|---|
| `010-submission-pipeline/kaggle_kernel/submission.py` | `predict_soundscape_xcpretrain` + logit-space 5-tap Gaussian smoothing |
| `010-submission-pipeline/scripts/export_xcpretrain.py` | emit `gaussian_smooth_logit_space`/`gaussian_kernel` in mel_config |
| `010-submission-pipeline/kaggle_kernel/kernel-metadata-xcpretrain-effv2s-fullft-pp.json` | NEW distinct PP kernel (reuses -fullft-weights) |
| `010-submission-pipeline/scripts/upload_kaggle.py` | `xcpretrain-effv2s-fullft-pp` MODELS entry |
| `src/birdclef/lightning/xcpretrain_finetune.py` | KD branch: `_prep`/`_step_kd`/`_perch_embed` + KD `__init__` params + training_step |
| `034-pretrained-backbone/train.py` | wire KD config + REPO_DIR (repo-relative Perch path) |
| `034-pretrained-backbone/configs/xcpretrain-effv2s-kd{,-joint}.yaml` | NEW KD configs (detach + joint) |
| `034-pretrained-backbone/pyproject.toml` | `onnxruntime-gpu==1.26.0` (PerchTeacher) |
| `034-pretrained-backbone/{results,tasks}.md`, `MODEL-RUNS.md` | LB scores + gap audit + Phase K |
| `tests/034-pretrained-backbone/test_xcpretrain_kd.py` | NEW 5 KD tests (PASS) |

**Recent commits (latest first):** `09dea9e` (KD scaffold) · `fc47673` (distinct PP
kernel) · `6417d52` (PP smoothing) · `3701a57` (LB scores + gap audit) ·
`fa71d2b` (upload routing fix).

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current overall best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| distilled-sed-kd-5fold (exp 032) | 0.915 | Best Tucker-SED ensemble |
| distilled-sed-kd (exp 032, single-fold) | 0.906 | **The base exp 034 must beat** |
| **exp 034 full-FT (in-domain backbone, no stack)** | **0.865** | bare; +0.009 vs ImageNet control |
| distilled-sed-control (exp 032, ImageNet, no KD, +PP) | 0.856 | the apples-to-apples control |
| exp 034 frozen-backbone | 0.819 | weaker variant |
| exp 034 full-FT + PP | _pending_ | A/B for the PP lever |
| exp 034 KD (detach / joint) | _pending_ | Phase K, awaiting launch |

---

## How to resume after disconnect

1. Read this file top-to-bottom.
2. `git log --oneline -6` → `09dea9e` at top; branch `mg/exp034-pretrained-backbone` (PR #54).
3. Get the PP LB read (§1) and KD grid status (§2: launched? finished? not started?).
4. Proceed per §1–4 above.
5. **Per [[feedback_user_runs_commands]]: don't run SLURM jobs / Kaggle uploads / kernel
   pushes yourself — give Murilo the commands.** Local pytest is fine (interactive venv
   at `/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv`; see [[feedback_use_uv_venv]]).
6. Venv policy: NEVER create a venv under home/project; SLURM jobs build on `$TMPDIR`
   (CLAUDE.md). PACE compute is NOT the bottleneck — prefer parallel grids
   ([[feedback_pace_compute_not_bottleneck]]).

**Prior context (exp 032/033):** exp 032 merged to main (`05f4063`); exp 033 own-teacher
PL = REJECT (PR not merged, branch `mg/exp033-own-teacher-pl`). Detail in those experiments'
`results.md`. Exp 034 superseded both as the active track.

**Credentials**: `user/murilo/.env` (gitignored), loaded by `train.py`.
```
