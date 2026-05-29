# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-29. Exp 034 (xcpretrain) first LB landed
(frozen 0.819 / full-FT 0.865, below the 0.906 base — bare model, gap = the
omitted stack). PP lever LANDED +0.014 (full-FT 0.865 → 0.879). This session: KD grid finished + exported.
**Joint-KD + PP scored LB 0.878 ≈ full-FT+PP 0.879 → KD does NOT transfer** to the in-domain
backbone (despite +0.034 offline — anti-correlation). **KD DROPPED.** exp 034 single-fold
ceiling ≈ 0.879. **Next = Phase F 5-fold + Phase I ensemble** (exp 034 as a backbone-diverse
member), plus a new final-experiment idea under discussion before the 2026-06-03 deadline.

**Competition deadline**: 2026-06-03 (5 days out).

**Branch**: `mg/exp034-pretrained-backbone` (PR **#54**, open). Latest commit `ee160dc`
(doc updates for the KD-grid result + export wiring are UNCOMMITTED in the working tree).

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

1. **PP smoothing (cheapest, no-retrain) — DONE, LB-validated +0.014.**
   Logit-space 5-tap Gaussian smoothing in `predict_soundscape_xcpretrain` (mirrors
   our 0.906 Tucker recipe), shipped as a distinct kernel
   `birdclef-2026-xcpretrain-effv2s-fullft-pp` reusing the `-fullft-weights` dataset.
   **LB 0.865 → 0.879 (+0.014)**, in the predicted +0.012–0.020 band → stack-levers
   compose on the in-domain backbone. **Keep PP on for all future exp 034 submissions.**
   (Sydorskyi himself uses NO smoothing — this is OUR lever; [[reference_sydorskyi_inference_recipe]].)

2. **KD on the in-domain backbone (biggest lever) — LAUNCHED, running overnight.**
   Phase K. Ported exp 032's Perch v2 embedding distillation into
   `XCPretrainFinetuneModule._step_kd` (DistillHead GAP+Linear 256→1536, MSE vs
   frozen Perch v2 ONNX teacher; `total = FocalBCE(cls) + alpha_distill·MSE`).
   `kd_detach_cls_features` selects Tucker-faithful (detach) vs joint. 5 unit tests
   + real-teacher integration check PASS. **2-cell grid launched 2026-05-29 EOD:**
   - `xcpretrain-effv2s-kd` (detach): **SLURM 9299507** (gpu-rtx6000)
   - `xcpretrain-effv2s-kd-joint` (joint): **SLURM 9299513** (gpu-rtx6000)
   50 ep each; expect completion overnight. DistillHead is discarded at inference →
   the SAME `export_xcpretrain.py` + kernel apply (KD changes weights, not the graph).

**On hold (deliberately):** 5-fold (Phase F) + ensemble (Phase I) — wait until the
PP + KD LB reads come back before fanning out (Murilo's call, 2026-05-29).

---

## Pick up here (next session)

### 0. Read state first
1. This file.
2. `experiments/034-pretrained-backbone/results.md` § LB results + § Gap to our best.
3. `experiments/034-pretrained-backbone/tasks.md` § Phase K (KD), § E.4 (LB verdict).
4. `git log --oneline -6` → top should be `09dea9e`.

### 1. PP read is IN — full-FT + PP = 0.879 (+0.014 over 0.865). ✅ DONE.
Smoothing stacks on this backbone. **Keep PP on for every future exp 034 submission**
(the KD kernels must use the same `predict_soundscape_xcpretrain` smoothing). No action
needed here — this just green-lights the bigger levers.

### 2. KD grid (Phase K) — DONE 2026-05-29. Joint wins decisively (offline). ✅
Both 50-ep runs finished clean. **joint `tptu` val/auroc 0.9217 ≫ detach `29fz` 0.8847**
(+0.037) — exactly the prior (detach forces a strong backbone to mimic Perch; joint adds
Perch as an auxiliary). Joint's 0.9217 also > exp 032 distilled-sed-kd's own offline (0.908,
the LB-0.906 model) on the SAME val set, and +0.034 over full-FT (0.883). Epoch-1 balance
gate **PASS** (`dist_loss` ≈25% of `cls_loss`, never swamped → `alpha_distill=1.0` fine).
Save-last quirk: `last.ckpt` = last top-k save (joint ep43 val 0.9226), uniform across the
experiment (full-FT's LB ckpt = ep22), so the ep43 export is apples-to-apples. Detail:
`results.md § Phase K` + `MODEL-RUNS.md § Exp 034`. (Offline anti-correlates with LB → the LB read decides.)

### 3. Export joint-KD + PP kernel — DONE 2026-05-29 → **LB 0.878 ≈ full-FT+PP 0.879 → KD dropped.**
(Verdict in `results.md § KD LB verdict`: KD is a multiplier on a weak base, not an additive lever
on Sydorskyi's already-focal-pretrained backbone. Single-fold ceiling ≈0.879. Wiring kept for
provenance; the export/submit commands below are how it was produced.)
KD discards the DistillHead (`distill_head.*` ≠ `model.*`; export filters `model.*`) →
ONNX graph **identical** to full-FT; `export_xcpretrain.py` bakes PP into mel_config → the
kernel runs WITH 5-tap logit-space Gauss smooth by default. 4 exp-010 files wired +
`kernel-metadata-xcpretrain-effv2s-kd-joint.json` (title slug-resolves to id ✅). Kernel
**validated on Kaggle** (commit run clean: `Species mapping: 234 mapped`, `5-tap LOGIT-space
Gauss smooth`, weights dataset resolves, 24 preds/2 files, CPU ~1s/file).
```bash
cd user/murilo/experiments/010-submission-pipeline
bash sbatch/run.sh export --model xcpretrain-effv2s-kd-joint      # GPU, ~10 min
python scripts/upload_kaggle.py --model xcpretrain-effv2s-kd-joint           # NEW dataset; wait 2-3 min
python scripts/upload_kaggle.py --model xcpretrain-effv2s-kd-joint --push-kernel  # then submit in UI
```
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
| `010-submission-pipeline/sbatch/export.sbatch` | NEW `export_xcpretrain_effv2s_kd_joint()` + 234-map case + dispatch case |
| `010-submission-pipeline/scripts/upload_kaggle.py` | NEW `xcpretrain-effv2s-kd-joint` MODELS entry (own `-kd-joint-weights` slug) |
| `010-submission-pipeline/kaggle_kernel/kernel-metadata-xcpretrain-effv2s-kd-joint.json` | NEW kernel (PP baked in via export; title slug-resolves ✅) |
| `010-submission-pipeline/kaggle_kernel/submission.py` | `…-kd-joint-weights` added to `MODEL_SLUGS` (first in exp-034 group) |
| `034-pretrained-backbone/{results,tasks}.md`, `CONTEXT.md`, `MODEL-RUNS.md` | KD grid result + export wiring recorded |

**Recent commits (latest first):** `ee160dc` (PP LB-validated 0.879) · `eaec243` (KD grid
launched) · `09dea9e` (KD scaffold) · `fc47673` (distinct PP kernel) · `6417d52` (PP smoothing).
**Uncommitted:** the 4 export-wiring files + new kernel-metadata + the doc updates above.

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current overall best |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| distilled-sed-kd-5fold (exp 032) | 0.915 | Best Tucker-SED ensemble |
| distilled-sed-kd (exp 032, single-fold) | 0.906 | **The base exp 034 must beat** |
| **exp 034 KD-joint + PP** | **0.878** | = full-FT+PP 0.879 → KD adds ~0 on LB (offline 0.9217 was an artifact); KD dropped |
| **exp 034 full-FT + PP** | **0.879** | PP lever +0.014 over 0.865 (kernel-only) |
| exp 034 full-FT (in-domain backbone, no stack) | 0.865 | bare; +0.009 vs ImageNet control |
| distilled-sed-control (exp 032, ImageNet, no KD, +PP) | 0.856 | the apples-to-apples control |
| exp 034 frozen-backbone | 0.819 | weaker variant |
| exp 034 KD-detach | _skip_ | offline 0.8847 (expected-weaker; paired data point only) |

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
