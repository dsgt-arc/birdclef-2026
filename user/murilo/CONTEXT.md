# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first to resume;
follow the linked files for full detail.

**Last updated**: 2026-05-31 (PM). **Active experiment: exp 038 — broad Neotropical XC supervised
pretraining (EffNetV2-S).** Stage 0 (curate + download) DONE; **Stage 1 (3-cell pretrain grid) DONE +
documented** (results.md/MODEL-RUNS.md/metrics committed); **Stage 2 (fine-tune Cell A & Cell C →
LB) BUILT + LOCAL-TESTED, ready for Murilo to launch.** exp 037 CLOSED/REJECT (merged, PR #57).

**Competition deadline**: 2026-06-03 (~3 days).

**Deadline picture (corrected this session):** the **0.929 exp-023 ensemble is ALREADY submitted +
scored on the LB** — so Track B is essentially done; the only remaining deadline action is **selecting
it as a final submission** (+ choosing the 2nd pick). That de-risks the clock and frees the next pulls
for the exp 038 Stage-2 A-vs-C test. exp 038 is still the **diversity/paper** bet with an honest
**~0.88–0.90 LB ceiling** (exp 034/036) — not expected to beat 0.906/0.929; its Stage-2 number is a
candidate 2nd final pick *iff* its prob-space blend beats 0.929. LB rate-limited (~5/day); ~14 left.

**▶ NEXT SESSION STARTS HERE:** Stage 2 is built. (1) Murilo runs the **fast-dev-run pre-flight**, then
**launches the 10-job grid** (Cell A & C × 5 folds, rtx6000) — commands in exp 038 `tasks.md` §Stage 2.
(2) When folds finish: export (exp 010 `export_distilled_sed.py`) → **LB single-fold A vs C vs 0.906**
(2 pulls = the decisive read), then 5-fold ensembles + 0.929 blend only if warranted. (3) Don't forget
to **select the 0.929 as a final submission**. See "Pick up here".

---

## Where we are — exp 038 (XC pretraining)

**Goal:** our own broad Neotropical bird backbone (EffNetV2-S, ImageNet-21k) → fine-tune on the 234
targets → backbone-diverse ensemble member + working-note. Branch `mg/038-xc-pretraining` (**pushed,
NOT merged** — open feature branch).

**Stage 0 — DONE.** Curated BirdSet-XCL (HF `DBD-research-group/BirdSet`, config XCL, metadata via
`XCL_metadata.parquet`) → **2,683 Neotropical bird species / 160,542 recs / 2,202 h**, **158/162** bird
targets (`palhor3`/`strher2` absent). Downloaded all 98 shards → **~151k OGGs on disk** (confirmed
**32 kHz mono**, no resample; 9,281 ≈5.8% genuinely missing → `missing_files.json`). Data at
`/storage/project/ps-clef2026_birdclef-0/shared/pretrain/birdset-neotropical/`
(`manifest.parquet`, `species_list.json`, `audio/<ebird_code>/<XCid>.ogg`). Download = `sbatch` worker
`sbatch/download_birdset_audio.sh` (resumable).

**Stage 1 — BUILT + RUNNING. Methodology (load-bearing decisions, locked with Murilo):**
- **SED-mirror head**: reuse `DistilledSEDFinetuneModule` **UNCHANGED** (`AttHeadLogitSpace`,
  `num_classes=2683`, **KD off**) → the backbone trains under the EXACT downstream pooling/mel regime
  (best transfer) AND it's the lowest-risk path (battle-tested module, no new training code). Chosen
  over a vanilla GAP+linear classifier. Only NEW code = the data layer
  `src/birdclef/data/xc_pretrain_dataset.py` (`XCPretrainDataset` multi-hot incl. in-vocab secondaries
  / random 5 s crop / `apply_aug`+focal×focal mixup ported from `FocalDS`; stratified per-species
  `XCPretrainDataModule`) + a **gated additive** `loss_type="focalbce"` + `label_smoothing` hparam in
  the SED module (the `bce`/`softauc` paths are **bit-identical**, regression-tested; LS applied in-loss
  to a local target copy so `labels.int()` metrics stay hard; focalbce lazy-imports torchvision).
- **Recipe**: Tucker mel n_mels=256 sr=32000, 5 s window, dual-supervision `0.5·BCE(clip)+0.5·BCE(frame_max)`
  + LS 0.05, multi-label = primary ∪ in-vocab secondaries (34.6% of recs have secondaries), AdamW
  lr 5e-4 → 2-ep warmup → cosine→1e-6, 16-mixed, grad-clip 1.0, **25 ep `save_top_k=-1`** (pick the
  XC-val peak), num_workers 8, stratified **2% XC val** = the clean sanity signal (NOT a 2026-LB
  predictor — labeled-SS/offline anti-correlate with LB: [[project_cv_measurement_failure]],
  [[feedback_offline_metrics_unreliable]]).
- **Run strategy = proven primary + parallel insurance arms** (compute is free
  [[feedback_pace_compute_not_bottleneck]]; LB is the only reliable ranker, so we don't agonize over
  un-rankable recipe knobs). 3 cells: **A** dual-BCE+LS (PRIMARY) / **B** focal-BCE (`FocalLossBCE`
  α0.25/γ2) / **C** no-mixup. Fine-tune **A** to the LB first; B/C are ready backups (no second wait).
- **Verified local**: 23 new tests + 238 regression (032/033/037) pass. **fast-dev-run** (a100, batch
  128) PASS + **gpu-check** (rtx6000, batch 64) PASS — GPU-bound 87% util, data-load 1.4%
  (num_workers=8 fine), peak reserved 14.2 GB/24 GB, **~10 min/epoch → ~4.2 h/cell** (`outputs/gpu_check/*.{txt,png}`).

### In-flight jobs (rtx6000, launched ~00:14 May 31, ETA ~04:30)
| Cell | Job | Config | Role |
|---|---|---|---|
| A | 9351785 | `configs/pretrain_cellA_dualbce.yaml` | PRIMARY (dual-BCE + LS 0.05) |
| B | 9351788 | `configs/pretrain_cellB_focalbce.yaml` | insurance (focal-BCE) |
| C | 9351789 | `configs/pretrain_cellC_nomixup.yaml` | insurance (no mixup) |

Effective batch 128 (~1160 opt steps/ep). Epoch 0: cellA `val/auroc` 0.513 (climbing from random;
`epoch00.ckpt` saved → `save_top_k=-1` + `monitor=val/auroc` both confirmed working). Checkpoints →
`/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/038-xc-pretraining/checkpoints/<run_name>/`.
W&B project `birdclef-2026-exp038`. **These cells launched BEFORE the metric-parity commit** → live
W&B/TB metrics + `pretrain_results.json` (`best_val_auroc`) YES; `outputs/test_results/` +
auto-`save_metrics` NO → regenerate `metrics.json` post-hoc (fallback reads `pretrain_results.json`).

---

## Pick up here (next session)

### A) exp 038 — after the grid finishes
1. `squeue -u $USER` → confirm A/B/C done. Read `val/auroc` peaks from W&B (`birdclef-2026-exp038`) or
   each `checkpoints/<run>/pretrain_results.json` → `best_val_auroc`.
2. Regenerate the 3 structured artifacts: `cd user/murilo/experiments/038-xc-pretraining` then
   `python save_metrics.py --config configs/pretrain_cell{A,B,C}_*.yaml` → `artifacts/training/<variant>/metrics.json`.
   Compare `best_val_auroc` (sanity gate; ≳0.9 expected in-domain; A vs B vs C = which recipe learned
   XC best — a PROXY, not LB). Pick the Cell A backbone (B/C backups).
3. **Build Stage 2 (NOT YET BUILT):** load the Cell A backbone (`epochNN.ckpt` at the val/auroc peak)
   → re-init a 234-class `AttHeadLogitSpace` → run the exp 032 KD-SED fine-tune recipe (KD-from-Perch +
   labeled-SS + dual-BCE + mixup; reuse `DistilledSEDFinetuneModule` with KD on) → exp 010 ONNX export
   → **LB** (single-fold vs 0.906, 5-fold vs 0.915, prob-space blend into 0.929). LB is the arbiter.

### B) Deadline END-GAME — secure the 0.929 final picks (STILL PENDING; higher deadline-EV)
1. Re-confirm the exp 023 **0.929** ensemble is reproducible + submittable (kernel builds/scores;
   canary `Species mapping: 234 mapped` — [[feedback_species_mapping_canary]]).
2. **PP/TTA** on the best ensemble (Gaussian smoothing is +0.014-proven; TTA untried + cheap). A/B.
3. **Pick the 2 final submissions**: safe = 0.929; upside = best blend (possibly + the exp 038 Cell-A
   model as a backbone-diverse member iff the prob-space blend beats 0.929 on LB).

Weigh A vs B against the 3-day clock: **B is the competitive deadline work**; A is the diversity/paper
bet whose Stage-2 LB number could become the "upside" member of B.

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | **Overall best → secure as a final pick (END-GAME pending)** |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| distilled-sed-kd-5fold (exp 032) | 0.915 | Best Tucker-SED ensemble |
| distilled-sed-kd single-fold (exp 032) | 0.906 | Best single model (ImageNet + KD) — Stage-2 target to beat |
| exp 034 / 036 XC-backbone (+PP) | 0.878–0.882 | XC-backbone single-fold ceiling (the ~0.88 evidence) |
| **exp 038 Cell A (XC-pretrain → Stage 2)** | **pending** | grid running; Stage 2 → LB next session |

---

## How to resume after disconnect
1. Read this top-to-bottom, then exp 038 `proposal.md` + `tasks.md`.
2. `git log --oneline -8` on `mg/038-xc-pretraining` (**pushed, open — NOT merged**). Top commits are
   the doc updates + `save_metrics.py` parity. exp 037/036/034/032 are CLOSED on main.
3. Check the grid: `squeue -u $USER` + W&B `birdclef-2026-exp038`. Then go to "Pick up here".
4. **[[feedback_user_runs_commands]]**: give Murilo commands; don't run SLURM/Kaggle yourself. Local
   pytest + `save_metrics.py` + light analysis are fine. Interactive venv
   `/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv` ([[feedback_use_uv_venv]]).
5. Venv policy: NEVER create a venv under home/project; SLURM jobs build on `$TMPDIR` (CLAUDE.md).
   PACE compute is NOT the bottleneck ([[feedback_pace_compute_not_bottleneck]]); Kaggle LB IS
   rate-limited (~5/day). `--squash` for PR merges ([[feedback_squash_merge]]).
6. **GPU availability (PACE, via `scripts/check_gpus.py`)**: a100/h200 often full; **h100 free-GPU
   nodes frequently have 0 idle CPUs → unschedulable** (job pends). **rtx6000 (24 GB, ~97 free)** is the
   reliable parallel option. `run.sh --gpu` accepts fallback lists (`h100,a100,rtx6000`) + `--batch-size`/
   `--accumulate` for the 24 GB cards (rtx6000 fits eff-batch-128 via batch 64 × accum 2, or batch 128
   physically — both ran clean).

**Credentials**: `user/murilo/.env` (gitignored; HF_TOKEN, W&B), loaded by `train.py`. **Note**: the HF
token was pasted in chat this session — Murilo should rotate `hf-token-pace`.
