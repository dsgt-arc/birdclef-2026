# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-27 (end of day — exp 033 REJECT verdict +
Phase 5 audit landed, Test 1 a2_kd_v3 launched as final PL data point,
deep research prompt drafted, pivot decision deferred to 2026-05-28 AM).

**Competition deadline**: 2026-06-03 (7 days from this update).

---

## Where we are

**Exp 032 — Tucker distilled-SED full replication: COMPLETE (PARTIAL REPLICATION)**
- PR #51 merged to main as commit `05f4063`.
- Best Tucker-SED single: `distilled-sed-kd` (Phase D, fold 0) = **LB 0.906**.
- Best Tucker-SED ensemble: `distilled-sed-kd-5fold` = **LB 0.915** (within ~0.002 of Tucker's 0.917).
- HGNet+KD ablation: LB 0.901 → backbone NOT the bottleneck (refuted).
- Project best overall (UNCHANGED): exp 023 `perch-sed-ensemble-mlp-only` = **0.929 LB**.

**Exp 033 — Own-teacher pseudo-labels on distilled-sed-kd: REJECT verdict + Phase 5 diagnostic in progress**
- Branch: `mg/exp033-own-teacher-pl` (~24 commits ahead of main, all pushed).
- All 10 ablations trained. D.5 e2e validation: 10/10 PASS.
- 3 of 5 Day-1 LB submissions landed — all REJECT (Δ ≤ 0 vs 0.906 baseline):
  - a4-kd (warm + Uniform + T2 teacher):  LB **0.899** (Δ −0.007)
  - a2-kd ⭐ (random + Uniform + T2, canonical Antoine):  LB **0.886** (Δ −0.020)
  - a4-v5j (warm + Uniform + T1):  LB **0.897** (Δ −0.009)
- Paired teacher-axis Δ (a4-kd vs a4-v5j): +0.002 LB — refutes proposal's headline
  teacher-diversity hypothesis at LB scale.
- Remaining Day-1 (a3-kd, a3-v5j) + all Day-2 cells NOT submitted on the trajectory.

**Phase 5 — Sydorskyi-divergence diagnostic (NEW, 2026-05-27 PM)**:
Audit found exp 033 implementation diverges from the only published 0.93+ PL recipe
(Sydorskyi 2025 2nd place) on **16 of 20 axes**. Most important: we never implemented
focal × PL MixUp fusion, chunk discard, or trim probability. The REJECT verdict
cannot be cleanly interpreted as a refutation of the protocol — we never ran the
protocol. See:
  - `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md`
    (full 20-axis divergence table + decision matrix; commit `40708f7`)
  - `experiments/033-own-teacher-pl/diagnostic-tests.md`
    (Test 1/2/3 workflow + commands + decision criteria; **resume-after-disconnect anchor**)

**Phase 5 Test 1 status (READY TO LAUNCH)**:
- Code: `generate_pseudo_labels.py` modified with optional `filter:` section.
- Configs: `configs/pl_gen_v3.yaml` (T2 + Sydorskyi 0.5/0.1 filters) and
  `configs/a2_kd_v3.yaml` (canonical Antoine on v3 parquet) created.
- Steps: re-issue v3 PL parquet (~30s head node) → pre-flight → sbatch
  a2_kd_v3 (~6.5h) → D.5 → ONNX export → LB submit.
- Decision criterion: LB > 0.913 → protocol recoverable, scale to other cells;
  LB 0.906-0.913 → run Test 2 (focal × PL fusion); LB < 0.906 → run Test 2 then
  decide pivot.

---

## Pick up here (next session) — 2026-05-28 morning

### 0. Read both overnight outputs before doing anything else

When you wake up:

1. **a2_kd_v3 training result** — check the SLURM log + the `outputs/test_results/2026-exp033-a2-kd-v3-fold0.json` file. Expect completion in ~6.5h. If launched at, say, 23:00, finishes around 05:30.
2. **Deep research output** — if you fired the prompt at `notes/2026-05-27-deep-research-prompt.md` overnight, the response should be ready at `docs/references/deep-research/2026-05-27-pivot-assessment.md` (or wherever Claude.ai delivered it).

Read both. They drive the exp 034 commit/pivot decision.

### 1. If a2_kd_v3 still needs LB submission (training landed, no LB shot yet)

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/experiments/010-submission-pipeline
# Add a2-kd-v3 to MODELS dict + kernel-metadata + export.sbatch (same pattern as commit 4280e72)
# then:
bash sbatch/run.sh export --model distilled-sed-pl-a2-kd-v3
python scripts/upload_kaggle.py --model distilled-sed-pl-a2-kd-v3
sleep 180
python scripts/upload_kaggle.py --model distilled-sed-pl-a2-kd-v3 --push-kernel
# Submit to Competition via Kaggle UI
```

### 2. Decision matrix (this is the actual point)

After reading both inputs together:

| Scenario | What it tells us | Move |
|---|---|---|
| **v3 LB > 0.913 AND research recommends yoink** | PL recoverable + yoink validated. Both stack. | Yoink first (higher ceiling), PL fixes second. |
| **v3 LB > 0.913 AND research recommends staying on PL** | PL works on our base; yoink may be unnecessary | Continue PL with Test 2 (focal × PL fusion) + scale to other cells |
| **v3 LB 0.906-0.913 AND research recommends yoink** | Modest PL lift; yoink still the bigger lever | Pivot to yoink; come back to PL later if time |
| **v3 LB < 0.906 AND research recommends yoink** | PL refuted on our base; yoink is the only path | Full pivot to yoink |
| **v3 LB < 0.906 AND research recommends something else** | Surprise — read carefully and reassess | Whatever research recommends |

### 3. Commit to one path. No more spanning.

Once decided, the next 3-4 days are single-track on that path. No new experiments without LB validation.

### 4. Files to read in order (if returning cold)

1. `user/murilo/CONTEXT.md` (this file) — current state
2. `experiments/033-own-teacher-pl/results.md` — the REJECT verdict + paired Δ readings
3. `analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` — why exp 033 REJECT might be implementation, not protocol
4. `notes/2026-05-27-deep-research-prompt.md` — the deep research prompt fired overnight (and read its output)
5. `experiments/033-own-teacher-pl/diagnostic-tests.md` — Phase 5 workflow if needed

---

### HISTORICAL: 2026-05-27 morning pickup (Phase G completion)

---

### HISTORICAL: 2026-05-27 morning pickup (Phase G completion — REPLACED BY 2026-05-28 plan above)

### 0. Check overnight job completion

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026
git checkout mg/exp033-own-teacher-pl
git log --oneline -12
# Expected most recent: 11b25cd → 6185a77 → 87d668d → ecc3ccb → 84ab853 → 6ab8312 ...

squeue -u $USER   # All 10 GPU jobs should be done (or only stragglers remain)
```

The 10 jobs were submitted in two waves:
- **T1 wave (5 jobs, launched 2026-05-26 ~21:30)**: 9190539, 9190540, 9190542, 9190544, 9190545
- **T2 wave (5 jobs, launched 2026-05-26 ~22:55)**: 9191658, 9191659, 9191660, 9191661, 9191662

Expected completion times (at ~20 min/epoch with num_workers=8):
| Ablation set | Epochs | Wall | T1 finish ETA | T2 finish ETA |
|---|---|---|---|---|
| a1 / a2 / a5 (random init, 25 ep) | 25 | ~8.3h | ~05:50 | ~07:15 |
| a3 / a4 (warm-start, 10 ep) | 10 | ~3.3h | ~00:50 | ~02:15 |

Hard walltime cap on each job: 24h (`#SBATCH --time=24:00:00`), so plenty of margin
even if any node is slow.

### 1. Verify all 10 trained cleanly

```bash
# All logs should end with "Job completed successfully!"
for log in user/murilo/experiments/033-own-teacher-pl/logs/slurm_a*_*-fold0_91*.log; do
    name=$(basename "$log" .log)
    status=$(tail -5 "$log" | grep -E "Job completed|exit code" || echo "STILL RUNNING")
    echo "$name → $status"
done

# Confirm checkpoints exist
ls -la /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/033-own-teacher-pl/checkpoints/2026-exp033-*/fold0/last.ckpt
# Expect 10 paths: 2026-exp033-a{1..5}-{v5j,kd}/fold0/last.ckpt
```

Per metric: read `outputs/test_results/2026-exp033-<ablation>-<teacher>-fold0.json`
for `val_metrics.val/auroc` and `val_metrics.val/cmap`. These are training-side
numbers (offline-LB anti-correlation caveat applies).

### 2. D.5 end-to-end validation on all 10 ckpts (~10 min each, can run in parallel)

Reuse exp 032's `sbatch/validate_end_to_end.sh` — it accepts FOLD/CKPT/CONFIG/TRAIN_AUROC
env vars. Example launcher:

```bash
cd user/murilo/experiments/032-tucker-distilled-sed
CKPT_BASE=/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/033-own-teacher-pl/checkpoints
RESULTS=/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/experiments/033-own-teacher-pl/outputs/test_results

for AB in a1 a2 a3 a4 a5; do
  for T in v5j kd; do
    SLUG="2026-exp033-${AB}-${T}"
    TRAIN_AUROC=$(jq -r '.val_metrics["val/auroc"]' "${RESULTS}/${SLUG}-fold0.json")
    CKPT="${CKPT_BASE}/${SLUG}/fold0/last.ckpt" \
    CONFIG="../033-own-teacher-pl/configs/${AB}_${T}.yaml" \
    FOLD=0 \
    TRAIN_AUROC="${TRAIN_AUROC}" \
    sbatch sbatch/validate_end_to_end.sh
  done
done
```

Gate: `|train_auroc - e2e_auroc| ≤ 0.02` PASS. Rank by **e2e val/auroc** to pick
the top-5 for LB Day 1. Per exp 032 Phase 2: e2e is uniformly slightly HIGHER than
training-side (~+0.004-0.011) due to 5-tap Gaussian smoothing in logit space.

### 3. LB submission (5/day cap)

```
Day 1 (today): export ONNX + push kernel + LB submit top-5 by e2e
Day 2 (next day): same for remaining 5
```

Per `feedback_kaggle_upload.md`: two-step Kaggle upload (dataset first, then kernel).
Per `feedback_kaggle_kernel_sources.md`: single-kernel pattern, kernel_sources uses
push-run output.

ONNX export wrappers: pattern matches exp 032's `010-submission-pipeline/`. New
`MODELS` entries needed for `distilled-sed-pl-a{N}-{T}` slugs — refer to exp 032
patterns for the wrappers.

### 4. Apply decision matrix (proposal.md § Falsifiable)

| Bucket | LB lift over 0.906 | Reading |
|---|---|---|
| **CONFIRM** | ≥ +0.014 | Round 2 (iterate teacher) + Sydorskyi-yoink for XC base |
| **PARTIAL** | +0.007 to +0.013 | Ensemble with exp 023 (0.929) |
| **WEAK** | +0.001 to +0.006 | Extend to 5-fold to disambiguate |
| **REJECT** | ≤ 0 | Pivot to Sydorskyi-yoink directly |

Secondary read: cross-ablation LB spread (std across A1-A5 within each teacher).
Spread > 0.010 → sampler choice is a recipe parameter to tune. Spread < 0.003 →
protocol is robust.

### 5. Closeout

- Update `experiments/033-own-teacher-pl/results.md` with full LB table + verdict
- Update `user/murilo/MODEL-RUNS.md` with all 10 ablation results
- Update this CONTEXT.md with verdict + next-session pickup
- Open PR (currently DRAFT NOT YET CREATED) — once verdict in:
  ```bash
  gh pr create --title "Exp 033: own-teacher pseudo-labels on distilled-sed-kd" \
    --body "<summary with LB table>" --base main --head mg/exp033-own-teacher-pl
  ```
  Squash-merge per `feedback_squash_merge.md`.

---

## What landed this session (2026-05-26, full day)

### Morning — exp 033 scaffolding (Phases A-E code)

Per the 2026-05-22 strategic direction note: Track B = own-teacher PL on top of
Track A = Tucker. 5 disclosed competitor recipes (Antoine 0.943, OverfitOracle 0.946,
Boredom 0.938+, chenwenqiang_001, Tucker) all converge on the same PL protocol.
Our prior PL attempts (exp 014/015/025) used **wrong protocols** — `loss_weight=0.5`
+ hardcoded 50/50 GT/PL batches — and regressed LB. The disclosed 1:1 weighting +
uniform sampling protocol had never been tested in our codebase.

| Phase | Commit |
|---|---|
| A — Scaffold metadata (proposal, tasks, 11 configs, preflight 7/7 PASS on v1) | `5aa5558` |
| B — Prereq tools (cache_inference.py + sbatch wrappers + generate_pseudo_labels.sh) | `0729e21` |
| C — Core code integration (loss_weight, PLDS, GTDatasetWrapper, MixSamp 3-source, collate) | `4687b1f` |
| D — 32 tests in `tests/033-own-teacher-pl/`; 195/195 PASS combined with exp 032 | `1a241dd` |
| E (code) — train.py + sbatch wrappers (fast-dev-run still pending at this point) | `34e3314` |

### Afternoon — Phase E gate cleared + Phase F runtime

| What | Commit / Job |
|---|---|
| Fast-dev-run PASS on a2_v5j (PL loaded, 5-tuple batches, KD active, no NaN) | `6ab8312`, job 9185603 |
| tasks.md Phase 0-3 marked complete | `a977661` |
| **T2 cache: smoke run** (LIMIT=5) — 300 preds, schema matches T1 | job 9185680 |
| **T2 cache: full run** (10,658 files × 5 folds × 12 windows) — 639,480 preds, walltime 36m44s | job 9187954 |
| **PL generator bugfix**: `cfg["perch"]` was hard-required; made optional via `use_perch` flag (5 conditional guards). Backward-compat verified for v1. | `84ab853` |
| **T2 PL parquet generated** (v2-distilled-sed-kd-5fold) — 210,375 PLs >0.5, 166 species covered, walltime 43s | job 9188496 |
| **v2 parquet preflight**: 7/7 PASS (identical schema + species ordering to v1) | (interactive) |

### Evening — num_workers regression + Phase G launch

| What | Commit / Job |
|---|---|
| **CRITICAL BUG: num_workers=0**. Exp 032 inherited 0 from Tucker's Jupyter notebook (single-threaded default). Comment said "revisit if I/O-bound" — exp 033 copy-pasted it. With 127K-file PL pool and no OS cache hits, single-threaded loading caused **4× slower training** (40 min/epoch vs predicted 10 min/epoch). Caught after 5 jobs at epoch 6 had burned 4h. | — |
| First T1 ablations launched (with workers=0) — 5 jobs at ~40 min/epoch | jobs 9188227-9188231 |
| **Fix 1**: exp 033 `_common.yaml`: `num_workers: 0 → 8` (matching all exps 025-031 and SBATCH `--cpus-per-task=8`). | `ecc3ccb` |
| **Fix 2**: exp 032 configs (3 files): same fix preventively (its 624-file SS pool fit in cache so it worked, but configs are still on disk). | `87d668d` |
| **Cancelled + relaunched 5 T1 ablations** with num_workers=8 → ~20 min/epoch (2× speedup; not quite the 4× predicted, but acceptable). Sunk 20 GPU-hours but saved ~8h calendar time on a1/a2/a5. | jobs 9190539-9190545 |
| proposal.md: per-batch composition column added, ⭐ marker on A2 canonical | `6185a77` |
| tasks.md Phase 4.1 updated with num_workers fix story | `11b25cd` |
| **All 5 T2 ablations launched** | jobs 9191658-9191662 |

---

## Key learnings from this session

1. **`num_workers` is not optional for large unique-file pools.** Exp 032 got away with `num_workers=0` because its 624-file labeled-SS pool fit in OS cache. Exp 033's 127K-file PL pool does not. Default to `num_workers=8` (matches `--cpus-per-task=8`).
2. **Tucker's Jupyter defaults are notebook defaults, not production defaults.** Jupyter is inherently single-threaded; `num_workers=0` works there but not in DataLoader-with-large-pool training.
3. **The PL generator's `perch:` config block is now optional.** Use pure-SED mode by omitting the block (Tier C species fall to zero, equivalent to Tier D).
4. **Per-batch composition matters more than ratio names.** "Focal/SC 0.5/0.5" is ambiguous; "32/0/32 (focal/sc/pl) at B=64" is unambiguous. proposal.md updated.

---

## The 10-ablation grid (running now)

| ID | Init | Sampler | Sampler share | Per-batch (B=64) | LR | Ep | Teacher | Config |
|---|---|---|---|---|---|---|---|---|
| A1.v5j | random | MixSamp | 0.50/0.00/0.50 | 32/0/32 | 5e-4 | 25 | T1 | a1_v5j.yaml |
| **A2.v5j ⭐** | random | **Uniform** | ~0.18/~0.00/~0.82 | ~12/0/~52 | 5e-4 | 25 | T1 | a2_v5j.yaml |
| A3.v5j | warm-start | MixSamp | 0.20/0.00/0.80 | 13/0/51 | 5e-5 | 10 | T1 | a3_v5j.yaml |
| A4.v5j | warm-start | **Uniform** | ~0.18/~0.00/~0.82 | ~12/0/~52 | 5e-5 | 10 | T1 | a4_v5j.yaml |
| A5.v5j | random | MixSamp | 0.70/0.00/0.30 | 45/0/19 | 5e-4 | 25 | T1 | a5_v5j.yaml |
| A1.kd / A2.kd⭐ / A3.kd / A4.kd / A5.kd | (same as v5j, T2 teacher) | | | | | | T2 | a*_kd.yaml |

- **T1** = v5j-4fold (LB 0.913) → `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet`
- **T2** = distilled-sed-kd-5fold (LB 0.915) → `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`
- **Warm-start ckpt**: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/last-v2.ckpt`

A2 (⭐ random + uniform = pure Antoine protocol) is the canonical recipe. Others
sweep init × sampler × focal/PL ratio around it as sensitivity ablations.

### Pair structure for reading
- A1↔A2, A3↔A4: sampler (MixSamp vs Uniform) within same init
- A1↔A3, A2↔A4: init (random vs warm-start) within same sampler
- A{i}.v5j ↔ A{i}.kd: teacher (T1 vs T2) within same protocol (5 paired tests)
- A1 vs A5: PL fraction sensitivity (0.5 vs 0.3 under MixSamp)

---

## Key methodology decisions (locked in)

1. **Naming convention**: dropped competitor author names from artifacts. References stay in proposal.md "Provenance" + "Background" sections only.
2. **Diff-only configs**: each ablation YAML is ~20-30 lines, inherits from `_common.yaml` via `_deep_merge` in `train.py:load_config`. Tucker recipe + KD + pseudo_label defaults live in `_common.yaml`.
3. **PL filter**: drop val-fold SS files (prevent leak) + drop labeled-SS train files (Antoine: GT used for labeled, PL only for unlabeled). Effective PL pool = 127,104 unlabeled segments per teacher.
4. **Sampler choice**: uniform = pure 1:1 protocol; MixSamp = explicit ratio control.
5. **KD loss stays unweighted**: only the BCE task branch gets per-sample loss_weight. Perch v2 MSE alignment is sample-confidence-independent.
6. **Schema contract**:
   - Tucker default dataloader → 4-tuple `(wavs, labels, wts, mks)`
   - PL-enabled dataloader → 5-tuple `(wavs, labels, wts, mks, loss_weights)`
   - `training_step` detects via `len(batch) > 4 and batch[4].ndim == 1`
7. **num_workers = 8** in both `_common.yaml` files (exp 032 + exp 033). Matches SBATCH allocation.

---

## File index (exp 033)

| File | Purpose |
|---|---|
| **NEW (exp 033)** | |
| `user/murilo/experiments/033-own-teacher-pl/proposal.md` | Hypothesis + falsifiable matrix + 10-ablation grid w/ per-batch composition |
| `user/murilo/experiments/033-own-teacher-pl/tasks.md` | Phase 0-4 checklist + closeout (num_workers bug captured in 4.1.0) |
| `user/murilo/experiments/033-own-teacher-pl/README.md` | Navigation + quickstart |
| `user/murilo/experiments/033-own-teacher-pl/pyproject.toml` | Pinned deps |
| `user/murilo/experiments/033-own-teacher-pl/configs/_common.yaml` | Shared Tucker recipe + KD + pseudo_label defaults (num_workers=8) |
| `user/murilo/experiments/033-own-teacher-pl/configs/a{1..5}_{v5j,kd}.yaml` | 10 ablation configs (diff-only) |
| `user/murilo/experiments/033-own-teacher-pl/configs/pl_gen_v2.yaml` | T2 PL gen config (pure-SED, no perch tier) |
| `user/murilo/experiments/033-own-teacher-pl/scripts/preflight_checks.py` | 7-check diagnostic + `--parquet-path` arg |
| `user/murilo/experiments/033-own-teacher-pl/scripts/cache_inference.py` | T2 ONNX inference cache |
| `user/murilo/experiments/033-own-teacher-pl/sbatch/{cache_inference,generate_pseudo_labels,_job,run}.sh` | SLURM wrappers |
| `user/murilo/experiments/033-own-teacher-pl/train.py` | Extends exp 032 train.py |
| `user/murilo/tests/033-own-teacher-pl/test_{loss_weighting,pseudo_label_integration,mixsamp_3source,pl_filter}.py` | 32 tests, all PASS |
| **MODIFIED (src/)** | |
| `user/murilo/src/birdclef/lightning/distilled_sed_finetune.py` | Optional `loss_weight` arg in `_dual_supervision_loss` + `training_step` |
| `user/murilo/src/birdclef/data/distilled_sed_dataset.py` | + PLDS class + GTDatasetWrapper (5→6-tuple) |
| `user/murilo/src/birdclef/data/distilled_sed_datamodule.py` | + 8 PL kwargs; PL parquet filter; sampler dispatch (uniform vs MixSamp); 5/6-tuple collate |
| `user/murilo/experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` | Made `cfg["perch"]` optional via `use_perch` flag (5 guards; backward-compat for v1) |
| **GENERATED ARTIFACTS** | |
| `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/fold{0..4}/logits.parquet` | T2 cache, 639,480 rows total |
| `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` | T2 PL parquet, 210,375 PLs>0.5, 166 species covered |
| **EXISTING (inherited from exp 032)** | |
| `user/murilo/src/birdclef/preprocessing/tucker_mel.py` | TuckerMelSpec (unchanged) |
| `user/murilo/src/birdclef/heads/sed_head.py` | AttHeadLogitSpace + GeMFreqPool (unchanged) |

---

## Pending decisions / open items for next session

1. **Verify 10 jobs completed** (squeue empty or only stragglers).
2. **D.5 e2e validation** on all 10 ckpts (mandatory pre-flight before LB).
3. **Pick top-5 by e2e val/auroc** for LB Day 1 submission.
4. **Decision matrix** (proposal.md § Falsifiable) → drives Round 2 / Sydorskyi-yoink / ensemble / pivot.
5. **Update results.md + MODEL-RUNS.md + this CONTEXT.md** with verdict after LB lands.
6. **Open PR** (still no PR yet — open after verdict).

---

## Branch / git state at session end (2026-05-26 ~23:00)

**Current branch**: `mg/exp033-own-teacher-pl`. 12 commits ahead of main, all pushed.

**Most recent commits** (latest first):
- `11b25cd` — tasks.md: T1/T2 launches + num_workers fix in Phase 4
- `6185a77` — proposal.md: per-batch composition column + canonical marker
- `87d668d` — exp 032 num_workers 0→8 (preventive)
- `ecc3ccb` — exp 033 num_workers 0→8 (the actual fix)
- `84ab853` — Phase F: Perch tier C optional in PL generator
- `a977661` — tasks.md: Phase 0/1/2/3 sub-items complete
- `6ab8312` — Phase E gate: fast-dev-run PASS on a2_v5j
- `fc71a78` — (obsolete) CONTEXT.md for end of scaffolding session
- `34e3314` — Phase E (partial): train.py + sbatch
- `1a241dd` — Phase D: 32 tests, 195/195 PASS
- `4687b1f` — Phase C: PL integration
- `0729e21` — Phase B: T2 prereq tools
- `5aa5558` — Phase A: scaffold
- `05f4063` — (main) Exp 032: full Tucker replication

**No PR yet for exp 033.** Open as ready-for-review after LB verdict lands.

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current overall best |
| distilled-sed-kd-5fold (exp 032 Phase 2.A) | **0.915** | Best Tucker-SED ensemble |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| distilled-sed-kd (exp 032 Phase D, single-fold) | 0.906 | Best Tucker-SED single; exp 033 baseline |
| **exp 033 target (single-fold CONFIRM)** | **≥ 0.920** | Antoine-style PL lift on top of 0.906 base |
| distilled-sed-kd-hgnet (exp 032 Phase 2.D) | 0.901 | HGNet backbone; refuted as bottleneck |
| distilled-sed-control (exp 032 Phase D) | 0.856 | Recipe alone w/o KD |

---

## Key reference docs

| Doc | What it contains |
|---|---|
| `experiments/033-own-teacher-pl/proposal.md` | Hypothesis, falsifiable matrix, method, per-batch grid, pool-size math, variables, limitations |
| `experiments/033-own-teacher-pl/tasks.md` | Phase 0-4 itemized checklist (num_workers fix story in 4.1.0) |
| `experiments/032-tucker-distilled-sed/results.md` | Phase D + Phase 2 LB results (CONFIRM verdict) |
| `notes/kaggle-discussion-non-perch-limit.md` | Antoine + OverfitOracle + Boredom recipe disclosures |
| `notes/2026-05-22-strategic-direction-tracks-A-B.md` | Track A (Tucker) → Track B (own-teacher PL) strategic context |
| `experiments/025-iter-pseudo-labeling/results.md` | Prior failed PL attempt — uses 0.5 weight + StratifiedBatchSampler (the protocols exp 033 fixes) |

---

## How to resume after disconnect

1. **Read this file top-to-bottom** (it's the source of truth for current state).
2. `git log --oneline -12` should show `11b25cd` at the top.
3. `squeue -u $USER` → all GPU jobs done (or only stragglers).
4. Verify 10 last.ckpt files exist (paths in § 1 above).
5. Launch D.5 e2e validation in parallel (§ 2 above).
6. Rank by e2e val/auroc → top-5 to LB Day 1.
7. Per `feedback_user_runs_commands.md`: don't run SLURM jobs / Kaggle uploads / kernel pushes yourself — give Murilo the commands.

**Slack / Kaggle / W&B credentials**: in `user/murilo/.env` (gitignored).
Loaded by `train.py`. If W&B login fails on SLURM, `WANDB_MODE=offline` or
`--fast-dev-run` skips W&B entirely.
