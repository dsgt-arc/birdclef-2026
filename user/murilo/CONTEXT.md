# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-26 (end of exp 033 scaffolding session — Phases A-D
DONE + Phase E code DONE, pending fast-dev-run + Phase F/G runtime work)

**Competition deadline**: 2026-06-03 (8 days from this update).

---

## Where we are

**Exp 032 — Tucker distilled-SED full replication: COMPLETE (PARTIAL REPLICATION)**
- PR #51 merged to main as commit `05f4063` (Phase D + Phase 2 full work).
- Best Tucker-SED single: `distilled-sed-kd` (Phase D, fold 0) = **LB 0.906**.
- Best Tucker-SED ensemble: `distilled-sed-kd-5fold` = **LB 0.915** (within ~0.002 of Tucker's published 0.917).
- HGNet+KD ablation: LB 0.901 → **backbone was NOT the bottleneck** (refuted).
- Fold 4 diagnostic: LB 0.903 despite offline +0.023 over fold 0 → **23rd data point of offline-LB anti-correlation**.
- Project best overall (UNCHANGED): exp 023 `perch-sed-ensemble-mlp-only` = **0.929 LB**.

**Exp 033 — Own-teacher pseudo-labels on distilled-sed-kd: SCAFFOLDED (Phases A-D done + Phase E code done)**
- Branch: `mg/exp033-own-teacher-pl` (5 commits ahead of main, all pushed).
- Hypothesis: 1:1 own-teacher PL (Antoine/OverfitOracle/Boredom protocol) on Phase D `distilled-sed-kd` → LB ≥ 0.920 (CONFIRM).
- 10-ablation grid: 5 protocols × 2 teachers (T1=v5j-4fold, T2=distilled-sed-kd-5fold).
- All source code changes made + backward-compat verified (195/195 tests PASS).
- **Fast-dev-run sanity check NOT yet executed** (compute session ended; flag for next session).

---

## Pick up here (next session) — 2026-05-26

### 0. Verify session state

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026
git checkout mg/exp033-own-teacher-pl
git log --oneline -8
# Expected most recent: 34e3314 → 1a241dd → 4687b1f → 0729e21 → 5aa5558 → 05f4063 ...

# Verify tests still pass (no source drift since last session)
~/clef/birdclef-2026/.venv/bin/python -m pytest \
    user/murilo/tests/032-tucker-distilled-sed/ \
    user/murilo/tests/033-own-teacher-pl/ -q
# Expected: 195 passed (163 exp032 + 32 exp033)
```

### 1. Fast-dev-run sanity (MANDATORY before Phase F/G)

Phase E left this undone. Run on `a2_v5j.yaml` (simplest config: random init,
uniform sampler, existing v1 PL parquet — no warm-start, no T2 PLs required):

```bash
cd user/murilo/experiments/033-own-teacher-pl
WANDB_MODE=offline ~/clef/birdclef-2026/.venv/bin/python train.py \
    --config configs/a2_v5j.yaml \
    --fast-dev-run \
    --num-workers 0
```

Verify in the log:
- `Exp 033 PL mode enabled: /storage/.../v1/sed_pseudo_labels.parquet`
- `DistilledSEDDataModule PL filter: 127,896 → 127,728 → 127,104 segments` (val-fold + labeled-sc filters)
- `train_dataloader (PL/uniform): focal=..., sc=..., pl=127104, total=~150K`
- 1 training step + 1 validation step complete without NaN/Inf
- Output mentions `loss_weight` somewhere in the loss path (5-tuple batch detected)

If fast-dev-run PASSES, proceed to Phase F. If it FAILS, diagnose before Phase F.

### 2. Phase F — generate T2 PLs (~3hr runtime; do NOT run if fast-dev-run fails)

```bash
cd user/murilo/experiments/033-own-teacher-pl

# Step 0.1: cache distilled-sed-kd-5fold ONNX inference on all 10,658 soundscapes
sbatch sbatch/cache_inference.sh
# ~30-60 min on rtx6000. Writes to:
#   shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/fold{0..4}/logits.parquet

# Step 0.2: generate PL parquet from the cache (after Step 0.1 completes)
sbatch sbatch/generate_pseudo_labels.sh
# ~10-30 min on CPU. Writes:
#   shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet

# Step 0.3: validate the new T2 parquet (re-runs 7/7 PASS checks on v2)
~/clef/birdclef-2026/.venv/bin/python scripts/preflight_checks.py \
    --parquet-path /storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet
```

### 3. Phase G — submit 10 ablations (parallel, ~4.5hr walltime each)

```bash
cd user/murilo/experiments/033-own-teacher-pl
for AB in a1 a2 a3 a4 a5; do
  for T in v5j kd; do
    bash sbatch/run.sh configs/${AB}_${T}.yaml --fold 0
  done
done
# All 10 jobs queue in parallel on gpu-rtx6000.
# T1 ablations (a*_v5j) can run BEFORE Phase F completes — they use existing v1 parquet.
# T2 ablations (a*_kd) need Phase F to complete first.
```

### 4. D.5 end-to-end validation + LB submission (after training)

D.5 script is in exp 032; reuse it pointing at exp 033 ckpts:

```bash
cd user/murilo/experiments/032-tucker-distilled-sed
# For each of 10 new ckpts:
CKPT=/storage/project/.../2026-exp033-<ablation>-<teacher>/last.ckpt \
CONFIG=../033-own-teacher-pl/configs/<ablation>_<teacher>.yaml \
FOLD=0 \
TRAIN_AUROC=<from metrics.json> \
sbatch sbatch/validate_end_to_end.sh
```

Rank by e2e val/auroc. Submit top 5 to LB Day 1, top 5 Day 2 (5/day cap).

### 5. Apply decision matrix (exp 033 proposal.md § Falsifiable)

| Bucket | LB lift over 0.906 | Reading |
|---|---|---|
| **CONFIRM** | ≥ +0.014 | Round 2 (iterate teacher) + Sydorskyi-yoink for XC base |
| **PARTIAL** | +0.007 to +0.013 | Ensemble with exp 023 (0.929) |
| **WEAK** | +0.001 to +0.006 | Extend to 5-fold to disambiguate |
| **REJECT** | ≤ 0 | Pivot to Sydorskyi-yoink directly |

### 6. PR + closeout

Open draft PR for exp 033 after Phase E fast-dev-run PASS:
```bash
gh pr create --draft --title "Exp 033: own-teacher pseudo-labels on distilled-sed-kd" \
  --body "<summary>" --base main --head mg/exp033-own-teacher-pl
```

Update on completion (Phase G results): mark non-draft, squash-merge per `feedback_squash_merge.md`.

---

## What landed this session (2026-05-26)

### Exp 032 close-out (early session)

- PR #51 merged to main as squash commit `05f4063` (38 commits → 1).
- All exp 032 docs (proposal, tasks, results, MODEL-RUNS) updated with final LB outcomes.

### Exp 033 design + scaffolding (rest of session)

Per the 2026-05-22 strategic direction note (Track B = own-teacher PL on top of Track A = Tucker), 5 disclosed competitor recipes (Antoine 0.943, OverfitOracle 0.946, Boredom 0.938+, chenwenqiang_001, Tucker) all converge on the same PL protocol. Our prior PL attempts (exp 014/015/025) used **wrong protocols** — `loss_weight=0.5` + hardcoded 50/50 GT/PL batches — and regressed LB. The disclosed 1:1 weighting + uniform sampling protocol has **never been tested** in our codebase.

### Phases completed this session

| Phase | What | Commit |
|---|---|---|
| **A — Scaffold metadata** | proposal.md, tasks.md, README.md, pyproject.toml, 11 ablation configs (_common.yaml + a{1..5}_{v5j,kd}.yaml + pl_gen_v2.yaml), preflight_checks.py (7/7 PASS) | `5aa5558` |
| **B — Prereq tools** | scripts/cache_inference.py (5-fold ONNX → per-fold logits.parquet), sbatch/cache_inference.sh, sbatch/generate_pseudo_labels.sh (invokes exp 025 generator with v2 config) | `0729e21` |
| **C — Core code integration** | DistilledSEDFinetuneModule.training_step + _dual_supervision_loss accept optional loss_weight; PLDS + GTDatasetWrapper added to distilled_sed_dataset.py; DistilledSEDDataModule gets 8 new PL kwargs + sampler dispatch (uniform vs MixSamp); _collate_drop_extras handles 5-tuple → 4-tuple AND 6-tuple → 5-tuple; backward-compat verified (163/163 exp 032 tests PASS unchanged) | `4687b1f` |
| **D — Tests** | 32 new tests in tests/033-own-teacher-pl/: test_loss_weighting (9), test_pseudo_label_integration (11), test_mixsamp_3source (8), test_pl_filter (4). Combined 195/195 PASS (163 + 32). | `1a241dd` |
| **E (partial) — train.py + sbatch** | exp 033 train.py with _deep_merge for _common.yaml inheritance + --pseudo-label-path arg + warm-start ckpt loading; sbatch wrappers copied + renamed. **Fast-dev-run NOT yet executed.** | `34e3314` |

---

## The 10-ablation grid

| ID | Init | Sampler | Focal/SC | LR | Epochs | Teacher | Config file |
|---|---|---|---|---|---|---|---|
| A1.v5j | random | MixSamp | 0.5/0.0/0.5 | 5e-4 | 25 | T1 | a1_v5j.yaml |
| A2.v5j | random | **Uniform** | natural ~17/0/83 | 5e-4 | 25 | T1 | a2_v5j.yaml |
| A3.v5j | **warm-start** | MixSamp | 0.2/0.0/0.8 | 5e-5 | 10 | T1 | a3_v5j.yaml |
| A4.v5j | **warm-start** | **Uniform** | natural | 5e-5 | 10 | T1 | a4_v5j.yaml |
| A5.v5j | random | MixSamp | 0.7/0.0/0.3 | 5e-4 | 25 | T1 | a5_v5j.yaml |
| A1.kd / A2.kd / A3.kd / A4.kd / A5.kd | (same as v5j, but T2 teacher) | | | | | T2 | a*_kd.yaml |

- **T1** = v5j-4fold (LB 0.913, existing `v1/sed_pseudo_labels.parquet` — pre-flight 7/7 PASS).
- **T2** = distilled-sed-kd-5fold (LB 0.915, **parquet to be generated in Phase F**).
- **Warm-start ckpt**: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/2026-exp032-distilled-sed-kd/last-v2.ckpt` (Phase D fold 0, LB 0.906).

---

## Key methodology decisions (locked in)

1. **Naming convention**: dropped competitor author names from artifacts (branch, dirs, configs). References to disclosed protocols (Antoine, OverfitOracle, Boredom) stay in proposal.md "Provenance" + "Background" sections.
2. **Diff-only configs**: each ablation YAML is ~20-30 lines, inherits from `_common.yaml` via `_deep_merge` in `train.py:load_config`. Tucker recipe + KD + pseudo_label defaults live in `_common.yaml`.
3. **PL filter**: drop val-fold SS files (prevent leak) + drop labeled-SS train files (Antoine: GT used for labeled, PL only for unlabeled). Effective PL pool = 127,104 unlabeled segments per teacher.
4. **Sampler choice**: uniform = pure 1:1 protocol (treat PL as labeled, natural pool proportions). MixSamp = explicit ratio control (focal/sc/pl). Both supported via `DistilledSEDDataModule.pseudo_label_sampler` config.
5. **KD loss stays unweighted**: only the BCE task branch gets per-sample loss_weight. Perch v2 MSE alignment is sample-confidence-independent.
6. **Schema contract**:
   - Tucker default dataloader → 4-tuple `(wavs, labels, wts, mks)`
   - PL-enabled dataloader → 5-tuple `(wavs, labels, wts, mks, loss_weights)`
   - `training_step` detects via `len(batch) > 4 and batch[4].ndim == 1`

---

## Pool-size sampler math (justifies the ablation grid)

With 26K focal + ~0.6K labeled SS + 127K PL segments, `nst = 2390 steps/epoch`:

| Setup | Focal exposure/epoch | PL exposure/epoch |
|---|---:|---:|
| Tucker 0.9/0.1 (naive transplant) | 5.3× | **0.001×** ← invisible |
| A5: 0.7/0.3 | 4.1× | 0.003× |
| A1: 0.5/0.5 | 2.9× | 0.006× |
| A3: 0.2/0.8 | 1.2× | 0.010× |
| A2/A4: Uniform | **1.0×** | **1.0×** |

Tucker's 0.9/0.1 was tuned for 792 SS. With 127K PL it makes the PL pool effectively invisible. Only uniform sampling achieves Antoine's "treat as labeled" semantics structurally.

---

## File index (exp 033)

| File | Purpose |
|---|---|
| **NEW (exp 033)** | |
| `user/murilo/experiments/033-own-teacher-pl/proposal.md` | Hypothesis + falsifiable success matrix + method + variables |
| `user/murilo/experiments/033-own-teacher-pl/tasks.md` | Phase 0-4 checklist + closeout |
| `user/murilo/experiments/033-own-teacher-pl/README.md` | Navigation + quickstart |
| `user/murilo/experiments/033-own-teacher-pl/pyproject.toml` | Pinned deps (mirrors exp 032) |
| `user/murilo/experiments/033-own-teacher-pl/configs/_common.yaml` | Shared Tucker recipe + KD + pseudo_label defaults |
| `user/murilo/experiments/033-own-teacher-pl/configs/a{1..5}_{v5j,kd}.yaml` | 10 ablation configs (diff-only) |
| `user/murilo/experiments/033-own-teacher-pl/configs/pl_gen_v2.yaml` | T2 PL generation config (invokes exp 025 generator) |
| `user/murilo/experiments/033-own-teacher-pl/scripts/preflight_checks.py` | 7-check diagnostic + `--parquet-path` arg for v2 |
| `user/murilo/experiments/033-own-teacher-pl/scripts/cache_inference.py` | T2 ONNX inference cache (5-fold → per-fold logits.parquet) |
| `user/murilo/experiments/033-own-teacher-pl/sbatch/cache_inference.sh` | SLURM wrapper for cache_inference.py |
| `user/murilo/experiments/033-own-teacher-pl/sbatch/generate_pseudo_labels.sh` | SLURM wrapper for exp 025's generator with pl_gen_v2.yaml |
| `user/murilo/experiments/033-own-teacher-pl/sbatch/{_job.sh,run.sh}` | Training launchers (copied from exp 032 + renamed) |
| `user/murilo/experiments/033-own-teacher-pl/train.py` | Extends exp 032 train.py: load_config _common.yaml merge + --pseudo-label-path + DataModule PL kwargs + warm-start ckpt |
| `user/murilo/tests/033-own-teacher-pl/test_loss_weighting.py` | 9 tests: _dual_supervision_loss + training_step 2/4/5-tuple |
| `user/murilo/tests/033-own-teacher-pl/test_pseudo_label_integration.py` | 11 tests: PLDS + GTDatasetWrapper + ConcatDataset + collate |
| `user/murilo/tests/033-own-teacher-pl/test_mixsamp_3source.py` | 8 tests: 3-source MixSamp + 2-source backward-compat |
| `user/murilo/tests/033-own-teacher-pl/test_pl_filter.py` | 4 tests: val-fold + labeled-sc filters + soundscape_dir required |
| **MODIFIED (src/)** | |
| `user/murilo/src/birdclef/lightning/distilled_sed_finetune.py` | _dual_supervision_loss + training_step: optional loss_weight (backward-compat) |
| `user/murilo/src/birdclef/data/distilled_sed_dataset.py` | + PLDS class (6-tuple soft-label loader) + GTDatasetWrapper (5→6-tuple) |
| `user/murilo/src/birdclef/data/distilled_sed_datamodule.py` | + 8 PL kwargs in __init__; setup() loads + filters PL parquet; train_dataloader() dispatches uniform vs MixSamp; _collate_drop_extras handles both 5- and 6-tuple inputs |
| **EXISTING (inherited from exp 032)** | |
| `user/murilo/src/birdclef/preprocessing/tucker_mel.py` | TuckerMelSpec (no change) |
| `user/murilo/src/birdclef/heads/sed_head.py` | AttHeadLogitSpace + GeMFreqPool (no change) |

---

## Pending decisions / open items for next session

1. **Fast-dev-run on a2_v5j** (~5 min) — gate to Phase F/G.
2. **Phase F runtime** (~3hr total): cache_inference.sh → generate_pseudo_labels.sh → preflight on v2 parquet.
3. **Phase G runtime** (~4.5hr training × 10 parallel + LB cap 5/day = ~3 LB days).
4. **Decision matrix** (proposal.md § Falsifiable) → drives Round 2 / Sydorskyi-yoink / ensemble / pivot.
5. **Update results.md + MODEL-RUNS.md + this CONTEXT.md** with verdicts after LB lands.
6. **PR for exp 033** — open as draft after fast-dev-run PASS; squash-merge after Phase G verdict.

---

## Branch / git state at session end

- **Current branch**: `mg/exp033-own-teacher-pl`. 5 commits ahead of main, all pushed.
- **Most recent commits**:
  - `34e3314` — Phase E (partial): train.py + sbatch (fast-dev-run pending)
  - `1a241dd` — Phase D: 32 tests, 195/195 PASS
  - `4687b1f` — Phase C: PL integration in DistilledSEDFinetuneModule + DataModule
  - `0729e21` — Phase B: T2 prereq tools (inference cache + PL generation)
  - `5aa5558` — Phase A: scaffold proposal + tasks + 10 ablation configs
  - `05f4063` — (main) Exp 032: Tucker distilled-SED full replication (PARTIAL REPLICATION)
- **No PR yet for exp 033** (waiting on fast-dev-run gate).

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | Current overall best (unchanged this session) |
| distilled-sed-kd-5fold (exp 032 Phase 2.A) | **0.915** | Best Tucker-SED ensemble; within ~0.002 of Tucker's 0.917 |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble (slightly above ours; cross-hardware variance) |
| distilled-sed-kd (exp 032 Phase D, single-fold) | 0.906 | Best Tucker-SED single; matches v5j-original |
| **exp 033 target (single-fold CONFIRM)** | **≥ 0.920** | Antoine-style PL lift on top of 0.906 base |
| distilled-sed-kd-hgnet (exp 032 Phase 2.D) | 0.901 | HGNet backbone; refuted as bottleneck |
| distilled-sed-control (exp 032 Phase D) | 0.856 | Recipe alone w/o KD |

---

## Key reference docs

| Doc | What it contains |
|---|---|
| `experiments/033-own-teacher-pl/proposal.md` | Hypothesis, falsifiable matrix, method, pool-size sampler math, variables, limitations |
| `experiments/033-own-teacher-pl/tasks.md` | Phase 0-4 itemized checklist |
| `experiments/032-tucker-distilled-sed/results.md` | Phase D + Phase 2 LB results (CONFIRM verdict) |
| `notes/kaggle-discussion-non-perch-limit.md` | Antoine + OverfitOracle + Boredom recipe disclosures |
| `notes/2026-05-22-strategic-direction-tracks-A-B.md` | Track A (Tucker) → Track B (own-teacher PL) strategic context |
| `experiments/025-iter-pseudo-labeling/results.md` | Prior failed PL attempt — uses 0.5 weight + StratifiedBatchSampler (the protocols exp 033 fixes) |

---

## How to resume after disconnect

1. Read this file top-to-bottom.
2. `git log --oneline -8` should show `34e3314` → `1a241dd` → ... most recent first.
3. Verify tests: `~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/032-tucker-distilled-sed/ user/murilo/tests/033-own-teacher-pl/ -q` → expect **195 passed**.
4. Run fast-dev-run sanity (§ 1 above) before any Phase F/G work.
5. If fast-dev-run PASS → proceed to Phase F (T2 PL generation).
6. If fast-dev-run FAIL → diagnose; do NOT proceed.

**Slack / Kaggle / W&B credentials**: in `user/murilo/.env` (gitignored).
Loaded by `train.py`. If W&B login fails on SLURM, `WANDB_MODE=offline` or
`--fast-dev-run` skips W&B entirely.
