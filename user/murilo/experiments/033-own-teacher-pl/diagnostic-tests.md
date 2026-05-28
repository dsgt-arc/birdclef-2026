---
experiment: "033-own-teacher-pl"
phase: 5 (diagnostic)
status: ready-to-execute
created: 2026-05-27
author: Murilo Gustineli
---

# Exp 033 Phase 5 — Sydorskyi-divergence diagnostic tests

**Purpose**: Resolve whether exp 033's REJECT verdict (a4-kd 0.899, a2-kd 0.886, a4-v5j 0.897 vs baseline 0.906) reflects a real refutation of own-teacher PL OR implementation divergences from the published 0.93+ recipe.

**Status (2026-05-27)**: Audit complete (5 high-impact divergences identified). Test 1 code + configs landed; ready to launch. Tests 2, 3 deferred until Test 1 LB lands.

**Reference**: `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` — full audit with 20-axis divergence table and detailed analysis. Read it first if you're returning to this work.

**Budget**: ~1.5 days for Test 1, +1.5 days each for Tests 2-3. Total ~3-4 days of the 7 remaining before 2026-06-03 deadline.

---

## Background (1-paragraph TL;DR)

Exp 033 REJECT verdict is suspicious because it landed below the baseline despite "matching" the disclosed Antoine protocol. The audit found that our implementation diverges from the only published 0.93+ PL recipe (Sydorskyi 2025 2nd place) on 16 of 20 axes — most importantly, we never implemented the **focal × PL MixUp fusion** (Sydorskyi's "cleverest piece"), the **trim probability** filter (Ali Ozan flagged this in 2026-05-27 thread), or the **chunk discard** filter. The REJECT verdict therefore can't be cleanly read as a refutation of the protocol — we never ran the protocol. The diagnostic tests below fix divergences incrementally and re-test.

---

## Current state checklist

Read this section first if returning to this work after a break.

- [x] Audit committed (`40708f7` — `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md`)
- [x] `generate_pseudo_labels.py` modified with optional `filter:` section (`min_max_prob`, `trim_min_prob`)
- [x] `configs/pl_gen_v3.yaml` created (T2 source + Sydorskyi-canonical 0.5/0.1 filter values)
- [x] `configs/a2_kd_v3.yaml` created (a2-kd canonical Antoine cell pointed at v3 parquet)
- [ ] Test 1: re-issue v3 PL parquet (run `generate_pseudo_labels.py --config pl_gen_v3.yaml` — head node, ~30s)
- [ ] Test 1: pre-flight v3 parquet (`scripts/preflight_checks.py --parquet-path ...`)
- [ ] Test 1: train a2-kd-v3 (sbatch — 25 ep ~6.5h on rtx6000)
- [ ] Test 1: D.5 e2e validation
- [ ] Test 1: LB submit + read decision
- [ ] Test 2 (conditional, if Test 1 LB ≥ 0.906): focal × PL fusion in FocalDS
- [ ] Test 3 (conditional, if Tests 1-2 inconclusive): add focal loss term

---

## Test 1 — Chunk discard + trim probability

**What it fixes** (divergences #2 + #3 from the audit):
- Drop PL chunks where `max(prob) < 0.5` (Sydorskyi: ~83% discard rate).
- Zero per-class probs `< 0.1` (Sydorskyi `trim_min_prob`).

**What it does NOT fix**: focal × PL MixUp fusion (#1), 40% swap sampling (#4), focal loss (#5). These remain divergent in Test 1.

**Why this is Test 1**: lowest cost, highest information. ~1h work, 30 min compute. If a2-kd-v3 lifts to >0.906, we have evidence the protocol is recoverable; if it doesn't, we know these two filters alone aren't the problem and we can target Test 2 with confidence.

### Step 1.1 — Re-issue v3 PL parquet (~30s on head node)

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026
~/clef/birdclef-2026/.venv/bin/python user/murilo/experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py \
    --config user/murilo/experiments/033-own-teacher-pl/configs/pl_gen_v3.yaml
```

**Expected output**:
- Initial merge stats show ~120K+ chunks with various >0.5 counts (same as v2).
- "Filter: chunk discard (min_max_prob=0.5)" block reports ~120K → ~20K chunks (~83% discard expected per Sydorskyi).
- "Filter: per-class trim (trim_min_prob=0.1)" block reports several million zeroed cells.
- Final parquet written to `/storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/v3-distilled-sed-kd-5fold-trim01-discard05/sed_pseudo_labels.parquet`.

**Sanity check**:
- `~20K rows × 234 cols` parquet (down from `127K × 234` in v2).
- Each row should be sparse (many zeros after trim).

### Step 1.2 — Pre-flight verify v3 parquet (~5 min)

Use the existing pre-flight script which can target either parquet:

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026
~/clef/birdclef-2026/.venv/bin/python user/murilo/experiments/033-own-teacher-pl/scripts/preflight_checks.py \
    --parquet-path /storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/v3-distilled-sed-kd-5fold-trim01-discard05/sed_pseudo_labels.parquet
```

**Expected**: 7/7 PASS (same as v1/v2 baselines). Schema must match (filename + chunk_idx + 234 species cols). Species ordering must match.

### Step 1.3 — Train a2-kd-v3 (~6.5h on rtx6000)

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/experiments/033-own-teacher-pl
bash sbatch/run.sh configs/a2_kd_v3.yaml --fold 0
```

**Note**: Smaller PL pool (~20K vs 127K) means the uniform sampler will see PL ~50% of slots instead of 81%. This is a side-effect of chunk discard, not a separate change.

**Caveat**: `steps_per_epoch` is computed from total pool size. With a smaller PL pool, total = 28906 focal + 624 sc + ~20K pl ≈ 49K. At BS=64 → ~770 steps/epoch. Should still see meaningful PL exposure.

### Step 1.4 — D.5 end-to-end validation (~5 min)

Reuse the existing D.5 launcher with a single-cell override, OR run directly:

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/experiments/032-tucker-distilled-sed
CKPT=/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/033-own-teacher-pl/checkpoints/2026-exp033-a2-kd-v3/fold0/last.ckpt \
CONFIG=../033-own-teacher-pl/configs/a2_kd_v3.yaml \
FOLD=0 \
TRAIN_AUROC=$(jq -r '.val_metrics["val/auroc"]' \
  ../033-own-teacher-pl/outputs/test_results/2026-exp033-a2-kd-v3-fold0.json) \
sbatch sbatch/validate_end_to_end.sh
```

**Gate**: `|train_auroc − e2e_auroc| ≤ 0.02` PASS.

### Step 1.5 — Export ONNX + LB submit

Reuse the exp 010 pipeline already wired for exp 033 in commit `4280e72`. New ablation needs a new pipeline entry:

```bash
# Add to scripts/upload_kaggle.py MODELS dict:
#   "distilled-sed-pl-a2-kd-v3": { slug, weights_dir, kernel_metadata }
# Add to sbatch/export.sbatch dispatch + species-mapping case blocks.
# Generate kaggle_kernel/kernel-metadata-distilled-sed-pl-a2-kd-v3.json.
# Add slug to kaggle_kernel/submission.py MODEL_SLUGS.

cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/experiments/010-submission-pipeline
bash sbatch/run.sh export --model distilled-sed-pl-a2-kd-v3
python scripts/upload_kaggle.py --model distilled-sed-pl-a2-kd-v3
sleep 180
python scripts/upload_kaggle.py --model distilled-sed-pl-a2-kd-v3 --push-kernel
# Click Submit to Competition in Kaggle UI.
```

### Step 1.6 — Read result, decide

| LB outcome | Reading | Next move |
|---|---|---|
| **LB > 0.913** | trim+discard explain REJECT alone. Protocol is recoverable on our base. | Scope a full v3 re-run of the strongest cells (a3-kd-v3, a4-kd-v3, etc.). |
| **LB 0.906-0.913** | Modest lift; trim+discard help but other divergences still matter. | Run Test 2 (focal × PL fusion). |
| **LB 0.886-0.906** | Trim+discard alone don't help. Probably implementation issue or wrong divergence isolated. | Run Test 2; if it also fails, pivot. |
| **LB < 0.886** | Worse than original a2-kd. Filters HURT — unlikely but possible if signal was in low-confidence chunks. | Investigate; possibly skip Test 2. |

Record in `results.md` under a new "Phase 5 — diagnostic" section.

---

## Test 2 — focal × PL MixUp fusion (CONDITIONAL on Test 1)

**What it fixes** (divergence #1 from the audit — CRITICAL):
- Implement Sydorskyi's "cleverest piece": when a focal sample fires its MixUp branch and the partner is a PL chunk, mix waveforms 50/50 AND sum-clip the targets (focal hard + PL soft → mixed target in [0, 1]^234).

**Effort**: ~0.5 day code + ~6.5h training.

### Step 2.1 — Add Path D to FocalDS

Modify `user/murilo/src/birdclef/data/distilled_sed_dataset.py:FocalDS`:

1. Add new constructor args:
   - `pl_pool: PLDataPool | None = None` — wraps PLDS for sampling from
   - `focal_pl_mixup_prob: float = 0.5` — fire probability

2. Add Path D in `__getitem__` BEFORE Path A (focal × focal):
```python
# --- Path D: Focal × PL fusion mixup (Sydorskyi 2025 mechanism) ---
if (self.aug and self.pl_pool is not None
        and self.rng.random() < self.focal_pl_mixup_prob):
    pl_wav, pl_soft = self.pl_pool.sample(self.rng)
    if pl_wav is not None:
        ch_mix = (0.5 * ch1 + 0.5 * pl_wav).astype(np.float32)
        ch_mix = apply_aug(ch_mix, self.rng, aug_prob=self.aug_prob)
        # SUM AND CLIP (not max) — fuses hard GT + soft teacher
        lb = np.clip(lb1 + pl_soft, 0.0, 1.0)
        self.mixup_counts["focal_pl"] += 1
        return (
            torch.from_numpy(ch_mix).unsqueeze(0),
            torch.from_numpy(lb).float(),
            torch.ones(self.num_classes, dtype=torch.float32),
            torch.ones(self.num_classes, dtype=torch.float32),
            "focal",  # tag — model treats it as a focal sample
        )
```

3. Define `PLDataPool` (in same file or in a new module) — lightweight wrapper that:
   - Holds `pl_df` (the v3 parquet) and a `soundscape_dir`
   - On `sample(rng)`: pick a random row, load the wav chunk, return `(wav, soft_label_vec)`
   - Reuses the existing `audio_loader` semantics from PLDS

### Step 2.2 — Wire PLDataPool into DataModule

Modify `DistilledSEDDataModule.train_dataloader()`:

When `pseudo_label_path` is set AND `pseudo_label_sampler == "focal_fusion"` (new option):
- Build `PLDataPool` instead of PLDS
- Pass to FocalDS constructor as `pl_pool=...`
- No ConcatDataset / MixSamp needed — just standard FocalDS + ScDS path

Add a new ablation config `a2_kd_v3_fusion.yaml`:
```yaml
run_name: "2026-exp033-a2-kd-v3-fusion"
pseudo_label:
  path: ".../v3-...trim01-discard05/sed_pseudo_labels.parquet"
  sampler: "focal_fusion"
  exclude_labeled_sc: true
  weight: 1.0  # unused in fusion path (loss_weight semantics don't apply)
training:
  focal_pl_mixup_prob: 0.5  # Sydorskyi default
```

### Step 2.3 — Sanity tests

- Unit test: a focal sample + a PL sample → fusion path → mixed wav with `(lb1 + pl_soft).clip(0, 1)` target.
- Smoke test (`--fast-dev-run`): verify `focal_pl` mixup_count > 0; verify targets stay in [0, 1].
- Backward-compat: existing a2_kd.yaml (with `sampler: "uniform"`) should still work unchanged.

### Step 2.4 — Train + LB

Same flow as Test 1 (Steps 1.3-1.5) with `--config configs/a2_kd_v3_fusion.yaml`.

### Step 2.5 — Decision

| LB outcome | Reading | Next move |
|---|---|---|
| **LB > 0.920** | Sydorskyi fusion mechanism transfers. Strong recovery. | Scale to other cells (a3, a4 × T1/T2) with v3+fusion. |
| **LB 0.913-0.920** | Substantial lift but below target. | Try Test 3 (focal loss). |
| **LB 0.906-0.913** | Modest lift. | Try Test 3 and/or accept the partial recovery. |
| **LB < 0.906** | Fusion doesn't help on its own. | Strong evidence base-quality is the ceiling. Pivot to Sydorskyi-yoink (Path A). |

---

## Test 3 — Add focal loss (CONDITIONAL on Tests 1, 2 inconclusive)

**What it fixes** (divergence #5):
- Add `sigmoid_focal_loss(α=0.25, γ=2.0)` to the BCE branch in `_dual_supervision_loss`.
- Matches Sydorskyi: `loss = 1.0 * BCE + 1.0 * focal`.

**Effort**: ~2h code + retrain.

### Step 3.1 — Modify Lightning module

`user/murilo/src/birdclef/lightning/distilled_sed_finetune.py:_dual_supervision_loss`:

```python
def _dual_supervision_loss(self, clip_logits, frame_logits, labels, loss_weight=None):
    frame_max = frame_logits.max(dim=-1).values
    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, labels.float())
    bce_frame = F.binary_cross_entropy_with_logits(frame_max, labels.float())
    loss = self.hparams.clipwise_weight * bce_clip + self.hparams.framewise_weight * bce_frame
    if self.hparams.get("focal_loss_weight", 0.0) > 0:
        from torchvision.ops import sigmoid_focal_loss
        focal_clip = sigmoid_focal_loss(
            clip_logits, labels.float(),
            alpha=0.25, gamma=2.0, reduction="mean"
        )
        loss = loss + self.hparams.focal_loss_weight * focal_clip
    return loss, frame_max
```

Add to `_common.yaml`:
```yaml
training:
  focal_loss_weight: 0.0  # default OFF for backward compat
```

Test 3 config: `a2_kd_v3_fusion_focal.yaml` with `focal_loss_weight: 1.0`.

### Step 3.2 — Train + LB

Same flow as before.

---

## If all 3 tests fail (LB < 0.906)

This is the **genuine refutation** outcome. Pivot decisively:

1. **Document the negative result.** All-tests-failed is a meaningful scientific finding — it would be 25th+ datapoint in our offline-LB anti-correlation pattern, but specifically calibrated against Sydorskyi's published mechanism.
2. **Pivot to Sydorskyi-yoink (Path A).** Download `vladimirsydor/bird-clef-2025-all-pretrained-models`, attach 1028-attention-dim head, fine-tune on 2026. This is the only remaining validated path to 0.94+ disclosed by competitors.
3. **Skip remaining exp 033 work.** Don't run a3-kd, a3-v5j (Day 1) or any Day 2 cells.

---

## File index

| File | Purpose |
|---|---|
| `experiments/033-own-teacher-pl/diagnostic-tests.md` | THIS file — Phase 5 workflow |
| `analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` | Full 20-axis divergence analysis |
| `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` | Modified 2026-05-27 to support `filter:` section |
| `experiments/033-own-teacher-pl/configs/pl_gen_v3.yaml` | NEW — T2 + Sydorskyi-canonical 0.5/0.1 filter |
| `experiments/033-own-teacher-pl/configs/a2_kd_v3.yaml` | NEW — a2-kd canonical Antoine on v3 parquet |
| `experiments/033-own-teacher-pl/configs/pl_gen_v2.yaml` | v2 config (no filter) — kept for comparison |
| `experiments/033-own-teacher-pl/configs/a2_kd.yaml` | a2-kd v2 config (LB 0.886) — kept for paired Δ |
| `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` | v2 parquet, no filter (127K chunks) |
| `shared/pseudo-labels/birdclef-2026/v3-distilled-sed-kd-5fold-trim01-discard05/...` | v3 parquet, TO BE GENERATED in Step 1.1 (~20K chunks expected) |

## Naming convention reminder

- `v1` = HGNet v5j-4fold teacher (LB 0.913), no filter
- `v2` = distilled-sed-kd-5fold teacher (LB 0.915), no filter
- `v3` = distilled-sed-kd-5fold teacher + trim 0.1 + discard 0.5 (Sydorskyi filters)
- `v4` (reserved) = above + focal × PL fusion in FocalDS
- `v5` (reserved) = above + focal loss

Naming the parquets, configs, and run_names with the version number keeps the chain of evidence clear when we look at this in three days and forget what's what.
