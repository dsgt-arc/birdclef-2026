# 2026-05-10 — v5j codebase drift: training pipeline cannot reproduce v5j-original 0.906 LB

## TL;DR

Three retrains of the v5j recipe on today's codebase (seed=42, same config, same data,
deterministic training) all land at LB **0.896-0.897**, **−0.009 below** v5j-original's
0.906 LB. The inference pipeline is verified clean (v5j-resubmit reproduces 0.906 from
the original ep 19 checkpoint). The training pipeline produces measurably different
weights now: same val/auroc (≈0.873), but **−0.016 val/cmap** and **−0.025 val/f1_macro**
vs v5j-original's training metrics. **Cause is in the training path between the exp 019
era (Mar 30 2026) and HEAD (May 10 2026)** — 8 PRs touched `sed_finetune.py`, adding
+1300 lines and structurally reorganizing `__init__`, `training_step`, and the loss
computation. This note ranks candidate root causes and proposes diagnostic experiments.

**Critical implication**: 0.906 IS the real achievable LB for the v5j recipe (proven by
v5j-resubmit: exp 019 ep 19 checkpoint → today's submission pipeline = 0.906 exactly).
The regression is a **bug to find and fix**, not a baseline to lower. Every comparison
made against the 0.906 baseline since exp 026 is silently contaminated by ~0.009 of
training-side regression — but the answer is to restore reproducibility, not to
re-baseline at 0.897.

## The smoking gun

### Same recipe, same seed → different model

| Run | Date | Codebase commit | val/auroc | val/cmap | val/f1_macro | best_ep | LB |
|---|---|---|---:|---:|---:|---:|---:|
| **v5j-original** (exp 019) | 2026-03-31 00:33 | `e420633` era | **0.8737** | **0.6429** | **0.5515** | **19** | **0.906** |
| v5j_control_v2 (exp 030) | 2026-04-27 | post-exp 026/027 | 0.8728 | 0.6268 | 0.5270 | 15 | 0.896 |
| v5j_control (exp 031) | 2026-05-10 | HEAD | 0.8728 | 0.6268 | 0.5270 | 15 | 0.896 (inherited) |
| v5j_control_resave (exp 031) | 2026-05-10 | HEAD | 0.8728 | 0.6268 | 0.5270 | 15 | — |
| **v5j_control_last** (ep 19, exp 031) | 2026-05-10 | HEAD | 0.8728 (final) | 0.6268 | 0.5270 | 19 | **0.897** |

The exp 030 → exp 031 retrains are bit-identical to each other (4-dp metric match,
state_dict hash match), confirming determinism is intact within today's codebase.

The exp 019 → exp 030 jump is the drift: **same config, same seed, different code → different weights**.

### Quantifying the gap

- val/auroc Δ: −0.0009 (within noise)
- **val/cmap Δ: −0.0161** (large; precision/calibration metric)
- **val/f1_macro Δ: −0.0245** (large; threshold-based)
- LB Δ: **−0.009 to −0.010** (4 retrains all in [0.896, 0.897])

The val/auroc parity initially masked the drift — the rank-ordering of predictions is similar,
but the **magnitude/calibration is meaningfully worse**. This is exactly the offline-LB anti-correlation
pattern in reverse: cmap/f1 dropped, and LB dropped with them. cmap and f1 turned out to be the
correct directional signal here.

### Inference pipeline is clean (rules out a class of causes)

Per exp 030's `v5j-resubmit` diagnostic (2026-04-28): re-exporting the original
`017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt`
through today's `experiments/010-submission-pipeline/` produces LB **0.906** exactly.

This rules out:
- ONNX export drift
- Mel pipeline differences at inference
- TTA / smoothing / TopN / priors drift in the kernel
- Submission CSV format changes

The drift is squarely in the training pipeline.

## Codebase delta: scope of the churn

`v5j-original` was trained at commit `e420633` (2026-03-30 23:46) on the
`mg/exp019-perch-guided-sed` branch. The training job started shortly after
and the ckpt was finalized at 00:33 on Mar 31. Between then and HEAD on
`mg/exp031-perch-kd-distillation`, **8 commits** touched `sed_finetune.py`:

```
35054d8  Exp 021: SED training improvements (#23)
459713f  Exp 022: PCEN normalization (#24)
1472dab  Exp 025: Iterative pseudo-labeling, tiered teacher (#27)
4301a22  Exp 026: Domain-shift training, dual validation (#28)
2b0ab35  Exp 027: Model training fixes — additive MixUp + CE loss (#30)
58dd8c2  Exp 030: Sydorskyi-faithful MixUp 2x2 ablation
ede03e6  Implement exp 031 Perch v2 KD branch
5f4b6aa  Fix exp 031 fast-dev-run
```

`git diff e420633..HEAD --stat` on the training path:

| File | Lines added | Lines deleted | Net |
|---|---:|---:|---:|
| `lightning/sed_finetune.py` | 651 | 104 | +547 |
| `data/sed.py` | 433 | 12 | +421 |
| `heads/sed_head.py` | 211 | (small) | +200 |
| `preprocessing/nnaudio_mel.py` | 90 | (small) | +90 |
| `encoders/efficientnet.py` | 19 | (small) | +19 |
| **Total** | **~1400** | **~120** | **+1280** |

Plus: `pyproject.toml` added `tensorflow>=2.20.0` and `marimo>=0.11.0`. New
dependencies pulled in by `uv pip install -e .` may shift transitively-installed
package versions (torch, lightning, torchmetrics, NNAudio) even without explicit
version bumps.

## Ranked candidate root causes

### 1. Cumulative `nn.Module` registration order drift in `__init__` (HIGH)

`SEDFinetuneModule.__init__` between v5j-era and HEAD has accumulated:

- **New conditional submodules** registered when their flag is on (none of these
  fire for v5j_control config): `self.distill_head`, `self.perch_teacher`,
  `_val_sc_pos_counts` buffer.
- **New `self.asl_loss` instance** when `loss_type == "asl"` (doesn't fire for BCE).
- **Possibly different metric registration order** depending on what `for split in (...)` iterates produced under newer torchmetrics.

PyTorch's parameter init for `nn.Linear`, `nn.Conv*` etc. uses the global RNG
(`torch.default_generator` → CUDA generator after `seed_everything`). Each call
consumes RNG state. Order matters: if module A is constructed before module B,
A consumes its slice of RNG first. Reordering or adding/removing intermediate
calls shifts everything downstream.

For v5j_control (KD off, no PCEN, no ASL, no dual_val) the conditional submodules
shouldn't fire — but if any unconditional code path now calls `nn.Linear` etc.
in a different order, the encoder + head re-init will differ subtly.

**How to confirm**: SHA256 hash the encoder + sed_head state_dicts at step 0
(after init, before any training), comparing v5j-era code-path to HEAD code-path.
If hashes differ, init drift is confirmed.

### 2. `_compute_loss` / `_compute_chunked_loss` floating-point divergence (MEDIUM-HIGH)

v5j-era loss computation:
```python
loss = clipwise_weight * _compute_loss(clipwise_logits, labels, primary_idx)
       + framewise_weight * _compute_loss(framewise_max_logits, labels, primary_idx)
```

HEAD loss computation:
```python
loss = _compute_chunked_loss(clipwise_logits, framewise_logits, labels, primary_idx, loss_weight)
```

`_compute_chunked_loss` internally calls `_compute_loss` with the same arguments,
plus `loss_weight=None` → falls through to `F.binary_cross_entropy_with_logits(logits, labels.float())`
in both eras. Should be mathematically identical.

**But**: the new `_compute_loss` signature has a new optional parameter `loss_weight`.
The function's bytecode and call frames differ. Floating-point arithmetic ordering
(the order in which intermediate accumulations happen, not just operations) can
produce non-bit-identical results across mathematically-equivalent code paths,
especially under `16-mixed` precision. If this happens on every training step,
small differences accumulate into different gradients → different weights.

**How to confirm**: zero-step diagnostic. Run training with `lr=0` for 1 step on
both code paths, dump gradients, compare. If gradients differ at lr=0, fp drift
is the cause.

### 3. `training_step` structural reorganization (MEDIUM)

v5j-era `training_step` had a flat `if mel_transform is not None and waveform_mixup:` /
`elif mel_transform is not None:` / `else:` chain. HEAD has nested conditionals
introducing `do_kd`, `use_manifold`, `K = max_trick_factor if max_trick else 1`,
extra `if loss_weight is not None` branches.

For v5j_control config, all the gates (`do_kd=False`, `use_manifold=False`, `K=1`,
`loss_weight=None`) should route to the same effective code path. But:

- The new code calls `self._mel_grad_context()` instead of `torch.no_grad()`. The
  helper checks `mel_transform.parameters()` — a dict iteration that may produce
  different ordering under different Python/torch versions (unlikely but possible).
- The order of `if-elif-else` evaluation may have changed even though the executed
  branch is the same.

Same RNG/tensor-allocation argument applies. Subtle drift possible.

### 4. Library version drift (`torch`, `lightning`, `nnaudio`, `torchmetrics`) (MEDIUM)

`pyproject.toml` added `tensorflow>=2.20.0` (for Perch teacher) on commit `ede03e6`.
TensorFlow has aggressive transitive deps (numpy, protobuf, etc.) that can pin
or upgrade other libraries. `uv.lock` was not committed in v5j-era (need to verify).

Even pinning the same direct deps, `uv pip install -e .` resolves indirect deps
fresh each install. CUDA / cuDNN / cuBLAS minor versions can change, and these
control kernel selection in deterministic mode (Lightning's `Trainer(deterministic=True)`
forces `torch.use_deterministic_algorithms(True)`, which raises on unsupported
ops but doesn't pin algorithm choice across CUDA versions for ops it considers
deterministic).

**How to confirm**: hash the `.venv` after install and compare across runs; pin
to specific torch/torchmetrics/lightning/nnaudio versions and rerun.

### 5. `data/sed.py` mixup pair selection or sampling order (MEDIUM)

+421 lines in `data/sed.py` — substantial change. Possibilities:

- **DataLoader shuffle order** — if any new code path calls `torch.randint`, `np.random.*`
  during `__getitem__` or during DataLoader worker initialization, the per-worker
  RNG state shifts. Even with `worker_init_fn` setting seeds deterministically,
  the BODY of `__getitem__` is what counts.
- **Mixup pair selection** — v5j config doesn't use waveform mixup (`raw_waveform_mixup: false`),
  so spec mixup happens later in `_apply_mixup` not in dataset. But if dataset
  changed how it presents (B, ...) batches (e.g., a new collate_fn), batch
  composition differs.
- **`_split_focal_dataframe`** behavior — the 4 single-sample species
  duplicated train+val behavior is documented in CONTEXT.md. If this logic
  changed at all, the split changes, the train data changes, training diverges.

**How to confirm**: hash the first N batches' (x, labels, primary_idx) tuples
under v5j-era vs HEAD code paths with same seed. If batches differ, dataset
drift is confirmed.

### 6. `head_lr`/scheduler interaction quirk (LOW)

CONTEXT.md flags: "head_lr quirk: silently ignored under scheduler_type: one_cycle
(only used in cosine branch of configure_optimizers)". v5j config has
`scheduler_type: "one_cycle"` and `head_lr: 5e-4` (= lr). Quirk doesn't bite for
v5j because they're equal — no split-LR. But if the optimizer's param groups
construction order changed, parameter assignment to AdamW's state slots differs
→ momentum/m2 buffers initialize differently → different update trajectory.

**How to confirm**: print param group sizes and orderings under both code paths.

### 7. The "soundscape val" change (LOW)

v5j metrics.json (exp 019) didn't have `dual_validation`, `mixup_warmup_epochs`,
`mixup_theta`, `mixup_per_sample`, `mixup_mode`, `loss_type`, `framewise_weight`
fields written to the artifact. Today's metrics.json does. This means
`save_metrics.py` was extended, but it could ALSO mean the **v5j-original
training was on a config that DID NOT have these knobs at all** — they came in
later via exp 021/026/027/030. New knobs with default values can still
indirectly change behavior if the defaults differ (e.g., `mixup_warmup_epochs`
default was 0 in v5j-era? But our v5j_kd config sets it to 5...).

**Action**: dig into the exp 019 v5j config (`hgnet_b0_sed_v5j.yaml` at commit
`e420633`) and compare to today's. Document defaults at each commit.

## Diagnostic experiments (cheap → expensive)

### A. Zero-step gradient diagnostic (cheap, ~10 min)

1. Check out `e420633`, `pip install -e .`, run 1 forward + 1 backward at lr=0
   on a fixed (seed=42) batch. Save gradients per-parameter as a tensor dict.
2. Check out HEAD, repeat.
3. Compare gradient hashes per-parameter. If any differ, the forward/backward
   path produces different math.

### B. Init-state hash diagnostic (cheap, ~5 min)

1. At `e420633`, instantiate `SEDFinetuneModule` with the v5j config, dump
   `self.encoder` + `self.sed_head` `state_dict()` SHA256.
2. At HEAD, repeat.
3. If init hashes differ, RNG ordering during `__init__` is the cause.

### C. First-batch hash diagnostic (cheap, ~5 min)

1. At `e420633`, instantiate the dataset + DataLoader, pull the first 5 batches
   (with `seed_everything(42, workers=True)`), hash each `(x, labels, primary_idx)`
   tuple.
2. At HEAD, repeat.
3. If batches differ, data path RNG drifted.

### D. Bisection (medium cost, ~hours)

If A/B/C all pass (rare), bisect across the 8 commits between `e420633` and HEAD.
Pick midpoint commit, retrain v5j_control with `save_top_k: -1`, submit ep 19 to
LB. Each LB is ~30 min training + ~10 min export + Kaggle quota slot.

Likely candidate cuts:
- After exp 021 (`35054d8`): if LB ≈ 0.906, exp 021 didn't break it
- After exp 026 (`4301a22`, dual_validation): exp 026 is the most-suspected since
  it added the conditional `_val_sc_pos_counts` buffer and reorganized validation.
- After exp 027 (`2b0ab35`, additive mixup + CE loss): added mixup_mode dispatch.

### E. Pin and lock (medium cost)

Generate `uv lock` for `e420633`'s pyproject.toml and HEAD's pyproject.toml.
Diff the locked versions. If torch/lightning/torchmetrics/nnaudio differ, pin
HEAD to v5j-era versions and rerun. If LB recovers, library version drift confirmed.

## Path forward

1. **Immediate** (within this session):
   - Document this finding (this note, MODEL-RUNS.md, results.md, tasks.md). DONE.
   - Keep 0.906 as the real v5j baseline (proven by v5j-resubmit). Treat 0.897 as a
     known regression to fix, not a corrected baseline.
   - Within-codebase KD-vs-control deltas are still meaningful for exp 031 verdict:
     0.890 vs 0.896 = −0.006 (REJECT) at best-by-val, 0.891 vs 0.897 = −0.006 (REJECT)
     at ep 19. KD verdict consistent across both regimes.

2. **Near-term** (next 1-2 sessions):
   - Run diagnostics A, B, C (cheap, ~30 min total). They should localize the cause.
   - If A passes (gradients match), the issue is RNG/init drift from B or C.
   - If A fails, the issue is forward/backward math drift (most likely from a new code path executing differently).

3. **Medium-term** (depending on diagnostic findings):
   - If init drift: add a `freeze_init_seed` flag that resets RNG state immediately after
     module construction, before the first training step. Defensive against future drift.
   - If lib version drift: pin `pyproject.toml` to a known-good lockfile. Generate
     `uv.lock` and commit it.
   - If `data/sed.py` drift: hash check + audit the `_split_focal_dataframe` function
     for behavioral changes.
   - If `training_step` math drift: identify the specific path that changed and either
     add a back-compat flag or accept the new path as the new baseline.

4. **Long-term** (process improvement):
   - **Reproducibility CI**: add a CI job that retrains v5j_control on every PR
     touching `src/birdclef/lightning/sed_finetune.py` or `src/birdclef/data/sed.py`
     and asserts val/auroc within ±0.0001 of a pinned reference. Catches drift at
     PR time.
   - **Lockfile in git**: commit `uv.lock` so installs are bit-reproducible across machines.
   - **Save more in metrics.json**: include git commit hash, lockfile hash, and a
     hash of the first batch's (x, labels) tensor. Future archaeology becomes easier.

## What this means for exp 031

The KD verdict was already REJECT (0.890 < 0.896 control) and remains REJECT against
the corrected 0.897 baseline (0.890 < 0.897). v5j_kd_last (ep 19) result is still
pending; even if it lands at 0.900, it's only +0.003 over 0.897 — still inside seed
noise, not a CONFIRM.

The post-mortem hypotheses (LSEHead wrong consumer of Perch features / 20 epochs too
short / α miscalibrated / requires Tucker's full stack) all still apply, but with
the additional caveat that we are now testing on a codebase that demonstrably
trains differently than the recipe was originally proven on. Some part of the
"KD doesn't transfer" finding could be entangled with the codebase drift —
e.g., maybe KD on the v5j-era codebase would have given +0.014. We can't know
without resolving the drift first.

## What this means for the project

This is an **infrastructure-level finding**: every comparison made since
exp 026 has been silently contaminated by ~0.009 of training-side regression
relative to the v5j-original baseline. The 0.906 baseline is the real
achievable target for the recipe (verified by v5j-resubmit), but the
training pipeline that produces it can't currently be reproduced from scratch.

| Baseline | Real value | Currently reproduces? |
|---|---:|:---:|
| v5j single-fold | 0.906 (proven by v5j-resubmit) | ❌ training falls short by ~0.009 |
| v5j 4-fold pp2 | 0.917 (achieved) | ❌ likely also affected (needs retrain to confirm) |
| Single-fold seed band | ±0.013 (per exp 030) | ✅ unchanged within today's codebase |

Several past "WEAK" / "REJECT" verdicts compared training-recipe variants
against the 0.906 baseline. Their TRUE delta-vs-recipe-as-originally-trained
is contaminated by the ~0.009 regression. For example:

- exp 030 sydorskyi mixup hit 0.892 → vs 0.906 = −0.014 (looked like
  meaningful regression). But part of that −0.014 is the codebase
  regression that affects EVERY current run, not the mixup change.
  Actual within-codebase delta vs same-codebase v5j_control = −0.004 (null).
- exp 029 2025 pretrain hit 0.885 → vs 0.906 = −0.021. Within-codebase
  delta vs same-codebase v5j_control = −0.012.

Strategic conclusions (training-recipe levers are weak on this codebase)
remain valid because within-codebase deltas don't change. But the
**magnitudes were overstated** by ~0.009, and the right reading is "X
recipe is null vs same-codebase control" rather than "X recipe is −0.014
vs v5j-original".

**Most importantly**: until reproducibility is restored, we can't run any
experiment that aims to BEAT v5j-original 0.906 from scratch. We're 0.009
in the hole before we start. The codebase regression is the highest-priority
fix on the project, not a baseline to accept.

## Cross-references

- v5j-original training metrics: `experiments/019-sed-ablations/artifacts/training/hgnet_b0_sed_v5j/metrics.json`
- v5j_control_resave training metrics: `experiments/031-perch-kd-distillation/artifacts/training/v5j_control_resave/metrics.json`
- exp 030 v5j-resubmit (inference is clean): `experiments/030-sydorskyi-mixup/results.md`
- exp 031 best-vs-last LB results: `experiments/031-perch-kd-distillation/results.md` § Best-vs-last LB results
- v5j-era code reference commit: `e420633`
- Lightning save_top_k pitfall (related but distinct): `feedback_save_top_k_does_not_preserve_last.md`
