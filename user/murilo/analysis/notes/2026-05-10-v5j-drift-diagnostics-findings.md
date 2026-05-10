# 2026-05-10 — v5j codebase drift: diagnostic findings (in-progress)

Following up on `2026-05-10-v5j-codebase-drift-analysis.md`. We ran the
diagnostics proposed there and **eliminated every static cause**. The drift
is real but requires actual GPU training to manifest. The next step is the
ground-truth worktree-training experiment.

## Ground-truth observation

v5j-original (Mar 31) and v5j_control_resave (today) **diverge on val/auroc
starting at epoch 0**:

| Epoch | v5j-original (Mar 31) | v5j_control_resave (today) | Δ |
|---:|---:|---:|---:|
| 0 | 0.50607 | 0.50880 | +0.00273 |
| 1 | 0.57539 | 0.56379 | −0.01160 |
| 2 | 0.77416 | 0.74631 | **−0.02785** |
| 3 | 0.83772 | 0.83047 | −0.00725 |
| 5 | 0.85724 | 0.85353 | −0.00371 |
| 19 (final) | 0.87366 | 0.87277 | −0.00089 |

The divergence is **already present after ONE epoch of training**. So
whatever causes it executes within the first ~511 training batches.

The val/auroc difference shrinks back toward zero by epoch 19, but the LB
gap persists at −0.009 — the final weights occupy a slightly different
optimum that has marginally similar val/auroc but worse downstream
calibration (cmap −0.016, f1_macro −0.025).

## What we ruled out

All under the SAME login venv (torch 2.10.0+cu128, pytorch-lightning 2.6.1,
timm 1.0.25):

### ✅ Diagnostic A: init-state hash (PASSES, identical)

Instantiated `SEDFinetuneModule` at v5j-era code (`e420633`, via git worktree)
and HEAD code, both with `seed_everything(42, workers=True)` and same kwargs.

| Stage | v5j-era hash | HEAD hash | Match? |
|---|---|---|---|
| Encoder alone | `0c2c3720e6035402` | `0c2c3720e6035402` | ✅ |
| Sed_head alone | `b9f134b06cd3253b` | `b9f134b06cd3253b` | ✅ |
| Module-wrapped encoder | `0c2c3720e6035402` | `0c2c3720e6035402` | ✅ |
| Module-wrapped sed_head | `78098686f5ad6bbd` | `78098686f5ad6bbd` | ✅ |
| Full module state_dict | `119068b6bce19f1b` | `119068b6bce19f1b` | ✅ |
| Post-init `torch.randn(1024)` | `d8607ea4ae1d5a73` | `d8607ea4ae1d5a73` | ✅ |

**Conclusion**: cumulative RNG drift from new submodules in `__init__` is
NOT the cause. The init state is bit-identical.

### ✅ timm pretrained weights hash (PASSES, identical across versions)

Loaded `hgnetv2_b0.ssld_stage2_ft_in1k` from three timm versions:

| timm version | hf-hub version | Pretrained state_dict hash |
|---|---|---|
| 1.0.25 | 1.6.0 | `0273c6a215cb0965e682151422a35660...` |
| 1.0.26 | 1.8.0 | `0273c6a215cb0965e682151422a35660...` |
| 1.0.27 | 1.14.0 | `0273c6a215cb0965e682151422a35660...` |

**Conclusion**: even though timm and hf-hub versions changed between v5j-
original training (1.0.26 / 1.8.0) and today's training (1.0.27 / 1.14.0),
the pretrained weights they download are bit-identical. NOT the cause.

### ✅ Diagnostic C: forward+backward gradient hash (PASSES, identical)

Constructed module + ran 1 forward+backward step with a fixed synthetic
mel input (deterministic seed=123). Compared per-parameter gradient hashes.

| | v5j-era | HEAD |
|---|---|---|
| loss | 0.707846 | 0.707846 |
| loss hash | `14e629675476` | `14e629675476` |
| combined gradient hash | `46c777e75845d49e` | `46c777e75845d49e` |
| per-param hashes (3 sampled) | identical | identical |

**Conclusion**: the forward + backward math is bit-identical between
v5j-era and HEAD code under the same lib stack. NOT the cause.

### ✅ Diagnostic B: train/val split hash (PASSES, identical)

| | v5j-era | HEAD |
|---|---|---|
| train rows | 32071 | 32071 |
| val rows | 3482 | 3482 |
| train filename hash | `f4d3fecb706646b2` | `f4d3fecb706646b2` |
| val filename hash | `5c48e8d592c3b27a` | `5c48e8d592c3b27a` |
| train ORDER hash | `80931c2c616abf64` | `80931c2c616abf64` |

**Conclusion**: data partition + ordering is identical. NOT the cause.

### ✅ Optimizer config (PASSES, identical)

`git diff e420633..HEAD -- user/murilo/src/birdclef/lightning/sed_finetune.py`
shows `configure_optimizers` is bit-identical between eras. Same AdamW,
same OneCycleLR config, same param groups.

### ✅ Library versions in actual training (PASSES, mostly identical)

From SLURM `pip install` logs of v5j-original (Mar 31) vs v5j_kd_resave (today):

| Library | Mar 31 (v5j-original) | May 10 (today) | Same? |
|---|---|---|:---:|
| torch | 2.5.1+cu121 | 2.5.1+cu121 | ✅ |
| torchaudio | 2.5.1+cu121 | 2.5.1+cu121 | ✅ |
| torchvision | 0.20.1+cu121 | 0.20.1+cu121 | ✅ |
| pytorch-lightning | 2.6.1 | 2.6.1 | ✅ |
| nnaudio | 0.3.4 | 0.3.4 | ✅ |
| torchmetrics | 1.9.0 | 1.9.0 | ✅ |
| numpy | 2.4.4 | (similar) | ✅ |
| timm | 1.0.26 | 1.0.27 | ⚠️ (no behavioral effect — pretrained weights identical) |
| huggingface-hub | 1.8.0 | 1.14.0 | ⚠️ (no behavioral effect — same weights downloaded) |
| nvidia-cudnn-cu12 | 9.1.0.70 | 9.1.0.70 | ✅ |
| nvidia-cublas-cu12 | 12.1.3.1 | 12.1.3.1 | ✅ |
| nvidia-nccl-cu12 | 2.21.5 | 2.21.5 | ✅ |
| NVIDIA driver | 575.57.08 | 575.57.08 | ✅ |
| CUDA | 12.9 | 12.9 | ✅ |
| Python | 3.11.11 | 3.11.11 | ✅ |
| cuda-pathfinder | 1.5.0 | 1.5.4 | ⚠️ (path resolver only, doesn't affect math) |

**Conclusion**: the active ML/CUDA stack is bit-identical between Mar 31
and May 10 training. Only timm + hf-hub + cuda-pathfinder differ, all
proven non-load-bearing.

### ✅ Data files (PASSES, unchanged since Mar 11)

`/storage/.../shared/data/birdclef-2026/train.csv` mtime is 2026-03-11.
Audio files in `train_audio/` weren't checked file-by-file but the directory
hasn't been touched in this experiment cycle.

## What's left as the cause

After eliminating everything testable statically:

| Possible cause | Status |
|---|---|
| 1. Init RNG drift | ❌ ruled out (Diagnostic A) |
| 2. Pretrained weight drift | ❌ ruled out (timm hash test) |
| 3. Forward+backward math drift | ❌ ruled out (Diagnostic C, synthetic input) |
| 4. Data partition / order drift | ❌ ruled out (Diagnostic B) |
| 5. Optimizer config drift | ❌ ruled out (diff) |
| 6. CUDA / cuDNN / cublas / NCCL version drift | ❌ ruled out (pip log) |
| 7. Python version drift | ❌ ruled out (3.11.11 both) |
| 8. Data file changes | ❌ ruled out (mtime check) |
| 9. **Subtle full-pipeline divergence (mel transform + multi-batch + multi-worker)** | ⚠️ untested |
| 10. **GPU-level fp non-determinism between specific Quadro RTX 6000 cards** | ⚠️ untested |
| 11. **DataLoader worker RNG state under `num_workers=8`** | ⚠️ untested |
| 12. **Wheel binary drift at unchanged version numbers** (cu121 wheel republish) | ⚠️ untested |

The divergence happens at **epoch 0**, after ~511 training steps with
`num_workers=8`. None of my single-step synthetic-input diagnostics reach
the same conditions as a real training epoch. Possible per-step
amplification:

- **DataLoader worker RNG**: Lightning's `seed_everything(workers=True)`
  seeds each worker, but the EXACT worker initialization sequence depends
  on the spawn vs fork start method (could differ across Python micro-versions
  or load conditions).
- **NNAudio mel computation**: relies on `torch.fft.rfft` under the hood,
  which calls cuFFT. cuFFT plan caching state could differ across runs.
- **Mixed-precision GradScaler**: scaler state evolves per step; if any single
  step has different inf/nan detection due to fp jitter, scaler diverges.
- **cudnn.benchmark behavior**: even with `Trainer(deterministic=True)`,
  ops not in the deterministic set may still choose different kernels on
  different hardware/load.

## Decisive next step: ground-truth worktree training

The only way to definitively resolve this is to **retrain v5j_control using
the v5j-era code from the worktree** on PACE. The worktree at `e420633` has
all the v5j-era source code. We submit a SLURM job that:

1. Uses `/tmp/birdclef-v5j-era/user/murilo/experiments/019-sed-ablations/` as
   the experiment root
2. Installs deps from that era's `pyproject.toml` (resolves to whatever
   PyPI / PyTorch wheels are CURRENT today — same fresh-resolution that
   today's runs do)
3. Trains v5j single-fold for 20 epochs
4. We submit the ep 19 ckpt to LB

| Outcome | Interpretation |
|---|---|
| LB ≈ 0.906 | The codebase IS the cause (some path my static diagnostics miss). Static diagnostics passing means the cause is subtle — pipeline-level or multi-step-amplified. Bisect through commits to localize. |
| LB ≈ 0.897 | The codebase is NOT the cause. Even v5j-era code on today's env produces 0.897. The cause is environmental (data subtly changed, hardware-as-fleet variance, or a non-pinned dep version we missed). Investigate further. |

Cost: 1 SLURM job (~40 min) + 1 LB slot.

## Status of recommendation

The earlier note said "init RNG drift highest likelihood." Diagnostic A
rules that out. **Updated ranking**:

1. **Untested pipeline-level divergence (epoch 0 amplification)** — HIGH:
   real training execution involves multi-worker DataLoader + mel transform +
   GradScaler state + 511 steps of fp accumulation. Any of these could
   diverge subtly even when single-step synthetic tests pass.
2. **Hardware/fleet variance** — MEDIUM: 4 retrains on different nodes all
   landed 0.896-0.897, so consistent across nodes today; but the v5j-original
   node (atl1-1-02-004-35-0) is different from today's nodes.
3. **Subtle unpinned dep** — LOW-MEDIUM: every ML-relevant dep matches,
   but non-ML deps could have transitive numerical effects (e.g.,
   `huggingface-hub` 1.14.0 download caching could affect tensor I/O paths).
4. **Wheel binary drift** — LOW: PyPI usually doesn't republish wheels,
   but possible.
