---
experiment: "032-tucker-distilled-sed"
status: approved
date: 2026-05-23
author: Murilo Gustineli
---

# Proposal: Full Tucker distilled-SED replication

**Status**: approved (2026-05-23)
**Created**: 2026-05-23
**Author**: Murilo Gustineli

## Reference materials (read before implementing)

These are the load-bearing docs for this experiment. Implementation cross-references will point here rather than re-explaining.

| Doc | What's in it |
|---|---|
| [`analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`](../../analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md) | Forward pass, AttHead aggregation domain (logit space, not probability), loss formula, mixup defaults, optimizer, mel pipeline. |
| [`analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md`](../../analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md) | Dataset class internals, fold construction, MixSamp 90/10 sampler, per-sample weights (effectively inert), all-zero-label SS handling (53 windows / 0.7%), checkpoint selection, inference in logit space, 5-tap Gaussian smoothing, two notebook bugs to NOT inherit. |
| [`notes/kaggle-discussion-tucker-distilled-sed-baseline.md`](../../notes/kaggle-discussion-tucker-distilled-sed-baseline.md) | Tucker's discussion thread with the +0.02 LB quantification, training logs, waveform preprocessing script, MengYe smoothing trick (+0.004). |
| [`notes/2026-05-22-strategic-direction-tracks-A-B.md`](../../notes/2026-05-22-strategic-direction-tracks-A-B.md) | Strategic context: Track A (Tucker primary) → Track B (own-teacher PL). Why this is the high-confidence path. |
| [`notes/figures/tucker-distilled-sed-arch.png`](../../notes/figures/tucker-distilled-sed-arch.png) | Architecture diagram (1536-d teacher default). |
| [`notes/figures/tucker-distilled-sed-arch2.png`](../../notes/figures/tucker-distilled-sed-arch2.png) | Cheaper 384-d MBConv stage variant (same LB, 20% faster). |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` | Tucker's verbatim notebook (the source). Re-pulled 2026-05-23 — byte-identical to 2026-05-09 vendored copy. |
| [`experiments/031-perch-kd-distillation/results.md`](../031-perch-kd-distillation/results.md) | Exp 031 REJECT post-mortem. Critically: REJECT is scoped to "KD as single-variable add on v5j HGNet+LSEHead+20ep+α=1.0" — Tucker's full recipe is NOT falsified. |

## Hypothesis

Faithful replication of Tucker Arrants's full distilled-SED recipe (EffNet-B0 NS-JFT + logit-space AttHead + dual-supervision loss + Perch v2 embedding KD + waveform mixup with hard-max labels + SpecAugment + 25 ep cosine + 90/10 focal/SS MixSamp + Tucker fold construction) on our codebase will land a single-fold LB in the **0.910 – 0.917** range, matching Tucker's publicly-quantified +0.02 KD lift and his published 0.917 single-model LB after 5-tap Gaussian smoothing.

Mechanism: exp 031 already showed KD alone on v5j (HGNet+LSEHead) doesn't transfer (−0.006 paired Δ). The post-mortem ranked four candidate causes; **#1 (LSEHead is wrong consumer of Perch features) and #4 (KD lift requires the full stack) are tested simultaneously by Tucker replication**, which is the highest-EV next step in the 12-day budget.

### Falsifiable

Read against **same-codebase paired control** (Tucker recipe with KD off, same SLURM job batch). Per exp 031 drift investigation: cross-run LB has ~0.007 hardware variance + ~0.002 code variance, so paired-within-codebase is the only reliable methodology. **Do NOT read against v5j-original 0.906.**

| Bucket | Paired Δ (tucker_kd vs tucker_control) | LB band for tucker_kd |
|---|---|---|
| **CONFIRM** | ≥ +0.014 | LB ≥ control + 0.014 |
| **PARTIAL** | +0.007 to +0.013 | Lift detectable, smaller than Tucker's +0.020 |
| **WEAK** | +0.001 to +0.006 | Inside ±0.013 single-fold seed band — needs 4-fold to disambiguate |
| **REJECT** | ≤ 0.000 | KD doesn't transfer even on bug-fixed AttHead + full stack → diagnose stack-vs-codebase |

Secondary read: **tucker_control LB vs v5j_control_last 0.897** (exp 031). This isolates the recipe lift independent of KD:

| tucker_control LB vs v5j_control_last 0.897 | Reading |
|---|---|
| ≥ 0.910 | Recipe alone (backbone + AttHead + dual-supervision + mixup + SpecAugment + schedule + MixSamp) is the lever |
| 0.900 – 0.909 | Recipe helps modestly; KD branch is the remaining variable |
| < 0.900 | Recipe alone regressed — investigate before reading the KD delta |

## Background

### What this experiment tests

Exp 031 verdict (REJECT, Δ = −0.006) was confined to **"KD as a single-variable add on v5j HGNet+LSEHead+20ep+α=1.0"**. Four post-mortem candidates were ranked (`experiments/031-perch-kd-distillation/results.md` § Post-mortem); the highest-likelihood was "LSEHead is the wrong downstream consumer of Perch features."

Tucker's actual proven recipe is **multi-variable**:
- EffNet-B0 NS-JFT (not HGNet)
- Logit-space AttHead with dual-supervision (not LSEHead)
- 25 ep + 2 ep warmup → cosine to 1e-6 (not 20 ep OneCycle)
- Waveform mixup with `MIXUP_HARD=True` (not spec mixup with interpolated labels)
- Focal × labeled-SS mixup (not present)
- SpecAugment 1×freq×10 + 2×time×10 (not present)
- 90/10 focal/SS MixSamp batch composition (not present)
- 5-fold KFold filenames + MultiLabelStrat focal (we're single-fold from v5j)
- Logit-space inference blending with single sigmoid at end (we sigmoid per-window)

Phase 1 is **single-fold paired control + KD** at the full Tucker recipe to test whether the lever transfers in its native stack. This is multi-variable by design; the paired Δ isolates the KD branch within the new stack.

### Why we expect transfer

| Argument | Direction |
|---|---|
| Tucker quantifies +0.02 LB on this exact pipeline (discussion thread) | Positive |
| Three independent +0.013 to +0.020 data points (Tucker, EliKal, chenwenqiang_001) | Positive |
| Exp 031 post-mortem #1: LSEHead drops the spatial signal that KD installs; logit-space AttHead consumes it directly. This is the head-side fix. | Positive |
| Exp 031 post-mortem #4: KD lift may compound with hard-max mixup, dual-supervision, SpecAugment. Full replication tests this. | Positive |
| Our exp 031 KD branch + Perch ONNX + DistillHead + sed_finetune.py wired already — ~30% of implementation already done | Positive |
| Multi-variable change makes attribution harder if WEAK/PARTIAL — but Phase 1 isn't trying to attribute, it's trying to clear the runway. Attribution comes in Phase 2 ablations. | Uncertain |
| Single fold has ~0.013 seed band on HGNet (exp 030); EffNet may have different variance | Uncertain |

## Method

### Approach

End-to-end faithful port of `tuckerarrants/bc2026-distilled-sed` (verbatim notebook in `vendor/`). Two paired SLURM jobs:

- **`tucker_control.yaml`**: KD off, everything else faithful Tucker
- **`tucker_kd.yaml`**: KD on, otherwise identical to control

Both submitted at the same time on the same partition. Their LB delta is the KD branch's contribution within the full Tucker stack.

### Setup

| Component | Details |
|---|---|
| Backbone | `tf_efficientnet_b0.ns_jft_in1k` (timm) — NS-JFT pretrained EfficientNet-B0, 1-channel, no global pool, drop_path_rate=0.1. ~5M params. |
| Head | New `AttHeadLogitSpace` in `src/birdclef/heads/sed_head.py`. GeMFreqPool(p=3, clamp(min=1.0)) → Dropout(0.25) → Linear(C, 512) → ReLU → Dropout(0.5) → 2× Conv1d(512, 234, k=1, xavier init) → softmax(tanh(att)) × cls aggregated in logit space. |
| Loss | `cls = 0.5 · BCE(clip_logits, lb) + 0.5 · BCE(frame_max_logits, lb)` + `1.0 · MSE(student_emb, perch_emb)` (KD on). With stop-gradient on the cls branch when KD is on. |
| Data | Focal recordings + labeled soundscape windows (66 files × 12 = 792 windows; 739 with positive labels, 53 all-zero — treated as all-negative per the dataset audit). Unlabeled SS files (10,592 × 12 = 127k windows) NOT used — they're the Track B PL pool. |
| Mel | torchaudio MelSpectrogram, n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000, power=2.0, AmplitudeToDB(top_db=80), per-sample mean/std norm. SpecAugment after norm: 1×freq×10 + 2×time×10. |
| Mixup | Waveform, Beta(0.4, 0.4) symmetric, `MIXUP_HARD=True` (max labels), p=0.5 focal-focal; p=0.5 focal × labeled-SS within batch. Per-sample at most one mixup applies. |
| Sampler | `MixSamp(shares={"focal": 0.9, "sc": 0.1}, BATCH=64)` → 58 focal + 6 SC per batch, with-replacement. `nst = max(100, sum(sizes) / BATCH)`. |
| Folds | 5-fold; focal `StratifiedKFold(n_splits=5)` on `primary_label` (with `drop_duplicates("original_idx")` before split), SS `GroupKFold(n_splits=5)` on `filename`. All 66 SS files participate. Rare-species upsampling to MIN_SAMPLE=20 after fold assignment. **Phase 1 trains fold 0 only**; Phase 2 extends to 5-fold if Phase 1 passes the gate. |
| Optimizer | AdamW(lr=5e-4, wd=1e-4) + LinearLR(start_factor=1/25, total_iters=2 epochs) → CosineAnnealingLR(T_max=23 epochs, eta_min=1e-6) via SequentialLR. fp16 autocast + GradScaler + clip_grad_norm 1.0. |
| Epochs | 25 (Tucker's default; v5j was 20). save_top_k=-1 to capture both best-by-macro and best-by-non_s22_macro. |
| Teacher | Perch v2 via ONNX (`vendor/kaggle/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx`). Re-using the wiring from exp 031. |
| Inference | Per-fold `0.5·clip_logits + 0.5·frame_max_logits` in logit space → fold-mean → 5-tap Gaussian smooth `[0.1, 0.2, 0.4, 0.2, 0.1]` across the 12 windows per file in logit space → single sigmoid at the end. Site/hour priors NOT in this proposal (would port from EoS.6 in Phase 3 if needed). |
| Compute | rtx6000 single GPU. Tucker's published walltime: ~60s/epoch with KD on RTX 6000 Pro @ batch 64 = ~25 min training. Our env will likely be slower (ONNX teacher overhead + slower data loading); budget ~90 min per single-fold paired job. |
| Dependencies | `experiments/032-tucker-distilled-sed/pyproject.toml` — pyproject as source of truth. Direct ML pins matching exp 031 uv.lock resolved versions (torch 2.5.1, timm 1.0.26, hf-hub 1.9.2, nnAudio 0.3.4, onnxruntime-gpu 1.26.0). No uv.lock. |
| Code | `mg/exp032-tucker-distilled-sed-replication` branch, PR #51 |

### Procedure

See `tasks.md` for the full ordered checklist. Top-level phases:

1. **Phase 0 — port** (no SLURM yet): AttHeadLogitSpace, backbone swap, dual-supervision loss, Tucker mixup, SpecAugment, schedule swap, MixSamp, 5-fold construction. Unit-test each.
2. **Phase 1 — paired single-fold** (fold 0 only): fast-dev-run gradient check, then full 25-ep paired SLURM (tucker_control + tucker_kd), LB submission. Apply success matrix.
3. **Phase 2 — if Phase 1 passes** (CONFIRM or PARTIAL): extend to full 5-fold, ensemble at inference, optionally port site/hour priors from EoS.6.
4. **Phase 3 — if Phase 1 fails**: diagnose stack-vs-codebase. Possible candidates: ALPHA_DISTILL sweep, mel-pipeline alignment with Perch, longer schedule.

### Variables

- **Independent (between paired runs)**: `use_perch_distill` (False vs True)
- **Independent (between exp 031 and exp 032)**: backbone, head, loss, mixup, SpecAugment, schedule, sampler, fold construction — all changed at once. Intentional multi-variable.
- **Dependent**: paired LB Δ; tucker_control LB vs v5j_control_last 0.897; offline val/cmap; dist_loss floor (sanity).
- **Controlled (within paired runs)**: same seed, same data, same code commit, same SLURM partition, same wall-clock window, paired sbatch submissions.

## Evaluation

### Metrics

| Metric | Description |
|---|---|
| Public LB | Primary signal. Read paired Δ within exp 032 + secondary vs v5j_control_last 0.897. |
| val/macro_auc_all | Sanity check (Tucker's checkpoint-selection metric per his discussion comment). |
| val/non_s22_macro | Tucker's secondary metric (excludes the dominant S22 site). |
| train/cls_loss, train/dist_loss | Training trajectory health. dist_loss should decay to ~0.003 by epoch 25 (Tucker's published floor). |

### Baseline

- **Paired control**: tucker_control LB (TBD, same job batch as tucker_kd)
- **Secondary baseline**: v5j_control_last LB 0.897 (exp 031, ep 19, save_top_k=-1)
- **Optimistic target**: Tucker's 0.917 LB (his 5-fold + smoothing). Phase 1 = single fold so we expect to land below this; target 0.910–0.916 single-fold band.

### Success Criteria

See § Falsifiable above. Two thresholds — paired Δ and tucker_control LB — both read together.

## Limitations

- Single fold for Phase 1. Per exp 030 + exp 031 drift investigation, single-fold v5j HGNet has ~0.013 LB seed band; EffNet variance unknown. WEAK/PARTIAL outcomes can't be distinguished from noise without 4-fold or seed sweep.
- Multi-variable port — if Phase 1 lands CONFIRM, the lift is attributable to "Tucker recipe + KD" as a unit, not to individual components. Ablations come in Phase 3.
- Tucker's `bc2026-distilled-sed` notebook has two known bugs (mel-size mismatch in SEDExportWrapper, checkpoint filename inversion) — we fix both in our port; this means our pipeline ≠ Tucker's exact pipeline by intention. Expected impact: small to positive.
- Site/hour priors not in this proposal. Tucker's "0.93+ with PP" claim includes priors that are NOT in the public notebook. Our 0.910–0.917 target is for the publicly-replicable subset (notebook + 5-tap Gaussian only).
- ONNX teacher overhead — Tucker reports ~30 s/epoch added by teacher forward. Our SLURM environment may be slower; budget 90 min wall time per single-fold paired run.
