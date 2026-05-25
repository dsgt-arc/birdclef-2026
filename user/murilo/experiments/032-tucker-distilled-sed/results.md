---
experiment: "032-tucker-distilled-sed"
status: phase-2-partial-replication
date: 2026-05-25
author: Murilo Gustineli
---

# Results: Tucker distilled-SED full replication

**Date Completed (Phase D)**: 2026-05-24
**Date Updated (Phase 2 LB)**: 2026-05-25
**Author**: Murilo Gustineli

## Status: PARTIAL REPLICATION — within ~0.002 of Tucker's publicly-published 0.917 5-fold LB; the "0.93+ with PP" claim is out of reach without his unpublished post-processing

- **Phase D (LB landed)**: KD lever transfers — distilled_sed_kd single-fold LB **0.906** (matches v5j-original; paired Δ vs control +0.050).
- **Phase 2 (LB landed)**: 5-fold KD ensemble **0.915 LB** (+0.009 over Phase D fold 0; −0.002 below Tucker's published 0.917). HGNet+KD ablation **0.901 LB** (−0.005 vs EffNet fold 0) → **backbone was NOT the bottleneck**. Fold 4 single-fold diagnostic **0.903 LB** (despite strongest offline val/auroc 0.931 = +0.023 over fold 0) → **offline-LB anti-correlation; single-fold LB band is narrow (~0.003)**.
- **Verdict**: replicated what's publicly documented; cannot fully reach Tucker's 0.917 within seed noise let alone his 0.93+ stretch number. The +0.009 single→5-fold ensemble lift is pure variance-reduction (Gaussian smoothing), not picking up better folds. Next levers: ensemble with exp 023 (0.929) and/or port site/hour priors.

## Status: CONFIRM (Phase D) — KD lever transfers, +0.050 paired Δ on LB

After the mel-pipeline mismatch was found + fixed + retrained + LB-resubmitted, Phase D landed:

| Run | LB | Δ paired ctrl | Δ vs v5j_control_last 0.897 | Δ vs Tucker target 0.910 |
|---|---:|---:|---:|---:|
| distilled_sed_control (Phase D retrain, last-v2.ckpt) | **0.856** | — | **−0.041** | **−0.054** |
| distilled_sed_kd (Phase D retrain, last-v2.ckpt) | **0.906** | **+0.050** | **+0.009** | **−0.004** |

**Paired Δ = +0.050**, vs success-matrix CONFIRM threshold +0.014. **The KD lever transfers in our codebase** — much larger than Tucker's published +0.020. distilled_sed_kd matches v5j-original 0.906 LB **exactly** and lands within 0.004 of Tucker's published single-fold target.

### Reading the secondary axis: recipe-alone vs v5j

- distilled_sed_control LB 0.856 vs v5j_control_last 0.897 = **−0.041**. Per the proposal's secondary-axis matrix (< 0.900 → "recipe alone regressed — investigate"), the Tucker recipe **without** KD underperforms v5j by ~0.04 LB.
- But the +0.050 KD lift on top of that recipe is unambiguous and not noise.
- Net interpretation: in our codebase, the Tucker recipe is not by itself an LB win over v5j (EffNet-B0 NS-JFT + AttHeadLogitSpace + dual-supervision is roughly equivalent-or-worse than HGNet + LSEHead at single-fold). **KD does all the heavy lifting.** The full Tucker recipe matters because KD requires the AttHeadLogitSpace consumer of Perch features (post-mortem candidate #1 from exp 031 results.md).

This validates the exp 031 post-mortem hypothesis: "LSEHead is the wrong downstream consumer of Perch features." On AttHeadLogitSpace, the KD lever lands +0.050. On LSEHead, exp 031 measured −0.006. Same KD branch, different head, +0.056 LB delta.

## Phase 1 (broken mel pipeline) — REJECTED

The original Phase 1 LB results (with the broken triple-normalized training mel) are preserved here for the historical comparison:

Phase 1 LB submissions landed **~0.38 LB below v5j_control_last 0.897**.
Both control + KD collapsed to barely-above-random. The paired Δ (KD − control =
−0.009) is unreadable because the control itself is broken. We CANNOT conclude
anything about Tucker's recipe or the KD branch from these numbers.

**This is not noise; it is not a small regression. Something is structurally
wrong in our port of Tucker's pipeline.**

We are pausing here. Next session = comprehensive side-by-side audit of our
port against Tucker's verbatim notebook
(`vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`,
re-pullable via `kaggle kernels pull tuckerarrants/bc2026-distilled-sed`).

## Summary

**Verdict**: REJECTED — but verdict is on the implementation, not the hypothesis.
The Tucker recipe is NOT falsified by this result. Our port is.

**One-liner**: both `distilled_sed_control` and `distilled_sed_kd` collapsed to
LB ~0.52 (vs target band 0.910–0.917, vs paired-control baseline 0.897). Offline
val/cmap on the held-out SS fold tracked Tucker's published trajectory closely
(0.889 at ep 6, matching Tucker's ~0.91), but LB barely beats random.
**Offline metrics are now untrustworthy**.

## Observations

### Quantitative

| Run | Ckpt | Tucker macro AUC (val fold 0) | LB | Δ vs paired ctrl | Δ vs v5j_control_last 0.897 | Δ vs target 0.910 |
|---|---|---:|---:|---:|---:|---:|
| `distilled_sed_control` | last (ep 24) | 0.8928 | **0.522** | — | **−0.375** | **−0.388** |
| `distilled_sed_kd` | last (ep 24) | TBD (KD ranker re-running after strict-load fix) | **0.513** | **−0.009** | **−0.384** | **−0.397** |

Single fold (fold 0). Both ran on `gpu-rtx6000`, distinct nodes:
- control: atl1-1-03-006-17-0
- KD: atl1-1-03-006-21-0

Cross-node variance typically ~0.007 LB (exp 031 drift investigation) —
nowhere near enough to explain −0.38.

### Decision-matrix read

Per `proposal.md` § Falsifiable:

| Bucket | Threshold (paired Δ) | Result |
|---|---|---|
| CONFIRM | ≥ +0.014 | ❌ |
| PARTIAL | +0.007 to +0.013 | ❌ |
| WEAK | +0.001 to +0.006 | ❌ |
| REJECT | ≤ 0 | technically ✅ (Δ = −0.009) |

**But this verdict is uninformative**. The paired Δ is small and within seed
noise; what is huge is the joint collapse of both runs to LB 0.52. The
hypothesis tested by the paired Δ ("does KD lift on top of Tucker's stack?") is
unanswerable when the stack itself is broken.

Per the secondary read (control LB vs v5j_control_last 0.897):
- < 0.900 → "recipe alone regressed — investigate before reading the KD delta"
- Our result (0.522) is far below 0.900 — **investigation gate triggered**.

### W&B / offline trajectory

In-training W&B `val/cmap` and `val/auroc` showed 0.038–0.111 (broken-metric
range — torchmetrics' 0-positive-class bug, fixed via `TuckerMacroAUC` per
yesterday's commit `8d64be8`). Post-hoc ranking with Tucker's correct macro
AUC (skip 0-positive classes) showed 0.85–0.92 across all 25 epochs (control
TSV at `outputs/checkpoint_ranking_control.tsv`). This matched Tucker's
published 0.91 trajectory exactly at epoch 6 (0.889 ours vs 0.91 his).
**The model appears to be learning Tucker's recipe** — yet LB shows it's
almost random.

This is the core paradox: a model that scores 0.89 macro AUC on the same val
data Tucker uses cannot score 0.52 on the LB if our inference + LB mapping
are correct.

## Phase D update (2026-05-24 night): retrain offline metrics — strong recovery

Both runs retrained successfully with `mel.type: "tucker"` (jobs 9136231 control + 9136232 KD; walltime 3h39m + 4h26m on gpu-rtx6000 nodes 11-0 + 1-0). W&B runs: `g5rblf6o` (control) + `0w29iaut` (KD), project `birdclef-2026-exp032`.

| Run | val/auroc (Tucker macro AUC) | val/cmap (Tucker macro AP) | best ckpt val/cmap | Δ vs Phase 1 broken trajectory |
|---|---:|---:|---:|---|
| distilled_sed_control (retrain, ep 24) | **0.887** | 0.645 | 0.659 (top-ranked epoch) | Phase 1 W&B showed 0.111 (broken metric); post-hoc TuckerAUC was 0.893 on a triple-normalized val that LB never saw. |
| distilled_sed_kd (retrain, ep 24) | **0.908** | 0.692 | 0.709 | Phase 1 W&B showed 0.111; same caveat. |
| **paired Δ (KD − control)** | **+0.021** | **+0.047** | +0.050 | — |

**Critical caveat**: these offline numbers are computed on val fold 0 SS windows using the training pipeline (which now matches the inference pipeline post-D.3). They are *consistent* with Tucker's published 0.91 single-fold band (his ns22_macro peaks ~0.93; ours macro 0.887 / 0.908). They are NOT a guarantee that LB will be similar — that requires D.5 end-to-end validation to confirm pipeline consistency, then LB submission to read.

The paired Δ matches Tucker's published +0.020 LB lift directionally on val/auroc. This is the first internally-consistent signal that the KD branch adds value in our codebase.

### D.5 end-to-end validation PASSED (2026-05-24 night)

Jobs 9156567 (control) + 9156568 (KD) ran `validate_end_to_end.py` against the Phase D `last-v2.ckpt` files. Each took ~1m walltime.

| Run | Training-side val/auroc | End-to-end val/auroc | Gap | Status |
|---|---:|---:|---:|---|
| control | 0.887 | 0.898 | 0.011 | ✅ PASS (< 0.02 tol) |
| kd | 0.908 | 0.912 | 0.004 | ✅ PASS (< 0.02 tol) |

**End-to-end is slightly HIGHER than training-side for both**, because the kernel path applies 5-tap Gaussian smoothing across the 12 windows per file in LOGIT space (audit § 7.3), and that acts as a temporal denoiser. Tucker's discussion thread notes the same effect (~+0.004 lift from this smoothing trick).

**Pipeline is now confirmed end-to-end consistent. LB submission is safe.**

For comparison: had we run validate_end_to_end on the Phase 1 broken checkpoints, training-side would have been ~0.89 and end-to-end would have been ~0.52 (matching LB). Gap ~0.37 → FAIL. The pre-flight check would have caught that bug before we burned the LB submission slot.

### What the retrain log + metrics confirm

- ✅ TuckerMacroAUC + TuckerMacroAP are now logging correctly (no more 0.11 dragging from 0-positive classes).
- ✅ Training loss + dist_loss are healthy: control reached val/loss 0.0399, KD reached val/loss 0.0307 + `dist_loss` 0.003 (Tucker's published floor).
- ✅ No NaN or instability anywhere across 25 epochs.
- ✅ `last.ckpt` saved for both. All 25 best-by-cmap checkpoints saved (save_top_k=25 effective).

### Diagnostic — phase 1 broken val vs phase D fixed val

This table reframes what we know now about offline metrics across the two runs:

|  | Phase 1 (broken mel + broken metric) | Phase D retrain (fixed) |
|---|---|---|
| Training mel pipeline | NNAudio `amplitude_to_db` (3 norm passes) | TuckerMelSpec + per-sample z-score (1 norm pass) |
| Inference mel pipeline (`compute_mel_tucker`) | torchaudio + AmpToDB + z-score (Tucker-faithful) | Same as Phase 1 (still Tucker-faithful) |
| Train vs inference distribution diff | max\|d\| = 3.20 (z-scored) | predicted < 1e-5 (same code paths) |
| W&B `val/auroc` (during training) | 0.111 (broken MultilabelAUROC) | 0.887 / 0.908 (TuckerMacroAUC) |
| Post-hoc Tucker AUC on val | 0.889 (matched Tucker reference) | 0.887 / 0.908 (matches W&B) |
| Mean of val/cmap (train metric) | 0.058 (broken MultilabelAP) | 0.645 / 0.692 (TuckerMacroAP) |
| LB | 0.522 / 0.513 | **TBD — needs D.5 then re-submit** |

The Phase 1 post-hoc Tucker AUC was 0.889 (control ep 6), measured on the broken-normalized val set. The Phase D Tucker AUC is 0.887, measured on the Tucker-normalized val. The fact that these match closely is consistent with the model "learning to predict labels equally well on either input distribution" — the bug was never in label-learning capacity, it was in train↔inference distribution drift.

## Phase 2 (2026-05-25 EOD): 5-fold KD ensemble + HGNet+KD ablation — TRAINING DONE, LB PENDING

Per Phase D CONFIRM verdict, Phase 2 extends in two directions:
- **2.A 5-fold KD ensemble**: train folds 1-4 alongside Phase D fold 0 (LB 0.906). Tucker's published 0.917 LB = 5-fold + 5-tap Gaussian smoothing across windows.
- **2.D HGNet+KD ablation**: single-fold (fold 0) backbone swap `tf_efficientnet_b0.ns_jft_in1k` → `hgnetv2_b0.ssld_stage2_ft_in1k` (v5j's backbone) with `use_post_head: true → false`. Tests whether the −0.041 Phase D control-vs-v5j gap is backbone-driven.

All 5 runs trained 25 ep + 2 ep warmup + cosine, AdamW lr=5e-4 wd=1e-4, fp16, grad_clip=1.0, BCE dual-supervision 50/50, blend 50/50, Beta(0.4,0.4) hard-max mixup, SpecAugment, 90/10 focal/SS MixSamp, seed=42, `save_top_k: -1`. Configs: `distilled_sed_kd.yaml` (folds 1-4 with `train.py --fold N`) + `distilled_sed_kd_hgnet.yaml` (HGNet).

### Phase 2 training offline metrics (val on each ckpt's held-out fold)

| Slug | Backbone | Fold | val/auroc (last ep24) | val/cmap (last) | val/top1 | val/top5 | best val/cmap | Best Ep | Walltime | SLURM |
|------|----------|------|---------------------:|----------------:|---------:|---------:|--------------:|--------:|----------|------:|
| `distilled-sed-kd-fold1` | EffNet-B0 NS-JFT | 1 | **0.9281** | 0.6654 | 0.8029 | 0.9197 | 0.6777 (ep11) | 11 | 4h22m | 9158704 |
| `distilled-sed-kd-fold2` | EffNet-B0 NS-JFT | 2 | 0.9079 | 0.6092 | 0.8082 | 0.9658 | 0.6237 (ep21) | 21 | 4h22m | 9158705 |
| `distilled-sed-kd-fold3` | EffNet-B0 NS-JFT | 3 | 0.9236 | 0.6927 | 0.7919 | 0.9396 | 0.6954 | ~24 | 4h22m | 9158706 |
| **`distilled-sed-kd-fold4`** | EffNet-B0 NS-JFT | 4 | **0.9310** | **0.7295** | **0.9013** | **0.9737** | 0.7335 (ep21) | 21 | 4h22m | 9159168 |
| `distilled-sed-kd-hgnet` | HGNetV2-B0 SSLD | 0 | 0.9069 | 0.6706 | 0.7613 | 0.9097 | 0.6963 | ~17 | 4h19m | 9158713 |

Phase D fold 0 reference (LB 0.906): val/auroc 0.908, val/cmap 0.692.

**5-fold KD val/auroc set**: {0.908 (fold 0), 0.9281, 0.9079, 0.9236, 0.9310}, mean **0.920**, std 0.011.

Read-outs:
- **Fold 0 was the bottom of the 5-fold band** (0.908 vs mean 0.920) — explains why Phase D single-fold LB 0.906 sits at the low end of Tucker's published single-fold trajectory (~0.91-0.92). Fold 4 is the strongest individual fold (0.931); fold 2 the weakest (0.908).
- **HGNet+KD val/auroc (0.907) ≈ Phase D EffNet+KD fold 0 (0.908)** — no obvious backbone advantage at the offline-metric level. Both `val/cmap` and `val/top1` are similar between HGNet and EffNet fold 0. LB will say whether the backbone matters for soundscape generalization.

W&B run IDs: `pmbh` (fold1) · `hoqe` (fold2) · `bpuk` (fold3) · `lwy9` (fold4) · `hyu0` (HGNet), all in `dsgt-arc/birdclef-2026-exp032`.

### D.5 end-to-end validation — ALL PASS ✅ (pre-flight before LB submission)

Each Phase 2 ckpt evaluated via `validate_end_to_end.py` (kernel inference path: `kernel_load_audio_chunk → compute_mel_tucker → encoder → AttHeadLogitSpace → LOGIT-space blend → 5-tap Gauss smooth → single sigmoid` on the same per-fold val SS pool). Gate: |training-side val/auroc − e2e val/auroc| ≤ 0.02.

| Slug | Training-side val/auroc | E2E val/auroc | Gap | Status | n_evaluable | Val windows |
|------|----------------------:|--------------:|----:|--------|------------:|------------:|
| `distilled-sed-kd-fold0` (Phase D) | 0.908 | 0.912 | +0.004 | ✅ PASS | — | 156 |
| `distilled-sed-kd-fold1` | 0.9281 | 0.9361 | +0.008 | ✅ PASS | 31 | 156 |
| `distilled-sed-kd-fold2` | 0.9079 | 0.9190 | +0.011 | ✅ PASS | 39 | 156 |
| `distilled-sed-kd-fold3` | 0.9236 | 0.9302 | +0.007 | ✅ PASS | 41 | 156 |
| `distilled-sed-kd-fold4` | 0.9310 | 0.9366 | +0.006 | ✅ PASS | 45 | 156 |
| `distilled-sed-kd-hgnet` | 0.9069 | 0.9108 | +0.004 | ✅ PASS | 29 | 156 |

E2E is uniformly slightly **higher** than training-side — the kernel path adds 5-tap Gaussian smoothing across the 12 windows per file in LOGIT space, which acts as a temporal denoiser. Matches Tucker's "+0.004 from MengYe smoothing trick" pattern (also seen in Phase D). **Pipeline is end-to-end consistent across all 5 new ckpts. LB submission is safe.**

Per-fold n_evaluable varies (29-45 classes with positives in that fold's val pool) — fold 4 has the most coverage (which may explain why it trained highest), fold 0 the least.

### D.5 methodology bug + fix (2026-05-25)

The first D.5 run on the 4 EffNet fold ckpts (jobs 9178659-62) showed e2e AUC ≈ **0.995** for all four — gap +0.066 to +0.090, all marked FAIL. Root cause: `validate_end_to_end.py` only read `fold_k` from the YAML config (which defaults to 0). The `--fold N` argparse override that `train.py` accepts was never wired into the validation script. So for ckpts trained with `--fold 1/2/3/4`, the script loaded **fold 0's** val pool — which was *training data* for those models → measuring memorization, not generalization. HGNet (job 9178663) PASSED on the first round because it was trained on fold 0, so the val pool matched correctly.

Fix:
- `scripts/validate_end_to_end.py`: added `--fold N` CLI arg that overrides `cfg["data"]["fold_k"]` (with the same `[0, 4]` guard as `train.py`).
- `sbatch/validate_end_to_end.sh`: added `FOLD` env var forwarding (`FOLD=N → --fold N`).

Re-running jobs 9178859-62 with `FOLD=N` produced the corrected PASS table above. **Training was never broken** — only the validation script was. This is consistent with the training-side val/auroc values being sensible held-out numbers (0.91-0.93 range); had training been evaluated on its own data, those would already have been ~0.995 in the W&B/`metrics.json`.

### Phase 2 checkpoints

- Top-level `last-v2.ckpt` at `2026-exp032-distilled-sed-kd/last-v2.ckpt` — Phase D KD fold 0 (LB 0.906).
- `2026-exp032-distilled-sed-kd/fold{1,2,3,4}/last.ckpt` — Phase 2.A folds 1-4 (last epoch 24, save_top_k=-1).
- `2026-exp032-distilled-sed-kd-hgnet/last.ckpt` — Phase 2.D HGNet+KD single-fold (last epoch 24).

### LB expectations (predictions)

| Submission | Predicted LB | Decision (per § Phase 2 matrix) |
|---|---|---|
| `distilled-sed-kd-5fold` (5 folds, logit-space mean + 5-tap smoothing) | **0.910–0.920** | Matches Tucker's 0.917 if recipe works at 5-fold scale; consider ensembling with exp 023 (0.929) for further lift |
| `distilled-sed-kd-hgnet` (single-fold, fold 0) | **0.90-0.92** | If LB > 0.910, backbone was a real bottleneck → pivot to HGNet for future training; if ~ 0.906, EffNet was fine |

The 5-fold mean val/auroc 0.920 is encouraging. With 5-tap Gaussian smoothing across folds in LOGIT space, the ensemble's effective resolution-per-window improves AND the per-fold noise averages out. Tucker's 0.917 LB came from exactly this stack at 5-fold.

## Phase 2 LB results (2026-05-25 PM): PARTIAL REPLICATION

| Submission | LB | Δ vs Phase D fold 0 (0.906) | Δ vs Tucker target | Decision-matrix bucket |
|---|---:|---:|---:|---|
| `distilled-sed-kd-5fold` (5-fold ensemble) | **0.915** | **+0.009** | **−0.002 vs 0.917** | **0.910–0.919 band** → matches Tucker's published 0.917 within noise; consider exp 023 ensemble |
| `distilled-sed-kd-hgnet` (single-fold ablation) | **0.901** | **−0.005** | — | **0.900–0.910 band** → marginal/slightly worse; EffNet was the right backbone |
| `distilled-sed-kd-fold4` (single-fold diagnostic) | **0.903** | **−0.003** | — | offline-LB anti-correlation: best-offline fold gave near-worst LB |

### Read-out

**5-fold ensemble: replicates Tucker's publicly-quantified pipeline within ±0.002 LB.**
- +0.009 single→5-fold lift matches Tucker's published +0.011 directionally.
- 0.002 gap to Tucker's 0.917 is well inside the documented cross-hardware variance (~0.007 LB on the same training code; exp 031 drift investigation).
- **What we did NOT replicate**: Tucker's "0.93+ with PP" stretch claim, which depends on his private post-processing (site/hour priors + per-class temperature; NOT in the public notebook).
- Bottom line: the publicly-replicable subset of his recipe yields 0.91–0.92 LB on this codebase, the same band his discussion thread reports.

**HGNet+KD: backbone was NOT the bottleneck.**
- HGNet (0.901) underperforms EffNet fold 0 (0.906) by −0.005 on LB despite near-identical offline val/auroc (0.907 vs 0.908).
- Refutes the Phase D hypothesis that the −0.041 control-vs-v5j gap was backbone-driven.
- Implication: the recipe-vs-recipe gap (Tucker recipe control 0.856 vs v5j_control 0.897) is driven by other ingredients — `AttHeadLogitSpace` vs LSEHead, dual-supervision vs single-clipwise, etc. NOT the EffNet→HGNet swap.

### What was actually replicated

| Tucker's published claim | Our result | Status |
|---|---|---|
| Single-fold KD lift ≈ +0.020 LB over no-KD | +0.050 paired Δ on Tucker recipe (Phase D) | ✅ exceeded |
| Single-fold LB band ~0.91 | distilled-sed-kd fold 0 LB 0.906 | ✅ in band (lower end) |
| 5-fold + smoothing LB 0.917 | distilled-sed-kd-5fold LB 0.915 | ⚠️ within ~0.002 (likely cross-hardware noise) |
| "0.93+ with PP" | not attempted (PP code is private) | ❌ out of reach |
| HGNet backbone improves over EffNet | tested, refuted on this stack (HGNet < EffNet) | ✅ ablation done, hypothesis rejected |

### Single-fold variance characterization (fold 0 + fold 4 LB landed)

| Fold | Offline val/auroc | E2E val/auroc | LB | Δ offline vs fold 0 | Δ LB vs fold 0 | Notes |
|---|---:|---:|---:|---:|---:|---|
| 0 (Phase D) | 0.908 | 0.912 | **0.906** | — | — | bottom of offline band; LB-validated reference |
| 1 | 0.928 | 0.936 | (not submitted) | +0.020 | — | middle of offline band |
| 2 | 0.908 | 0.919 | (not submitted) | 0.000 | — | tied with fold 0 offline |
| 3 | 0.924 | 0.930 | (not submitted) | +0.016 | — | middle of offline band |
| **4** | **0.931** | **0.937** | **0.903** | **+0.023** | **−0.003** | **strongest offline → LB came back near-worst of the two folds tested** |

**Fold 4 result is a strong offline-LB anti-correlation data point** — `+0.023 offline → −0.003 LB`. Two single-fold LB points (0.903, 0.906) cover a narrow 0.003 range despite their offline values spanning 0.023. Implications:
- **Single-fold LB band on this codebase is narrow (~0.003)** — much tighter than the offline-implied 0.023 spread. Offline val/auroc is essentially uninformative for ranking folds at LB.
- **The +0.009 single→5-fold ensemble lift (0.906 → 0.915) is pure variance-reduction**, not "picking up better folds". The Gaussian smoothing across 5 fold logits in LOGIT space is the lever.
- **Don't submit folds 1/2/3** as additional diagnostic — we already have the answer. The remaining 3 folds would land in the same 0.903-0.906 band with high confidence; low marginal information for high LB-slot cost.
- **Aligned with broader exp 030/031 pattern**: offline metrics consistently anti-correlate with LB on this competition. Documented across 22+ prior recipe attempts; this is data point #23.

This also explains why our Phase D single-fold landed at 0.906 (below Tucker's published 0.91-0.92 single-fold band): the single-fold band on Tucker's hardware may genuinely be wider than ours, OR his published "fold-1 = 0.922" was just a lucky fold instance. Either way, the 5-fold ensemble is what closes the gap (0.915 ≈ his 0.917), and the path forward doesn't need more single-fold submissions.

### Implication for next steps

The recipe is replicated; the path to higher LB is no longer "fix the recipe":
1. **Ensemble distilled-sed-kd-5fold (0.915) with exp 023 perch-sed-ensemble-mlp-only (0.929)** — highest-EV move. Two strong, decorrelated stacks (Tucker SED + Perch probes) average to a higher number than either alone. Estimated band: 0.93–0.94.
2. **Port site/hour priors (EoS.6 style) onto distilled-sed-kd-5fold** — exp 021 priors lifted v5j-4fold by +0.004; similar lift here would land 0.918–0.920.
3. **Accept the publicly-replicable limit** of this recipe and pivot LB-lift effort to other levers (Sydorskyi-yoink, MixIT-based PL, etc.).

LB submission slots are precious near deadline (2026-06-03, 9 days out); options 1 and 2 are the highest-EV with what's already built.

## Phase D update (2026-05-24 EOD): mel mismatch CONFIRMED + fix landed

`scripts/diag_mel_pipeline.py` on a real soundscape (5s @ 32 kHz) confirms
the leading hypothesis numerically:

| Comparison | max\|d\| | Pearson r | Status |
|---|---:|---:|---|
| TUCKER vs OURS-KERNEL (compute_mel_tucker) | 4.77e-07 | 1.000000 | ✅ match |
| TUCKER vs OURS-TRAINING (NNAudio amp2db) | **3.20** | 0.9356 | ❌ mismatch |
| OURS-TRAINING vs OURS-KERNEL | **3.20** | 0.9356 | ❌ mismatch |

Kernel is Tucker-faithful to machine epsilon. **Training is not.** The
model was trained on a distribution max|d| = 3.20 away from what the kernel
feeds it at inference — sufficient to explain the LB collapse.

### Fix (commit pending): TuckerMelSpectrogram

- NEW `user/murilo/src/birdclef/preprocessing/tucker_mel.py` — verbatim
  port of Tucker's `MelSpecTransform` (torchaudio MelSpec + AmplitudeToDB(80),
  zero internal normalization).
- `train.py` dispatches on `mel.type == "tucker"` → instantiates `TuckerMelSpectrogram`.
- `DistilledSEDFinetuneModule._compute_mel.per_sample_mel_norm=True` (unchanged)
  applies the single z-score on top. Combined pipeline matches Tucker's
  training pipeline to atol=1e-5 (verified by 12 unit tests).
- Both configs `distilled_sed_{control,kd}.yaml` updated: `mel.type: "tucker"`,
  `norm_type` removed. `power: 2.0` made explicit.
- `test_diverges_from_nnaudio_amplitude_to_db` regression guard locks the
  fix in place — any accidental revert to the broken NNAudio path fails CI.

**Tests: 163/163 pass** (was 151 + 12 new).

Next step: retrain paired control + KD with the new mel pipeline. Expected
outcome on retrain: offline val/cmap macro AUC remains similar (~0.89 at
ep 6, matches Tucker), **AND** end-to-end inference on
`train_soundscapes_labels.csv` now also reads ~0.89 macro AUROC (D.5
validation). If both hold, LB submission should land in Tucker's published
0.910–0.917 band.

## Analysis: leading hypothesis

### 🚨 Mel-pipeline mismatch between training and inference (HIGHEST-LIKELIHOOD) — CONFIRMED 2026-05-24 EOD

Tucker's reference pipeline (verified against
`vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.py` lines 400–407, 947–949, 1240–1246):

```
power_mel → AmplitudeToDB(top_db=80) → per-sample (mel - mean) / (std + eps)
```

**One normalization pass. Identical at training and inference.**

Our **training** pipeline (commit `8d64be8` HEAD):

1. `NNAudioMelSpectrogram(norm_type="amplitude_to_db")` does:
   - `10 * log10(mel + 1e-10)`
   - clamp to `mel_max - top_db`
   - per-sample z-score (mean, std)
   - **per-sample min-max scale to [0, 1]** ← not in Tucker
2. Then `DistilledSEDFinetuneModule._compute_mel` does **another** per-sample
   z-score (mean, std) over (1, F, T) ← second normalization on top of step 1

So our training does **three normalizations** (z-score → min-max → z-score)
while Tucker does **one** (z-score after AmplitudeToDB).

Our **inference** pipeline (`submission.py:compute_mel_tucker`, NEW today):
matches Tucker's published recipe verbatim — torchaudio MelSpec +
AmplitudeToDB(80) + single per-sample z-score.

**Consequence**: model trained on a triple-normalized distribution sees
single-normalized features at inference. The input distribution is
completely different from what the model learned to interpret — output is
near-random.

This single bug is sufficient to explain a 0.522 LB given a model that
scores 0.89 macro AUC on its training-distribution val data.

### Other candidate causes (lower-likelihood, to be ruled in/out during audit)

1. **NNAudio vs torchaudio MelSpectrogram numerical drift**. Even with
   matched n_fft/hop/n_mels, the two backends differ in mel filterbank
   construction (slaney vs htk default, normalization). Should be small but
   not zero.
2. **AttHeadLogitSpace forward shape**. Tucker's notebook line 546 permutes
   framewise to `(B, T, C)` BEFORE returning; ours returns `(B, C, T)`.
   The `frame_max_logits = frame_logits.max(dim=-1).values` in our wrapper
   maxes over T (axis -1 = T for `(B, C, T)`), giving `(B, C)`. Tucker maxes
   over `dim=1` on `(B, T, C)`, also giving `(B, C)`. Both equivalent.
   **Likely OK; needs verified test**.
3. **Audio loading SR / chunk indexing**. `submission.py:load_audio_chunk`
   loads `sr * duration_s` samples; for 5s @ 32kHz = 160000 samples,
   matching Tucker. **Likely OK**.
4. **ONNX wrapper output**. Our wrapper outputs single pre-blended logits
   `0.5·clip + 0.5·frame_max`. Tucker's ONNX outputs `(clip_logits, framewise_logits)`
   separately and does the blend in the kernel. Semantically equivalent
   when blends match. **Likely OK; needs verified test on identical mel input**.
5. **Species mapping**. 234/234 identity. Used same `species_mapping_2026_234.json`
   that exp 030/031 use and which produced 0.897 LB there. **Almost certainly OK**.
6. **Training fold (only fold 0)**. Single-fold v5j has ~0.013 seed band;
   doesn't explain 0.38.
7. **`save_top_k=-1` interactions**. Both control + KD used `last.ckpt` (ep 24).
   The ranker showed ep 24 in the top cluster (0.8928 AUC); same checkpoint
   that's now LB 0.522.

### Why ALL other audited components look correct

- ✅ Encoder: `tf_efficientnet_b0.ns_jft_in1k`, `use_post_head=True`, 1280-d post-conv_head — verified by 11 tests in `test_efficientnet_encoder_post_head.py`.
- ✅ Head: `AttHeadLogitSpace` with GeMFreqPool(p=3, clamp), Dropout(0.25)→Linear→ReLU→Dropout(0.5), xavier conv init — verified by 14 tests.
- ✅ Loss: dual-supervision (0.5·BCE_clip + 0.5·BCE_frame_max) — synthetic comparison vs Tucker formula gives diff 1.19e-07 (verified by 26 tests).
- ✅ Mixup: waveform mixup Beta(0.4,0.4), MIXUP_HARD=True (max labels), p=0.5 focal-focal + p=0.5 focal×SS — verified by 18 tests including audit-rate validation (50/25/25%).
- ✅ SpecAugment: 1 freq×10 + 2 time×10 — verified by 12 tests.
- ✅ Schedule: AdamW(5e-4, 1e-4) + LinearLR(start_factor=1/25, 2ep warmup) → CosineAnnealingLR(T_max=23ep, eta_min=1e-6) — verified by 4 schedule tests.
- ✅ MixSamp: 90/10 focal/SS, per_src=[58, 6] — verified by 14 tests including DataLoader integration.
- ✅ Folds: StratifiedKFold focal + GroupKFold SS, 66 labeled SS files → 792 windows (53 all-zero) — verified by 20 tests.
- ✅ Tucker macro AUC metric: skips 0-positive classes — verified by 11 tests; reproduces Tucker's 0.889 at control epoch 6.

The component-level audit is clean. The bug is in **pipeline composition** —
specifically the mel normalization stack.

### What the offline metric can't tell us

The val pool is 156 SS windows from fold 0 (13 files × 12 windows). The
labels matrix has ~29 evaluable classes (out of 234) with any positives.
A model trained on triple-normalized features and validated on the SAME
triple-normalized features can easily score 0.89 macro AUC on those 29
classes — the model just learned the distribution it was given. **Offline
val/cmap/auroc cannot detect a training-inference distribution mismatch**
because the offline val uses the training pipeline.

This is the key methodology failure. We need to:
1. Validate offline with the actual inference pipeline (mel + everything).
2. Ideally validate offline against held-out audio loaded the same way the
   Kaggle kernel loads it.

## Implementation pivot needed

### Immediate (next session) — comprehensive audit

- [ ] **Pull Tucker's notebook fresh** (`kaggle kernels pull tuckerarrants/bc2026-distilled-sed`) and diff against vendored copy at `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`. Confirm no upstream change since 2026-05-23.
- [ ] **Line-by-line diff** of our `_compute_mel` flow vs Tucker's `MelSpecTransform` + per-sample norm loop (notebook lines 400–407, 945–950).
- [ ] **Side-by-side numerical test**: feed the SAME waveform through Tucker's pipeline (vendored `.py`) and through our `DistilledSEDFinetuneModule._compute_mel`. Compare mel outputs element-wise. Expected: identical to ~1e-6 if pipelines match; large diff if mismatched.
- [ ] **Audit `compute_mel_tucker` in submission.py** against Tucker's inference cell (`librosa.feature.melspectrogram` + `librosa.power_to_db(top_db=80)` + per-sample z-score). Confirm it matches the **training** distribution our model was actually trained on — currently it matches Tucker's training pipeline, NOT ours.
- [ ] **Fix training pipeline** to match Tucker's single-pass normalization. Likely fix: change config `norm_type` from `"amplitude_to_db"` (NNAudio's triple-norm) to something that does ONLY `10*log10 + top_db_clamp` and let the module's `per_sample_mel_norm` do the single z-score on top. Or add a new `norm_type: "tucker"` that does exactly `power_to_db + top_db_clamp` without z-score or min-max (lets `_compute_mel`'s single z-score complete the pipeline).
- [ ] **Retrain both runs** with the fixed mel pipeline.
- [ ] **Re-export + re-submit** with the same `compute_mel_tucker` inference path (which IS Tucker-faithful — the inference path is what we want).

### Strategic question to revisit

Is the right fix to (a) align our training mel to Tucker's verbatim, or
(b) accept our codebase's mel and re-engineer the inference kernel to match
our training? Option (a) preserves audit-validated Tucker faithfulness;
option (b) is faster but exits the Tucker-replication framing.

Default = (a).

### Don't trust offline metrics until validated end-to-end

After the mel fix, validate the training distribution by:
1. Re-running training for 1 epoch.
2. Loading the resulting checkpoint, running inference on `train_soundscapes/`,
   computing macro AUROC against `train_soundscapes_labels.csv`, and comparing
   against the **same checkpoint's offline val** macro AUC. Should match to
   2-3 dp; if not, the mismatch is still there.

This validation should happen BEFORE any LB submission going forward.

## Confounders

- Two confounders go away once the mel fix lands:
  - Multi-variable port: irrelevant — the recipe didn't actually fail; the
    ate-its-tail mel pipeline did.
  - Single-fold seed noise: −0.38 LB is two orders of magnitude beyond any
    plausible seed band.

## Artifacts

| Type | Location |
|---|---|
| Configs | `configs/distilled_sed_{control,kd}.yaml` |
| Logs | `logs/slurm_distilled_sed_{control_9088255,kd_9088258}.log` |
| W&B | `dsgt-arc/birdclef-2026-exp032/{nn9inb1u,rsxukve7}` |
| TensorBoard | `outputs/tensorboard/` |
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints/{2026-exp032-distilled-sed-control,2026-exp032-distilled-sed-kd}/` (all 25 epochs + last.ckpt) |
| Metrics JSON | `artifacts/training/{distilled_sed_control,distilled_sed_kd}/metrics.json` |
| Checkpoint ranking (control) | `outputs/checkpoint_ranking_control.tsv` (post-hoc; ep 1 ranked #1 due to small evaluable class pool; ep 24 in top 9) |
| Checkpoint ranking (KD) | `outputs/checkpoint_ranking_kd.tsv` (rerun after strict-load fix) |
| ONNX exports | `experiments/010-submission-pipeline/artifacts/weights/{distilled-sed-control,distilled-sed-kd}/model.onnx` |
| Kaggle kernels | `murilogustineli/birdclef-2026-distilled-sed-{control,kd}` |
| Diagnostic script | `scripts/diag_val_auroc.py` (val-metric bug, fixed yesterday) |
| Code | branch `mg/exp032-tucker-distilled-sed-replication`, PR #51 |
| Tucker source (vendored) | `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.{ipynb,py}` (re-pullable via `kaggle kernels pull tuckerarrants/bc2026-distilled-sed`) |

## Next Steps

See `tasks.md` Phase D (new). Do NOT proceed to Phase 2 (5-fold ensemble) or
Phase 3 (KD alpha sweep, longer schedule, etc.) until the mel-pipeline audit
lands and we have a re-trained checkpoint whose LB approaches the secondary
target of 0.900 (`distilled_sed_control LB vs v5j_control_last 0.897 ≥ 0.900`).

**Leads to**: comprehensive audit → mel-pipeline fix → re-train both runs →
re-submit. If retrained control still doesn't approach 0.90 LB, investigate
the lower-likelihood candidates ranked above (NNAudio backend drift, head shape,
audio loading).
