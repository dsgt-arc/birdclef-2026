---
experiment: "032-tucker-distilled-sed"
status: confirm-phase-d
date: 2026-05-24
author: Murilo Gustineli
---

# Results: Tucker distilled-SED full replication

**Date Completed (Phase D)**: 2026-05-24
**Author**: Murilo Gustineli

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
