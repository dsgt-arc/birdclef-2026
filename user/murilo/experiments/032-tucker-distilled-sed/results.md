---
experiment: "032-tucker-distilled-sed"
status: rejected-blocked
date: 2026-05-24
author: Murilo Gustineli
---

# Results: Tucker distilled-SED full replication

**Date Completed (Phase 1)**: 2026-05-24
**Author**: Murilo Gustineli

## Status: REJECTED (catastrophic) — PIPELINE COMPROMISED, PAUSE FOR AUDIT

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

## Analysis: leading hypothesis

### 🚨 Mel-pipeline mismatch between training and inference (HIGHEST-LIKELIHOOD)

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
