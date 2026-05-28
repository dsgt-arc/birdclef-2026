# Tasks — Exp 034 Pretrained-backbone yoink

## Tests

Tests will live in `user/murilo/tests/034-pretrained-backbone-yoink/` (mirroring the experiment dir).

Run with the interactive venv:

```bash
~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/034-pretrained-backbone-yoink/ -v
```

Test files (planned — to be created in Phase B):

| Test file | Coverage |
|---|---|
| `test_nnaudio_mel.py` | nnAudio mel pipeline output shape/dtype, parity against captured reference tensor |
| `test_sed_head.py` | SED-attention head forward pass + dropout positions + GeM freq pool numerical correctness |
| `test_checkpoint_loader.py` | Sydorskyi pretrained ckpt loads with strict=True; conv weight norms non-zero post-load |
| `test_focal_bce_loss.py` | FocalBCELoss matches reference α/γ behavior; gradient finite on synthetic batch |
| `test_class_balanced_sampler.py` | γ=−0.5 sqrt-balanced WeightedRandomSampler empirical sample frequency |
| `test_augmentation_pipeline.py` | Crop → MixUp(sum-clip) → BG noise → peak-norm → mel → RandomFiltering → SpecAugment full chain |
| `test_parity_5audio.py` | Phase C gate: cosine ≥0.999 on 5 fixed audios vs Sydorskyi's inference notebook outputs |

**Backward-compat gate**: `pytest user/murilo/tests/032-tucker-distilled-sed/ -q` + `pytest user/murilo/tests/033-own-teacher-pl/ -q` must still pass after any shared-src changes (we should NOT modify exp 032/033 src; build exp 034 src as a new module).

---

## Reference materials

Read these BEFORE starting any phase. Implementation tasks reference them by short name.

| Short name | File |
|---|---|
| **Proposal** | [`proposal.md`](proposal.md) |
| **Deep research review** | `user/murilo/notes/2026-05-28-research-report.md` |
| **Sydorskyi 2025 paper** | `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` (vendored) + CEUR Vol-4038 paper_256 |
| **Divergence audit (exp 033)** | `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` |
| **Source repo** | `github.com/VSydorskyy/BirdCLEF_2025_2nd_place` |
| **Pretrained weights (Kaggle)** | `vladimirsydor/bird-clef-2025-all-pretrained-models` |
| **Reference inference notebook (Kaggle)** | `vladimirsydor/bird-clef-2025-minimul-inference` (v245080754) — Phase C parity anchor |
| **Tucker recipe module (exp 032)** | `user/murilo/src/birdclef/lightning/distilled_sed_finetune.py` — reuse LightningModule patterns where compatible |
| **Inference path (exp 010)** | `experiments/010-submission-pipeline/sbatch/export.sbatch` — extend with new yoink-export helper |

---

## Phase A — Source reading + verified config

### A.1 Get the source — DONE 2026-05-28

- [x] A.1.1 Cloned `github.com/VSydorskyy/BirdCLEF_2025_2nd_place` as git submodule at `vendor/sydorskyi-2025/` (commit `053594f`).
- [x] A.1.2 Downloaded Kaggle dataset `vladimirsydor/bird-clef-2025-all-pretrained-models` (219 MB → 237 MB unzipped, ~10s) to `/storage/project/ps-clef2026_birdclef-0/shared/pretrained-models/sydorskyi-2025/`.
- [x] A.1.3 Verified 3 ckpts present: `models_2025/eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt` (84 MB), `models_2025/tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt` (77 MB, SWA), `models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth` (77 MB, legacy).
- [x] A.1.4 Inspected ckpt keys: flat OrderedDicts (no Lightning wrapping); both stems are `in_chans=1`; NFNet `final_conv` shape `(2304, 1536, 1, 1)` → 2304 ch ✅; EffNetV2-S missing `conv_head` + `bn2` → loadable backbone-out = **256 ch** (NOT 1280) ⚠️. Findings captured at `docs/sydorskyi-config-verified.md`.

### A.2 Read the source line-by-line — DONE 2026-05-28

- [x] A.2.1 `code_base/models/wave_clasifier.py` + `code_base/models/blocks.py` (AttHead, NormalizeMelSpec, TraceableMelspec, GeMFreq, ChannelAgnosticAmplitudeToDB) — head structure verified. Backbone instantiation uses `features_only=True` (explains EffNetV2-S ckpt truncation at `blocks.5.14.bn3`).
- [x] A.2.2 `code_base/datasets/wave_dataset.py` — mel pipeline + dataset augmentation chain verified. Standardize axes = `(1, 2)` per-sample over full spectrogram. MixUp with `alpha=None` = equal-weight wave + sum-clip target (the "cleverest piece" from audit, verified line-by-line).
- [x] A.2.3 `code_base/losses/focal_loss.py` (`FocalLossBCE`) + `code_base/losses/combined_losses.py` (`BCEFocal2WayLoss`) — FocalLossBCE: α=0.25, γ=2.0, bce_weight=1.0, focal_weight=1.0, reduction="mean". With `loss_type="baseline"`, applied to `clipwise_logits_long` only (no framewise-max term).
- [x] A.2.4 `code_base/train_functions/train_lightning.py:220-265` — sampler / class-balancing logic. Confirmed `power_value=-0.5` default. NFNet `class_weights_path="sqrt"` → `w_c = (n_c/N)^-0.5`; EffNetV2-S `class_weights_path="balanced"` → `w_c = 1/n_c`. ⚠️ Proposal had this backwards; corrected in verified-config doc.
- [x] A.2.5 `scripts/create_pretrain_backbone.py` — ckpt key format confirmed. Strips `model.backbone.` prefix from Lightning ckpt → outputs flat `OrderedDict`. Distributed ckpts already in this format. Compatible with `timm features_only + strict=True` load.
- [x] A.2.6 `train_configs/{selected_ebs,selected_eca,best_solo_ebs}.py` — fully-resolved hparams captured. ⚠️ Key finding: EffNetV2-S configs reference `models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth` (NOT the 2025 SWA ckpt). NFNet references `models_2025/eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt`.
- [x] A.2.7 `code_base/models/augmentations.py` (RandomFiltering) — 4-band STFT-domain random EQ. min_db=-20, n_fft=1024 (different from mel n_fft=2048), peak-normalize after iSTFT.
- [x] A.2.8 `code_base/augmentations/spec_augment.py` + `code_base/augmentations/transforms.py` (OneOf, BackgroundNoise) — SpecAugment wraps torchaudio FrequencyMasking + TimeMasking. OneOf picks one transform uniformly; each branch has internal p=0.5. Net BG noise: 25% nocall + 25% ESC50 + 50% none.
- [x] A.2.9 `code_base/forwards/cls_forward.py` (`MultilabelClsForwardLongShort`) — forward wrapper. `loss_type="baseline"` + `use_bce_focal_loss=True` → FocalLossBCE on `clipwise_logits_long`. RandomFiltering applied to waveform before mel.

### A.3 Write verified config doc — DONE 2026-05-28

- [x] A.3.1 Created `docs/sydorskyi-config-verified.md` with every config cell line-cited against source. Replaces the in-flight skeleton with comprehensive Phase A.2 findings.
- [x] A.3.2 All "Open questions" from proposal table resolved:
  - Head hidden dim: **512** ✅
  - GeM freq pool: `p=3.0` learnable, `eps=1e-6` ✅
  - Standardize axes: `(1, 2) = (n_mels, time), per-sample` ✅
  - mel `normalized=True`: divides by `||hann_window||₂` after STFT ✅
  - FocalLossBCE: α=0.25, γ=2.0, 1.0/1.0 weights, reduction="mean" ✅
  - Class-balancing γ: ⚠️ CORRECTED (NFNet=sqrt/γ=-0.5, EffNetV2-S=balanced/γ=-1)
  - Ckpt key prefix: NONE (flat OrderedDict, no wrapping) ✅
  - Backbone output channels: ⚠️ CORRECTED (EffNetV2-S = **256** via features_only, NFNet = **2304**)
  - Mel library: ⚠️ partial correction — nnAudio STFT + torchaudio MelScale (hybrid, not pure nnAudio)
  - Which ckpt for EffNetV2-S: ⚠️ CORRECTED — **2024** ckpt, NOT 2025 SWA

**Phase A exit criterion**: ✅ `docs/sydorskyi-config-verified.md` landed with every config cell either HIGH-confidence VERIFIED or explicitly out-of-scope. No "MED" or "LOW" cells remain in the setup table. 5 corrections to original proposal documented.

---

## Phase B — Implementation

### B.1 Dependencies — DONE 2026-05-28

- [x] B.1.1 `experiments/034-pretrained-backbone-yoink/pyproject.toml` created. Pins inherit exp 033 ML stack + adds `torchvision==0.20.1` (needed for `torchvision.ops.focal_loss.sigmoid_focal_loss` in B.5). `nnAudio==0.3.4` already present.
- [x] B.1.2 No new install needed — active venv already has nnAudio 0.3.4, torchvision 0.25.0, timm 1.0.25, torch 2.10. Versions exceed pyproject pins but tests pass cleanly.
- [x] B.1.3 `import nnAudio; from nnAudio.Spectrogram import STFT as nnAudioSTFT` works (Sydorskyi uses STFT, not full MelSpectrogram — see verified-config doc § Mel pipeline).

### B.2 nnAudio mel pipeline — DONE 2026-05-28

- [x] B.2.1 `src/birdclef/preprocessing/yoink_mel.py` — direct port of Sydorskyi's mel chain (`TraceableMelspec` = nnAudio STFT + torchaudio MelScale; `ChannelAgnosticAmplitudeToDB`; `NormalizeMelSpec`; `GeMFreq`; `Swish`).
- [x] B.2.2 `build_sydorskyi_mel_extractor` convenience function composes the full chain (TraceableMelspec → ChannelAgnosticAmplitudeToDB → NormalizeMelSpec) with Sydorskyi-default params.
- [x] B.2.3 Unit tests: 10 tests in `test_yoink_mel.py` pass. Coverage: GeMFreq learnable p=3 init + exportable/non-exportable parity; NormalizeMelSpec per-sample standardize over (1,2) + output range [0,1]; ChannelAgnosticAmplitudeToDB 3D/4D acceptance + top_db=80 clip; TraceableMelspec output shape (2, 128, 313) for 5s @ 32kHz; full chain end-to-end finite + range.

### B.3 SED-attention head — DONE 2026-05-28

- [x] B.3.1 `src/birdclef/heads/yoink_sed_head.py` — `YoinkAttHead` port. Backbone-features → GeMFreq pool → squeeze + permute → dense(in_chans → 512) + dropout(0.25) + ReLU + dropout(0.5) → 2 parallel Conv1d (attention + value) → soft-attention-weighted aggregation in BOTH logit-space (loss) and prob-space (metric).
- [x] B.3.2 Default `p=0.5` master → dropout(p/2=0.25) before linear, dropout(p=0.5) after ReLU. ReLU activation.
- [x] B.3.3 Default `num_class=234` (BirdCLEF 2026). `train_period == infer_period` asserted (simplified vs original Sydorskyi which supports short-window crop path at inference).
- [x] B.3.4 Unit tests: 9 tests in `test_yoink_sed_head.py` pass. Coverage: constructor defaults match Sydorskyi (hidden 512); dropout positions/rates verified; ReLU activation verified; output dict has 3 keys with correct shapes; clipwise_pred in [0,1] (prob-space); clipwise_logits unbounded (logit-space); simplification assertion fires when train_period != infer_period; gradient flows to GeMFreq.p, attention.weight, fix_scale.weight.

### B.4 Checkpoint loader — DONE 2026-05-28

- [x] B.4.1 `experiments/034-pretrained-backbone-yoink/scripts/load_sydorskyi_backbone.py` — instantiates `timm.create_model(features_only=True, in_chans=1)` and loads Sydorskyi ckpt with `strict=True` (after stripping defensive `model.backbone.` / `backbone.` prefixes). CLI + Python API.
- [x] B.4.2 Both backbones (EffNetV2-S 2024 + NFNet-L0 2025) load with strict=True; no missing/unexpected keys. Conv weight norms confirmed non-zero post-load.
- [x] B.4.3 `backbone.feature_info.channels()[-1]` confirmed: **256 for EffNetV2-S** (features_only strips conv_head/bn2 — matches stripped ckpt) and **2304 for NFNet-L0**.
- [x] B.4.4 Unit tests: 6 tests in `test_yoink_backbone_loader.py` pass. Coverage: strict=True load for both backbones; output channels verification; conv weights non-zero; forward pass produces finite (B, 256/2304, H, W); unknown-backbone raises ValueError.

### B.5 WaveCNNAttenClasifier orchestrator — DONE 2026-05-28 (added beyond original Phase B scope)

- [x] B.5.1 `src/birdclef/models/yoink_wave_classifier.py` — `YoinkWaveClassifier` simplified port. Composes mel chain + features_only backbone + (deferred) spec_augment + head.
- [x] B.5.2 Unit tests: 10 tests in `test_yoink_wave_classifier.py` pass. Coverage: construction with defaults; extract_spec shape (2, 1, 128, 313); forward output dict; finite outputs; precomputed-spec path; short-circuit return_spec_feature / return_cnn_emb; spec_augment_config raises NotImplementedError (deferred); custom num_class; load pretrained backbone end-to-end forward.

### B.6 FocalBCELoss — PENDING (next Phase B sub-commit)

- [ ] B.6.1 `src/birdclef/losses/focal_bce.py` — direct port from `vendor/sydorskyi-2025/code_base/losses/focal_loss.py:28-54`. Composition: `1.0 * BCEWithLogitsLoss + 1.0 * torchvision.ops.focal_loss.sigmoid_focal_loss(α=0.25, γ=2.0)`. Reduction="mean".
- [ ] B.6.2 Unit tests: synthetic batch finite loss + gradient finite; matches torchvision impl when bce_weight=0; matches BCE-only when focal_weight=0.

### B.7 Class-balancing sampler — PENDING

- [ ] B.7.1 `src/birdclef/data/yoink_sampler.py` — `WeightedRandomSampler` factory with γ knob. NFNet path: γ=−0.5 ("sqrt"); EffNetV2-S path: γ=−1 (equivalent to "1/n_c"). Verified-config doc § Training hyperparameters.
- [ ] B.7.2 Unit test: empirical sample frequency over a fixed dataset matches expected within tolerance.

### B.8 Augmentation pipeline — PENDING

- [ ] B.8.1 `src/birdclef/preprocessing/yoink_spec_augment.py` — `CustomFreqMasking` + `CustomTimeMasking` ports (torchaudio FreqMasking/TimeMasking wrappers; per-sample stochastic apply with p, n_masks).
- [ ] B.8.2 `src/birdclef/preprocessing/random_filtering.py` — `RandomFiltering` port (4-band STFT-domain random EQ; min_db=-20; applied to waveform pre-mel).
- [ ] B.8.3 `src/birdclef/data/yoink_dataset.py` — `WaveDataset` port (focal + labeled-SS + PL with 40% Bernoulli swap; MixUp with sum-clip target; label smoothing GT-only; BG noise OneOf). Skip the unused features (unlabeled, montage, curation, OOF, deep_supervision).
- [ ] B.8.4 Unit tests: augmentation chain ends with finite tensor; MixUp sum-clip target rule verified; PL swap probability empirically ~0.4; label smoothing math correct.

### B.9 PyTorch Lightning module — PENDING

- [ ] B.9.1 `src/birdclef/lightning/yoink_finetune.py` — `YoinkFinetuneModule` Lightning wrapper. training_step calls YoinkWaveClassifier + RandomFiltering(wave) + FocalLossBCE on clipwise_logits_long. validation_step computes val ROC-AUC. Optimizer + scheduler from config.
- [ ] B.9.2 Fast-dev-run gate: launch on a tiny subset (no full ckpt load), verify no NaN/Inf, exit 0.

**Phase B exit criterion**: every component has a unit test that passes; fast-dev-run completes cleanly.

**Phase B sub-commit 1 (CURRENT) exit criterion**: ✅ B.1-B.5 done. Mel + head + backbone-loader + classifier orchestrator + 35 tests pass + backward-compat (exp 032/033's 195 tests still pass = 230 total). Ready for Phase C Gate 1 parity test.

### B.5 FocalBCELoss

- [ ] B.5.1 `src/birdclef/losses/focal_bce.py` — `FocalBCELoss = 1.0 * BCE + 1.0 * sigmoid_focal(α, γ)` with verified α/γ.
- [ ] B.5.2 Label smoothing α=0.05 mass-preserving applied to targets before loss.
- [ ] B.5.3 Unit test: synthetic batch, loss tensor finite, gradient finite, matches reference behavior at α=γ extremes.

### B.6 Class-balancing sampler

- [ ] B.6.1 Use `WeightedRandomSampler` with weights = `(class_freq / total)^γ`, γ from Phase A.
- [ ] B.6.2 Unit test: empirical sample frequency over a fixed dataset matches expected within tolerance.

### B.7 Augmentation pipeline

- [ ] B.7.1 Random 5s crop (training) + non-overlapping 5s windows (inference).
- [ ] B.7.2 MixUp(p=0.5) with `target = clip(t1 + t2, 0, 1)` (sum-clip, not max).
- [ ] B.7.3 BG noise (p=0.5, OneOf nocall-SS | ESC-50) with convex blend amp ∈ U[0.25, 0.75]. (defer ESC-50 sourcing if it adds friction).
- [ ] B.7.4 Peak-normalize after augmentation.
- [ ] B.7.5 RandomFiltering (4-band random EQ via STFT, U[−20, 0] dB per band) — may be optional for Phase E first LB; full reproducibility for Phase G full FT.
- [ ] B.7.6 SpecAugment freq L=10 n=3 p=0.3 + time L=20 n=3 p=0.3.

### B.8 PyTorch Lightning module

- [ ] B.8.1 `src/birdclef/lightning/sydorskyi_yoink_finetune.py` — extend exp 032's `DistilledSEDFinetuneModule` pattern, replacing the backbone + head + loss + sampler.
- [ ] B.8.2 `training_step`, `validation_step`, optimizer config (AdamW lr=1e-4 cosine→1e-6, 50 epochs).
- [ ] B.8.3 Fast-dev-run on a tiny subset; confirm no NaN/Inf and W&B/log integration works.

**Phase B exit criterion**: every component has a unit test that passes; fast-dev-run completes cleanly.

---

## Phase C — Gate 1: feature parity (mandatory, no LB)

- [ ] C.1 Pick 5 fixed BirdCLEF 2026 test audio files; document their paths.
- [ ] C.2 Run Sydorskyi's published Kaggle inference notebook on those 5 audios; dump backbone embeddings (or final logits if no embedding hook available).
- [ ] C.3 Run our Phase B pipeline on the same 5 audios (frozen backbone, our wrappers).
- [ ] C.4 Compute per-audio cosine similarity on embeddings.
- [ ] C.5 Compute mean L2 distance.
- [ ] C.6 Target: cosine ≥ **0.999** on all 5, mean L2 < 1e-4.

**Phase C exit criterion**: cosine ≥0.999 on every audio. NO LB submission until this passes.

**Phase C pivot (if cosine <0.99 after extended debug)**: abandon Path A → Path C fallback (skip directly to Phase I ensemble of existing models without yoink).

---

## Phase D — Gate 2: frozen-backbone head sanity (no LB)

- [ ] D.1 Freeze backbone (`for p in backbone.parameters(): p.requires_grad = False`).
- [ ] D.2 Random-init head.
- [ ] D.3 Train head only on fold 0, 5 epochs.
- [ ] D.4 Compute local OOF AUC on fold-0 holdout SS pool.
- [ ] D.5 Target: AUC ≥ **0.95**. (Sydorskyi reference: 0.984.)

**Phase D exit criterion**: local OOF AUC ≥ 0.95. If 0.93-0.95, marginal — investigate head + augmentation. If <0.93, return to Phase C parity audit.

---

## Phase E — Frozen-backbone fine-tune + first LB

### E.1 Training

- [ ] E.1.1 Config: `configs/yoink_effv2s_frozen.yaml` (to be created in Phase B).
- [ ] E.1.2 Train frozen-backbone fold 0 for 25-50 epochs (start 25; extend if not converged).
- [ ] E.1.3 Confirm `last.ckpt` written + W&B log uploaded + `metrics.json` saved.

### E.2 D.5 e2e validation

- [ ] E.2.1 Reuse exp 032's `validate_end_to_end.sh` adapted for the yoink config.
- [ ] E.2.2 Gate: `|train_auroc − e2e_auroc| ≤ 0.02`.

### E.3 Export + LB submit

- [ ] E.3.1 Extend `experiments/010-submission-pipeline/sbatch/export.sbatch` with `export_sydorskyi_yoink()` helper.
- [ ] E.3.2 Add MODELS entry in `scripts/upload_kaggle.py`.
- [ ] E.3.3 Create `kaggle_kernel/kernel-metadata-pretrained-backbone-yoink-effv2s-frozen.json` (verify title→slug derivation).
- [ ] E.3.4 Add weights slug to `kaggle_kernel/submission.py` MODEL_SLUGS.
- [ ] E.3.5 Submission notebook must use nnAudio mel inference (verify in kernel logs).
- [ ] E.3.6 ONNX export — may not be straightforward with nnAudio in the forward path; consider TorchScript fallback if ONNX trips on nnAudio ops.
- [ ] E.3.7 Submit to LB.

### E.4 Decision matrix

- [ ] E.4.1 Read LB; record in `results.md` (to be created).
  - LB ≥ 0.91 → proceed to Phase F.
  - LB 0.90-0.91 → re-audit Phase C parity.
  - LB < 0.90 → debug Phase B for silent bug; fall back to Phase C pivot if unresolvable.
  - LB > 0.92 → consider skipping Phase G full-FT, jumping to Phase I ensemble.

**Phase E exit criterion**: LB ≥ 0.91 OR active debug plan with cause hypothesis.

---

## Phase F — 5-fold frozen-backbone ensemble + LB

- [ ] F.1 Train remaining 4 folds frozen-backbone (parallel SLURM submissions; ~4-6h each on rtx6000).
- [ ] F.2 D.5 e2e validation on each fold.
- [ ] F.3 Implement 5-fold logit-space ensemble inference (mirror exp 032's `predict_soundscape_tucker_ensemble`).
- [ ] F.4 Export ONNX (per-fold) + Kaggle upload + push kernel + LB submit.
- [ ] F.5 Read LB.

**Phase F exit criterion**: 5-fold LB ≥ single-fold LB + 0.005 (typical fold-ensemble lift). LB ≥ 0.92 confirms on-track.

**Phase F pivot (if 5-fold < 0.92)**: try NFNet-L0 variant (parallel branch in Phase I) OR accept ceiling and skip to Phase H + Phase I.

---

## Phase G — Full fine-tune (unfreeze backbone)

- [ ] G.1 Config: `configs/yoink_effv2s_fullft.yaml` (lr=1e-4 cosine→1e-6, 50 epochs, AdamW).
- [ ] G.2 Train full FT EffNetV2-S fold 0.
- [ ] G.3 D.5 e2e validation.
- [ ] G.4 Export + LB submit.
- [ ] G.5 Paired comparison vs Phase E frozen-backbone fold-0.

**Phase G exit criterion**: full-FT LB > frozen-backbone LB. If yes, scale to 5 folds (separate sub-phase).

**Phase G pivot (if full-FT regresses)**: revert to frozen-backbone for final. Common cause: catastrophic forgetting of in-domain pretrain on small 2026 train set. Try lower LR (1e-5) before reverting.

---

## Phase H — Postprocessing + TTA

Run only after Phase F or G yields a single base LB ≥ 0.92 (Jack's warning: PP layers easily hurt on weaker bases).

- [ ] H.1 TopN N=1 postprocessing (Sydorskyi paper §5.6).
- [ ] H.2 Site/hour priors layer (reference: `ferariz/birdclef2026` `priors.py`).
- [ ] H.3 Gaussian smoothing in logit space (5-tap, matches exp 032 path).
- [ ] H.4 Sliding-window TTA (2.5-sec left/right shifts, Sydorskyi solo +0.012).
- [ ] H.5 Stepwise LB submissions: base only → +TopN → +priors → +smooth → +TTA. Attribute lift per layer.
- [ ] H.6 Drop any individual layer that regresses LB.

**Phase H exit criterion**: cumulative PP lift ≥ +0.005 LB.

---

## Phase I — Ensemble

Run anytime once a yoinked base ≥ 0.91 exists (Phase E onward).

- [ ] I.1 Rank-ensemble: yoink (Phase E/F/G best) ⊕ exp 032 distilled-sed-kd-5fold (0.915) ⊕ exp 023 Perch-MLP (0.929).
- [ ] I.2 Stepwise LB: yoink+032 → +023 → maybe +NFNet-L0 variant.
- [ ] I.3 Optional: NFNet-L0 yoink branch (different pretrain backbone for diversity) — gate on remaining compute budget.
- [ ] I.4 Optional: 2-seed identical-architecture ensemble (Nikita's stability lever).
- [ ] I.5 Drop any added member that doesn't lift.

**Phase I exit criterion**: ensemble LB > best single-model. Target ≥ 0.93; stretch ≥ 0.94.

**Path C fallback path** (if Phase C/E aborted): direct rank-ensemble exp 023 + exp 032 5-fold; submit. Realistic ceiling ~0.92.

---

## Phase J — Final submission selection

- [ ] J.1 Kaggle allows 2 final picks. Choose 1 high-LB-high-variance + 1 safer-lower-LB-lower-variance.
- [ ] J.2 Sanity inference on 100-soundscape local holdout.
- [ ] J.3 Verify submission kernel runs in <90 min CPU (Kaggle hidden-test inference budget).
- [ ] J.4 Submit by **2026-06-03 18:00 UTC** (safety margin against deadline 23:59).

**Phase J exit criterion**: both final picks tested end-to-end; inference time confirmed.

---

## Closeout

- [ ] C.1 Update `results.md` with per-phase LB outcomes + verdict.
- [ ] C.2 Update `user/murilo/MODEL-RUNS.md` with exp 034 section.
- [ ] C.3 Update `user/murilo/CONTEXT.md` for the next session.
- [ ] C.4 Open PR — squash-merge per `feedback_squash_merge.md`.
- [ ] C.5 If CONFIRM/PARTIAL: scope follow-up (own-teacher PL on yoinked base, OR additional ensemble members).
- [ ] C.6 If REJECT: document negative result; revert to Path C ensemble.

---

## Pivot-triggers — cross-phase reference

| Trigger | Pivot |
|---|---|
| Phase C cosine <0.99 after extended debug | Abandon Path A → Path C fallback (Phase I direct, no yoink) |
| Phase D OOF AUC <0.93 | Return to Phase C parity audit |
| Phase E LB <0.90 | Pipeline broken — debug Phase B; if still broken after extended debug, fall back to Path C |
| Phase F 5-fold <0.92 | Try NFNet-L0 variant OR accept ceiling + skip to Phase H+I |
| Phase G full-FT regresses vs Phase E | Try lower LR; if still regresses, revert to frozen for final |
| Phase H individual PP layer regresses | Drop that layer (Jack's warning) |
| Phase I added member doesn't lift | Drop that member |
| **Any phase: silent bug detected post-LB** | Re-run last 3 LB submissions' offline counterparts to check invalidation (per feedback memory + exp 032 fold-0 hardcode incident) |
