# Tasks — Exp 034 Pretrained-backbone xcpretrain

## Tests

Tests will live in `user/murilo/tests/034-pretrained-backbone/` (mirroring the experiment dir).

Run with the interactive venv:

```bash
~/clef/birdclef-2026/.venv/bin/python -m pytest user/murilo/tests/034-pretrained-backbone/ -v
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
| **Inference path (exp 010)** | `experiments/010-submission-pipeline/sbatch/export.sbatch` — extend with new xcpretrain-export helper |

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

- [x] B.1.1 `experiments/034-pretrained-backbone/pyproject.toml` created. Pins inherit exp 033 ML stack + adds `torchvision==0.20.1` (needed for `torchvision.ops.focal_loss.sigmoid_focal_loss` in B.5). `nnAudio==0.3.4` already present.
- [x] B.1.2 No new install needed — active venv already has nnAudio 0.3.4, torchvision 0.25.0, timm 1.0.25, torch 2.10. Versions exceed pyproject pins but tests pass cleanly.
- [x] B.1.3 `import nnAudio; from nnAudio.Spectrogram import STFT as nnAudioSTFT` works (Sydorskyi uses STFT, not full MelSpectrogram — see verified-config doc § Mel pipeline).

### B.2 nnAudio mel pipeline — DONE 2026-05-28

- [x] B.2.1 `src/birdclef/preprocessing/xcpretrain_mel.py` — direct port of Sydorskyi's mel chain (`TraceableMelspec` = nnAudio STFT + torchaudio MelScale; `ChannelAgnosticAmplitudeToDB`; `NormalizeMelSpec`; `GeMFreq`; `Swish`).
- [x] B.2.2 `build_sydorskyi_mel_extractor` convenience function composes the full chain (TraceableMelspec → ChannelAgnosticAmplitudeToDB → NormalizeMelSpec) with Sydorskyi-default params.
- [x] B.2.3 Unit tests: 10 tests in `test_xcpretrain_mel.py` pass. Coverage: GeMFreq learnable p=3 init + exportable/non-exportable parity; NormalizeMelSpec per-sample standardize over (1,2) + output range [0,1]; ChannelAgnosticAmplitudeToDB 3D/4D acceptance + top_db=80 clip; TraceableMelspec output shape (2, 128, 313) for 5s @ 32kHz; full chain end-to-end finite + range.

### B.3 SED-attention head — DONE 2026-05-28

- [x] B.3.1 `src/birdclef/heads/xcpretrain_sed_head.py` — `XCPretrainAttHead` port. Backbone-features → GeMFreq pool → squeeze + permute → dense(in_chans → 512) + dropout(0.25) + ReLU + dropout(0.5) → 2 parallel Conv1d (attention + value) → soft-attention-weighted aggregation in BOTH logit-space (loss) and prob-space (metric).
- [x] B.3.2 Default `p=0.5` master → dropout(p/2=0.25) before linear, dropout(p=0.5) after ReLU. ReLU activation.
- [x] B.3.3 Default `num_class=234` (BirdCLEF 2026). `train_period == infer_period` asserted (simplified vs original Sydorskyi which supports short-window crop path at inference).
- [x] B.3.4 Unit tests: 9 tests in `test_xcpretrain_sed_head.py` pass. Coverage: constructor defaults match Sydorskyi (hidden 512); dropout positions/rates verified; ReLU activation verified; output dict has 3 keys with correct shapes; clipwise_pred in [0,1] (prob-space); clipwise_logits unbounded (logit-space); simplification assertion fires when train_period != infer_period; gradient flows to GeMFreq.p, attention.weight, fix_scale.weight.

### B.4 Checkpoint loader — DONE 2026-05-28

- [x] B.4.1 `experiments/034-pretrained-backbone/scripts/load_sydorskyi_backbone.py` — instantiates `timm.create_model(features_only=True, in_chans=1)` and loads Sydorskyi ckpt with `strict=True` (after stripping defensive `model.backbone.` / `backbone.` prefixes). CLI + Python API.
- [x] B.4.2 Both backbones (EffNetV2-S 2024 + NFNet-L0 2025) load with strict=True; no missing/unexpected keys. Conv weight norms confirmed non-zero post-load.
- [x] B.4.3 `backbone.feature_info.channels()[-1]` confirmed: **256 for EffNetV2-S** (features_only strips conv_head/bn2 — matches stripped ckpt) and **2304 for NFNet-L0**.
- [x] B.4.4 Unit tests: 6 tests in `test_xcpretrain_backbone_loader.py` pass. Coverage: strict=True load for both backbones; output channels verification; conv weights non-zero; forward pass produces finite (B, 256/2304, H, W); unknown-backbone raises ValueError.

### B.5 WaveCNNAttenClasifier orchestrator — DONE 2026-05-28 (added beyond original Phase B scope)

- [x] B.5.1 `src/birdclef/models/xcpretrain_wave_classifier.py` — `XCPretrainWaveClassifier` simplified port. Composes mel chain + features_only backbone + (deferred) spec_augment + head.
- [x] B.5.2 Unit tests: 10 tests in `test_xcpretrain_wave_classifier.py` pass. Coverage: construction with defaults; extract_spec shape (2, 1, 128, 313); forward output dict; finite outputs; precomputed-spec path; short-circuit return_spec_feature / return_cnn_emb; spec_augment_config raises NotImplementedError (deferred); custom num_class; load pretrained backbone end-to-end forward.

### B.6 FocalLossBCE — DONE 2026-05-28 ✅

- [x] B.6.1 `src/birdclef/losses/focal_bce.py` — verbatim port of `focal_loss.py:28-54`. `1.0*BCEWithLogitsLoss + 1.0*torchvision.ops.focal_loss.sigmoid_focal_loss(α=0.25, γ=2.0)`, reduction="mean", applied to `clipwise_logits_long`.
- [x] B.6.2 `test_focal_bce.py` — 5 tests: finite loss+grad; == torchvision focal when bce_weight=0; == BCEWithLogits when focal_weight=0; default params; weighted-sum composition.

### B.7 Class-balancing sampler — DONE 2026-05-28 ✅

- [x] B.7.1 `src/birdclef/data/xcpretrain_sampler.py` — `compute_class_weights` + `build_xcpretrain_sampler`. mode "balanced"→1/n_c (EffNetV2-S), "sqrt"→(n_c/N)^-0.5 (NFNet). Mirrors `train_lightning.py:240-263`.
- [x] B.7.2 `test_xcpretrain_sampler.py` — 6 tests incl. empirical equalization of a skewed 90/10 distribution. (Standalone utility — exp 032 datamodule uses its own MixSamp by default; available to swap in for Sydorskyi-faithful sampling.)

### B.8 Augmentation ports — DONE 2026-05-28 ✅ (B.8.1/B.8.2); B.8.3 = REUSE (see decision)

- [x] B.8.1 `src/birdclef/preprocessing/xcpretrain_spec_augment.py` — `CustomMasking`/`CustomFreqMasking`/`CustomTimeMasking` verbatim port. Wired into `XCPretrainWaveClassifier` (replaces the prior NotImplementedError; applied in forward when training).
- [x] B.8.2 `src/birdclef/preprocessing/random_filtering.py` — `RandomFiltering` verbatim port (4-band STFT EQ, min_db=-20, peak-normalize). Applied to the waveform in the module's training_step (Sydorskyi `batch_aug`).
- [x] B.8.3 **DECISION — reuse exp 032's `DistilledSEDDataModule` instead of porting Sydorskyi's `WaveDataset` from scratch.** An Explore-agent audit confirmed the datamodule is model-agnostic: it yields raw-waveform `(wavs (B,1,T), labels (B,C), wts, mks)` batches and already owns the 2026 5-fold split (seed=42), data paths, rare-species upsampling, and PL hooks. Porting a stateful WaveDataset would re-introduce exactly the silent-bug surface Gate 1 just eliminated, and can't be data-faithful anyway (different competition). **Rationale + 3-option tradeoff + fidelity-escalation path recorded in `results.md` § B.8.3.** Sydorskyi-specific data dynamics NOT yet matched (MixUp sum-clip, 40% PL swap, BG-noise OneOf, class-balanced sampler) are LOW-risk per the research and can be swapped in incrementally and A/B'd at Phase G / PL round. Empirical arbiter = Gate 2 OOF AUC + Phase E LB.
- [x] B.8.4 `test_xcpretrain_augment.py` — 6 tests: masking shape/energy-non-increase/p=0 noop/p=1 changes/inplace; RandomFiltering wave roundtrip finite + peak-capped + spec-path gain.

### B.9 PyTorch Lightning module — DONE 2026-05-28 ✅

- [x] B.9.1 `src/birdclef/lightning/xcpretrain_finetune.py` — `XCPretrainFinetuneModule`. Consumes exp 032 datamodule batches → squeeze → (train) RandomFiltering → XCPretrainWaveClassifier → FocalLossBCE on `clipwise_logits_long`; mass-preserving label smoothing (0.05); optional frozen-backbone; AdamW lr=1e-4 cos→1e-6 per-step; val AUROC via TuckerMacroAUC (matches exp 032 e2e gate). `test_xcpretrain_finetune.py` — 6 smoke tests.
- [x] B.9.2 Fast-dev-run gate: `experiments/034-pretrained-backbone/train.py` (reuses exp 032 data wiring). `--fast-dev-run` ran 1 train + 1 val batch on real 2026 data — finite losses (train 1.023, val 0.821), AUROC computed, exit 0, no NaN/Inf.

**Phase B exit criterion**: ✅ every component unit-tested (58 exp034 tests pass); fast-dev-run completes cleanly; 195 backward-compat tests still pass (253 total green).

---

## Phase C — Gate 1: feature parity (mandatory, no LB) — DONE 2026-05-28 ✅ PASS

**Approach change (vs original plan)**: the proposal specced parity against
Sydorskyi's *published Kaggle inference notebook*. Since we now have his full
source vendored as a git submodule (`vendor/sydorskyi-2025/`), we ran the parity
test LOCALLY against his *own* `WaveCNNAttenClasifier` — a stronger, faster,
fully-reproducible oracle for "is our PORT faithful?", with no Kaggle round-trip.
Both classifiers expose identical `return_spec_feature` / `return_cnn_emb` taps;
both backbones loaded from the *same* cleaned pretrained `state_dict`, so any
Stage-2 divergence would be attributable purely to the mel front-end.
(Library version-skew vs Sydorskyi's original training env is NOT tested here —
that lower-risk concern is what Gate 2 frozen-backbone OOF AUC covers.)

Script: `scripts/gate1_parity.py`. Results JSON: `outputs/gate1_parity{,_nfnet}.json`.

- [x] C.1 Picked first 5 sorted `train_soundscapes` `.ogg` (deterministic): `BC2026_Train_0001..0005_S08`. Same waveform tensor fed to both classifiers (audio backend not part of the comparison).
- [x] C.2/C.3 Ran our `XCPretrainWaveClassifier` AND vendored `WaveCNNAttenClasifier` on the same 5 audios, frozen, eval mode.
- [x] C.4/C.5 Computed per-audio cosine + RMS-L2 + max-abs-diff at BOTH the mel stage and the backbone-embedding stage.
- [x] C.6 Result: **bit-identical** — cosine = 1.00000000, max-abs-diff = 0.0, RMS-L2 = 0.0 on all 5 audios, BOTH stages, BOTH backbones (EffNetV2-S 256-ch + NFNet-L0 2304-ch). Far exceeds the cosine ≥0.999 / L2 <1e-4 bar.

**Phase C exit criterion**: ✅ cosine ≥0.999 on every audio — PASS (bit-identical). Clears the highest-probability catastrophic-bug class (nnAudio mel mismatch, standardize/min-max order/axes, head dim, ckpt keys). NO LB burned. Ready for Phase D Gate 2.

**Phase C pivot (if cosine <0.99 after extended debug)**: N/A — passed cleanly.

---

## Phase D — Gate 2: frozen-backbone head sanity (no LB)

**Setup DONE 2026-05-28** (ready to launch; Murilo runs the SLURM job):
- [x] D.1 Freeze backbone — `--freeze-backbone` (verified: 371K trainable / 19.8M frozen in pre-flight).
- [x] D.2 Random-init head — head is random by default; `--load-pretrained` loads only the backbone.
- [x] Training-loop audit vs exp 021 / 032 / 033 / Sydorskyi — no bugs; Sydorskyi-faithful. See `results.md § Training-loop audit`.
- [x] `train.py` persists OOF `val/auroc` to `outputs/<run>-fold<k>.json`. Submission uses the exp 032/033 **launcher + `_job.sh`** pattern: `bash sbatch/gate2.sh` (plain bash launcher) → `sbatch sbatch/_job.sh` ($TMPDIR venv, GPU, `$SLURM_SUBMIT_DIR`). (Earlier self-submitting single-file was replaced — it broke under interactive allocations + sbatch script-spooling.)
- [x] Local CPU pre-flight (`--fast-dev-run --load-pretrained --freeze-backbone`) PASS — frozen+pretrained path runs clean.
- [x] D.3 Trained head-only on fold 0, 5 epochs (job 9277013, config-driven).
- [x] D.4 OOF `val/auroc` = **0.791** (`outputs/test_results/gate2-effv2s-fold0.json`; rose 0.541→0.741→0.791).
- [x] D.5 ⚠️ **0.95 gate was mis-set** (cited Sydorskyi full-FT 0.984; but our best full model exp 032 tops out at **0.908** on this same 2026 SS val set, and Gate 2 is 5-ep frozen-head). Corrected read: frozen features clearly informative (0.79 vs 0.91 ceiling vs 0.5 chance) → pipeline + transfer VALIDATED. Full analysis in `results.md § Phase D — Gate 2 result`.

**Launch command** (per `feedback_user_runs_commands` — Murilo runs SLURM jobs):
```bash
cd user/murilo/experiments/034-pretrained-backbone
bash sbatch/gate2.sh                 # configs/gate2-effv2s.yaml, fold 0, 5 ep (self-submits)
# monitor:                           squeue -u $USER ; tail -f logs/slurm_gate2-effv2s-fold0_*.log
# result:                            cat outputs/test_results/gate2-effv2s-fold0.json
```

**Phase D exit (corrected)**: Gate 2 is a pipeline/transfer sanity, NOT an absolute-AUC
bar — frozen-head clearly above chance + a sensible fraction of the full-model ceiling
(0.91) + sane artifacts → PASS. The 0.95 number is retired. Offline AUC anti-correlates
with LB; **Phase E LB is the real arbiter**. Frozen-head caps below the full-model ceiling
(0.79 vs 0.91) → closing the gap likely needs full fine-tune (Phase G unfreeze).

---

## Phase E — Frozen-backbone fine-tune + first LB

**PRE-STAGED 2026-05-28** (ready to launch the moment Gate 2 clears).

### Epoch decision (25 vs 50) — RESOLVED

`train.py` default is 50 (Sydorskyi `selected_ebs.py` + `feedback_bump_epochs`), but
that is his **full-fine-tune** number. **Phase E is frozen-backbone (head-only, 371K
params on fixed features)** → converges fast, and the cosine LR must anneal over the
*actual* run length. So:
- **Phase E frozen primary = 25 epochs** (our proven exp 032/033 cadence → LB 0.906/0.915; proper annealing; ½ compute).
- **50 epochs = cheap parallel ablation** (`MAX_EPOCHS=50 RUN_SUFFIX=-50ep bash sbatch/train.sh`) — let LB arbitrate 25 vs 50 (`feedback_pace_compute_not_bottleneck`).
- **Phase G full-FT = 50** (Sydorskyi-faithful, whole network adapts; `MODE=fullft`).

### E.1 Training — `sbatch/train.sh` (self-submitting; `--save-ckpt` save_top_k=3 + save_last → use `last.ckpt`)

```bash
cd user/murilo/experiments/034-pretrained-backbone
bash sbatch/train.sh                              # Phase E: frozen, fold 0, 25 ep
MAX_EPOCHS=50 RUN_SUFFIX=-50ep bash sbatch/train.sh   # 25-vs-50 LB ablation
for f in 0 1 2 3 4; do FOLD=$f bash sbatch/train.sh; done   # Phase F: 5-fold
# result: cat outputs/xcpretrain-tf_efficientnetv2_s_in21k-frozen-fold0.json
# ckpts:  /storage/project/.../experiments/034-pretrained-backbone/checkpoints/<run>-fold0/last.ckpt
```
- [x] E.1.1 Frozen fold-0, 25 ep — DONE (SLURM 9279181, W&B `qfm8`): **val/auroc 0.859**, `last.ckpt` written.
- [x] E.1.2 Full-FT fold-0, 50 ep (Phase G, run in parallel) — DONE (SLURM 9279209, W&B `gvx4`): **val/auroc 0.883**, `last.ckpt` written.
- [x] E.1.3 Both confirmed: `outputs/test_results/<run>-fold0.json` + `artifacts/training/<run>/fold0/metrics.json` written; clean tqdm logs. Trajectory: frozen 0.791(5ep)→0.859(25ep); full-FT 0.883 > frozen. (See `results.md § Phase E/G`.) **Decision point: LB submit both — offline can't tell us if the yoink lever beats 0.906.**

### E.2 e2e validation

- [ ] E.2.1 Adapt exp 032's `validate_end_to_end.sh` for the xcpretrain forward path (nnAudio mel + clip-only `clipwise_pred_long`).
- [ ] E.2.2 Gate: `|train_auroc − e2e_auroc| ≤ 0.02`.

### E.3 Export + LB submit — **build once a trained ckpt exists** (depends on E.1)

⚠️ **Watch-items (from training-loop audit):**
- **Inference scoring must match training** = clip-only `clipwise_pred_long` (NOT the exp 021/032 σ(clip)+σ(frame) blend) — or deliberately A/B a blend. Avoid silent train/infer mismatch.
- **ONNX + nnAudio**: nnAudio STFT ops may not export to ONNX cleanly. `TraceableMelspec` is built `exportable=True`, but plan a **TorchScript fallback** if ONNX trips (E.3.6). Validate the exported mel vs the eager mel before trusting the kernel.

- [ ] E.3.1 Extend exp 010 export with an `export_xcpretrain()` helper (ONNX → TorchScript fallback).
- [ ] E.3.2 Add MODELS entry in `scripts/upload_kaggle.py`.
- [ ] E.3.3 `kaggle_kernel/kernel-metadata-...-xcpretrain-effv2s-frozen.json` (verify title→slug).
- [ ] E.3.4 Add weights slug to `kaggle_kernel/submission.py` MODEL_SLUGS.
- [ ] E.3.5 Submission kernel uses nnAudio mel inference (verify in kernel logs).
- [ ] E.3.6 ONNX export (TorchScript fallback if nnAudio ops fail).
- [ ] E.3.7 Submit to LB.

### E.4 Decision matrix — RESULT (2026-05-28)

- [x] E.4.1 LB landed (after fixing the 41/234 species_mapping bug): **frozen 0.819, full-FT 0.865** — both below the 0.906 base (`results.md § LB results`).
  - Per the matrix this is the "< 0.90" bucket, BUT the cause is **NOT a silent bug** (Gate 1 bit-identical + kernel parity 9e-7 + canary 234-mapped). The cause is **configuration**: exp 034 is a BARE single-fold model with NO KD / PP / PL / ensemble. full-FT 0.865 ≈ exp 032's no-KD control (0.856) → the in-domain backbone is a small real lever, not a broken one.
  - **Decision: don't debug; STACK.** The gap to 0.906/0.929 is the missing levers, by LB-evidenced value: KD +0.050, PL +0.027, PP +0.012–0.020, 5-fold +0.009, ensemble. See `results.md § Gap to our best + path forward`.

**Phase E exit criterion**: ⚠️ LB < 0.91 — but pipeline validated; path = add the stack (PP cheapest → KD biggest → ensemble), not debug.

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

- [ ] G.1 Config: `configs/xcpretrain_effv2s_fullft.yaml` (lr=1e-4 cosine→1e-6, 50 epochs, AdamW).
- [ ] G.2 Train full FT EffNetV2-S fold 0.
- [ ] G.3 D.5 e2e validation.
- [ ] G.4 Export + LB submit.
- [ ] G.5 Paired comparison vs Phase E frozen-backbone fold-0.

**Phase G exit criterion**: full-FT LB > frozen-backbone LB. If yes, scale to 5 folds (separate sub-phase).

**Phase G pivot (if full-FT regresses)**: revert to frozen-backbone for final. Common cause: catastrophic forgetting of in-domain pretrain on small 2026 train set. Try lower LR (1e-5) before reverting.

---

## Phase K — Perch v2 KD on the in-domain backbone (post-LB pivot, 2026-05-29)

**Why**: KD is our biggest LB-evidenced lever — exp 032 control 0.856 → distilled-sed-kd 0.906 = **+0.050** — but that delta was measured on an ImageNet b0 backbone. This phase stacks the SAME mechanism (Perch v2 embedding distillation) onto Sydorskyi's in-domain backbone (which already scored full-FT 0.865), to test whether KD still pays when the backbone is already strong.

**Mechanism** (ported from `DistilledSEDFinetuneModule`, lives in `XCPretrainFinetuneModule._step_kd`): `DistillHead` (GAP→Linear, 256→1536) MSE'd against the frozen Perch v2 ONNX teacher's clip embedding; total = FocalLossBCE(cls) + `alpha_distill`·MSE. Teacher embeds the same post-RandomFiltering 5s waveform the student does. DistillHead is discarded at inference → **export/kernel are unchanged** (export takes `model.*` only).

- [x] K.1 Add KD branch to `XCPretrainFinetuneModule` (config-gated, backward-compatible); `onnxruntime-gpu==1.26.0` dep; wire Perch path in `train.py`. **5 unit tests + real-teacher integration check PASS** (DistillHead 256→1536 == Perch 1536, MSE-compatible).
- [x] K.2 Configs: `xcpretrain-effv2s-kd.yaml` (detach=True, Tucker-faithful) + `xcpretrain-effv2s-kd-joint.yaml` (detach=False, joint backbone grad). Both inherit fullft (trainable backbone, 50 ep).
- [ ] K.3 Train the 2-cell grid (detach vs joint), fold 0, in parallel:
  - `CONFIG=configs/xcpretrain-effv2s-kd.yaml bash sbatch/train.sh`
  - `CONFIG=configs/xcpretrain-effv2s-kd-joint.yaml bash sbatch/train.sh`
- [ ] K.4 **Epoch-1 watch**: `train/cls_loss` vs `train/dist_loss` (W&B). FocalBCE ≪ exp 032's plain BCE, so `alpha_distill=1.0` may over-weight the MSE — if `dist_loss` dominates, kill + lower alpha.
- [ ] K.5 Export (same `export_xcpretrain.py` — KD changes weights, not graph) + distinct kernel + LB submit each.

**Phase K exit criterion**: KD LB > full-FT 0.865 (the no-KD base). Stretch: approach the +0.050 paired delta → ~0.91.
**Phase K pivot**: if detach regresses (likely — forcing a strong backbone to mimic Perch can lose task-specific signal) but joint helps, keep joint. If neither helps, KD doesn't transfer to an in-domain backbone → drop, focus on 5-fold + ensemble.

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

Run anytime once a xcpretrained base ≥ 0.91 exists (Phase E onward).

- [ ] I.1 Rank-ensemble: xcpretrain (Phase E/F/G best) ⊕ exp 032 distilled-sed-kd-5fold (0.915) ⊕ exp 023 Perch-MLP (0.929).
- [ ] I.2 Stepwise LB: xcpretrain+032 → +023 → maybe +NFNet-L0 variant.
- [ ] I.3 Optional: NFNet-L0 xcpretrain branch (different pretrain backbone for diversity) — gate on remaining compute budget.
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
- [ ] C.5 If CONFIRM/PARTIAL: scope follow-up (own-teacher PL on xcpretrained base, OR additional ensemble members).
- [ ] C.6 If REJECT: document negative result; revert to Path C ensemble.

---

## Pivot-triggers — cross-phase reference

| Trigger | Pivot |
|---|---|
| Phase C cosine <0.99 after extended debug | Abandon Path A → Path C fallback (Phase I direct, no xcpretrain) |
| Phase D OOF AUC <0.93 | Return to Phase C parity audit |
| Phase E LB <0.90 | Pipeline broken — debug Phase B; if still broken after extended debug, fall back to Path C |
| Phase F 5-fold <0.92 | Try NFNet-L0 variant OR accept ceiling + skip to Phase H+I |
| Phase G full-FT regresses vs Phase E | Try lower LR; if still regresses, revert to frozen for final |
| Phase H individual PP layer regresses | Drop that layer (Jack's warning) |
| Phase I added member doesn't lift | Drop that member |
| **Any phase: silent bug detected post-LB** | Re-run last 3 LB submissions' offline counterparts to check invalidation (per feedback memory + exp 032 fold-0 hardcode incident) |
