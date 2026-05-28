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

### A.1 Get the source

- [ ] A.1.1 Clone `git@github.com:VSydorskyy/BirdCLEF_2025_2nd_place.git` into `vendor/` (or wherever is `.gitignore`-safe).
- [ ] A.1.2 Download Kaggle dataset `vladimirsydor/bird-clef-2025-all-pretrained-models` to PACE shared storage: `/storage/project/ps-clef2026_birdclef-0/shared/pretrained-models/sydorskyi-2025/`.
- [ ] A.1.3 Verify checkpoint files present: `eca_nfnet_l0_*.pth`, `tf_efficientnetv2_s_in21k_*.pth` (or whatever the actual file naming is — confirm from repo README).
- [ ] A.1.4 `torch.load` each ckpt, print `list(ckpt.keys())[:20]` and `{k: v.shape for k, v in list(ckpt.items())[:10]}` — capture in `docs/sydorskyi-config-verified.md`.

### A.2 Read the source line-by-line

- [ ] A.2.1 `code_base/models.py` — confirm head structure (hidden dim, GeM, dropout positions, ReLU vs other activation, SED-attention impl).
- [ ] A.2.2 `code_base/dataset.py` — confirm mel pipeline (nnAudio call, standardize axes, min-max[0,1] axes, augmentation chain order).
- [ ] A.2.3 `code_base/losses.py` — confirm FocalBCELoss exact α, γ, reduction.
- [ ] A.2.4 `code_base/samplers.py` (or wherever sampler lives) — confirm class-balancing γ values per backbone.
- [ ] A.2.5 `create_pretrain_backbone.py` — confirm ckpt key format output (prefix, wrapping under `state_dict`).
- [ ] A.2.6 Configs (`configs/*.yaml` in the source repo) — confirm hparams per backbone variant.

### A.3 Write verified config doc

- [ ] A.3.1 Create `docs/sydorskyi-config-verified.md` with every config cell from `proposal.md § Setup` table updated to HIGH confidence with line-number citation.
- [ ] A.3.2 Specifically resolve:
  - Head hidden dim: **512 or other?**
  - GeM freq pool params (p, eps)
  - Standardize axes (per-spectrogram vs per-batch vs per-channel)
  - mel `normalized` flag handling (True vs False)
  - FocalBCELoss α, γ
  - Class-balancing γ for EffNetV2-S vs NFNet-L0
  - Ckpt key prefix + whether wrapped under `state_dict`

**Phase A exit criterion**: `docs/sydorskyi-config-verified.md` landed. No "MED" or "LOW" confidence cells remain in the setup table.

---

## Phase B — Implementation

### B.1 Dependencies

- [ ] B.1.1 Add `nnAudio` to `pyproject.toml` deps in `experiments/034-pretrained-backbone-yoink/pyproject.toml` (to be created).
- [ ] B.1.2 `uv pip install -e ".[dev]"` inside active interactive venv (NOT in cwd).
- [ ] B.1.3 Verify `import nnAudio; from nnAudio.Spectrogram import MelSpectrogram` works.

### B.2 nnAudio mel pipeline

- [ ] B.2.1 `src/birdclef/preprocessing/sydorskyi_mel.py` — wrap `nnAudio.Spectrogram.MelSpectrogram` with verified params (Phase A).
- [ ] B.2.2 Compose: nnAudio mel → `AmplitudeToDb(top_db=80, amin=1e-10)` → standardize (verified axes) → min-max[0,1].
- [ ] B.2.3 Unit test against captured reference tensor (small fixed audio, deterministic).

### B.3 SED-attention head

- [ ] B.3.1 `src/birdclef/heads/sydorskyi_sed_head.py` — backbone-features → GeM freq pool → SED-attention time pool → hidden(512) → dropout(0.5) → ReLU → classifier(num_classes).
- [ ] B.3.2 Initial dropout 0.25 after backbone, before pooling.
- [ ] B.3.3 Unit test forward pass on synthetic backbone features.
- [ ] B.3.4 Verify `head.classifier.out_features == num_classes_2026` after init.

### B.4 Checkpoint loader

- [ ] B.4.1 `scripts/load_sydorskyi_backbone.py` — `torch.load` ckpt → inspect keys → strip head if needed → `load_state_dict(strict=True)`.
- [ ] B.4.2 Assert: every loaded conv layer's `weight.abs().mean()` > 1e-3 (catches all-zeros silent load).
- [ ] B.4.3 Verify `backbone.num_features` matches expected (1280 for EffNetV2-S, 2304 for NFNet-L0).
- [ ] B.4.4 Unit test: load ckpt, run dummy forward pass, no NaN/Inf.

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
