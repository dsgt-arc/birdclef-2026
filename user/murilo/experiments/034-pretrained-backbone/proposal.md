---
experiment: "034-pretrained-backbone"
status: draft
date: 2026-05-28
author: Murilo Gustineli
---

# Proposal: Reuse a published in-domain-pretrained backbone (Sydorskyi 2025) on BirdCLEF 2026

**Status**: draft (2026-05-28)
**Created**: 2026-05-28
**Author**: Murilo Gustineli
**Branch (planned)**: `mg/exp034-pretrained-backbone`

## Provenance

This experiment reuses the **publicly-released backbone weights from the 2nd place team in BirdCLEF+ 2025** (Sydorskyi & Gonçalves), pretrained on an 819,032-recording / 7,489-species Xeno-Canto corpus, as a starting point for fine-tuning on the BirdCLEF 2026 task. The pretrained backbone is downloaded from a published Kaggle dataset; we attach our own head and fine-tune on 2026 data.

The lever: replace our current ImageNet-only-pretrained EfficientNet-B0 NS-JFT (LB 0.906) with an in-domain-pretrained EfficientNetV2-S (or eca_NFNet-L0) that has already seen ~10× more bird audio than the 2026 training set. Sydorskyi's own ablation (CEUR Vol-4038 paper_256, Table 5) attributes **+0.018 Private LB** to in-domain pretraining alone, isolated from the PL and PP layers.

Five independent disclosed precedents on this lever:

- **Sydorskyi & Gonçalves (2nd, BirdCLEF+ 2025)** — published 819K-recording / 7,489-species pretrained EfficientNetV2-S + NFNet-L0 checkpoints. Final Private 0.928 / Public 0.925.
- **Ali Ozan Memetoglu (7th, BirdCLEF 2026 in-comp)** — discloses LB **0.941** via Sydorskyi backbone + own head + fine-tune on 2026.
- **Jack (18th, BirdCLEF 2026 in-comp)** — also reports use of Sydorskyi's backbone; warns about mel/head mismatch costing **−0.012 LB** if not exact.
- **Antoine Masq (33rd, LB 0.943)** — uses own XC pretrain (not Sydorskyi's), but confirms in-domain pretraining as the lever ("XC pretrain + own-teacher PL").
- **Nikita Babych (1st, BirdCLEF+ 2025 + currently 1st BirdCLEF 2026)** — multi-iterative noisy-student PL on top of strong base; confirms pretraining as the base-quality lever.

The exp 033 REJECT verdict (3/3 Day-1 cells regressed, Phase 5 v3 filters HURT at LB 0.879) plus the deep research re-read of Sydorskyi's Table 5 (PL is a multiplier on a strong base, not a substitute for one) plus 4 independent confirmations from Anil Ozturk / Boredom / Zejun_ / Salman that PL plateaus at one pass on solo LB convergently point at base quality as the gap between 0.906 and 0.93+. This experiment buys the lever directly.

Cited materials:

| Doc | Content |
|---|---|
| `user/murilo/notes/2026-05-28-research-report.md` | Independent deep research review — realistic landing 0.92-0.93 (mode 0.925); enumerated 10 implementation risks; mandatory pre-flight gates |
| `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` | Vendored Sydorskyi writeup |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Top-team disclosures incl. Ali Ozan + Jack + Nikita + Boredom updates |
| `user/murilo/notes/kaggle-discussion-non-perch-limit.md` | Companion thread — Antoine 0.936→0.943, Arunodhayan 0.946 |
| `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` | Audit that uncovered our 16/20 divergences; informs which protocol pieces matter |
| `experiments/033-own-teacher-pl/results.md` | exp 033 REJECT + Phase 5 v3 filters HURT (0.879 LB) |
| `experiments/032-tucker-distilled-sed/results.md` | Best Tucker-SED stack so far (single 0.906, 5-fold 0.915); the base we layer on top of (or ensemble with) |
| Source repo | `github.com/VSydorskyy/BirdCLEF_2025_2nd_place` |
| Pretrained ckpts | Kaggle dataset `vladimirsydor/bird-clef-2025-all-pretrained-models` |
| Reference inference notebook | `kaggle.com/code/vladimirsydor/bird-clef-2025-minimul-inference` (scriptVersionId 245080754) — used for Gate 1 parity test |
| Paper | Sydorskyi & Gonçalves 2025, CEUR Vol-4038 paper_256 — ablation table 5, config table 11 |

## Hypothesis

**Reusing Sydorskyi's published in-domain-pretrained backbone (EffNetV2-S) + attaching an exact-parity SED head + fine-tuning on BirdCLEF 2026 single-fold lifts LB from 0.906 to 0.91-0.93.** The +0.018 Private gain Sydorskyi attributes to in-domain pretraining alone (Table 5) is the disclosed transferable lever; the ImageNet-only EfficientNet-B0 NS-JFT base we currently use lacks this prior.

Mechanism: our exp 032 distilled-sed-kd recipe (LB 0.906) is recipe-saturated — adding more recipe tricks (KD, mixup, PCEN) doesn't lift. The base's prior is the missing ingredient. Sydorskyi's pretraining corpus is ~10× larger than what we can pretrain in 7 days; downloading the released weights is the only realistic path to a 0.93+ base within the remaining deadline.

The 2nd-order benefit: a 0.92+ single-model base unlocks **two further levers** that are currently below the noise floor of our 0.906 base — full PP package (Nikita: +0.020 from 0.935 → 0.955) and PL iteration done correctly (Anil Ozturk +0.009 from 0.931 → 0.940). Both stack on top, not in place of, the base improvement.

### Falsifiable success matrix

Read against the paired **same-codebase baseline** (`distilled-sed-kd` Phase D fold 0 = LB 0.906, reproducible at HEAD). Per the deep research probability bands (Sydorskyi-xcpretrain mode 0.925, conditional on operator track record):

| Bucket | Single-fold LB (frozen-backbone) | Reading | Active path |
|---|---|---|---|
| **CONFIRM** | ≥ 0.93 | XCPretrain transferred cleanly; in-domain pretraining is the lever. | Scale to 5-fold + full fine-tune + PP + ensemble for 0.94+ |
| **PARTIAL** | 0.91-0.93 | XCPretrain paying out at expected mode. PP + ensemble + maybe full FT close the 0.94 gap. | Continue scaling; layer PP early (Phase H) |
| **WEAK** | 0.906-0.91 | Pretrain barely transfers. Likely a silent mel/head/ckpt parity bug. | Re-run Gate 1 + Gate 2; debug parity before more LB |
| **REJECT** | < 0.906 | Pipeline broken OR pretrain doesn't transfer to 2026's class distribution. | Pivot to Path C fallback (rank-ensemble exp 023 0.929 + exp 032 5-fold 0.915) |

Per the research's 80-85% aggregate-bug probability on first-attempt implementations of this complexity, **WEAK is the most likely outcome on the first LB submission**. The plan is structured to detect and recover from WEAK without burning LB slots in panic-debug loops.

## Background

### Why this lever, why now

Exp 032 confirmed our recipe is healthy (distilled-sed-kd 0.906 single, 0.915 5-fold, +0.020 KD lift). Exp 033 confirmed the recipe is stuck — PL doesn't transfer at LB on this base (3/3 Day-1 cells REJECT, Phase 5 v3 filter diagnostic LB 0.879 worse than v2). The deep research re-read of Sydorskyi's published ablation table shows the +0.029 PL Public delta lands on top of a base that already has in-domain pretraining; on an ImageNet-only base, the same protocol caps around 0.92. Salman Ahmed (43rd, LB 0.939 solo) reports PL adds only +0.005 on his stronger base — typical PL solo lift is now median +0.005, not the upper-band +0.012-0.013 of Antoine/Boredom/chenwenqiang.

What's left as a real lever within 7 days: the base.

### Why this specific implementation (frozen-backbone first, then unfreeze)

The deep research argues — and we accept — that full-finetuning the unfreezed Sydorskyi backbone from the first LB submission carries 80-85% silent-bug risk. The pre-flight is therefore:

1. **Phase A**: read the actual source code (not just the paper) and write a verified config table. The paper's Appendix B leaves several parameters (standardize axes, focal α/γ, exact dropout positions) unspecified.
2. **Phase B**: implement the head + mel + ckpt-loader.
3. **Phase C — Gate 1**: feature-parity test against Sydorskyi's published inference notebook on 5 fixed audios. Cosine ≥0.999 on backbone embeddings = parity confirmed.
4. **Phase D — Gate 2**: frozen-backbone head sanity check. Local OOF AUC ≥ 0.95 on fold 0 head-only train.

ONLY after Gates 1+2 pass do we burn the first LB slot (Phase E). This decouples "did the pipeline work?" from "did fine-tuning work?" — a missing decoupling that cost exp 033 multiple LB slots on REJECT cells that were never going to lift.

### What we already know that other teams don't (asymmetric advantages)

| Asset | Use |
|---|---|
| exp 023 Perch-MLP ensemble at LB 0.929 | Available as ensemble member in Phase I |
| exp 032 distilled-sed-kd-5fold at LB 0.915 | Same |
| Phase D fold-0 distilled-sed-kd at LB 0.906 (reproducible, bit-deterministic) | Paired-baseline anchor |
| Tucker-faithful mel + 5-tap Gaussian smooth + logit-space ensemble inference path | Reusable inference scaffold from exp 010-submission-pipeline |
| Validated 5-fold split (seed=42) from exp 032 | Reusable splits; consistent paired-LB methodology |
| `validate_end_to_end.sh` D.5 pipeline (with `_common.yaml` deep-merge patch) | Reusable for offline e2e validation gate |

### What we DON'T have (forced workarounds)

- **No XC pretrain compute budget**: Sydorskyi's pretrain took weeks on 819K recordings; we can't redo it. We must xcpretrain.
- **No nnAudio installed**: Sydorskyi uses `nnAudio` for on-the-fly GPU mel (not torchaudio). 60% silent-bug risk if we translate to torchaudio. **Decision**: install `nnAudio` and use directly. ~5 lines in pyproject.toml.
- **No 2-seed identical-architecture ensemble baseline** (Nikita's stability lever): out of scope for first pass; can be added in Phase I if compute allows.

## Method

### Approach

Phase A → J. Phases are **sequential by dependency**, not by calendar. Two phases can run in parallel where independent (e.g., reading code while downloading the pretrained dataset). LB submissions are budgeted per phase, not per day, and only fire after the in-phase gate passes.

### Setup (verified config — to be confirmed in Phase A)

| Component | Value | Confidence |
|---|---|---|
| Backbone | `tf_efficientnetv2_s_in21k` (primary) + `eca_nfnet_l0` (Phase I diversity) | HIGH (paper Appendix B, confirmed by GitHub README) |
| Pretrained weights | `vladimirsydor/bird-clef-2025-all-pretrained-models` (Kaggle dataset) | HIGH |
| Backbone output channels | EffNetV2-S = 1280, NFNet-L0 = **2304** (NOT 1024) | HIGH (timm, research-verified) |
| Head | SED-attention head, **512 hidden** (NOT 1028 — that was a typo in earlier planning) | MED — confirm in Phase A from `code_base/models.py` |
| Head pooling | GeM frequency pool + SED-attention time pool (no RNN) | MED — confirm in Phase A |
| Head dropout | 0.25 after backbone, 0.5 after hidden, ReLU activation | MED — confirm in Phase A |
| Mel library | **nnAudio** (NOT torchaudio); `normalized=True` flag | HIGH — research-flagged 60% bug risk if mistranslated |
| Mel params | sr=32000, n_mels=128, n_fft=2048, hop=512, f_min=20 | HIGH (paper Table 11) |
| Mel post-norm | `AmplitudeToDb(top_db=80, amin=1e-10)` → standardize → **min-max to [0, 1]** | MED — standardize axes unspecified in paper; read source |
| Loss | `FocalBCELoss` = 1.0·BCE + 1.0·sigmoid_focal(α=0.25, γ=2.0 default; **confirm in Phase A**) | MED — α/γ not in paper, read `code_base/losses.py` |
| Label smoothing | α = 0.05, mass-preserving | MED (paper §5.2; confirm in source) |
| Class balancing | `WeightedRandomSampler` γ=−0.5 (sqrt) for EffNetV2-S; γ=−1 (equal) for NFNet-L0 | MED |
| Augmentation | Random 5s crop → MixUp(p=0.5) → BG noise (p=0.5, OneOf nocall-SS or ESC-50) → peak-normalize → mel → RandomFiltering (4-band) → SpecAugment (freq L=10×3 p=0.3 + time L=20×3 p=0.3) | MED — full pipeline to verify in source |
| MixUp target rule | `target = clip(target_focal + target_other, 0, 1)` (sum-clip) | HIGH (paper + audit) |
| Optimizer | EffNetV2-S: AdamW lr=1e-4 cos→1e-6. NFNet-L0: RAdam lr=1e-3 cos→1e-6. | HIGH (paper Table 11) |
| Epochs / BS | 50 epochs, batch 64 | HIGH |
| Checkpoint selection | `last.ckpt` (NOT best-val) | HIGH (matches exp 032 Phase D pattern) |
| Folds | exp 032 5-fold (StratifiedKFold focal + GroupKFold SS, seed=42), fold 0 first | — |
| Postprocessing | TopN N=1 (Phase H); site/hour priors (Phase H stretch); sliding-window TTA (Phase H stretch) | HIGH (paper §5.6) |

### Procedure — phases (not days)

#### Phase A — Source reading + verified config

**Goal**: confirm every "MED" cell in the setup table by reading the actual Sydorskyi source code (not the paper).

- Clone `github.com/VSydorskyy/BirdCLEF_2025_2nd_place` and download `vladimirsydor/bird-clef-2025-all-pretrained-models` from Kaggle to PACE.
- Read line-by-line: `code_base/models.py` (head structure), `code_base/dataset.py` (mel + standardize axes + augmentation), `code_base/losses.py` (FocalBCELoss α/γ), `code_base/samplers.py` (class-balancing γ), `create_pretrain_backbone.py` (checkpoint key format).
- Write verified config to `docs/sydorskyi-config-verified.md` with exact line-number references back to the source.
- Specifically resolve: head hidden dim (**512 or other?**), standardize axes (per-spectrogram vs per-batch), focal α/γ, mel `normalized` flag handling, ckpt key prefix.

**Phase A exit criterion**: verified config doc landed. Every parameter in the setup table either marked HIGH or has a source-file line reference.

#### Phase B — Implementation

**Goal**: standalone implementation of mel + head + checkpoint loader + loss + sampler, ready to plug into PyTorch Lightning training.

- Install `nnAudio` via `uv pip install` inside the active venv.
- Implement `nnAudio.Spectrogram.MelSpectrogram` wrapper with verified params (Phase A).
- Implement SED-attention head (512 hidden + GeM freq + dropout + ReLU). Reuse exp 032's framework where compatible (LightningModule base class).
- Implement checkpoint loader: `torch.load` → print first 20 keys → `load_state_dict(strict=True)` → assert non-zero conv weight norms. **NEVER `strict=False` on first attempt**.
- Implement `FocalBCELoss` with verified α/γ; label smoothing α=0.05; class-balancing `WeightedRandomSampler` with γ from Phase A.
- Implement augmentation pipeline (random crop → MixUp sum-clip → BG noise → peak-norm → mel → RandomFiltering → SpecAugment).
- Write unit tests (mirroring `tests/032-tucker-distilled-sed/` pattern).

**Phase B exit criterion**: every component has a unit test that passes; no NaN/Inf on a single forward+backward pass; loss tensor finite.

#### Phase C — Gate 1: feature parity (mandatory, no LB)

**Goal**: prove our reimplemented pipeline produces backbone embeddings cosine ≥0.999 against Sydorskyi's published inference notebook on the same audio.

- Pick 5 fixed BirdCLEF 2026 test_soundscapes audios. Use Sydorskyi's published Kaggle inference notebook (`bird-clef-2025-minimul-inference`, version 245080754) to run them and dump the 1280-dim backbone embeddings.
- Run the same 5 audios through our Phase B pipeline (frozen-backbone, our wrappers, our nnAudio mel) and dump embeddings.
- Compute cosine similarity per audio. Compute mean L2 distance. Target: **cosine ≥0.999** on all 5; mean L2 < 1e-4.
- If fails: enumerate which Phase A "MED" cell is wrong. Diff intermediate tensors (mel → standardize → min-max → backbone input → embedding) to pinpoint the parity break.

**Phase C exit criterion**: cosine ≥0.999 on all 5 audios. No LB submission until this passes.

**Phase C pivot trigger**: if cosine <0.99 after 6+ hours of debug, **abandon Path A**. Pivot to Path C fallback (Phase I-only path: rank-ensemble exp 023 + exp 032 5-fold).

#### Phase D — Gate 2: frozen-backbone head sanity (no LB)

**Goal**: confirm the head wiring + feature path produces meaningful OOF AUC before any LB burn.

- Freeze the entire backbone. Random-initialize the head.
- Train head only on BirdCLEF 2026 fold 0, 5 epochs.
- Compute local OOF AUC on fold 0 holdout SS pool.

**Phase D exit criterion**: local OOF AUC ≥ **0.95** (Sydorskyi's reference: 0.984 pre-PL on EffNetV2-S, paper Table 5). If <0.93, head wiring or feature path is wrong — return to Phase C.

#### Phase E — Frozen-backbone fine-tune + first LB

**Goal**: train the full frozen-backbone setup on a single fold and submit to LB. **This LB submission is a pipeline-validation read, NOT a final score read.**

- Train frozen-backbone head fine-tune on fold 0 for 25-50 epochs (start at 25, extend if not converged).
- Run D.5 e2e validation (gap < 0.02).
- Export ONNX via `experiments/010-submission-pipeline/` (extend export.sbatch with a new helper for xcpretrain models).
- Upload to Kaggle, push kernel, submit.

**Phase E exit criterion + decision matrix**:

| LB outcome | Reading | Next move |
|---|---|---|
| ≥ 0.91 | Pipeline works. Pretrain transferred. | Proceed to Phase F (5-fold). |
| 0.90-0.91 | Borderline. Likely pipeline issue. | Re-audit Phase C parity. Sanity-check fold split + augmentation. |
| < 0.90 | Pipeline broken OR pretrain doesn't transfer at all. | **Stop scaling**. Debug Phase B for a silent bug. If still broken after 1 phase of debug, fall back to Path C (Phase I direct). |
| > 0.92 | Ahead of mode. | Consider skipping Phase G full-FT and going direct to Phase I ensemble. |

#### Phase F — 5-fold frozen-backbone average + LB

**Goal**: validate the lever scales.

- Train remaining 4 folds frozen-backbone.
- Average 5-fold predictions (logit-space, matching exp 032's `predict_soundscape_tucker_ensemble` path).
- LB submission.

**Phase F exit criterion**: 5-fold LB ≥ frozen-backbone single-fold + 0.005 (typical fold-ensemble lift). LB ≥ 0.92 confirms on-track.

**Phase F pivot trigger**: if 5-fold LB < 0.92, pretrain may not transfer well. Try NFNet-L0 variant (Phase I.optional) or accept 0.91-0.92 ceiling and skip to Phase H + Phase I.

#### Phase G — Full fine-tune (unfreeze backbone)

**Goal**: test whether unfreezing the backbone with a small LR lifts over the frozen-backbone baseline. Sydorskyi paper Table 11 reports this as the production config (full FT, not frozen).

- Full FT EffNetV2-S fold 0 with AdamW lr=1e-4 cosine→1e-6, 50 epochs.
- Compare paired LB to Phase E frozen-backbone fold 0.

**Phase G exit criterion**: full-FT LB > frozen-backbone LB. If yes, scale to 5 folds.

**Phase G pivot trigger**: if full-FT regresses, revert to frozen-backbone for final. Common cause is catastrophic forgetting of the in-domain pretrain on small 2026 train set.

#### Phase H — Postprocessing + TTA

**Goal**: layer in the disclosed PP + TTA levers. Nikita's discussion comment (2026-05-28) reports +0.020 from full PP package; Sydorskyi solo branch reports +0.012 from sliding-window TTA. These stack additively in most reports.

- TopN N=1 postprocessing (Sydorskyi paper §5.6).
- Site/hour priors (per Tucker / Ferariz reference repo `priors.py`).
- Gaussian smoothing in logit space (5-tap, matches exp 032 path).
- Sliding-window TTA (2.5-sec left/right shifts).
- LB submissions: one with PP only; one with PP+TTA.

**Phase H exit criterion**: cumulative PP lift ≥ +0.005 LB. If a specific PP layer regresses, drop it (Jack's warning: "most things hurt my score").

#### Phase I — Ensemble

**Goal**: ensemble the xcpretrained model with our existing strong models. **This is the lever that bridges 0.92-0.93 single-model to 0.94+ final.**

- Rank-ensemble: xcpretrain-FT (Phase G or E) ⊕ exp 032 distilled-sed-kd-5fold (0.915) ⊕ exp 023 Perch-MLP (0.929).
- Optional: add NFNet-L0 xcpretrain variant for backbone-diversity.
- Optional: 2-seed identical-architecture ensemble (Nikita's stability lever).
- LB submissions: stepwise — xcpretrain+032, xcpretrain+032+023, xcpretrain+032+023+NFNet — to attribute lift per added member.

**Phase I exit criterion**: ensemble LB > best Phase G/H single. Stop adding members when a new member doesn't lift.

#### Phase J — Final submission selection

**Goal**: pick 2 final submissions (Kaggle allows 2) — one high-LB-high-variance + one safer-lower-LB-lower-variance.

- Sanity inference on a local 100-soundscape holdout for both candidates.
- Confirm submission notebook runs in <90 min CPU (Kaggle inference time budget for hidden test set).
- Submit by 2026-06-03 18:00 UTC (safety margin against deadline).

**Phase J exit criterion**: both final picks tested end-to-end; inference time confirmed.

### Pivot-triggers summary (cross-phase)

| Trigger | Pivot |
|---|---|
| Phase C cosine <0.99 after extended debug | **Abandon Path A**. Path C fallback (ensemble existing models only). |
| Phase E LB < 0.90 | Pipeline broken. Re-audit Phase B/C. If still broken, fall back to Phase C/Path C. |
| Phase F 5-fold LB < 0.92 | Pretrain doesn't transfer well. Try NFNet-L0 variant OR accept ceiling, jump to Phase H+I. |
| Phase G full-FT regresses vs Phase E frozen | Revert to frozen-backbone for final. |
| Phase H PP layer hurts LB | Drop that PP layer specifically. |
| **Any phase: silent bug detected post-LB** | Re-run last 3 LB submissions' offline counterparts to check invalidation. |

### Variables

- **Independent**: backbone (EffNetV2-S vs NFNet-L0), training mode (frozen-backbone vs full-FT), fold count (1 vs 5), PP layers on/off, ensemble members on/off.
- **Dependent**: paired LB, fold-0 OOF AUC, D.5 e2e val/auroc, training stability (NaN/Inf, gradient norm).
- **Controlled**: 5-fold split seed=42 from exp 032; 5s training windows; 32kHz sample rate; PACE rtx6000 GPU; same paired-LB methodology as exp 030/031/032.

## Evaluation

### Metrics

| Metric | Description | Role |
|---|---|---|
| Paired LB lift | LB(xcpretrain) − LB(distilled-sed-kd Phase D 0.906) | **Primary read** |
| Frozen-vs-full-FT LB delta | Paired test within xcpretrain branch | Validates Phase G decision |
| 5-fold lift | LB(5-fold avg) − LB(single fold) | Validates Phase F scaling |
| Ensemble lift per added member | Stepwise LB | Validates Phase I composition |
| Local OOF AUC | Fold-0 holdout SS pool | Gate 2 + Phase F sanity |
| D.5 e2e val/auroc | Kernel-inference path | Mandatory pre-LB gate |

### Baselines

- **Primary**: `distilled-sed-kd` Phase D fold 0 = LB 0.906 (paired-within-codebase, reproducible at HEAD).
- **Secondary** (for ensemble Phase I): exp 023 Perch-MLP = 0.929, exp 032 5-fold = 0.915.
- **External anchors**: Ali Ozan 0.941, Jack 0.92x range (xcpretrain confirmation points); Nikita 0.935 no-PL (top of xcpretrain-without-PL band).

### Success criteria

See § Falsifiable success matrix above.

## Limitations

- **Reproducibility risk (research-quantified at 80-85%)**: at least one of nnAudio mel / standardize axes / head hidden dim / checkpoint keys / focal α/γ has a non-trivial silent-bug probability. The Phase A→C→D pre-flight gates exist to catch these before LB burn but may not catch all of them.
- **PP variance (Jack's warning)**: "most things I've tried have only hurt my score." Phase H PP/TTA is high-variance and adds individual LB burns to validate each layer.
- **Compute budget for ensemble variants (Phase I)**: a full NFNet-L0 xcpretrain branch + a 2-seed ensemble adds substantial wall-clock; we may skip these and accept a smaller ensemble if compute or LB-slot budget constrains.
- **Deadline-coupled risk**: 2026-06-03 23:59 UTC final-submission deadline. Phase J has no slack for re-doing if a silent bug surfaces at the last day.
- **Hardware variance**: cross-PACE-node LB has 0.007-0.013 single-fold variance (per exp 030/031). The Phase E first-LB "≥ 0.91 = pipeline works" gate has WEAK boundary noise — re-submission on a different node may be needed to disambiguate.
- **Single iteration of PL only (if revisited later)**: 4 independent confirmations (Anil Ozturk / Boredom / Zejun_ / Salman) that PL plateaus at round 1. If PL is added on top of xcpretrain in a future round, plan one iteration, not many.

## Next-step plan (after exp 034)

| Phase 034 outcome | Next experiment / action |
|---|---|
| CONFIRM (LB ≥ 0.93 single-fold) | Layer PL on top of xcpretrained base (one iteration, Sydorskyi-faithful with trim + chunk-discard + focal × PL fusion). Stretch toward 0.94+. |
| PARTIAL (0.91-0.93) | Iterate Phase H PP variants + Phase I ensemble compositions. Target final LB 0.93+ via stacking. |
| WEAK (0.906-0.91) | Re-audit Phase B/C parity. If still WEAK after second pass, accept ceiling and go direct to Phase I ensemble. |
| REJECT (< 0.906) | Path C fallback (ensemble existing 0.906 + 0.915 + 0.929 without xcpretrain). No further xcpretrain iteration. |

## Acknowledgments

- Sydorskyi & Gonçalves for publishing complete reproducible weights + code + paper.
- Top-team disclosures in `notes/kaggle-discussion-best-single-model-lb.md` for the LB-coupled feedback that anchored Phase E gate thresholds.
- The independent deep research review (2026-05-28) for the implementation-risk quantification and the frozen-backbone-first sequencing that distinguishes this plan from a 3-day full-FT bet.
