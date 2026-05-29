# BirdCLEF 2026 — Independent Pivot Assessment (7 Days to Deadline)

**Author:** Independent research review
**Date:** 2026-05-27 (final submission 2026-06-03, 23:59 UTC)
**Subject:** Honest, non-validating challenge of the Sydorskyi-yoink pivot vs. PL-fix vs. hybrid alternatives.

---

## 1. Executive Summary

**Verdict: Commit to Sydorskyi-yoink as the primary path, but execute it as a *frozen-backbone-first* protocol — not the 3–4 day full-finetune the user is currently scoping. The honest realistic landing band is 0.915–0.930 (mode ≈ 0.925), NOT Ali Ozan's 0.941.** Treat 0.94 as a stretch goal contingent on a clean ensemble with the existing 0.906 distilled-sed-kd model, not on yoink alone.

**Strongest evidence supporting the pivot:** Two independent operators (Ali Ozan Memetoglu LB 0.941, "Jack" per the user's bundle) have already reproduced Sydorskyi's published 2025 backbone on 2026 data, and Sydorskyi's own CEUR ablation attributes +0.018 Private to in-domain pretraining alone — a transferable gain. The user's current 0.906 base has fundamentally weaker pretraining (timm ImageNet only); no amount of PL ironing will close that gap.

**Strongest counter-evidence the user should consider:** The user's track record (3 silent bugs found post-hoc, 24 anti-correlation points, exp 033 inversion) means a 1:1 Ali Ozan replication is *unlikely* in 7 days. The Sydorskyi recipe uses **nnAudio (not torchaudio) on-the-fly GPU mel + a non-obvious standardize→min-max[0,1] order + a SED-attention head with 512 hidden dim + GeM frequency pooling + label smoothing α=0.05 + FocalBCELoss + nnAudio's `normalized=True` flag** — at least four of these are the exact class of silent-mismatch bug that previously cost the user LB 0.522. Murphy's law on this implementation puts the median outcome around 0.92, not 0.94.

**Bottom line: The user is right about the direction and wrong about the timeline.** A 7-day Sydorskyi-yoink with verified parity testing, frozen-backbone first, and a planned ensemble fallback is the right plan. A "implement Path A and Path B in parallel" plan or "spend 3-4 days doing full finetune" plan will fail.

---

## 2. Question-by-Question Answers (Q1–Q7)

### Q1. Is the Sydorskyi-yoink pivot the right move, or is the user confirmation-biased?

**Yes, the pivot is the right move — but the reasoning the user is using to justify it is partly confirmation-biased, and the corrected reasoning gives a lower expected LB.**

The user's audit identified 16/20 protocol divergences in exp 033 and concluded "we got PL wrong." A different framing asks: *what is the actual ceiling of the 0.906 distilled-sed-kd base even if PL were perfectly implemented?* Sydorskyi's own ablation (CEUR Vol-4038 paper_256, Table 5) shows pseudo-labelling on a pretrain-only EfficientNetV2-S base going from 0.881 Public / 0.889 Private to 0.908 Public / 0.906 Private — a **+0.027/+0.017** delta. That delta is anchored on a base that already includes in-domain XC pretraining (+0.046 Public / +0.018 Private over enhancements alone). On the user's 0.906 base (ImageNet-only pretrain, no in-domain XC), the same +0.017 Private gain would land somewhere around 0.92–0.925 — *exactly the user's stated "50% PARTIAL recovery to LB 0.910–0.913" outcome*.

So the contrarian framing: **exp 033 didn't just have PL bugs; it was anchored on a base that even Sydorskyi's perfectly-implemented PL would only push to ~0.925.** PL is a multiplier on a strong base, not a substitute for one. The +0.018 Private from in-domain pretraining — Sydorskyi & Gonçalves explicitly state: *"The final pretraining dataset comprised 819,032 recordings spanning 7,489 species"* (CEUR Vol-4038 paper_256, §5.3) — is the more important lever, and yoink delivers it for free.

The confirmation-bias risk: an implicit assumption that *"Sydorskyi-yoink → 0.941 because Ali Ozan got 0.941."* Ali Ozan represents one data point of clean execution; the user's track record gives a 60–70% probability of at least one mel/head/checkpoint mismatch bug on a first-attempt implementation. The honest expected value is closer to 0.92 than 0.94.

**Bayesian outside view:** weight Ali Ozan's 0.941 + Jack's reported 0.941 + the user's prior bug rate → Expected single-model LB = 0.915–0.930 with std ~0.012. Reaching 0.94 requires ensembling.

### Q2. Realistic LB landing band for Sydorskyi-yoink

**Three-band probability distribution:**

| Outcome band | Probability | Reasoning |
|---|---|---|
| **0.91–0.92** (Murphy's law: one silent bug like exp 023→033 pattern) | **35%** | One mel parameter off (`win_length`, `f_max`, standardize axis), one normalization order swap, one checkpoint key remap missing, OR backbone channel mismatch (NFNet-L0 is 2304 channels per `timm`, not the 1024 the user assumed). |
| **0.92–0.93** (clean head/mel, no ensemble lift) | **40%** | Single-fold or 5-fold-average frozen-backbone fine-tune, matches Sydorskyi's own single-iteration PL Public score band. |
| **0.93–0.94** (clean execution + 5-fold average + minor TTA) | **20%** | Approaches Ali Ozan band; requires perfect mel parity AND postprocessing (TopN N=1). |
| **0.94–0.95** (yoink + ensemble with existing 0.906 distilled-sed-kd) | **5%** | Only path to 0.94+ within 7 days given user's track record. |

**Mode: ~0.925.** This is approximately the user's "50% PARTIAL recovery" prior but for a *different reason*: not because PL "partially recovers," but because the base improvement from yoink is itself worth ~0.015–0.020 LB over the current 0.906 baseline, *before* any PL.

The 0.94 cliff is real: Salman Ahmed (LB 0.939 solo / 0.944 w/ PL / 0.952 ensemble) — single models don't routinely clear 0.94. Even Sydorskyi's ensemble only hit Private 0.928 in 2025: per CEUR Vol-4038 paper_256 abstract, *"placing 2nd on the Private Leaderboard with a ROC AUC of 0.928 on the Private test set and 0.925 on the Public test set."*

### Q3. Realistic LB landing for "fix all 5 PL divergences" alternative

**This path lands in the 0.910–0.918 band with high confidence. The base is too weak for PL alone to clear 0.94.**

Evidence:
- **Salman Ahmed disclosed PL adds +0.005 solo** on a 0.939 base (per bundle). On a 0.906 base, the gain may even be smaller because PL benefits scale with teacher quality.
- **Boredom went 0.938-0.939 → 0.950+ with 1-iter PL** (per bundle, +0.012). That is a +0.012 jump on a base that already clears 0.94. Replicating Boredom's PL on a 0.906 base would land near 0.918, not 0.95.
- **Sydorskyi's own paper (Table 5)** shows the largest PL jump was 0.881 → 0.908 Public (+0.027) on a base that *already had in-domain XC pretraining*. The "PL works on a strong base" pattern is consistent across all top-team disclosures.
- The Phase 5 Test 1 v3 run the user already launched (chunk-discard + trim) has best-case prior of ~0.913 per the user's own honest estimate.

**Conclusion: Path B alone cannot clear 0.94 in 7 days.** It can at best recover to the 0.91–0.92 band, which is the same band Path A delivers more reliably. Path B as a *secondary* lever on top of the yoinked base is interesting but should not be the primary path.

### Q4. Implementation risk enumeration for Sydorskyi-yoink (ranked)

Verified via the published CEUR paper (Sydorskyi & Gonçalves 2025, Appendix B / Table 11) and the GitHub README. The subagent investigation confirmed:

| Rank | Risk | Likelihood | LB Impact if Wrong | Mitigation |
|---|---|---|---|---|
| **1** | **Mel library mismatch: Sydorskyi uses `nnAudio` (not `torchaudio.transforms.MelSpectrogram`).** nnAudio has different default `window`, `normalized=True` semantics, and uses 1D-conv kernels internally — subtle numerical differences cascade. | **High (60%)** | **−0.05 to −0.15** (cf. exp 023 LB 0.522 collapse) | Use `nnAudio.Spectrogram.MelSpectrogram` directly. Do NOT translate to torchaudio. Test parity by feeding 5 known audio clips through Sydorskyi's published inference notebook (`bird-clef-2025-minimul-inference` v245080754) and the user's pipeline; compare logits via cosine similarity (target >0.999). |
| **2** | **Standardize → min-max[0,1] order with unspecified standardize axes.** Paper says "AmplitudeToDb (top_db=80, amin=1e-10) → standardize → min-max to [0,1]." Axes (per-spectrogram vs per-batch vs per-channel) are NOT stated in the paper. | **High (50%)** | **−0.02 to −0.08** | Read `code_base/dataset.py` or `code_base/models.py` from the GitHub repo *before* training. The standardize step is the most easily-broken normalization in audio pipelines. |
| **3** | **Head hidden dim is 512, not 1280, not 1028.** The bundle's "attention dim 1028" is a typo — the correct value (per CEUR paper Appendix B) is **512 hidden** with **GeM frequency pooling** and **SED-attention time pooling (no RNN)**, dropout 0.25 after backbone, 0.5 after hidden, ReLU activation. Backbone output is 1280 (EffNetV2-S `tf_efficientnetv2_s_in21k`) or 2304 (NFNet-L0 `eca_nfnet_l0`, NOT 1024). | **Medium (40%)** | **−0.01 to −0.05** | Match the head exactly from `code_base/models.py`. If the user has been targeting 1028, they have been wrong. Verify backbone channel count by `print(backbone.num_features)` after loading. |
| **4** | **Checkpoint key remapping.** Sydorskyi's `create_pretrain_backbone.py` outputs a backbone-only state_dict, but PyTorch Lightning typically wraps it under a `'state_dict'` key with `'backbone.'` prefix. Silent `load_state_dict(strict=False)` may load zero weights. | **Medium (35%)** | **−0.05 to −0.20** | After `torch.load`, print `list(ckpt.keys())[:20]`. Then `model.load_state_dict(ckpt, strict=True)` — never `strict=False` on a first attempt. Assert that mean of conv weights is non-zero post-load. |
| **5** | **Sample rate / resampling mismatch.** Sydorskyi trains and infers at 32 kHz. If the user's pipeline resamples to a different rate or uses a different librosa/soundfile backend (mp3 vs ogg), the mel will be subtly different. | **Medium (30%)** | **−0.02 to −0.05** | Force 32 kHz, use `torchaudio.load` with `backend='soundfile'`, verify by comparing 5 sample spectrograms numerically against Sydorskyi's reference notebook. |
| **6** | **FocalBCELoss alpha/gamma not specified in paper.** Reference [38] (Lin et al.) defaults to α=0.25, γ=2.0, but Sydorskyi's exact values must be read from `code_base/losses.py`. | **Medium (30%)** | **−0.005 to −0.02** | Read source file directly. Use exact values. |
| **7** | **Label smoothing α=0.05 missing.** Paper §5.2 specifies α=0.05. Easy to forget. | **Low (20%)** | **−0.005 to −0.01** | Apply LS=0.05 to targets before BCE. |
| **8** | **5-second clip handling: random vs first/last 7-sec window.** Paper §5.5 found "last approach" (random 5s from first OR last 7s) marginally better. The exact crop strategy at training vs inference time matters. | **Low (15%)** | **−0.005 to −0.01** | Use random 5-sec crops at training; non-overlapping 5-sec at inference. |
| **9** | **ONNX/OpenVINO export not needed for training**, only for final submission notebook. Less of a risk during the 7 days; only matters in last 24 hours. | **Low (15%)** | **−0.005 to −0.01** | Skip ONNX until day 6 — only convert if final submission notebook is timing-bound. |
| **10** | **Training-time class balancing**: Sydorskyi uses Sqrt balancing (γ=−0.5) for NFNet and Equal (γ=−1) for EffNetV2-S; mismatched balancing on the user's 2026 class distribution will subtly mis-train. | **Low (15%)** | **−0.005 to −0.015** | Use γ=−0.5 sqrt-balancing by default for EffNetV2-S yoink branch; verify on local CV. |

**Aggregate "at least one bug" probability for a first-attempt implementation by an operator with the user's track record: ~80–85%.** This is the central reason the realistic landing band is 0.92, not 0.94.

### Q5. Pre-flight verification steps (before any LB burn)

The user has already wasted 3 LB submissions on exp 033. **No LB submission should be burned on Path A until all of the following gates pass offline.**

**Gate 1: Feature parity (≤ 2 hours).** Run 5 fixed BirdCLEF 2026 test audio files through both: (a) Sydorskyi's published inference notebook on Kaggle (`vladimirsydor/bird-clef-2025-minimul-inference`, version 245080754) executing his frozen pretrained checkpoint, and (b) the user's reimplemented pipeline with the same checkpoint. Compare the 1280-dim backbone-output embeddings (or final logits if head is matched). **Cosine similarity must be ≥ 0.999** per audio file. Mean L2 distance < 1e-4. If not, stop and fix.

**Gate 2: Frozen-backbone head sanity check (≤ 4 hours).** Freeze the entire backbone. Train ONLY the head (random init) on a single fold of BirdCLEF 2026 train data for 5 epochs. Local OOF AUC must beat 0.95 (Sydorskyi's baseline pretrain achieved 0.984 local CV pre-PL on EffNetV2-S, per CEUR paper Table 5). If local OOF < 0.93, the head wiring or feature path is wrong.

**Gate 3: Class-count assert.** Print `model.head.classifier.out_features` — must equal the 2026 class count exactly. The pretrained head outputs 7,489 classes (CEUR §5.3); it must be re-initialized to the BirdCLEF 2026 number of classes (verify against `taxonomy.csv`).

**Gate 4: Loss-scale sanity.** First training step total loss should be in a reasonable BCE range (initial loss ≈ −log(0.5) = 0.693 for mean-reduction BCE). If first-step loss is wildly different, the head outputs are not raw logits or the targets are wrong.

**Gate 5: D.5 validation script re-audit.** Given the prior hardcoded `fold_k=0` false-PASS bug, before trusting any offline metric the user MUST grep their validation code for hardcoded indices and rerun the previous false-PASS scenarios to verify they now FAIL where they should.

**Only after Gates 1–5 pass should the first LB submission be burned** (and that submission should be a frozen-backbone head-only fine-tune, NOT a full fine-tune — to isolate "did the pipeline work" from "did the fine-tune work").

### Q6. Unknown levers the user may be missing

**(a) BirdCLEF 2024 Kefir 1st place (Anil Ozturk et al.):** Per the public Zenn summary (https://zenn.dev/yuto_mo/articles/ad43c630729073) of `kaggle.com/competitions/birdclef-2024/discussion/512197`, Kefir's key levers were: (i) **Google bird-vocalization-classifier-based filtering of train_audio** — drop chunks where the classifier's max prediction doesn't match the primary label; replace primary with secondary if the classifier matches the secondary; (ii) **soft-distributed labels** when secondary labels exist (primary=0.5, remaining 0.5 distributed among secondaries); (iii) **Google-classifier pseudo-labels at coefficient 0.05** (treated as label smoothing, not as ground truth); (iv) **multistage training** (train on labeled, finetune on unlabeled-soundscape-PL, or vice versa); (v) **train on 10-second chunks, inference on 5-second**; (vi) **random fold0** as default validation. Of these, **(i) and (iii) are orthogonal to Sydorskyi's recipe and the user has not tried them.** A Google-classifier-based train-audio filter, applied to BirdCLEF 2026 train_audio before fine-tuning the yoinked backbone, is a cheap +0.003–0.008 lever.

**(b) Perch 2.0** (van Merriënboer, Dumoulin, Hamer, Harrell, Burns, Denton; arXiv:2508.04665v2, 5 Jan 2026): Google's Perch 2.0 explicitly introduces (i) **multi-source mixup (>2 sources mixed simultaneously)** and (ii) **prototype-learning classifier self-distillation** as the two innovations over Perch v1. Per the arXiv abstract: *"Perch 2.0 obtains state-of-the-art performance on the BirdSet and BEANS benchmarks. It also outperforms specialized marine models on marine transfer learning tasks, despite having almost no marine training data."* **The relevant orthogonal lever: replace the user's 2-source mixup with 3- or 4-source mixup.** This is a ~5-line change.

**(c) BirdSet / Bird-MAE / AVES family (BirdSet ICLR 2025 spotlight; Bird-MAE arXiv:2504.12880):** Bird-MAE-Large/Huge sets SOTA on BirdSet's seven downstream tasks via masked autoencoder pretraining on the 6,800-hour BirdSet corpus, *outperforming Perch on multi-label probing*. **Orthogonal lever: as a tertiary backbone for ensembling, a frozen Bird-MAE-Base embedding + small MLP head is fast to train and uncorrelated with Sydorskyi/EffNetV2 features.** Probably too much new code for 7 days, but if the yoink fails on Day 3, this is the next-best backbone alternative.

**(d) Site/hour priors + Gaussian smoothing (Tucker / "Goron" recipe lever).** Per the BirdCLEF 2026 strategy playbook (lamsade.dauphine.fr) and the `ferariz/birdclef2026` reference repo (which explicitly ships `priors.py` site/hour module + `postprocess.py` smoothing/temperature/blending), site/hour priors and temperature scaling are routine postprocessing scoring +0.005–0.010 LB. The pattern dates to BirdCLEF 2021 10th place (Kumar Shubham et al., github.com/kumar-shubham-ml/kaggle-birdclef-2021) which built a postprocessing model using site (longitude, latitude), rarity, and co-occurrence features. **For the user, this is a Day 6 cheap +0.005 add — but only worth it if a single model is at 0.92+; below that, fixing the base is higher EV.**

**(e) Sliding-window TTA (Antoine Masq's recipe — LB 0.943 single).** Per the bundle and tekkix.com's BirdCLEF+ 2025 top-5 summary, Sydorskyi's solo branch went *"TTA with 2.5-second left/right shifts (0.91 → 0.922)"* — i.e., +0.012 from sliding-window TTA. Same Day 6 candidate.

**(f) Soft-AUC loss (BirdCLEF 2025 5th place, "Nikita" / myso1987).** Per tekkix.com, the team using `SoftAUCLoss` was a specific top-5 team running a "zoo" of (tf_efficientnet_b0_ns, v2_b3, v2_s.in21k, mnasnet_100, spnasnet_100) — single team, not several. Marginal lever (probably +0.001–0.005); not worth a code rewrite at this stage.

**(g) Noisy-student-style multi-iterative PL — BirdCLEF+ 2025 1st place (Nikita Babych, solo).** Per Kaggle Winners Walkthrough playlist and the 1st-place writeup at kaggle.com/competitions/birdclef-2025/discussion/583577 *"1st Place Solution: Multi-Iterative Noisy Student Is All You Need"* — Nikita's primary insight was that **iterative noisy-student PL with diverse augmentations** (not single-iter PL) drives the top of the LB. This is consistent with the user's bundle observation that "PL doesn't work for more than one pass" being the *common* finding but not universal — for sufficiently noised students, multi-iter can work. **Out of scope for 7 days, but flag for future.**

**Of these, the highest-EV orthogonal additions in 7 days are: Kefir's Google-classifier train-audio filter (Day 4) and Tucker-style site/hour priors + TopN N=1 postprocessing (Day 6). Both together are realistically worth +0.005–0.012 on top of the yoinked backbone.**

### Q7. Concrete day-by-day plan

**Path commitment: Path A (Sydorskyi-yoink) primary, Path B (PL fix) deprecated except as a fallback. Path C (ensemble with existing 0.906 distilled-sed-kd) is the 0.94-reach lever.**

**LB-slot accounting:** 5/day Kaggle quota. 2 already used 2026-05-27 on diagnostics → 3 left today. The plan below conservatively budgets 1 LB submission per day on average, banking slots for Days 5–7 when ensemble decisions need rapid iteration.

| Day | Date | Primary deliverables | LB budget | Continue-gate (must pass to proceed) |
|---|---|---|---|---|
| **0 (today)** | **2026-05-27** | (a) Clone `VSydorskyy/BirdCLEF_2025_2nd_place`. (b) Download `vladimirsydor/bird-clef-2025-all-pretrained-models`. (c) Read `code_base/models.py`, `code_base/dataset.py`, `code_base/losses.py` manually (≤2 hours). (d) Write down exact `win_length`, `f_max`, `power`, standardize axes, focal α/γ. (e) Inspect checkpoint keys via `torch.load`. (f) Phase 5 Test 1 result captured for record (do NOT submit). | **0** (bank remaining 3 of today's slots) | Source files actually read; exact parameter table written to `docs/sydorskyi-config-verified.md`. |
| **1** | **2026-05-28** | (a) Implement Sydorskyi mel pipeline with `nnAudio` (NOT torchaudio). (b) Implement SED-attention head with 512 hidden, GeM freq pool, dropout 0.25/0.5. (c) Run **Gate 1 parity test**: 5 fixed audios, cosine sim ≥0.999 vs Sydorskyi inference notebook. (d) Run **Gate 2 frozen-backbone head sanity** on fold 0 (5 epochs). | **0** | Gate 1 PASS (cosine ≥0.999); Gate 2 local fold-0 AUC ≥ 0.95. **If Gate 1 fails: stop, debug all day 2 if needed — do NOT submit.** |
| **2** | **2026-05-29** | (a) Full frozen-backbone head fine-tune, single fold (fold 0), 30 epochs. (b) **First LB submission**: this frozen-backbone single-fold model only — pipeline validation, NOT final score. (c) Begin all-5-folds frozen-backbone training overnight. | **1** | LB ≥ 0.91 confirms pipeline works. **If LB < 0.90: pipeline is broken — pivot to Path C (ensemble of existing 0.906 + Perch v2 baseline). If LB 0.90–0.91: proceed but expect ceiling at 0.93. If LB > 0.92: ahead of schedule.** |
| **3** | **2026-05-30** | (a) All-5-folds frozen-backbone arithmetic-mean ensemble. (b) Begin Sydorskyi-recommended full fine-tune (unfreeze backbone) on EffNetV2-S, fold 0. (c) **Second LB submission**: 5-fold frozen-backbone average. | **1** | LB ≥ 0.92 confirms in-domain pretraining transferred. **If LB < 0.92: re-audit mel/head. If LB 0.92–0.93: on-track. If LB > 0.93: drop full-finetune plan, go straight to ensemble.** |
| **4** | **2026-05-31** | (a) Full fine-tune all 5 folds (EffNetV2-S). (b) Optional: Kefir-style Google bird-vocalization-classifier train_audio filter applied. (c) **Third LB submission**: full-finetune 5-fold average. | **1** | LB ≥ 0.93 confirms yoink is paying out. **If LB < 0.92 vs Day 3: full-finetune is regressing → revert to frozen-backbone for final. If LB 0.93+: proceed to ensemble.** |
| **5** | **2026-06-01** | (a) Add TopN postprocessing (N=1, per Sydorskyi paper §5.6). (b) Ensemble: average yoink-finetune ⊕ existing 0.906 distilled-sed-kd model. (c) Two LB submissions: one with PP, one ensemble with/without PP. | **2** | LB ≥ 0.935 for single-model + PP. Ensemble target ≥ 0.94. **If ensemble < 0.93: ensemble hurt — submit non-ensemble. If ensemble 0.93–0.94: keep ensemble. If > 0.94: stop optimizing, stress-test.** |
| **6** | **2026-06-02** | (a) Tucker site/hour priors postprocessing layer. (b) Sliding-window TTA (2.5-sec shift, per Sydorskyi solo branch which lifted 0.91 → 0.922). (c) NFNet-L0 yoink branch (if compute allows) — uncorrelated backbone for ensemble diversity. (d) Two LB submissions: one with TTA+priors, one with NFNet added to ensemble. | **2** | Identify the best stable submission. **If best > 0.94: lock it as Final Submission #1 candidate; reserve Day 7 for safer #2 (lower-variance) selection.** |
| **7** | **2026-06-03** | (a) Final submission selection (Kaggle allows 2 final picks). (b) Pick one high-LB high-variance + one safer lower-LB lower-variance. (c) Sanity inference on local 100-soundscape holdout to confirm submission notebook runs in <90 min CPU. (d) **Final LB submission** ≤ 18:00 UTC for safety margin. | **1** | Both final picks tested in submission notebook. Inference < 90 min CPU. |

**"If X by Day N, pivot to Y" decision matrix:**

| Condition | Pivot |
|---|---|
| Day 1: Gate 1 parity cosine < 0.99 after 6+ hours debug | Abandon Path A. Commit Day 2 to **Path C: ensemble existing 0.906 distilled-sed-kd + Perch v2 baseline** (kailyn2359 baseline notebook on Kaggle gives starter Perch v2 model). Realistic ceiling 0.92. |
| Day 2: First LB submission < 0.90 | Mel/head pipeline is broken. Spend Day 3 on parity test rather than scaling. If still broken after Day 3, **fall back to Path C ensemble + Path B's v3 PL fix on existing distilled-sed-kd**. |
| Day 3: 5-fold frozen-backbone LB < 0.92 | Pretrained backbone is not transferring. **Pivot to NFNet-L0 yoink variant** (different pretrain dataset). If still <0.92, accept 0.91–0.92 ceiling and shift remaining days to ensemble+postprocess maximization. |
| Day 4: Full-finetune regresses vs frozen-backbone | Revert to frozen-backbone. Spend Day 5–6 on ensemble + postprocess, not training. |
| Day 5: No single-model > 0.93 | **0.94 is out of reach.** Optimize for safest submission near current best; do not chase variance. |
| Day 6: Ensemble doesn't lift over single-model | Submit single-model. Use Day 7 for inference-time safety. |
| Any day: silent bug detected in offline validation | **Immediately re-run last 3 LB submissions' offline counterparts** to check whether prior LB results are now invalidated. |

---

## 3. Risk Enumeration Table

(See Q4 above for full table.) Highlights:

| Axis | Likelihood | LB impact | Mitigation |
|---|---|---|---|
| nnAudio vs torchaudio mel mismatch | High (60%) | −0.05 to −0.15 | Use nnAudio directly; parity test |
| Standardize→min-max order/axes | High (50%) | −0.02 to −0.08 | Read source; verify on 5 sample inputs |
| Head dim wrong (512, not 1028/1280) | Med (40%) | −0.01 to −0.05 | Match exactly from `models.py` |
| Checkpoint key mismatch | Med (35%) | −0.05 to −0.20 | strict=True load; assert non-zero weights |
| Resampling backend | Med (30%) | −0.02 to −0.05 | Force 32 kHz, soundfile backend |
| Focal α/γ unknown | Med (30%) | −0.005 to −0.02 | Read `losses.py` directly |
| LS=0.05 missing | Low (20%) | −0.005 to −0.01 | Apply at target prep |
| Class-balance γ wrong | Low (15%) | −0.005 to −0.015 | γ=−0.5 (EffNetV2-S) / γ=−1 (NFNet) default |
| Anti-correlation pattern recurring | High | Continued LB inversion | Trust LB > offline; Day 1 Gate 5 false-PASS audit |
| 90-min CPU inference timeout (final day) | Med | Submission = 0 | Day 7 dry run on Kaggle CPU notebook |
| Final-selection variance | Med | Lose 0.005–0.015 from best | Always pick 1 stable + 1 high-LB |

---

## 4. References

### Primary sources
- **Sydorskyi & Gonçalves (2025).** "Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation." CEUR Vol-4038 paper_256. https://ceur-ws.org/Vol-4038/paper_256.pdf — verified ablation: pretrain +0.018 Private (EffNetV2-S), PL +0.017–0.020 Private, PP (TopN N=1) +0.015 Private. Verbatim ablation final: *"placing 2nd on the Private Leaderboard with a ROC AUC of 0.928 on the Private test set and 0.925 on the Public test set."* Pretrain corpus quote: *"The final pretraining dataset comprised 819,032 recordings spanning 7,489 species."*
- **GitHub: VSydorskyy/BirdCLEF_2025_2nd_place.** https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place — README confirms `create_pretrain_backbone.py` strips classification head; pretrained models in `bird-clef-2025-all-pretrained-models`.
- **Sydorskyi Kaggle minimal inference notebook.** https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-minimul-inference (scriptVersionId=245080754).
- **Sydorskyi Kaggle writeup.** https://www.kaggle.com/competitions/birdclef-2025/discussion/583699.

### Verified mel/head config (CEUR paper Table 11 + Appendix B)
- sr=32000, n_mels=128, n_fft=2048, hop=512, f_min=20, mel `normalized=True`
- AmplitudeToDb `top_db=80`, `amin=1e-10`
- Standardize → min-max [0,1] (axes not in paper, verify in source)
- **nnAudio (NOT torchaudio) on-the-fly GPU mel**
- Head: 512 hidden, dropout 0.25/0.5, ReLU, GeM freq pool, SED-attention time pool (no RNN), per [Adavanne & Virtanen 2017]
- Loss: FocalBCELoss = BCE + sigmoid-focal (α/γ in source); Label smoothing α=0.05
- 50 epochs, batch 64. EffNetV2-S: AdamW lr=1e-4 cos→1e-6. NFNet-L0: RAdam lr=1e-3 cos→1e-6.

### BirdCLEF 2024 1st place (Kefir / Anil Ozturk et al.)
- Discussion thread: https://www.kaggle.com/competitions/birdclef-2024/discussion/512197.
- Summary: https://zenn.dev/yuto_mo/articles/ad43c630729073 — Google bird-vocalization-classifier filter on train_audio; soft-distributed secondary labels (primary 0.5 + secondaries 0.5); Google-classifier PL as label smoothing at coefficient 0.05; multistage training; 10-sec train / 5-sec inference; "random fold0" validation.

### BirdCLEF 2024 2nd–3rd place (PL pioneers)
- Theo Viel 3rd place (CPMP+Henkel+Viel): https://github.com/TheoViel/kaggle_birdclef2024 — pseudo-labels added at 25–45% rate, target = max(hard, soft), 3 iterations.
- JFP CPMP part: https://github.com/jfpuget/birdclef-2024.

### BirdCLEF+ 2025 1st place (Nikita Babych, solo)
- "1st Place Solution: Multi-Iterative Noisy Student Is All You Need" — https://www.kaggle.com/competitions/birdclef-2025/discussion/583577.
- Multi-iterative noisy-student PL was the disclosed primary lever (out of scope for 7 days but documents that multi-iter PL can work with sufficient student noise).

### Bioacoustic foundation models (orthogonal levers)
- **Perch 2.0** (van Merriënboer, Dumoulin, Hamer, Harrell, Burns, Denton; arXiv:2508.04665v2, 5 Jan 2026): https://arxiv.org/pdf/2508.04665v2 — multi-source mixup; prototype-learning self-distillation; *"Perch 2.0 obtains state-of-the-art performance on the BirdSet and BEANS benchmarks."*
- **BirdSet** (Rauch et al., ICLR 2025 Spotlight, arXiv:2403.10380): https://openreview.net/forum?id=dRXxFEY8ZE — 6,800 hours / 10,000 classes benchmark.
- **Bird-MAE** (arXiv:2504.12880): https://arxiv.org/html/2504.12880v1 — SOTA on BirdSet via MAE pretraining; prototypical probing.
- **BirdAVES** (earthspecies/aves v1.0.0, 2025-04-11): https://github.com/earthspecies/aves — self-supervised HuBERT-style bird-tuned.

### BirdCLEF 2026 strategy playbook
- https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2026.pdf — confirms site/hour priors and temperature scaling as standard postprocessing.
- ferariz/birdclef2026 reference repo: https://github.com/ferariz/birdclef2026 — explicit `priors.py` site/hour module + `postprocess.py` (smoothing, temperature, blending).

### BirdCLEF 2025 top-5 summary (bundle corroboration)
- https://tekkix.com/articles/ai/2025/07/birdclef-2025-overview-of-the-competition-a — confirms top teams used tf_efficientnetv2_s + RAdam, eca_nfnet_l0 + AdamW, 50 epochs, Focal+BCE, cosine LR; pretraining on entire Xeno Archive 0.84→0.87; PL 2 rounds 0.87→0.91; *TTA with 2.5-second left/right shifts (0.91 → 0.922)*. SoftAUCLoss is attributed to a single team's `(tf_efficientnet_b0_ns, v2_b3, v2_s.in21k, mnasnet_100, spnasnet_100)` "zoo" pipeline.

### LifeCLEF 2025 overview
- CEUR Vol-4038 paper_232: https://ceur-ws.org/Vol-4038/paper_232.pdf — overview of BirdCLEF+ 2025; confirms pseudo-labeling, TTA, SED+CNN ensembles as the dominant pattern.

### BirdCLEF 2025 5th place (myso1987)
- https://github.com/myso1987/BirdCLEF-2025-5th-place-solution — SED EfficientNet B0/B3/V2-B3/V2-S architecture variants; 30s/60s cropped training.

---

**End of independent assessment.**

**Save to:** `docs/references/deep-research/2026-05-27-pivot-assessment.md`
