---
experiment: "034-pretrained-backbone"
status: in-progress
date: 2026-05-28
author: Murilo Gustineli
---

# Results — Exp 034 Pretrained-backbone (in-domain XC pretrain reuse)

Reuse the BirdCLEF+ 2025 2nd-place published in-domain-pretrained backbone
(EffNetV2-S / NFNet-L0, pretrained on 819K XC recordings) + our own SED head,
fine-tuned on BirdCLEF 2026. Baseline to beat: `distilled-sed-kd` fold 0 = **LB 0.906**.

Naming note: the code prefix is `xcpretrain` (names the lever — Xeno-Canto
in-domain pretraining). Earlier scaffold used the codename "yoink"; renamed
2026-05-28. Attribution to the source author stays in `proposal.md` § Provenance.

## Phase status

| Phase | What | Status |
|---|---|---|
| A | Source read + verified config (`docs/sydorskyi-config-verified.md`) | ✅ DONE (5 corrections) |
| B.1–B.5 | Forward path: mel + SED head + backbone loader + orchestrator (+35 tests) | ✅ DONE |
| B.6–B.9 | Training path: focal loss + sampler + augmentation ports + Lightning module + train.py | ✅ DONE (58 tests + fast-dev-run) |
| **C — Gate 1** | **Feature parity vs vendored source** | ✅ **PASS (bit-identical)** |
| D — Gate 2 | Frozen-backbone head sanity (OOF AUC ≥0.95) | ⬜ next |
| E → J | Frozen FT + LB → 5-fold → full-FT → PP/TTA → ensemble → final | ⬜ |

## Phase C — Gate 1: feature parity (2026-05-28) ✅ PASS

**Method**: ran our `XCPretrainWaveClassifier` against Sydorskyi's own vendored
`WaveCNNAttenClasifier` (`vendor/sydorskyi-2025/` @ `053594f`) on 5 fixed
`train_soundscapes` audios, locally. Both fed the same waveform tensor; both
backbones loaded from the same cleaned pretrained `state_dict`. Compared at the
mel stage (`return_spec_feature`) and the backbone-embedding stage
(`return_cnn_emb`). Script: `scripts/gate1_parity.py`.

| Backbone | Out ch | Mel cosine / max-abs-diff | Emb cosine / max-abs-diff | Verdict |
|---|---|---|---|---|
| `tf_efficientnetv2_s_in21k` (2024 ckpt) | 256 | 1.00000000 / 0.0 | 1.00000000 / 0.0 | **PASS** |
| `eca_nfnet_l0` (2025 ckpt) | 2304 | 1.00000000 / 0.0 | 1.00000000 / 0.0 | **PASS** |

Result: **bit-identical** on all 5 audios, both stages, both backbones — far
beyond the cosine ≥0.999 / L2 <1e-4 gate. Confirms our B.1–B.5 port faithfully
reproduces Sydorskyi's mel + backbone forward path. Clears the research's
highest-probability catastrophic-bug class (nnAudio mel mismatch −0.05/−0.15;
standardize→min-max order/axes; head dim; ckpt keys). **No LB slot burned.**

Caveat (by design): this tests PORT fidelity, not library version-skew vs
Sydorskyi's original training environment. Version-skew is a lower-risk concern
that fine-tuning absorbs; Gate 2 (frozen-backbone OOF AUC) is the test that the
loaded backbone produces useful features for 2026.

## § B.8.3 — Data-pipeline fidelity decision (reuse vs port)

**What is bit-faithful to Sydorskyi (Gate-1-validated or verbatim-ported + tested):**
mel front-end, backbone weights+arch, SED att-head, FocalLossBCE, SpecAugment,
RandomFiltering, label smoothing (0.05 mass-preserving), class-balanced sampler,
AdamW lr=1e-4 cos→1e-6, 32-true precision, last-ckpt, 5-fold. The MODEL is faithful.

**What diverges (data pipeline / training dynamics), and why it's acceptable now:**
We reuse exp 032's `DistilledSEDDataModule` rather than Sydorskyi's `WaveDataset`.
Divergences: focal/SC mixing (0.9/0.1 MixSamp vs his class-balanced sampler), MixUp
variant (additive-hard vs sum-clip), no 40% PL swap, no BG-noise OneOf, val on
labeled-SS vs his held-fold full-file. These are **inherently un-reproducible in part**
(2026 data ≠ his 2025 data/CSVs) and the deep research ranks each remaining piece in
the **LOW-risk tier** (−0.005…−0.015), vs the HIGH-risk mel/head/ckpt class that Gate 1
already cleared bit-identically.

**Precedent for "own recipe on yoinked backbone":** Ali Ozan (LB 0.941 in-comp) yoinked
this backbone + his OWN head + own fine-tune — he did NOT reproduce Sydorskyi's full
recipe. The validated high-score path is exactly "reuse backbone, own everything else."

**For frozen-backbone (Phases D/E — done first), recipe divergence is low-stakes**: the
valuable, fragile asset (the pretrained features) is frozen and bit-validated, so the
head-training recipe cannot corrupt it; it's a standard supervised problem on fixed
features, where exp 032's recipe is proven (0.906/0.915 LB).

**Three options considered:**
- **A (chosen now)** — reuse exp 032 datamodule + thin XCPretrain module. Lowest bug risk
  (both halves validated), fast, "our own solution." Divergences in LOW-risk tier.
- **B** — port Sydorskyi's WaveDataset from scratch. HIGHEST new bug surface (the exact
  thing the gate philosophy avoids); can't be data-faithful anyway. Rejected.
- **C** — import Sydorskyi's vendored WaveDataset directly, adapt 2026 data into its schema.
  Max fidelity, near-zero port risk, but medium data-glue effort + 2025 coupling. Held as
  the **fidelity-escalation path** if Phase E LB underperforms or for the full-FT/PL round.

**Empirical arbiters (we don't decide A-vs-C in the abstract):** Gate 2 OOF AUC ≥0.95
(cheap, no LB) validates the head+pipeline on fixed features; Phase E first LB (0.91-0.92
band expected) validates the whole thing. If either underperforms, escalate the diverging
piece toward Sydorskyi (Option C / selective swaps using the ported transforms) and A/B
against LB. Offline metrics anti-correlate with LB historically, so Phase E LB is the
real arbiter.

## Training-loop audit vs prior experiments (2026-05-28)

Cross-checked `XCPretrainFinetuneModule` + `train.py` against exp 021 (SED),
exp 032/033 (distilled-SED, whose datamodule we reuse), and Sydorskyi's
`selected_ebs.py` / `train_lightning.py`, plus the relevant feedback memories.

| Knob | exp 021 SED | exp 032/033 | Sydorskyi | **exp 034** | Verdict |
|---|---|---|---|---|---|
| Optimizer | AdamW wd=0.01, 2 groups | AdamW lr=5e-4 wd=1e-4 | AdamW lr=1e-4 eps=1e-8 | AdamW lr=1e-4 eps=1e-8 | ✅ Sydorskyi-faithful |
| Scheduler | Cosine+warmup / OneCycle | Cosine + 2ep warmup, per-step | Cosine, **no warmup**, per-step | Cosine, no warmup, per-step | ✅ faithful |
| max_epochs | 50 | 25 | 50 | 50 (Gate2=5) | ✅ (=feedback_bump_epochs + Sydorskyi) |
| grad_clip | none (cosine path) | 1.0 | **none** (trainer_params={}) | **none** (opt-in `--grad-clip`) | ✅ faithful |
| Precision | config | 16-mixed | **32-true** (no AMP) | **32-true** | ✅ faithful |
| Loss frame-aggregation | logit | logit | **logit** (`clipwise_logits_long`) | **logit** | ✅ correct — avoids the prob-space pitfall (`feedback_atthead_aggregation_domain`) |
| Val metric | TuckerMacroAUC, prob blend σclip+σframe | TuckerMacroAUC, prob blend 0.5/0.5 | ROC-AUC on `clipwise_pred_long` (clip-only prob) | **TuckerMacroAUC on `clipwise_pred_long`** | ✅ faithful + same metric class as exp 032 |
| num_workers | — | 8 | 12 | 8 | ✅ (`feedback_num_workers_default`) |
| Val set | held fold | labeled-SS held fold | held-fold full-file | labeled-SS held fold (reuse) | ✅ consistent w/ exp 032 |
| Mixup/sampler | additive-hard | additive-hard + MixSamp | sum-clip + class-bal + 40% PL swap + BG | exp 032 datamodule (additive-hard + MixSamp) | ⚠️ data divergence — see § B.8.3 (low-risk; escalation path) |

**Conclusion: no training-loop bugs.** Every load-bearing knob is correct and
Sydorskyi-faithful; the deltas vs exp 032 (no grad-clip, 32-true, no warmup, 50 ep)
are deliberate moves toward Sydorskyi, and all sit within our codebase's normal
range (exp 021 also runs 50 epochs and a per-step schedule). The loss aggregates in
LOGIT space and the metric reads `clipwise_pred_long` — the correct side of the known
prob-space-aggregation issue. The only divergence is the DATA pipeline (§ B.8.3).

**Two Phase-E watch-items (not bugs, not Gate-2 concerns):**
1. **Checkpointing:** Phase E should use `ModelCheckpoint(save_top_k=-1, save_last=True)`
   and select **`last.ckpt`** — matches Sydorskyi (last), exp 032's full-data path, and
   `feedback_save_top_k_does_not_preserve_last`. (Gate 2 needs no ckpt; reads val/auroc.)
2. **Inference blend:** exp 021/032 blend σ(clip)+σ(frame) at inference; Sydorskyi/exp 034
   score clip-only. When wiring exp 034 into the exp 010 submission pipeline (Phase E.3),
   keep inference scoring consistent with training (clip-only `clipwise_pred_long`), or
   deliberately A/B a blend — avoid a silent train/infer mismatch.

## LB results

_(none yet — Phase C/D are offline gates; first LB is Phase E.)_
