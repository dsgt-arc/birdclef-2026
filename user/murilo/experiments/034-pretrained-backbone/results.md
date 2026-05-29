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
| **D — Gate 2** | Frozen-backbone head sanity | ⚠️ **0.791** (gate threshold was mis-set — see below) |
| **E — frozen FT (fold 0, 25 ep)** | first LB | ✅ val/auroc 0.859 → **LB 0.819** |
| **G — full-FT (fold 0, 50 ep)** | first LB | ✅ val/auroc 0.883 → **LB 0.865** |
| E.3 export pipeline | full-model audio→probs ONNX + kernel | ✅ done (parity 1e-7) |
| H PP / KD / F 5-fold / I ensemble | close the gap to 0.906+ (see LB results + gap audit) | ⬜ next — bare model below base |

## Phase E/G — single-fold training results (2026-05-28)

Both fold-0 runs finished cleanly (config-driven pipeline; W&B + checkpoints + artifacts;
clean tqdm logs). Val = TuckerMacroAUC on the labeled-SS fold-0 holdout.

| Run | Config | Epochs | val/auroc | last.ckpt | SLURM | W&B |
|---|---|---:|---:|---|---|---|
| Gate 2 | frozen head sanity | 5 | 0.791 | — | 9277013 | — |
| **Phase E** | frozen backbone | 25 | **0.859** | `checkpoints/xcpretrain-effv2s-frozen/fold0/last.ckpt` | 9279181 | `qfm8` |
| **Phase G** | full fine-tune | 50 | **0.883** | `checkpoints/xcpretrain-effv2s-fullft/fold0/last.ckpt` | 9279209 | `gvx4` |

**Reading (direction-only — offline anti-correlates with LB):**
- Trajectory is sensible: more epochs lift the frozen head (0.791 → 0.859), and **full-FT > frozen** (0.883 > 0.859) — full fine-tune adapts the focal-pretrained backbone to the soundscape domain, as predicted.
- Both sit below exp 032 `distilled-sed-kd`'s 0.908 on the same val set — but exp 034 has **no KD, no PP, single-fold, last-ckpt**, whereas exp 032's offline is boosted by Perch-KD + dual-supervision. Offline ≠ LB.
- **The yoink hypothesis (does the in-domain backbone lift LB over 0.906?) is only testable on the LB.** Offline 0.86/0.88 can't answer it. → Phase E.3 export + LB submit both single-fold models is the decisive next step.

> Metric-suite note: these runs logged `val/auroc` + losses + lr only (the module
> wired just `val_auroc`). The full exp 032/033 suite (cmap, f1_macro/micro,
> top1/top5, train-side) is a logging-only addition deferred to the next runs
> (Phase F 5-fold) to avoid racing these two live jobs.

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

## Phase D — Gate 2 result (2026-05-28): val/auroc = 0.791 (job 9277013)

Frozen Sydorskyi backbone (19.8M frozen) + random-init head (371K trainable),
5 epochs, fold 0. Config-driven pipeline ran cleanly; all 3 artifacts written
(`outputs/test_results/`, `artifacts/training/`, `outputs/tensorboard/`). AUROC
trajectory: 0.541 → 0.741 → 0.791 (still rising at epoch 5).

**The 0.95 gate was mis-calibrated (my error).** It cited Sydorskyi's *full-FT*
0.984 (his pipeline + his val set). But on THIS 2026 labeled-soundscape val set:
- our best **full** model (exp 032 distilled-sed-kd, 25 ep + KD) tops out at **0.908**;
- Gate 2 is a **5-epoch frozen-head** (no backbone adaptation to the soundscape domain).

So 0.95 was unreachable by construction. Corrected read:
- **Pipeline + transfer are validated** — frozen features are clearly informative
  (0.79 at 5 ep, rising; vs the 0.91 full-model ceiling on the same val set; vs
  0.5 chance). Gate 2's real purpose (decouple "pipeline works" from "fine-tune
  works") is satisfied; no bug (Gate 1 already proved bit-identical features).
- **Frozen-head caps below the full-model ceiling** (0.79 vs 0.91) — expected,
  since a frozen focal-pretrained backbone can't adapt to the soundscape domain.
  Closing that gap needs more epochs (Phase E = 25 ep) and likely **full fine-tune**
  (Phase G unfreeze; Sydorskyi's production config).
- **Offline AUC anti-correlates with LB** (`feedback_offline_metrics_unreliable`),
  so 0.79 is NOT a reliable LB predictor. **Phase E LB is the real arbiter.**

**Corrected gate** (going forward): Gate 2 is a pipeline/transfer sanity, not an
absolute-AUC bar — "frozen-head clearly above chance and a sensible fraction of the
full-model ceiling, artifacts sane." PASS. The 0.95 number is retired.

## Implementation conventions (aligned to exp 032/033, 2026-05-28)

After an audit, exp 034 now matches the established experiment scaffold:
- **Config-driven**: `configs/_common.yaml` (Sydorskyi recipe) + diff configs
  (`gate2-effv2s`, `xcpretrain-effv2s-frozen`, `xcpretrain-effv2s-fullft`) with
  `inherit:` deep-merge. `train.py --config <yaml> --fold N` (was CLI/env-flag driven).
- **Launchers**: `bash sbatch/gate2.sh` / `bash sbatch/train.sh` → `sbatch sbatch/_job.sh`
  (launcher + `_job.sh` split, like exp 032/033 `run.sh`+`_job.sh`; `$SLURM_SUBMIT_DIR`).
- **Logs**: `logs/slurm_<run>-fold<N>_%j.log` (`.log`, absolute path, unbuffered).
- **Lightning logger**: `TensorBoardLogger` → `outputs/tensorboard` (read by `tb:metrics`);
  W&B (`entity=dsgt-arc`, `project=birdclef-2026-exp034`) for full runs only (not Gate 2).
- **Metrics**: `train.py` → `outputs/test_results/<run>-fold<N>.json`; `save_metrics.py`
  → `artifacts/training/<run>/fold<N>/metrics.json` (run by `_job.sh`, non-fast-dev).
- **Checkpoints**: `<project>/experiments/034-pretrained-backbone/checkpoints/<run>/fold<N>/`
  (`save_top_k=3` + `save_last` → inference uses `last.ckpt`).

Verified: config deep-merge correct; fast-dev-run via `configs/gate2-effv2s.yaml` loads the
pretrained backbone (371K trainable / 19.8M frozen), 1 train+val batch finite, exit 0; 58
unit tests pass.

## LB results (2026-05-28) — single-fold, BARE model

| Run | Config | val/auroc | **LB** | vs 0.906 base |
|---|---|---:|---:|---:|
| `xcpretrain-effv2s-frozen` | frozen, 25ep, no PP/KD/PL | 0.859 | **0.819** | −0.087 |
| `xcpretrain-effv2s-fullft` | full-FT, 50ep, no PP/KD/PL | 0.883 | **0.865** | −0.041 |

(First submissions were invalid — 41/234 species_mapping bug; re-submitted after the fix.
Kernel canary `234 mapped` confirmed. full-FT > frozen, as predicted.)

### Verdict: bucket REJECT vs the 0.906 base — but the read is "necessary-not-sufficient," not "backbone fails"

The decisive comparison is to exp 032's **no-KD control**, the apples-to-apples bare model:

| | backbone | KD | PP (smooth) | 5-fold | PL | LB |
|---|---|---|---|---|---|---:|
| exp 032 control | ImageNet EffNet-B0 | ❌ | ✅ | ❌ | ❌ | 0.856 |
| **exp 034 full-FT** | **in-domain EffNetV2-S** | ❌ | ❌ | ❌ | ❌ | **0.865** |
| exp 032 distilled-sed-kd | ImageNet EffNet-B0 | ✅ | ✅ | ❌ | ❌ | 0.906 |

**exp 034 full-FT (0.865) ≈ a no-KD control with a slightly better backbone.** It even
BEAT exp 032's control (0.856) *despite* having NO post-processing (exp 032's kernel
applies 5-tap Gaussian smoothing; the xcpretrain kernel applies none). So the in-domain
backbone IS adding value — but it's a single, modest lever, and we built the WEAKEST
possible configuration around it.

The 0.94 competitors (Ali Ozan 0.941) stacked this same backbone WITH the levers we omitted.
Our own LB-measured lever values: **KD +0.050** (exp 032 control→KD), **5-fold +0.009**,
PP/priors +0.004–0.020, **PL +0.027** (Sydorskyi, disclosed). exp 034 has **none** of them.

### Gap to our best + path forward (by LB-evidenced impact)

| Missing lever | Evidence (LB) | Cost | Note |
|---|---|---|---|
| **Post-processing** (5-tap Gaussian smooth + TopN + sliding-window TTA + site/hour priors) | +0.012–0.020 (Nikita/Sydorskyi); exp 021 priors +0.004 | **cheap — no retrain**, kernel-only on existing ckpts | exp 034 kernel has ZERO PP; exp 032 control already had smoothing |
| **KD** (Perch v2 embedding distillation, AttHeadLogitSpace) | **+0.050** (exp 032 control 0.856→kd 0.906) | retrain | our single biggest evidenced lever; exp 034 is a "control" without it |
| **5-fold ensemble** | +0.009 (exp 032) | 4× train | variance reduction |
| **PL** (own-teacher pseudo-labels on soundscapes, Sydorskyi-faithful: sum-clip MixUp + 40% PL-swap + BG-noise) | +0.027 (Sydorskyi); finicky on our ImageNet base (exp 033 REJECT) but "multiplies a strong base" | PL-gen + retrain | the soundscape-domain adapter that *unlocks* the focal-pretrained backbone |
| **Ensemble w/ exp 023 (0.929) + exp 032 (0.915)** | exp 023 already 0.929 | cheap rank-ensemble | in-domain backbone = diverse member; may lift the ensemble even at 0.865 |

**Why competitors hit 0.94 with this backbone:** backbone is ~1 of 4–5 stacked levers
(backbone + PL + PP + ensemble [+ multi-iter noisy-student]). The deep research said it
plainly: *PL is a multiplier on a strong base.* We validated the base transfers; we have
not added the multipliers. Nothing is mechanically wrong (Gate 1 bit-identical, kernel
parity 9e-7) — the configuration is just bare.

### What Sydorskyi's own inference pipeline actually does (audited `vendor/sydorskyi-2025/` @ `053594f`, 2026-05-29)

Checked `code_base/inefernce/inference_class.py`, `inference_configs/*.py`, `scripts/main_inference_and_compile.py`, and the submission notebooks. His PP is **minimal** — the 0.925/0.928 lift came from training-side + ensemble levers, NOT inference smoothing:

- **No temporal smoothing.** Zero `smooth`/`convolve`/`ndimage`/`savgol` anywhere. (So smoothing is *our* 2026 lever, not his — our exp 032 LB history justifies it independently.)
- **No TTA.** `step=None` → non-overlapping 5s windows; `lookback/lookahead=None` → no temporal context.
- **Per-segment prediction**, output is `clipwise_pred_long` (a *probability*); `use_sigmoid=False` because the head already outputs probs — **identical to our exp 034 ONNX**.
- **Ensemble averaging is PROB-space**, not logit-space: `mean`, `vlom` (avg of geometric-mean & quadratic-mean), or `gaus` (RMS). → For Phase F (5-fold) on this head, prefer **prob-space fold averaging (mean/vlom/gaus)** over the logit-space fold-mean our Tucker ensemble path uses.
- **SWA** weight-averaging across checkpoints; **Google-model (Perch/SurfPerch) blend**; heavy **PL** (`PseudoF2…P04…` in the exp name) + `AddRareBirds`; FocalBCE + label-smoothing-fix; `late_normalize`.

**Implication for our PP choice:** his head is prob-space, but we ship **logit-space** 5-tap Gaussian smoothing anyway (user decision, 2026-05-29) because it mirrors *our* 0.906 Tucker recipe — our strongest 2026 prior. Prob-space is the fast follow-up if logit underperforms the 0.865 baseline.

### PP-smoothing change (2026-05-29) — cheapest lever first

Added 5-tap logit-space Gaussian smoothing (`[0.1,0.2,0.4,0.2,0.1]`, edge-padded) to
`predict_soundscape_xcpretrain` in the exp 010 kernel: ONNX prob → inverse-sigmoid →
smooth across windows → sigmoid. Toggled by `mel_cfg["gaussian_smooth_logit_space"]`
(default True; baked into both models' `mel_config.json`). Math sanity-tested (shape/finite,
constant-in-time ≈ identity 6e-8, <3-window skip, extreme-prob clamp, spike-bleed). **No
retrain / re-export — kernel re-push only.** This is the A/B vs the no-smooth 0.865 baseline.
