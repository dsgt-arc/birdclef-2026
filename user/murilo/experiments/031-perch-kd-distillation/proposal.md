---
experiment: "031-perch-kd-distillation"
status: approved
date: 2026-05-09
author: Murilo Gustineli
---

# Proposal: Perch v2 Knowledge Distillation on v5j (Tucker recipe, Phase 1)

**Status**: approved (2026-05-09)
**Created**: 2026-05-09
**Author**: Murilo Gustineli

## Hypothesis

Adding Perch v2 embedding-space knowledge distillation as a single-variable
change on top of the v5j_control_v2 recipe (the deterministic-reproducible
seed=42 single-fold v5j config from exp 030 → LB 0.896) will move LB by
**+0.014 to +0.020**, landing the single-fold checkpoint in the **0.910 –
0.916** range.

The mechanism: the backbone is forced to match Perch v2's 1536-d clip
embedding (via MSE), while the LSEHead trains on **stop-gradient-detached**
features — so the backbone's only optimization signal is "produce a
Perch-aligned representation." This engages the 04-25 label-noise mechanism
because the supervision is now on a *representation* (Perch's, distilled
from 14,795 species at scale), not on the noisy recording-level labels.

### Why we read against v5j_control_v2 (0.896), not v5j-original (0.906)

Per exp 030: a single-fold deterministic retrain of the v5j config today
(`v5j_control_v2`) lands at LB **0.896**, not 0.906. The 0.010 gap is
training-side (best-by-val plateau ep 14-15 in new runs vs ep 19 in
v5j-original; `save_top_k=1` doesn't serialize ep 19). The
`v5j-resubmit` diagnostic (04-28) confirmed the inference pipeline today
reproduces v5j-original's LB exactly when given the original ep-19
checkpoint — so the gap is NOT inference-side, it's training-side.

Phase 1 is structurally a single-variable add of KD on top of
v5j_control_v2's training pipeline, not a re-creation of v5j-original's
ep-19 state. The right comparator is therefore v5j_control_v2 = 0.896,
not v5j-original = 0.906. Reading against the wrong baseline would
under-credit a real KD lift by 0.010.

**Falsifiable** (corrected baseline):
- **CONFIRM if** single-fold LB ≥ **0.910** (≥ +0.014 over 0.896 baseline,
  i.e. outside the ±0.013 single-fold seed band measured in exp 030).
- **REJECT if** single-fold LB ≤ **0.896**, i.e. KD didn't add anything
  on top of the no-KD baseline.
- **PARTIAL** if LB 0.903 – 0.909: lift detectable but under +0.014 → diagnose.
- **WEAK** if LB 0.897 – 0.902: lift inside seed noise, can't conclude → diagnose.

## Background

### What this experiment tests

The 2026-05-09 audit of Tucker Arrants's published `bc2026-distilled-sed`
notebook
([note](../../analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md))
confirmed three things about his publicly-quantified +0.02 LB lift:

1. **Embedding-space MSE distillation** — `dist_loss = F.mse_loss(distill_emb,
   perch_emb)` where `distill_emb = Linear(GAP(backbone_features))` and
   `perch_emb` is Perch v2's frozen 1536-d output for the same 5s
   waveform.
2. **Stop-gradient on the SED branch** — `feat_for_head = feat.detach()`
   when KD is on. Backbone updates come exclusively from `dist_loss`. SED
   head trains on a "frozen" (from its perspective) feature.
3. **The classification side stays simple** — BCE on the ground-truth
   labels through an off-the-shelf SED head. No KLD on logits, no
   teacher-classifier-output supervision; the teacher is a *representation
   provider*, not a label provider.

This is structurally distinct from what we tried in exp 014/015/025 (PL on
unlabeled SS, "category #1" in
[`project_kd_categories_2026_05_09.md`](../../../../.claude/projects/.../memory/project_kd_categories_2026_05_09.md)).
Tucker's recipe is "category #3" — KD on labeled, embedding space, with
two-headed gradient flow. Never tried by us.

### Why we expect transfer to our codebase

| Argument | Direction |
|---|---|
| Tucker quantifies +0.02 LB on his pipeline (2026-05-09 thread quote: "without distillation around 0.89, with around 0.91") | Positive |
| Three independent data points cluster around the same effect: Tucker (~+0.02), chenwenqiang_001 own-teacher (+0.013), EliKal published 0.909 with KLD-distillation | Positive |
| Our v5j is at 0.906 LB. Tucker's "without KD" is ~0.89, "with KD" is ~0.91 — relative offset suggests his "with KD" lands ≥ our 0.906 | Positive |
| 04-25 per-frame finding (`v5j is structurally honest on hard crops`) implies the lever is *more information*, not training-recipe tweaks. Embedding KD is an information lever (Perch was trained on 14,795 species, dwarfing our 234). | Positive |
| Our prior PL attempts (exp 014/015/025) failed → could mean we have a structural issue with teacher-student setups; or could mean only category #1 was broken, categories #2/#3 untested | Uncertain |
| Tucker's Perch teacher mel != our student mel (he uses torchaudio MelSpectrogram + AmpToDB; we use NNAudio + AmpToDB) — small distribution mismatch may bound the dist_loss floor | Uncertain |
| Single fold has ±0.013 seed band on HGNet (exp 030) → if the lift is really +0.014, it's at the noise floor | Uncertain |

### Why we picked Phase 1 = single fold

Cheapest test of "does the lever transfer." If LB ≥ 0.920, KD is real and
we go to Phase 2 (4-fold ensemble OR 384-d teacher variant). If LB <
0.906, KD didn't transfer — diagnose before spending more compute. Single
fold gives us a yes/maybe/no decision in ~2-3 hours wall time and 1 LB
submission slot.

4-fold from the start would cost 4× training time + 4× ONNX exports + 4×
Kaggle dataset uploads. Justifiable if we expect a clear positive; not
justifiable while we're testing whether the lever transfers at all.

### Why we're using Tucker's published Perch ONNX as the Phase-1 teacher

Three options (full discussion in
[code audit](../../analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md)):

- **A: Vendor `tuckerarrants/perch-v2-no-dft-onnx`** — fastest, identical
  teacher to Tucker's, same embedding distribution.
- **B: Export our own Perch v2 no-DFT ONNX** — owned by us but adds
  several hours of careful work and risks introducing a teacher mismatch.
- **C: PyTorch Perch teacher** — keeps everything GPU-resident, no
  cpu↔gpu round-trip per batch, but requires deeper integration.

**Phase 1 uses Option A.** Phase 2 ports to Option B (and possibly C) for
ownership of the full submission pipeline before any "live" submissions.

This is a deliberate diagnostic-first choice: if Phase 1 fails with
Tucker's exact teacher, we know KD didn't transfer to our codebase; we
don't have to debug "did I export Perch correctly?" simultaneously. If
Phase 1 succeeds, we then pay the ownership cost knowing it's worth it.

### Cross-references

- Code audit: `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`
- Discussion summary: `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md`
- Tucker's notebook (vendored): `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`
- Strategic context: `user/murilo/analysis/notes/2026-04-28-next-experiment-direction.md` (ADDENDUM)
- Memory: KD categories taxonomy, AttHead aggregation-domain bug

## Method

### Approach

Single-variable add of Perch v2 embedding-space KD on top of v5j. Train
one HGNetV2-B0 + LSEHead + BCE model with the distillation branch enabled.
Submit to LB. Compare to v5j-original 0.906 baseline. Use Tucker's
published Perch ONNX as the teacher for Phase 1; defer ownership of the
Perch export to Phase 2.

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 focal (~35,549 files) + labeled soundscape segments (~1,478 segments). 90/10 split, `seed=42`. Same data as v5j_control_v2. |
| Compute | 1 SLURM job on rtx6000 (or v100 if rtx6000 queue is full). ~40-60 min training (single backbone, 20 epochs). Phase 0 fast-dev-run on the same partition. |
| Dependencies | timm (HGNetV2-B0), nnAudio, onnxruntime-gpu (for Perch teacher), Tucker's published ONNX dataset `tuckerarrants/perch-v2-no-dft-onnx` |
| Code | New module `DistillHead` + `PerchTeacher` in `user/murilo/src/birdclef/lightning/sed_finetune.py`. New config `031-perch-kd-distillation/configs/v5j_kd.yaml`. |

### Procedure

**Phase 0 — implementation prerequisites (no GPU training)**

0a. **Mel-pipeline equivalence diagnostic**: numerically compare our NNAudio
   mel (`hop_length=313`, etc.) against torchaudio MelSpectrogram with
   Tucker's exact config (`hop_length=512`, etc.) on a fixed 100-sample
   batch of training waveforms. Compute: per-pixel max abs difference in
   dB space (after AmpToDB), mean abs difference, **per-sample cosine
   similarity** (the load-bearing metric), frame-count difference, and
   the same metrics on a "STFT/library only" comparison (matched hop, no
   normalization) to isolate library numerics from hop-length and
   normalization choices.
   - **Decision gate (corrected after 2026-05-09 first-run analysis)**:
     The original gate ("max abs diff > 5.0 dB → refactor") proved
     unreliable because the abs-diff metric is dominated by global offset
     differences from `power → dB` conventions and post-normalization
     scale choices, NOT by signal disagreement. The corrected gate uses
     cosine similarity:
     - **Library agreement** (STFT-LIB-ONLY comparison, matched hop, no
       norm): cos sim ≥ 0.95 → libraries agree on STFT and mel filter
       bank, no library switch needed. cos sim < 0.95 → genuine library
       disagreement, switch to torchaudio.
     - **Hop length**: any choice is fine as long as the teacher takes
       **raw waveforms** (which Perch v2's no-DFT ONNX does — it computes
       its own mel internally). Different hops produce different student
       feature-map shapes but don't break the KD math because the
       projection layer bridges to Perch's clip embedding regardless.
     - **Post-normalization**: any per-sample standardize and/or min-max
       is fine. These are linear remappings that the backbone's first
       BatchNorm absorbs.
   - Document findings in `phase0_mel_equivalence.md`. **Phase 1 always
     runs the same v5j mel pipeline (NNAudio + hop=313 + AmpToDB +
     standardize + min-max)** unless the corrected gate triggers a
     library switch. This preserves single-variable discipline vs v5j.
   - **Runtime check carried into Phase 1**: monitor `dist_loss` curve.
     Tucker's published floor is ~0.003 by epoch 25. If our floor is
     significantly higher (> 0.01 by epoch 5, > 0.005 by epoch 25),
     mel-pipeline mismatch is contributing — document and consider
     refactor as a Phase 2 diagnostic.

1. Vendor Tucker's published Perch ONNX:
   `kaggle datasets download tuckerarrants/perch-v2-no-dft-onnx`. Place
   under `vendor/kaggle/datasets/perch-v2-no-dft-onnx/`. **No re-export.**
2. Add `DistillHead(nn.Module)` class to `sed_finetune.py`. GAP over
   `dim=[2, 3]` of backbone features → Linear projection to 1536-d.
   Direct port of Tucker's notebook (cell 10).
3. Add `PerchTeacher` class to `sed_finetune.py`. ONNX inference via
   `onnxruntime`, `@torch.no_grad()`, returns `(B, 1536)` float32.
4. Modify `SEDFinetuneModule` forward path to support a
   `use_perch_distill` flag:
   - When `True`: compute `feat = backbone(mel)`, then both
     `distill_emb = self.distill_head(feat)` (gradient flows to backbone)
     AND `feat_for_head = feat.detach()` for the LSEHead branch (no
     gradient to backbone from BCE).
   - When `False`: existing behavior unchanged.
5. Modify the loss path: `loss = cls_loss + alpha_distill * distill_loss`
   where `distill_loss = F.mse_loss(distill_emb, perch_teacher.embed(wav_5s))`.
6. Fast-dev-run on a 100-sample subset. Verify:
   - `dist_loss` decreases over a handful of steps.
   - `cls_loss` still trains normally.
   - Backbone gradient norm is non-zero (driven by `dist_loss`).
   - SED head gradient norm is non-zero (driven by `cls_loss` through
     detached features).
   - No NaN/Inf in either branch.

**Phase 1 — first KD training run**

7. Create `031-perch-kd-distillation/configs/v5j_kd.yaml`. Identical to
   `030-sydorskyi-mixup/configs/v5j_control.yaml` (the deterministic-baseline
   config that reproduces v5j-original training trajectory) except:
   - `use_perch_distill: true`
   - `perch_onnx_path: <vendored path from Phase 0 step 1>`
   - `perch_embed_dim: 1536`
   - `alpha_distill: 1.0`
   - `run_name: 2026-exp031-v5j-kd`
8. Submit SLURM job. Monitor `dist_loss` curve in W&B; expect monotone
   decay with floor ~0.003 by epoch 25 (per Tucker's published logs).
   **Halt if `dist_loss` plateaus above 0.01 by epoch 5 OR if `cls_loss`
   diverges** — indicates Perch teacher / mel mismatch issue worth
   diagnosing before continuing.
9. After training completes, ONNX-export the LSEHead-only inference
   path (the distillation head is discarded at inference). Verify ONNX
   output is `(B, 234)` non-degenerate on a fixed random mel input.
10. Upload weights to Kaggle (new dataset slug
    `birdclef-2026-hgnet-b0-sed-v5j-kd-weights`), push kernel.
11. Submit kernel to LB. Wait for score.

**Phase 2 (gated on Phase 1 verdict — defer if Phase 1 rejects)**

12. **Ownership work (B/C from P0.1)**: Export our own Perch v2 no-DFT
    ONNX (Option B) OR wire PyTorch Perch teacher (Option C). Validate
    that the new teacher's `embed(wav)` returns vectors with `cosine
    similarity > 0.999` to Tucker's ONNX on a held-out 100-sample subset.
    If similarity < 0.999, diagnose mel-spec or weight differences before
    proceeding.
13. Re-train `v5j_kd_v2` with our owned Perch teacher (single fold,
    seed=42). Confirm LB lands within ±0.005 of Phase 1 LB. If yes,
    ownership is established; if no, diagnose teacher difference.
14. **Multi-fold scaling**: 4-fold version of the winning Phase-1 config.
    4 SLURM jobs + 4 ONNX exports + 4 Kaggle uploads + 1 LB submission
    of the ensemble.

### Variables

- **Independent**: presence/absence of the Perch v2 embedding KD branch
  (`use_perch_distill: true` vs `false`) AND the `feat.detach()` on the
  LSEHead's input. These two are tied — KD on means detach on; KD off
  means standard gradient flow. Cannot ablate independently in this
  experiment.
- **Dependent**: public LB score (primary), val/auroc, val/cmap,
  val/f1_macro, dist_loss curve, training walltime.
- **Controlled**: backbone (HGNetV2-B0 `hgnetv2_b0.ssld_stage2_ft_in1k`),
  head (LSEHead, hidden=1024), loss (BCE), optimizer (AdamW lr=5e-4
  OneCycleLR), epochs (20), seed (42), data (90/10 split), precision
  (16-mixed), mel pipeline (NNAudio + AmpToDB, n_mels=256, n_fft=2048,
  hop=512), mixup (spec-domain Beta(1,1) interpolated, theta=0.8 cap,
  warmup=5, p=1.0 — v5j default), no `dual_validation`, no SpecAugment,
  no BG noise injection, no auxiliary augmentations. **The only thing
  that changes vs v5j-original is the addition of the KD branch.**

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **public LB** | Primary. Single-fold v5j_kd checkpoint submitted via Kaggle kernel. Compares directly to v5j-original 0.906. |
| `dist_loss` curve | Diagnostic. Should decay monotonically. Floor expected ~0.003 by epoch 25 (Tucker's published number). Plateau > 0.01 suggests mel-spec mismatch or teacher hookup issue. |
| `val/auroc` | Diagnostic only. Per FINDINGS.md, val/auroc is **anti-correlated with LB** for cross-architecture comparisons. Used here as a sanity check the model trained, not as an LB predictor. |
| `val/cmap` + `val/f1_macro` | Diagnostic. Have given directionally-cleaner LB signals than val/auroc in exp 030 (cmap +0.020 backbone main effect was confirmed on LB). |
| Backbone gradient norm | Implementation check during Phase 0 fast-dev-run. Confirms KD loss actually drives the backbone (rather than the cls loss leaking through some bug). |
| Training walltime | Operational. Tucker reports ~60s/epoch with KD vs ~30s/epoch without on G4. On rtx6000 with ONNX teacher, expect ~2-3 min/epoch. |

### Baseline

- **Primary baseline**: **v5j_control_v2 LB 0.896** (exp 030). Same codebase,
  same config, same seed, same single fold, same checkpoint-selection regime
  as Phase 1 will use — *minus the KD branch*. Structurally identical
  comparator. The v1↔v2 audit in exp 030 proved this trains
  bit-identically when only val-side flags change, so it is the most
  reproducible baseline available to us.
- **Reference point**: v5j-original LB 0.906 (exp 019). Inference pipeline
  reproduces this exactly via `v5j-resubmit` (exp 030, 04-28). The
  0.010 v5j-original → v5j_control_v2 gap is training-side (best-by-val
  plateau ep 14-15 in new runs vs ep 19 in v5j-original; `save_top_k=1`
  never serializes ep 19). NOT used as the comparator for the success
  criteria — but reported alongside for context.
- **External reference**: Tucker's "without PP" ≈ 0.89, "with PP" ≈ 0.91+
  (his thread quote). Phase 1 is single-fold + no Tucker-style PP (we use
  our existing pp2 inference path), so the closest external comparison
  point is "Tucker without PP" ≈ 0.89-0.91. He stacks 5-fold ensembling
  + site/hour priors + Gaussian smooth on top to reach 0.93+; we'd need
  Phase 2's 4-fold scaling + matching PP to chase that.

### Success Criteria

Read against **v5j_control_v2 = 0.896** baseline (the structurally-identical
no-KD comparator), with ±0.013 single-fold seed band per exp 030:

| Outcome | Definition | Action |
|---|---|---|
| **CONFIRM** | LB ≥ **0.910** (≥ +0.014 over 0.896, outside ±0.013 noise) | KD lever transfers to our codebase. Proceed to Phase 2 (ownership work + 4-fold). Investigate whether dist_loss floor matches Tucker's published 0.003. |
| **PARTIAL** | LB **0.903 – 0.909** (+0.007 to +0.013 over 0.896) | KD transferred but lift smaller than Tucker's +0.020. Most likely cause: mel pipeline mismatch raising dist_loss floor (NNAudio vs torchaudio + hop_length 313 vs 512). Diagnose: dist_loss floor, run Phase 0.5 mel-equivalence remediation. |
| **WEAK** | LB **0.897 – 0.902** (≤ +0.006 over 0.896, inside seed noise) | Effect indistinguishable from noise. Need 4-fold to escape seed band before drawing conclusions. Diagnose dist_loss curve before paying 4-fold compute. |
| **REJECT** | LB ≤ **0.896** | KD didn't add anything as a single-variable add. Either a bug in our Phase 0 implementation (verified by Phase 0.4 fast-dev-run gradient checks), or the lift requires Tucker's full stack (AttHead + dual-supervision + hard mixup + SpecAugment). Halt Phase 2; write a focused post-mortem identifying which structural difference is most likely responsible. |

### Bonus reference: if v5j-original ep-19 weights got KD treatment

Hypothetical only — we can't easily test this without a `save_top_k=-1`
v5j retrain. But for orientation: v5j-original is 0.906; if KD adds the
same +0.014 it adds to v2, ep-19-equivalent LB would be ~0.920, matching
Tucker's "without PP" range. This is the upper bound of what's plausible
on a single fold. Phase 2's 4-fold scaling is what should target the
0.917+ "with PP" zone.

## Limitations

- **`v5j_control_v2` baseline is 0.896, not v5j-original 0.906.** The
  0.010 LB gap between the two is training-side and unsolved (best-by-val
  plateau ep 14-15 vs ep 19; `save_top_k=1` doesn't serialize ep 19).
  We compare Phase 1 against v5j_control_v2 because that's the
  structurally-identical no-KD comparator on the same training pipeline.
  A `save_top_k=-1` retrain to recover the ep-19 0.906 baseline is
  deferred work — not blocking Phase 1.
- **Mel pipeline mismatch with Tucker is the largest known risk.**
  Comprehensive comparison in
  `v5j-vs-tucker-pipeline-comparison.md`. Two sub-issues:
  (a) NNAudio vs torchaudio MelSpectrogram (different STFT/filter-bank
  numerics with same params); (b) `hop_length=313` vs Tucker's `512`
  (1.6× temporal-resolution difference, ~512 frames per 5s vs ~313).
  Phase 0a is the diagnostic + remediation gate.
- **Tucker's pipeline differs from v5j on 7+ axes** (mel pipeline, head
  architecture, mixup target combination, dual-supervision loss,
  SpecAugment, schedule shape, batch composition sampler). Phase 1 holds
  v5j defaults for everything except KD — single-variable discipline.
  Possible the +0.02 KD lift Tucker reports interacts with one or more
  of those other components and won't transfer cleanly to our codebase.
  If Phase 1 lands in REJECT / WEAK, the comparison table is the menu
  of follow-up experiments.
- **Single fold has ±0.013 noise.** Phase 1 cannot disambiguate "lift
  +0.014" from "lift +0.001 + noise +0.013." Phase 2's 4-fold ensemble
  is the disambiguator. We accept this — Phase 1's job is yes/maybe/no
  signal, not precise effect size.
- **Tucker's recipe was developed end-to-end with multiple components**
  (AttHead, dual-supervision, SpecAugment, hard mixup, focal-SS mixup,
  90/10 batch composition). We're only adding the KD branch. Possible
  Tucker's lift compounds with his other choices; possible the +0.02
  *requires* his head architecture. If Phase 1 rejects, this is a
  primary diagnostic question.
- **Phase 1 uses a third-party (Tucker's) Perch ONNX.** We violate
  `feedback_own_datasets.md` for the *teacher* during Phase 1 only,
  intentionally to remove "is the Perch export correct?" from the failure
  surface. Phase 2 owns the full pipeline. The student weights and
  submission kernel are entirely ours either way.
- **`alpha_distill = 1.0` is Tucker's default.** Untested for our codebase.
  If Phase 1 lands in PARTIAL/WEAK, an `alpha_distill` sweep
  (0.1, 0.5, 1.0, 2.0) is a candidate diagnostic. Not in scope for
  Phase 1.
- **No SpecAugment in Phase 1.** Tucker uses 1 freq mask × 10 + 2 time
  masks × 10 by default. We don't. Tucker himself flagged that
  SpecAugment-on-student-but-not-teacher contributes to the dist_loss
  floor. By omitting SpecAugment, Phase 1 should have a *lower* dist_loss
  floor than Tucker's published 0.003, which is a free indication that
  the teacher hookup is working. Adding SpecAugment is a Phase 3+
  question.
- **`tf_efficientnet_b0.ns_jft_in1k` (Tucker's backbone) is not tested
  here.** HGNetV2-B0 is — because v5j is HGNet, and the single-variable
  discipline says: change KD only. Backbone comparison is Phase 3+.
- **Inference recipe held at v5j-default**: TTA + smoothing + TopN +
  priors. We do NOT add Tucker's 5-tap Gaussian smoothing / dual-
  supervision blend at inference. The LSEHead-only output uses our
  existing pp2 inference path. Direct comparison to v5j-original
  inference, no inference-side variables.
- **The MSE loss target dimension is fixed at 1536-d.** Tucker's 384-d
  cheap variant ("distill from Perch's last MBConv stage") is NOT tested
  here. Phase 2 candidate.
