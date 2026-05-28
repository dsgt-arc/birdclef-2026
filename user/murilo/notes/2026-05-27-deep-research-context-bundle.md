# Deep Research Context Bundle — Exp 033 Pivot Decision (2026-05-27)

This bundle concatenates the deep research prompt + 8 companion documents
into a single file for upload to Claude.ai. Each document is delimited by
a level-1 heading. The first section is the prompt itself; the rest are
the materials Claude.ai should read before answering.

**Instructions for Claude.ai**:
- Read the prompt in § 1 first.
- Reference the companion docs in § 2–9 as needed to answer.
- Output a markdown document with the structure requested in § 1.
- Be willing to tell me I'm wrong. Don't validate; assess.

---


# § 1. THE PROMPT

# Deep Research Prompt — Pivot Decision after Exp 033 REJECT (2026-05-27)

**Purpose**: Independent assessment of the BirdCLEF 2026 pivot decision after exp 033's REJECT verdict. Designed to **challenge** the project's current direction (Sydorskyi backbone yoink), not validate it.

**How to use**: Paste the body below into Claude.ai's deep research interface. Attach the listed companion docs (or paste their contents). Fire it overnight; review the output tomorrow morning along with the a2_kd_v3 LB result.

**Why this prompt is shaped this way**: Our prior research docs (2026-04-13 Claude + Gemini, 2026-04-11 Claude) all answered "what should I try?" — they were strategy-from-zero. We now have specific failures (exp 033 REJECT + 24 data points of offline-LB anti-correlation) and a specific candidate pivot (Sydorskyi-yoink). This prompt asks for independent ranking against that landscape, not new ideas from scratch.

---

## The prompt body (paste this into Claude.ai's deep research)

I'm in the BirdCLEF 2026 Kaggle competition. **7 days remaining** before the 2026-06-03 deadline. **5 LB submissions/day** quota, with 2 already used today on diagnostic runs.

My goal: break **LB 0.94** with the highest probability of success in the remaining time. Current best LB is 0.929 (`exp 023 perch-sed-ensemble-mlp-only`). Best single model is 0.906 (`exp 032 distilled-sed-kd` Phase D, fold 0).

I need an **independent assessment** — not validation of my current plan. Challenge the assumptions. Identify whether I'm wrong about anything before I commit to a 3-4 day pivot.

## What I've done (verbatim, including failures)

1. **Exp 032: Tucker distilled-SED replication.**
   - EffNet-B0 NS-JFT + AttHeadLogitSpace + dual-supervision BCE + Perch v2 embedding KD + Tucker mel + 5-fold.
   - Single fold LB 0.906 (Phase D), 5-fold ensemble LB 0.915, HGNet backbone ablation LB 0.901 (backbone refuted as the bottleneck).
   - Verdict: PARTIAL REPLICATION (within 0.002 of Tucker's published 0.917 single-fold).

2. **Exp 033: 1:1 own-teacher pseudo-labeling on distilled-sed-kd.**
   - 10 ablations: 2 teachers (v5j-4fold LB 0.913, distilled-sed-kd-5fold LB 0.915) × 5 protocol cells (init × sampler × focal/PL ratio).
   - All 10 trained to full max_epochs; D.5 e2e validation all 10 PASS.
   - 3 LB submissions, all REJECT (Δ ≤ 0 vs 0.906 baseline):
     - a4-kd (warm + Uniform + T2): LB 0.899 (Δ −0.007)
     - a2-kd ⭐ (random + Uniform + T2, canonical Antoine): LB 0.886 (Δ −0.020)
     - a4-v5j (warm + Uniform + T1): LB 0.897 (Δ −0.009)
   - Paired teacher-axis LB Δ (a4-kd vs a4-v5j): +0.002 (T2 over T1). Refutes the proposal's "teacher diversity" hypothesis at LB scale (T2 won offline by +0.011 auroc).

3. **Audit of exp 033 vs Sydorskyi 2025 2nd place protocol** (the only published 0.93+ PL recipe with line-by-line code).
   - **16 of 20 protocol axes diverge** between our implementation and Sydorskyi's published mechanism.
   - Top 5 high-impact divergences:
     - PL × focal MixUp fusion missing (Sydorskyi: focal hard + PL soft → 50/50 audio mix + sum-clip targets → implicit distillation. Us: PL flows as independent batch slots with no fusion. Sydorskyi explicitly calls this "the cleverest piece" of the pipeline.)
     - Trim probability missing (Sydorskyi: per-class probs < 0.1 zeroed before training. Us: temperature + power sharpening but no zero-out.)
     - Chunk discard missing (Sydorskyi: drop chunks where max(prob) < 0.5; ~83% discard rate. Us: kept all 127K chunks.)
     - 40% primary-class-matched swap sampling missing.
     - Focal loss missing (Sydorskyi: BCE + sigmoid_focal(α=0.25, γ=2.0); us: dual-sup BCE only).

4. **Phase 5 Test 1 launched**: re-issued PL parquet with chunk-discard + trim filters (v3 = 85K chunks, 33% discard rate, 98% per-cell trim). Training a2_kd_v3 right now. My honest prior on outcome: 50% PARTIAL recovery to LB 0.910-0.913 / 20% true PL benefit > 0.915 / 30% genuine REJECT (LB < 0.906).
   - **Important note**: even my best-case v3 outcome (CONFIRM) lands at ~0.92, not 0.94.

## What top teams disclosed (BirdCLEF 2026 discussion thread, 2026-05-27)

- **Sydorskyi 2025 2nd place (Public 0.925, Private 0.928)**: in-domain XC pretrain on 819K recordings + iterative semi-supervised distillation + TopN N=1 postprocessing. Ablations show pretrain = +0.018 Private, pseudo-labels = +0.020 Private, PP = +0.015 Private.
- **Ali Ozan Memetoglu (7th, LB 0.941)**: yoinked Sydorskyi's published 2025 backbone (`vladimirsydor/bird-clef-2025-all-pretrained-models`), attached own head (attention dim 1028, melspec match required to within ±0.012 LB), fine-tuned on 2026 data. No XC retraining.
- **Antoine Masq (33rd, LB 0.943 single, 0.948 ensemble)**: own XC pretrain + own-teacher PL + custom ASL + sliding-window TTA.
- **Boredom (2nd, LB 0.950+ single)**: timm + SED, non-Perch, 1-iter PL on unlabeled. Without PL: 0.938-0.939.
- **Salman Ahmed (43rd, LB 0.939 solo / 0.944 w/ PL / 0.952 ensemble)**: attention investigation breakthrough. EffNetV2-B0. PL adds +0.005 solo; primary value is enabling diversified ensembling.
- **Multiple confirmations on PL**: "doesn't work for more than one pass" (Anil Ozturk, Boredom, Zejun_, Salman). "Trim probability" (Ali Ozan, Jiacheng Ma) — confirmed our exp 033 was missing this.

## My implementation track record (the honest part)

Our codebase has had **three significant silent bugs** found post-hoc, all of which passed offline validation:
1. Exp 032 Phase 1: training/inference mel pipeline mismatch caused LB 0.522 collapse (offline showed nothing).
2. Exp 033: num_workers=0 inherited from Tucker's Jupyter notebook → 4× slower training (offline showed nothing).
3. D.5 validation script: fold_k hardcoded to 0; folds 1-4 evaluated against their own training data (gave false-PASS at 0.995).

**Pattern**: 24+ data points of offline-LB anti-correlation. Most recent: exp 032 fold 4 best offline (auroc 0.931, e2e 0.937) → LB 0.903 (near-worst of the folds submitted).

## The candidate pivot I'm leaning toward

**Sydorskyi-yoink (Path A)**: download `vladimirsydor/bird-clef-2025-all-pretrained-models`, attach my own head matching attention dim 1028 + melspec exactly (n_mels=128, n_fft=2048, hop=512, fmin=20, top_db=80, AmplitudeToDB → standardize → min-max), fine-tune on 2026 data. Validated to LB 0.941 by Ali Ozan and Jack. Estimated 3-4 days implementation.

## Questions I need answered

Be specific and provide reasoning, not just verdicts.

1. **Is the Sydorskyi-yoink pivot the right move, or am I confirmation-biased?** Consider: my audit was self-conducted, my probability estimates are self-estimated. What would a different framing recommend?

2. **What's the realistic LB landing band for Sydorskyi-yoink** on a competitor codebase like mine (3 prior pipeline bugs found post-hoc, anti-correlation pattern)? Should I expect 0.91-0.92 (Murphy's law on implementation), 0.93-0.94 (clean execution matches Ali Ozan), or 0.94-0.95 (additional ensemble lift)?

3. **What's the realistic LB landing for the "fix all 5 PL divergences" alternative?** If I instead spent the same 3-4 days correctly implementing Sydorskyi's PL protocol (focal × PL fusion, chunk discard, trim, 40% swap, focal loss) on top of our existing 0.906 distilled-sed-kd base, where does it land? Is the base too weak to extract PL signal regardless?

4. **Implementation risk enumeration for Sydorskyi-yoink.** Where are bugs most likely to land in this pivot? Mel mismatch, head dim mismatch, ONNX export, checkpoint key remapping, training-time class balancing? Rank by likelihood × LB impact.

5. **Pre-flight verification steps** for each path. How would I know I'd implemented Sydorskyi-yoink correctly before burning a LB submission?

6. **What unknown levers might I be missing?** Specifically:
   - Anything from BirdCLEF 2024 winners I haven't considered (Kefir 1st place)?
   - Any 2025/2026 papers on bird audio classification that propose orthogonal levers?
   - Any disclosed but underweighted lever in the BirdCLEF 2026 discussion threads?
   - Site/hour priors + Gaussian smoothing (Tucker's +0.005-0.010 lever) — is this big enough to matter as a closing move on whatever I land on?

7. **Concrete day-by-day plan** for the remaining 7 days (starting tomorrow, 2026-05-28). Should include:
   - Which path (Sydorskyi-yoink / PL fix / ensemble / other) to commit to.
   - Day-by-day milestones with realistic compute + LB-slot budgets.
   - Pre-flight gates between days (what triggers continuation vs pivot).
   - A specific "if X happens by Day N, pivot to Y" decision matrix.

## Output format

A markdown document I can save to my repo at `docs/references/deep-research/2026-05-27-pivot-assessment.md`. Sections:

1. Executive summary (3-5 sentences with the verdict).
2. Question-by-question answers (1-7 above) with sources cited.
3. Concrete day-by-day plan with milestones + decision gates.
4. Risk enumeration table (axis, what could go wrong, mitigation).
5. References (URLs, papers, Kaggle threads, code repos).

Be willing to tell me I'm wrong. Don't validate; assess.

---

## Companion documents to attach (paste content if Claude.ai can't read files directly)

| File | What it provides |
|---|---|
| `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` | Full 20-axis divergence table + audit (the document Question 1 should challenge) |
| `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` | Sydorskyi 2025 writeup (479 lines, line-by-line protocol detail) |
| `user/murilo/experiments/032-tucker-distilled-sed/results.md` | Tucker recipe results — confirms our pipeline can match 0.906 / 0.915 |
| `user/murilo/experiments/033-own-teacher-pl/proposal.md` + `results.md` | Exp 033 design + REJECT verdict |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | All competitor disclosures captured to date |
| `user/murilo/notes/kaggle-discussion-non-perch-limit.md` | Companion thread (non-Perch ceiling discussions) |
| `user/murilo/MODEL-RUNS.md` | Full experiment ledger (28+ runs) |

If Claude.ai's interface doesn't accept attachments, paste the content of each into the chat or summarize what's needed.

## Expected output

A markdown file at `docs/references/deep-research/2026-05-27-pivot-assessment.md`. Read it tomorrow morning alongside the a2_kd_v3 LB result. Together they drive the exp 034 commit/pivot decision.


---

# § 2. COMPANION DOC: Exp 033 ↔ Sydorskyi divergence audit

*Source: `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md`*

---
date: 2026-05-27
author: Murilo Gustineli
type: divergence-audit
related:
  - experiment: 033-own-teacher-pl
  - reference: docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md
---

# Exp 033 ↔ Sydorskyi 2025 Pseudo-Label Protocol Divergence Audit

**Purpose**: Determine whether exp 033's REJECT verdict (a4-kd 0.899, a2-kd 0.886, a4-v5j 0.897 vs baseline 0.906) reflects a real refutation of own-teacher PL, or implementation divergences from the only published 0.93+ PL recipe we have line-by-line code for (Sydorskyi 2025 2nd place, Private LB 0.928).

**TL;DR**: Our exp 033 is **structurally different** from Sydorskyi's published PL protocol. We have 5 high-impact divergences and 7 lower-impact ones. The most damaging is the **missing PL × focal MixUp fusion mechanism** — Sydorskyi's published protocol fuses the soft PL target into the focal target via summed-and-clipped MixUp targets at p=0.5, while our exp 033 lets PL samples flow through the model in their own independent batch slots with no fusion. Sydorskyi explicitly calls this fusion mechanism the "cleverest piece" of their pipeline. We did not implement it.

We also missed: trim probability (`< 0.1 → 0`), chunk discard (`max < 0.5 → drop`), 40% primary-class-matched swap sampling, and focal loss (BCE+Focal vs our dual-sup BCE). These four would each be roughly day-of-work fixes.

Recommendation: do NOT pivot from exp 033 yet. Reissue the PL parquet with trim + chunk discard (1 hour code + 30 min compute), then re-run a2-kd canonical cell. If still REJECT, *then* pivot. The current REJECT verdict is not yet a refutation of the protocol — it's a refutation of OUR version of the protocol.

---

## Context

- **Our exp 033 baseline**: distilled-sed-kd Phase D fold 0, LB 0.906 (no PL).
- **Sydorskyi 2025 published baseline**: EfficientNetV2-S with enhancements + in-domain pretrain, Public 0.881, Private 0.889 (no PL).
- **Sydorskyi PL gain**: +0.029 Public, +0.020 Private — single largest LB lever in their pipeline.
- **Our PL outcome (exp 033)**: −0.007 to −0.020 LB across 3 cells. **Δ-vs-Sydorskyi = ~0.049 LB worth of expected lift we did not achieve.**

The hypothesis we tested in exp 033 was the *Antoine* protocol ("1:1 own-teacher PL"), which Antoine described at a high level but did not publish code for. We synthesized our implementation from his discussion comments + a generic semi-supervised distillation framing. We did NOT use Sydorskyi's published code — even though it's the only repo with executable detail and the only writeup with quantified ablations.

This audit checks whether our synthesis matched Sydorskyi's actual published mechanism. Spoiler: it didn't.

## Sydorskyi 2025 protocol (canonical reference)

Source: `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` + paper [CEUR Vol-4038, paper_256](https://ceur-ws.org/Vol-4038/paper_256.pdf) + code [github.com/VSydorskyy/BirdCLEF_2025_2nd_place](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place).

### PL generation

```python
# Sydorskyi soundscape_pseudo_config:
{
    "primary_label_min_prob": 0.5,   # discard whole chunks where max(prob) < 0.5
    "trim_min_prob": 0.1,            # zero out per-class probs < 0.1
    "sampling_prob": 0.4,            # 40% chance per training slot to swap focal → PL
}
```

Pipeline:
1. Ensemble 8 models, predict on all 9,726 unlabeled SS files × 12 chunks = ~117K chunks.
2. **Discard** chunks where `max(prob) < 0.5`. (Iteration 1: 4,430 files / 19,405 chunks retained; about 17% retention rate.)
3. For retained chunks, **zero out** per-class probs `< 0.1`.
4. Keep as **soft labels**, not binarized.
5. Save to parquet.

### PL training-side fusion (the "clever piece")

For every training sample drawn from the labeled set, the dataloader does:

```python
if soundscape_df is not None and bernoulli(p=0.4):
    initial_class = original_sample.primary_label
    if initial_class in pseudo_birds:
        # SWAP this sample with a PL chunk whose primary class matches
        idx = random_choice(np.where(pseudo_df.primary_label == initial_class))
        sample = pseudo_chunk_at(idx)
        target = pseudo_chunk.soft_label_vector  # soft, multi-class
        is_pseudo_sample = True
```

So 40% of training slots carry a PL chunk **matched by primary class** to whatever class the class-balanced sampler originally drew. The class-balance is preserved.

### Then MixUp on top (p=0.5)

When MixUp fires, **focal (hard) × PL (soft)** are mixed at 50/50 audio:

```python
wav = 0.5 * wave_focal + 0.5 * wave_pl
target = clip(target_focal + target_pl, 0, 1)  # sum-and-clip, NOT max
wav = librosa.util.normalize(wav)  # peak-normalize
```

The summed-and-clipped target carries BOTH "hard, primary species at 1.0" AND "soft, multi-species teacher probabilities" in the same vector. This implicitly fuses classification (matching GT) with distillation (matching teacher) without a separate KD loss term.

### Loss

```python
loss = 1.0 * BCE(reduction=mean) + 1.0 * sigmoid_focal(alpha=0.25, gamma=2.0)
# Applied to clipwise_logits ONLY (no framewise term in selected configs)
```

### Auxiliary
- Label smoothing 0.05 mass-preserving.
- Random 5s crop → MixUp(p=0.5) → BG noise(p=0.5, OneOf nocall-SS | ESC-50) → peak-normalize → batch → RandomFiltering → mel(`AmplitudeToDB → standardize → min-max`) → SpecAugment(freq L=10×3 p=0.3 + time L=20×3 p=0.3).
- Class balancing: `WeightedRandomSampler` with `(c_i/Σc_j)^γ` (γ=-1 fully balanced).
- 3 PL iterations, cumulative.
- 50 epochs, BS=64, `last.ckpt` (NOT best-val).
- Postprocessing: TopN N=1 at ensemble level only.

## Divergence matrix

20 axes audited. ✗ = divergent, ⚠ = partial match, ✓ = match.

| # | Axis | Sydorskyi 2025 | Our exp 033 | Status | Impact |
|---|---|---|---|---|---|
| 1 | **PL × focal MixUp fusion** | `wav = 0.5*focal + 0.5*PL`, target = `clip(focal+PL, 0, 1)`. Sum-clip soft+hard targets. | None. PL samples flow as independent batch slots through their own dataloader path. | ✗ | **CRITICAL** |
| 2 | **Trim probability** | `trim_min_prob=0.1` (zero per-class probs < 0.1) | None. T=0.6 + P^1.5 sharpening but no zero-out below 0.1. | ✗ | **HIGH** |
| 3 | **Chunk discard threshold** | `primary_label_min_prob=0.5` (drop chunk if max < 0.5) | None. All 127K chunks retained in parquet. | ✗ | **HIGH** |
| 4 | **Per-slot 40% swap sampling** | 40% Bernoulli swap; matched by primary_label to preserve class balance | Uniform shuffle over 3-source ConcatDataset OR MixSamp with hand-set shares | ✗ | **HIGH** |
| 5 | **Loss function** | BCE + sigmoid_focal(α=0.25, γ=2.0), 1:1, clipwise only | BCE only, dual-sup (0.5·BCE_clip + 0.5·BCE_frame_max) | ✗ | **MEDIUM** |
| 6 | **Soft PL labels (not binarized)** | Yes — soft probabilities preserved | Yes — soft probabilities (T+P scaled) | ✓ | — |
| 7 | **Label smoothing** | 0.05 mass-preserving (`(1-0.05)*y + 0.05*y.sum()/K`) | None | ✗ | LOW |
| 8 | **BG noise augmentation** | p=0.5 OneOf nocall-SS \| ESC-50, convex blend amp ∈ U[0.25, 0.75] | `apply_aug` does gain + white noise. No nocall-SS or ESC-50 mix. | ✗ | LOW |
| 9 | **RandomFiltering** | 4-band random EQ via STFT, U[-20, 0] dB per band | None | ✗ | LOW |
| 10 | **SpecAugment strength** | Freq L=10 n=3 p=0.3 + Time L=20 n=3 p=0.3 | Freq param=10 n=1 + Time param=10 n=2 (Tucker recipe) | ⚠ | LOW |
| 11 | **Backbone** | `tf_efficientnetv2_s_in21k` (or `eca_nfnet_l0`) | `tf_efficientnet_b0.ns_jft_in1k` (Tucker recipe; the exp 032 Phase D base) | ⚠ | UNKNOWN |
| 12 | **In-domain pretrain** | 819K-recording, 7,489-species pretrain on backbone | ImageNet (NS-JFT only) | ✗ | **HIGH** (separate exp / Path A) |
| 13 | **Iterative PL rounds** | 3 iterations, cumulative pool, exclude already-pseudo'd files | 1 iteration | ✗ | MEDIUM |
| 14 | **Teacher model** | 8-model ensemble (EffNetV2-S × 4 + NFNet-L0 × 4) | Single teacher: v5j-4fold (T1, LB 0.913) OR distilled-sed-kd-5fold (T2, LB 0.915) | ✗ | MEDIUM |
| 15 | **OOF vs Full prediction** | Both run independently; merged. OOF prevents PL leakage. | "Full" only — teacher is trained on all 5 folds, so PL probs carry fold-leakage from training files | ✗ | MEDIUM |
| 16 | **PL exclude val-fold files** | Yes (OOF strategy) | Yes (`val_sc_files` filter in DataModule) | ⚠ | partial |
| 17 | **PL exclude labeled-SS train** | Yes — labeled SS uses GT, PL only for unlabeled | Yes (`exclude_labeled_sc=True`) | ✓ | — |
| 18 | **Mel post-norm** | AmplitudeToDB → standardize → **min-max to [0, 1]** | AmplitudeToDB → standardize (no min-max) (Tucker) | ✗ | LOW |
| 19 | **Class balancing (sampler)** | `WeightedRandomSampler` γ=-1 (fully balanced) | None — uniform shuffle | ✗ | MEDIUM |
| 20 | **Last vs best ckpt** | `last.ckpt` (NOT best-val) | `last.ckpt` for D.5/submit (matches) | ✓ | — |

### Score: 3/20 match, 1/20 partial match, 16/20 divergent

## Top 5 high-impact divergences — detailed analysis

### Divergence #1: PL × focal MixUp fusion missing (CRITICAL)

**What Sydorskyi does**: At p=0.5, every training sample gets MixUp'd with another sample. When the "another sample" is a PL chunk and the original is a focal chunk, the targets are **summed and clipped to [0, 1]** — so the resulting target vector simultaneously contains:
- Focal's hard labels (e.g., primary class = 1.0, all others = 0)
- PL's soft teacher-ensemble probabilities (e.g., 5-10 species with confidence between 0.1 and 0.95)

The mixed target trains the model to predict BOTH the GT primary class AND the teacher's soft distribution simultaneously. This is implicit distillation via MixUp, not via a separate KD loss.

Sydorskyi explicitly calls this "the cleverest piece of the pipeline" in his discussion comments. The paper Section 3.5 details how it enables "joint supervised + semi-supervised" training without a KD head.

**What we do (exp 033)**:
- `FocalDS` has its own MixUp: focal × focal at p=0.5 with `mixup_hard=True` → `max(lb1, lb2)`, OR focal × labeled-SS at p=0.5 with same hard-max.
- `PLDS` has NO MixUp at all. PL samples flow through the dataloader as independent items.
- `ConcatDataset([gt_focal, gt_sc, pl_ds])` stacks the three datasets; the sampler (uniform or MixSamp) picks from each independently.
- **There is no code path where a focal hard-labeled sample gets MixUp'd with a PL soft-labeled sample in a single training slot.**

**Why this likely matters**: Sydorskyi's mechanism implicitly distills the teacher into the student via the focal sample's loss. When a focal sample sees `target = focal_hard + PL_soft (clipped)`, the model is being asked to match both the GT and the teacher in one BCE step. The teacher's soft signal is *fused* into the supervised signal.

In our setup, the PL sample contributes loss separately — the model sees a PL chunk with soft labels in its OWN batch slot. There's no fusion. The PL is a separate, weaker signal competing for batch slots with the cleaner focal signal. This may be precisely why our LB regressed: we gave the model 82% PL-per-batch (uniform sampler over a 127K-chunk pool vs 26K focal) and **none of it was fused into the focal supervised path**.

**Effort to fix**: Implementing this correctly requires either:
- (a) A new `PLFusedFocalDS` that, when fired, draws a PL chunk and returns `(focal_wav + PL_wav)/2, clip(focal_hard + PL_soft, 0, 1), ...` in place of the focal-only return. Conceptually fits into the existing `FocalDS.__getitem__` Path A/B/C structure as a new "Path D: focal × PL fusion".
- (b) ~0.5 day of code + tests + a single retrain to verify.

### Divergence #2: Trim probability missing (HIGH)

**What Sydorskyi does**: After ensemble prediction, all per-class probabilities below 0.1 are zeroed in the saved parquet. This eliminates noise from species the teacher is uncertain about — only the top few species per chunk carry signal.

**What we do**: Our PL generator (`experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py`) does temperature scaling (T=0.6, sharpens) + power transform (P=1.5, sharpens) but **never zeros below a threshold**. The parquet contains the full 234-class soft vector per chunk, including noise.

**Why this likely matters**: With BCE loss and equal-weight (1.0) PL samples, every per-class probability — including the noise floor — contributes to the loss. If a chunk has actual signal at 0.8 for species A and noise at 0.05 for 200 other species, our setup trains the model to match the 0.05 noise floor for those 200 species too. Sydorskyi's trim-to-zero forces the model to ignore those species for that chunk.

The discussion thread (2026-05-27 update) confirms this: Ali Ozan (7th) directly recommended "trim probability — cell-level any pseudo probability below it gets zeroed" as the protocol detail. Jiacheng Ma confirmed "same as the 2nd place 2024 solution." We did NOT implement this.

**Effort to fix**: ~5 lines of code in `generate_pseudo_labels.py` after the `power_transform` step, before saving the parquet:

```python
TRIM_MIN_PROB = 0.1
merged[merged < TRIM_MIN_PROB] = 0.0
```

Plus a re-run of the PL generator (~30s). The parquet schema is unchanged. Existing DataModule code consumes the new parquet without changes.

### Divergence #3: Chunk discard threshold missing (HIGH)

**What Sydorskyi does**: Chunks where `max(prob) < 0.5` are entirely discarded from the PL parquet. Their published numbers: iteration 1 = 19,405 retained chunks (out of ~117K possible), so **~83% of chunks are discarded** because the teacher has no high-confidence call in them.

This dramatically shrinks the pool — from "every unlabeled SS window" to "only windows the teacher recognizes as a bird call." The retained chunks are biased toward actual bird-call events, not noise-only segments.

**What we do**: All 127,104 unlabeled SS chunks remain in our parquet. The PLDS pool size is 127K, not Sydorskyi's ~19K.

**Why this likely matters**: When the sampler is Uniform (canonical Antoine), our model sees ~82% of training samples drawn from a 127K-pool where ~80%+ are likely noise-only (no bird call). Sydorskyi's same Uniform-style sampling sees the same 82% from a 19K-pool of bird-call-rich chunks. We're training on a much noisier distribution.

**Effort to fix**: ~3 lines of code in `generate_pseudo_labels.py` immediately before saving:

```python
MIN_MAX_PROB = 0.5
chunk_max = merged.max(axis=1)
keep_mask = chunk_max >= MIN_MAX_PROB
meta_df = meta_df[keep_mask].reset_index(drop=True)
merged = merged[keep_mask]
```

Combined with #2 (trim), one re-run of the PL generator handles both. ~30 min compute total.

### Divergence #4: 40% per-slot swap sampling missing (HIGH)

**What Sydorskyi does**: The sampler is the standard `WeightedRandomSampler` (class-balanced). For every sample drawn from the labeled set, there's a 40% chance to **swap** it with a PL chunk whose `primary_label` matches the originally-drawn class. The class-balance intent is preserved — if the sampler wanted a sample of class X, it still gets a sample of class X, just possibly a PL chunk with X as the primary instead of a focal chunk.

This mechanism does two things:
1. Maintains class balance under PL.
2. Couples each PL sample to a specific primary class (so loss for class X is dominated by samples actually about X, whether focal or PL).

**What we do**: Three options were implemented in exp 033:
- **Uniform**: shuffle the ConcatDataset of focal+sc+pl. PL probability per slot = `127K / 156K ≈ 81%`. Class balance is destroyed — proportional to pool sizes.
- **MixSamp**: hand-set per-batch shares (focal=0.2, sc=0.0, pl=0.8 in a3/a4). Class balance still not enforced.
- Neither implements primary-class matching.

**Why this likely matters**: The Antoine 1:1 protocol we tried gives PLs the same weight as focal in the loss. But:
- Sydorskyi: 40% of slots are PL'd, each matching a specific focal class. PL is class-matched.
- Us (a2 canonical Antoine): 81% of slots are PL. PL is class-blind (could be any class).
- Us (a3 MixSamp 0.2/0.8): 80% of slots are PL. PL is class-blind.

We over-exposed PL by ~2× AND lost class balance AND lost primary-class matching. Three failures stacked.

**Effort to fix**: ~1 day of code. Requires:
- A new `PLSwapWrappedFocalDS` that wraps a FocalDS, accepts a `pseudo_df`, and at p=0.4 swaps the focal sample with a PL chunk where `pseudo_df.primary_label == focal.primary_label`.
- The base FocalDS sampler stays untouched (could even add WeightedRandomSampler for full Sydorskyi parity).
- ConcatDataset/MixSamp paths can be deleted from the new design.

### Divergence #5: Loss function (MEDIUM)

**What Sydorskyi does**: `loss = 1.0 * BCE(reduction=mean) + 1.0 * sigmoid_focal(alpha=0.25, gamma=2.0)`. Applied to clipwise_logits ONLY. No framewise-max term in their selected configs (`loss_type="baseline"`).

Focal loss with α=0.25, γ=2.0 down-weights well-classified examples (where p ≈ 1 or 0) and focuses gradient on hard examples. With PL data introducing many partial-confidence labels, focal loss is well-matched.

**What we do (Tucker recipe)**: `loss = 0.5 * BCE(clip_logits) + 0.5 * BCE(frame_max_logits)`. Both branches are plain BCE, no focal. The framewise-max term is Tucker's dual-supervision contribution.

**Why this likely matters**: With many partial-confidence PL labels (especially in the [0.1, 0.5] range that survives trim but doesn't dominate), BCE treats all losses linearly. Focal loss specifically de-emphasizes these mid-range labels at γ=2.0, which may be what Sydorskyi needed to make the soft-label fusion work without overfitting to teacher uncertainty.

**Effort to fix**: ~20 lines in `_dual_supervision_loss`. Add an optional `focal_weight` hparam that adds `sigmoid_focal_loss(α=0.25, γ=2.0)` to the BCE branch. Could be tested orthogonally to other fixes.

## What we got right (sanity check)

The other items we DID match Sydorskyi on:

- **Soft PL labels** (#6): We preserve soft probabilities; we did not binarize. ✓
- **Exclude val-fold SS files from PL pool** (#16-17): We filter `val_sc_files` and `labeled_sc_files`. Sydorskyi does this implicitly through OOF construction. ✓
- **Last.ckpt selection** (#20): We use `last.ckpt` for D.5 and submission. ✓
- **5s windows** (not in divergence table, but worth noting): We use 5s windows; Sydorskyi's selected configs also use `train_period=infer_period=5.0`. ✓

These confirm we got the basics right. The divergences are in the protocol mechanism, not in the data plumbing.

## What the discussion thread told us about each divergence

Cross-referenced against the 2026-05-27 thread update (commit `c7beb60`):

| Divergence | Disclosed by | Quote |
|---|---|---|
| Trim probability (#2) | Ali Ozan Memetoglu (7th) | "Cell level any pseudo probability below it gets zeroed" |
| Trim probability (#2) | Jiacheng Ma (25th) | "same as the 2nd place solution" |
| One-pass plateau (vs #13) | Anil Ozturk (6th), Boredom (2nd), Zejun_ (15th), Salman (43rd) | "doesn't work for more than one pass" / "only first student is smart" |
| Same-teacher correlation (vs #14) | Antoine Masq (33rd) | "Do you ensemble models that were trained with the same teachers? If yes, it might be because they are too correlated" |

We had access to all these signals BEFORE running exp 033. We did not apply them.

## Diagnostic plan — three controlled experiments

The cheapest test of "is exp 033 REJECT a real refutation or a missing protocol piece?":

### Test 1: Re-issue PL parquet with chunk discard + trim (cost: ~1h work + 30 min compute)

Goal: Eliminate PL noise. Tests whether divergences #2 and #3 alone explain the regression.

```bash
# Modify experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py
# - Add MIN_MAX_PROB = 0.5 chunk filter
# - Add TRIM_MIN_PROB = 0.1 per-class zero-out
# - Re-run with v2 config (T2 = distilled-sed-kd-5fold teacher)

# Expected: parquet shrinks from 127K chunks to ~20-25K chunks (Sydorskyi reports ~83% discard)
# Expected: per-row label vector becomes much sparser

# Then retrain a2-kd canonical Antoine with the new parquet:
bash sbatch/run.sh configs/a2_kd.yaml --fold 0
```

**LB decision criteria**:
- LB > 0.913 (above teacher) → strong evidence the trim+discard divergences explain the REJECT verdict alone. Protocol is salvageable on our base.
- LB 0.906-0.913 → modest lift, suggests trim+discard helped but other divergences (esp. #1 fusion) still matter.
- LB < 0.906 → these two divergences alone don't explain REJECT. Try Test 2.

### Test 2: Add focal × PL MixUp fusion (cost: ~0.5 day code + retrain)

Goal: Implement the "cleverest piece" of Sydorskyi's pipeline (#1).

Approach: Modify FocalDS to support a "Path D: focal × PL fusion" branch:

```python
# In FocalDS.__getitem__, BEFORE Path A (focal × focal mixup):
if self.aug and self.pl_pool is not None and self.rng.random() < self.focal_pl_mixup_prob:
    pl_idx = int(self.rng.integers(0, len(self.pl_pool)))
    pl_wav, pl_soft_target = self.pl_pool[pl_idx]
    lam = 0.5  # Sydorskyi uses equal-weight
    wav_mix = lam * ch1 + (1 - lam) * pl_wav
    target = np.clip(lb1 + pl_soft_target, 0, 1)  # SUM-CLIP, not max
    return wav_mix, target, ...
```

PL becomes a *pool drawn from in FocalDS*, not a separate Dataset. The 3-source ConcatDataset goes away.

**LB decision criteria**:
- LB > 0.920 → Sydorskyi mechanism transfers strongly to our base.
- LB 0.910-0.920 → partial transfer; pivot to fixing more divergences.
- LB < 0.906 → fusion alone doesn't help; pivot.

### Test 3 (optional): Add focal loss (cost: ~2h code)

Goal: Test divergence #5 in isolation. Add `sigmoid_focal_loss(α=0.25, γ=2.0)` to the BCE branch.

Run only if Tests 1+2 are inconclusive.

## Implementation cost estimates

| Fix | Code time | Compute time | Risk |
|---|---|---|---|
| Trim probability (#2) | 30 min | 5 min (re-gen parquet) | Very low — pure data filter |
| Chunk discard (#3) | 30 min | 5 min (re-gen parquet) | Very low — pure data filter |
| Re-train a2-kd with trim + discard | — | 6.5h (25 ep) or 2.7h (10 ep) | Low — same training code |
| Focal × PL fusion (#1) | 4-6h (new FocalDS path + tests) | + 6.5h training | Medium — new code paths |
| 40% swap sampling (#4) | 4-6h | + 6.5h training | Medium — sampler rework |
| Focal loss (#5) | 1-2h | + 6.5h training | Very low |

**Total minimal diagnostic budget**: 1.5 days (Test 1) + 1.5 days (Test 2 if needed) = ~3 days of the 7 remaining.

## What if all three tests fail?

If after fixing #1, #2, #3 (and optionally #4, #5) we STILL don't lift LB above 0.906:

1. **Real refutation**: own-teacher PL with Sydorskyi-style fusion does not lift our distilled-sed-kd base. This would be a strong scientific finding worth documenting.
2. **Base-quality theory confirmed**: our 0.906 base is too weak to extract PL signal. The 0.929 LB ceiling we're stuck at is the recipe's ceiling on ImageNet-init backbones. The disclosed 0.94+ teams all rely on in-domain pretrain (Sydorskyi 819K corpus, Antoine XC). We don't have that, and PL can't substitute.
3. **Pivot to Path A**: Sydorskyi backbone yoink becomes the only viable path. We accept the ~3 days "spent" on this diagnostic as the cost of being certain we're not pivoting prematurely.

This is a much better outcome than pivoting now and never knowing whether exp 033 was a bug or a refutation.

## My honest read

We did not implement the disclosed PL protocol. We implemented something we *called* the disclosed PL protocol, but the actual published mechanism (Sydorskyi's sum-clip MixUp fusion + chunk discard + trim + 40% swap) was either misunderstood or simplified away. The REJECT verdict cannot be cleanly interpreted as "the protocol doesn't transfer to our base" because we never ran the protocol.

The good news: 3 of the 5 high-impact divergences are very cheap to fix (~1 day combined). One re-trained a2-kd canonical cell with trim + discard + fusion would tell us a lot.

The bad news: even if we fix all 5, we may still hit a base-quality ceiling. The disclosed 0.94+ recipes all stack PL on top of stronger bases (XC pretrain or Perch KD on the right head architecture). Our distilled-sed-kd base (0.906) is below the 0.92 threshold where PL gains became visible for Sydorskyi.

The honest prior: ~50% probability that fixing the divergences lifts a2-kd to 0.910-0.913 range (recovers the regression). ~20% probability of >0.915 (true PL benefit). ~30% probability of staying at 0.886-0.899 (genuine refutation).

## Recommendation

**Do Test 1 today.** It's 1 hour of code + 30 min compute. The decision criterion is clean: if a2-kd jumps above 0.906 we know we have something. If it doesn't, we've ruled out the cheapest fixes and Test 2 (Sydorskyi's fusion) is the next move.

**Do NOT submit Day-2 exp 033 ablations.** Their setup has the same divergences. They will fail for the same reasons.

**Hold off on Sydorskyi yoink until Test 1 lands.** The yoink is a 4-day commitment; spending 1 day on Test 1 first costs us almost nothing.

## Cross-references

- `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` — full Sydorskyi 2025 writeup
- `experiments/033-own-teacher-pl/proposal.md` — exp 033 design
- `experiments/033-own-teacher-pl/results.md` — REJECT verdict
- `notes/kaggle-discussion-best-single-model-lb.md` — competitor disclosures (incl. Ali Ozan's trim recommendation in Thread 20)
- `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` — PL generator to modify for Tests 1
- `user/murilo/src/birdclef/data/distilled_sed_dataset.py:FocalDS` — where to add Path D for Test 2
- `user/murilo/src/birdclef/data/distilled_sed_datamodule.py` — current PL DataModule (to simplify if Test 2 lands)


---

# § 3. COMPANION DOC: BirdCLEF 2025 2nd-place writeup (Sydorskyi)

*Source: `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md`*

---
slug: "sydorskyi-2nd-place-2025"
type: "writeup"
competition: "birdclef-2025"
date_captured: "2026-04-26"
---

# BirdCLEF+ 2025 2nd Place — Sydorskyi & Gonçalves

**Paper**: [CEUR Vol-4038, paper_256](https://ceur-ws.org/Vol-4038/paper_256.pdf) — *"Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation: A Case Study on BirdCLEF+ 2025"*
**Authors**: Volodymyr Sydorskyi (KPI, Kyiv) & Fernando Gonçalves
**GitHub**: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
**Kaggle Discussion**: https://www.kaggle.com/competitions/birdclef-2025/discussion/583699
**Final score**: Public 0.925, **Private 0.928** (2nd place)

> Same lead author as the 2023 1st place solution (`docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md`). The 2025 pipeline is an evolution of that recipe with two big additions: (a) a much larger in-domain pretraining corpus (819K recordings, 7,489 species) and (b) iterative semi-supervised distillation on unlabeled soundscapes — both motivated as **domain-shift mitigation**.

## Summary

The solution has five pillars:

1. **Strong baseline** — CNN backbone + GeMFreq + AttHead (a refined version of the 2023 recipe).
2. **In-domain pretraining** — pretrain backbones on a 7,489-species, 819,032-recording corpus assembled from prior BirdCLEF competitions + Xeno-Canto (everything except 2025 species, to avoid data leakage).
3. **Iterative pseudo-labeling / semi-supervised distillation** — **the largest single LB lever**. Use an 8-model ensemble to label unlabeled soundscapes, mix soft pseudo-labels into the training distribution at 40% sampling probability, retrain, repeat for 3 iterations. **+0.025 to +0.035 LB.**
4. **Postprocessing** — multiplicative TopN with N=1: `prob_a,s,c *= max(prob_a,:,c)`. **+0.01 to +0.015 LB**.
5. **Ensembling** — 5-fold × 3 selected experiments = 15 models, simple average. Optuna-selected ensemble vs Public-LB-selected ensemble (Public+0.929 Private vs Public+0.925 Private). +0.005 over best single.

Final ensemble = 15 models. Best single fold = 0.917 LB Public. Best ensemble = 0.925 Public / 0.929 Private.

## Architecture (`WaveCNNAttenClasifier`)

Raw waveform → on-the-fly nnAudio mel-spectrogram → 2D CNN backbone → GeMFreq pooling → attention-based SED head.

### Backbones (final ensemble)

- **EfficientNetV2-S** (`tf_efficientnetv2_s_in21k`) — primary
- **ECA-NFNet-L0** (`eca_nfnet_l0`) — diversity

Both pretrained from ImageNet-21k → in-domain pretraining → fine-tuning on 2025 with pseudo-labels. EfficientNetV2-S replaced ConvNeXt from the 2023 recipe; the paper notes both work well but EfficientNetV2-S converged faster after transfer-learning warm starts.

### Mel-spectrogram (nnAudio + AmplitudeToDB)

Identical across pretrain, fine-tune, and inference. Implemented as `TraceableMelspec` for ONNX export.

| Parameter | Value |
|---|---|
| Sample rate | 32,000 Hz |
| n_mels | 128 |
| n_fft | 2,048 |
| hop_length | 512 |
| f_min | 20 Hz |
| f_max | (none — uses full Nyquist) |
| normalized | True |
| top_db | 80 |
| amin | 1e-10 |
| Post-mel norm | AmplitudeToDB → standardize (zero-mean, unit-std) → min-max scale to [0, 1] |

Code: `code_base/models/wave_clasifier.py:WaveCNNAttenClasifier` + `code_base/models/blocks.py:NormalizeMelSpec`.

### Attention head (`AttHead`)

Closely matches the 2023 1st-place head:

```
GeMFreq(p=3, learnable)
→ Dropout(0.25)
→ Linear(in_chans, 512)
→ ReLU
→ Dropout(0.5)
→ [Conv1D attention branch:  out = num_class, kernel=1, +tanh]
→ [Conv1D value branch:      out = num_class, kernel=1]
```

**Clipwise prediction** (training and inference, when `train_period == infer_period`):

```python
clipwise_pred  = sum( sigmoid(value)  * softmax(tanh(att), dim=time), dim=time )
clipwise_logit = sum(         value   * softmax(tanh(att), dim=time), dim=time )
```

Clipwise pred is in **probability space** — sigmoid is applied to value before attention-weighted sum. Logit version exists for loss computation.

When `train_period > infer_period`, an additional center-cropped pathway computes `framewise_clipwise_pred_short = framewise_max * 0.5 + clipwise * 0.5` (the 75/25 blend from the 2023 paper has been changed to **50/50** here; `infer_framewise_max_coef=0.5` in the head). The selected experiments use `train_period = infer_period = 5.0`, so this branch is unused in the final models.

Code: `code_base/models/blocks.py:AttHead`.

### Hidden channels

- Fine-tuning configs: 512 (default `hidden_chans=512`)
- Pretraining configs: **1024** (`hidden_chans=1024` for the 7,591-class head — they widen to absorb the extra label space)

The classification head is **discarded** between pretrain and fine-tune; only the CNN backbone is carried over (`scripts/create_pretrain_backbone.py`).

## Loss function (`FocalLossBCE`)

```python
loss = bce_weight * BCEWithLogitsLoss + focal_weight * sigmoid_focal_loss
     = 1.0 * BCE + 1.0 * Focal(alpha=0.25, gamma=2.0)
```

Two losses **added together**, equal weight. Computed on `clipwise_logits_long` only (no framewise-max term in the selected configs — `loss_type="baseline"`). Applied with reduction="mean".

Code: `code_base/losses/focal_loss.py:FocalLossBCE`, `code_base/forwards/cls_forward.py:MultilabelClsForwardLongShort`.

### Multi-hot targets with secondary labels

Targets are built as multi-hot vectors over **primary + all secondary** labels (each at value 1.0, summed and clipped to [0,1]). XC recordings carry secondary labels (4,958 of 28,564 in 2025); iNat / CSA do not.

### Label smoothing (mass-preserving variant)

```python
all_tgt = all_tgt * (1 - 0.05) + 0.05 * all_tgt.sum() / num_classes
```

Note this is a **mass-preserving** form: smoothing is proportional to the number of active classes per sample, not a flat 1/K. So a sample with one positive gets `0.05/K` smoothing on negatives and `0.95 + 0.05/K` on the positive; a sample with two positives gets `0.10/K` on negatives.

Code: `code_base/datasets/wave_dataset.py:WaveDataset._prepare_target` (line ~599).

## Augmentation cascade

Applied **per sample in the dataloader** (CPU, before batch collate), with a few exceptions:

### 1. Random 5s crop

Random 5-second segment from full audio. **No "first 7s" or "last 7s" trick** — they tested it, found mild improvement, then tried surgical curation (cutting alien speech), found it hurt LB, abandoned it. Final pipeline: pure random 5s.

### 2. Audio-domain MixUp (p=0.5)

- Pick another sample uniformly (or via weighted sampler if `weights_path` set, but selected configs don't use this).
- Mix at **equal weight** (`alpha=None` → `lam=0.5`): `wave = 0.5 * wave1 + 0.5 * wave2`.
- Targets summed and clipped to [0, 1] (`target_aggregation="sum"`).
- Then renormalize the wave via `librosa.util.normalize` (peak-normalize to [-1, 1]).

This is **simple averaging-with-summed-labels MixUp on raw waveforms**, not Beta-distribution interpolation. The paper notes (Table 8): removing MixUp drops Public 0.835→0.835 (no change) but Private 0.871→0.835 (**−0.036**). MixUp is a Private-set robustness lever.

Code: `code_base/datasets/wave_dataset.py:WaveDataset.__getitem__` (lines 762-820).

### 3. Background noise (p=0.5, OneOf two sources)

Wrapped in `OneOf([source_A, source_B])` — picks one source uniformly each call:

**Source A — nocall soundscapes from BirdCLEF 2023** (`data/soundscapes_nocall/train_audio` + `data/v1_no_call_meta.csv`).

**Source B — ESC-50** filtered to 20 categories: `dog, rain, insects, hen, engine, hand_saw, pig, rooster, sea_waves, cat, crackling_fire, thunderstorm, chainsaw, train, sheep, wind, footsteps, frog, cow, crickets`.

Each source has its own `p=0.5`, but inside `OneOf` only one is picked, so the effective per-sample BG-noise application rate is `0.5 × 0.5 + 0.5 × 0.5 = 0.5`.

**Mixing math**:
```python
back_amp = uniform(0.25, 0.75)             # random amplitude per call
augmented = y * (1 - back_amp) + back_sample * back_amp
augmented = librosa.util.normalize(augmented)
```

This is **convex linear mixing**, not additive — the original signal is reduced as the noise is added. Equivalent to a random SNR draw. Paper Table 8: removing **RandomFiltering** alone changes Public +0.019, Private −0.010 (BG noise removal isn't ablated separately in the paper; the 2023-1st recipe used it, so 2025 inherited it).

Code: `code_base/augmentations/transforms.py:BackgroundNoise`.

### 4. SpecAugment (in-model, GPU)

Applied to the mel-spectrogram inside the model's forward pass when training:

- **FreqMasking**: max length 10, max masks 3, p=0.3
- **TimeMasking**: max length 20, max masks 3, p=0.3

Per the ablation (Table 8), removing SpecAugment moves Public **+0.010** but Private **−0.010** — Private-side robustness lever again.

Code: `code_base/augmentations/spec_augment.py:CustomFreqMasking, CustomTimeMasking`.

### 5. RandomFiltering (batch-level GPU augmentation)

Applied as `batch_aug` inside `MultilabelClsForwardLongShort`. Operates on the waveform via STFT → multiplicative filtering → iSTFT.

- 4 control points uniformly placed across frequency bins, each drawn from `uniform(min_db, 0) = uniform(-20, 0)` dB.
- Linearly interpolated to a per-frequency-bin gain, applied to the magnitude spectrogram.
- Inverse STFT recovers a re-equalized waveform.

The 2023 paper called this a "simplified random equalizer to simulate channel distortions." Paper ablation (Table 8): removing it moves Public +0.019, Private −0.010 — same Private-vs-Public trade as SpecAugment.

Code: `code_base/models/augmentations.py:RandomFiltering`.

### Augmentation pipeline summary

```
pick random 5s crop
→ MixUp(p=0.5, audio domain, equal-weight, label-sum)
→ BG noise (p=0.5, OneOf nocall-soundscape | ESC-50, convex blend at amp ~ U[0.25, 0.75])
→ peak-normalize waveform
→ [batch collate, move to GPU]
→ RandomFiltering (4 bands × U[-20, 0] dB)
→ nnAudio mel + AmplitudeToDB + standardize + min-max
→ SpecAugment (freq mask, time mask)
→ Backbone forward
```

## Training: two-stage transfer learning

### Stage 1 — Pretrain on 819K recordings, 7,489 species

- Collected taxonomy of all species from prior BirdCLEF competitions: 16,607 unique.
- Downloaded audio + metadata via Xeno-Canto, iNaturalist, CSA, prior BirdCLEF train sets.
- Pruned: removed all 2025 species (no leakage), corrupted files, invalid eBird codes, species with < 10 recordings → **819,032 recordings, 7,489 species**.
- 5% holdout for evaluation; only species with ≥ 100 recordings used for holdout metric (1,627 species in eval).
- Trained with the **same baseline recipe**, just no class balancing (the corpus is too large; balancing not needed at this scale).
- Best checkpoint by holdout macro ROC AUC. **Discard classification head; keep CNN backbone only** (`scripts/create_pretrain_backbone.py`).

Pretraining configs: `train_configs/pretrain_ebs_2025_extended.py`, `train_configs/pretrain_eca_2025_extended.py`. AdamW lr=2.5e-4, 50 epochs, BS=64, CosineAnnealingWarmRestarts to 1e-6.

> **Important note**: the authors did NOT include the pretraining audio in the released artifacts (~1TB). They provide metadata CSVs and download scripts (`scripts/download_all_xeno_canto.py`, `scripts/download_inaturalist.py`) so you can rebuild the corpus yourself. They also publish the pretrained checkpoints directly on Kaggle: [`bird-clef-2025-all-pretrained-models`](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models).

### Stage 2 — Fine-tune on 2025 + pseudo-labels

- **Data**: `train_audio_2025` (28,564 recordings, 206 species) + ~10K extra recordings from XC dump (28.03.2025) and previous-comp scraping (only the **two** highlighted "+impactful" sources from Table 3 are kept in `selected_ebs.py`; the all-data variant was *worse*: Public 0.835 → 0.832 in Table 7).
- **Same model** as pretrain (backbone weights loaded), but **head re-initialized** for 206 classes, hidden_chans dropped to 512.
- AdamW (EfficientNetV2-S) at **lr=1e-4** / RAdam (NFNet-L0) at **lr=1e-3**, both with `eta_min=1e-6` and `betas=(0.9, 0.999)` for AdamW. CosineAnnealingWarmRestarts, T_0 = 50 epochs × steps-per-epoch (so essentially one cosine cycle).
- **50 epochs**, batch size 64, **no AMP** (`precision="32-true"`).
- **No early stopping**; final checkpoint is `last.ckpt` (best-by-val also saved with `save_top_k=3`, but the inference configs use **`last.ckpt`** by default).

### Class balancing (sampler)

Per the paper Equation 1, weights are `(c_i / Σc_j) ^ γ`:

- **EfficientNetV2-S**: `class_weights_path="balanced"` → γ = −1 (fully balanced sampling).
- **ECA-NFNet-L0**: `class_weights_path="sqrt"` → γ = −0.5 (square-root balanced, gentler) **plus** explicit minor-class oversampling (the `OVERSAMPLE_CONFIG` dict in `selected_eca.py` duplicates rows for 58 specific rare species up to between 10 and 96 extra rows each). Different sampling strategies per backbone — the paper notes this is one source of ensemble diversity.

Implemented via PyTorch `WeightedRandomSampler`.

### CV split

- 5-fold StratifiedGroupKFold by `primary_label`, **grouped by recordist** (to prevent recordist leakage from XC's habit of uploading bursts of recordings in the same session).
- Two variants for handling rare species:
  1. Add them to validation (assess them, lose train data).
  2. Add them only to train (no eval, more data, less label noise).
- The selected configs use the **train-only** variant (`split_path` filenames contain `allbirds` or `allrarebirds_noleak`).

The paper Section 5.1 notes: **CV-LB correlation breaks down once Public > 0.9** (Pearson −0.13, Spearman −0.12 in that regime). They could not find a CV strategy that tracks LB once pseudo-labeling is engaged. **Optuna-on-OOF still produced the highest-Private ensemble** despite low individual correlation, suggesting OOF carries diversity signal even when it doesn't track absolute scores.

## Pseudo-labeling / semi-supervised distillation

This is **the largest single LB lever** (Table 5: Public 0.881 → 0.910–0.917, Private 0.889 → 0.910).

### Pipeline

1. Start with 8 baseline models (EffNetV2-S × 4 + NFNet-L0 × 4) trained from in-domain pretrained checkpoints.
2. Predict on **all 9,726 unlabeled soundscape files** (= 116,712 chunks of 5s each).
3. Mean-ensemble predictions across the 8 models.
4. **Pseudo selection logic** per chunk:
   - Compute `max(prob_c)` across all 206 classes.
   - **Discard** chunks where `max < 0.5`.
   - For retained chunks, **zero out** all class probabilities `< 0.1`.
   - Keep the rest as **soft labels** (not binarized).
5. Save the surviving (chunk → soft-label vector) pairs as the pseudo dataset.
6. Retrain models with pseudo-data fused into the training distribution (next subsection).
7. Generate next iteration of predictions, repeating with stronger models.

Iteration sample counts (Table 4):

| Iteration | Files selected | Chunks selected |
|---|---:|---:|
| 1 | 4,430 | 19,405 |
| 2 | 1,483 | 3,750 |
| 3 | 1,437 | 4,108 |

Each iteration's pseudo dataset is **cumulative** with prior iterations (with the "remove already-used files before re-predicting" logic in Figure 3).

### Two strategies (run independently then merged)

- **Full mode**: every iteration predicts on all soundscapes (excluding files already in the pseudo set), accumulates new ones. Each chunk gets predicted by 10 models (5 folds × 2 architectures).
- **OOF mode**: every iteration *re-predicts* every soundscape, with each fold's predictions coming **only from models not trained on that fold**. Each chunk gets only 2 models' predictions, but pseudo-labels refresh each round.

The two strategies were used independently (paper Table 5) and also **merged** into a single dataset using "remove from previous iteration" logic. Merging adds ensemble diversity. Best results (Table 5): **OOF I2 + Full I2** ensemble (Public 0.917, Private 0.910).

### Training-side fusion: 40% sampling probability + soft-label MixUp

This is the cleverest piece of the pipeline (`code_base/datasets/wave_dataset.py:WaveDataset._prepare_sample_target_from_idx`, lines ~604-735).

For every training sample drawn from the labeled set:

```python
if soundscape_df is not None and bernoulli(soundscape_pseudo_config["sampling_prob"] = 0.4):
    initial_class = original_sample.primary_label
    if initial_class in pseudo_birds:
        # Replace this sample with a pseudo-labeled chunk
        # whose primary class matches the originally-drawn class
        idx = random_choice(np.where(pseudo_df.primary_label == initial_class))
        sample = pseudo_chunk_at(idx)
        target = pseudo_chunk.soft_label_vector
        is_pseudo_sample = True
```

So **40% of training batches** carry a soundscape chunk with a soft-label vector (matched by primary class to preserve the class-balanced sampler's intent). The other 60% are normal hard-labeled focal samples.

**Then MixUp runs on top** at p=0.5 — and the MixUp here is the *interesting* fusion mechanism the paper emphasizes:

- A focal hard-labeled sample (target ∈ {0, 1}^206) and a soundscape soft-labeled sample (target ∈ [0, 1]^206) get audio-mixed at 50/50.
- Their targets are **summed and clipped to [0, 1]**, so the resulting target vector simultaneously carries "hard, primary species at 1.0" alongside "soft, multi-species probabilities from the teacher ensemble" in the same vector.
- This implicitly fuses **distillation** (matching the teacher's soft predictions) with **classification** (matching the ground-truth label) without needing a separate KD loss term.

Pseudo-label thresholds (`soundscape_pseudo_config`):

```python
{
    "primary_label_min_prob": 0.5,   # discard chunks with no high-confidence class
    "trim_min_prob": 0.1,            # zero out per-class probs below this
    "sampling_prob": 0.4,            # 40% chance to swap in pseudo per batch slot
}
```

Soft pseudo-labels were chosen explicitly **against** binarized pseudo-labels — paper Section 2 critiques [DS@GT 2024 (Miyaguchi/Cheung/**Gustineli**/Kim)](https://arxiv.org/abs/2407.06291) for using binarized pseudo-labels that produce overconfident predictions.

> **Direct callout to our prior work**: Sydorskyi cites Miyaguchi/Cheung/Gustineli/Kim 2024 (BirdCLEF 2024 DS@GT) as an example of "classical semi-supervised learning with binarized pseudo-labels" that overconfidently predicts. The 2025 2nd-place pipeline is explicitly engineered to fix that failure mode. Worth knowing as we revisit pseudo-labeling for 2026.

## Postprocessing — TopN, N=1

The 2nd-place team tested four PP methods (Section 5.6):

1. **Mean** — `prob_a,s,c *= mean_s'(prob_a,s',c)`
2. **TopN** — `prob_a,s,c *= mean over top-N segments(prob_a,:,c)`
3. **Convolution** — temporal smoothing across neighboring segments
4. **L2 model** — train a second-stage CNN on pseudo-labeled chunks to model species-specific rhythmic patterns

**Final winner: TopN with N=1**, applied at the **ensemble level only** (applying it per-model regressed). PP gain: **+0.01 to +0.015 LB** (Table 5: Full I2 → Full I2 + PP: Public 0.910 → 0.918, Private 0.909 → 0.924).

The L2 model didn't beat TopN N=1, even though it explicitly modeled per-species call rhythms.

Pipeline order: **PP → rank transform → arithmetic average** (when ensembling). Applying PP after rank-transform was tried but degraded results.

## Ensembling

- Per experiment: 5 folds × 1 architecture = 5 models, simple arithmetic mean.
- Final ensemble: **3 selected experiments × 5 folds = 15 models**.
- Selection strategies:
  - **Public-LB selection** → Public 0.925 / Private **0.928**.
  - **Optuna optimizing OOF AUROC** → Public 0.921 / Private **0.929**. Despite weaker CV-LB correlation, Optuna-on-OOF won Private. The paper interpretation: OOF still captures diversity / model-correlation signal even when it doesn't track absolute scores.

Single-model gain from ensembling: **only +0.005**. The single models already score Public 0.917, so the ensemble is mostly Private-set robustness.

Inference precision: **FP16, OpenVINO** (`scripts/main_inference_and_compile.py`, `code_base/utils/onnx_utils.py:ONNXEnsemble`). Mel extraction is done **once and reused across all ensemble models** for inference speed (90-min Kaggle CPU budget).

## Ablation findings (paper Tables 5, 7, 8)

### Stepwise contribution (Table 5)

Reading the EfficientNetV2-S column from baseline to final:

| Stage | Public | Private |
|---|---:|---:|
| Baseline (no enhancements) | 0.837 | 0.859 |
| + enhancements (label smoothing, balanced sampling) | 0.835 | 0.871 |
| + transfer learning (pretrain on 7,489-species corpus) | 0.881 | 0.889 |
| + pseudo-labels Full I2 | **0.910** | 0.909 |
| + postprocessing (TopN N=1) | **0.918** | **0.924** |
| Final ensemble (15 models) | 0.925 | 0.928 |

So per-component approximate contributions:

- Baseline → Enhancements: **~+0.012 Private** (label smoothing + sampling tweaks).
- Enhancements → Transfer learning: **+0.046 Public, +0.018 Private**.
- Transfer learning → Pseudo-labels: **+0.029 Public, +0.020 Private**.
- Pseudo-labels → PP: **+0.008 Public, +0.015 Private**.
- Single → 15-model ensemble: **+0.007 Public, +0.004 Private**.

**The two big levers are transfer learning and pseudo-labels**, contributing ~+0.07 each in stacked Public.

### Additional data ablation (Table 7)

| Config | Mean CV | Public | Private |
|---|---:|---:|---:|
| Train data only (no extras) | 0.972 | 0.839 | 0.868 |
| + 2 selected extras (XC dump 28.03.2025 + prev-comp samples) | 0.975 | 0.835 | 0.871 |
| + ALL extras (incl. CSA, iNat, more XC) | 0.976 | 0.832 | 0.864 |

**Adding more data hurt LB once past the two highlighted sources.** The paper attributes this to: (a) 2025 train data is already large enough; (b) systematic alien-speech contamination in some extras; (c) overrepresentation of recordists' biases. They did not pursue more aggressive curation.

### Augmentation ablation (Table 8) — measures Public / Private trade

| Setup | Mean CV | Public | Private |
|---|---:|---:|---:|
| Enhanced baseline (all augs) | 0.975 | **0.835** | **0.871** |
| − SpecAugment | 0.974 | 0.845 | 0.861 |
| − RandomFiltering | 0.973 | 0.854 | 0.861 |
| − MixUp | 0.964 | 0.835 | **0.835** |

**Striking**: Public score *increases* when SpecAugment / RandomFiltering are removed, but Private *decreases* every time. The augmentations help the harder, larger Private set at a small cost to Public. **MixUp is essential** — removing it kills CV (−0.011) and Private (−0.036).

The paper argues this is consistent with a smaller density of overlapping species per chunk in the Public split — augmentations help Private's harder distribution.

## Hyperparameter table (final fine-tune configs)

| Component | EfficientNetV2-S | ECA-NFNet-L0 |
|---|---|---|
| Backbone (timm) | `tf_efficientnetv2_s_in21k` | `eca_nfnet_l0` |
| Mel | nnAudio, n_mels=128, n_fft=2048, hop=512, fmin=20, normalized | same |
| Mel norm | AmplitudeToDB(top_db=80) → standardize → min-max | same |
| Head | `AttHead`, hidden=512, dropout=0.5 (p/2 + p) | same |
| Loss | BCE(reduction=mean) + sigmoid_focal(α=0.25, γ=2.0) (1:1) | same |
| Label smoothing | 0.05 (mass-preserving) | same |
| Optimizer | AdamW, lr=1e-4, ε=1e-8, β=(0.9, 0.999) | RAdam, lr=1e-3 |
| Scheduler | CosineAnnealingWarmRestarts(T_0 = 50 × steps), eta_min=1e-6 | same |
| Epochs | 50 | 50 |
| Batch | 64, drop_last, ddp_find_unused_params_true | same |
| Precision | 32-true (no AMP) | same |
| Window | 5s, random crop | same |
| Class sampling | balanced (γ = −1) | sqrt (γ = −0.5) + manual oversampling of 58 minor classes |
| Init | In-domain pretrain backbone (head re-init) | In-domain pretrain backbone (head re-init) |
| Pseudo iter used | I2 (post hoc selected) | I3 + minor oversample variant |
| Final ckpt | `last.ckpt` (NOT best-val) | same |
| Augs (per-sample) | MixUp p=0.5 (audio, 50/50) → BG noise OneOf(nocall-SS, ESC-50) p=0.5 → peak-normalize | same |
| Augs (in-model) | RandomFiltering(4 bands, U[-20, 0] dB) → mel → SpecAugment(freq L=10×3, time L=20×3, both p=0.3) | same |
| Inference | OpenVINO FP16, ensemble extracts spec once, mean across folds, then 3-experiment mean, then TopN N=1 | same |

## Their stated limitations

From Section 7.1:

1. **Validation doesn't reflect deployment.** Soundscapes are biased toward one geographic site; pseudo-labels may not generalize to other regions.
2. **No serious hyperparameter optimization.** Especially pseudo-thresholds (0.5 / 0.1 / 0.4) are educated guesses; the search space wasn't explored.

Future-work directions they list:
- More diverse soundscape evaluation across geographies.
- Model-driven curation to remove non-vocalized fragments (they tried it, hurt LB; revisiting later).
- Use the AttHead's framewise predictions at inference (currently only clipwise is used).
- Better undersampled-class handling.

## What's notably *absent* (compared to other top solutions)

- **No knowledge distillation from BirdNET / Perch / Google Bird Vocalization Classifier.** They explicitly note this is what the 2024 DS@GT paper did and that it underutilizes transfer learning.
- **No BirdNET embeddings / Perch features.** Pure timm CNN backbones, raw waveform input.
- **No CE loss.** Pure BCE+Focal multi-label.
- **No model soup / SWA in the final selected ensemble.** `n_swa_models=1` in the inference configs; they pick `last.ckpt` (which differs from `best-val.ckpt`) — interesting choice, presumably because the cosine schedule lands on a sensible final checkpoint.
- **No attention-architecture innovation.** AttHead is the same one as the 2023 paper, with the framewise blend coefficient nudged from 75/25 → 50/50 — essentially a copy of the 2018 Adavanne–Virtanen weakly-labeled SED head.
- **No multi-context / longer windows.** Pure 5s training and 5s inference.
- **No insect/amphibian-specific pipeline** (Nikita Babych 1st place 2025 had a separate one). Sydorskyi treats all 206 species uniformly.
- **No 2nd-stage temporal model** (the L2 model was tested and abandoned).

The pipeline is deliberately **boring and big**: huge pretraining corpus, lots of pseudo-iterations, simple averaging ensemble. Almost no architectural cleverness.

## Available artifacts

| Artifact | Location |
|---|---|
| Trained inference models (OpenVINO FP16) | [Kaggle: bird-clef-2025-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-models) |
| In-domain pretrained backbones | [Kaggle: bird-clef-2025-all-pretrained-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models) |
| Pseudo-label CSVs (3 iterations, both Full and OOF) | [Kaggle: bird-clef-2025-pseudo](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pseudo) |
| Additional 2025 training data (XC dump, prev-comp scrapes) | [Kaggle: bird-clef-2025-add-data](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-add-data) |
| Pretraining metadata CSVs | [Kaggle: bird-clef-2025-pretrained-metadata](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pretrained-metadata) |
| Inference notebook (final selected) | [Kaggle: bird-clef-2025-ensemble-v2-final-final](https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-ensemble-v2-final-final) |
| Code repo | https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place |

## Key takeaways

1. **The 2025 ceiling without unlabeled data is ~0.89 Public**, just from a careful baseline + in-domain pretrain. Everything past that — for this team — came from **using the unlabeled soundscapes**, exactly as Boredom (1st place) and Salman (13th place 2025, single 0.937) also reported.

2. **Transfer learning at scale is more powerful than competition-year-only training.** The 819K-recording, 7,489-species corpus is the foundation. Without it, the baseline tops out at 0.837 Public.

3. **Soft pseudo-labels, not hard.** The paper explicitly diagnoses binarized PL (the DS@GT 2024 approach) as overconfident. Soft labels with `trim_min_prob=0.1` zero-out preserve uncertainty.

4. **Pseudo-distillation by 40% sampling + sum-MixUp.** The pseudo-target is fused into training implicitly via MixUp-summed targets, not as a separate KD loss. Cleanest semi-supervised recipe published in this competition.

5. **Augmentations help Private more than Public.** Three of four augmentations (SpecAugment, RandomFiltering, MixUp) all reduce Public score slightly while raising Private significantly. The harder distribution is Private; that's where to optimize.

6. **PP TopN N=1 at ensemble level only**: ~+0.015 LB. Per-model PP regressed. Order matters: PP → rank → average is wrong; ensemble average → PP is right.

7. **OOF correlations break down past 0.9 Public, but Optuna-on-OOF still picks the best Private ensemble.** OOF carries diversity signal even when absolute scores diverge.

8. **No KD from frozen embeddings** (BirdNET / Perch / GBVC) was used. The pipeline is pure timm-CNN-on-mel — distillation is from in-domain *student-of-student* ensembles, not external models.

## File-level pointers (for later replication)

- Baseline architecture: `code_base/models/wave_clasifier.py`, `code_base/models/blocks.py`
- Augmentations: `code_base/augmentations/transforms.py`, `code_base/augmentations/spec_augment.py`, `code_base/models/augmentations.py:RandomFiltering`
- Loss: `code_base/losses/focal_loss.py:FocalLossBCE`
- Training forward + MixUp + batch aug: `code_base/forwards/cls_forward.py:MultilabelClsForwardLongShort`
- Pseudo-label fusion: `code_base/datasets/wave_dataset.py:WaveDataset._prepare_sample_target_from_idx` (lines ~604-735)
- Pseudo creation pipeline: `notebooks/create_pseudo.ipynb`
- Pretraining configs: `train_configs/pretrain_*_2025_extended.py`
- Final fine-tune configs: `train_configs/selected_ebs.py`, `train_configs/selected_eca.py`
- Inference + ONNX/OpenVINO compile: `scripts/main_inference_and_compile.py`, `code_base/inefernce/inference_class.py`, `code_base/utils/onnx_utils.py`
- One-button reproduction: `rock_that_bird.sh`


---

# § 4. COMPANION DOC: Exp 033 proposal

*Source: `user/murilo/experiments/033-own-teacher-pl/proposal.md`*

---
experiment: "033-own-teacher-pl"
status: draft
date: 2026-05-25
author: Murilo Gustineli
---

# Proposal: Own-teacher pseudo-labels on distilled-sed-kd

**Status**: draft (2026-05-25)
**Created**: 2026-05-25
**Author**: Murilo Gustineli

## Provenance

This experiment implements **own-teacher pseudo-labeling** — using a strong SED model's predictions on unlabeled soundscapes as additional training supervision, treated as 1:1 equivalent to ground-truth labels. The technique is classical semi-supervised learning (noisy student, self-training) and is the modal recipe at the top of the BirdCLEF 2026 LB. Five independent competitor disclosures converge on this lever:

- **Antoine Masq (49th, LB 0.943)** — "I just incorporated [PLs] by using the unlabeled data just as the labeled one but the labels are predicted by the teacher models." Reports +0.007 LB from teacher iteration (0.936 → 0.943) on top of XC pretrain.
- **OverfitOracle (685th, LB 0.946)** — SED + teacher-generated pseudo frame/event labels on XC + competition data.
- **Boredom (5th, LB 0.938+ no PL, 0.950+ with 1-iter PL)** — confirms +0.012 LB from a single PL iteration.
- **chenwenqiang_001** — independently reports +0.013 LB from own-teacher PL on a 0.886 base.
- **Tucker Arrants** — references the same iterative pattern in his distilled-SED discussion.

The protocol's name is generic in the literature; we drop competitor-specific naming from artifacts. Author attribution is preserved here in references, not in directory or branch names.

Cited materials (vendored / captured):

| Doc | Content |
|---|---|
| `notes/kaggle-discussion-non-perch-limit.md` | Antoine + OverfitOracle + Arunodhayan + Boredom disclosures |
| `notes/kaggle-discussion-best-single-model-lb.md` | Cross-reference for the same recipe pattern |
| `notes/2026-05-22-strategic-direction-tracks-A-B.md` | Strategic context: this is "Track B (own-teacher PL)" on top of "Track A (Tucker)" |
| `experiments/025-iter-pseudo-labeling/results.md` | Prior PL attempt (REJECTED, but for a different protocol) |
| `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` | PL generation script — reused for T2 parquet (Step 0.3) |
| `experiments/032-tucker-distilled-sed/results.md` | Phase D distilled-sed-kd (LB 0.906) — the base student |
| `experiments/032-tucker-distilled-sed/configs/distilled_sed_kd.yaml` | Tucker recipe + KD config — basis for exp 033 configs |
| `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Tucker code audit; recipe foundation |
| `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet` | **T1**: pre-generated PL parquet, v5j-4fold teacher (ready to use; pre-flight 7/7 PASS) |
| `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` | **T2**: PL parquet to be generated in Step 0 (distilled-sed-kd-5fold teacher) |
| `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/` | T1 cached inference (already exists) |
| `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/` | T2 cached inference to be generated in Step 0 |
| `experiments/010-submission-pipeline/artifacts/weights/distilled-sed-kd-5fold/model_fold{0..4}.onnx` | T2 ONNX weights for the inference caching script |

## Hypothesis

**1:1 own-teacher PL on the `distilled-sed-kd` recipe lifts single-fold LB from 0.906 to ≥ 0.920.** Treating unlabeled soundscape teacher predictions as 1:1 equivalent training labels (no down-weighting, no waveform mixing) engages the lever that lifted Antoine 0.936 → 0.943, Boredom +0.012, and chenwenqiang_001 +0.013.

Mechanism: our prior PL attempts (exp 014/015/025) all used **down-weighted PLs** (loss_weight=0.5 in Phase B) or **waveform-mixing** (Phase C). Both regressed LB (−0.009 / −0.012). The five competitor disclosures uniformly use 1:1 weighting. This experiment isolates the **weighting + sampler protocol** as the variable, and additionally tests **teacher architecture diversity** (v5j-4fold vs `distilled-sed-kd-5fold`) as an orthogonal axis.

The two teachers are ~equal in LB quality (Δ = +0.002, within seed noise) but architecturally different (HGNet+LSEHead+BCE vs EffNet+AttHeadLogitSpace+KD). This is NOT a teacher-strength test — it's a **teacher-diversity test**: do PLs from a structurally different but equivalent-quality teacher produce a different student?

### Falsifiable success matrix

Read against the paired **same-codebase baseline** (`distilled-sed-kd` Phase D fold 0 = LB 0.906, reproducible at HEAD per the bit-deterministic Phase D retrain). Per the exp 031 drift investigation: cross-run LB has ~0.007 hardware variance, so paired-within-codebase is the only reliable methodology.

| Bucket | Single-fold LB lift over 0.906 | Reading |
|---|---|---|
| **CONFIRM** | ≥ +0.014 | Antoine-protocol PL transfers; explore Round 2 + Sydorskyi-yoink |
| **PARTIAL** | +0.007 to +0.013 | Smaller than Antoine's +0.007-0.013; still useful for ensemble |
| **WEAK** | +0.001 to +0.006 | Inside seed band; needs 5-fold + LB resubmission to disambiguate |
| **REJECT** | ≤ 0.000 | PL doesn't transfer on this base; reroute to Sydorskyi-yoink directly |

Secondary read across ablations: the **MixSamp ratio × initialization** matrix should expose which protocol variant is the lever. If all 5 ablations land within ±0.003 of each other, the protocol is robust; if they spread > 0.010, the sampler choice is itself a recipe parameter to tune.

## Background

### Why prior PL attempts failed and this one is different

Exp 014/015/025 ran what looked like "PL on unlabeled SS" but with two critical departures from the disclosed Antoine protocol:

| Variable | Prior failed attempts (exp 014/015/025) | This experiment (exp 033) |
|---|---|---|
| PL loss weight | **0.5** (down-weighted vs GT's 1.0) | **1.0** (equal to GT) |
| Batch composition | **Hardcoded 50/50** GT/PL via `StratifiedBatchSampler` (oversamples GT 5× given pool sizes) | **Natural proportions** (uniform sampler) OR ablation of MixSamp ratios that respect the pool-size mismatch |
| Teacher | Perch (cross-arch, mismatch with student) for exp 014/015; mostly v5j-SED (own-arch) for exp 025 with Perch fallback for 44 species | **Two own-arch SED teachers paired** (T1=v5j-4fold, T2=distilled-sed-kd-5fold); no Perch tier C, no cross-arch contamination |
| Base recipe | v5j HGNet+LSEHead (LB 0.906 in 4-fold; ~0.897 retrained single-fold) | **distilled-sed-kd** (LB 0.906 reproducible at HEAD; KD-validated stack matching Tucker recipe) |

The exp 025 verdict was scoped to *"PL with Phase B down-weighting OR Phase C waveform-mixing"*. The 1:1 weighting + uniform sampling protocol that 5 competitors disclose has **never been tested** in our codebase.

### Why we expect this to work

| Argument | Direction |
|---|---|
| 5 independent disclosed data points at 0.93+ all use this exact protocol | Positive |
| Antoine's +0.007 came from **teacher iteration**, not a stronger first-round teacher; suggests the protocol scales | Positive |
| Both our teachers (T1=v5j-4fold cached at LB 0.913, T2=distilled-sed-kd-5fold at LB 0.915) are strong by competitor standards; Antoine's first-round teacher was likely ~0.92-grade. Two own-arch teachers also enable a paired diversity test. | Positive |
| Pool-size math: 127K PL segments vs 26K focal → MixSamp 0.9/0.1 makes the PL pool invisible (each PL seen 0.001× per epoch). Sampler choice is structurally important and a documented variable. | Positive (we're now correcting for this) |
| distilled-sed-kd's `AttHeadLogitSpace` consumes spatial features in logit space — Tucker recipe is engineered for this. PL targets are clip-level soft probabilities → should integrate cleanly with the dual-supervision loss | Positive |
| Pre-flight checks (7/7 PASS) confirm parquet schema + audio + mel + ckpt all wired correctly | Positive |
| Our base lacks XC pretrain that Antoine has; absolute ceiling likely below Antoine's 0.943 | Uncertain |
| Offline-LB anti-correlation extends to single-fold variance (fold 4 result, exp 032); offline metrics during training may underread or overread true LB lift | Uncertain |

## Method

### Approach

Train **distilled-sed-kd recipe** (Tucker recipe + Perch v2 KD, exp 032 base) on **focal + labeled SS + pseudo-labeled unlabeled SS** using the **Antoine protocol** (1:1 loss weighting, no down-weighting, no waveform-mixing). Run 5 paired ablations exploring **initialization × sampler × focal/SC ratio**. Submit the top-performing ablations to LB.

### Setup

| Component | Detail |
|---|---|
| Base recipe | `distilled-sed-kd` (exp 032 Phase D): EffNet-B0 NS-JFT + AttHeadLogitSpace + dual-supervision BCE (0.5/0.5) + Perch v2 KD (α=1.0) + TuckerMelSpec + waveform mixup hard-max (Beta(0.4,0.4)) + SpecAugment (1×freq×10 + 2×time×10) + 25 ep AdamW + cosine + grad_clip=1.0 |
| Teachers (paired comparison) | **(T1) v5j-4fold** — LB-validated 0.913 4-fold via cached `v1/sed_pseudo_labels.parquet` (own-arch HGNet+LSEHead+BCE, generated 2026-04-10).<br>**(T2) `distilled-sed-kd-5fold`** — LB 0.915 5-fold; cached predictions to be generated to `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/`, then PL parquet generated to `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`. Same logical schema as v1 (filename, chunk_idx, 234 species cols) — pure own-arch SED, no tiered Perch fallback. |
| Student | New runs on exp 032 fold 0 split (StratifiedKFold focal + GroupKFold SS, seed=42). Optionally warm-start from `distilled-sed-kd` Phase D `last-v2.ckpt` (LB 0.906) for A3/A4 variants |
| PL parquets | T1: `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet` (exists, pre-flight verified).<br>T2: `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` (to be generated; ~3hr work — inference caching + PL generation). Both pure-SED own-arch (no Perch tier C). |
| PL filter | Drop the 14 fold-0 labeled SS files (168 segments) to prevent val-fold leak. Drop the 52 labeled-SS train files (624 segments) from PL pool to avoid GT/PL label conflict — those files appear in the labeled SS pool with GT labels. Effective PL pool = **127,104 unlabeled SS segments** (10,592 files × 12 windows) per teacher. |
| PL loss weight | **1.0** (1:1 protocol — equal to GT) |
| Mel | TuckerMelSpec (n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000, top_db=80, single per-sample z-score), applied to all 3 sources (focal / labeled SS / unlabeled SS) — verified path-invariant by pre-flight Check 3 |
| Folds | exp 032 5-fold (StratifiedKFold focal + GroupKFold SS, seed=42), fold 0 only for Round 1 |
| Compute | rtx6000 GPU, ~4.5h walltime per ablation (10 ablations × ~4.5h, parallel-submittable on PACE). Plus ~3hr for T2 inference cache + PL generation prerequisite. |
| Code | Branch `mg/exp033-own-teacher-pl`. Extends exp 032's `DistilledSEDFinetuneModule` + `DistilledSEDDataModule` with PL data ingestion (port from exp 025's `data/sed.py` PL machinery). New inference caching script for T2 teacher. |

### Procedure

#### Step 0 — Prerequisites (before any training)

| # | Task | Cost |
|---|---|---|
| 0.1 | Write inference caching script — port `predict_soundscape_tucker_ensemble` (5-fold logit-mean + smooth + sigmoid) to a PACE-GPU offline script that writes `logits.parquet` + `framewise_logits.parquet` per fold to `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/`. | ~1-2 hr coding |
| 0.2 | Run inference caching on all 10,658 SS files (5 folds × 128K windows) | ~30-60 min on rtx6000 |
| 0.3 | Run `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` with the new T2 cache (re-use the same temperature=0.6, power_transform=1.5, max-across-folds config; no Perch tier). Output `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`. | ~30 min |
| 0.4 | Re-run pre-flight checks (Checks 1, 4, 5, 6 on the new parquet) to confirm schema/leakage | ~5 min |

#### Step 1 — Ablation grid (10 paired runs)

Each ablation is the same Tucker recipe (KD on) + Antoine 1:1 PL protocol. Variables: init × sampler × focal/pl share × **teacher**. **A2** is marked ⭐ as the canonical Antoine protocol (random init + uniform sampler = "treat PL as labeled"). All other rows are sensitivity ablations around it.

The "Sampler share" column carries different meanings per sampler:
- **MixSamp** rows: hand-set per-batch fractions `focal/sc/pl` (deterministic; sums to 1)
- **Uniform** rows: pool-size-proportional sampling from `ConcatDataset(focal=28906, sc=624, pl=127104)` → natural ~18/0/82 batch composition

| ID | Init | Sampler | Sampler share (focal/sc/pl) | Per-batch B=64 (focal/sc/pl) | LR | Epochs | Teacher | Hypothesis tested |
|---|---|---|---|---|---|---|---|---|
| **A1.v5j** | random | MixSamp | 0.50 / 0.00 / 0.50 | 32 / 0 / 32 | 5e-4 | 25 | T1 (v5j-4fold) | "Moderate PL emphasis from scratch, T1 teacher." |
| **A2.v5j** ⭐ | random | **Uniform** | ~0.18 / ~0.00 / ~0.82 (natural) | ~12 / ~0 / ~52 | 5e-4 | 25 | T1 | **Pure 1:1 Antoine protocol** from scratch, T1 teacher (canonical recipe). |
| **A3.v5j** | **warm-start** (`last-v2.ckpt`) | MixSamp | 0.20 / 0.00 / 0.80 | 13 / 0 / 51 | **5e-5** | **10** | T1 | "Continued training, PL-heavy mix, T1 teacher." |
| **A4.v5j** | **warm-start** | **Uniform** | ~0.18 / ~0.00 / ~0.82 (natural) | ~12 / ~0 / ~52 | **5e-5** | **10** | T1 | "Warm-start + pure Antoine protocol (compound), T1 teacher." |
| **A5.v5j** | random | MixSamp | 0.70 / 0.00 / 0.30 | 45 / 0 / 19 | 5e-4 | 25 | T1 | "Conservative under-exposure (predicted to underperform A1) — sensitivity check." |
| **A1.kd** | random | MixSamp | 0.50 / 0.00 / 0.50 | 32 / 0 / 32 | 5e-4 | 25 | **T2 (`distilled-sed-kd-5fold`)** | A1 paired across teachers. |
| **A2.kd** ⭐ | random | **Uniform** | ~0.18 / ~0.00 / ~0.82 (natural) | ~12 / ~0 / ~52 | 5e-4 | 25 | T2 | A2 paired across teachers (canonical recipe on T2). |
| **A3.kd** | **warm-start** | MixSamp | 0.20 / 0.00 / 0.80 | 13 / 0 / 51 | **5e-5** | **10** | T2 | A3 paired across teachers. |
| **A4.kd** | **warm-start** | **Uniform** | ~0.18 / ~0.00 / ~0.82 (natural) | ~12 / ~0 / ~52 | **5e-5** | **10** | T2 | A4 paired across teachers. |
| **A5.kd** | random | MixSamp | 0.70 / 0.00 / 0.30 | 45 / 0 / 19 | 5e-4 | 25 | T2 | A5 paired across teachers. |

The per-batch counts come from the datamodule's `MixSamp per_src=[focal, sc, pl]` log and from `batch_size × pool_share` for uniform. Note that the labeled-SS pool (624 segments) is never sampled directly in exp 033 — `sc_share=0.0` for all MixSamp ablations, and uniform's 624/156634 ≈ 0.4% rounds to 0 per batch. Labeled-SS audio still appears as **ground-truth focal anchors** in the focal × labeled-SS mixup (the Tucker recipe's `focal_sc_mixup_prob=0.5`), just not as a primary sampling source here.

Stretch (decided after Round 1):
- **A0.ensemble**: best-performing protocol (TBD) using **mean(PL_T1, PL_T2)** as targets — tests ensemble-of-teachers.

#### Pair structure for attribution

| Comparison | Variable isolated |
|---|---|
| A1.v5j ↔ A2.v5j, A3.v5j ↔ A4.v5j, A1.kd ↔ A2.kd, A3.kd ↔ A4.kd | **Sampler** (MixSamp vs Uniform) — 4 paired tests |
| A1.v5j ↔ A3.v5j, A2.v5j ↔ A4.v5j, A1.kd ↔ A3.kd, A2.kd ↔ A4.kd | **Init** (random vs warm-start) — 4 paired tests |
| A1.v5j ↔ A1.kd, A2.v5j ↔ A2.kd, A3.v5j ↔ A3.kd, A4.v5j ↔ A4.kd, A5.v5j ↔ A5.kd | **Teacher** (T1 vs T2) — 5 paired tests |
| A1.v5j vs A5.v5j, A1.kd vs A5.kd | **Sanity** — over-exposure vs under-exposure on fixed teacher |

#### LB submission strategy under daily cap (5/day)

| Day | Action |
|---|---|
| Day 1 morning | Submit T2 inference caching + PL generation prereq (Step 0) |
| Day 1 afternoon | Submit all 10 training jobs to PACE (parallel) |
| Day 2 morning | All 10 jobs complete (~4.5h); run D.5 e2e validation on all 10 ckpts. Rank by e2e val/auroc. |
| Day 2 | Submit **top 5 by e2e** to LB. Read results when they land. |
| Day 3 | Submit remaining 5 (or stretch A0 if Day 2 results suggest ensemble-of-teachers is promising). |
| Day 4 | Apply decision matrix. Plan Round 2 if CONFIRM. |

### Pool-size sampler math (already validated in design discussion)

With 127K PL segments + 26K focal samples and `nst = (focal + sc) / batch_size = 2390 steps/epoch`:

| Setup | Focal exposure/epoch | PL exposure/epoch |
|---|---:|---:|
| Tucker 0.9/0.1 (original ratio, transplanted) | 5.3× | 0.001× |
| A5: 0.7/0.3 | 4.1× | 0.003× |
| A1: 0.5/0.5 | 2.9× | 0.006× |
| A3: 0.2/0.8 | 1.2× | 0.010× |
| A2 / A4: Uniform | 1.0× | 1.0× |

Tucker's published 0.9/0.1 was tuned for **792 SS segments**. Naive transplant with 127K SS is structurally broken — the PL pool becomes invisible. A2/A4 (uniform) is the only setup that achieves Antoine's "treat as labeled" semantics; A3 (warm-start + 0.2/0.8) tests whether MixSamp's focal anchor still has value when the model is already trained.

### Variables

- **Independent (between ablations within exp 033)**:
  - **Init**: random Tucker-init vs warm-start from `distilled-sed-kd` Phase D `last-v2.ckpt`
  - **Sampler**: MixSamp (focal/SC ratio knob) vs Uniform (natural proportions over ConcatDataset)
  - **Focal/SC ratio** (when MixSamp): 0.5/0.5, 0.7/0.3, 0.2/0.8
  - **Teacher**: T1 (v5j-4fold cached, LB 0.913 4-fold) vs T2 (`distilled-sed-kd-5fold`, LB 0.915 5-fold)
- **Independent (between exp 033 and exp 025)**: loss_weight (1.0 vs 0.5), sampler family (uniform/MixSamp vs StratifiedBatchSampler hardcoded 50/50), teacher purity (own-arch SED vs tiered SED+Perch). Intentional multi-variable correction.
- **Dependent**: paired LB lift, val/auroc (with caveats from offline-LB anti-correlation history), val/cmap, e2e val/auroc (D.5 pre-flight)
- **Controlled (within exp 033)**: same Tucker recipe, same fold 0 split, same seed=42, same SLURM partition. Teacher swaps in T1/T2; everything else identical within each pair.

## Evaluation

### Metrics

| Metric | Description | Role |
|---|---|---|
| Paired LB lift | LB(student) − LB(distilled-sed-kd Phase D 0.906) | **Primary read** — Antoine reports +0.007-0.013 from this lever |
| Cross-ablation LB spread | std(LB across A1-A5) | **Secondary read** — characterizes protocol robustness |
| val/auroc (Tucker macro AUC) | Training-side, on fold 0 SS val pool | Sanity check; not the LB predictor (offline-LB anti-correlation) |
| End-to-end val/auroc (D.5) | Kernel inference path on fold 0 val SS | **Mandatory pre-flight before LB**; gap to training-side must be < 0.02 |
| train/cls_loss, train/dist_loss | Training trajectory health | Sanity (KD still working) |

### Baseline

- **Primary baseline**: `distilled-sed-kd` Phase D fold 0 = **LB 0.906** (paired-within-codebase, reproducible at HEAD)
- **Secondary baseline (historical)**: distilled-sed-kd-5fold = 0.915 LB (Phase 2.A ensemble; informative but ensemble is a different scope from single-fold PL)
- **Optimistic target**: 0.920+ (Antoine's +0.014 single-fold lift band lower edge; reaches CONFIRM)

### Success Criteria

See § Falsifiable success matrix above. CONFIRM (≥ +0.014) → proceed to Round 2 (teacher iteration) and/or Sydorskyi-yoink layering. PARTIAL (+0.007-0.013) → useful for ensemble with exp 023 (0.929). WEAK (+0.001-0.006) → ambiguous, need 5-fold to disambiguate. REJECT (≤ 0) → reroute to Sydorskyi-yoink as primary.

## Limitations

- **No XC pretrain**: Antoine's recipe is XC pretrain + own-teacher PL. We have only the PL half. Without XC, our backbone's in-domain prior is weaker → expected ceiling 0.92-0.93 single-fold, not 0.94+. XC layering (Sydorskyi-yoink) is reserved for follow-up if exp 033 confirms.
- **Single-fold for Round 1**: 5-fold extension reserved for if Round 1 CONFIRMs. Per exp 030/031 drift investigation, single-fold has ~0.007-0.013 cross-node LB variance — WEAK outcomes can't be reliably disambiguated without 5-fold or seed sweep.
- **Single-round PL**: Antoine's lift came from iterating teacher (Round 1 student → Round 2 teacher). We're testing only Round 1 first. Round 2 deferred until LB lands.
- **Teacher LB delta is small (Δ = +0.002)**: T1 (v5j-4fold) and T2 (`distilled-sed-kd-5fold`) are ~equal in quality. This is a teacher-architecture-diversity test, NOT a teacher-strength test. Antoine's teacher iteration was a much larger quality delta — we don't expect to replicate that mechanism, only the diversity benefit.
- **Offline-LB anti-correlation history**: 23+ data points in this project show offline metrics don't predict LB rank. We mandatory-gate LB submission on D.5 end-to-end validation but expect the LB number to still be the only reliable read.
- **Daily LB cap (5/day)**: 10 ablations cannot all be submitted on a single day. Top-5-by-e2e ranking determines Day 2 submissions; remainder go on Day 3. Risk: the e2e ranking might not perfectly correlate with LB (consistent with offline-LB anti-correlation history). Mitigation: D.5 gap (training vs e2e) has been more LB-correlated than training-side val/auroc in exp 032 (0.004-0.011 gap matched 0.906 LB landing).
- **T1 cached PL parquet was generated 2026-04-10**: pre-flight Check 1 confirms filename + species set still align with current data. T2 parquet (to be generated in Step 0) inherits the same data setup.
- **T2 inference caching prereq adds ~3hr** before the first training. Sequential dependency for the 5 T2-teacher ablations; T1 ablations can start immediately.

## Next-step plan (post-Round-1)

| Round 1 outcome | Round 2 plan |
|---|---|
| **CONFIRM** (LB ≥ 0.920) | Regenerate PLs using Round-1 student as teacher → train Round-2 student → submit (Antoine's iteration). Parallel track: layer Sydorskyi-yoink for XC base. |
| **PARTIAL** (+0.007-0.013) | Ensemble with exp 023 (0.929) for combined LB. Consider XC pretrain as the next lever rather than another PL round. |
| **WEAK** (+0.001-0.006) | Extend to 5-fold to disambiguate from single-fold seed noise. |
| **REJECT** (≤ 0) | Pivot to Sydorskyi-yoink as primary remaining lever. Document the protocol findings for archival. |


---

# § 5. COMPANION DOC: Exp 033 results (REJECT verdict + v3 status)

*Source: `user/murilo/experiments/033-own-teacher-pl/results.md`*

---
experiment: "033-own-teacher-pl"
status: reject-protocol-fails-on-base
date: 2026-05-27
author: Murilo Gustineli
---

# Results: Own-teacher pseudo-labels on distilled-sed-kd

**Date Updated**: 2026-05-27 PM (3/5 Day-1 LB submissions landed — all REJECT)
**Author**: Murilo Gustineli

## Status: REJECT — 1:1 own-teacher PL protocol does NOT lift LB on the distilled-sed-kd base

3 LB submissions across both teachers, including the canonical Antoine cell + the
paired teacher-axis test, all regressed −0.007 to −0.020 vs the 0.906 baseline.
The teacher-axis paired Δ on the warm+Uniform cell is effectively zero
(0.899 vs 0.897 LB), refuting the proposal's headline teacher-diversity claim
at LB scale. **The protocol does not transfer on this base regardless of teacher.**

### LB summary (2026-05-27 PM)

| Ablation | Cell | Teacher | LB | Δ vs 0.906 | Decision-matrix bucket |
|---|---|---|---:|---:|:---:|
| `2026-exp033-a4-kd` | warm + Uniform | T2 (distilled-sed-kd-5fold) | **0.899** | **−0.007** | REJECT (edge of cross-hardware noise) |
| `2026-exp033-a2-kd` ⭐ | random + Uniform (canonical Antoine) | T2 | **0.886** | **−0.020** | REJECT (firmly inside REJECT band) |
| `2026-exp033-a4-v5j` | warm + Uniform | T1 (v5j-4fold) | **0.897** | **−0.009** | REJECT |
| `2026-exp033-a3-kd` | warm + MixSamp 0.2/0.8 | T2 | (not submitted) | — | — |
| `2026-exp033-a3-v5j` | warm + MixSamp 0.2/0.8 | T1 | (not submitted) | — | — |

Per `proposal.md § Falsifiable`: CONFIRM ≥ +0.014, PARTIAL +0.007 to +0.013,
WEAK +0.001 to +0.006, REJECT ≤ 0. **All 3 landed in REJECT.** The remaining
2 Day-1 slots (a3-kd, a3-v5j) and the 5 Day-2 cells (a1, a5 × teachers) are
unlikely to surprise positive given the trajectory and were not submitted.

### Paired Δ readings

- **Teacher axis (a4-kd vs a4-v5j, same protocol cell)**: Δ = 0.899 − 0.897 = **+0.002 LB** (T2 vs T1). Inside cross-hardware noise. The proposal's headline teacher-diversity hypothesis (T2 wins offline by +0.011 auroc) does NOT transfer to LB. Refutes the "T2 is the teacher to keep iterating with" claim from this morning's read.
- **Protocol axis (a4-kd warm vs a2-kd random, same teacher)**: a2-kd canonical Antoine LB 0.886 vs warm-variant a4-kd LB 0.899. Δ = **−0.013** (canonical < warm-start). Inverts the offline reading (a4-kd e2e 0.955 vs a2-kd e2e 0.940 = warm > canonical). Another offline-LB anti-correlation data point.

### Decision-matrix read

| Bucket | LB threshold | Result |
|---|---|---|
| CONFIRM | ≥ +0.014 | ❌ |
| PARTIAL | +0.007 to +0.013 | ❌ |
| WEAK | +0.001 to +0.006 | ❌ |
| **REJECT** | **≤ 0** | **✅ all 3 cells** |

**Verdict per proposal**: pivot to Sydorskyi-yoink (XC pretrain) as the primary
remaining lever. The own-teacher PL protocol is refuted on the distilled-sed-kd
base.

### Offline-LB anti-correlation: data point #24

This adds 3 paired data points to the project's documented offline-LB
anti-correlation pattern (24+ total now):

| Cell | Offline e2e val/auroc | LB | Offline rank | LB rank |
|---|---:|---:|---:|---:|
| a4-kd | 0.955 (rank 1/10) | 0.899 (rank 2 of 3) | strongest | middle |
| a2-kd ⭐ | 0.940 (rank 3/10) | 0.886 (rank 3 of 3) | middle | worst |
| a4-v5j | 0.940 (rank 4/10) | 0.897 (rank 1 of 3) | middle | best |

**The offline-ranked best (a4-kd) was middle on LB; the offline-ranked weakest
of the three (a4-v5j) was the LB best.** Same pattern as exp 032 fold 4 (+0.023
offline → −0.003 LB) and 22+ prior recipe data points.

---

## Status: TRAINING COMPLETE (10/10 fold-0 ablations) — D.5 e2e validation + LB submission pending (HISTORICAL — superseded by LB results above)

All 10 ablations of the Antoine-style 1:1 own-teacher PL protocol on top of the
Phase D `distilled-sed-kd` recipe finished overnight 2026-05-26 → 2026-05-27 on
gpu-rtx6000. Wall-time matched predictions (25-ep runs ~6.5h, 10-ep runs ~2.6h
at `num_workers=8`). Every run reached its full `max_epochs` (early stopping was
disabled with `patience: 999`); every `last.ckpt` written; every `metrics.json`
backfilled.

The slurm FAIL emails are a false alarm — `_job.sh:127` called a missing
`save_metrics.py` that wasn't ported from exp 032 in Phase E. `Training
complete!` line is present in all 10 logs; `outputs/test_results/*.json` exist
for all 10. Fix landed in this session (commit pending): new
`save_metrics.py` with `_common.yaml` inheritance, plus backfilled metrics for
all 10 runs. Future runs will exit 0.

**Caveat going in**: offline val/auroc has been **anti-correlated with LB** for
23+ data points in this project (most recently exp 032 fold 4: best offline
0.931 → near-worst LB 0.903). The offline rankings below indicate *direction*
only; D.5 end-to-end validation + actual LB submission is what closes the loop.

## Offline metrics (val on fold-0 held-out SS pool)

Baseline (`distilled-sed-kd` Phase D fold 0, LB **0.906**): val/auroc 0.908,
val/cmap 0.692, best val/cmap 0.709 (at epoch 10).

### Ranked by training-side val/auroc (last epoch)

| Rank | Ablation | Init | Sampler | Teacher | val/auroc | val/cmap | best val/cmap | Best Ep | Walltime | SLURM | W&B |
|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| 1 | `2026-exp033-a3-kd` | warm-start | MixSamp 0.2/0.0/0.8 | T2 | **0.9486** | 0.8311 | 0.8350 | 7 | 2h40m | 9191660 | `ax6t5qdz` |
| 2 | `2026-exp033-a4-kd` | warm-start | Uniform | T2 | **0.9477** | **0.8323** | 0.8335 | 7 | 2h38m | 9191661 | `htfgco92` |
| 3 | `2026-exp033-a2-kd` ⭐ | random | Uniform | T2 | 0.9390 | 0.8182 | 0.8201 | 3 | 6h29m | 9191659 | `1ye9s2ni` |
| 4 | `2026-exp033-a2-v5j` ⭐ | random | Uniform | T1 | 0.9362 | 0.7923 | 0.7977 | 14 | 7h20m | 9190540 | `4y0h7h1a` |
| 5 | `2026-exp033-a1-kd` | random | MixSamp 0.5/0.0/0.5 | T2 | 0.9357 | 0.8242 | 0.8390 | 12 | 6h46m | 9191658 | `2cg0kcux` |
| 6 | `2026-exp033-a4-v5j` | warm-start | Uniform | T1 | 0.9352 | 0.8043 | 0.8058 | 7 | 2h39m | 9190544 | `dl7pz6kv` |
| 7 | `2026-exp033-a3-v5j` | warm-start | MixSamp 0.2/0.0/0.8 | T1 | 0.9345 | 0.8061 | 0.8095 | 7 | 2h38m | 9190542 | `f781pbyz` |
| 8 | `2026-exp033-a1-v5j` | random | MixSamp 0.5/0.0/0.5 | T1 | 0.9288 | 0.7821 | 0.7859 | 11 | 7h25m | 9190539 | `ct3by7ax` |
| 9 | `2026-exp033-a5-kd` | random | MixSamp 0.7/0.0/0.3 | T2 | 0.9285 | 0.7915 | 0.7954 | 10 | 6h45m | 9191662 | `vn3vyrp5` |
| 10 | `2026-exp033-a5-v5j` | random | MixSamp 0.7/0.0/0.3 | T1 | 0.9113 | 0.7586 | 0.7595 | 21 | 6h41m | 9190545 | `opv1942s` |

⭐ = canonical Antoine protocol (random init + uniform sampler over ConcatDataset).
T1 = `v5j-4fold` teacher (`v1/sed_pseudo_labels.parquet`, LB 0.913).
T2 = `distilled-sed-kd-5fold` teacher (`v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`, LB 0.915).

All 10 ablations land above the baseline 0.908 val/auroc; even the worst
(a5.v5j 0.911) is +0.003 over baseline. **Offline rankings are NOT a reliable LB
predictor on this project** — direction only, D.5 + LB submission required.

### Paired Δ across the three axes

#### Teacher axis: T2 (`distilled-sed-kd-5fold`) vs T1 (`v5j-4fold`)

5 paired tests across the same protocol cell:

| Pair | T1 val/auroc | T2 val/auroc | Δ auroc | T1 val/cmap | T2 val/cmap | Δ cmap |
|---|---:|---:|---:|---:|---:|---:|
| A1 (random / MixSamp 0.5/0.5) | 0.9288 | 0.9357 | **+0.0069** | 0.7821 | 0.8242 | **+0.0421** |
| A2 ⭐ (random / Uniform) | 0.9362 | 0.9390 | +0.0028 | 0.7923 | 0.8182 | +0.0259 |
| A3 (warm / MixSamp 0.2/0.8) | 0.9345 | 0.9486 | **+0.0142** | 0.8061 | 0.8311 | +0.0250 |
| A4 (warm / Uniform) | 0.9352 | 0.9477 | **+0.0125** | 0.8043 | 0.8323 | +0.0280 |
| A5 (random / MixSamp 0.7/0.3) | 0.9113 | 0.9285 | **+0.0172** | 0.7586 | 0.7915 | **+0.0330** |
| **mean** | 0.9292 | 0.9399 | **+0.0107** | 0.7887 | 0.8195 | **+0.0308** |

**T2 wins all 5 pairs on auroc AND cmap.** Despite the two teachers having
~equal LB (0.913 vs 0.915 = +0.002), the T2 student is uniformly stronger
offline. This is consistent with a **teacher-diversity benefit** — T2's
distilled-sed-kd is architecturally + recipe-distinct from the student
(same backbone family, but different LB-validated soft-label distribution than
the v5j ensemble). If the LB tracks the offline ranking even directionally,
T2 is the teacher to keep iterating with.

#### Sampler axis: Uniform vs MixSamp (within init + teacher)

4 paired tests:

| Pair | MixSamp ratio | MixSamp val/auroc | Uniform val/auroc | Δ (Uniform − MixSamp) |
|---|---|---:|---:|---:|
| A1.v5j vs A2.v5j (random / T1) | 0.5/0.5 | 0.9288 | 0.9362 | **+0.0074** |
| A1.kd vs A2.kd (random / T2) | 0.5/0.5 | 0.9357 | 0.9390 | +0.0033 |
| A3.v5j vs A4.v5j (warm / T1) | 0.2/0.8 | 0.9345 | 0.9352 | +0.0007 |
| A3.kd vs A4.kd (warm / T2) | 0.2/0.8 | 0.9486 | 0.9477 | −0.0009 |

**Uniform wins on random init; flat on warm-start.** This is consistent with
the proposal's prediction that pool-size structure matters more than ratio
names — uniform sampling over `ConcatDataset(focal=29K, sc=624, pl=127K)`
gives natural proportions ~18/0.4/82 per batch, very close to A3/A4's hand-set
0.2/0.0/0.8, so the two converge on warm-start. From scratch, the implicit PL
emphasis of uniform (82% PL exposure) appears stronger than A1's 50% MixSamp.

#### Init axis: warm-start vs random (within sampler + teacher)

4 paired tests (note: warm-start runs use 10 ep at lr=5e-5; random runs use
25 ep at lr=5e-4 — confounded with epochs/lr, see Limitations):

| Pair | Random val/auroc | Warm-start val/auroc | Δ (Warm − Random) |
|---|---:|---:|---:|
| A1.v5j vs A3.v5j (MixSamp / T1) | 0.9288 | 0.9345 | **+0.0057** |
| A1.kd vs A3.kd (MixSamp / T2) | 0.9357 | 0.9486 | **+0.0130** |
| A2.v5j vs A4.v5j (Uniform / T1) | 0.9362 | 0.9352 | −0.0010 |
| A2.kd vs A4.kd (Uniform / T2) | 0.9390 | 0.9477 | **+0.0086** |

**Warm-start helps on 3/4 pairs; flat on T1 + uniform.** The strongest lift is
A1.kd → A3.kd (+0.013) — warm-starting from the Phase D ckpt + heavier PL share
under MixSamp + T2 teacher is the best stack offline. With early-stop-by-cmap
on the 10-ep runs (best at ep 7), the model converges quickly and doesn't
need 25 ep. This is the canonical Antoine "iterate teacher on existing
student" pattern.

#### PL-exposure sanity: A1 (50% PL) vs A5 (30% PL)

| Pair | A1 (PL 50%) | A5 (PL 30%) | Δ (A1 − A5) |
|---|---:|---:|---:|
| v5j (T1) | 0.9288 | 0.9113 | **+0.0175** |
| kd (T2) | 0.9357 | 0.9285 | +0.0072 |

**More PL exposure helps**, as predicted. A5 was the "conservative
under-exposure" sanity check and lands at the bottom of both teacher families.
Confirms the proposal's pool-size analysis: Tucker's published 0.9/0.1 ratio
(transplanted naively) would be even worse — PL becomes structurally invisible
to the student.

### Per-batch composition × teacher heat map (val/auroc)

|                                          | T1 (v5j-4fold) | T2 (distilled-sed-kd-5fold) | Δ T2−T1 |
|---|---:|---:|---:|
| **A5** — random / MixSamp 0.7/0.0/0.3 (45/0/19) | 0.9113 | 0.9285 | +0.0172 |
| **A1** — random / MixSamp 0.5/0.0/0.5 (32/0/32) | 0.9288 | 0.9357 | +0.0069 |
| **A2** ⭐ — random / Uniform (~12/~0/~52)        | 0.9362 | 0.9390 | +0.0028 |
| **A4** — warm-start / Uniform (~12/~0/~52)      | 0.9352 | 0.9477 | +0.0125 |
| **A3** — warm-start / MixSamp 0.2/0.0/0.8 (13/0/51) | 0.9345 | 0.9486 | +0.0142 |

Reading top-to-bottom: as PL-per-batch share grows from 30% → ~82% (under
random init) the student's val/auroc grows monotonically (+0.025 on T1, +0.011
on T2). Warm-start at the high-PL end (A3, A4) extracts another +0.009-0.013
over random-init at the same PL share. The protocol's two strongest levers are
(a) ensure PL-per-batch ≥ ~50% and (b) warm-start when possible.

## Training dynamics

### Best-checkpoint epoch distribution

| Epoch range | Count | Ablations |
|---|---:|---|
| ≤ 3 | 1 | a2.kd (ep 3) |
| 7 | 4 | a3.kd, a3.v5j, a4.kd, a4.v5j (all warm-start runs converged at ep 7) |
| 10–14 | 4 | a1.kd (12), a1.v5j (11), a2.v5j (14), a5.kd (10) |
| 21 | 1 | a5.v5j |

Warm-start runs (a3/a4 × T1/T2) converge by epoch 7 — past the 10-epoch
budget they would not improve (this is consistent with the proposal's
"10-epoch warm-start fine-tune" sizing). Random-init runs typically peak
around epoch 10-14. `a5.v5j` (the conservative under-PL'd config) peaks
latest at epoch 21, consistent with the under-PL signal being slower to
saturate the focal-rich training distribution.

A2.kd's epoch-3 best is a likely false positive (val/cmap monitor on a small
held-out SS pool is noisy at the start); the last-epoch val/cmap (0.8182)
matches the trend on this row well.

### Training-side losses (sanity)

Final-epoch `train/cls_loss` is in the ~0.02 range across all 10 runs (vs
Tucker's recipe published 0.02-0.03 band). `train/dist_loss` (Perch v2 KD MSE)
plateaus at 0.002-0.003 (Tucker's published floor). No NaN or instability
anywhere; the dual-supervision loss + per-sample loss_weight integration is
healthy at scale.

## Compute summary

| Cell | Walltime (per run) | × 5 runs | Notes |
|---|---|---|---|
| 25-ep ablations (A1/A2/A5) | 6.5–7.4h | ~33h | matches 20 min/epoch × 25 ep + ~10 min warmup |
| 10-ep ablations (A3/A4) | 2.6–2.7h | ~13h | matches 16 min/epoch × 10 ep at lr=5e-5 |
| **All 10 ablations (parallel)** | — | wall ~7.5h | gpu-rtx6000, 5 nodes (15 + 19 + 21 + 7 + 11/3) |

Total compute consumed (T1 wave was the cancelled `num_workers=0` jobs +
relaunched fix; sunk ~20 GPU-hours on the cancelled runs). Effective parallel
overnight: ~7.5h wall-clock for all 10 ablations.

`num_workers=8` regression fix (commits `ecc3ccb` + `87d668d`) was the key
performance win — `num_workers=0` inherited from Tucker's notebook would have
made every job 4× slower, missing the overnight window.

## Artifacts

| Type | Location |
|---|---|
| Configs | `configs/_common.yaml` + `configs/a{1..5}_{v5j,kd}.yaml` |
| Logs | `logs/slurm_a{1..5}_{v5j,kd}-fold0_91*.log` (10 files) |
| Test results JSON | `outputs/test_results/2026-exp033-a{1..5}-{v5j,kd}-fold0.json` (10 files) |
| Structured metrics | `artifacts/training/a{1..5}_{v5j,kd}/fold0/metrics.json` (10 files; backfilled 2026-05-27) |
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/033-own-teacher-pl/checkpoints/2026-exp033-a{1..5}-{v5j,kd}/fold0/{best-epochepoch=NN.ckpt, last.ckpt}` |
| T2 PL parquet | `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` (210,375 PLs > 0.5; 166 species; this session) |
| T2 inference cache | `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/fold{0..4}/logits.parquet` (639,480 rows; this session) |
| W&B | `dsgt-arc/birdclef-2026-exp033/{ct3by7ax,2cg0kcux,4y0h7h1a,1ye9s2ni,f781pbyz,ax6t5qdz,dl7pz6kv,htfgco92,opv1942s,vn3vyrp5}` |
| Code | branch `mg/exp033-own-teacher-pl` |

### T1 ckpt name caveat

For the T1 (`v5j`) ablations, `last.ckpt` in each fold0 directory is from the
**cancelled `num_workers=0` job** (epoch 4–5 snapshot, sat there from
~21:35 on 2026-05-26). The **actual final checkpoint** is `last-v1.ckpt`
(epoch 24 or 9, from the relaunched run with `num_workers=8`). D.5 + ONNX
export must point at `last-v1.ckpt` for the 5 T1 runs; T2 runs use the
standard `last.ckpt`.

## Decision-matrix read (OFFLINE ONLY — LB closes the loop)

Per `proposal.md § Falsifiable` (LB-only buckets; baseline LB 0.906):

| Bucket | LB threshold | Interpretation |
|---|---|---|
| **CONFIRM** | LB ≥ 0.920 (+0.014) | Antoine-protocol PL transfers; explore Round 2 (teacher iteration) and/or Sydorskyi-yoink XC base |
| **PARTIAL** | LB +0.007 to +0.013 | Useful for ensemble with exp 023 (0.929) |
| **WEAK** | LB +0.001 to +0.006 | Inside seed band; need 5-fold or seed sweep to disambiguate |
| **REJECT** | LB ≤ 0.906 | Reroute to Sydorskyi-yoink as primary |

Offline val/auroc lifts over the Phase D baseline (0.908) are **+0.020 to
+0.041** across the 10 ablations. Per exp 032's anti-correlation evidence
(fold 4: +0.023 offline → −0.003 LB), this does NOT translate 1:1 to LB. The
LB submission itself is the only reliable read.

The pre-LB ranking for the 5/day cap is therefore best made on **D.5
end-to-end val/auroc**, not training-side val/auroc — D.5 was a tighter
LB predictor in exp 032 Phase D (gap of +0.004 to +0.011 between e2e and
training-side, consistent with the 5-tap smoothing trick).

## Phase 5 — Sydorskyi-divergence diagnostic (in flight, 2026-05-27 EOD)

After the REJECT verdict, audit identified that exp 033 implementation diverges
from the only published 0.93+ PL recipe (Sydorskyi 2025 2nd place) on **16 of 20
axes** — full table in
`user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md`.
The REJECT verdict therefore can't cleanly be read as a refutation of the
protocol; we never actually ran the protocol.

### Test 1: a2_kd_v3 (chunk discard + per-class trim)

**Status**: training launched 2026-05-27 EOD. LB result expected 2026-05-28
morning.

| Setup | Value |
|---|---|
| Run name | `2026-exp033-a2-kd-v3` |
| Base config | Canonical Antoine (random init + Uniform sampler + T2 teacher + 25 ep) |
| Diff vs v2 (LB 0.886) | PL parquet has Sydorskyi-canonical filters applied: `min_max_prob=0.5` (chunk discard) + `trim_min_prob=0.1` (per-class trim) |
| PL parquet | `v3-distilled-sed-kd-5fold-trim01-discard05/sed_pseudo_labels.parquet` (85K chunks, 33% chunk-discard rate, 98% per-cell trim) |
| Note on discard rate | Below Sydorskyi's ~83% — our existing T=0.6 + P^1.5 sharpening + max-across-folds inflates max probs before the filter. v4 (Sydorskyi-faithful PL gen) is scaffolded as fallback. |

**Decision criteria** (see `diagnostic-tests.md § Step 1.6`):

| LB outcome | Reading |
|---|---|
| > 0.913 | trim+discard explain REJECT; protocol recoverable on our base |
| 0.906-0.913 | Modest lift; deeper protocol fixes (focal × PL fusion) may help |
| 0.886-0.906 | Filters alone don't help; protocol mechanism likely the gap |
| < 0.886 | Worse than original — investigate parquet directly |

**Phase 5 closure decision (recorded 2026-05-27 EOD)**: regardless of v3
outcome, the next major effort goes to **Sydorskyi-yoink (exp 034)** rather
than continuing PL diagnostics (Test 2 / Test 3). v3 is the final data point
on exp 033's PL track. Rationale: even best-case PL lift lands at ~0.92, not
0.94; the disclosed 0.94+ paths all rely on a data move, not a recipe move.

## Next Steps

D.5 + 3-cell LB landed; verdict is REJECT. v3 (Test 1) is the final PL
data point. Next major effort goes to **Sydorskyi-yoink (exp 034)** — XC
pretrain via Ali Ozan's path (download Sydorskyi 2025 published backbone,
attach own head, fine-tune on 2026). Decision finalized after v3 LB lands +
deep research overnight output (`notes/2026-05-27-deep-research-prompt.md`)
is read.

1. **Closeout**: ✅ results.md updated · MODEL-RUNS.md updated · tasks.md
   closeout phase pending PR.
2. **Pivot**: scope `experiments/034-sydorskyi-yoink-xc-pretrain/` (or
   similar). Goal = layer XC pretrain (Sydorskyi's recipe) on top of the
   distilled-sed-kd recipe to unlock the lever Antoine has but we don't.
3. **Don't run** Day-1 a3-kd + a3-v5j or the 5 Day-2 ablations — the LB cap
   slots are better spent on the pivot's first iteration.

### Historical (pre-LB plan)

The original ladder, kept here for archival:

1. D.5 end-to-end validation on all 10 ckpts (✅ done — all 10 PASS).
2. Rank by e2e val/auroc → top 5 to LB Day 1 (3 submitted, 2 abandoned).
3. LB submit, apply decision matrix → REJECT.
4. CONFIRM → Round 2 + Sydorskyi-yoink. (Not triggered.)
5. PARTIAL → ensemble with exp 023. (Not triggered.)
6. WEAK → 5-fold. (Not triggered.)
7. REJECT → pivot to Sydorskyi-yoink directly. **← active path**

## Methodology notes

- **Slurm FAIL email gotcha**: All 10 trained to completion (`Training
  complete!` + JSON + `last.ckpt` + W&B upload all present in every log).
  `save_metrics.py` was missing from exp 033 (Phase E omission), causing
  `_job.sh` to exit non-zero → `--mail-type=END,FAIL` fired. Fix landed this
  session: `save_metrics.py` ported with `_common.yaml` deep-merge logic;
  10 backfills run on the head node.
- **Early stopping disabled**: `_common.yaml` set `patience: 999` on `val/cmap`
  monitor. All runs reached `max_epochs` (25 or 10). The best ckpt selection
  via `save_top_k: -1` + monitor `val/cmap` correctly identifies the
  per-epoch best (see Best Ep column above).
- **Offline-LB anti-correlation**: per exp 032 fold 4 (+0.023 offline → −0.003
  LB) and 22+ prior data points, we don't expect the offline rankings to
  translate 1:1. The +0.030 offline lift over baseline across the canonical
  Antoine protocol (A2.kd / A4.kd) is encouraging but not guarantee.
- **Confounds in the init axis**: warm-start runs (A3/A4) also use lr=5e-5
  + 10 epochs + warmup_epochs=0; random-init runs use lr=5e-4 + 25 epochs +
  warmup_epochs=2. The "warm-start helps" reading is a compound observation,
  not a clean isolated lever. The proposal acknowledges this — the cell-grid
  was designed around competitor recipes, not orthogonal ablations.


---

# § 6. COMPANION DOC: Exp 032 results (Tucker replication, PARTIAL)

*Source: `user/murilo/experiments/032-tucker-distilled-sed/results.md`*

---
experiment: "032-tucker-distilled-sed"
status: phase-2-partial-replication
date: 2026-05-25
author: Murilo Gustineli
---

# Results: Tucker distilled-SED full replication

**Date Completed (Phase D)**: 2026-05-24
**Date Updated (Phase 2 LB)**: 2026-05-25
**Author**: Murilo Gustineli

## Status: PARTIAL REPLICATION — within ~0.002 of Tucker's publicly-published 0.917 5-fold LB; the "0.93+ with PP" claim is out of reach without his unpublished post-processing

- **Phase D (LB landed)**: KD lever transfers — distilled_sed_kd single-fold LB **0.906** (matches v5j-original; paired Δ vs control +0.050).
- **Phase 2 (LB landed)**: 5-fold KD ensemble **0.915 LB** (+0.009 over Phase D fold 0; −0.002 below Tucker's published 0.917). HGNet+KD ablation **0.901 LB** (−0.005 vs EffNet fold 0) → **backbone was NOT the bottleneck**. Fold 4 single-fold diagnostic **0.903 LB** (despite strongest offline val/auroc 0.931 = +0.023 over fold 0) → **offline-LB anti-correlation; single-fold LB band is narrow (~0.003)**.
- **Verdict**: replicated what's publicly documented; cannot fully reach Tucker's 0.917 within seed noise let alone his 0.93+ stretch number. The +0.009 single→5-fold ensemble lift is pure variance-reduction (Gaussian smoothing), not picking up better folds. Next levers: ensemble with exp 023 (0.929) and/or port site/hour priors.

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

## Phase 2 (2026-05-25 EOD): 5-fold KD ensemble + HGNet+KD ablation — TRAINING DONE, LB PENDING

Per Phase D CONFIRM verdict, Phase 2 extends in two directions:
- **2.A 5-fold KD ensemble**: train folds 1-4 alongside Phase D fold 0 (LB 0.906). Tucker's published 0.917 LB = 5-fold + 5-tap Gaussian smoothing across windows.
- **2.D HGNet+KD ablation**: single-fold (fold 0) backbone swap `tf_efficientnet_b0.ns_jft_in1k` → `hgnetv2_b0.ssld_stage2_ft_in1k` (v5j's backbone) with `use_post_head: true → false`. Tests whether the −0.041 Phase D control-vs-v5j gap is backbone-driven.

All 5 runs trained 25 ep + 2 ep warmup + cosine, AdamW lr=5e-4 wd=1e-4, fp16, grad_clip=1.0, BCE dual-supervision 50/50, blend 50/50, Beta(0.4,0.4) hard-max mixup, SpecAugment, 90/10 focal/SS MixSamp, seed=42, `save_top_k: -1`. Configs: `distilled_sed_kd.yaml` (folds 1-4 with `train.py --fold N`) + `distilled_sed_kd_hgnet.yaml` (HGNet).

### Phase 2 training offline metrics (val on each ckpt's held-out fold)

| Slug | Backbone | Fold | val/auroc (last ep24) | val/cmap (last) | val/top1 | val/top5 | best val/cmap | Best Ep | Walltime | SLURM |
|------|----------|------|---------------------:|----------------:|---------:|---------:|--------------:|--------:|----------|------:|
| `distilled-sed-kd-fold1` | EffNet-B0 NS-JFT | 1 | **0.9281** | 0.6654 | 0.8029 | 0.9197 | 0.6777 (ep11) | 11 | 4h22m | 9158704 |
| `distilled-sed-kd-fold2` | EffNet-B0 NS-JFT | 2 | 0.9079 | 0.6092 | 0.8082 | 0.9658 | 0.6237 (ep21) | 21 | 4h22m | 9158705 |
| `distilled-sed-kd-fold3` | EffNet-B0 NS-JFT | 3 | 0.9236 | 0.6927 | 0.7919 | 0.9396 | 0.6954 | ~24 | 4h22m | 9158706 |
| **`distilled-sed-kd-fold4`** | EffNet-B0 NS-JFT | 4 | **0.9310** | **0.7295** | **0.9013** | **0.9737** | 0.7335 (ep21) | 21 | 4h22m | 9159168 |
| `distilled-sed-kd-hgnet` | HGNetV2-B0 SSLD | 0 | 0.9069 | 0.6706 | 0.7613 | 0.9097 | 0.6963 | ~17 | 4h19m | 9158713 |

Phase D fold 0 reference (LB 0.906): val/auroc 0.908, val/cmap 0.692.

**5-fold KD val/auroc set**: {0.908 (fold 0), 0.9281, 0.9079, 0.9236, 0.9310}, mean **0.920**, std 0.011.

Read-outs:
- **Fold 0 was the bottom of the 5-fold band** (0.908 vs mean 0.920) — explains why Phase D single-fold LB 0.906 sits at the low end of Tucker's published single-fold trajectory (~0.91-0.92). Fold 4 is the strongest individual fold (0.931); fold 2 the weakest (0.908).
- **HGNet+KD val/auroc (0.907) ≈ Phase D EffNet+KD fold 0 (0.908)** — no obvious backbone advantage at the offline-metric level. Both `val/cmap` and `val/top1` are similar between HGNet and EffNet fold 0. LB will say whether the backbone matters for soundscape generalization.

W&B run IDs: `pmbh` (fold1) · `hoqe` (fold2) · `bpuk` (fold3) · `lwy9` (fold4) · `hyu0` (HGNet), all in `dsgt-arc/birdclef-2026-exp032`.

### D.5 end-to-end validation — ALL PASS ✅ (pre-flight before LB submission)

Each Phase 2 ckpt evaluated via `validate_end_to_end.py` (kernel inference path: `kernel_load_audio_chunk → compute_mel_tucker → encoder → AttHeadLogitSpace → LOGIT-space blend → 5-tap Gauss smooth → single sigmoid` on the same per-fold val SS pool). Gate: |training-side val/auroc − e2e val/auroc| ≤ 0.02.

| Slug | Training-side val/auroc | E2E val/auroc | Gap | Status | n_evaluable | Val windows |
|------|----------------------:|--------------:|----:|--------|------------:|------------:|
| `distilled-sed-kd-fold0` (Phase D) | 0.908 | 0.912 | +0.004 | ✅ PASS | — | 156 |
| `distilled-sed-kd-fold1` | 0.9281 | 0.9361 | +0.008 | ✅ PASS | 31 | 156 |
| `distilled-sed-kd-fold2` | 0.9079 | 0.9190 | +0.011 | ✅ PASS | 39 | 156 |
| `distilled-sed-kd-fold3` | 0.9236 | 0.9302 | +0.007 | ✅ PASS | 41 | 156 |
| `distilled-sed-kd-fold4` | 0.9310 | 0.9366 | +0.006 | ✅ PASS | 45 | 156 |
| `distilled-sed-kd-hgnet` | 0.9069 | 0.9108 | +0.004 | ✅ PASS | 29 | 156 |

E2E is uniformly slightly **higher** than training-side — the kernel path adds 5-tap Gaussian smoothing across the 12 windows per file in LOGIT space, which acts as a temporal denoiser. Matches Tucker's "+0.004 from MengYe smoothing trick" pattern (also seen in Phase D). **Pipeline is end-to-end consistent across all 5 new ckpts. LB submission is safe.**

Per-fold n_evaluable varies (29-45 classes with positives in that fold's val pool) — fold 4 has the most coverage (which may explain why it trained highest), fold 0 the least.

### D.5 methodology bug + fix (2026-05-25)

The first D.5 run on the 4 EffNet fold ckpts (jobs 9178659-62) showed e2e AUC ≈ **0.995** for all four — gap +0.066 to +0.090, all marked FAIL. Root cause: `validate_end_to_end.py` only read `fold_k` from the YAML config (which defaults to 0). The `--fold N` argparse override that `train.py` accepts was never wired into the validation script. So for ckpts trained with `--fold 1/2/3/4`, the script loaded **fold 0's** val pool — which was *training data* for those models → measuring memorization, not generalization. HGNet (job 9178663) PASSED on the first round because it was trained on fold 0, so the val pool matched correctly.

Fix:
- `scripts/validate_end_to_end.py`: added `--fold N` CLI arg that overrides `cfg["data"]["fold_k"]` (with the same `[0, 4]` guard as `train.py`).
- `sbatch/validate_end_to_end.sh`: added `FOLD` env var forwarding (`FOLD=N → --fold N`).

Re-running jobs 9178859-62 with `FOLD=N` produced the corrected PASS table above. **Training was never broken** — only the validation script was. This is consistent with the training-side val/auroc values being sensible held-out numbers (0.91-0.93 range); had training been evaluated on its own data, those would already have been ~0.995 in the W&B/`metrics.json`.

### Phase 2 checkpoints

- Top-level `last-v2.ckpt` at `2026-exp032-distilled-sed-kd/last-v2.ckpt` — Phase D KD fold 0 (LB 0.906).
- `2026-exp032-distilled-sed-kd/fold{1,2,3,4}/last.ckpt` — Phase 2.A folds 1-4 (last epoch 24, save_top_k=-1).
- `2026-exp032-distilled-sed-kd-hgnet/last.ckpt` — Phase 2.D HGNet+KD single-fold (last epoch 24).

### LB expectations (predictions)

| Submission | Predicted LB | Decision (per § Phase 2 matrix) |
|---|---|---|
| `distilled-sed-kd-5fold` (5 folds, logit-space mean + 5-tap smoothing) | **0.910–0.920** | Matches Tucker's 0.917 if recipe works at 5-fold scale; consider ensembling with exp 023 (0.929) for further lift |
| `distilled-sed-kd-hgnet` (single-fold, fold 0) | **0.90-0.92** | If LB > 0.910, backbone was a real bottleneck → pivot to HGNet for future training; if ~ 0.906, EffNet was fine |

The 5-fold mean val/auroc 0.920 is encouraging. With 5-tap Gaussian smoothing across folds in LOGIT space, the ensemble's effective resolution-per-window improves AND the per-fold noise averages out. Tucker's 0.917 LB came from exactly this stack at 5-fold.

## Phase 2 LB results (2026-05-25 PM): PARTIAL REPLICATION

| Submission | LB | Δ vs Phase D fold 0 (0.906) | Δ vs Tucker target | Decision-matrix bucket |
|---|---:|---:|---:|---|
| `distilled-sed-kd-5fold` (5-fold ensemble) | **0.915** | **+0.009** | **−0.002 vs 0.917** | **0.910–0.919 band** → matches Tucker's published 0.917 within noise; consider exp 023 ensemble |
| `distilled-sed-kd-hgnet` (single-fold ablation) | **0.901** | **−0.005** | — | **0.900–0.910 band** → marginal/slightly worse; EffNet was the right backbone |
| `distilled-sed-kd-fold4` (single-fold diagnostic) | **0.903** | **−0.003** | — | offline-LB anti-correlation: best-offline fold gave near-worst LB |

### Read-out

**5-fold ensemble: replicates Tucker's publicly-quantified pipeline within ±0.002 LB.**
- +0.009 single→5-fold lift matches Tucker's published +0.011 directionally.
- 0.002 gap to Tucker's 0.917 is well inside the documented cross-hardware variance (~0.007 LB on the same training code; exp 031 drift investigation).
- **What we did NOT replicate**: Tucker's "0.93+ with PP" stretch claim, which depends on his private post-processing (site/hour priors + per-class temperature; NOT in the public notebook).
- Bottom line: the publicly-replicable subset of his recipe yields 0.91–0.92 LB on this codebase, the same band his discussion thread reports.

**HGNet+KD: backbone was NOT the bottleneck.**
- HGNet (0.901) underperforms EffNet fold 0 (0.906) by −0.005 on LB despite near-identical offline val/auroc (0.907 vs 0.908).
- Refutes the Phase D hypothesis that the −0.041 control-vs-v5j gap was backbone-driven.
- Implication: the recipe-vs-recipe gap (Tucker recipe control 0.856 vs v5j_control 0.897) is driven by other ingredients — `AttHeadLogitSpace` vs LSEHead, dual-supervision vs single-clipwise, etc. NOT the EffNet→HGNet swap.

### What was actually replicated

| Tucker's published claim | Our result | Status |
|---|---|---|
| Single-fold KD lift ≈ +0.020 LB over no-KD | +0.050 paired Δ on Tucker recipe (Phase D) | ✅ exceeded |
| Single-fold LB band ~0.91 | distilled-sed-kd fold 0 LB 0.906 | ✅ in band (lower end) |
| 5-fold + smoothing LB 0.917 | distilled-sed-kd-5fold LB 0.915 | ⚠️ within ~0.002 (likely cross-hardware noise) |
| "0.93+ with PP" | not attempted (PP code is private) | ❌ out of reach |
| HGNet backbone improves over EffNet | tested, refuted on this stack (HGNet < EffNet) | ✅ ablation done, hypothesis rejected |

### Single-fold variance characterization (fold 0 + fold 4 LB landed)

| Fold | Offline val/auroc | E2E val/auroc | LB | Δ offline vs fold 0 | Δ LB vs fold 0 | Notes |
|---|---:|---:|---:|---:|---:|---|
| 0 (Phase D) | 0.908 | 0.912 | **0.906** | — | — | bottom of offline band; LB-validated reference |
| 1 | 0.928 | 0.936 | (not submitted) | +0.020 | — | middle of offline band |
| 2 | 0.908 | 0.919 | (not submitted) | 0.000 | — | tied with fold 0 offline |
| 3 | 0.924 | 0.930 | (not submitted) | +0.016 | — | middle of offline band |
| **4** | **0.931** | **0.937** | **0.903** | **+0.023** | **−0.003** | **strongest offline → LB came back near-worst of the two folds tested** |

**Fold 4 result is a strong offline-LB anti-correlation data point** — `+0.023 offline → −0.003 LB`. Two single-fold LB points (0.903, 0.906) cover a narrow 0.003 range despite their offline values spanning 0.023. Implications:
- **Single-fold LB band on this codebase is narrow (~0.003)** — much tighter than the offline-implied 0.023 spread. Offline val/auroc is essentially uninformative for ranking folds at LB.
- **The +0.009 single→5-fold ensemble lift (0.906 → 0.915) is pure variance-reduction**, not "picking up better folds". The Gaussian smoothing across 5 fold logits in LOGIT space is the lever.
- **Don't submit folds 1/2/3** as additional diagnostic — we already have the answer. The remaining 3 folds would land in the same 0.903-0.906 band with high confidence; low marginal information for high LB-slot cost.
- **Aligned with broader exp 030/031 pattern**: offline metrics consistently anti-correlate with LB on this competition. Documented across 22+ prior recipe attempts; this is data point #23.

This also explains why our Phase D single-fold landed at 0.906 (below Tucker's published 0.91-0.92 single-fold band): the single-fold band on Tucker's hardware may genuinely be wider than ours, OR his published "fold-1 = 0.922" was just a lucky fold instance. Either way, the 5-fold ensemble is what closes the gap (0.915 ≈ his 0.917), and the path forward doesn't need more single-fold submissions.

### Implication for next steps

The recipe is replicated; the path to higher LB is no longer "fix the recipe":
1. **Ensemble distilled-sed-kd-5fold (0.915) with exp 023 perch-sed-ensemble-mlp-only (0.929)** — highest-EV move. Two strong, decorrelated stacks (Tucker SED + Perch probes) average to a higher number than either alone. Estimated band: 0.93–0.94.
2. **Port site/hour priors (EoS.6 style) onto distilled-sed-kd-5fold** — exp 021 priors lifted v5j-4fold by +0.004; similar lift here would land 0.918–0.920.
3. **Accept the publicly-replicable limit** of this recipe and pivot LB-lift effort to other levers (Sydorskyi-yoink, MixIT-based PL, etc.).

LB submission slots are precious near deadline (2026-06-03, 9 days out); options 1 and 2 are the highest-EV with what's already built.

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


---

# § 7. COMPANION DOC: Kaggle thread "best single model LB"

*Source: `user/murilo/notes/kaggle-discussion-best-single-model-lb.md`*

# Kaggle Discussion: What is your best single model LB score?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/683791
- **Posted**: 2026-03-21 (Antoine Masq)
- **Topic author**: Antoine Masq (33rd at 2026-05-27 capture; was 49th at 2026-05-21)
- **Comment count at capture**: 65
- **Captured**: 2026-05-27 (prior pulls 2026-04-23, 2026-04-28, 2026-05-09, 2026-05-21)
- **Companion thread**: [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md) — "What's the limit without any Perch involved?"
- **Companion Perch-stack note**: [`2026-05-21-nina2025-eos6-protossm-stack-analysis.md`](2026-05-21-nina2025-eos6-protossm-stack-analysis.md)

Notation in the transcript below:
- `↳` indicates one level of nesting (reply); `↳↳` two levels; `↳↳↳` three levels; etc.
- ⭐ marks comments referenced by the synthesis sections at the bottom of this file.
- `"This comment has been deleted."` is preserved verbatim wherever Kaggle displays it.

---

## Original post

**Antoine Masq (49th, Topic Author) — Posted 2 months ago** ⭐

> Hi everyone,
>
> As most of us are probably still improving our baseline models, I was curious to see what your best LB score is for a single model without using the unlabeled data.
>
> So far, my best single model performance is 0.889 with an EfficientNetv2_b0 backbone

---

## Key competitor scores (at-a-glance, 2026-05-27)

Synthesis table — *see comment transcript below for source quotes*. Ranks are as of 2026-05-27.

| Person | Rank | LB | Details |
|---|---|---|---|
| **Arunodhayan** | **7th** | **0.946** | No Perch, no details (cross-ref: [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md)) |
| **Boredom** | **2nd** (was 5th) | **0.950+** w/ unlabeled, **0.938-0.939** w/o | Non-Perch, timm+SED, ~11min inference. 1st iter on unlabeled. **No TTA**; "simple PP" worth ~+0.004 (corrected from 0.03-0.04 typo). 2026-05-27 update: confirms **"doesn't work for more than one pass"** in pseudo iteration thread. |
| **Ali Ozan Memetoglu** | **7th** (was 6th) | **0.941** single | Uses Sydorskyi 2nd-place 2025 published checkpoint directly ([dataset](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models)). 1028 attention dim. ONNX 5-fold ~30 min total. 2026-05-27 update: recommends **"trim probability"** (cell-level pseudo prob below threshold zeroed) — also says to **"increase the trim probability"** when pseudo fails. |
| **Antoine Masq** | **33rd** (was 49th, Topic Author) | **0.943** single, **0.948** ensemble | EffNetV2-B0 + XC pretrain + own-teacher PL + custom ASL + sliding-window TTA + smoothing. 10 min inference. Trajectory: 0.889 → 0.936 → 0.943 (cross-ref: [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md)). 2026-05-27 update: warns against ensembling models trained on the **same teachers** (too correlated). |
| **Salman Ahmed** | **43rd** (was 39th) | **0.937** solo v2B0 w/o PL, **0.939** w/o PL, **0.944** w/ PL, **0.952** ensemble | Modeling breakthrough — attention investigation on hardest/easiest loss examples. B0_NS now matching v2B0. Uses PP/TTA. 20s segments, ~50-60 min inference. 2026-05-27 update: PL lift +0.005 (0.939 → 0.944); second PL iteration doesn't help; teacher is a 0.944 ensemble. |
| **Anil Ozturk** | **6th** (was 2nd) | **0.940** w/ "very weird customized pseudo" | Two trajectory points: supervised 0.931 → pseudo 0.940; supervised 0.934 → pseudo 0.939. **"Doesn't work for more than one pass."** GKF on sites OOF=0.608 (extreme CV/LB disconnect). Recommends "cell level any pseudo probability below it gets zeroed". |
| **Jack** | **17th** (was 21st) | **0.933** 5-folds no pseudo / no distill, **0.934** w/ pseudo (1 fold) | No TTA / no PP. "Pseudo not lifting solos but allowing ensembling of solos to get a much better score in same compute." Confirms Sydorskyi backbone yoink + own head with melspec match. |
| **Tucker Arrants** | **685th** (was 196th) | **0.946** best after 6 rounds of pseudo, virtually zero ensemble benefit | "6 rounds of pseudo, best model 0.946, but virtually zero ensemble benefit 😆". Earlier 0.93+ recipe (Perch-distilled B0, 5-fold + priors + smoothing) holds. Also: 0.925 w/o deep learning (frozen Perch + clustering + metadata). |
| **Zejun_** | 15th | — | "totally same, only first student is smart" — independent confirmation of pseudo-iteration plateau. |
| **Joseph** | 317th | **0.949** single, **0.947** ensemble | Solo > ensemble case — opposite of Jack's pattern. |
| **Jiacheng Ma** | **25th** | — | Tried 2nd-place 2024 pseudo recipe, didn't work down to stage 1. Asks Tucker about adding priors in logit space. Reports same pattern as Jack: solo + pseudo + ensemble not lifting much. |
| **MarkDjadchenko** | **109th** | 0.899 (clustering only) | Asks Tucker about clustering recipe. Also: "Maybe you are not training backbone during the pseudo-only distillation" — to Tucker re: 6-round plateau. |
| **To be happy** | 1149th | — | Rank-ensembling teacher boosts LB but distorts probabilities → hard to use for PL targets. |
| **BNBU Anpeng Yuan** | 918th | **0.910** 4-fold HGNetv2 no distill / no pseudo | New data point: HGNetv2 4-fold matches our 0.906 ballpark exactly without KD / PL. |
| **Starry** | 3rd | — | Asked Boredom about 0.950+ unlabeled flow |
| **Anil Ozturk** | 2nd | 0.879 | GKF on sites OOF=0.608 (extreme CV/LB disconnect) |
| **Tom** | 14th | ~similar to hengck23 | "slightly different approach which fits on 4x5sec" |
| **yukiZ** | 17th | 0.904 (single fold, exp #107) | Long table of CV/LB pairs across experiments |
| **Jack** | 21st | not disclosed in this thread; "didn't manage to see the increase" | Also yoinked Sydorskyi's backbone + own head; matched melspec; −0.012 LB swap to that melspec. Confirms Perch KD causes backbone stability |
| **tennogh** | 33rd | — | ~0.02 diff between backbones; KD reduces diversity (cos sim 0.5→0.67) |
| **Jake** | 34th | **0.915 w/o unlabeled, 0.922 w/ unlabeled** | Solo model; small ~0.007 lift from unlabeled |
| **MarkDjadchenko** | 74th | 0.899 (clustering only) | Asks Tucker about clustering recipe |
| **Kurise** | 115th | — | Discusses Sydorskyi-checkpoint reuse |
| **!!!!!** | 122nd | 0.88 | Pretrained timm; later reports the same |
| **Tucker Arrants** | **196th** | **0.93+** w/ PP (5-fold ensemble), **0.922** w/o PP | Perch-distilled B0 + 5s clips + BCE + waveform augs + site/hour priors + Gaussian smooth (5 folds). Quantified Perch KD lift at **~+0.02 LB**. Also: 0.925 w/o deep learning (frozen Perch + clustering + metadata), 0.91 w/ Perch+GRU. |
| **OpPrime** | 655th | 0.835 (fold 0) / 0.826 (5-fold) | Notebook + weights shared |
| **Aic** | 837th | — | Asks about LB stability |
| **Gaurav Rawat** | 1065th | 0.908-0.926 | GroupK on site CV; shows poor CV-LB correlation |
| **goevian** | 1075th | — | Asks about multi-context head |
| **Jiacheng Ma** | 1332nd | — | Asks Tucker how to add priors (logit space?) |
| **Amit Aharoni** | 1531st | — | Asks yukiZ about val data |
| **Seeing Times** | 1490th | — | Asks Salman about SED head |
| **XuKong Ji** | 1609th | **0.899** single fold | EffNet-B0+LSEHead+BCE, 5s. 5-fold=0.889 (worse!) |
| **Murilo Gustineli** (us) | 1715th | **0.906** (HGNet+LSEHead) | No KD, no Perch in SED |
| **hengck23** | 1813th | **0.896** CNN only | Long input wave, multi-context head, no Perch, no PP |
| **EliKal** | 2044th | **0.909** single (EffNet-B0) | Perch KD (KLD loss) + 12ep finetune, time-shift TTA. Multiple backbones cluster 0.906-0.909 |
| **chenwenqiang_001** | 2073rd | 0.899 student | Own 0.886 teacher → student +0.013 from own-teacher distillation |
| **Shiro** | 2192nd | — | Asks Salman about no-PP/TTA score |
| **vion** | 2220th | — | 0.940 ceiling math |
| **Mahog** | 2364th | — | Reports ±0.005 LB shift on seed change |
| **Leon** | 2394th | — | Working to match Salman's 0.922 single B0 |
| **D.M. (Don Mathis)** | 2573rd | 0.880 (~0.875 reproducible) | CNN + logit-LSE, BCE, 5s clips. **Pain point**: ".921 with mixup on waves + train_audio + labelled SS — I get .865 doing same." |
| **lhwcv** | 2687th | — | Discord ping to Salman |

---

## Full comment transcript (Kaggle thread display order)

### Thread 1 — Salman Ahmed's April 11 update (LB 0.932 ensemble / 0.922 solo)

**Salman Ahmed (39th) — Posted a month ago** ⭐ MAJOR (20s training, no KD, ensemble 0.932 / solo 0.922)

> As of April 11. Updated a couple of things in my train settings.
>
> Trained on 20 sec with Train Audio + Labelled Soundscapes.
>
> No KD. No Claude for now. Just wanted to solve it / write it myself after a longtime relying on claude. [Just for the fun of it].
>
> No Prior or additional features.
>
> Ensemble of 4 scores 0.932.
>
> Individual scores around 0.922.
>
> 10 seconds / 15 seconds scores almost the same for me / sometimes better [Didn't include those in my ensemble].
>
> Will play with some cool attention for next week.

**↳ Jack (21st) — Posted a month ago**

> Impressive stuff!

**↳ Antoine Masq (49th, Topic Author) — Posted a month ago**

> So this score is still without using the unlabeled soundscapes ? If so, that's super impressive!

**↳ Gaurav Rawat (1065th) — Posted a month ago**

> Awesome dunno getting single models similar but infer too slow for me :/

**↳ XuKong Ji (1609th) — Posted a month ago**

> That's impressive! If you don't mind, could you please share some details about the data processing pipeline? Specifically, do you directly convert the entire 15-second audio clip into a mel spectrogram and feed it into the model for prediction and training? Or do you split the 15-second spectrogram into three 5-second segments, convert each segment into a mel spectrogram separately, perform model inference on each one individually, and then aggregate the three results?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (20s NOT split+max)

> xD. Yes it's trained on 20 seconds audio segments. Not the split and max logits.

**↳↳↳ Mahog (2364th) — Posted a month ago**

> Does training on longer audio segments increase training time by a lot for you? 20 second segments should in theory increase training time by 4x right?

**↳ !!!!! (122nd) — Posted a month ago**

> If you don't mind me asking, what was the runtime for your submission?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (50-60 min runtime)

> Around 50 to 60 minutes I think.

**↳ Leon (2394th) — Posted a month ago**

> Awesome! Is it still the efficientnetb0 backbone?

**↳↳ Salman Ahmed (39th) — Posted a month ago**

> Yes. Solo B0 was around 0.922.

**↳↳↳ Jiacheng Ma (1332nd) — Posted a month ago**

> Great！It's still efficientnet_b0 or efficientnet_v2b0?

**↳ Starry (3rd) — Posted a month ago**

> Hi Salman~ Nice to see you in BirdClef again. I wonder do you use external data in your current pipeline?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (BG noise from past comps)

> The data I am using is used in solutions for previous competitions.
>
> I have a question for host @stefankahl ; Do we need to share that again? If yes, then where? and by what time / day?
>
> for example; no call / Background noise was used by 2nd place solution in 2023 [Background Noise(nocall in 2020, 2021 comp + rainforest + environment sound + nocall in freefield1010, warblrb, birdvox)] https://www.kaggle.com/datasets/honglihang/background-noise

**↳↳↳ Salman Ahmed (39th) — Posted a month ago**

> btw, I think I get same LB without any other data than this competition's. But I think using that previous competition's data, might be helpful to make my model generalize better.

**↳↳↳ lhwcv (2687th) — Posted a month ago**

> @salmanahmedtamu Hi Salman, are you still on Discord?

**↳↳↳↳ Salman Ahmed (39th) — Posted a month ago**

> Hi @lihaoweicvch, Great to see you in this year's competition as well. Yes, but not too active there. 😅

**↳ Ali Ozan Memetoglu (6th) — Posted a month ago**

> Thank you :)) Are you using the soundscapes too or only this?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (SS HURTS HIS LB — possible bug)

> Just Train Audios. Just started experimenting with Soundscapes, and for now, adding Soundscapes reduce my LB to around 0.85.

**↳↳↳ Salman Ahmed (39th) — Posted a month ago**

> As I was typing this, i remember something, I think there is a bug in Soundscapes training. 😂 I'll fix it soon.

**↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago**

> No, I mean for the background noise I was only using soundscapes so I'm thinking of combining them with the dataset you provided

**↳↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (does NOT use SS as BG noise)

> Ah I see. No I don't use the SS as background noise, because there are species in those SS recordings.

**↳ MengYe (558th) — Posted a month ago**

> does your cv correlate with lb?

---

### Thread 2 — Salman Ahmed's 0.937 single-model breakthrough ("modeling matters")

**Salman Ahmed (39th) — Posted a month ago** ⭐⭐ (0.937 SINGLE — attention investigation)

> Finally. Single model LB 0.937 without using Unlabeled data. Recently started paying attention to the modeling part in this competition, Apparently, it was not that important in previous competitions, but in this one, for me it is one of the most important part. I have been investigating attention values on some examples with highest loss during training, and minimum loss during training, and now it's much better compared to previous models I had.

**↳ Leon (2394th) — Posted a month ago**

> Wow that is amazing! May I ask is it still efficientnet b0 ns?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (EffNetV2-B0 is the 0.937 backbone)

> xD LB 0.937 is with Efficientnet v2B0 because it's faster to train compared to others. Will try other variants (B0 NS / mixnet / nfnet etc) soon.

**↳↳↳ Seeing Times (1490th) — Posted a month ago**

> an amazing score, May I ask did you change sed head?

**↳ Leon (2394th) — Posted a month ago**

> hh sorry for keeping asking this. I am working on making my b0 single model to match your score xD. I think you mentioned before b0 MBConv doesn't learn well, and ebv2 is using Fused-MBConv, might be reason why its better (not sure). and may I also ask if your b0 model (0.922) is with any post processing (I assume only model soup)?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (no model soup; B0_NS now matches v2B0)

> xD No worries, For that previous model architecture / training strategy MBConvs were not learning, but now B0_NS is matching or performing even better than v2B0. Yeah, I am not using Model Soup this time, simple model checkpoint.

**↳ Shiro (2192nd) — Posted a month ago**

> Impressive results, with efficientnetb1 I am not able to reach that results with single models, trying different architectures. May I ask if you include postprocessing in the inference step or is it just a vanilla prediction on 5 seconds?

**↳↳ Salman Ahmed (39th) — Posted a month ago** ⭐ (relies on PP/TTA)

> Hi @ludovick Yes, I definitely rely on some post processing / TTA.

**↳ D.M. (2573rd) — Posted a month ago**

> How can cv correlate with LB? The training data are "clean" clips not embedded in soundscapes. Mixup of clean clips with unlabeled soundscapes doesn't work because there is unlabeled signal in the soundscapes. Mixup with labeled soundscapes helps got from .80 to .86 but there's not enough of them to go further. I'm stumped.

**↳ Shiro (2192nd) — Posted a month ago**

> thanks, do you know your LB on single model without postprocessing/TTA?

**↳ XuKong Ji (1609th) — Posted 22 days ago**

> Hi Salman, May I ask your single model means 5fold average or just 1fold model?

---

### Thread 3 — hengck23's diagnostic framework

**hengck23 (1813th) — Posted a month ago** ⭐⭐ (diagnostic framework — failure modes + monitoring)

> why some people get lb0.87+ and why get as high as 0.91+ for simple CNN model? (exclude effects, pseudo-labels, extreme tricks … we focus on base model performance)
>
> the reason are:
> - long tail (some class has few train samples)
> - domain shift. focal audio: one species, high snr soundscape: multiple species, low snr (background noise, bird far from mic)
>
> how to solve:
> - long tail: pretrained model, sampling, more information per sample (e.g. 10,15 sec instead of 5 sec), regularization
> - domain shift: augmentation (others: using unlabelled, external data)
>
> what should you monitor
> - auc: sensitive to few samples
> - bce: can skews towards low sample (bce hacking … like reward hacking)
> - logit std: monitor this as a "proxy to logit distribution"
>
> how a robust model would look like?
> - agreement for train and validation
> - lower logit std
> - low bce
> - high auc (split into acu for normal, rare, problematic classes etc)
> - can train for every long epoch (if augmentation is good)." Good grad descent at high lr"
> - low auc variance in different folds
> - flat loss valley for low sample class
>
> how a overfit model would look like?
> - disagreement for train and validation
> - high train logit std (growing as training iteration pogress)
> - low train bce
> - kigh train auc
>
> finally, cut and paste this post into you favourite AI llm and ask for comment. If useful, add to the skills in agent.

**hengck23 (1813th) — Posted a month ago** ⭐ (methodology: "turning insights into measurable, testable pipelines")

> how to learn to be an expert in modelling. distill ChatGPT knowledge to yourself.
>
> instead of asking chaptgpt to "improve the model", …
> Pick a point from the above post, e.g. "flat loss valley for low sample class"
> ask chagpt:
> - how to measure flatness. how to prove current model methods is flat or not
> - suggest methods to improve flatness
> - implement and carry experiments
> - verify flatness is improved and lead to better train,validation and public LB.
> - if it does not, ask chatgpt why and most importantly how to prove?
>
> "turning insights into measurable, testable pipelines"

---

### Thread 4 — Tucker Arrants's Perch+GRU and Perch-only-clustering

**Tucker Arrants (196th) — Posted a month ago** ⭐ (frozen Perch + clustering only → 0.925 no DL)

> Frozen Perch embeddings + learned pool + GRU
>
> 5 folds, 3x time-shift TTA
>
> 0.91 public leaderboard
>
> Edit: I was able to score 0.925 without any deep learning, just some simple clustering on the frozen perch embeddings and metadata engineering. The learned pool + GRU was total over kill.

**↳ MarkDjadchenko (74th) — Posted 11 days ago**

> Could you tell me more about the clustering? Which clustering algorithm did you use? The best LB score I managed to achieve with clustering was 0.899.

---

### Thread 5 — hengck23's CNN-only 0.896 with multi-context head

**hengck23 (1813th) — Posted 2 months ago** ⭐⭐ (CNN-only 0.896 LB with multi-context)

> how to use CNN only (no perch2, no post processing) to get 0.896 lb
> context and augmentation is the key. here we cheat by using CNN and long input wave to capture temporal information
>
> hint: how to get even better results: multi-conext head

*(Attached image: ASCII pseudo-code for multi-context head. Reproducing from prior captures: single 15s spectrogram → HGNetV2-B0 backbone → feature map (B, C, F', T') → split T' into 3 chunks of T'/3 → LSEHead per-chunk → 3 predictions, all supervised during training with their own 5s label. LB 0.876 → 0.896 (+0.020), 4-fold, no PP. Wave mix augmentation specially designed for 3×5sec input. Training: 35k focal clips + 66 labeled soundscapes. Image vendored at `notes/figures/hengck23-cnn-birdclef.png`.)*

**↳ Tom (14th) — Posted 2 months ago**

> This is similar to my current approach, but I made a slightly different approach which fits on 4x5sec.

**↳↳ hengck23 (1813th) — Posted 2 months ago**

> I checked your Claude report. He actually knows this trick the very first day. That is impressive

**↳↳↳ This comment has been deleted.**

**↳↳↳ D.M. (2573rd) — Posted 2 months ago**

> I sort of see the overall goal, but this is too sketchy for me. I don't know how many context_nets there are or what clip length they've been trained on, nor what the '3' is in the shape of prob3. I think the bottom loop is sliding width-4 windows within a 12 timestep clip and averaging the last 2 dimensions of prob3 in each of the 12 cells. But then I still don't know what the 3 dimension is, nor what 'probs' is at the end.

**↳ goevian (1075th) — Posted a month ago**

> may i ask, what do you mean by"multi-context head"?

---

### Thread 6 — Salman Ahmed's 0.921 solo / 0.923 with priors (no Perch KD)

**Salman Ahmed (39th) — Posted 2 months ago** ⭐ (0.921 solo / 0.917 no-priors / 0.923 with priors, WITHOUT Perch KD)

> Alright, Finally have 2 SED models with ensemble of LB 0.923. and solo around 0.921 Simple training with mixup on waves. 25 epochs. CE Loss but Sigmoid at Infer Trained with train_audio and labelled soundscapes only. Without priors it achieves 0.917 and with priors -> 0.923. Nothing fancy, simple model training setup. Another thing I noticed, for melspecs, MBConvs from EfficientNets are not learning that well, compared to other architectures.
>
> WITHOUT any Perch KD

**↳ Boredom (5th) — Posted 2 months ago** ⭐ (EffNet works for Boredom at 0.925+)

> Interesting findings. In your experiments, did EfficientNet underperform compared to other architectures? Also, may I ask what you mean by "Priors" here? In my experiments, a single EfficientNet model can easily achieve an LB score above 925+.

**↳↳ Salman Ahmed (39th) — Posted 2 months ago** ⭐ (priors = species×site×hour)

> Yes, for me Efficientnet NS (with MBConv Architecture) were not converging fast and scored much lower on LB, compared to other models. Not sure why. By Priors I mean, prior table of species based on site and hours.

**↳ Murilo Gustineli (1715th) — Posted 2 months ago**

> Great score with a single EfficientNet! I'm struggling to get EfficientNet-B0 to go beyond 0.855 LB, even when trying multiple different heads and loss functions. Could you share what loss function you're using (BCE vs CE) and whether you're using an attention-based head or something simpler? Trying to understand what unlocks 925+ on EfficientNet specifically

**↳ Murilo Gustineli (1715th) — Posted 2 months ago**

> Congrats on the 0.921! My best single model so far got 0.906 LB with HGNetV2-B0 + LSEHead + BCE. I tried switching to AttHead (with tanh attention, sigmoid-space clipwise, dual 50/50 supervision) but it got worse LB results both with BCE and CE loss. Two quick questions if you don't mind: (1) what backbone did you switch to from EfficientNet? and (2) are you using 5s or longer windows for training? Thanks!

**↳↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (claims 0.906 must come from distillation — WRONG re us, but interesting framing)

> i suppose your 0.906 comes from distillation. Comparing with your previous result, you would note that the improvement comes from distillation and not data.
> What to distill? logit? embedding or spatial embedding or all?
>
> if the results is from distillation, then maybe modeling is less impotant.
>
> if you look at public notebook which is just based on perch2 emebdding, they are :
> lb 0.930 = (perch2 + proxy class + temporal + PCA) + prior + state space + perclass normalisation ….
>
> you can estimate
> lb 0.92+ = (perch2 + proxy class + temporal + PCA)
>
> this means to get lb 0.92+ from timm, you need to add temporal

**↳↳↳ Murilo Gustineli (1715th) — Posted 2 months ago** ⭐ (clarifies: no KD, no Perch; multi-context regressed)

> I'm not using knowledge distillation at all actually. The 0.906 is just HGNetV2-B0 trained from ImageNet init on focal + labeled soundscapes with BCE loss. No Perch at all.
>
> Really interesting point about temporal. I've tried your 3x5s multi-context approach (backbone sees 15s, split into 3 chunks, LSEHead per-chunk) and got best-ever offline val metrics, but it regressed on LB (0.896 with 4×5s avg, vs 0.906 baseline). What exactly do you mean by "add temporal"? is it the multi-context you showed, or something else like temporal embeddings or sequence modeling?

**↳ Aic (837th) — Posted a month ago**

> May I ask, do you experience any score fluctuations in your experiments due to different numbers of epochs or changes in other hyperparameters?…it's been giving me a headache lately.

**↳↳ Mahog (2364th) — Posted a month ago** ⭐ (seed sensitivity)

> Have you managed to fix this? I'm having the same issue, even a seed change shifts LB by 0.005 :V

**↳ D.M. (2573rd) — Posted a month ago** ⭐ (CAN'T REPRODUCE SALMAN 0.921)

> 0.921 Simple training with mixup on waves. Trained with train_audio and labelled soundscapes only. No clue how you are doing that. I get .865 when I do that :/

---

### Thread 7 — Murilo Gustineli's 0.86 ablations + calls for help

**Murilo Gustineli (1715th) — Posted 2 months ago** ⭐ (our public ask)

> My best single SED model is 0.86 LB with EfficientNet-B0 - and I've been struggling to break past that despite extensive ablations.
>
> My setup: EfficientNet-B0 SED (ImageNet init), GeM + attention head, n_fft=4096, n_mels=224, AmplitudeToDB norm, soft CE dual loss (clipwise + framewise max), 10s windows, spectrogram MixUp (α=0.5), 90/10 split, 50 epochs with CosineAnnealing. Trained on focal + labeled soundscapes (234 classes).
>
> I've run 15+ ablations (BCE, ASL, LSEHead, HGNetV2-B0, waveform MixUp, weight decay, StochasticDepth) and can't break past 0.86 - offline val improvements consistently fail to translate to LB gains.
>
> I also reproduced ttahara's HGNetV2-B0 recipe (LSEHead, BCE, OneCycleLR, per-sample MixUp, 5s windows, 20 epochs) and got 0.872 val/auroc offline, but only 0.855 LB on EfficientNet with the same recipe. The val-to-LB disconnect has been consistent across all ablations - improvements in offline val rarely translate to LB gains.
>
> Curious what's driving the gap to 0.9+ for other single EfficientNet-B0 SEDs. A few things we haven't explored:
>
> - NS-JFT pretrained weights (tf_efficientnet_b0.ns_jft_in1k) vs standard ImageNet init
> - 3-channel mel input (repeat mono to match pretrained stem) vs 1-channel with weight averaging
> - Longer context windows (15-20s) for capturing insect/amphibian textures
> - Training on focal only vs focal + soundscapes - currently training on focal + soundscapes
>
> Would love to hear from folks who've broken 0.9+ @aliozanmemetoglu @hengck23 @tom99763 @cudacoding @salmanahmedtamu @hideyukizushi what was the single biggest change that got you there? Any pointers would be appreciated!

**↳ Salman Ahmed (39th) — Posted 2 months ago** ⭐ (key advice to us)

> In my experiments, more epochs make the model too confident on it's predictions and that hurt the AUC (Evaluation metric).
>
> In my experiments, SED simple Attn head's framewise max is better than lse head.
>
> I mixup waves rather than Spectrograms.
>
> For now, Efficientnet B0 didn't help.
>
> Try to listen to a audio from train soundscape and then look at the attentionhead's output. That might help.

**↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (Perch-baseline framing)

> it is better to think of baseline
> 1) use perch embedding to get cv and public lb
>
> just train a linear or MLP probe on clip and evaluate on soundscape ss (in addition to clip). this give you an idea on effect of domain shift.
>
> 2) repeat (1) but using your model. train on clip only
>
> read perch paper and see how many train samples they use and compare that with kaggle data. this gives you an idea of how num and variety of data affects results.
>
> from these two, it would be enough to see if your results is reasonable or no and other kagger top results is due to tricks, post processing, better modelling, better data, etc

**↳ yukiZ (17th) — Posted 2 months ago** ⭐ (waveform augmentations: Pink Noise + Time Stretch)

> Hi! Regarding the contribution of CV/LB in my experiment, data augmentation was effective.
>
> I believe many publicly available notebooks include Mixup. On the other hand, data augmentation on raw waveforms is not seen very often, but it has been frequently used in solutions from past BirdCLEF competitions. For example, "Pink Noise" and "Time Stretch".

---

### Thread 8 — Salman Ahmed's earliest 0.908 (10s, CE, raw wave mixup)

**Salman Ahmed (39th) — Posted 2 months ago** ⭐ (original 0.908 baseline recipe)

> My best single model is SED Efficientnet B0 (Trained on 10 sec). Mixup on Raw Waves, and CrossEntropyLoss on clipwise and framewise max. Trained for 20+ epochs. This single model scores 0.908
>
> I have a ensemble of these 2 models and it scores 0.914
>
> No other augmentation. No unlabeled data.

**↳ This comment has been deleted.**

**↳ Murilo Gustineli (1715th) — Posted 2 months ago**

> Is it safe to assume that you're only training on train_audio or also using the train_soundscapes to achieve the 0.908 LB score?

**↳↳ Salman Ahmed (39th) — Posted 2 months ago**

> Yup. Train audio + label soundscapes.

**↳ Jiacheng Ma (1332nd) — Posted 2 months ago**

> What are label soundscapes and unlabel soundscapes? Thanks.

**↳↳ Salman Ahmed (39th) — Posted 2 months ago**

> train_soundscapes_labels.csv are labeled soundscapes.

**↳ This comment has been deleted.**

---

### Thread 9 — yukiZ's experiment table (CV/LB pairs)

**yukiZ (17th) — Posted 2 months ago** ⭐ (CV/LB correlation table)

> I believe our interest lies in whether there is a correlation between CV and LB, so I will share my experimental results. However, due to the highly skewed distribution of the data, the validation strategy will likely differ depending on the participant.
>
> The following experimental results show the results of training/submitting using only Fold0 with 5Fold SKF, assuming future Blending of the BB of the SED Model with B0 (approximately 5-7M parameters) or Alternatively, it uses a backbone of around 15M, similar to v2s.
>
> | MyExpNo | AUC (LocalValidation, Fold1/5) | LB (Fold1/5) |
> |---------|-------------------------------|--------------|
> | #107 | 0.9774 | 0.904 |
> | #105 | 0.9756 | 0.902 |
> | #042v4 | 0.9768 | 0.898 |
> | #042 | 0.9744 | 0.888 |
> | #041 | 0.9737 | 0.885 |
> | #040 | 0.9722 | 0.878 |
>
> [2026/03/27 Upd. Add Exp "#042v4","#105","#107"]

**↳ Amit Aharoni (1531st) — Posted 2 months ago**

> Thanks for sharing Based on the AUC score, is it safe to assume that your local validation is using train_audio and not train_soundscape?

**↳↳ yukiZ (17th) — Posted 2 months ago**

> Yes, your way of thinking is correct.

---

### Thread 10 — Boredom's progression 0.922 → 0.947+ → 0.950+

**Boredom (5th) — Posted 2 months ago** ⭐ (0.922 single-fold, no unlabeled)

> Single model / Single Fold: 0.922 without using the unlabeled/extra data

**↳ vion (2220th) — Posted 2 months ago**

> Amazing score considering that 0.940 is the maximum achievable with 206 of 234 classes. Congrats!

**↳ Jiacheng Ma (1332nd) — Posted 2 months ago**

> You only use train_audio? no train_soundscape?

**↳↳ Boredom (5th) — Posted 2 months ago**

> train_audio + train_soundscape (w/ label)，w/o pseduo label

**↳ Boredom (5th) — Posted 2 months ago** ⭐ (estimates 0.94-0.945 ceiling w/o unlabeled)

> I guess without using unlabeled data, the highest LB (leaderboard score) we can achieve is around 0.94-0.945.

**↳ !!!!! (122nd) — Posted 2 months ago**

> May I ask you use Perch or just timm-based model? For me, I get low score (0.88) using pretrained timm model.

**↳ Boredom (5th) — Posted a month ago** ⭐ (0.947+ update, 13 min)

> update: Single model: 0.947+ about 13min

**↳↳ Kurise (115th) — Posted a month ago**

> Amazing results, did you used the unlabeled data?

**↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (confirms unlabeled critical for >0.945)

> Yes, absolutely. I would be very surprised if someone could achieve above 0.945 with a single model without using unlabeled data😭. My best single model without unlabeled data is around 0.937-0.938.

**↳↳↳↳ Aic (837th) — Posted a month ago**

> Really impressive! Was your best single model based on Perch distillation, or was it a non-Perch model?

**↳↳↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (NON-PERCH)

> non-Perch, maybe Perch can do better.

**↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago** ⭐⭐ (0.941 SINGLE w/ 2025 PRETRAIN)

> My single best model is 0.941 but I use last year's pretrained checkpoint so I believe it is possible :D

**↳↳↳↳↳↳ Aic (837th) — Posted a month ago**

> Impressive stuff!

**↳↳↳↳↳↳ Boredom (5th) — Posted a month ago**

> 😱Did you use extra data?I've been struggling for a long time to break 0.94 without using unlabeled data..

**↳↳↳↳↳↳ Kurise (115th) — Posted a month ago**

> You did what I want to do tomorrow! May I ask how much will it improve by using last year's checkpoint?

**↳↳↳↳↳↳ Kurise (115th) — Posted a month ago**

> Not sure whether I'm right, using last year's checkpoint just like 2 stage training(pretrain on 25, retrain on 26), but there is no extra computation cost. So that we can get a model trained on both 25 and 26 data.

**↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago** ⭐ (Sydorskyi XC pretrain)

> They used the entire Xeno-Canto dataset by filtering it and fixing a bug during pretraining, but I didn't use anything beyond the checkpoint.

**↳↳↳↳↳↳↳↳ Cody_Null (310th) — Posted a month ago**

> Im not super excited about the idea of doing all these models 5 fold on the entire XC dataset but we gotta do what we gotta do 😂

**↳↳↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted a month ago**

> it is pretraining, so there are no folds

**↳↳↳↳↳↳ Jack (21st) — Posted 20 days ago** ⭐ (Sydorskyi backbone yoink CONFIRMED)

> You just yoinked a backbone from 2nd place and trained another head on it, yeah? They mentioned doing that with the full xc data backbone I believe

**↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted 20 days ago** ⭐ (PUBLISHED CHECKPOINT URL)

> yes https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models

**↳↳↳↳↳↳↳↳ Jack (21st) — Posted 18 days ago**

> Did you match their melspec? My first experiment's results went well but my notebook runtime is pretty much your team's name smh

**↳↳↳↳↳↳↳↳↳ Ali Ozan Memetoglu (6th) — Posted 18 days ago** ⭐ (ATTN DIM 1028 + ONNX RUNTIME)

> Yes, 1028 is the attention dimension too. When I convert them to ONNX, all 5 folds only take half an hour

**↳↳↳↳↳↳↳↳↳↳ Jack (21st) — Posted 17 days ago** ⭐ (MELSPEC MISMATCH = −0.012 LB)

> Is your 0.941 with tta/postprocessing/priors? Swapping to match their melspec lowered my score 0.012.. the runtime is a LOT faster though hahaha

**↳ Boredom (5th) — Posted a month ago** ⭐⭐ (0.950+ SINGLE w/ 1-iter PL)

> Update: Single model LB 0.950+ (11 minutes). After reading Salman's latest post, I realized I might have been constrained by my experience from past competitions, which may have limited the optimal single-model performance of this pipeline to around 0.953–0.955. I will soon build the next pipeline from scratch for ensembling, hoping to push even higher.

**↳↳ Starry (3rd) — Posted a month ago**

> Hi Boredom, it is an amazing result, may I ask if you use unlabeled train_soundscapes in your 0.950+ model. And if so, what the score with only train_audio + labeled train_soundscapes ?

**↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (1-iter PL on unlabeled; 0.938-0.939 w/o)

> Yep！this is the result of the first iteration on unlabeled data. Maybe one or two more iterations are possible. A single model using only labeled data achieves an LB of 0.938–0.939. I've never been able to break 0.94 with a single model. Compared to others with solid work, there might still be a significant gap in potential.

**↳↳ Shiro (2192nd) — Posted a month ago**

> Impressive! does it include TTA/postprocessing during inference? Is there a big gap between with/without those postprocessing in term of LB score?

**↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (no TTA, simple PP)

> no tta，950+ lb include some simple pp，these can improve about 0.03-0.04

**↳↳↳↳ Kurise (115th) — Posted a month ago**

> Are you sure? I don't know how "simple" pp can enhance 0.04😨

**↳↳↳↳↳ Boredom (5th) — Posted a month ago** ⭐ (PP correction to +0.004)

> sorry，0.004。haha😂

**↳ Cody_Null (310th) — Posted 22 days ago**

> @cudacoding what sort of cv are you getting for your single model .95+ LB? If you are feeling extra generous I would also be super curious about your cv method? I was really excited about this competition because I saw they included a validation set this year, but there has proven to be a significant gap, curious if that still exists at the top end, and to what degree. I am sure the data is tough to manage but I really wish we simply had a collection of data we want to build a model for and we split files across train, public LB, and private LB so there is clear correlation for all of them but it must not be as simple as it seems.

**↳ Gaurav Rawat (1065th) — Posted a month ago**

> nice SED or vision model ?

**↳↳ Boredom (5th) — Posted a month ago** ⭐ (timm + SED)

> timm backbone+SED

---

### Thread 11 — Ali Ozan Memetoglu's initial 0.925 (no unlabeled)

**Ali Ozan Memetoglu (6th) — Posted 2 months ago** ⭐ (0.925 baseline, no unlabeled)

> Mine is at 0.925 without using the unlabeled data. I couldn't make good use of it, and it might not even be that useful in this competition. But I might be wrong.

**↳ Tom (14th) — Posted 2 months ago**

> I think the floor of top scores would be very high in the end

**↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (winning solution = unlabeled)

> "nd it might not even be that useful in this competition. But I might be wrong."
>
> it should be very useful! in fact winng solution is how you make use of it

---

### Thread 12 — Jake's 0.915 / 0.922 (small unlabeled lift)

**Jake (34th) — Posted 25 days ago** ⭐ (small unlabeled lift +0.007)

> Best solo model 0.915 w/o using any of the unlabelled soundscape data. 0.922 using the unlabelled soundscape data.

**↳ BNBU Anpeng Yuan (491st) — Posted 18 days ago**

> May i ask how to choose the number of unlabelled filels?

---

### Thread 13 — Tucker Arrants's 0.93+ Perch-distilled B0 5-fold

**Tucker Arrants (196th) — Posted 25 days ago** ⭐⭐ (0.93+ recipe + 0.922 no-PP, KD lift +0.02)

> Finally broke 0.93 with a single model only using focal recordings and labeled soundscapes.
>
> Perch distilled B0 trained on 5s clips. Simple BCE loss, waveform augmentations, and moderate post processing (site/hour priors + Gaussian smooth)
>
> Without any post processing it is 0.922

**↳ Salman Ahmed (39th) — Posted 25 days ago**

> HI @tuckerarrants Did you use site/hour priors from labeled soundscapes only? Also is this a single model or 5 folds of the same model?

**↳↳ Tucker Arrants (196th) — Posted 25 days ago** ⭐ (5-fold KFold filenames + MultiLabelStrat)

> Labeled soundscapes only for the site/hour priors, for now…
>
> This is 5 folds of the same model, simple KFold on filenames and MultiLabelStrat on focal labels.

**↳↳↳ Jiacheng Ma (1332nd) — Posted 14 days ago**

> Hi, How do you add "priors" to your model? Transfer to logits space?

**↳ chenwenqiang_001 (2073rd) — Posted 22 days ago**

> what is the public score without distilling.the public score is 0.732 without using distilling on your Perch distilled B0.

**↳↳ Tucker Arrants (196th) — Posted 20 days ago** ⭐⭐ (PERCH KD LIFT QUANTIFIED AT ~+0.02)

> It is about 0.02 less than with distillation, so around 0.89, definitely much higher than 0.732. A simple linear probe on frozen perch features will get you in the mid 0.8s, so something is wrong if you are not above that.

**↳↳↳ chenwenqiang_001 (2073rd) — Posted 14 days ago** ⭐ (own-teacher KD = +0.013)

> appreciate your reply.I once used a single mode of 0.886 as the teacher and trained a student mode the same as the teacher.It increased 0.013.

---

### Thread 14 — XuKong Ji's 5-fold regression + 10s drop

**XuKong Ji (1609th) — Posted a month ago** ⭐ (5-fold WORSE than 1-fold; 10s drops 0.86)

> I have a single fold(one of 5folds) model(effecient b0+LSEHead + BCE) score 0.899 with 5s input/time smooth/time shift tta(shift 0s and 2.5s). I got something weird:
>
> - 5fold ensemble score 0.889;
> - After incresing input to 10s, score drop to 0.86(1fold)…

---

### Thread 15 — OpPrime's 0.826 + EliKal's advice

**OpPrime (655th) — Posted a month ago**

> I got 0.826 on EfficientNetB0 with 5-fold training, 0.835 on fold 0, 10 epochs per fold, feeding in 5 seconds clips only. Notebook and weights shared now.
>
> Following Salman's post below, I then got a 0.801 from SED HGNetV2 feeding it 20s of mel with a temporal attention head.
>
> I am now tracing how and where I am going wrong as my numbers don't match with others.

**↳ EliKal (2044th) — Posted a month ago** ⭐ (0.915+ ensemble of 2 from 5s models)

> I'd start by focusing on your 5s models. You can get 0.915+ with an ensemble of just two models. For 20s inputs, I'd try adding segment-wise labels from the labeled soundscapes. That said, you'll probably need to adjust the backbone or increase the resolution, since a larger context usually means a larger model. It might be worth trying 10s inputs first with the same backbone and resolution, just to see how things behave. My 10s model (EfficientNet-B0 as a starting point) isn't ready yet, I hope my intuition will work.

---

### Thread 16 — Gaurav Rawat's CV/LB data + GKF discussion

**Gaurav Rawat (1065th) — Posted a month ago** ⭐ (CV/LB data — poor monotonicity)

> - 0.82 cv lb 908
> - 0.94 cv lb 913
> - 0.97 cv lb 926
> - 0.87 cv lb 916 :/
> - 0.942 cv lb 915
> - SED 0.87 - 0.75 :(
>
> Weird but

**↳ Aic (837th) — Posted a month ago**

> May I ask about your CV strategy? Mine is quite unreliable, even though my SS_AUC is closely aligned with the LB.

**↳↳ Gaurav Rawat (1065th) — Posted a month ago**

> pretty simple groupk on site

**↳↳↳ EliKal (2044th) — Posted a month ago**

> Did you validate on xc, inat and ss data together?

**↳↳↳↳ Gaurav Rawat (1065th) — Posted a month ago**

> No, I validate strictly on Soundscapes (SS) using GroupKFold by Site as of now .

**↳↳↳↳↳ EliKal (2044th) — Posted a month ago**

> Interesting. When I tried to group the Soundscapes by site I had many missing species in the validation set. Your cv scores are OOF scores? It would make sense much more if they are

**↳↳↳↳↳ Anil Ozturk (2nd) — Posted a month ago** ⭐ (GKF site OOF=0.608 / LB=0.879 EXTREME DISCONNECT)

> Soundscape-val-only GKF on sites OOF: 0.60775 LB: 0.879
>
> lol

**↳↳↳↳↳↳ Tucker Arrants (196th) — Posted a month ago** ⭐ (DUAL-VAL PROPOSAL)

> GKF is overly harsh for this competition because the test set has site overlap with training. GKF models can't learn site-level species priors that would help at test time — they're forced to hold out entire sites during training, so they never see the "this site has these species" signal that exists in reality. But removing GKF entirely is also risky: the model might memorize site-specific artifacts that don't transfer, and you'd have no way to detect that the model relies on site memorization versus learning general features.
>
> But you can try dual validation on a single training run: train on everything, but hold out two disjoint val sets. Val-A holds out specific recordings from seen sites (tests the regime: new recordings at seen sites). Val-B holds out an entire site (tests whether the model has any ability to generalize to unseen sites as a hedge against test sets that include unseen sites). Combine them into a weighted metric for early stopping, weighted toward Val-A because it matches the test regime more directly.

---

### Thread 17 — EliKal's 4-backbone Perch-KD cluster (0.906-0.909)

**EliKal (2044th) — Posted 2 months ago** ⭐⭐ (4 backbones at 0.906-0.909 with KLD-on-Perch)

> EfficientNet-B0 – 0.909 EfficientNet-B3 – 0.906 DINOv3-small – 0.906 ConvNeXt-small – 0.907 All models were trained using the same pipeline for 12 epochs (after 10 epochs of distillation with KLD loss), with mixup applied to spectrograms. After reading the comments, I'm considering replacing this with mixup on raw waveforms. Inference was performed using only time-shift TTA.

**↳ tennogh (33rd) — Posted 2 months ago**

> Interesting that you have so little difference between backbones. With my pipeline I have up to .02 difference between model families so far.

**↳↳ Jack (21st) — Posted a month ago** ⭐ (KD causes backbone stability)

> It's the distillation (assuming he's distilling perch) causing that stability. I experience the same swings as you with different backbones

**↳↳↳ EliKal (2044th) — Posted a month ago**

> Correct. Perch embeddings for all the architectures. Did you solve this stability "issue"?

**↳↳↳↳ tennogh (33rd) — Posted a month ago** ⭐ (KD reduces diversity: cos sim 0.5→0.67)

> Not sure if this is an issue, I guess it could make the models more similar and ensembling less beneficial. I'll check when I have a stable KD pipeline.
>
> Edit: with KD and 4 different backbones, I still have a ~0.02 difference between the worst and the best (the ranking is a little bit different too). KD also reduces diversity to an extent (cosine similarity ~0.5->~0.67), to the point where is might be worth ensembling KD and non-KD models.

---

### Thread 18 — D.M. (Don Mathis)'s 0.880 + replicability pain

**D.M. (2573rd) — Posted 2 months ago** ⭐ (0.880 not replicable; mamba-from-scratch ceiling 0.772)

> Mine is only 0.880, fine-tuning an imagenet-pretrained CNN backbone that takes spectrogram input. The backbone outputs low-temporal resolution BCT (T=10, C=1024), that I then map to 234 and pool with logit-LSE to get clip logits, trained with BCE loss. The .880 isn't really replicable, more like .875 only. Trained on train_audio and the labeled soundscapes. Using 5-second clips.
>
> I tried lots of mixing schemes with the unlabeled soundscapes but that didn't transfer to the LB for me. I can't imagine how it's possible to get something like .922 using the same data I'm using.
>
> I also trained a 4-layer residual mamba model from scratch (no pretrained network at all), but that topped out at .772.

**↳ hengck23 (1813th) — Posted 2 months ago** ⭐ (long-tail framing)

> "The .880 isn't really replicable, more like .875 only." this is due to small sample size in some class. bird clef is domain and long tail problem. the issue can be solved by better validation and training methods. Hence the potential of timm and "train from pretained CNN/scratch" can be higher than you think.
>
> although "train from pretained CNN/scratch" can achieve good results, using perchv2 may be faster and easier.

---

### Thread 19 — BNBU Anpeng Yuan's 0.910 4-fold HGNetv2 (no KD, no PL)

**BNBU Anpeng Yuan (918th) — Posted a day ago** ⭐ (HGNetv2 4-fold = 0.910 baseline; matches our ballpark)

> No distill, no pseudo label. 4 Folds = 0.910 with HGNetv2

---

### Thread 20 — Jack's "pseudo doesn't help solos" cluster (2026-05-26/27, VERY RECENT)

**Jack (17th) — Posted a day ago** ⭐⭐ (0.933 5-fold no PL / 0.934 single fold w/ PL — "huge +0.001 from PP")

> I've got 0.933 with 5-folds and no pseudo or distill, 0.934 with pseudo but one fold. No TTA or postprocessing on either.
>
> What increases are people getting from postprocessing? I've tried several methods and have achieved a HUGE +0.001. Feels like a waste of time with stronger models.
>
> edit: asked about postprocessing, thread became about pseudo lol

**↳ Jiacheng Ma (25th) — Posted a day ago**

> My pseudo is not up😭.

**↳↳ Jack (17th) — Posted a day ago** ⭐ (PL enables stronger ensembling, not solo lift)

> Tbh, I haven't seen much gain either from pseudo on solos, but it's allowed me to ensemble those solos and get a much better score in the same amount of time as that 5-fold model.

**↳↳↳ Jiacheng Ma (25th) — Posted a day ago**

> My situation is just like yours.😅

**↳↳ Anil Ozturk (6th) — Posted a day ago** ⭐⭐ (CUSTOMIZED PL: 931→940, 934→939; one pass max)

> yeah kinda same
>
> - supervised 931 - a very weird customized pseudo 940
> - supervised 934 - same very weird customized pseudo 939
>
> and it doesn't work for more than one pass. 😅

**↳↳↳ Jiacheng Ma (25th) — Posted a day ago**

> Could you share your mean "very weird customized pseudo"😜haha ? I tried the 2th place in the past year, but it don't work, even down to stage1.

**↳↳↳↳ Ali Ozan Memetoglu (7th) — Posted a day ago** ⭐⭐ (TRIM PROBABILITY recommendation)

> Try to increase the trim probability

**↳↳↳↳↳ Jiacheng Ma (25th) — Posted a day ago**

> Thank you very much. Could you clarify what you mean by 'trim probability'? Are you suggesting to change the sampling ratio between pseudo-labeled and focal samples? 😀

**↳↳↳↳↳↳ Ali Ozan Memetoglu (7th) — Posted a day ago** ⭐⭐ (TRIM = CELL-LEVEL ZERO BELOW THRESHOLD)

> Cell level any pseudo probability below it gets zeroed

**↳↳↳↳↳↳↳ Jiacheng Ma (25th) — Posted a day ago** ⭐ (Jiacheng tried 0.1 threshold, didn't work for him)

> Thank you for the clarification. 😁So you mean: before training, for each pseudo-labeled soft label vector, set any probability below a threshold (e.g., 0.1) to zero.（just like the 2nd place solution） Yes — I already do this when generating the pseudo labels. However, even with this, my self-training still does not work well.😭

**↳↳↳ Jack (17th) — Posted a day ago**

> Man, if I could get 0.940/0.939 from my pseudomodels I'd be straight chillin rn. I guess I should keep working on that hahaha. You doing any postprocessing?

**↳↳↳ Boredom (2nd) — Posted a day ago** ⭐ (CONFIRMS one-pass plateau)

> it doesn't work for more than one pass. +1 🤣

**↳↳↳↳ Jack (17th) — Posted a day ago**

> Same for you?

**↳↳↳↳↳ Zejun_ (15th) — Posted 8 hours ago** ⭐ (3RD CONFIRMATION: "only first student is smart")

> totally same, only first student is smart.

**↳↳↳ Salman Ahmed (43rd) — Posted a day ago** ⭐⭐ (0.939 solo no PL → 0.944 w/ PL → 0.952 ensemble; PL = +0.005)

> My single model without pseudo is 0.939 and with pseudo its 0.944 Ensemble goes to around 0.952 Second iteration of pseudo doesn't work for me. 😶😐

**↳↳↳↳ Jack (17th) — Posted a day ago**

> 0.952 is nice. What was the teacher score for getting 0.944?

**↳↳↳↳↳ Salman Ahmed (43rd) — Posted a day ago** ⭐ (TEACHER WAS 0.944 ENSEMBLE)

> Teacher is an ensemble that scores 0.944 without pseudo.

**↳↳↳ Tucker Arrants (685th) — Posted a day ago** ⭐⭐ (6 ROUNDS PL → 0.946 BUT ZERO ENSEMBLE BENEFIT)

> I have the opposite problem - 6 rounds of pseudo, best model 0.946, but virtually zero ensemble benefit 😆

**↳↳↳↳ MarkDjadchenko (109th) — Posted 21 hours ago** ⭐ (training-backbone-during-PL hypothesis)

> Maybe you are not training backbone during the pseudo-only distillation

**↳↳↳↳ Antoine Masq (33rd, Topic Author) — Posted 12 hours ago** ⭐ (SAME-TEACHER ENSEMBLE TRAP)

> Do you ensemble models that were trained with the same teachers ? If yes, it might be because they are too correlated with each other

**↳↳↳↳ To be happy (1149th) — Posted 6 hours ago** ⭐ (RANK ENSEMBLE BREAKS PL TARGETS)

> My teacher model is an ensemble. Because the models are quite diverse, I need to use rank ensembling to boost the LB, otherwise the score drops. However, when generating pseudo-labels, I noticed that the submission after rank ensembling doesn't represent true probabilities. So it seems I can only use hard labels, and I'm not sure how I should generate soft labels instead. Has anyone encountered a similar situation? I'd love to get some advice from the top competitors here.

**↳↳↳ Joseph (317th) — Posted a day ago** ⭐ (SOLO 0.949 > ENSEMBLE 0.947 — opposite of Jack's pattern)

> wow, your single model is 940 but ensemble to top 5? my best single model for now is 949 but 947 after ensemble…

---

## Synthesis: proven 0.93+ recipes (2026-05-21)

Single-model recipes that have publicly hit ≥0.93 LB.

### 1. Antoine Masq — 0.943 single, 0.948 ensemble (XC pretrain + own-teacher PL + custom ASL)

| Component | Value |
|---|---|
| Backbone | EfficientNetV2-B0 only |
| Pretrain | Xeno-Canto (own pretrain) |
| Fine-tune data | 2026 train_audio + labeled SS + **pseudo-labeled unlabeled SS** (own teacher) |
| Loss | Custom asymmetric loss (ASL variant; params undisclosed) |
| TTA | Sliding window + smoothing patterns |
| PL iteration | At least 2 teacher generations (0.936 → 0.943 from teacher upgrade) |
| Inference | 10 min |
| LB trajectory | 0.889 → 0.936 → 0.943 |
| Source | "non-Perch limit" thread, cross-ref [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md) |

### 2. Arunodhayan — 0.946 single (no details)

7th place. Posted only "my single model LB is 0.946" with no further explanation. Strongest non-Perch single-model data point disclosed.

### 3. Salman Ahmed — 0.937 v2B0 SED (no KD, no unlabeled)

| Component | Value |
|---|---|
| Backbone | EfficientNetV2-B0 (Fused-MBConv) — also B0_NS now matches |
| Head | "simple Attn head's framewise max is better than LSE head" |
| Loss | CE on clipwise + framewise max; **sigmoid at inference** |
| Window | 20s training segments (not split+max). 10s/15s "almost the same / sometimes better" |
| Mixup | Wave mixup |
| Epochs | 25; "more epochs make the model too confident" |
| Data | train_audio + labeled soundscapes (Salman says SS adds ~0 to LB; he sometimes only uses train_audio) |
| KD | None |
| External | BG noise from prior comps (nocall 2020/21, freefield1010, etc.) |
| PP / TTA | "I definitely rely on some post processing / TTA" |
| Soundscape-as-BGN | Does NOT use SS as BG noise (species in them) |
| Inference time | ~50-60 min |
| Breakthrough | "investigating attention values on highest/lowest loss examples" — modeling > tricks |

### 4. Tucker Arrants — 0.93+ Perch-distilled B0 SED (5-fold ensemble)

| Component | Value |
|---|---|
| Backbone | EfficientNet-B0 with **Perch distillation** |
| Window | 5s clips |
| Loss | BCE |
| Mixup | Waveform augmentations (no specifics; "simple") |
| Folds | **5-fold ensemble** (KFold on filenames + MultiLabelStrat on focal labels) |
| Data | Focal + labeled soundscapes only (no unlabeled) |
| PP | Site/hour priors (from labeled SS only) + Gaussian smooth — "moderate" |
| Without PP | 0.922 |
| With PP | 0.93+ |
| **KD lift quantified** | **~+0.02 LB** — without distillation ≈0.89, with distillation ≈0.91 (Tucker) |
| Comparison: own-teacher KD | chenwenqiang_001: 0.886 self-teacher → student +0.013 |

### 5. Boredom — 0.938-0.939 timm+SED (no unlabeled), 0.950+ (with unlabeled, 1 iter)

| Component | Value |
|---|---|
| Backbone | timm + SED, non-Perch |
| Inference | ~11 min total; **no TTA** |
| PP | "simple PP" worth ~+0.004 (corrected from 0.03-0.04 typo) |
| Unlabeled | 1 iteration of pseudo-labels → 0.938-0.939 → 0.950+ |
| Pipeline ceiling | 0.953-0.955 estimated |

### 6. Ali Ozan Memetoglu — 0.941 (Sydorskyi 2025 published checkpoint + own head)

| Component | Value |
|---|---|
| Approach | "Yoink" Sydorskyi's published 2025 backbone (819K-recording XC pretrain + bug fix) + train new classification head on 2026 |
| Pretrain corpus | Filtered Xeno-Canto (entire 2025 checkpoint, no folds) — Sydorskyi did this; Ali Ozan didn't redo it |
| Backbone source | Public dataset `vladimirsydor/bird-clef-2025-all-pretrained-models` (EfficientNetV2-S + ECA-NFNet-L0) |
| Attention dim | **1028** (Ali Ozan: "1028 is the attention dimension too") |
| Inference | ONNX, 5 folds, ~30 min total |
| Melspec | Must match Sydorskyi's exact mel pipeline (Jack: −0.012 LB swing on mismatch) |
| 2026 training | Standard fine-tune on 2026 data with the new head |

### What 0.93+ recipes have in common

1. **At least one of**: in-domain pretrain (Antoine XC, Ali Ozan Sydorskyi-yoink), Perch KD (Tucker, EliKal cluster), modeling refinement on attention (Salman 0.922→0.937).
2. **Simple post-processing**: site/hour priors + Gaussian smoothing (Tucker, Boredom). PP alone worth +0.004 to +0.010.
3. **5-fold ensemble OR a single-model breakthrough** — Tucker uses 5-fold; Salman/Boredom/Ali Ozan/Antoine get there single-model.
4. **Either CE or BCE** (Salman: CE+sigmoid-infer; Tucker, Antoine: BCE/ASL).
5. **5s, 20s, or "doesn't matter" window** — Tucker, Boredom, EliKal use 5s; Salman uses 20s. 10s is the "death zone" (XuKong Ji, OpPrime).
6. **NOT**: special head architecture, exotic loss (vanilla ASL without tuning gets −0.03 per To be happy / Dhruv), 3-channel mel, 15s multi-context as the silver bullet.

### Where our 0.906 stalls vs 0.93+

| Lever they have / we don't | Estimated LB lift | Source |
|---|---:|---|
| In-domain pretrain (Sydorskyi backbone yoink OR own XC pretrain) | **+0.030** | Ali Ozan 0.941; Antoine 0.936 (own XC) |
| Own-teacher PL on top of stronger base | **+0.007 to +0.013** | Antoine 0.936→0.943; Boredom 0.938→0.950; chenwenqiang_001 0.886→0.899 |
| Perch knowledge distillation (KLD/MSE on Perch outputs) | **+0.020** | Tucker (with KD ≈0.91, without ≈0.89) |
| Site/hour priors at inference (from labeled SS) | +0.005-0.010 | Tucker, our pp2 |
| 4 or 5-fold ensemble (proper KFold + ensembling) | +0.005-0.010 | v5j → v5j-4fold (caveat: XuKong Ji reports 5-fold WORSE than 1-fold) |
| Attention-quality investigation (Salman) | +0.015 | Salman 0.922→0.937 |

---

## Key takeaways

### Top scores & unlabeled data (2026-05-21)

1. **Arunodhayan: 0.946 single, no Perch** — 7th place, no details disclosed.
2. **Boredom: 0.950+ single model WITH unlabeled data** — non-Perch, timm+SED, ~11min. First iteration on unlabeled. Without unlabeled: 0.938-0.939. Says pipeline ceiling may be 0.953-0.955.
3. **Antoine Masq: 0.943 single, no Perch** — EffNetV2-B0 + XC pretrain + own-teacher PL + custom ASL. 10 min inference. Cross-ref non-Perch limit thread.
4. **Ali Ozan: 0.941 single using Sydorskyi 2025 published backbone** — direct download of `vladimirsydor/bird-clef-2025-all-pretrained-models`, then own head on 2026. Attention dim 1028. ONNX 5-fold ~30 min. Jack confirms parallel attempts; melspec must match exactly (−0.012 LB cost from mismatch).
5. **Salman: 0.937 solo EfficientNetV2-B0** — jump from 0.922 to 0.937 via attention investigation on hardest/easiest loss examples. B0_NS now matching v2B0. Uses PP/TTA. Simple checkpoint (no model soup).
6. **Unlabeled data is THE differentiator for 0.94+** — Boredom: "never been able to break 0.94 with a single model" using labeled data only. 1 iteration of unlabeled → 0.950+.
7. **Tucker quantifies Perch KD at ~+0.02 LB**: with distillation ≈0.91, without ≈0.89. chenwenqiang_001: own-teacher KD (0.886 → 0.899, +0.013).

### Salman's 0.922→0.937 breakthrough

8. **"Modeling" is the key, not just data/augmentation** — Salman: "for me it is one of the most important parts."
9. **Attention investigation on loss extremes** — examined attention head outputs on hard vs easy training examples, debugged from there.
10. **EfficientNetV2-B0 (Fused-MBConv)** at 0.937. B0_NS now also matches with current training strategy.
11. **No model soup, simple checkpoint.**
12. **Post-processing + TTA both required.**
13. **Soundscape training currently hurts Salman** (0.85 with SS) — suspects a bug. Does NOT use soundscapes as background noise ("there are species in those SS recordings").

### Architecture & training

14. **20s full-audio training** (not split+max) was Salman's 0.922 approach. 10s/15s "almost the same / sometimes better."
15. **CE loss + sigmoid inference** remains Salman's loss approach.
16. **Waveform augmentations** (pink noise, time stretch) recommended by yukiZ.
17. **Offline val/auroc is unreliable**: yukiZ's data shows correlation but with significant variance.
18. **Seed sensitivity**: Mahog reports seed changes shift LB by 0.005.

### 5-fold regression

19. **XuKong Ji: 5-fold ensemble (0.889) WORSE than single fold (0.899)** — with EffNet-B0 + LSEHead + BCE. Also confirms 10s input drops to 0.86. Naive ensembling can hurt; 5s is the right window size.

### CV/LB correlation

20. **Gaurav Rawat's CV/LB data (GroupKFold on site)** — non-monotonic (e.g. 0.82 CV → 0.908 LB but 0.87 CV → 0.916 LB).
21. **Anil Ozturk: soundscape-val-only GKF on sites** — OOF 0.608, LB 0.879. Extremely poor predictive value.
22. **Tucker's dual-validation proposal**: train on everything, hold out two disjoint val sets. Val-A holds out specific recordings from seen sites. Val-B holds out an entire site. Combine into a weighted metric for early stopping.

### Multi-context & architecture

23. **hengck23: "context and augmentation is the key"** — 0.896 LB with CNN only (no Perch, no PP) using long input wave.
24. **Tom (14th) confirms similar approach** — "fits on 4x5sec."
25. **Multi-context (hengck23)** = ONE backbone pass on 15s, then SPLIT temporal features into 3 chunks, apply LSEHead per-chunk → 3 predictions. The backbone's convolutional receptive field provides cross-window context, but each prediction is still from a 5s chunk. Different from Tattaka's train_period/infer_period crop. **LB 0.876 → 0.896 (+0.020)**, 4-fold, no PP.
26. **Our v5i (10s single window) got 0.880 LB (worse than v5j's 0.906 at 5s)** — different from multi-context because v5i pools over the full 10s. hengck23's approach preserves 5s pooling resolution per chunk while adding backbone-level context.
27. **Our v7 Phase D (AttHeadMultiContext) failed badly** (−0.054) because we changed the head AND aggregated over all 15s.

### Knowledge distillation & Perch KD

28. **EliKal: KLD loss distillation from Perch** — 10 epochs distillation + 12 epochs finetune achieves 0.909 with EfficientNet-B0. Four backbones (EffNet-B0/B3, DINOv3-small, ConvNeXt-small) all land at 0.906-0.909.
29. **EliKal confirms using Perch embeddings** for all architectures.
30. **Jack: Perch KD causes backbone stability** — less difference between backbones. Without KD, same swings as tennogh.
31. **tennogh: 0.02 diff between model families w/o KD; with KD: cos similarity increases 0.5→0.67.** "Might be worth ensembling KD and non-KD models."

### Perch without deep learning

32. **Tucker Arrants: 0.925 without deep learning** — "simple clustering on the frozen perch embeddings and metadata engineering." MarkDjadchenko (74th) asks for the clustering algorithm — no public reply.

### hengck23's diagnostic framework

33. **Root causes for 0.87+ vs 0.91+**: long tail (some class has few train samples) + domain shift (focal=high SNR single species, soundscape=low SNR multi-species).
34. **Key metric to monitor**: logit std as "proxy to logit distribution" — overfit models show high train logit std growing over training.
35. **Robust model traits**: low logit std, train-val agreement, can train for very long epochs, flat loss valley for low sample classes, low AUC variance across folds.
36. **Overfit model traits**: train-val disagreement, high growing logit std, low train BCE + high train AUC.

### Salman's head preference

37. **Salman says "Attn head's framewise max is better than LSE head"** — we use LSEHead, he uses AttHead.
38. **hengck23 suggests our 0.906 may benefit from "temporal"** — but we tried multi-context and it regressed.

### External data & 2-stage training

39. **Salman uses background noise from prior competitions** — nocall/BG noise from BirdCLEF 2020/2021 + rainforest + environment sound + freefield1010/warblrb/birdvox. Says "same LB without it" but "helpful to generalize better." Does NOT use soundscapes as BG noise.
40. **Ali Ozan: BirdCLEF 2025 pretrained checkpoint → 0.941** — using Sydorskyi's *publicly downloadable* checkpoint with a new head trained on 2026. Jack confirms parallel attempts. Attention dim 1028; melspec must match exactly.
41. **Boredom: 0.938-0.939 w/o unlabeled, 0.950+ with** — first iteration of pseudo-labels on unlabeled data. Uses timm backbone + SED.
42. **Antoine Masq: 0.943 single, XC pretrain + own-teacher PL** — own XC pretrain (not Sydorskyi-yoinked). PL from own teacher, simple (no confidence threshold).

### Sydorskyi backbone reuse

43. **Ali Ozan + Jack are both reusing Sydorskyi's published 2025 pretrained backbone** without redoing the 819K-recording XC pretrain. Public dataset: `vladimirsydor/bird-clef-2025-all-pretrained-models`.
44. **Two architectures in the dataset**: EfficientNetV2-S (`tf_efficientnetv2_s_in21k`) and ECA-NFNet-L0. Both ImageNet-21k → 819K-recording in-domain pretrain.
45. **Attention dim 1028** (Sydorskyi-specific) — must be matched in any new head.
46. **Melspec must match exactly**: 32 kHz / n_mels=128 / n_fft=2048 / hop=512 / f_min=20 / top_db=80 / AmplitudeToDB → standardize → min-max. Jack: −0.012 LB swing on mismatch.
47. **ONNX inference runtime**: Ali Ozan reports ~30 min for 5 folds.

### Own-teacher pseudo-labels (NEW 2026-05-21)

48. **Own-teacher PL is structurally distinct from PL-on-unlabeled with Perch-as-teacher** (which our exp 014/015/025 rejected).
49. **Own-teacher PL lift is roughly +0.007 to +0.013**, mostly independent of teacher strength within the 0.88-0.94 band:
    - Antoine 0.936 → 0.943 (+0.007) from improved teacher
    - Boredom 0.938 → 0.950 (+0.012) from 1-iter
    - chenwenqiang_001 0.886 → 0.899 (+0.013) from own-teacher KD
50. **Antoine's PL protocol is the simplest**: pseudo-label unlabeled SS with current best model, treat as additional labeled rows, retrain. No filtering, no confidence threshold disclosed.

### Pseudo-label iteration findings (NEW 2026-05-27 — Thread 20, Jack's cluster)

This is the most important new content. Multiple top teams independently
report a consistent pattern that supersedes some of the 2026-05-21 takeaways.

51. **PL doesn't lift solo models much for several top teams.** Jack (17th):
    "I haven't seen much gain either from pseudo on solos." Anil Ozturk (6th):
    supervised 0.931 → PL 0.940 (+0.009) and 0.934 → 0.939 (+0.005) — modest,
    not headline. Salman: 0.939 → 0.944 (+0.005) — also modest. **Earlier
    Boredom/Antoine/chenwenqiang headline numbers (+0.012, +0.007, +0.013) are
    upper-band, not typical.** Median real-world solo PL lift is closer to
    +0.005.

52. **"Trim probability" is the critical detail (Ali Ozan, 7th).** Cell-level
    rule: any pseudo-probability below a threshold (e.g., 0.1) gets zeroed
    in the parquet *before training*. "Same as the 2nd place solution"
    (Jiacheng Ma). **Our exp 033 did NOT trim** — we kept soft probabilities
    everywhere ≥ 0.5 but didn't zero the noise below that. This may be one
    cause of our REJECT verdict (the protocol can amplify low-confidence
    noise on a weak base).

53. **PL doesn't iterate past round 1 — multiple independent confirmations.**
    Anil Ozturk: "doesn't work for more than one pass." Boredom: "+1." Zejun_
    (15th): "totally same, only first student is smart." Salman: "Second
    iteration doesn't work for me." **Antoine's 0.936→0.943 may be the
    exception, not the rule.** This changes the strategic value of own-teacher
    PL — it's a one-shot lever, not an iterative ladder.

54. **Where PL DOES pay off: ensembling.** Jack: "[PL] allowed me to ensemble
    those solos and get a much better score in the same amount of time as
    that 5-fold model." Salman: 0.939 solo → 0.944 w/ PL → **0.952 ensemble**
    (+0.008 from PL → +0.008 from ensemble). **PL's primary value is enabling
    diversified ensembles, not lifting solo LB.**

55. **Tucker's contrarian data point**: 6 rounds of PL → best model 0.946
    BUT virtually zero ensemble benefit. Inverts the Jack/Salman pattern.
    MarkDjadchenko's hypothesis: "Maybe you are not training backbone during
    the pseudo-only distillation" — i.e., distillation-only continuation
    may freeze the encoder, losing diversity. Antoine: "Do you ensemble
    models that were trained with the same teachers? If yes, it might be
    because they are too correlated." **Implication**: PL-iterated models
    share the teacher's signal → tight cosine similarity → poor ensemble
    decorrelation.

56. **Rank-ensemble teacher → can't generate soft PL targets.** "To be happy"
    (1149th): rank-ensembling for teacher boosts LB but distorts probabilities
    → can only use hard labels for PL. Open problem on the discussion.

57. **Joseph (317th) reports the inverse pattern**: solo 0.949 > ensemble 0.947.
    Some pipelines saturate solo before ensembling can help. Less common.

### Implications for our exp 033 REJECT verdict (NEW 2026-05-27)

58. **Exp 033 omitted "trim probability"** (Ali Ozan's critical detail). Our
    PL parquet kept all probabilities ≥ 0.5, but did NOT zero cell-level
    probabilities below ~0.1. This is the closest thing to a "missing
    protocol piece" identified in this thread.

59. **Solo PL lift is modest even on stronger bases.** Anil Ozturk 0.931→0.940
    is the strongest disclosed; Salman 0.939→0.944 is more typical. Our base
    (0.906) is lower than either; if the protocol gives +0.005 on a 0.94
    base, it may give less on a 0.906 base — or net negative if the trim
    detail isn't right.

60. **PL → ensemble is the real lever, not PL → solo.** Even if exp 033's
    individual cells stayed at 0.886-0.899, an ensemble of those 10 cells
    might lift if we trust Jack/Salman's pattern. **Untested possibility**:
    rank-ensemble of our 10 PL'd students vs the distilled-sed-kd 5-fold
    ensemble (0.915 LB) as a stretch.

61. **One-shot, not iterative.** Antoine's "0.936 → 0.943 from teacher
    iteration" is now an outlier — 4 other top teams say iteration plateaus
    at round 1. Don't plan Round 2 PL experiments.

---

## Occam's-razor priorities for us (updated 2026-05-27, post-exp-033-REJECT)

Three paths to ≥0.93+ for a non-Perch SED, ranked by EV in the remaining ~7 days (deadline 2026-06-03):

**Path A — Sydorskyi backbone yoink (Ali Ozan / Jack path).** Download
`vladimirsydor/bird-clef-2025-all-pretrained-models`. Attach own head matching
attention dim 1028 + melspec exactly (Jack: −0.012 LB on mismatch). Fine-tune
on 2026. Single-model LB target **0.92-0.94**. Cost: ~2-3 days. **Highest EV
remaining lever; Antoine's 0.943 and Ali Ozan's 0.941 both rely on this kind
of XC-pretrained base.**

**Path B — Ensemble lever (cross-decorrelated stacks).** Average exp 023
(Perch probes ensemble, 0.929 LB) with exp 032 distilled-sed-kd-5fold (0.915
LB). Pure submission-time, no retraining. Estimated landing 0.92-0.93.
Cost: ~1 hour code + 1 LB slot. **Best EV/cost ratio of any remaining lever
given exp 023's 0.929 is already validated.**

**Path C — Own-teacher PL with trim probability (re-attempt exp 033 with
the missing detail).** Re-run exp 033 a2-kd canonical cell BUT add the
cell-level trim (zero pseudo-probabilities < 0.1) before training. Ali Ozan
flagged this 2026-05-27. May fix exp 033's REJECT verdict. Cost: ~1 day
retraining + 1 LB submission. **Speculative — Jiacheng Ma already does this
trim and STILL fails, so the effect size is uncertain.**

**Path D (deferred) — Salman's attention investigation.** Compute attention
maps on hard-loss and easy-loss training examples. Look for uniform attention,
frame saturation, attention not localizing on call onsets. Iterate head/loss
to fix observed failure modes. Pairs with any base model. Cost: 1-2 days of
debugging + 1 retrain. **Defer until Path A or B lands.**

**Path E (deferred) — Salman's PP/TTA layer.** Site/hour priors at inference
+ Gaussian smoothing — Tucker quantified ~+0.005-0.010 from this. Pure
inference-time. **Defer; small lever vs. Path A.**

**Combined ceiling estimate**: 0.94-0.95 single-model, 0.95+ ensemble.
Arunodhayan 0.946 and Boredom 0.950+ are the empirical ceilings for non-Perch
single-model right now.

This supersedes the earlier exp 030 mixup-recipe and exp 031 KD-on-HGNet-LSE
work — those levers are smaller than the unexplored levers above.
**Exp 033 own-teacher PL (without XC pretrain base) is now also empirically
refuted on our codebase** — the protocol doesn't transfer at LB without the
stronger base. Path A unlocks Paths B/C/D/E by giving them a 0.93+ starting
point.

### Revised PL strategy (post-Thread 20)

If we revisit own-teacher PL after Path A, do it Salman-style:
- **One pass only** — don't plan Round 2 (4 independent confirmations of
  plateau).
- **Trim probability < 0.1 at cell level** before training (Ali Ozan).
- **Goal = enable ensembling, not lift solo** — train multiple base models
  with diverse seeds/augmentations, PL each with the same teacher, ensemble
  the PL'd students. Jack/Salman both confirm PL's primary value is
  diversified ensembling (Salman: 0.939 → 0.944 → **0.952 ensemble**).
- **Avoid same-teacher correlation trap** (Antoine warning) — ensemble PL'd
  models trained from DIFFERENT teachers if possible.

---

## Cross-references

- `user/murilo/notes/kaggle-discussion-non-perch-limit.md` — companion thread; Arunodhayan 0.946, Antoine 0.936→0.943 full trajectory, Jack's "tried every method" failure mode, MengYe ASL+HGNet recipe.
- `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` — Tucker's notebook deep-dive.
- `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` — code audit of Tucker's notebook.
- `user/murilo/notes/2026-05-21-nina2025-eos6-protossm-stack-analysis.md` — the Perch-heavy 0.949 ensemble side; what this thread is rebelling against.
- `user/murilo/notes/yuriy-protossm-v5-annotated.md` — annotated ProtoSSM v5 reference (LB 0.925).
- Our exp 016 (ASL rejected on HGNet baseline) — worth re-running with Sydorskyi backbone + PL.
- Our exp 029 (2025-only pretrain rejected at 28K-row scale) — was the right idea, wrong scale; Sydorskyi-yoink supersedes.
- Our exp 014/015/025 (PL rejected with Perch-as-teacher) — was the right concept, wrong teacher; own-teacher PL is what Antoine/Boredom/chenwenqiang_001 use.


---

# § 8. COMPANION DOC: Kaggle thread "non-Perch limit"

*Source: `user/murilo/notes/kaggle-discussion-non-perch-limit.md`*

# Kaggle Discussion: What's the limit without any Perch involved?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/ (BirdCLEF+ 2026 discussion forum, post by an unidentified user asking about non-Perch ceilings)
- **Posted**: ~2026-05-17 (estimated from "Posted 4 days ago" at capture time)
- **Captured**: 2026-05-21
- **Status at capture**: 14 comments
- **Companion thread**: [`kaggle-discussion-best-single-model-lb.md`](kaggle-discussion-best-single-model-lb.md) — many of the same people show up in both
- **Companion vendored notebook**: `vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/` (the Perch-heavy 0.949 stack — the thing this thread is asking how to compete with *without* Perch)

---

## TL;DR

Three independent recipes hit 0.93+ on the LB **without any Perch** as of 2026-05-21:

| Person | Single-model LB | Recipe summary |
|---|---:|---|
| **Arunodhayan** (7th) | **0.946** | No details disclosed |
| **Antoine Masq** (49th) | **0.943** | EffNetV2-B0 + XC pretrain + custom ASL loss + own-teacher PL + sliding-window TTA + smoothing. 10 min inference. |
| **Boredom** (5th) | **0.938-0.939** (no PL); 0.950+ (1-iter PL) | timm + SED, ~11 min, non-Perch |
| **Salman Ahmed** (39th) | **0.937** | EffNetV2-B0 SED, "modeling on attention" investigation, PP/TTA. Documented in the other thread. |

The two big public takeaways from this thread:

1. **XC pretraining is necessary, not optional.** Antoine pretrained EffNetV2-B0 on Xeno-Canto BEFORE 2026 fine-tuning. Without it, his EffNetV2-B0 stalls at 0.88 (his original baseline). The Sydorskyi-yoink (Ali Ozan 0.941) is the same lever via a different XC checkpoint.
2. **Own-teacher pseudo-labels** add +0.007 LB on top of an already-strong (0.936) teacher (Antoine 0.936 → 0.943 from improved-teacher PL). Boredom reports +0.012 LB from 1-iter PL on his 0.938 base.

This thread also documents the **failure mode for Jack** (rank 21, training from scratch): "I've tried every method that has been discussed this year alongside the top two methods from last year. I haven't managed to see that increase people are getting." → confirms that copying-the-recipe-list-only-half-works without the unspoken details.

---

## Original post

> Hello everyone, I notice nearly all top-ranked notebooks on the leaderboard (including my own solutions) rely heavily on Perch in one way or another. Most either extract audio embeddings via Perch or adopt Perch-based knowledge distillation. I think this may have caused severe lack of model diversity among our approaches. In my exps, merging another extra Perch-based model into ensembles barely brings any effective gain — the final LB improvement is only around 0.001 at most, which is basically within random score fluctuation. For this reason, I've been trying to train pure CNN branches only using mel-spectrogram input with EfficientNet-B0, without introducing any Perch weights, embeddings or distillation strategies at all. Unfortunately, all my attempts can hardly break 0.88 on public LB. Here are my two main questions:
>
> 1. What is the approximate upper bound public LB score we can reach if we stick purely to CNN pipelines with zero Perch involvement?
> 2. Under current mainstream ensemble frameworks, can we still obtain valid leaderboard gains by adding well-trained non-Perch models, even if their single-model scores are below 0.9?

---

## Comments (chronological)

### Arunodhayan (7th) — Posted 3 days ago ⭐ CEILING DATAPOINT

> my single model LB is 0.946

No further details disclosed. This is the strongest non-Perch single-model data point published as of 2026-05-21. Rank suggests serious work behind it.

### yh-nnn (164th) — Posted 3 days ago

> How did you do it? Unbelievable.

(No recorded reply from Arunodhayan.)

### Zejun_ (11th) — Posted 3 days ago

> impressive score

### tennogh (33rd) — Posted 4 days ago ⭐ DIVERSITY CONFIRMS

> If you look at the best LB scores threads you will see people made strong models without any Perch involvement. Even using Perch, different methods can still be complementary, e.g. the public notebooks using Perch embeddings, the one using SED with distillation aren't very correlated.

**Reading**: pure-CNN models DO have non-Perch ensemble value. Tucker's distilled-SED CNN and the ProtoSSM-Perch-pipeline are uncorrelated enough to ensemble usefully (this is what the EoS.6 stack exploits).

### Antoine Masq (49th) — Posted 4 days ago ⭐ THE FULL RECIPE

> I'm only using custom CNN pipelines (EfficientNetV2-B0 only) without using Perch, and now my best single model gets a 0.936 LB score and I can reach 0.948 by ensembling different runs

### To be happy (1781st) — Posted 4 days ago

> Did you initialize your EfficientNetV2-B0 with pretrained weights?

### Antoine Masq — Posted 4 days ago ⭐ XC PRETRAIN CONFIRMED

> Yes and I also pretrained it on Xeno-Canto data

### Antoine Masq — Posted a day ago ⭐ MAJOR UPDATE TO 0.943

> Update: now my best single model is leading to a 0.943 LB score with efficientnetv2_b0 (10min for inference)

### Jack (21st) — Posted a day ago

> Is this with pseudolabels or no? What about tta or postprocessing?

### Antoine Masq — Posted a day ago ⭐ PL CONFIRMED

> Yes with pseudo-labels and the classic TTA with the sliding window and smoothing patterns

### Jack — Posted a day ago ⭐ FAILURE MODE

> Am I the only moron who hasn't managed to benefit from his pseudolabels in this competition? 🫠

### Antoine Masq — Posted a day ago

> Given your rank, you seem to do great anyway haha. Are you also training from scratch or do you use Perch? If it's the former you can have a look at last year's write-ups, they often discuss it

### Jack — Posted a day ago ⭐ "TRIED EVERYTHING, STILL STUCK" PATTERN

> From scratch. I've tried every method that has been discussed this year alongside the top two methods from last year. I haven't managed to see that increase people are getting. I've only managed to distill the ensemble into a solo model that achieves a slightly weaker score than the ensemble. I must be screwing something up but I haven't figured out what…

**Reading**: even rank-21 competitors training from scratch struggle to capture all the published levers. The published recipes are necessary but not sufficient — there's tacit knowledge in the implementation. This is comforting context for our 0.906 → 0.93 gap.

### Antoine Masq — Posted a day ago

> Strange, personally I just incorporated it by using the unlabeled data just as the labeled one but the labels are predicted by the teacher models

**Reading**: Antoine's PL recipe is the simplest possible — pseudo-label the unlabeled SS with a strong teacher, then treat them as additional training data alongside labeled data. No filtering, no confidence thresholding mentioned. Self-distillation in the classical sense.

### Jack — Posted 21 hours ago

> If you don't mind me asking, what loss are you using?

### Antoine Masq — Posted 21 hours ago ⭐ ASL CONFIRMED, DETAILS WITHHELD

> It's a custom ASL, but the details are a secret haha

**Reading**: Antoine's loss is Asymmetric Loss (ASL) with custom modifications. ASL is a multi-label classification loss (Ridnik et al. 2021) designed for long-tail problems. We tested ASL in exp 016 and rejected it on baseline HGNet. **Worth reconsidering now that we have a different starting recipe (XC pretrain + PL)** — the same loss may behave very differently when the backbone has already been XC-aligned and the targets are richer (own-teacher PL).

### Jack — Posted 20 hours ago

> Keep your secrets 😂 Was your 0.936 with pseudolabels and TTA as well or no? Thanks for the answers so far

### Antoine Masq — Posted 19 hours ago ⭐ 0.936 ALSO HAD PL

> The 0.936 was indeed with pseudo labels (in fact it was essentially the same but I improved the teacher model)

**Reading**: Antoine's trajectory was NOT "XC pretrain alone → 0.936 → +PL → 0.943". It was "XC pretrain + PL → 0.936 → improved teacher + PL → 0.943". So PL is load-bearing through the whole 0.88 → 0.943 climb. XC pretrain is what unlocked the recipe; PL is what compounds it.

The 0.936 → 0.943 delta (+0.007) is from **improving the teacher used to generate pseudo-labels**, not from adding more PL iterations.

### Dhruv Pai Dukle (1021st) — Posted a day ago

> Impressive without Perch Distillation was this with or without XC pre-training?

### Antoine Masq — Posted a day ago

> With Pre-training

### Donghui Zhang (1547th) — Posted 10 hours ago

> I'm curious how many iterations your pseudo-tags have gone through?

(No recorded reply.)

### To be happy (1781st) — Posted 9 hours ago ⭐ THE BOOTSTRAP QUESTION

> Curious to know, at what single-model LB score do you typically start ensembling, pseudo-labeling, or distillation?

### Dhruv Pai Dukle — Posted 3 hours ago

> You don't know. That's the beauty, only thing you can reliably do is check the model co-relation

**Reading**: there's no published threshold for "when is the teacher strong enough to PL." Empirically: Tucker quantified +0.013 LB from a 0.886 own-teacher; chenwenqiang_001 confirmed +0.013 from a 0.886-grade self-teacher. Boredom got +0.012 from his 0.938 base. Antoine got +0.007 from his 0.936 base.

The pattern: **PL lift is roughly +0.007-0.013, mostly independent of teacher strength** within the 0.88-0.94 band. So even a 0.886-0.906 teacher is "good enough" to start.

### MengYe (558th) — Posted 4 days ago ⭐ PUBLIC HGNET PATH

> you can use asymmetric loss with the public hgnet notebook and some tuning to reach 0.90+ lb, some public post-processing methods can make it reach 0.91+ lb

**Reading**: a concrete cheap recipe → **ASL + public HGNet notebook + tuning → 0.90+ → +PP → 0.91+**. Two independent levers stacked.

### BNBU Anpeng Yuan (491st) — Posted 3 days ago

> Hi Meng! In the public HGNet notebook, after changing the output chunk to 10 seconds, what corresponding adjustments should be made to the inference code? Since this is not an SED model and cannot produce frame‑level outputs, if the input becomes 10 seconds, the output also seems to become 10 seconds.

### To be happy — Posted 3 days ago ⭐ ASL REPRO FAILURE

> Are you referring to this asymmetric loss? Why did my LB score drop by 0.03 after implementing the modification based on the public hgnet notebook?

(Image attached: ASL code snippet, content not preserved in this capture.)

### Dhruv Pai Dukle — Posted 2 days ago

> Mine To honestly

**Reading**: ASL drops LB by 0.03 for at least one re-implementer. **ASL is recipe-sensitive** — wrong γ_+ / γ_- / threshold values destroy it. This is the same trap Jack flagged ("tried every method").

---

## Key competitor scores (this thread + cross-ref)

| Person | Rank | Single LB | Method | Notes |
|---|---:|---:|---|---|
| **Arunodhayan** | **7th** | **0.946** | No Perch | No details |
| **Boredom** | 5th | 0.938-0.939 (no PL); 0.950+ (PL) | timm+SED, non-Perch | Other thread |
| **Ali Ozan** | 6th | 0.941 | Sydorskyi 2025 XC backbone + own head | Cross-ref [`kaggle-discussion-best-single-model-lb.md`](kaggle-discussion-best-single-model-lb.md) |
| **Salman Ahmed** | 39th | 0.937 | EffNetV2-B0, "modeling on attention" | Other thread |
| **Antoine Masq** | **49th** | **0.943** | EffNetV2-B0 + XC pretrain + own-teacher PL + ASL + sliding-window TTA + smoothing | 10 min inference, full recipe disclosed minus ASL details |
| **Jack** | 21st | (not disclosed in this thread; "didn't manage to see the increase") | From scratch, tried "every method" | Public-recipe-only failure mode |
| **tennogh** | 33rd | (cross-thread) | KD on Perch | Confirms ensemble value across non-Perch and Perch models |
| **MengYe** | 558th | not stated, claims path to 0.91+ | Public HGNet + ASL + PP | The cheap-recipe path |

---

## Antoine Masq's full recipe (synthesized from this thread + best-single-model thread)

This is the cleanest disclosed non-Perch recipe at the 0.94+ tier:

| Stage | Detail |
|---|---|
| **Backbone** | EfficientNetV2-B0 (only — single architecture, no ensembling across architectures within single-model score) |
| **Pretrain** | Xeno-Canto (Antoine's own pretrain, not Sydorskyi-yoinked) |
| **Fine-tune data** | 2026 train_audio + labeled_soundscapes + **pseudo-labeled unlabeled soundscapes** (PL targets from a teacher model) |
| **Loss** | Custom asymmetric loss (ASL variant; specific parameters undisclosed) |
| **Mixup** | (not disclosed in this thread; probably waveform mixup per the broader pattern) |
| **PL recipe** | Use unlabeled SS just like labeled SS, with teacher's predictions as labels. Iterate teacher: each generation of student becomes next teacher. |
| **TTA** | "Classic sliding window" — most likely 5s window slid by some hop across the 60s clip, predictions averaged |
| **Post-processing** | Smoothing patterns (specifics undisclosed) |
| **Inference** | 10 min total |
| **Single-model LB trajectory** | 0.889 (baseline, March) → 0.936 (XC pretrain + initial PL) → 0.943 (improved teacher PL) |
| **Ensemble LB** | 0.948 (ensembling "different runs" of the same recipe) |

**Critical implementation gap** between this and our codebase:
- We don't have XC-pretrained EffNetV2-B0 (we have HGNet+LSEHead from-scratch at 0.906, and Tucker's NS-JFT init at the timm-default level).
- We've never run own-teacher PL (exp 014/015/025 used Perch as teacher, not a 0.906 SED model).
- ASL is implemented but was rejected on baseline HGNet (exp 016).

---

## Implications for our path

### What this thread proves

1. **Non-Perch single-model 0.93+ is reproducible** by at least 4 different people with disclosed (or partly-disclosed) recipes. It's not a one-off lucky run.
2. **The two essential levers are**: (a) in-domain pretrain (XC), (b) own-teacher PL on unlabeled SS. Stack them, and you get 0.94+.
3. **The ASL / loss / head choice is recipe-sensitive but small**: it's the 3rd lever, ~+0.005-0.010 swing depending on tuning. Get the first two right or it doesn't matter.
4. **PL recipe is simple**: pseudo-label unlabeled SS with current best model, treat them like additional labeled rows, retrain. No filtering/confidence-threshold protocol disclosed by anyone. The "iteration" is teacher-quality iteration, not threshold tuning.

### What this thread does NOT settle

1. **How much XC pretrain is "enough"**: Sydorskyi pretrained on 819K-row XC; Antoine doesn't disclose his scale; exp 029 tried 28K and rejected it. The right scale is somewhere between 28K and 819K — we don't know.
2. **ASL parameters**: γ_+/γ_-/clip threshold are the make-or-break knobs. To be happy & Dhruv both report −0.03 LB from wrong-ASL.
3. **Antoine's TTA specifics**: "classic sliding window" — window size, hop, smoothing kernel all undisclosed.
4. **Whether multi-architecture diversity matters within single-model PL chains**: Antoine uses only EffNetV2-B0. Some Perch-stacks blend many architectures. The non-Perch tier seems okay with single-arch.

### Direct implications for our Track 1 (B → A) plan

Track B (Sydorskyi backbone yoink) skips the XC-pretrain blocker entirely:
- We download Sydorskyi's 819K-pretrained EffNetV2-S / ECA-NFNet-L0.
- We attach a new head (matching attention dim 1028 + melspec exactly per Ali Ozan/Jack).
- We fine-tune on 2026 focal + labeled SS.
- LB target: **0.92-0.94** (Ali Ozan 0.941 is the proof point; we'd probably land a bit lower without his unstated tweaks).

Track A (Antoine recipe) then layers PL on top:
- Use the Track B model as the initial teacher.
- Pseudo-label all unlabeled SS.
- Retrain Track B with focal + labeled SS + pseudo-labeled SS.
- LB target: **+0.007 to +0.013** on top of Track B (so 0.93-0.95 if Track B lands 0.92-0.94).
- Iterate teacher: feed student-2 back as teacher for round-3.

**Combined ceiling estimate**: 0.93-0.95 single-model, 0.94-0.96 ensemble. Antoine + Boredom are the proof points.

---

## Open questions to resolve before committing

1. **Storage cost of Sydorskyi backbones**: how large are the checkpoint files? Local download + Kaggle upload needed for inference.
2. **Does Sydorskyi's 2025 backbone work on 2026 species?** The 2025 classifier head saw 2025 species. The backbone is generic acoustic features. Ali Ozan trains a new head on top — confirmed working at 0.941.
3. **Confidence threshold for PL**: should we filter pseudo-labels below threshold X, or trust the teacher as Antoine does (no filtering)?
4. **Inference time budget**: 10 min (Antoine) vs 30 min (Ali Ozan 5-fold ONNX). Kaggle hard limit 90 min. Our v5j-4fold takes ~40 min including ensemble. Plenty of headroom.

---

## Cross-references

- `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` — companion thread; many same people; canonical recipe-comparison home (also being updated 2026-05-21)
- `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` — the Perch-distilled side that this thread is rebelling against
- `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` — Tucker's code audit
- `user/murilo/notes/2026-05-21-nina2025-eos6-protossm-stack-analysis.md` — the Perch-heavy 0.949 ensemble; this thread is the alternative path
- Our exp 016 (ASL rejected on HGNet baseline) — worth re-running with Sydorskyi backbone + PL
- Our exp 029 (2025-only pretrain rejected at 28K-row scale) — was the right idea, wrong scale; Sydorskyi-yoink supersedes
- Our exp 014/015/025 (PL rejected with Perch-as-teacher) — was the right concept, wrong teacher; own-teacher PL is what Antoine/Boredom use


---

# § 9. COMPANION DOC: MODEL-RUNS.md (full experiment ledger)

*Source: `user/murilo/MODEL-RUNS.md`*

# Model Runs

Single source of truth for all trained models, their checkpoints, training data, and scores.
Referenced by experiment proposals, pseudo-labeling configs, and submission pipelines.

## Naming Convention

Model slugs follow: `{encoder}-{year}-{variant}`

- `{encoder}`: `protoclr`, `convnext`, `eat`
- `{year}`: `2025`, `2026`
- `{variant}`: training data / method (e.g., `soundscape`, `fulldata`, `pseudo-v1`)

## BirdCLEF 2025 Models (exps 006–009)

Trained on BirdCLEF 2025 focal recordings (28,564 files, 206 species).

| Slug | Encoder | Experiment | Training Data | Classes | Test AUROC | LB | Checkpoint |
|------|---------|------------|--------------|---------|-----------|-----|------------|
| `protoclr-2025` | ProtoCLR/CvT-13 | 009 | 2025 focal 80/10/10 | 206 | 0.942 | 0.531 | `009-protoclr-finetune/.../best-epochepoch=20.ckpt` |
| `convnext-2025` | ConvNeXt-Base | 007 | 2025 focal 80/10/10 | 206 | 0.941 | — | `007-convnext-finetune/.../best-epochepoch=12.ckpt` |
| `eat-2025` | EAT-base | 006 | 2025 focal 80/10/10 | 206 | 0.945 | — | `006-lora-finetune/.../best-epochepoch=16.ckpt` |

## BirdCLEF 2026 Models — Focal Only (exp 011)

Trained on BirdCLEF 2026 focal recordings (35,549 files, 206 species).

| Slug | Encoder | Variant | Training Data | Classes | Test AUROC | LB | Checkpoint |
|------|---------|---------|--------------|---------|-----------|-----|------------|
| `protoclr-2026` | ProtoCLR/CvT-13 | 80/10/10 | 2026 focal | 206 | 0.937 | 0.721 | `011-.../2026-protoclr-lora-attn/best-epochepoch=21.ckpt` |
| `convnext-2026` | ConvNeXt-Base | 80/10/10 | 2026 focal | 206 | 0.938 | 0.720 | `011-.../2026-convnext-lora-mlp-mlphead/best-epochepoch=09.ckpt` |
| `eat-2026` | EAT-base | 80/10/10 | 2026 focal | 206 | 0.929 | 0.701 | `011-.../2026-eat-lora-attn-mlphead/best-epochepoch=17.ckpt` |
| `protoclr-2026-fulldata` | ProtoCLR/CvT-13 | 100% focal | 2026 focal (all) | 206 | — | 0.736 | `011-.../2026-protoclr-lora-attn-fulldata/last-v1.ckpt` |
| `convnext-2026-fulldata` | ConvNeXt-Base | 100% focal | 2026 focal (all) | 206 | — | 0.695 | `011-.../2026-convnext-lora-mlp-mlphead-fulldata/last.ckpt` |
| `eat-2026-fulldata` | EAT-base | 100% focal | 2026 focal (all) | 206 | — | 0.698 | `011-.../2026-eat-lora-attn-mlphead-fulldata/last.ckpt` |

## BirdCLEF 2026 Models — Soundscape Fine-Tuned (exp 012)

Fine-tuned from exp 011 (80/10/10) checkpoints on focal + 739 labeled soundscape segments.
Head expanded from 206 → 234 species.

| Slug | Encoder | Training Data | Classes | Test AUROC | LB | Checkpoint |
|------|---------|--------------|---------|-----------|-----|------------|
| `protoclr-2026-soundscape` | ProtoCLR/CvT-13 | 2026 focal + soundscape | 234 | 0.845 | **0.767** | `012-.../2026-protoclr-soundscape/best-epochepoch=11.ckpt` |
| `convnext-2026-soundscape` | ConvNeXt-Base | 2026 focal + soundscape | 234 | 0.844 | **0.755** | `012-.../2026-convnext-soundscape/best-epochepoch=02.ckpt` |
| `eat-2026-soundscape` | EAT-base | 2026 focal + soundscape | 234 | 0.838 | **0.720** | `012-.../2026-eat-soundscape/best-epochepoch=03.ckpt` |

### ONNX Exports

All exp 012 models are exported to ONNX via exp 010 pipeline.

| Slug | ONNX Path | Mel Type | fmax |
|------|-----------|----------|------|
| `protoclr-2026-soundscape` | `010-.../artifacts/weights/protoclr-2026-soundscape/model.onnx` | protoclr_mel | 8,000 Hz |
| `convnext-2026-soundscape` | `010-.../artifacts/weights/convnext-2026-soundscape/model.onnx` | mel | 16,000 Hz |
| `eat-2026-soundscape` | `010-.../artifacts/weights/eat-2026-soundscape/model.onnx` | kaldi_fbank | 8,000 Hz |

### Known Limitations (from error analysis)

| Model | Insect AUROC | Issue |
|-------|-------------|-------|
| `eat-2026-soundscape` | **0.673** | kaldi fbank fmax=8kHz misses insect stridulation |
| `protoclr-2026-soundscape` | 0.856 | fmax=8kHz but handles insects better |
| `convnext-2026-soundscape` | 0.848 | fmax=16kHz, best spectral coverage |

EAT excluded from pseudo-labeling teacher ensemble (exp 014) due to insect failure.

## Pseudo-Label Runs (exp 014)

| Run Name | Teachers | Method | Split | Files | Segments | Path |
|----------|---------|--------|-------|-------|----------|------|
| `pcr-cnx-soundscape-v1` | protoclr + convnext (exp 012) | Direct average | labeled | 66 | 792 | `shared/pseudo-labels/birdclef-2026/pcr-cnx-soundscape-v1/labeled/` |
| `pcr-cnx-soundscape-v1` | protoclr + convnext (exp 012) | Direct average | unlabeled | 10,592 | 127,104 | `shared/pseudo-labels/birdclef-2026/pcr-cnx-soundscape-v1/unlabeled/` |

### Labeled Split Validation (ensemble vs ground truth, 75/234 species)

| Model | Macro AUROC | cmAP |
|-------|-----------|------|
| **Ensemble (mean)** | **0.8851** | **0.3876** |
| protoclr-2026-soundscape | 0.8543 | 0.3583 |
| convnext-2026-soundscape | 0.8224 | 0.3085 |

Per-taxon ensemble AUROC: Insecta 0.904, Aves 0.888, Mammalia 0.864, Amphibia 0.862, Reptilia 0.794.

## BirdCLEF 2026 Models — Student Training (exp 015, round 0)

Fine-tuned from exp 012 checkpoints on focal + labeled soundscape + 127K pseudo-labeled
soundscape segments (soft labels from exp 014 `pcr-cnx-soundscape-v1`). Per-sample weighted
BCE loss: w=1.0 for ground truth, w=0.5 for pseudo-labels. Temperature T=1.0 (no scaling).

| Slug | Encoder | Training Data | Classes | Test AUROC | Test cmAP | Best Epoch | LB | Checkpoint |
|------|---------|--------------|---------|-----------|----------|------------|-----|------------|
| `protoclr-2026-student-v1` | ProtoCLR/CvT-13 | focal + soundscape + pseudo (340K) | 234 | 0.846 | 0.485 | 5 | **0.762** | `015-.../2026-protoclr-student-v1/best-epochepoch=05.ckpt` |
| `convnext-2026-student-v1` | ConvNeXt-Base | focal + soundscape + pseudo (340K) | 234 | 0.845 | 0.479 | 1 | pending | `015-.../2026-convnext-student-v1/best-epochepoch=01.ckpt` |
| `eat-2026-student-v1` | EAT-base | focal + soundscape + pseudo (340K) | 234 | 0.842 | 0.465 | 2 | **0.696** | `015-.../2026-eat-student-v1/best-epochepoch=02.ckpt` |

### Comparison vs Exp 012 Baseline

| Model | Exp 012 Test AUROC | Exp 015 Test AUROC | Delta | Exp 012 LB | Exp 015 LB | LB Delta |
|-------|-------------------|-------------------|-------|-----------|-----------|----------|
| ProtoCLR | 0.845 | 0.846 | +0.001 | 0.767 | 0.762 | **−0.005** |
| ConvNeXt | 0.844 | 0.845 | +0.001 | 0.755 | pending | — |
| EAT | 0.838 | 0.842 | +0.004 | 0.720 | 0.696 | **−0.024** |

ConvNeXt student completed: test AUROC 0.845 (+0.001), LB pending.

**Conclusion (round 0)**: Pseudo-labels at w=0.5, T=1.0 **hurt** LB performance.
Both ProtoCLR (−0.005) and EAT (−0.024) regressed despite flat/improved offline
test metrics. EAT's larger drop may reflect the pseudo-label quality mismatch —
EAT was excluded from the teacher ensemble (exp 014) due to poor insect AUROC,
yet the pseudo-labels encode the ProtoCLR+ConvNeXt teachers' behavior which may
conflict with EAT's learned representations. The clip-level architecture is the
primary bottleneck — exp 016 (SED) addresses this directly.

## BirdCLEF 2026 Models — SED Architecture (exp 016)

EfficientNet-B0 with SED attention head (GeM freq pooling + attention + framewise max).
Trained from ImageNet-pretrained backbone (no prior BirdCLEF checkpoint). 10s windows,
spectrogram mixup, 32kHz. v1: soft CE dual loss, standard ImageNet init.
v2: standard CE with label_smoothing=0.05, NS-JFT pretrained (`tf_efficientnet_b0.ns_jft_in1k`).
**v2 regressed** on all offline metrics vs v1 (−0.023 AUROC, −0.054 cmAP).

| Slug | Encoder | Architecture | Classes | Test AUROC | Test cmAP | Best Epoch | LB | Checkpoint |
|------|---------|-------------|---------|-----------|----------|------------|-----|------------|
| `effnet-b0-sed-v1` | EfficientNet-B0 | SED (GeM + AttHead) | 234 | **0.860** | **0.593** | 24 | 0.795 / 0.823 (+pp) / 0.826 (+pp +10s) / **0.827** (+pp2 +10s) | `016-.../2026-effnet-b0-sed-v1/best-epochepoch=24.ckpt` |
| `effnet-b0-sed-v2` | EfficientNet-B0 (NS-JFT) | SED (GeM + AttHead) + CE | 234 | 0.837 | 0.539 | 24 | **0.755** (+pp) | `016-.../2026-effnet-b0-sed-v2/best-epochepoch=24.ckpt` |
| `effnet-b0-sed-v3a` | EfficientNet-B0 (TF) | SED (TF-ported weights) | 234 | 0.857 | 0.599 | 25 | **0.813** (+pp) | `016-.../2026-effnet-b0-sed-v3a/best-epochepoch=25.ckpt` |
| `effnet-b0-sed-v3b` | EfficientNet-B0 | SED + raw waveform mixup | 234 | **0.862** | **0.607** | 25 | **0.802** (+pp) | `016-.../2026-effnet-b0-sed-v3b/best-epochepoch=25.ckpt` |
| `effnet-b0-sed-v3c` | EfficientNet-B0 | SED + CE loss | 234 | 0.850 | 0.552 | 25 | pending | `016-.../2026-effnet-b0-sed-v3c/best-epochepoch=25.ckpt` |
| `effnet-b0-sed-v3d` | EfficientNetV2-B0 | SED (V2 architecture) | 234 | 0.841 | 0.523 | 13 | pending | `016-.../2026-effnet-b0-sed-v3d/best-epochepoch=13.ckpt` |

### Comparison: SED vs Clip-Level (offline test set)

| Model | Architecture | Test AUROC | Test cmAP | LB |
|-------|-------------|-----------|----------|-----|
| EfficientNet-B0 SED v3b | SED + raw waveform mixup | **0.862** | **0.607** | **0.802** (+pp) |
| **EfficientNet-B0 SED v1** | **SED (GeM + AttHead)** | **0.860** | **0.593** | 0.795 / 0.823 (+pp) / **0.826** (+pp +10s) |
| EfficientNet-B0 SED v3a | SED (TF-ported weights) | 0.857 | 0.599 | **0.813** (+pp) |
| EfficientNet-B0 SED v3c | SED + CE loss | 0.850 | 0.552 | pending |
| EfficientNetV2-B0 SED v3d | SED (V2 architecture) | 0.841 | 0.523 | pending |
| EfficientNet-B0 SED v2 | SED + NS-JFT + CE | 0.837 | 0.539 | **0.755** (+pp) |
| ProtoCLR (exp 012) | Clip (global avg pool) | 0.845 | 0.481 | 0.767 |
| ConvNeXt (exp 012) | Clip (global avg pool) | 0.844 | 0.470 | 0.755 |
| EAT (exp 012) | Clip (global avg pool) | 0.838 | 0.452 | 0.720 |

SED architecture achieves +0.015 AUROC and +0.112 cmAP over the best clip-level model,
using a smaller backbone (3.6M params vs 19-90M for our LoRA encoders). The cmAP
improvement is particularly notable — SED's framewise detection captures brief calls
that clip-level averaging misses.

**Post-processing**: TTA (2.5s shifts) + temporal smoothing (Gaussian) + TopN (file-level
max multiplication) added +0.028 LB (0.795 → 0.823) with zero retraining.

**Competition reference**: EfficientNet-B0 SED = 0.908 LB (competitor). Remaining gap
0.082 — likely from: mel resolution (n_fft=1024 vs 4096, 128 vs 224 mels),
standard ImageNet init, spectrogram mixup (vs raw waveform), and head design.
10s inference alignment added +0.003, pp2 (temp scaling + taxa smoothing + metadata priors) added +0.001. See exp 016 summary.md for gap analysis;
next steps in [concept: strong-baseline-mel-overhaul](docs/concepts/strong-baseline-mel-overhaul.md).

### Training Dynamics

Strong learning curve from random init:
- Epoch 0: val/auroc 0.517 (random)
- Epoch 8: val/auroc 0.833 (rapid improvement)
- Epoch 18: val/auroc 0.844 (plateauing)
- Epoch 24: val/auroc 0.845 (best, early stopped at epoch 34)

## BirdCLEF 2026 Models — Mel Overhaul (exp 017)

EfficientNet-B0 SED with high-resolution mel: n_fft=4096, n_mels=224 (vs v1's 1024/128).
All other parameters identical to exp 016 v1 to isolate the mel resolution variable.
Mel computed on GPU (n_fft=4096 too slow on CPU DataLoader workers).

| Slug | Encoder | Architecture | Mel | Classes | Test AUROC | Test cmAP | Best Epoch | LB | Checkpoint |
|------|---------|-------------|-----|---------|-----------|----------|------------|-----|------------|
| `effnet-b0-sed-v4a` | EfficientNet-B0 | SED (GeM + AttHead) | n_fft=4096, n_mels=224 | 234 | **0.862** | **0.615** | 24 | 0.825 / **0.830** (+pp2) | `017-.../2026-effnet-b0-sed-v4a/best-epochepoch=24.ckpt` |

### Comparison: v4a vs v1

| Metric | v1 (exp 016) | v4a (exp 017) | Delta |
|--------|-------------|---------------|-------|
| test/auroc | 0.860 | **0.862** | +0.002 |
| test/cmap | 0.593 | **0.615** | **+0.022** |
| test/top1_acc | 0.762 | 0.772 | +0.010 |
| test/top5_acc | 0.887 | 0.894 | +0.007 |
| LB (raw) | 0.795 | **0.825** | **+0.030** |
| LB (+pp2) | 0.827 | **0.830** | **+0.003** |

**Mel resolution is a major lever.** v4a raw (0.825) nearly matches v1+pp2 (0.827) with
zero post-processing. PP2 adds only +0.005 on v4a (vs +0.032 on v1) — diminishing returns
as the model already captures distinctions that post-processing was compensating for.
Decision gate passed → continue to Step 2.

Inference optimized with full-signal mel (compute STFT once per 60s file, slice frames)
to stay within the 90-minute Kaggle CPU budget at n_fft=4096.

### Step 2: Parallel Ablations (v4b-v4e)

Single-variable ablations from v4a. 90/10 split (no test set), 50 epochs.
No test metrics available — only val/auroc from checkpoint callback.

| Slug | Change | Mel Norm | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4b` | AmplitudeToDB norm | amplitude_to_db | AttHead | soft_ce | **0.865** | 30 | **0.859** / **0.860** (+pp2) | `017-.../2026-effnet-b0-sed-v4b/best-epochepoch=30.ckpt` |
| `effnet-b0-sed-v4c` | AttBlockV2 head | audioset | AttBlockV2 | soft_ce | 0.815 | 15 | skipped | `017-.../2026-effnet-b0-sed-v4c/best-epochepoch=15.ckpt` |
| `effnet-b0-sed-v4d` | ASL loss | audioset | AttHead | asl | 0.858 | 16 | 0.808 | `017-.../2026-effnet-b0-sed-v4d/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v4e` | StochasticDepth | audioset | AttHead | soft_ce | **0.868** | 29 | 0.840 | `017-.../2026-effnet-b0-sed-v4e/best-epochepoch=29.ckpt` |

### Step 3: Combined winners + re-ablations

| Slug | Change | Mel Norm | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4f` | v4b + StochasticDepth | amplitude_to_db | AttHead | soft_ce | 0.862 | 17 | 0.832 | `017-.../2026-effnet-b0-sed-v4f/best-epochepoch=17.ckpt` |
| `effnet-b0-sed-v4g` | v4b + AttBlockV2 + BCE | amplitude_to_db | AttBlockV2 | bce | **0.868** | 28 | 0.819 | `017-.../2026-effnet-b0-sed-v4g/best-epochepoch=28.ckpt` |
| `effnet-b0-sed-v4h` | v4b + AttBlockV2 + ASL | amplitude_to_db | AttBlockV2 | asl | 0.858 | 12 | 0.800 | `017-.../2026-effnet-b0-sed-v4h/best-epochepoch=12.ckpt` |

### Step 4: Fulldata Training

| Slug | Change | Mel Norm | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4b-fulldata` | v4b on 100% data | amplitude_to_db | AttHead | soft_ce | — (no val) | 44 (last) | 0.853 | `017-.../2026-effnet-b0-sed-v4b-fulldata/last.ckpt` |

**v4b (AmplitudeToDB)** is the current best at 0.859 LB (0.860 with pp2).
Fulldata training regressed −0.006 (0.853 vs 0.859) — likely overtrained without val monitoring.

### Step 5: Loss & Augmentation Ablations

Motivated by Salman Ahmed's 0.908 LB (single EfficientNet-B0 SED with CE + raw waveform MixUp).

| Slug | Change | Mel Norm | Head | Loss | Waveform MixUp | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4i` | Standard CE loss (primary_label) | amplitude_to_db | AttHead | ce | No | 0.835 | 12 | skip | `017-.../2026-effnet-b0-sed-v4i/best-epochepoch=12.ckpt` |
| **`effnet-b0-sed-v4j`** | **Raw waveform MixUp** | amplitude_to_db | AttHead | soft_ce | **Yes** | **0.868** | 34 | 0.853 | `017-.../2026-effnet-b0-sed-v4j/best-epochepoch=34.ckpt` |
| `effnet-b0-sed-v4k` | CE + waveform MixUp | amplitude_to_db | AttHead | ce | Yes | 0.835 | 18 | skip | `017-.../2026-effnet-b0-sed-v4k/best-epochepoch=18.ckpt` |

**v4j (waveform MixUp)** had best offline (+0.003 val/auroc) but regressed −0.006 on LB (0.853 vs 0.859).
Offline val/auroc continues to be a poor LB predictor (v4j: 0.868 val → 0.853 LB).
Standard CE loss (v4i, v4k) regressed −0.030 val/auroc even with the primary_label fix
(using actual primary label index instead of argmax of multi-hot). CE loss with our soft CE
setup consistently underperforms — the gap to Salman's 0.908 is from other factors.

**Note**: v4i and v4k used the fixed CE implementation (primary_label_idx from dataloader,
not argmax of multi-hot). The original broken CE (argmax) gave similar results (v4k
first run: 0.846), confirming the low mismatch rate (0.2% for focal recordings).

## BirdCLEF 2026 Models — SED Ablations (exp 019)

Single-variable ablations from v4b (0.860 LB). 90/10 split, 50 epochs, same data pipeline.
Goal: push SED from 0.860 → 0.885+ LB before pseudo-labeling.

### Step 1: Quick wins ablation

| Slug | Change | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v5a` | BCEWithLogitsLoss | AttHead | bce | **0.875** | 29 | **0.851** | `017-.../2026-effnet-b0-sed-v5a/best-epochepoch=29.ckpt` |
| `effnet-b0-sed-v5b` | Stronger MixUp (α=1.0, θ=0.8, warmup=5) | AttHead | soft_ce | **0.869** | 31 | **0.840** | `017-.../2026-effnet-b0-sed-v5b/best-epochepoch=31.ckpt` |
| `effnet-b0-sed-v5c` | weight_decay=1e-4 | AttHead | soft_ce | **0.867** | 35 | pending | `017-.../2026-effnet-b0-sed-v5c/best-epochepoch=35.ckpt` |
| `effnet-b0-sed-v5d` | LSEHead (GeM+LSE pooling) | **LSEHead** | soft_ce | **0.861** | 21 | pending | `017-.../2026-effnet-b0-sed-v5d/best-epochepoch=21.ckpt` |
| `effnet-b0-sed-v5e` | HGNetV2-B0 backbone (embed=1024) | AttHead | soft_ce | 0.861 | 23 | pending | `017-.../2026-hgnet-b0-sed-v5e/best-epochepoch=23.ckpt` |

**v5d (LSEHead)**: −0.004 val/auroc vs v4b baseline (0.861 vs 0.865). LSE pooling slightly
regresses compared to attention head on EfficientNet-B0.

### Step 2: ttahara full recipe reproduction

Faithful reproduction of ttahara's 0.884 LB pipeline: LSEHead(mean freq), BCE, per-sample
MixUp(α=1.0, θ=0.8), OneCycleLR(lr=5e-4, wd=1e-4), torchaudio mel(n_fft=2048, resize
256x256, min-max), 5s segments, 20 epochs. Ref: `user/murilo/notes/ttahara-hgnetv2-b0-annotated.md`.

| Slug | Backbone | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v5f` | EfficientNet-B0 | LSEHead(mean) | bce | **0.866** | 14 | **0.855** | `017-.../2026-effnet-b0-sed-v5f/best-epochepoch=14.ckpt` |
| `hgnet-b0-sed-v5g` | HGNetV2-B0 | LSEHead(mean) | bce | **0.872** | 18 | **0.892** | `017-.../2026-hgnet-b0-sed-v5g/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v5g-4fold` | HGNetV2-B0 ×4 | LSEHead(mean) | bce | 0.939/0.842/0.826/0.802 | 14/19/18/16 | **0.901** | `017-.../2026-hgnet-b0-sed-v5g/fold{0-3}/` |

### Step 4: v5g Recipe Ablations

Single-variable ablations from v5g (0.892 LB). All use HGNetV2-B0 + ttahara recipe, 20 epochs.

| Slug | Change from v5g | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|----------------|------|------|---------------|------------|-----|------------|
| `hgnet-b0-sed-v5h` | n_fft 2048→4096 | LSEHead(mean) | bce | 0.869 | 16 | **0.893** | `017-.../2026-hgnet-b0-sed-v5h/best-epochepoch=16.ckpt` |
| `hgnet-b0-sed-v5i` | window 5s→10s | LSEHead(mean) | bce | **0.878** | 17 | **0.880** | `017-.../2026-hgnet-b0-sed-v5i/best-epochepoch=17.ckpt` |
| **`hgnet-b0-sed-v5j`** | **AmplitudeToDB norm†** | LSEHead(mean) | bce | 0.874 | 19 | **0.906** | `017-.../2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt` |
| `hgnet-b0-sed-v5k` | 3-channel mel (in_chans=3) | LSEHead(mean) | bce | 0.868 | 16 | **0.886** | `017-.../2026-hgnet-b0-sed-v5k/best-epochepoch=16.ckpt` |

| `hgnet-b0-sed-v5l` | v5j + n_fft=4096 | LSEHead(mean) | bce | 0.874 | 16 | **0.904** | `017-.../2026-hgnet-b0-sed-v5l/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v5m` | EffNet-B0 + proj 320→1024‡ | LSEHead(mean) | bce | **0.873** | 16 | **0.898** | `017-.../2026-effnet-b0-sed-v5m/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v5n` | EffNet-B0, no proj (v5j recipe)‡‡ | LSEHead(mean) | bce | 0.871 | 19 | **0.875** | `017-.../2026-effnet-b0-sed-v5n/best-epochepoch=19.ckpt` |
| **`hgnet-b0-sed-v5j-4fold`** | **AmplitudeToDB norm† ×4** | LSEHead(mean) | bce | 0.951/0.843/0.823/0.808 | 15/19/19/18 | **0.913** | `017-.../2026-hgnet-b0-sed-v5j/fold{0-3}/` |

† v5j confounded: also changes win_length (626→2048), removes resize (256×256→256×511), uses NNAudio.
‡ v5m uses EfficientNet-B0 (320 native embed_dim) + 1x1 Conv2d projection to 1024. Tests whether HGNet's advantage is from embed_dim. All other params match v5j.
‡‡ v5n uses EfficientNet-B0 with v5j recipe (NNAudio+AmplitudeToDB) but NO projection (native 320 embed_dim). Control for v5m.

**Note on inference post-processing**: All LB scores from v5g onward include
inference-time TTA (±2.5s shifts, 50/25/25 weighting), temporal smoothing
(Gaussian), and TopN filtering, enabled globally in `submission.py`
(`ENABLE_TTA=True`, `ENABLE_TEMPORAL_SMOOTHING=True`, `ENABLE_TOPN=True`).
These are baked into every score — there is no "raw" v5j LB without them.

## BirdCLEF 2026 — Perch Probes (exp 018)

Frozen Perch v2 as feature extractor + probe classifiers. No GPU training —
probes train in ~2s on CPU. All inference on Kaggle CPU kernel (~85 min for ~600 test files).

| Slug | Pipeline | Probe Features | OOF | LB | Delta | Kaggle Kernel |
|------|----------|---------------|-----|-----|-------|---------------|
| `perch-raw` | Perch logits only | — | No | 0.825 | baseline | `submission_perch_raw.py` |
| `perch-cache-v1` | + genus proxy mapping | — | No | 0.836 | +0.011 | `submission_perch_cache.py` |
| `perch-probes-v1` | + LogReg probes, logit-space | 39-dim (PCA32 + 7) | No | 0.893 | +0.057 | `submission_perch_probes.py` v1 |
| `perch-probes-v2` | + OOF fold-safe priors | 39-dim (PCA32 + 7) | 5-fold | 0.898 | +0.005 | `submission_perch_probes.py` v2 (buggy) |
| **`perch-probes-v2-fix`** | **+ 6-cat fuse, sc_clean priors** | **39-dim (PCA32 + 7)** | **5-fold** | **0.911** | **+0.013** | **`submission_perch_probes.py` v2-fix** |
| `perch-protossm-v4` | + BiSSM×2 + cross-attn + TTA + ResidualSSM + 6-stage PP | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.918** | +0.007 | `submission_perch_protossm.py` v1 |
| `perch-protossm-v4.1` | v4 + OOF prior by filename, per-stage clipping, all-data train | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.918** | +0.000 | `submission_perch_protossm.py` v4 |
| `perch-protossm-v4.2` | v4.1 + taxonomy 5-way, fixed ensemble 0.5, MLP seed, 1-based sites | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.899** | **−0.019** | `submission_perch_protossm.py` v5 |

### Perch Probe Progression Analysis

| Transition | Key Change | LB Delta | Why it helped |
|-----------|-----------|----------|---------------|
| raw → cache | Genus proxy mapping for unmapped species | +0.011 | 31 unmapped species get non-zero predictions |
| cache → probes-v1 | LogReg probes on PCA-compressed embeddings | +0.057 | Learned species-specific decision boundaries |
| v1 → v2-fix | OOF fold-safe priors, 6-category fusion, sc_clean | +0.013 | Proper prior estimation, bug fixes (K_HOUR, labeled subset) |
| v2-fix → v4 | ProtoSSM temporal modeling + ResidualSSM + TTA | +0.007 | Captures temporal patterns across 12 windows |
| v4 → v4.2 | 4 simultaneous changes | **−0.019** | Batched changes; cannot isolate which one broke it |

## BirdCLEF 2026 — Perch ProtoSSM v5 (exp 020)

Faithful port of pantanal-distill V17 (0.930 LB reference). V17 config: d_model=256,
3 BiSSM layers, cross-attention, focal BCE + distill + mixup, 40 epochs. Uses
jaejohn/perch-meta cache for labeled embeddings. All training on Kaggle CPU kernel.

| Slug | Pipeline | Config | LB | Kaggle Kernel |
|------|----------|--------|-----|---------------|
| `perch-protossm-v5-v2` | pantanal port (V18 config, no cache) | d=320, 4L, 80ep | timeout | `020-.../submission_perch_protossm.py` v2 |
| `perch-protossm-v5-v3` | pantanal port (V18 config, cache) | d=320, 4L, 80ep | timeout | `020-.../submission_perch_protossm.py` v3 |
| `perch-protossm-v5-v4` | pantanal port (V17 config, cache) | d=256, 3L, 40ep | **0.922** | `020-.../submission_perch_protossm.py` v4 |
| `perch-protossm-v5-v5` | V17+ONNX, MLP-only (w=0.0) | d=256, 3L, 40ep | 0.911 | `020-.../submission_perch_protossm.py` v5 |
| `perch-protossm-v5-v7` | V18+ONNX | d=320, 4L, 80ep | 0.920 | `020-.../submission_perch_protossm.py` v7 |
| **`perch-protossm-v18`** | **two-kernel V18+TF** | **d=320, 4L, 80ep, TTA×5** | **0.925** | **`020-.../submission_perch_protossm.py` (infer_only)** |

V18 pipeline (two-kernel split): Training kernel saves pretrained artifacts to Kaggle dataset.
Inference kernel loads artifacts AFTER Perch TF inference (memory optimization: loading 75MB
before TF caused 2x slowdown). Pipeline: Perch v2 TF → MLP(256,128) probes (PCA-128) →
ProtoSSM v5 (d=320, 4 BiSSM, cross-attn, 80ep, 5-fold, focal γ=2.5, cosine restart, SWA) →
ResidualSSM (d=128) → TTA [0,±1,±2] → ensemble (0.5 proto + 0.5 mlp) →
6-stage PP (temp, confidence, rank-aware, delta-shift, V18 per-class thresholds).

Gap vs pantanal reference (0.930): −0.005. V18+TF gained +0.003 over V17+TF (0.922).

### v2-fix Details (0.911 LB)

Pipeline: Perch v2 frozen → StandardScaler + PCA(32) → 5-fold OOF GroupKFold by site →
52 LogReg probes (39-dim, C=0.25, class_weight='balanced') → Gaussian smoothing → sigmoid.

Key fixes over buggy v2 (0.898):
- K_HOUR = 8 (was 4)
- Prior tables from ALL 739 soundscape annotations (was labeled subset only)
- 6-category fuse_scores: proxy λ=0.8, prior-only REPLACE, inactive = -8.0
- Filtered to 59 fully-labeled files / 708 windows (was 66 files / 792 windows)

## BirdCLEF 2026 — SED Training Improvements (exp 021)

Post-processing and training recipe improvements targeting 0.920+ single-model LB.
Motivated by competitor insights from Kaggle discussion #683791.

**All exp 021 LB scores include TTA + temporal smoothing + TopN** (inherited from
`submission.py` global flags). See note under exp 019 Step 4.

### Phase A: Site/Hour Priors (no retraining)

Multiplicative site×hour×species priors from 66 labeled soundscapes (1478 segments,
9 sites, 13 hours). Taxa-aware lambda: Aves=0.3, Amphibia/Insecta=0.5, Mammalia/Reptilia=0.2.

| Slug | Base Model | Priors | LB | Delta | Notes |
|------|-----------|--------|-----|-------|-------|
| `hgnet-b0-sed-v5j` | HGNetV2-B0 single | No | **0.906** | — | Baseline (exp 019) |
| **`hgnet-b0-sed-v5j-pp2`** | HGNetV2-B0 single | **Yes** | **0.909** | **+0.003** | Site/hour priors, 11 min runtime |
| `hgnet-b0-sed-v5j-4fold` | HGNetV2-B0 ×4 | No | **0.913** | — | Baseline (exp 019) |
| **`hgnet-b0-sed-v5j-4fold-pp2`** | **HGNetV2-B0 ×4** | **Yes** | **0.917** | **+0.004** | **New best SED** |

Priors gave +0.003 single / +0.004 4-fold (not the +0.006 Salman reported). Possible
reasons: different prior implementation (multiplicative vs logit-space), different lambda
tuning, or model already captures site/time patterns that priors are redundant with.

### Phase B: Training Recipe Ablations

Single-variable changes from v5j (HGNetV2-B0, val/auroc 0.874, LB 0.906).
All trained with seed=42, deterministic=True, on PACE GPU.

| Slug | Change from v5j | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------------|-----------|----------|------------|-----|------------|
| `hgnet-b0-sed-v6a` | CE loss (primary_label) | 0.854 | 0.580 | 18 | pending | `021-.../2026-hgnet-b0-sed-v6a/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v6b` | Waveform MixUp | 0.871 | 0.625 | 16 | **0.887** | `021-.../2026-hgnet-b0-sed-v6b/best-epochepoch=16.ckpt` |
| `hgnet-b0-sed-v6c` | Waveform MixUp + noise + stretch | 0.861 | 0.608 | 16 | pending | `021-.../2026-hgnet-b0-sed-v6c/best-epochepoch=16.ckpt` |
| `hgnet-b0-sed-v6d` | 25 epochs (vs 20) | **0.873** | **0.651** | 22 | pending | `021-.../2026-hgnet-b0-sed-v6d/best-epochepoch=22.ckpt` |

CE loss (v6a) regressed −0.020 — single-label assumption hurts on multi-label soundscapes.
Waveform MixUp (v6b) confirmed hurts on LB: 0.887 (−0.019 vs v5j's 0.906). Spectrogram MixUp remains better.
25 epochs (v6d) essentially flat on AUROC but improved cmap (+0.026).

### Phase C: AttHead + Backbone Scaling

All use **AttHead with dual supervision** (clipwise_weight=0.5, framewise_weight=0.5,
blend 75% clip + 25% frame max). Replaces LSEHead from v5j. 20 epochs, seed=42.

| Slug | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|------|-----------|----------|------------|-----|------------|
| `hgnet-b2-sed-v6h` | HGNetV2-B2 | 12M | AttHead | BCE | **0.876** | **0.658** | 18 | **0.889** | `021-.../2026-hgnet-b2-sed-v6h/best-epochepoch=18.ckpt` |
| `convnextv2-nano-sed-v6i` | ConvNeXt-V2 Nano | 15M | AttHead | BCE | 0.874 | **0.663** | 16 | — | `021-.../2026-convnextv2-nano-sed-v6i/best-epochepoch=16.ckpt` |
| **`effnet-b0-ns-sed-v6j`** | **EffNet-B0 NS** | **5.3M** | AttHead | BCE | 0.874 | 0.626 | 15 | **0.893** | `021-.../2026-effnet-b0-ns-sed-v6j/best-epochepoch=15.ckpt` |
| `hgnet-b0-sed-v6e` | HGNetV2-B0 | 6M | AttHead | BCE | 0.870 | 0.611 | 18 | skip | `021-.../2026-hgnet-b0-sed-v6e/best-epochepoch=18.ckpt` |
| `effnet-b3-ns-sed-v6g` | EffNet-B3 NS | 12M | AttHead | CE | 0.859 | 0.611 | 19 | skip | `021-.../2026-effnet-b3-ns-sed-v6g/best-epochepoch=19.ckpt` |
| `hgnet-b0-sed-v6f` | HGNetV2-B0 | 6M | AttHead | CE | 0.850 | 0.577 | 18 | skip | `021-.../2026-hgnet-b0-sed-v6f/best-epochepoch=18.ckpt` |

### Phase F: Competition-Standard AttHead Fix (v6k)

Fixed AttHead to match Nikita 1st place 2025 / Tattaka exactly: pre-dropout(p/2),
softmax(tanh(x)) attention, probability-space clipwise. Dual supervision (50/50).
hidden_dim=1024 (matching v5j). Everything else identical to v5j.

| Slug | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|------|-----------|----------|------------|-----|------------|
| `hgnet-b0-atthead-v6k` | HGNetV2-B0 | 6M | AttHead (fixed) | BCE | 0.871 | 0.596 | 19 | **0.888** | `021-.../2026-hgnet-b0-atthead-v6k/best-epochepoch=19.ckpt` |

**Fixed AttHead regresses** −0.018 on LB (0.888 vs 0.906) and −0.003 offline.
Barely differs from old buggy AttHead v6e (0.871 vs 0.870). The three fixes
(pre-dropout, tanh, probability-space) are not the differentiator. AttHead
consistently underperforms LSEHead on HGNetV2-B0 on both offline and LB.

**Both Phase C AttHead models regressed on LB** vs v5j (LSEHead, 0.906): v6j 0.893
(−0.013), v6h 0.889 (−0.017). Offline metrics are anti-correlated with LB.

### Phase D: Multi-Context Training (AttHeadMultiContext)

Train on 15-20s windows with center 5s inference crop. `AttHeadMultiContext` head with
competition-standard attention (softmax(tanh(att)), probability-space clipwise).

| Slug | Backbone | Params | Head | Context | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|---------|-----------|----------|------------|-----|------------|
| `effnet-b0-ns-mc-sed-v7c` | EffNet-B0 NS | 5.3M | AttHeadMC | 3×5s (15s) | 0.851 | 0.605 | 19 | — | `021-.../2026-effnet-b0-ns-mc-sed-v7c/best-epochepoch=19.ckpt` |
| `hgnet-b0-mc-sed-v7a` | HGNetV2-B0 | 6M | AttHeadMC | 3×5s (15s) | 0.820 | 0.579 | 16 | — | `021-.../2026-hgnet-b0-mc-sed-v7a/best-epochepoch=16.ckpt` |
| `hgnet-b0-mc-sed-v7b` | HGNetV2-B0 | 6M | AttHeadMC | 4×5s (20s) | 0.799 | 0.542 | 16 | — | `021-.../2026-hgnet-b0-mc-sed-v7b/best-epochepoch=16.ckpt` |

**All v7 models massively regressed** vs v5j (0.874): v7c −0.023, v7a −0.054, v7b −0.075.
Longer context = worse. Changed too many variables from v5j (head type + window length +
batch size). Cannot isolate multi-context effect. hengck23 uses LSEHead with multi-context,
not AttHead — the head change likely caused most of the regression, not the context itself.

### Phase E: LSEHead + Temporal Split (correct multi-context)

Keep LSEHead exactly as v5j, add `n_chunks` to split temporal features into N chunks
before LSE pooling. Backbone sees full 15-20s context, each chunk pools at 5s resolution.
Inspired by hengck23 (0.896 LB, CNN-only, 3×5s). See `multi-context-data-flow.md`.

| Slug | Backbone | Head | Context | n_chunks | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|------|---------|----------|-----------|----------|------------|-----|------------|
| `hgnet-b0-mc-lse-sed-v8a` | HGNetV2-B0 | LSEHead | 3×5s (15s) | 3 | **0.882** | **0.719** | 13 | 0.882 (avg) / 0.883 (center) | `021-.../2026-hgnet-b0-mc-lse-sed-v8a/best-epochepoch=13.ckpt` |
| **`hgnet-b0-mc-lse-sed-v8b`** | **HGNetV2-B0** | **LSEHead** | **4×5s (20s)** | **4** | **0.881** | **0.727** | 9 | **0.896 (avg)** | `021-.../2026-hgnet-b0-mc-lse-sed-v8b/best-epochepoch=09.ckpt` |

Best offline metrics ever (+0.007-0.008 AUROC, +0.094-0.102 cmap vs v5j). LB results:
v8a 0.882 (avg) / 0.883 (center-chunk, +0.001 negligible), v8b **0.896** (avg). Center-chunk
inference didn't help — the offline-to-LB gap is not an inference strategy issue. v8b's
20s context outperforms v8a's 15s on LB (0.896 vs 0.883), suggesting longer backbone
context is valuable but the approach still regresses vs v5j's 0.906 single-model LB.

### Phase G: Manifold Mixup (v6l)

Manifold mixup inspired by Mix2 (Moummad et al. 2024). Each batch randomly alternates
between input-space spectrogram mixup (50%) and embedding-space manifold mixup (50%).
Single-variable change from v5j — same backbone, head, loss, schedule.

| Slug | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|------|-----------|----------|------------|-----|------------|
| `hgnet-b0-sed-v6l` | HGNetV2-B0 | 6M | LSEHead | BCE | 0.874 | 0.630 | 15 | **0.899** | `021-.../2026-hgnet-b0-sed-v6l/best-epochepoch=15.ckpt` |

**Manifold mixup regresses −0.007 on LB** (0.899 vs 0.906) despite matching v5j offline
(0.874). Embedding-space mixing disrupts soundscape generalization. Another training-time
change that fails to improve on v5j.

## BirdCLEF 2026 Models — PCEN Normalization (exp 022)

PCEN (Per-Channel Energy Normalization) replacing AmplitudeToDB in the mel pipeline.
All configs identical to v5j except normalization. Targets domain gap between focal
recordings and Pantanal soundscapes.

| Slug | Backbone | Params | Head | Normalization | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|---------------|-----------|----------|------------|-----|------------|
| `hgnet-b0-sed-v10a` | HGNetV2-B0 | 6M | LSEHead | Learnable PCEN | 0.871 | 0.620 | 18 | **0.880** | `022-.../2026-hgnet-b0-sed-v10a/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v10b` | HGNetV2-B0 | 6M | LSEHead | Fixed PCEN | 0.870 | 0.610 | 18 | — | `022-.../2026-hgnet-b0-sed-v10b/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v10c` | HGNetV2-B0 | 6M | LSEHead | PCEN + AmplitudeToDB | 0.870 | 0.602 | 17 | — | `022-.../2026-hgnet-b0-sed-v10c/best-epochepoch=17.ckpt` |
| `hgnet-b0-sed-v5j-repro` | HGNetV2-B0 | 3.1M | LSEHead | AmplitudeToDB | 0.873 | — | 18 | **0.903** | `022-.../2026-hgnet-b0-sed-v5j-repro/best-epochepoch=18.ckpt` |

**v10a regressed −0.026 on LB** (0.880 vs 0.906). PCEN normalization hurts soundscape
generalization. v10b/v10c not submitted (worse offline than v10a). **v5j-repro scored
0.903 LB** (−0.003 vs original v5j 0.906), confirming pipeline is correct and PCEN
is the cause of regression. **Experiment rejected.**

## BirdCLEF 2026 — Perch + SED Ensemble (exp 023)

Score-level fusion of Perch ProtoSSM v18 (ONNX) and SED v5j-4fold-pp2 in a single
Kaggle CPU kernel. Each pipeline runs its own full post-processing, then weighted
average produces the final submission. Estimated runtime ~64 min for ~600 test files.

| Slug | W_PERCH | Perch Backend | SED Model | LB | Kaggle Kernel |
|------|---------|--------------|-----------|-----|---------------|
| `perch-sed-ensemble-w05` | 0.50 | ONNX | v5j-4fold-pp2 | 0.927 | `023-.../submission_ensemble.py` |
| `perch-sed-ensemble-w03` | 0.30 | ONNX | v5j-4fold-pp2 | 0.927 | `023-.../submission_ensemble.py` |
| `perch-sed-ensemble-w07` | 0.70 | ONNX | v5j-4fold-pp2 | 0.926 | `023-.../submission_ensemble.py` |
| **`perch-sed-ensemble-mlp-only`** | **0.50** | **ONNX** | **v5j-4fold-pp2** | **0.929** | **`023-.../submission_ensemble.py` (proto_weight=0.0)** |
| `perch-sed-ensemble-no-pp` | 0.50 | ONNX | v5j-4fold-pp2 | 0.922 | `023-.../submission_ensemble.py` (PP disabled) |

**0.929 LB is the new best overall** (MLP-only, +0.004 vs Perch TF 0.925).
Key findings:
- Dropping ProtoSSM (MLP-only) gains +0.002 — ProtoSSM trained on TF embeddings hurts
  under ONNX inference mismatch. Matches pantanal-distill 0.930 approach.
- Post-processing is worth +0.005 (0.927 with PP vs 0.922 without).
- Weight curve flat (0.926-0.927) — both models contribute equally.
- Next: TF Perch + single SED (recover 0.005 ONNX penalty, could reach 0.932+).

## Experiment 026: Domain-Shift Training

Dual validation (separate focal/soundscape metrics) + window length + AttHead ablations.
Goal: close the single-model gap (0.903 → 0.922+).

### Phase 2: v5j-dualval baseline

| Slug | Window | Head | val/auroc | val_sc/auroc | LB | Checkpoint |
|------|--------|------|-----------|-------------|-----|------------|
| `v5j-dualval` | 5s | LSEHead | 0.849 | 0.956 | 0.897 | `026-.../2026-exp026-v5j-dualval/best-epochepoch=17.ckpt` |

*val_sc/auroc is broken — macro AUROC over 234 species when only ~35 appear in
86 soundscape val segments. See `026-domain-shift-training/results.md`.

### Phase 3: AttHead + window ablation

All Phase 3 jobs ran with old broken val_sc/auroc (~0.093). Compare using val/auroc.

| Slug | Window | Head | val/auroc | val/cmap | val/top1 | LB | Checkpoint |
|------|--------|------|-----------|----------|----------|-----|------------|
| `v9a-10s-att` | 10s | AttHead | 0.859 | 0.645 | 0.827 | **0.868** | `026-.../last.ckpt` (ep19) |
| `v9b-15s-att` | 15s | AttHead | 0.858 | 0.664 | 0.847 | **0.877** | `026-.../last.ckpt` (ep19) |
| `v9c-20s-att` | 20s | AttHead | 0.861 | 0.681 | 0.862 | **0.878** | `026-.../last.ckpt` (ep19) |

All AttHead models regressed on LB despite best-ever offline metrics. Train/inference
window mismatch (train on 10-20s, infer on 5s) likely causes attention calibration issues.

## Experiment 027: Model Training Fixes

Fix two implementation errors identified by deep research: additive waveform MixUp
(x1+x2 with max labels) and CE loss on primary label (sigmoid at inference).

| Slug | Loss | MixUp | val/auroc | val/cmap | val_sc/auroc | LB | Checkpoint |
|------|------|-------|-----------|----------|-------------|-----|------------|
| `v10a-addmix-ce` | CE | additive wfm | 0.835 | 0.571 | 0.863 | **0.494** | `027-.../last.ckpt` (ep19) |
| `v10b-addmix-bce` | BCE | additive wfm | 0.844 | 0.592 | 0.966 | **0.875** | `027-.../best-epochepoch=17.ckpt` |

**v10a = 0.494 LB (catastrophic).** CE logits have ~1 unit spread centered at zero;
sigmoid maps all 234 species to ~0.50 → random on soundscapes. CE is now confirmed
dead (7 tests, 0 successes). Root cause: CE optimizes relative ranking (softmax),
not absolute magnitudes; logits lack safety margin for focal→soundscape domain shift.
See `027-model-training-fixes/results.md` for full analysis.

**v10b = 0.875 LB (-0.031).** Additive waveform MixUp regresses even with BCE.
Worse than interpolative waveform MixUp (v6b = 0.887). All waveform MixUp variants
regress vs spectrogram MixUp. Offline val_sc/auroc improved +0.010 but LB regressed
-0.031 — offline metrics anti-correlated with LB again.

### I-1: v5j-10s (overlapping window inference, no retraining)

Tested 10s inference window on v5j (2.5s context each side). LB = **0.906** — flat vs
v5j baseline (0.906). Existing TTA (±2.5s shifts) already provides similar overlap.
Inference-only change, no new model.

### Phase 4: Salman-derived structural changes

| Slug | Change | val/auroc | val/cmap | val_sc/auroc | LB | Checkpoint |
|------|--------|-----------|----------|-------------|-----|------------|
| `v5j-control` | v5j baseline (W&B control) | 0.849 | 0.596 | 0.955 | — | `027-.../best-epochepoch=14.ckpt` |
| `v11a-maxtrick` | max trick + RMS sampling | 0.804 | 0.516 | 0.953 | — | `027-.../best-epochepoch=18.ckpt` |
| `v11b-melnorm` | simple mel norm (X+80)/80 | 0.847 | 0.599 | 0.960 | **0.895** | `027-.../best-epochepoch=18.ckpt` |

v5j-control matches v5j-dualval (exp 026): val/auroc 0.849 vs 0.849. Pipeline verified.

## Experiment 025: Iterative Pseudo-Labeling

### Phase B: Row-addition (FAILED)

Added PLs as separate dataset rows with stratified batch sampling (50/50 GT:PL).

| Slug | Method | val/auroc | val/cmap | LB | Checkpoint |
|------|--------|-----------|----------|-----|------------|
| `v5j-pl-v1` | Row-addition (w=0.5, fold 0) | 0.872 | 0.620 | **0.897** | `025-.../best-epochepoch=18.ckpt` |

### Phase C: Waveform mixing (FAILED)

Mario-style PL waveform mixing: 30% of focal samples overlaid with random PL soundscape
segment (amplitude-controlled). Labels: max(focal, PL). Standard 3-tuple, v5j recipe unchanged.

| Slug | Method | Epochs | val/auroc | val/cmap | val_sc/auroc | Best Epoch | LB | Checkpoint |
|------|--------|--------|-----------|----------|-------------|------------|-----|------------|
| `v3-waveform-mix-20ep` | Waveform mixing (30%, 90/10) | 20 | 0.866 | 0.603 | — | 16 | — | `025-.../best-epochepoch=16.ckpt` |
| `v3-waveform-mix-40ep` | Waveform mixing (30%, 90/10) | 40 | 0.851 | 0.612 | 0.991 | 38 | **0.894** | `025-.../best-epochepoch=38.ckpt` |

**Result: 0.894 LB — regressed -0.012 vs v5j baseline (0.906).** Worse than Phase B
row-addition (0.897). Both pseudo-label approaches failed.

## Experiment 029: 2025 Pretrain → 2026 Fine-tune (Ali Ozan path)

Sequential transfer learning. Stage 1 pretrains v5j on BirdCLEF 2025 focal data
(28,564 rows, 206 species) from ImageNet init. Stage 2 (pending) fine-tunes on
2026 with the 41 (2025 ∩ 2026) overlap species' classifier rows transferred;
remaining 193 rows (165 2026-only + 28 soundscape-only) kept at random init.
Single-variable change from v5j: 2025-pretrained init vs ImageNet init.

Motivated by 2026-04-25 per-frame analysis (v5j is structurally honest on hard
crops; further training-recipe tweaks won't move LB; adding more *information*
via 2025 data is the credible remaining lever). Ali Ozan (0.941 LB) used a
similar pretrain → finetune pattern with filtered Xeno-Canto.

### Stage 1: Pretrain on 2025 — DONE

| Slug | Backbone | Head | Loss | val/auroc | val/cmap | val/top1 | Best Epoch | Walltime | Checkpoint |
|------|----------|------|------|-----------|----------|----------|------------|----------|------------|
| `hgnet-b0-sed-v5j-pretrain-2025` | HGNetV2-B0 | LSEHead | BCE | **0.972** | 0.649 | 0.761 | 15 | 30m16s | `029-.../checkpoints/2026-exp029-v5j-pretrain-2025/best-epochepoch=15.ckpt` |

Trajectory: 0.543 (ep 0) → 0.794 (ep 2) → 0.945 (ep 5) → 0.968 (ep 11) →
**0.972 (ep 15)** → plateau through ep 19. Clean convergence under
OneCycleLR(20 ep), no overfitting, no instability.

### Stage 2: Fine-tune on 2026 — DONE (REGRESSED)

| Slug | Init | LR | val/auroc | val/cmap | val/top1 | Best Epoch | Walltime | LB | Δ vs v5j | Checkpoint |
|------|------|----|-----------|----------|----------|------------|----------|-----|----------|------------|
| `hgnet-b0-sed-v5j-finetune-2026` | 2025-pretrained (Stage 1) | 2e-4 | **0.850** | 0.590 | 0.757 | 12 | 38m40s | **0.885** | **−0.021** | `029-.../checkpoints/2026-exp029-v5j-finetune-2026/best-epochepoch=12.ckpt` |

**Result: 0.885 LB — regressed −0.021 vs v5j baseline (0.906). REJECTED.**

Offline val/auroc 0.850 was essentially identical to v5j-control's 0.851
(Δ −0.001), but LB regressed clearly. Offline-LB anti-correlation
continues — this is the 22nd training-side change where offline metrics
gave no LB signal.

The result is consistent with the 2026-04-25 per-frame finding: v5j
sits at a soundscape-generalization equilibrium, and any training-side
perturbation (different init, different LR) tends to hurt LB even
when offline metrics look identical. 2025 pretraining is itself a
training perturbation — different starting point, different LR, mixed
warm/random head — and the result is what the structural argument
predicted.

Trajectory: 0.566 (ep 0, warm start) → 0.829 (ep 5) → 0.850 (ep 12, best)
→ plateau through ep 19. Earlier peak than v5j-control (~ep 14) consistent
with warm-started overlap species calibrating sooner.

**Apples-to-apples offline comparison** (same dual_validation schema):

| Slug | Init | Best val/auroc | Best Epoch | Δ |
|---|---|---:|---:|---:|
| v5j-control (exp 027) | ImageNet | 0.851 | ~14 | — |
| **exp 029 stage 2** | **2025-pretrained** | **0.850** | **12** | **−0.001** |

Offline result is essentially identical to v5j-control — within seed noise
(Mahog reports ±0.005). The 41-species warm start gave an epoch-0 head
start (0.566 vs ~0.5 random) but this washed out as the 193 random-init
classifier rows trained up. Offline being neutral does not predict LB
direction (offline-LB has been anti-correlated throughout this project).

Comparison baselines (LB):

- v5j single fold (exp 019): **0.906 LB** (ImageNet init, no dual_val)
- v5j-pp2 (exp 021): 0.909 LB (priors)
- v5j-4fold-pp2 (exp 021): 0.917 LB (4-fold + priors)
- Reference: Ali Ozan single model (filtered XC pretrain, much larger): 0.941 LB

**Decision: REJECTED.** Pretrain at our scale (28K 2025 rows, 41-species
overlap, reduced LR for warm start) does not transfer usefully to 2026
LB. Cheap retry candidate: `head_lr: 5e-4` with `lr: 2e-4` (split LR)
to test whether random-init head was undertrained. Otherwise, pivot to
exp 030 silent-class or Salman recipe replication.

## Experiment 030: Sydorskyi-faithful MixUp on v5j

A/B comparison testing whether replacing v5j's Zhang-style interpolated
spectrogram MixUp with Sydorskyi 2025 2nd-place's faithful waveform recipe
(`wave = (w1+w2)/2` + peak-normalize, `target = max(y1, y2)`, p=0.5,
deterministic 0.5 lambda, no theta cap, no warmup) moves LB.

Phase A: v5j HGNetV2-B0 ablation (single-variable mixup change).
Phase B (pending): EfficientNetV2-B0 backbone + 2x2 ablation.

Motivated by 2026-04-26 mixup audit (PR #42) — v5j and prior wave-mixup
attempts (v4j/v6b/v10b) all use interpolated multi-label targets, not
Sydorskyi's `max(y1, y2)`. Plus target-distribution simulation showing 80%
of v5j's training target values are sub-1, 59% have NO full positive.

### Phase A + Phase B v1: Training + LB DONE (all with dual_validation=true)

All 4 cells use LSEHead + BCE + OneCycleLR (lr=5e-4) + 20 epochs + seed=42 +
**dual_validation=true**. Only backbone and mixup recipe differ. v1 LBs are
~0.009 below the original v5j regime due to dual_validation changing
checkpoint selection (see exp 026 v5j-dualval = 0.897 vs v5j = 0.906).

| Slug (v1) | Backbone | Mixup recipe | val/auroc | val/cmap | val/f1_macro | Best Ep | **LB** | Walltime |
|------|----------|-------------|-----------|----------|--------------|---------|------:|----------|
| `hgnet-b0-sed-v5j-control` | HGNetV2-B0 | spec, interp, theta=0.8 | 0.8490 | 0.5959 | 0.5039 | 14 | **0.890** | 37m 44s |
| `hgnet-b0-sed-v5j-sydorskyi` | HGNetV2-B0 | wave sydorskyi (avg+peak-norm+max) | 0.8472 | 0.6065 | 0.5467 | 14 | **0.892** | 41m 45s |
| `effnetv2b0-sed-control` | EffNetV2-B0 (192-ch) | spec, interp, theta=0.8 | 0.8510 | 0.6124 | 0.5320 | 19 | **0.893** | 41m 17s |
| **`effnetv2b0-sed-sydorskyi`** | **EffNetV2-B0** | **wave sydorskyi** | **0.8518** | **0.6303** | **0.5751** | 19 | **0.901** | 45m 37s |

### v1 2x2 LB matrix

| | control (spec interp) | sydorskyi (wave max) | mixup Δ |
|---|---:|---:|---:|
| HGNetV2-B0 | 0.890 | 0.892 | +0.002 (null) |
| EfficientNetV2-B0 | 0.893 | **0.901** | **+0.008** |
| backbone Δ | +0.003 | +0.009 | |

**Strong interaction**: Sydorskyi mixup helps on v2-B0 but not HGNet. The
diagonal corner (effnetv2b0_sydorskyi) is the best v1 cell at +0.011 LB
vs v5j_control. Best offline cell is also this one — offline 2x2 and LB
2x2 agree on the winner, even though absolute numbers are below v5j 0.906
due to the dual_validation regime cost.

### Phase A + Phase B v2: Training DONE (no dual_validation, LB pending)

All 4 variants retrained with `dual_validation: true` removed; run names
get `-v2` suffix so v1 checkpoints stay intact. Same seed, same recipe
otherwise. Trained overnight 2026-04-26 → 2026-04-27. All 4 SLURM jobs
exit 0; same checkpoint slugs/dirs as v1 (Kaggle export pipeline already
points at `-v2` dirs).

| Slug (v2) | val/auroc | val/cmap | val/f1_macro | Best Ep | best_val | Walltime | SLURM |
|------|----------|----------|-------------|--------:|---------:|---------|------:|
| `hgnet-b0-sed-v5j-control` | 0.8728 | 0.6268 | 0.5270 | 15 | 0.8747 | 40m 12s | 7455688 |
| `hgnet-b0-sed-v5j-sydorskyi` | 0.8700 | 0.6339 | 0.5663 | 14 | 0.8737 | 1h 14m 54s | 7455689 |
| `effnetv2-b0-sed-control` | 0.8745 | 0.6461 | 0.5545 | 19 | 0.8745 | 41m 5s | 7455690 |
| **`effnetv2-b0-sed-sydorskyi`** | **0.8760** | **0.6556** | **0.5922** | 19 | **0.8760** | 41m 16s | 7455691 |

W&B run ids: `586p423f`, `e6hdxkg8`, `fyqjktl5`, `cg9v7ksj`.

### v2 reproducibility check (vs v5j-original 0.874 → 0.906 LB regime)

| | val/auroc | Δ vs v5j 0.874 | Within seed noise (±0.005)? |
|---|---:|---:|---|
| v5j_control_v2 | **0.8728** | −0.001 | ✓ YES |
| v5j_sydorskyi_v2 | 0.8700 | −0.004 | ✓ YES |
| effnetv2b0_control_v2 | 0.8745 | +0.001 | ✓ YES |
| effnetv2b0_sydorskyi_v2 | **0.8760** | **+0.002** | ✓ YES |

**Pipeline reproducibility CONFIRMED in the v2 (no-dual_val) regime.**
v5j_control_v2 reproduces v5j-original within −0.001 val/auroc. The +0.024
val/auroc shift across all 4 cells from v1 → v2 is consistent with the
dual_validation hypothesis (val/auroc focal-only → combined focal+SS).

### v2 2x2 offline summary (val/auroc / val/cmap / val/f1_macro)

| | control (spec, interp) | sydorskyi (wave, max) |
|---|---|---|
| **HGNetV2-B0** | 0.8728 / 0.6268 / 0.5270 | 0.8700 / 0.6339 / 0.5663 |
| **EfficientNetV2-B0** | 0.8745 / 0.6461 / 0.5545 | **0.8760 / 0.6556 / 0.5922** |

Main effects (averaged across the other axis):
- **Backbone** (v2-B0 − HGNet): +0.004 auroc, **+0.020 cmap**, **+0.027 f1_macro**
- **MixUp** (sydorskyi − control): −0.001 auroc, +0.008 cmap, **+0.039 f1_macro**
- **Interaction**: small positive on auroc (+0.004), tiny on cmap (+0.002),
  near-flat on f1_macro (−0.002)

Best v2 cell on every offline metric: `effnetv2b0_sydorskyi_v2`. Compared to
v5j-original, +0.002 auroc / +0.029 cmap (vs v5j_control_v2) / +0.065 f1_macro.

The 2x2 *structure* reproduces across v1 and v2 (backbone +0.020 cmap, mixup
positive on cmap/f1_macro, additive interaction). LB ordering should match
v1 (`effnetv2b0_sydorskyi` > rest); each cell expected ~+0.009 LB above v1
if dual_val cost is fully recoverable.

### v2 LB predictions (decision matrix)

| Cell | v1 LB | Predicted v2 LB | Verdict if hits band |
|---|---:|---:|---|
| v5j_control_v2 | 0.890 | 0.899-0.906 | dual_val confound CONFIRMED |
| v5j_sydorskyi_v2 | 0.892 | 0.899-0.906 | mixup null on HGNet (consistent w/ v1) |
| effnetv2b0_control_v2 | 0.893 | 0.901-0.905 | backbone null on control |
| **effnetv2b0_sydorskyi_v2** | **0.901** | **0.908-0.912** | **first single-model > v5j 0.906** |

### Seed=43 retrain (2026-04-28): training DONE, partial LB, exp closed

Diagnostic for the seed=42 EffNet × Sydorskyi +0.008 LB asymmetry. Only
`data.seed` changes (42 → 43); everything else identical to v2. Different
seed → different 90/10 val split → absolute val/auroc not comparable across
seed groups. Within-seed deltas are the diagnostic.

| Slug (seed=43) | val/auroc | val/cmap | val/f1_macro | Best Ep | best_val | LB | SLURM |
|------|---------:|---------:|-------------:|--------:|---------:|---:|------:|
| `hgnet-b0-sed-v5j-control-seed43` | 0.8926 | 0.6426 | 0.5679 | 16 | 0.8933 | **0.883** | 7488877 |
| `hgnet-b0-sed-v5j-sydorskyi-seed43` | 0.8967 | 0.6437 | 0.5854 | 16 | 0.8978 | (skipped) | 7488880 |
| `effnetv2-b0-sed-control-seed43` | 0.8890 | 0.6455 | 0.5604 | 18 | 0.8930 | (skipped) | 7488881 |
| **`effnetv2-b0-sed-sydorskyi-seed43`** | **0.8932** | **0.6490** | **0.5945** | 18 | **0.8982** | (skipped) | 7488884 |

**`v5j_control_seed43` LB = 0.883**, −0.013 below `v5j_control_v2` (0.896).
This widens the documented single-fold seed band from ±0.005 to ~±0.013 on
HGNet for this codebase. Within-seed mixup Δs (±0.004 offline) sit inside
this wider noise floor → remaining 3 seed=43 LB submissions skipped as
**conscious decision** to redirect daily-cap budget to the next experiment.

### Exp 030 closing verdict (2026-04-28)

| Question | Answer |
|---|---|
| Does Sydorskyi waveform mixup beat Zhang interpolated mixup on this codebase? | **Null on HGNet**, borderline +0.008 on EffNet (inside ±0.013 noise floor) |
| Does EffNetV2-B0 backbone beat HGNetV2-B0? | **Null on LB** (offline cmap/f1 +; LB inside noise) |
| First single-model training-recipe LB lift? | Not confirmed |
| Best single-model still | **v5j 0.906** (exp 019, reconfirmed via v5j-resubmit) |
| Best single-model challenger | **`effnetv2b0_sydorskyi` 0.901** (v1≡v2 weights) |

The cmap +0.014 / f1 +0.039 offline mixup signals are real and reproduce
across seeds, but don't translate to single-fold LB above noise. If
revisited, the right setup is 4-fold ensemble (where seed noise averages
out), not another single-fold seed.

### 2x2 offline summary (val/auroc / val/cmap / val/f1_macro)

| | control (spec, interp) | sydorskyi (wave, max) |
|---|---|---|
| **HGNetV2-B0** (v5j arch) | 0.8490 / 0.5959 / 0.5039 | 0.8472 / 0.6065 / 0.5467 |
| **EfficientNetV2-B0** | 0.8510 / 0.6124 / 0.5320 | **0.8518 / 0.6303 / 0.5751** |

Main effects (averaged across the other axis):
- **Backbone** (v2-B0 − HGNet): +0.003 auroc, **+0.020 cmap**, **+0.028 f1_macro**
- **MixUp** (sydorskyi − control): −0.001 auroc, **+0.014 cmap**, **+0.043 f1_macro**
- Interaction: small positive on auroc/cmap (sydorskyi helps slightly more on v2-B0)

`effnetv2b0_sydorskyi` is the best cell on every offline metric. cmap +0.034
and f1_macro +0.071 vs v5j (HGNet + spec interp). LB will tell whether this
translates or hits the same offline-LB anti-correlation as our 22+ prior
recipe attempts.

### Reproducibility check (v5j_control)

| | val/auroc | Notes |
|---|---:|---|
| v5j-control (exp 027) | 0.851 | original baseline run |
| **v5j_control (exp 030)** | **0.849** | Δ −0.002, within seed noise |

**Pipeline reproducibility confirmed** — no codebase drift since exp 027.

### Phase B caveat (EffNetV2-B0 feature dim)

Encoder uses `tf_efficientnetv2_b0.in1k` stage-5 spatial features (192
channels), NOT the post-pool 1280-d projection set in the config. Both
v2-B0 cells share this same effective architecture, so the within-backbone
mixup comparison is internally clean. The v2-B0-vs-HGNet comparison is at
very different feature dims (192 vs 1024); take backbone main effect with
that caveat. A "faithful" v2-B0 setup might use a different feature
extraction layer with more channels.

See `experiments/030-sydorskyi-mixup/results.md` for full details.

## Experiment 031: Perch v2 Knowledge Distillation on v5j (Tucker recipe, Phase 1)

Single-variable add of Perch v2 embedding-space KD on top of v5j_control_v2.
DistillHead (GAP + Linear 1024→1536) MSE-aligned to frozen Perch v2 ONNX
clip embedding on the same pre-mix waveform. Backbone driven by `dist_loss`
only (`feat.detach()` on LSEHead branch), BCE-clip cls loss unchanged from
v5j. α_distill=1.0, single-fold seed=42, 20 epochs.

| Slug | Variant | val/auroc | best ep | LB | Δ vs v5j-original (0.906) | Checkpoint |
|------|---------|----------:|--------:|----:|---:|---|
| `hgnet-b0-sed-v5j-kd` | v5j + Perch v2 KD (best-by-val) | 0.8629 | 17 | **0.890** | **−0.016** | `031-perch-kd-distillation/.../2026-exp031-v5j-kd/best-epochepoch=17.ckpt` |
| `v5j_control` (exp 031) | back-compat baseline (no KD) | 0.8747 | 15 | **0.896**† | −0.010 | bit-identical to exp 030 v5j_control_v2 |
| **`hgnet-b0-sed-v5j-control-last`** | **v5j_control ep 19 (save_top_k=-1 rerun)** | 0.8728 | 19 | **0.897** | **−0.009** | `031-perch-kd-distillation/.../2026-exp031-v5j-control-resave/best-epochepoch=19.ckpt` |
| **`hgnet-b0-sed-v5j-kd-last`** | **v5j+KD ep 19 (save_top_k=-1 rerun)** | 0.8595 | 19 | **0.891** | **−0.015** | `031-perch-kd-distillation/.../2026-exp031-v5j-kd-resave/best-epochepoch=19.ckpt` |
| **`hgnet-b0-sed-v5j-eracode`** | **v5j retrained with e420633 (v5j-era) code on today's env (drift diag)** | 0.8732 (cmap **0.6429** matches v5j-original to 4 dp) | 19 | **0.899** | **−0.007** | `017-mel-overhaul/.../2026-exp019-v5j-diag-eracode-may10/best-epochepoch=19.ckpt` |

† Inherited from exp 030 v5j_control_v2 (LB 0.896). All 8 offline metrics
bit-identical to 4 dp under deterministic training (`seed_everything(42, workers=True) + Trainer(deterministic=True)`),
so weights are bit-identical.

### v5j reproduction: drift investigation closed (2026-05-11)

The 2026-05-10/11 diagnostic suite (best-vs-last LB submissions + v5j-era
worktree training + uv.lock pinning) resolved the question of why today's
v5j retraining lands at 0.897 instead of the historical 0.906.

| Question | Answer |
|---|---|
| Inference pipeline reproduces v5j-original? | ✅ Yes (v5j-resubmit LB 0.906, exact reproduction of original ep 19 ckpt through today's submission.py) |
| Was 0.906 → 0.896 a ckpt-selection artifact (ep 15 vs ep 19)? | ❌ No — ep 19 weights from today's training give LB 0.897, only +0.001 over ep 15's 0.896 |
| Is the regression caused by HEAD codebase vs e420633 codebase? | ⚠️ Partially — v5j-eracode (e420633 worktree code, today's env) hit LB 0.899, only 0.002 better than HEAD's 0.897. Code accounts for ~0.002 LB. |
| Is the bigger gap from GPU/node fp variance? | ✅ Yes — v5j-original (LB 0.906) and v5j-eracode (LB 0.899) used SAME code but DIFFERENT SLURM nodes; cross-hardware fp behavior accounts for ~0.007 LB. |
| Does val/cmap separate code from hardware effects? | ✅ Yes — same code different node → val/cmap matches to 4 dp (0.6429 ↔ 0.6429). Code change → 0.016 cmap drop (but only 0.002 LB drop). |

**Final attribution** of the 0.906 → 0.897 retrain gap:
- ~0.007 LB: cross-hardware GPU fp non-determinism (not fixable in code; needs same-node pinning or multi-fold averaging)
- ~0.002 LB: code change between e420633 and HEAD (now pinned via uv.lock + version pins in pyproject.toml — commit `9f9ec3d`)

**0.906 stays as the real v5j baseline** (proven by v5j-resubmit).
**Single-fold v5j has ~0.007 cross-node LB variance** that's fundamental to
GPU training; no amount of code reproducibility eliminates it. For
LB-comparable experiments going forward, use **paired within-run controls**
and read **deltas vs same-run control**, not deltas vs historical 0.906.

Comprehensive analysis at:
- `analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md`
- `analysis/notes/2026-05-10-v5j-drift-diagnostics-findings.md`
- `analysis/diagnostics/v5j-codebase-drift/README.md`

### Verdict: REJECT (final — 2026-05-11)

Per the success matrix in `experiments/031-perch-kd-distillation/proposal.md`,
read against v5j_control 0.896 with ±0.013 single-fold seed band:

| Bucket | LB threshold | v5j_kd 0.890 |
|---|---|---|
| CONFIRM | ≥ 0.910 | ❌ |
| PARTIAL | 0.903–0.909 | ❌ |
| WEAK | 0.897–0.902 | ❌ |
| **REJECT** | ≤ 0.896 | **✅** |

The KD lever did not transfer as a single-variable add on this codebase.
Phase 2 (ownership + 4-fold) is HALTED.

### Post-mortem candidates (ranked by likelihood)

1. **LSEHead is the wrong downstream consumer of Perch features.** LSE pools
   per-frame logits by magnitude; Perch encodes spatial localization that
   AttHead would consume directly. Highest-likelihood candidate.
2. **20-epoch budget too short.** KD's val/auroc trajectory still narrowing
   the gap to control through ep 17 (best). Tucker uses 25 + cosine.
3. **`alpha_distill = 1.0` miscalibrated** for our codebase's mel + feature dim.
4. **KD lift may require Tucker's full stack** (AttHead + dual-supervision +
   hard-max mixup + SpecAugment + focal×SS + 90/10 batch composition).

See `experiments/031-perch-kd-distillation/results.md` § Post-mortem for full
discussion + ranked Phase 3+ follow-up experiments.

### Diagnostic pending: best-vs-last checkpoint instability

The 0.890 was on best-by-val ep 17. v5j-original's 0.906 was on the LAST
epoch (ep 19). Both runs save `last.ckpt` (`save_last: true` in the config);
ep 19 weights are on disk for both v5j_kd and v5j_control but unsubmitted.
Two pending submissions:

- `v5j_kd_last`: tests "0.890 was ckpt-selection artifact" vs "KD doesn't transfer"
- `v5j_control_last`: tests "ep 19 weights recover 0.906 on new codebase" (isolates ckpt-selection from codebase drift)

## Experiment 032: Tucker distilled-SED Full Replication — PARTIAL REPLICATION (Phase 2 5-fold 0.915 LB vs Tucker 0.917; HGNet ablation refutes backbone hypothesis)

Faithful port of `tuckerarrants/bc2026-distilled-sed`: EffNet-B0 NS-JFT
(use_post_head=True, 1280-d) + AttHeadLogitSpace (LOGIT-space aggregation) +
dual-supervision (0.5·BCE clip + 0.5·BCE frame_max) + Perch v2 embedding-MSE KD
(feat.detach() on cls; α=1.0) + waveform mixup hard-max (Beta(0.4,0.4)) +
focal×SS mixup + SpecAugment (1×freq×10 + 2×time×10) + 25 ep AdamW + 2 ep
warmup + cosine + 90/10 focal/SS MixSamp + StratifiedKFold focal +
GroupKFold SS. Fold 0, seed=42.

### Phase 1 (broken mel pipeline) — REJECTED

| Slug | Variant | Tucker val macro AUC (fold 0) | LB | Δ paired ctrl | Δ vs v5j_control_last 0.897 | Checkpoint |
|------|---------|----------:|---:|---:|---:|---|
| `distilled-sed-control` (Phase 1, broken) | Tucker recipe, KD OFF (last.ckpt, ep 24) | 0.893 (post-hoc) | **0.522** | — | **−0.375** | job 9088255 |
| `distilled-sed-kd` (Phase 1, broken) | Tucker recipe + Perch v2 KD ON (last.ckpt, ep 24) | TBD (rerun) | **0.513** | **−0.009** | **−0.384** | job 9088258 |

### Phase D retrain (Tucker-faithful mel via TuckerMelSpectrogram) — CONFIRM ✅

After D.2 confirmed our training mel diverged from Tucker's by max\|d\| = 3.20 (Pearson r = 0.9356), D.3 landed `TuckerMelSpectrogram` (verbatim port of Tucker's `MelSpecTransform`). Configs flipped `mel.type: "nnaudio"` → `"tucker"`. Training and Kaggle inference now share the SAME torchaudio code paths — they cannot drift apart. Verified by 12 unit tests including the regression guard `test_diverges_from_nnaudio_amplitude_to_db`.

| Slug | Variant | val/auroc (Tucker macro AUC) | val/cmap (Tucker macro AP) | best_val_score | LB | Δ paired |
|------|---------|----------:|----------:|----------:|---:|---:|
| `distilled-sed-control` (Phase D, ep 24 last-v2.ckpt) | Tucker recipe, KD OFF, mel fix | 0.887 | 0.645 | 0.659 | **0.856** | — |
| `distilled-sed-kd` (Phase D, ep 24 last-v2.ckpt) | Tucker recipe + Perch v2 KD ON, mel fix | 0.908 | 0.692 | 0.709 | **0.906** | **+0.050** |
| paired Δ (KD − control) | — | +0.021 | +0.047 | +0.050 | **+0.050 LB** | — |

**Verdict: CONFIRM** per proposal's success matrix (+0.050 ≥ +0.014 threshold).
- `distilled-sed-kd` matches v5j-original 0.906 LB **exactly**, lands within 0.004 of Tucker's published single-fold target.
- The KD lever transfers in our codebase. The full Tucker recipe matters because KD requires the AttHeadLogitSpace consumer of Perch features — same KD branch on LSEHead (exp 031) got Δ = −0.006; on AttHeadLogitSpace it gets Δ = +0.050 (+0.056 LB swing from head choice).
- This validates exp 031 post-mortem hypothesis #1: "LSEHead was the wrong downstream consumer of Perch features."

Recipe alone (control 0.856 vs v5j_control_last 0.897 = −0.041) is **NOT** an LB win over v5j by itself. KD does all the heavy lifting on this stack.

End-to-end validation (D.5) predicted both LBs:
- Control: e2e 0.898 → LB 0.856 (−0.042 offline↔LB gap)
- KD: e2e 0.912 → LB 0.906 (−0.006 offline↔LB gap, normal exp 030/031 range)

Retrain jobs: 9136231 (control, walltime 3h39m, node atl1-1-03-007-11-0) + 9136232 (KD, walltime 4h26m, atl1-1-03-006-1-0). W&B: `dsgt-arc/birdclef-2026-exp032/g5rblf6o` (control) + `0w29iaut` (KD).

Checkpoints (Phase D retrains): `/storage/project/.../checkpoints/2026-exp032-distilled-sed-{control,kd}/last.ckpt`. Phase D `last.ckpt` overwrites Phase 1; Phase 1 epoch-best checkpoints carry the `-v1` suffix, Phase D ones don't.

### Phase 2: 5-fold KD ensemble (folds 1–4) + HGNet KD backbone ablation — LB LANDED (2026-05-25 PM)

Extends the Phase D CONFIRM verdict in two directions:
- **2.A 5-fold KD ensemble**: train folds 1–4 alongside the existing Phase D fold 0 (LB 0.906). Tucker's published 0.917 LB = 5-fold + 5-tap Gaussian smoothing. Inference: per-fold blended logits → LOGIT-space fold mean → 5-tap Gauss smooth → single sigmoid.
- **2.D HGNet+KD ablation**: single-fold (fold 0) backbone swap `tf_efficientnet_b0.ns_jft_in1k` → `hgnetv2_b0.ssld_stage2_ft_in1k` (v5j's backbone) with `use_post_head: true → false`. Tests whether the −0.041 Phase D control-vs-v5j gap is backbone-driven. Same Tucker recipe + KD otherwise.

All 5 runs trained 25 ep + 2 ep warmup + cosine, AdamW lr=5e-4 wd=1e-4, fp16, grad_clip=1.0, BCE dual-supervision 50/50, blend 50/50, Beta(0.4,0.4) hard-max mixup, SpecAugment, 90/10 focal/SS MixSamp, seed=42, `save_top_k: -1`. Configs: `distilled_sed_kd.yaml` (folds 1-4) + `distilled_sed_kd_hgnet.yaml` (HGNet).

| Slug | Backbone | Fold | val/auroc (last) | val/cmap (last) | val/top1 | val/top5 | best val/cmap | Best Ep | Walltime | SLURM | W&B |
|------|----------|------|-----------------:|----------------:|---------:|---------:|--------------:|--------:|----------|------:|-----|
| `distilled-sed-kd-fold1` | EffNet-B0 NS-JFT | 1 | **0.9281** | 0.6654 | 0.8029 | 0.9197 | 0.6777 | 11 | 4h22m | 9158704 | `pmbh` |
| `distilled-sed-kd-fold2` | EffNet-B0 NS-JFT | 2 | 0.9079 | 0.6092 | 0.8082 | 0.9658 | 0.6237 | 21 | 4h22m | 9158705 | `hoqe` |
| `distilled-sed-kd-fold3` | EffNet-B0 NS-JFT | 3 | 0.9236 | 0.6927 | 0.7919 | 0.9396 | 0.6954 | ~24 | 4h22m | 9158706 | `bpuk` |
| **`distilled-sed-kd-fold4`** | EffNet-B0 NS-JFT | 4 | **0.9310** | **0.7295** | **0.9013** | **0.9737** | 0.7335 | 21 | 4h22m | 9159168 | `lwy9` |
| `distilled-sed-kd-hgnet` | HGNetV2-B0 SSLD | 0 | 0.9069 | 0.6706 | 0.7613 | 0.9097 | 0.6963 | ~17 | 4h19m | 9158713 | `hyu0` |

Phase D fold 0 (for reference, on LB 0.906): val/auroc 0.908, val/cmap 0.692.

5-fold KD val/auroc set: {0.908 (fold 0), 0.9281, 0.9079, 0.9236, 0.9310}, mean **0.920**.

Fold 0 originally landed at the **bottom** of the 5-fold band — explains why Phase D's single-fold LB 0.906 sits at the low end of Tucker's published single-fold trajectory (~0.91-0.92). Fold 4 is the strongest individual fold; fold 2 the weakest.

### D.5 end-to-end validation (pre-flight before LB submission) — ALL PASS ✅

Each ckpt's training-side `val/auroc` compared against `validate_end_to_end.py` macro AUC (kernel inference path: `kernel_load_audio_chunk → compute_mel_tucker → encoder → AttHeadLogitSpace → LOGIT-space blend → 5-tap Gauss smooth → single sigmoid` on the same val SS fold). Gate: |gap| ≤ 0.02.

| Slug | Training-side val/auroc | E2E val/auroc | Gap | Status | n_evaluable | Val windows |
|------|----------------------:|--------------:|----:|--------|------------:|------------:|
| `distilled-sed-kd-fold0` (Phase D) | 0.908 | 0.912 | +0.004 | ✅ PASS | — | 156 |
| `distilled-sed-kd-fold1` | 0.9281 | 0.9361 | +0.008 | ✅ PASS | 31 | 156 |
| `distilled-sed-kd-fold2` | 0.9079 | 0.9190 | +0.011 | ✅ PASS | 39 | 156 |
| `distilled-sed-kd-fold3` | 0.9236 | 0.9302 | +0.007 | ✅ PASS | 41 | 156 |
| `distilled-sed-kd-fold4` | 0.9310 | 0.9366 | +0.006 | ✅ PASS | 45 | 156 |
| `distilled-sed-kd-hgnet` | 0.9069 | 0.9108 | +0.004 | ✅ PASS | 29 | 156 |

E2E is uniformly slightly higher than training-side because the kernel path applies 5-tap Gaussian smoothing across the 12 windows per file in LOGIT space (acts as a temporal denoiser; matches Tucker's published "+0.004 from MengYe smoothing trick"). Pipeline is end-to-end consistent across all 5 new checkpoints. **LB submission is safe.**

D.5 jobs: 9178659-63 first round (FAIL — fold_k hardcoded to 0; script bug, not model bug; ckpts were being eval'd against own training data → 0.995); 9178859-62 re-run with `--fold N` patch (PASS as above). HGNet's first-round PASS valid (trained + eval'd on fold 0).

Checkpoints (Phase 2): top-level `2026-exp032-distilled-sed-kd/last-v2.ckpt` (Phase D fold 0, LB 0.906) + `2026-exp032-distilled-sed-kd/fold{1,2,3,4}/last.ckpt` (Phase 2.A folds 1-4) + `2026-exp032-distilled-sed-kd-hgnet/last.ckpt` (Phase 2.D HGNet).

LB expectations:
- 5-fold ensemble: target 0.910–0.920 (matches Tucker's published 0.917). 5-fold mean val/auroc 0.920 + 5-tap smoothing across folds in logit space should land in this band.
- HGNet+KD single-fold: 0.90-0.92 if HGNet > EffNet for KD; otherwise marginal. Val metrics ≈ Phase D fold 0 EffNet (HGNet val/auroc 0.907 vs EffNet 0.908) → predicting LB also flat near 0.906.

### Phase 2 LB results (2026-05-25 PM): PARTIAL REPLICATION

| Slug | Variant | val/auroc | E2E val/auroc | LB | Δ vs Phase D fold 0 (0.906) | Δ vs Tucker target |
|------|---------|----------:|--------------:|---:|---:|---:|
| `distilled-sed-kd-5fold` | 5-fold KD ensemble | 0.920 (mean) | — | **0.915** | **+0.009** | **−0.002 vs 0.917** |
| `distilled-sed-kd-hgnet` | HGNet KD single-fold ablation | 0.907 | 0.911 | **0.901** | **−0.005** | — |
| `distilled-sed-kd-fold4` | EffNet KD fold 4 (strongest offline) | 0.931 | 0.937 | **0.903** | **−0.003** | offline-LB anti-correlation |

**5-fold ensemble (0.915 LB)**: matches Tucker's published 0.917 within ~0.002 (well inside the documented cross-hardware variance ~0.007). Direction (+0.009 single→5-fold) matches Tucker's reported +0.011 single→5-fold lift. **What we did NOT replicate**: Tucker's "0.93+ with PP" stretch number, which depends on his private site/hour priors + per-class temperature (NOT in the public notebook). The publicly-replicable subset of his recipe yields 0.91-0.92 LB on this codebase, the same band his discussion thread reports.

**HGNet+KD (0.901 LB)**: −0.005 below EffNet+KD fold 0 (0.906) despite near-identical offline val/auroc. **Backbone was NOT the bottleneck.** Refutes the Phase D hypothesis that the −0.041 control-vs-v5j gap was backbone-driven. The recipe-vs-recipe gap is driven by other ingredients (AttHeadLogitSpace vs LSEHead, dual-supervision vs single-clipwise, etc.). EffNet remains the right backbone for this recipe.

**Fold 4 single-fold (0.903 LB)**: strongest offline val/auroc (0.931, +0.023 over fold 0) but **near-worst LB**. Single-fold LB band is narrow (~0.003 across 2 data points spanning 0.023 of offline range). The +0.009 single→5-fold ensemble lift is **pure variance-reduction** (Gaussian smoothing across 5 fold logits in LOGIT space), not "picking up better folds". 23rd data point in the offline-LB anti-correlation pattern.

### Single-fold variance characterization (fold 0 + fold 4 landed)

| Fold | Offline val/auroc | E2E val/auroc | LB | Δ offline vs fold 0 | Δ LB vs fold 0 |
|---|---:|---:|---:|---:|---:|
| 0 (Phase D) | 0.908 | 0.912 | **0.906** | — | — |
| 4 | 0.931 | 0.937 | **0.903** | **+0.023** | **−0.003** |

**Conclusion**: offline rank ≠ LB rank on single-fold. Folds 1/2/3 not submitted (would land in the same 0.903-0.906 band with high confidence — low marginal information for high LB-slot cost). The ensemble +0.009 lift is from variance-reduction across folds, not from any "magic fold" inside the 5-fold set.

### Phase 2 implications for next steps

The Tucker recipe is replicated within noise of the publicly-quantified target. Path to higher LB is no longer "fix the recipe":
1. **Ensemble distilled-sed-kd-5fold (0.915) with exp 023 perch-sed-ensemble-mlp-only (0.929 LB)** — highest-EV move. Two strong, decorrelated stacks (Tucker SED + Perch probes); estimated landing 0.93-0.94 LB.
2. **Port site/hour priors (EoS.6 style)** onto distilled-sed-kd-5fold — exp 021 priors gave +0.004 on v5j-4fold-pp2; similar lift here would land 0.918-0.920. Likely the path Tucker took for "0.93+ with PP".
3. **Accept the publicly-replicable limit** of this recipe and pivot LB-lift effort to other levers (Sydorskyi-yoink stretch, MixIT-based PL, etc.).

Best single Tucker-SED model: `distilled-sed-kd` (Phase D, fold 0) = **0.906 LB**, matching v5j-original. Best Tucker-SED ensemble: `distilled-sed-kd-5fold` = **0.915 LB**, within ~0.002 of Tucker's published 0.917. Neither approaches the exp 023 ensemble's 0.929 (current overall best).

### Verdict: Phase 1 REJECTED (pipeline broken — historical, now resolved)

Both runs collapsed to barely-above-random LB despite offline Tucker macro AUC
tracking Tucker's published 0.91 trajectory exactly (control ep 6 = 0.889 vs
Tucker's ~0.91 — matches to 4 dp). The paired Δ (−0.009) is within seed noise
and is uninformative because the control itself is broken.

### Leading hypothesis (HIGH-likelihood, blocking)

**Training/inference mel-pipeline mismatch.**

- Tucker: `power_mel → AmplitudeToDB(top_db=80) → per-sample (mel - mean) / (std + eps)`. One normalization pass at both training AND inference.
- Our training: NNAudio `norm_type="amplitude_to_db"` does `power_to_db → z-score → min-max [0,1]`, THEN the module's `per_sample_mel_norm` does another z-score. **Three normalization passes**.
- Our inference (`compute_mel_tucker` in submission.py): single z-score after AmplitudeToDB — matches Tucker but NOT our training distribution.

Model trained on triple-normalized features sees single-normalized features at
inference → output is near-random. Single bug sufficient to explain LB 0.522
given a model that scores macro AUC 0.89 on its own training-distribution val.

### Next steps (see `experiments/032-tucker-distilled-sed/tasks.md` Phase D)

1. Side-by-side numerical test of our mel pipeline vs Tucker's (`scripts/diag_mel_pipeline.py`).
2. Fix training mel to match Tucker's single-pass normalization (add `norm_type: "tucker_amp2db"` to NNAudioMelSpectrogram).
3. Add end-to-end validation script (`scripts/validate_end_to_end.py`) — load ckpt, run kernel inference path on `train_soundscapes/`, compute macro AUC against labels, compare against offline val. Make this a pre-flight check before every LB submission.
4. Retrain paired control + KD with fixed mel.
5. Re-export + re-submit. Phase 2 (5-fold) and Phase 3 (KD diagnosis) BLOCKED until end-to-end validation passes AND retrained control LB ≥ 0.900.

### Methodology lesson

**Offline val/cmap/auroc cannot detect training-inference distribution
mismatch** because the offline val uses the training pipeline. End-to-end
validation against soundscape labels using the inference kernel's full path
must be a pre-flight check, not an after-the-fact diagnostic.

See `experiments/032-tucker-distilled-sed/results.md` for full LB table,
hypothesis ranking, and audit checklist.

## Experiment 033: Own-teacher Pseudo-Labels on distilled-sed-kd — REJECT (3/5 Day-1 LB landed all in REJECT band; protocol does not transfer at LB)

Applies the Antoine / OverfitOracle / Boredom 1:1 own-teacher PL protocol on top
of the exp 032 Phase D `distilled-sed-kd` recipe (baseline LB 0.906). 5
protocol cells × 2 SED teachers × fold 0 = 10 paired runs. Prior PL attempts
(exp 014/015/025) used down-weighted PLs (loss_weight=0.5) or hardcoded 50/50
StratifiedBatchSampler — both regressed LB. This experiment isolates the
disclosed competitor protocol (1:1 loss_weight + uniform-or-natural sampler +
own-arch SED teacher) for the first time on this codebase.

Teachers: T1 = `v5j-4fold` (`v1/sed_pseudo_labels.parquet`, teacher LB 0.913);
T2 = `distilled-sed-kd-5fold` (`v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`,
teacher LB 0.915 — generated this session). Effective PL pool 127,104 unlabeled
SS segments per teacher (val-fold leak + labeled-SS train files filtered).

### Phase G: 10-ablation grid — offline metrics (fold 0, val-side)

Baseline (`distilled-sed-kd` Phase D fold 0): val/auroc 0.908, val/cmap 0.692,
LB **0.906**. All 10 ablations land above the baseline offline; the spread is
+0.003 to +0.041 val/auroc. **Offline rankings are direction-only** (per exp 032
fold 4: +0.023 offline → −0.003 LB).

| Slug | Init | Sampler (focal/sc/pl share) | Teacher | val/auroc | val/cmap | best val/cmap | Best Ep | Walltime | SLURM | W&B |
|------|------|-----------------------------|---------|----------:|---------:|--------------:|--------:|----------|------:|-----|
| `2026-exp033-a3-kd` | warm | MixSamp 0.2/0.0/0.8 | T2 | **0.9486** | 0.8311 | 0.8350 | 7 | 2h41m | 9191660 | `ax6t5qdz` |
| `2026-exp033-a4-kd` | warm | Uniform              | T2 | **0.9477** | **0.8323** | 0.8335 | 7 | 2h38m | 9191661 | `htfgco92` |
| `2026-exp033-a2-kd` ⭐ | random | Uniform           | T2 | 0.9390 | 0.8182 | 0.8201 | 3 | 6h29m | 9191659 | `1ye9s2ni` |
| `2026-exp033-a2-v5j` ⭐ | random | Uniform          | T1 | 0.9362 | 0.7923 | 0.7977 | 14 | 7h20m | 9190540 | `4y0h7h1a` |
| `2026-exp033-a1-kd` | random | MixSamp 0.5/0.0/0.5 | T2 | 0.9357 | 0.8242 | 0.8390 | 12 | 6h46m | 9191658 | `2cg0kcux` |
| `2026-exp033-a4-v5j` | warm | Uniform             | T1 | 0.9352 | 0.8043 | 0.8058 | 7 | 2h39m | 9190544 | `dl7pz6kv` |
| `2026-exp033-a3-v5j` | warm | MixSamp 0.2/0.0/0.8 | T1 | 0.9345 | 0.8061 | 0.8095 | 7 | 2h38m | 9190542 | `f781pbyz` |
| `2026-exp033-a1-v5j` | random | MixSamp 0.5/0.0/0.5 | T1 | 0.9288 | 0.7821 | 0.7859 | 11 | 7h25m | 9190539 | `ct3by7ax` |
| `2026-exp033-a5-kd` | random | MixSamp 0.7/0.0/0.3 | T2 | 0.9285 | 0.7915 | 0.7954 | 10 | 6h45m | 9191662 | `vn3vyrp5` |
| `2026-exp033-a5-v5j` | random | MixSamp 0.7/0.0/0.3 | T1 | 0.9113 | 0.7586 | 0.7595 | 21 | 6h41m | 9190545 | `opv1942s` |

⭐ = canonical Antoine protocol (random init + uniform sampler).
Per-batch composition counts (focal/sc/pl @ B=64): A1 32/0/32, A2/A4 ~12/0/~52
(natural over ConcatDataset 29K/0.6K/127K), A3 13/0/51, A5 45/0/19.

### Phase G offline read-outs

**Teacher axis (T2 vs T1, 5 paired tests)**: T2 wins **all 5 pairs** on val/auroc
(mean Δ +0.011) and on val/cmap (mean Δ +0.031). Teacher-diversity benefit holds
despite the two teachers being LB-comparable (Δ +0.002 LB). T2 is the teacher to
keep iterating with offline; LB will say whether this transfers.

**Sampler axis (Uniform vs MixSamp, 4 paired tests)**: Uniform wins on random
init (+0.0033 to +0.0074 auroc); flat or marginally negative on warm-start
(−0.0009 to +0.0007). Consistent with the proposal's pool-size math —
Uniform's natural ~82% PL share converges with MixSamp 0.2/0.8 in the
warm-start cells.

**Init axis (warm-start vs random, 4 paired tests)**: warm-start helps 3/4
(strongest A1.kd→A3.kd +0.013 auroc); flat on T1+uniform. Confounded with
lr=5e-5 vs 5e-4 and 10ep vs 25ep.

**PL-exposure sanity (A1 50% vs A5 30%)**: A1 wins +0.018 (T1) / +0.007 (T2)
auroc. More PL exposure helps — Tucker's published 0.9/0.1 transplant would be
worse than A5.

### Phase G training dynamics

- All 10 ran to `max_epochs` (early stopping was disabled with `patience: 999`).
- Warm-start runs (A3/A4 × T1/T2) all peaked at epoch 7 — 10-epoch budget is
  generous; could be trimmed to 8 in future.
- A5.v5j peaked latest (ep 21) — under-PL'd config saturates focal slowly.
- Final-epoch `train/cls_loss` ~0.02 across all 10 (matches Tucker's published
  band); `train/dist_loss` plateaus at 0.002-0.003 (Tucker's floor). No NaN.

Per-ablation `metrics.json` at `experiments/033-own-teacher-pl/artifacts/training/{a1..a5}_{v5j,kd}/fold0/metrics.json` (backfilled 2026-05-27 after `save_metrics.py` was ported with `_common.yaml` deep-merge inheritance).

### Phase G checkpoints

- T2 (`*-kd`) ablations: `2026-exp033-a{N}-kd/fold0/last.ckpt` is the final
  (single launch, completed cleanly).
- T1 (`*-v5j`) ablations: `2026-exp033-a{N}-v5j/fold0/last-v1.ckpt` is the
  final (the relaunched-with-num_workers=8 run). `last.ckpt` in those dirs is
  the stale epoch ~4-5 snapshot from the cancelled `num_workers=0` job; do NOT
  use it.
- Best-by-`val/cmap` ckpts saved per-epoch via `save_top_k: -1` at
  `best-epochepoch=NN.ckpt`.

### Phase G compute

| Cell | Walltime | Runs | Total compute |
|---|---|---:|---|
| 25-ep ablations (A1/A2/A5 × T1/T2) | 6.5–7.4h each | 6 | ~41 GPU-h |
| 10-ep ablations (A3/A4 × T1/T2)    | 2.6–2.7h each | 4 | ~11 GPU-h |
| **All 10** (parallel overnight)    | wall ~7.5h    |    | **~52 GPU-h** |

Cancelled+relaunched `num_workers=0 → 8` fix (commits `ecc3ccb` + `87d662d`)
sunk an additional ~20 GPU-h on the T1 first wave; net effective compute ~72 GPU-h.
At `num_workers=8` per-epoch time on rtx6000 is ~20 min (~2× the proposal's
predicted 10 min, attributed to per-epoch worker startup + I/O dominance from
the 127K-file PL pool — acceptable, not a regression).

### Phase G D.5 end-to-end validation — ALL 10 PASS ✅

Same kernel-inference-path methodology as exp 032 (`validate_end_to_end.sh`).
Gate: `|train_auroc − e2e_auroc| ≤ 0.02`. All 10 PASS; max gap 0.0079; all 10
runs took 4–5s of inference on the fold-0 val SS pool (n_evaluable=29, val
windows=156). Used `last-v1.ckpt` for T1 ablations (the relaunched-with-
num_workers=8 run) and `last.ckpt` for T2 ablations.

| Slug | Train val/auroc | E2E val/auroc | Gap | LB | Δ vs 0.906 |
|------|----------------:|--------------:|----:|---:|---:|
| `2026-exp033-a4-kd` | 0.9477 | **0.9551** | +0.0074 | **0.899** | **−0.007** |
| `2026-exp033-a3-kd` | 0.9486 | **0.9540** | +0.0054 | (not submitted) | — |
| `2026-exp033-a2-kd` ⭐ | 0.9390 | 0.9403 | +0.0013 | **0.886** | **−0.020** |
| `2026-exp033-a4-v5j` | 0.9352 | 0.9399 | +0.0047 | **0.897** | **−0.009** |
| `2026-exp033-a3-v5j` | 0.9345 | 0.9387 | +0.0042 | (not submitted) | — |
| `2026-exp033-a1-kd` | 0.9357 | 0.9362 | +0.0005 | (not submitted) | — |
| `2026-exp033-a2-v5j` ⭐ | 0.9362 | 0.9332 | −0.0030 | (not submitted) | — |
| `2026-exp033-a1-v5j` | 0.9288 | 0.9282 | −0.0006 | (not submitted) | — |
| `2026-exp033-a5-kd` | 0.9285 | 0.9206 | −0.0079 | (not submitted) | — |
| `2026-exp033-a5-v5j` | 0.9113 | 0.9095 | −0.0018 | (not submitted) | — |

### Phase G LB results (2026-05-27 PM): REJECT

3 of the top-5 by D.5 e2e val/auroc submitted; all landed in the REJECT bucket
(per `proposal.md § Falsifiable`: CONFIRM ≥ +0.014, PARTIAL +0.007–0.013, WEAK
+0.001–0.006, REJECT ≤ 0). The remaining 7 ablations (2 Day-1 + 5 Day-2)
were NOT submitted on the trajectory.

**Paired Δ on the teacher axis (a4-kd vs a4-v5j)**: 0.899 vs 0.897 = +0.002 LB.
Inside cross-hardware noise. **Refutes the proposal's headline teacher-diversity
claim** (T2 won offline by +0.011 auroc, but does NOT win at LB on a paired
single-fold test).

**Paired Δ on the protocol-cell axis (a4-kd warm vs a2-kd random, same teacher)**:
0.899 vs 0.886 = **−0.013** (canonical Antoine < warm-start). Inverts the offline
ranking (a2-kd lower offline than a4-kd, but landed worse on LB too — both
regressing). Another offline-LB anti-correlation data point.

**Offline-LB anti-correlation data point #24**: the strongest offline cell
(a4-kd, e2e rank 1/10) landed in the middle on LB (rank 2 of 3 submitted); the
weakest of the 3 submitted by e2e (a4-v5j, e2e rank 4/10) was the best on LB
(rank 1 of 3). Same pattern as exp 032 fold 4 (+0.023 offline → −0.003 LB) and
22+ prior recipe data points.

### Phase G verdict

| Bucket | LB threshold (Δ vs 0.906) | Result |
|---|---|---|
| CONFIRM | ≥ +0.014 | ❌ |
| PARTIAL | +0.007 to +0.013 | ❌ |
| WEAK | +0.001 to +0.006 | ❌ |
| **REJECT** | **≤ 0** | **✅ all 3 cells (−0.007 / −0.020 / −0.009)** |

**The 1:1 own-teacher PL protocol does NOT lift LB on the distilled-sed-kd
base** despite all 10 offline ablations landing +0.003 to +0.041 val/auroc above
the baseline. Pivot per `proposal.md § Next-step plan`: Sydorskyi-yoink (XC
pretrain) as the primary remaining lever before the 2026-06-03 deadline (7 days
out).

Best LB this experiment: `distilled-sed-pl-a4-kd` = **0.899** (still REJECT
vs the distilled-sed-kd Phase D fold 0 baseline of 0.906).

See `experiments/033-own-teacher-pl/results.md` for the full per-ablation table
+ paired Δ across teacher/sampler/init/PL-exposure axes + decision-matrix
read-out.

## Checkpoint Base Path

All checkpoints under: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/`


---

# END OF BUNDLE
