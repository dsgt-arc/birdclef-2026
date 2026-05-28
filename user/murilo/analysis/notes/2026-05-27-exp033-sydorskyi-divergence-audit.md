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
