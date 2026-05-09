# 2026-04-28 — Next-experiment direction (post exp 030 close)

Strategic synthesis written immediately after closing exp 030 (mixup-recipe
null at single-fold resolution). Goal: capture the candidate directions and
the reasoning, so the next experiment scaffold has a documented basis rather
than picking a path under recency bias.

## Where we stand

| Model | LB | Notes |
|---|---:|---|
| Best ensemble | 0.929 | Perch ONNX + SED v5j-4fold MLP-only (exp 023) |
| Best single SED ensemble | 0.917 | v5j-4fold-pp2 (exp 021) |
| Best single SED | 0.906 | v5j-original (exp 019); reconfirmed 04-28 via v5j-resubmit |
| Closest single-model challenger | 0.901 | `effnetv2b0_sydorskyi` (exp 030) |
| Single-fold seed band on HGNet | ±0.013 | Measured 04-28 (was assumed ±0.005) |
| Target | 0.948 | Competition #1; gap from current best = 0.019 |

## What our analysis tells us about why we fail

**Strongest mechanistic claim (04-25 per-frame analysis)**: v5j is
*structurally honest*. On hard crops (low-SNR, weak/distant calls):
- GT-class attention entropy rises monotonically with failure severity
  (1.05 → 1.53 → 1.89 → 2.12 nats, vs uniform=2.77)
- GT-class logit std collapses (4.31 → 1.97 → 1.30)
- Same species produces qualitatively different head outputs across categories
  in the bimodal-species deep dive (entropy doubles/triples; logit std collapses
  3-6×)

Combined with the audio listening (mixed_uncertainty crops are noise-dominated
with weak/absent calls): the model is correctly being diffuse where there is
no signal. BCE+random-crop training penalizes that honesty (recording-level
label says species is present whether or not a 5s window contains it).

**Reproduced on the train side (04-26 D1 verdict)**: 84% top-1% by-loss overlap
between v5j (LSEHead) and v6k (AttHead). The hardest training samples are
the same samples for both heads. Failure pattern is data-driven, not
head-architecture-driven.

**Strategic implication that's been guiding our choices**: stop trying to
make the model more confident on hard crops (that teaches false positives).
Instead, use:
- *Information* levers (pretraining, external data, pseudo-distillation)
- *Inference* levers (PP/TTA)
- *Backbone* changes motivated by representation quality

This explains 22+ training-recipe failures including exp 030.

## What competitors at 0.93+ do

Captured from `user/murilo/notes/kaggle-discussion-best-single-model-lb.md`
(updated 04-28). Filtered to publicly-disclosed single-model recipes ≥0.93 LB.

| Person | LB | Key levers |
|---|---:|---|
| Tucker Arrants | 0.93+ | **Perch-PL student** + 5-fold + waveform augs + site/hour priors + Gaussian smooth (PP=+0.005, base=0.922) |
| Salman | 0.937 | EffNetV2-B0 + AttHead + 20s + CE+sigmoid-infer + waveform mixup + PP/TTA + **attention investigation on loss extremes** (+0.015 jump from 0.922) |
| Boredom | 0.938-0.939 | timm+SED + simple PP (+0.004); single-fold; no TTA |
| Ali Ozan | 0.941 | 2025-pretrained ckpt (filtered XC + bug fix) → 2026 finetune |
| EliKal | 0.909 | Perch KLD distillation (10ep) → finetune (12ep) |

## The Tucker puzzle

Tucker's recipe is **structurally identical** to our exp 015/025 attempts:
Perch as teacher → soft labels → student SED + BCE. We marked those "dead"
(cf. CONTEXT.md "Pseudo-labels are dead (2 approaches)"). Tucker hits 0.93+.

Possible deltas to investigate:
1. **Teacher choice** — Tucker uses Perch directly; we used a Perch + SED
   (or Perch + ConvNeXt + ProtoCLR) ensemble in exp 014/015. Different
   teacher quality → different label noise distribution.
2. **Fold structure** — Tucker: simple KFold on filenames + MultiLabelStrat
   on focal labels, 5-fold. We: single-fold student in exp 015 first round.
3. **Augmentation cascade** — Tucker: "waveform augmentations" (no specifics).
   We: tried various recipes, none matched.
4. **PP** — Tucker: site/hour priors + Gaussian smooth. We have v5j-4fold-pp2
   at 0.917 from a similar PP family.

Worth a focused diff before recommitting to Perch-PL.

## Three candidate directions

### (A) Tucker-style Perch-PL replication

Replicate the recipe with deliberate single-axis differences from exp 015/025
to identify what we missed.

| Step | Action |
|---|---|
| 1 | Read exp 015/025 results.md, isolate exact differences from Tucker's recipe |
| 2 | Single-axis test: rebuild PL pipeline with Perch-only as teacher (not ensemble) |
| 3 | 5-fold KFold + MultiLabelStrat + simple BCE student |
| 4 | Add Tucker's PP layer (site/hour priors + Gaussian smooth) at inference |

**Pros**: Tucker's lift is publicly proven (0.922 → 0.93+ with PP). Recipe
is simple. Most infrastructure exists (exp 015 had the student pipeline).

**Cons**: Pseudo-label engineering is fragile (label noise, teacher
miscalibration, distribution shift). We've been here twice already.

**Aligns with 04-25 finding**: ✓ (information lever, not training-recipe).

### (B) Pseudo-distillation on 2026 unlabeled SS

Use our 0.929 Perch+SED ensemble as teacher to label ~26K unlabeled SS
files; train SED on the result.

**Pros**:
- We have the teacher (0.929 ensemble) and unlabeled data already
- Sydorskyi 2025 reported **+0.029 from this** on top of in-domain pretrain
  — biggest published 2025 single-experiment lift
- Closes the focal→SS domain gap directly (operates on SS data, not focal)

**Cons**:
- Soundscape pseudo-labels are noisier than focal (ambient noise dominates)
- D.M. on the discussion thread: "I tried lots of mixing schemes with the
  unlabeled soundscapes but that didn't transfer to the LB for me"

**Aligns with 04-25 finding**: ✓ Same as (A).

### (C) Salman's attention-investigation move

Salman: "investigating attention values on highest/lowest loss examples
during training, and now it's much better" → **+0.015 jump (0.922 → 0.937)**.

Tension with our 04-25 finding: we concluded attention IS doing the right
thing (correctly diffuse on hard crops). If Salman's lift is real, the
actionable interpretation is:
- Find cases where attention is *incorrectly* diffuse on EASY crops
  (model failing to localize where it should)
- Or *incorrectly* peaked on noise on hard crops
- Then fix those specific failure modes

**Pros**:
- Cheap to probe (analysis on existing v5j ckpt; no training first)
- Engages with existing 04-25 tooling (`per_frame_analysis.py`,
  `train_side_analysis.py`)
- If we find a specific failure pattern, the fix has mechanistic justification

**Cons**:
- Salman is vague — "investigating attention values" doesn't say what action
  followed the investigation
- We may chase the wrong pattern
- Even if we find one, the fix might be non-trivial (regularizer, attention
  loss term, hard-negative mining)

**Tension with 04-25 finding**: 04-25 focused on `mixed_uncertainty` and
`gt_invisible` (hard cases). Salman framed his investigation around
**highest AND lowest loss** examples — easy-and-correct, easy-but-confident-wrong,
and confident-correct may be where the lever lives. We didn't fully look
at those buckets.

## User's noted idea (worth flagging, not yet scoped)

**Waveform mixup (Sydorskyi recipe from exp 030) + AttHead as a single
model**. Combines:
- The mixup recipe variant we built in exp 030 (`mixup_mode="average_max"`)
- An attention-based head (Salman's preference; we've tested AttHead in
  v6k/v7 with mixed results)

Status: noted as a possible 4th candidate. Reasonable to scope after the
(A)/(B)/(C) decision is made. Doesn't currently have a mechanistic argument
beyond "Salman uses both and gets 0.937" — but could be a clean single
variable test on top of whichever direction we pick.

Caveat: prior AttHead tests on our codebase regressed on LB (v6k 0.899 vs
v5j 0.906; v7 manifold-mixup 0.899). The 04-26 D1 verdict found v6k's
attention is essentially uniform because its inference recipe
(`blend_clip=0.0`, framewise-max only) leaves attention with no gradient
pressure. Any AttHead retest needs to fix that first.

## Suggested sequence

1. **(C)-style probe first** (1-2 hours, no training): re-run
   `per_frame_analysis.py` on v5j across ALL triage buckets, not just hard
   ones. Compare attention behavior on top 1% by training loss vs bottom 1%
   (Salman's framing). We have the train-side tooling from exp 026.

2. **Decision point**: if we find a clear actionable failure pattern, scope
   a (C)-style targeted training-recipe change with mechanistic justification.
   If not, default to (A) — Tucker-style Perch-PL replication with a
   single-axis comparison vs exp 015 first to find what we missed.

3. **Defer (B)** unless (A) hits a pseudo-label-quality wall — same
   structural lever as (A) but on noisier data.

4. **Defer the user's wave-mixup + AttHead idea** to after (A)/(B)/(C)
   resolves; potentially as a Phase B of whichever direction wins.

## Why this sequence

- **Cheapest first**: (C)-style probe is pure analysis on existing artifacts.
  No GPU spend. If it surfaces a pattern, we've learned something concrete
  before committing to a training run.
- **Hedges against repeated failure**: (A) and (B) both have "we tried this
  and failed" baggage. The 04-28 update gave new evidence (Tucker 0.93+ +
  Salman "modeling matters") suggesting the gap is closer to the head than
  the data. A short diagnostic before committing GPU budget protects against
  running into the same failure mode for the third time.
- **Single-fold seed band realization**: with ±0.013 noise, any
  single-experiment LB Δ below ±0.013 is inside noise. This argues for
  experiments that produce *bigger* expected lifts (Tucker +0.024 from
  Perch-PL+5-fold+PP, Sydorskyi +0.029 from pseudo-distillation, Salman
  +0.015 from attention investigation) over experiments where the predicted
  lift is comparable to noise.

## Open question for the next session

Tucker's "Perch distilled B0 + BCE" structurally equals our exp 015/025
recipe. Tucker hits 0.93+; we marked it dead. Either we have an
implementation flaw (likely) or our earlier conclusion was premature. The
exp 015/025 results.md is the right place to start for the (A) path —
identify what's load-bearing in Tucker's recipe that we didn't have.

---

## ADDENDUM (2026-04-28 PM): The PL vs KD distinction

Conversation surfaced a critical conflation. "Perch as teacher" can mean
three structurally-different things. We've only tried one of them.

| Concept | What it does | Loss structure | What WE did |
|---|---|---|---|
| **Pseudo-labeling on unlabeled data** | Teacher generates labels for UNlabeled audio; expand training set | BCE on PLs (treated as new ground truth) | exp 014/015/025 |
| **Knowledge distillation on labeled data** | Teacher provides soft targets alongside ground truth on already-labeled data; auxiliary loss during training | BCE on hard labels + KLD/MSE on teacher logits (joint) | **never** |
| **Teacher relabeling on labeled data** | Teacher generates per-crop predictions to REPLACE the recording-level label | BCE on teacher's crop-level predictions | **never** |

Tucker's "Perch distilled B0 + BCE" is most likely **#2 (joint KD)**, based
on EliKal's thread quote: "10 epochs of distillation with KLD loss". That's
structurally different from what we did. Our previous "Pseudo-labels are
dead" conclusion is about **#1 only** — categories #2 and #3 remain
unexplored.

User confirmed (2026-04-28): exp 015/025 only ran category #1 on unlabeled
SS. Focal recordings and labeled SS were never relabeled or distilled-into.

### Why the distinction matters for the 04-25 mechanism

Each strategy engages the BCE+random-crop label-noise mechanism differently:

| Strategy | Engages 04-25 mechanism? | Why |
|---|---|---|
| #1 PL on unlabeled (what we did) | **No.** | Adds more labels to fit; PLs from a model that ALSO trained on noisy crops inherit the same noise. Possibly amplifies failure. |
| #2 KD on labeled | **Partially.** | Teacher's soft labels carry calibrated uncertainty — Perch saying "0.3 for this species in this crop" captures "weak/maybe present," which BCE on hard recording-level labels does not. |
| #3 Crop-level relabeling | **Directly.** | Replaces noisy recording-level hard labels with Perch's per-crop verdict. Most aligned with our analysis. |

### Why our exp 015/025 likely failed (revised hypothesis)

1. We took unlabeled SS audio (heavy noise, often weak/no calls in many crops)
2. The teacher ensemble made predictions that were themselves noisy on those clips
3. Student trained on focal + labeled SS + noisy unlabeled-SS PLs
4. Net negative: PL volume hurt more than the noisy labels helped

We were trying to address the data/information bottleneck (more samples =
more signal) but the new "samples" came with bad labels. We did not address
the **crop-level label noise mechanism** the 04-25 finding actually
identified.

### Refined candidate experiments (replaces (A)/(B)/(C))

After post-mortem on exp 014/015/025, the highest-information-value next
experiment should be one of:

#### (α) KD-on-labeled (EliKal-style replication)

Train SED with combined loss: BCE on hard labels + KLD against Perch logits,
jointly. Closest to EliKal's published 0.909 single-model recipe.
Tucker's recipe is likely a variant of this with more PP.

**Pros**: published recipes (EliKal, likely Tucker), engages the 04-25
mechanism partially via Perch's soft uncertainty.

**Cons**: requires Perch logits/embeddings cached for every training crop;
implementation engineering on top of existing SED training pipeline.

#### (β) Crop-level relabeling (most aligned with 04-25)

Use Perch to generate per-crop soft labels for our existing labeled focal +
SS data. Replace recording-level hard labels with Perch's per-crop verdict.
Train SED with BCE on these refined labels.

**Pros**: most directly addresses the 04-25 label-noise mechanism. Sydorskyi
2025 calls this "soft pseudo-distillation" in his writeup.

**Cons**: weaker replication target — no public recipe specifies this
exactly. Risk of label drift if Perch is itself noisy on certain clips.

#### (γ) Hybrid: filtered PL + KD

Continue (#1)-style PL on unlabeled SS but with stricter quality filtering
(only PLs with teacher confidence > 0.7) + add KD on labeled data. Less
aggressive change.

**Pros**: incremental over what we tried; lower risk.

**Cons**: smaller expected lift; doesn't fully engage 04-25 mechanism.

### Phase 1 — focused post-mortem (still recommended next step, no GPU)

Three reads + one write:

1. `user/murilo/experiments/014-*/results.md` (PL generation pipeline)
2. `user/murilo/experiments/015-*/results.md` (student training)
3. `user/murilo/experiments/025-iter-pseudo-labeling/results.md` (iterative)
4. `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md`
   (Sydorskyi's "soft pseudo-distillation" definition — formal source)

Output: a comparison note (`2026-04-29-pl-kd-postmortem.md` or similar)
covering:
- Exact teacher composition, training data, loss, fold structure for exp
  014/015/025
- Sydorskyi's formal soft-pseudo-distillation definition
- EliKal's KLD-distillation recipe from the thread
- Tucker's recipe inferred from public posts
- Ranked list of candidate experiments with mechanistic justification

After Phase 1, scope a single experiment as Phase 2 (training run) based
on what the post-mortem reveals.

### What the user does NOT want to do (explicitly stated 2026-04-28)

- Skip the post-mortem and re-run a PL variation on hope. We've done that
  twice. Without a clear hypothesis about what changed, we'd be running
  experiment #3 on the same failed pattern.
- Pivot to a fundamentally new approach (SSL from scratch, etc.) — covered
  in earlier discussion: BirdCLEF data is too small for SSL from scratch,
  and Perch already provides the SSL representations we need.
