# 2026-04-26 — D1 train-side attention verdict (v5j vs v6k)

Output of [`train_side_analysis.py`](../../analysis/train_side_analysis.py)
on training-set inference outputs from `precompute_losses.py --data-split=train`
for both v5j (LSEHead, LB 0.906) and v6k (AttHead competition-standard, LB 0.888).

Plan:
[`2026-04-26-mixup-implementation-audit.md`](2026-04-26-mixup-implementation-audit.md)
→ [`2026-04-26-mixup-target-simulation-results.md`](2026-04-26-mixup-target-simulation-results.md)
→ this note (D1 verdict).

## TL;DR

Three findings, ranked by how surprising / load-bearing they are:

1. **v6k's learned attention is essentially uniform** (entropy 2.5 nats out of
   2.77 max; attn_max 0.13-0.15 vs 0.0625 uniform). The AttHead in our v6k
   checkpoint isn't doing what AttHead is supposed to do. It's not "peaked
   but wrong" or "peaked correctly on easy, diffuse on hard." It's flat
   everywhere.
2. **v5j and v6k agree on which training samples are hardest.** 84% of v5j's
   top-1% by training loss are also in v6k's top-1%; ZERO samples are in
   disjoint extremes (v5j top ∩ v6k bottom = 0). The failure pattern is
   data-driven, not head-architecture-driven.
3. **The 04-25 val finding reproduces on train.** For v5j, attention entropy
   is highest and logit std lowest on top-loss samples — same monotonic
   pattern we documented on val mixed_uncertainty crops on 04-25.

Verdict on H1 (information / data) vs H2 (modeling / head):

- **H1: concretely supported.** Both heads converge on the same hard samples;
  the "honest dispersal on hard crops" pattern holds head-agnostically.
- **H2 from v6k specifically: not testable.** v6k's AttHead doesn't produce
  meaningful attention at all (entropy ~uniform on EASY samples too). This
  is a property of v6k's training recipe (framewise-max-only inference →
  no gradient pressure on attention), not a refutation of H2 in general.
  To actually test H2 we'd need an AttHead that uses attention at inference
  (Salman-style blend), which we don't have.

So: the headline finding ("diffuse on hard crops") is now confirmed
generalizable across heads and across the train/val split. The right next
move is on the data side, not on the head — but only because the head
question hasn't been cleanly testable yet, not because we've ruled it out.

## Setup

Both models were run on the same 33,377 training samples (32,071 focal +
1,306 SS segments, with deterministic seed-fixed random 5s crops, no mixup
or augmentation in the forward pass — matches Salman's described post-hoc
inspection methodology).

Per-sample metrics:

- **`gt_attn_entropy`**: Shannon entropy of attention weights on the GT
  class. v5j uses LSE-implicit `softmax(α·z_t[gt])` (α=1.0). v6k uses
  learned `softmax(tanh(att_raw[gt]))`. Range [0, log(16)] = [0, 2.773]
  nats. Lower = more peaked.
- **`gt_attn_max`**: max attention weight on any single frame. 1/16 ≈
  0.0625 = uniform. Higher = more peaked.
- **`gt_logit_std`**: std of GT-class per-frame logits. Higher = head's
  per-frame value branch varies sharply ("this frame yes, that frame no").
- **`top1_majority_frac`**: fraction of frames where per-frame top-1
  matches clipwise top-1. Higher = stable, lower = wandering.

Buckets: top-1% by loss (~333), bottom-1% (~333), mid (~32,711).

## v5j (LSEHead) per-bucket

| bucket | n | mean_loss | gt_attn_entropy | gt_attn_max | gt_logit_std | top1_majority | gt_prob |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottom_1pct | 333 | 0.000 | **1.205** | 0.555 | **5.210** | 0.658 | 0.996 |
| mid | 32,711 | 0.008 | 1.200 | 0.567 | 3.471 | 0.434 | 0.714 |
| top_1pct | 333 | 0.083 | **1.359** | 0.529 | **2.852** | 0.366 | 0.493 |

**Pattern**: entropy slightly increases on hard samples (1.20 → 1.36, +0.15
nats), logit_std collapses (5.21 → 2.85, -45%), top1_majority_frac drops
(0.66 → 0.37). gt_prob falls from near-perfect (0.996) to coin-flip (0.493).

**Reading**: v5j's LSE-implicit attention is meaningfully peaked everywhere
(entropy ~1.2 vs 2.77 uniform). The peakedness varies modestly across
buckets, but the dominant difficulty signal is in the **logit magnitude**
(std goes from 5.21 → 2.85), not in the attention weights. This matches
the 04-25 val finding: the model "doesn't know" on hard samples mostly by
producing flat-magnitude per-frame logits, not by producing more diffuse
attention.

## v6k (AttHead) per-bucket

| bucket | n | mean_loss | gt_attn_entropy | gt_attn_max | gt_logit_std | top1_majority | gt_prob |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottom_1pct | 333 | 0.000 | **2.621** | 0.091 | 2.740 | 0.885 | 0.994 |
| mid | 32,711 | 0.010 | 2.486 | 0.153 | 2.795 | 0.463 | 0.717 |
| top_1pct | 333 | 0.083 | **2.541** | 0.141 | 2.276 | 0.386 | 0.498 |

**Pattern**: entropy is essentially constant across buckets (~2.5 nats; max
2.77). attn_max is 0.09-0.15 — slightly above uniform 0.0625 but close to
it. logit_std varies less than v5j (2.74-2.80 across easy/mid, 2.28 on
hard). top1_majority drops similarly to v5j.

**Reading — the surprise**: v6k's learned attention is **near-uniform on all
buckets, including the easy ones**. This is the opposite of what we expected.
AttHead has a learned `Conv1D → tanh → softmax` attention branch that
should be capable of localizing — but in this trained checkpoint it's
basically averaging frames.

Why? v6k's config has `blend_clip: 0.0, blend_frame: 1.0` — at **inference**
the attention output is **discarded entirely**, only framewise-max contributes.
During training the attention is supervised through the dual-loss
(`clipwise_weight=0.5, framewise_weight=0.5`), but the clipwise loss can be
satisfied by uniform attention as long as the framewise value branch
produces good per-frame logits. There's no gradient pressure to make the
attention peaked.

So v6k's attention is **architecturally present but functionally uniform**.
This isn't a property of AttHead in general — it's a property of v6k's
specific recipe (which de-emphasizes attention by design).

## Same-sample attention comparison (conditioned on v5j bucket)

For the SAME training samples, what does each head produce?

| bucket | n | entropy_v5j | entropy_v6k | Δ | logit_std_v5j | logit_std_v6k | Δ | top1_maj_v5j | top1_maj_v6k | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bottom_1pct | 333 | 1.205 | 2.476 | **+1.27** | 5.21 | 4.24 | -0.97 | 0.658 | 0.681 | +0.024 |
| mid | 32,711 | 1.200 | 2.487 | **+1.29** | 3.47 | 2.78 | -0.69 | 0.434 | 0.465 | +0.031 |
| top_1pct | 333 | 1.359 | 2.536 | **+1.18** | 2.85 | 2.34 | -0.52 | 0.366 | 0.392 | +0.026 |

**v6k's attention is consistently +1.2 nats more diffuse than v5j's** on
identical samples. The delta barely varies by bucket — v6k is uniformly
~1.2 nats more diffuse regardless of difficulty.

v6k's per-frame logit std is also lower than v5j's by ~0.5-1.0. So v6k
produces flatter outputs both in attention AND in per-frame value branches.

## Loss-bucket cross-tabulation

| | v6k bottom | v6k mid | v6k top | total |
|---|---:|---:|---:|---:|
| **v5j bottom** | 112 | 221 | **0** | 333 |
| **v5j mid** | 221 | 32,436 | 54 | 32,711 |
| **v5j top** | **0** | 54 | **279** | 333 |
| **total** | 333 | 32,711 | 333 | 33,377 |

**Strong agreement on extremes**:
- 84% of v5j top_1pct are in v6k top_1pct (279 / 333)
- 34% of v5j bottom_1pct are in v6k bottom_1pct (112 / 333)
- **0** samples in v5j top ∩ v6k bottom (or v5j bottom ∩ v6k top)

The bottom-1% overlap is weaker (34%) because many samples have near-zero
loss and the bucket boundaries are essentially tiebreakers. The top-1%
overlap is the load-bearing number: **84% of the hardest samples for v5j
are also among the hardest for v6k**.

The zero off-diagonals are the strongest signal: nothing v5j finds easy is
hard for v6k, and nothing v5j finds hard is easy for v6k. The two heads
aren't disagreeing about difficulty.

**Reading**: training-sample difficulty is dominated by sample properties,
not head architecture. This holds even though the two heads have very
different attention behaviors (v5j peaked, v6k uniform).

## Mapping to the H1 / H2 / H3 framework

I defined the answer regions upfront in
`2026-04-26-sydorskyi-2025-vs-v5j-comparison.md`. Mapping each:

### H1 (information / data) — concrete evidence supports it

Criteria from upfront:
- ✓ v5j and v6k both show diffuse / non-peaked attention on high-loss train
  (head-agnostic). v5j entropy goes up modestly on top_1pct; v6k is
  near-uniform everywhere; both heads' logit_std collapses on top_1pct.
- ✓ 84% of v5j top_1pct ∩ v6k top_1pct, zero disjoint extremes — strong
  failure agreement.
- ✓ 04-25 val pattern reproduces on train for v5j.

Read together: hard training samples are hard for both heads, in roughly
the same way. The model is honestly-confused on a specific population of
training samples (the same population both heads identify). The right
intervention is more soundscape-domain information.

### H2 (modeling / head) — not testable from this v6k

Criteria from upfront:
- ✗ v6k does NOT show "peaked but wrong" on high-loss train. It's near-uniform.
- ✗ v6k does NOT show "well-localized attention on actual call frames in
  low-loss train." It's near-uniform there too.

But neither of these criteria match because **v6k's attention is essentially
unused at inference** (`blend_clip=0.0, blend_frame=1.0`). The training
objective doesn't pressure the attention to become peaked. So we can't
distinguish "AttHead architecture is bad at localizing" from "v6k's recipe
doesn't pressure AttHead to localize."

To actually test H2, we'd need:
- An AttHead checkpoint where attention IS used at inference (`blend_clip > 0`)
- AND with a training recipe that gives attention real gradient pressure
- Salman's recipe is one example; we don't have a comparable model

**Verdict on H2 from v6k**: inconclusive. We've concretely shown that v6k's
attention is uniform, but that doesn't tell us whether a properly-trained
AttHead would localize.

### H3 (mixed) — partial

H1 has support; H2 isn't testable. So we have partial information leaning
toward H1.

## What's load-bearing about this verdict (and what isn't)

### Load-bearing

- **04-25 finding generalizes from val to train.** Same monotonic
  entropy/logit_std pattern. The "v5j is structurally honest on hard
  crops" mechanistic story carries over.
- **84% failure agreement**. This is a strong number. Hard training
  samples are sample-driven, not head-driven. If we want to fix them, we
  need to address the *samples* (information / supervision quality), not
  the head architecture.
- **v5j's logit_std collapse is the dominant difficulty signal.** Going
  from 5.21 (bottom) to 2.85 (top) is a 45% drop. The peakedness change
  (1.20 → 1.36) is modest. The model "doesn't know" on hard samples
  mostly by flattening per-frame magnitudes.

### Not load-bearing

- **v6k's specific attention pattern doesn't generalize to "all AttHead
  models."** Our v6k uses framewise-max-only inference. Salman's AttHead
  uses attention at inference. The two are different architectures
  practically, even if the modules look the same.
- **Whether a Sydorskyi-faithful AttHead would localize** is an open
  question that v6k can't answer. We need a different model.

## Combined picture so far (4 structural facts about v5j's training)

Stacking what we now know about v5j's training distribution and behavior:

1. **MixUp uses interpolated multi-label targets** (audit finding):
   v5j's `target = lam·y1 + (1-lam)·y2`. Sydorskyi's recipe uses
   `max(y1, y2)` instead. Different by design.
2. **80% of active training target values are sub-1** (simulation finding):
   on a typical mixup-active step, only 20% of target positives are at
   1.0. The model trains on uncertain target values most of the time.
3. **59% of training samples have NO class at full presence** in their
   target (simulation finding). The model is rarely supervised with "this
   class is fully here."
4. **NEW from D1: v5j's attention is meaningfully peaked but logit
   magnitude collapses on hard samples.** Both head agree on hard samples;
   84% top-1% overlap with v6k. Hard samples are sample-driven.

Together: v5j is being trained on a target distribution that DOESN'T match
the test distribution (multi-label binary), with 80% of supervision being
"partial presence" (sub-1). Then at inference, v5j produces *peaked*
per-frame logits that *vary in magnitude* by sample difficulty. The
attention layer says "I'm looking here" — but on hard samples, the answer
to "what's there" is uncertain (low logit std).

This is a coherent picture of "trained honesty + acoustic honesty" both
driving the diffuse-on-hard pattern. It doesn't say which one is the
bigger lever, just that both are real.

## What we know we can do next (ranked by EV-confidence)

Not picking a next experiment yet — still want to discuss with you. But
here's what the evidence supports, in rough order of how directly it ties
to what we've measured:

### Tier 1 — directly motivated by D1 + audit + simulation

**Sydorskyi-faithful MixUp on v5j (single variable)**: spec-mixup → wave-mixup,
interpolated targets → sum-clip (`max(y1,y2)` for binary), p=1.0 → p=0.5.
Tests whether the training-target-distribution mismatch is the lever.
Cheap (one retrain), single-variable, falsifiable. If it doesn't move LB,
we know the target distribution isn't the lever and we move on.

### Tier 2 — directly motivated by H1 reading

**Pseudo-distillation on 2026 unlabeled SS using Sydorskyi's recipe**: our
0.929 ensemble as teacher, soft labels with `primary_label_min_prob=0.5`
and `trim_min_prob=0.1`, 40% sampling, sum-MixUp fusion. Tests whether
adding soundscape-domain training data moves LB.

This is the "biggest published lever for 2025" but Sydorskyi's lift was
on top of in-domain pretrain. Without pretrain, the lift may be smaller.
Higher implementation cost than Tier 1.

### Tier 3 — would require infrastructure

**Faithful AttHead replication (Salman-style)**: AttHead with
`blend_clip > 0` (attention actually used at inference), trained with a
recipe that gives attention real gradient pressure. Would actually test
H2. But this is a multi-axis change (head + inference + training recipe);
probably worth doing only if Tier 1 and Tier 2 both fail.

## Caveats

- **v6k's near-uniform attention is itself a finding.** It tells us our
  AttHead implementation/recipe doesn't produce localized attention even
  when the head architecture is theoretically capable. Future AttHead
  experiments should verify the head IS localizing before assuming
  attention is the lever.
- **The "trained honesty" story is structurally consistent but not
  *proven* causal.** We've shown:
  (a) v5j is trained on uncertain targets,
  (b) v5j produces collapsed logit std on hard test samples.
  But (a) → (b) is a hypothesis. Could be reversed: the model is
  uncertain on hard samples regardless of training distribution.
- **All v5j and v6k inference here uses 5s deterministic crops with no
  augmentation.** This isn't quite what training-time saw (training has
  mixup, BG noise — wait, v5j has no BG noise, just mixup). The
  inference-time attention pattern might differ from gradient-time
  attention pattern. Hard to know without more invasive instrumentation.

## Files

- Inputs:
  - `data/losses_v5j_train.parquet` + `_perframe_logits.npy`
  - `data/losses_v6k_train.parquet` + `_perframe_logits.npy` + `_perframe_att.npy`
- Outputs:
  - `data/d1_train_side/v5j_train_enriched.parquet` (per-sample metrics + bucket)
  - `data/d1_train_side/v6k_train_enriched.parquet`
  - `data/d1_train_side/v5j_summary_by_bucket.csv`
  - `data/d1_train_side/v6k_summary_by_bucket.csv`
  - `data/d1_train_side/loss_bucket_crosstab.csv`
  - `data/d1_train_side/same_sample_attention_comparison.csv`
- Tooling: `train_side_analysis.py`, extended `precompute_losses.py` and
  `per_frame_analysis.py`.
- Pre-existing context: `2026-04-25-per-frame-attention-analysis.md`,
  `2026-04-26-mixup-implementation-audit.md`,
  `2026-04-26-mixup-target-simulation-results.md`,
  `2026-04-26-sydorskyi-2025-vs-v5j-comparison.md`.
