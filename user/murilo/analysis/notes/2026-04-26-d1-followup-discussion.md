# 2026-04-26 — D1 follow-up discussion: three open questions

After reviewing the [D1 verdict](2026-04-26-d1-train-side-attention-verdict.md),
three questions surfaced that matter for the next move. Capturing them here
because the answers shape what we do next, and some of the answers refine
or correct what's in the verdict note itself.

## Q1 — v6k uses attention during training but discards it at inference. Is that a bad idea?

**Short answer**: yes, structurally weird. It's also what Nikita 1st-place
2025 did. So it's defensible but worth questioning.

### The mechanics

v6k config (`hgnet_b0_atthead_v6k.yaml`):

```yaml
training:
  clipwise_weight: 0.5     # half the loss on clipwise (uses attention)
  framewise_weight: 0.5    # half the loss on framewise-max (no attention)
  blend_clip: 0.0          # ⟵ at inference, attention contribution = 0
  blend_frame: 1.0         # ⟵ at inference, framewise-max contribution = 100%
```

Training:
- Forward pass produces both clipwise output (`sum(att_weights · sigmoid(frame_logits))`)
  and framewise output (`max(frame_logits)`).
- Both contribute to loss with equal weight.
- Attention has gradient flow because the clipwise term depends on it.

Inference:
- Output = `1.0 · sigmoid(framewise_max)` + `0.0 · clipwise_pred` = framewise-max only.
- Attention is computed and immediately discarded.

### Why anyone would do this

The defense (per the v6k config comment, citing Nikita 1st place 2025):

1. **Attention as a regularizer / auxiliary task**: forcing the model to
   produce coherent attention-weighted clipwise predictions is a training
   signal that improves the per-frame feature quality, even if attention
   is then thrown away at inference.
2. **Framewise-max is simpler and more robust at inference**: no learned
   attention failure modes, no calibration risk, just `max(per-frame logits)`.
3. **Nikita won with this recipe**, so it's empirically validated for at
   least one top BirdCLEF 2025 solution.

### Why it's still weird (and matters for our interpretation)

Two real concerns:

1. **It makes v6k a structurally bad H2 test.** We trained an AttHead model
   to ask "does AttHead-style attention diagnose better than LSE-implicit?"
   But v6k's attention is *architecturally unused at inference*. So
   whatever attention it's producing has no inference cost or benefit, and
   the gradient signal it gets during training is weak (clipwise loss can
   be satisfied by uniform attention as long as framewise logits are
   good). Result: v6k's attention is functionally uniform — entropy ~2.5
   nats out of 2.77 max. We can't read this as "AttHead can't localize" —
   only as "v6k's specific recipe doesn't pressure AttHead to localize."

2. **Different top-3 finishers do different things here.** Nikita
   (1st 2025): framewise-max only at inference. Sydorskyi (2nd 2025):
   attention IS the inference output (`clipwise_pred = sum(sigmoid(value)
   · softmax(tanh(att), dim=time))`). Salman (13th 2025, single 0.937):
   "I definitely rely on some post processing / TTA" — implies attention
   is part of his stack. Nikita's recipe is one valid path; it's not the
   only one and adopting it without the rest of his pipeline (iterative
   PL across 4 rounds, SoftAUCLoss, separate insect pipeline) may be
   missing the point.

### What this changes about the verdict

The verdict note already calls this out as a caveat. But upgrading its
prominence: **v6k's near-uniform attention is at least partly an artifact
of the inference recipe choice, not a property of AttHead in general.**
If we want to actually test H2 ("does proper AttHead-style learned
attention help?") we need a different model — one with `blend_clip > 0`
where attention is structurally part of the inference output.

This doesn't invalidate the H1 evidence (84% failure agreement is still
load-bearing — that's about which samples are hard, not about attention
patterns). But it does mean the H2 reading should be "deferred until we
have a comparable model" rather than "weakly refuted."

### Worth doing as a tiny side experiment?

A "v6k-with-attention-at-inference" variant: same checkpoint, just change
inference to `blend_clip=1.0, blend_frame=0.0` and re-submit to LB. Zero
training cost. Tells us: did v6k's attention learn anything useful that
framewise-max-only inference is throwing away? Probably won't move LB
much (the attention is uniform), but it's a free check.

I don't think this is the highest-EV next move, but it's a sanity check
worth flagging.

## Q2 — Should we revise the training strategy? Silent class? Longer windows?

The user proposed two interventions in response to the
"80% of training target values are sub-1" finding from the simulation.
Walking through each:

### Option A: Add a "no signal" / silent class (235th class)

This is exp 030 candidate from CONTEXT.md, motivated by the 04-25
per-frame finding. The `2026-04-25-per-frame-attention-analysis.md` note
flagged it explicitly:

> Add a 235th class for "no signal in this 5s window," supervise it during
> training, and use it at inference to gate predictions. Matches what the
> model is already learning to do; it would just give it permission to
> say so explicitly.

**How it relates to the simulation finding**: partially. The simulation
shows that 59% of training samples have NO class at full presence after
mixup. That's a target-distribution problem. A silent class addresses a
*different* problem: training samples where the audio genuinely has no
species (true silent crops, like the rank02 roahaw clip from
`2026-04-25-audio-validation-low-snr-crops.md`). These overlap somewhat
but aren't the same population.

A silent class would help if:
- Many training crops genuinely contain no species (because of random
  cropping into long, mostly-silent recordings)
- Adding "silent" as a target lets the model say "I don't hear anything"
  without being penalized

The 04-25 note already flagged the implementation challenge: where do we
get supervision data for "silent"?
- Labeled SS rarely has truly silent segments (most have at least one
  species annotated).
- Synthetic silence is out-of-distribution from real soundscapes.
- Self-bootstrap from unlabeled SS via low-RMS filter is most pragmatic
  but couples to v5j's existing predictions.

So a silent class is plausible but has data-acquisition complexity. Worth
considering, but not the cheapest experiment.

### Option B: Longer windows (10-20s training, like Salman)

The intuition: longer windows make it more likely the bird call is
*somewhere* in the window, reducing the "label says present, audio is
empty" mismatch.

**We've tested this. It regressed every time.**

| Slug | Change | LB | Δ vs v5j |
|---|---|---:|---:|
| v5i (exp 019) | 5s → 10s window | 0.880 | −0.026 |
| v9a (exp 026) | 10s window AttHead | 0.868 | −0.038 |
| v9b (exp 026) | 15s window AttHead | 0.877 | −0.029 |
| v9c (exp 026) | 20s window AttHead | 0.878 | −0.028 |

Salman trains at 20s and gets 0.937. We train at 20s and lose 0.026-0.038.
The window length isn't the lever in our pipeline — at least, not on its
own. Salman's recipe has many other things going on (CE loss with
sigmoid inference, BG noise injection, EffNetV2-B0, attention-at-inference)
that may interact with window length.

**Note also**: longer windows make the "fraction of window with call"
problem WORSE, not better. A 5s window with a 0.5s bird call has the
call occupy 10% of the window. A 20s window with the same 0.5s call has
the call occupy 2.5%. The model's task gets harder per-frame even though
the recording-level supervision stays the same.

**So**: longer windows have been ruled out as a single-variable change.
If we revisit it, it would need to be as part of a multi-axis change
that includes the things Salman does differently (loss, BG noise,
inference recipe).

### What the simulation finding actually points to

The 80%/59% sub-1 numbers are about **target combination math** during
mixup, not about input audio content. The fix is in the target
combination, not in how we sample audio. Specifically:

- Sydorskyi: `target = max(y1, y2)` for binary multi-hot → 100% of active
  positives are at 1.0
- Our v5j: `target = lam·y1 + (1-lam)·y2` → most active positives are
  sub-1

Changing the *target combination* (Sydorskyi-faithful mixup) directly
addresses the simulation finding. Changing the *crop length* doesn't.
That makes Q3's proposal the right one.

## Q3 — Sydorskyi-faithful MixUp on v5j as the next experiment

**Yes — this is the right next experiment.** Most directly motivated by
what we've actually measured.

### Single-variable change vs v5j

The full Sydorskyi recipe diverges from v5j on multiple axes (per the
[comparison note](2026-04-26-sydorskyi-2025-vs-v5j-comparison.md)):

- **Domain**: spec → wave
- **Lambda distribution**: Beta(1,1) max-flipped to [0.5,1] → deterministic 0.5
- **Wave combination**: interpolation → `(w1+w2)/2` + peak-normalize
- **Target combination**: interpolation → `max(y1, y2)` (sum-clip for binary)
- **Theta cap**: 0.8 → none
- **Probability**: 1.0 → 0.5

The strict definition of "single variable" is unclear here — the whole
recipe is the variable. Two ways to scope:

**Option 3a: Full Sydorskyi-faithful recipe in one config.**
Test: does the entire recipe move LB? If yes, we know the target-
distribution lever (or one of several stacked changes) helps. If no, we
know the entire recipe is dead and don't need to ablate further. Most
informative as a yes/no signal.

**Option 3b: Just change target combination, keep everything else (spec-domain mixup, Beta lambda, theta cap).**
Test: is the `max(y1, y2)` vs `lam·y1 + (1-lam)·y2` target distinction
the lever? Cleanest single-variable test of the audit finding.

I lean **3a** for the first run. Reasoning:

- Sydorskyi's stack is internally coherent — wave-domain + max-targets +
  deterministic 0.5 lambda + peak-normalize all make sense together. The
  internal coherence might be what makes it work. Testing pieces in
  isolation could miss the interaction.
- We've already tested partial-Sydorskyi recipes 3 times (v4j, v6b, v10b)
  and all regressed. The full faithful version is what we haven't tested.
- If 3a fails, we then run 3b to determine if the target combination is
  the issue or it's something else.

### What I'd predict ahead of time (for falsifiability)

Honest predictions, in case we want to grade results against them later:

- **Optimistic**: full Sydorskyi mixup on v5j → 0.910-0.915 LB. The
  target-distribution mismatch was a real lever; closing it gets a
  meaningful but modest gain. Sydorskyi-style stack synergizes with the
  rest of v5j's recipe.
- **Pessimistic**: full Sydorskyi mixup on v5j → 0.895-0.905 LB
  (regression to flat). The mixup details aren't the lever; the audit
  finding was true but not load-bearing for LB.
- **Neutral**: Sydorskyi mixup on v5j → 0.901-0.908 LB (within seed noise).
  Inconclusive — we can't tell if the change matters.

I genuinely don't know which is most likely. The audit finding is
*concrete and falsifiable*, and the simulation quantifies a real
distribution mismatch — but neither *guarantees* LB will move.

### What we don't change in this experiment

To preserve "single experiment" discipline:
- Same v5j HGNetV2-B0 backbone, LSEHead, BCE loss
- Same mel pipeline (NNAudio + AmplitudeToDB)
- Same OneCycleLR schedule (lr=5e-4, 20 epochs)
- Same 90/10 split, same data
- Same inference (TTA + smoothing + TopN + priors)
- Only mixup config changes

We don't add label smoothing, BG noise, focal loss, or any other
Sydorskyi feature in this experiment. Those are separate axes for separate
tests.

### Implementation cost

Looking at `sed_finetune.py:_apply_mixup`, this needs a new
`mixup_mode="sydorskyi"` (or `"average_max"`) branch:

```python
elif self.hparams.mixup_mode == "average_max":
    indices = torch.randperm(x.size(0), device=x.device)
    x = (x + x[indices]) / 2
    labels = torch.maximum(labels, labels[indices])
    # peak-normalize after mixing (per Sydorskyi's late_normalize=True)
    if self.hparams.waveform_mixup:  # only applies on waveform domain
        x = x / (x.abs().amax(dim=-1, keepdim=True) + 1e-6)
```

Plus config: `raw_waveform_mixup: true`, `mixup_mode: "average_max"`,
`mixup_prob: 0.5`, `mixup_warmup_epochs: 0`, `mixup_theta: 1.0`.

~20 lines of code change. Single config file. No new infrastructure.

### Total experiment cost

- Code change: 1-2 hours
- Training: ~2-3 hours single fold (matches v5j's ~2.5h baseline)
- ONNX export + LB submission: ~1 hour
- **Total: ~half a day end-to-end**

Cheap relative to the information value.

## My honest synthesis

Three things I'd commit to:

1. **The Sydorskyi-faithful MixUp experiment (Q3) is the right next step.**
   It directly tests the most concrete finding we have (mixup audit +
   simulation), it's cheap, and it's falsifiable. Even a clean negative
   tells us something useful: target-distribution mismatch isn't the
   lever, refocus elsewhere.

2. **The v6k attention-only-during-training caveat (Q1) is more important
   than the verdict note made it.** v6k can't test H2 cleanly. Future
   AttHead experiments should verify the head is structurally doing
   something at inference before assuming "head architecture" is what's
   being tested.

3. **The longer-window proposal (Q2) is ruled out as a single-variable
   change** by our prior tests (v5i / v9a-c all regressed). Silent class
   is plausible but has data-acquisition complexity and addresses a
   different problem than the simulation finding. Set aside for now.

### What I'm NOT confident about

- Whether the Sydorskyi-faithful recipe will actually move LB. The
  predictions above reflect honest uncertainty, not a confident take.
  We've had 22 training-recipe regressions; the prior is that recipes
  don't help. The audit + simulation give us the cleanest mechanistic
  argument for why THIS recipe might be different, but mechanistic
  arguments have failed before.
- Whether the silent class would help. Plausibly yes, but it's a
  different problem and has implementation cost. Not advocating for it
  yet.

### What I want to push back on (from my earlier framing)

In the verdict note I called H2 "weak / not supported from v6k." Now I
think that's slightly mis-stated. The honest version: **H2 is not
testable from v6k** because v6k's attention is structurally unused at
inference. We have neither evidence for nor against H2 in our codebase.
Testing H2 properly would require a different model (e.g., a v6k variant
with `blend_clip > 0`), which is an open question we haven't budgeted for.

I'll update the verdict note to make this clearer.

## Concrete next-step proposal

Given the discussion, my recommendation:

1. **Run the Sydorskyi-faithful MixUp experiment** (Q3) as exp 030 or
   similar. Half-day cost. Tests the most concrete finding.
2. **In parallel**: run the cheap v6k inference-side check from Q1 (just
   change blend_clip/blend_frame at inference, re-submit to LB). Almost
   no cost. Tells us if v6k's attention learned anything inference-relevant.
3. **Defer** the silent class and any longer-window variants until 1+2
   produce a result.

If both 1 and 2 fail to move LB, then we're back to "no obvious next
training-recipe lever" and have to choose between:
- Pseudo-distillation (information lever, infrastructure-heavy)
- Silent class (information lever via supervision, data-acquisition-heavy)
- Faithful Salman replication (multi-axis, expensive)
- Something we haven't thought of yet

Let me know which direction you want to push. The Sydorskyi-faithful
MixUp test seems clearly worth doing first — it's the cheapest test that
directly engages the audit + simulation findings — but I want to confirm
before scaffolding an experiment.

## Caveats

- This note is partially overlapping with the verdict note. Not aiming
  for non-redundancy; aiming for capturing the *discussion* that
  followed the verdict, in addition to the verdict itself. Future-me
  can read either.
- Predictions in Q3 are honest guesses. Don't grade against them too
  heavily — the prior is that training-recipe changes don't help
  (22 failures, 0 wins), and one mechanistic argument doesn't break
  that prior cleanly.

## Files

- Verdict: [`2026-04-26-d1-train-side-attention-verdict.md`](2026-04-26-d1-train-side-attention-verdict.md)
- Audit: [`2026-04-26-mixup-implementation-audit.md`](2026-04-26-mixup-implementation-audit.md)
- Simulation: [`2026-04-26-mixup-target-simulation-results.md`](2026-04-26-mixup-target-simulation-results.md)
- Comparison vs Sydorskyi: [`2026-04-26-sydorskyi-2025-vs-v5j-comparison.md`](2026-04-26-sydorskyi-2025-vs-v5j-comparison.md)
- 04-25 per-frame: [`2026-04-25-per-frame-attention-analysis.md`](2026-04-25-per-frame-attention-analysis.md)
