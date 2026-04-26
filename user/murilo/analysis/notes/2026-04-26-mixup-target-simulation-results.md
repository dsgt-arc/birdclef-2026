# 2026-04-26 — MixUp target simulation results

Output of `mixup_target_simulation.py` (committed under `analysis/`).
Quantifies what the audit
([2026-04-26-mixup-implementation-audit.md](2026-04-26-mixup-implementation-audit.md))
showed structurally: **v5j is trained on uncertain (sub-1) target values most of
the time** because of its interpolated MixUp recipe.

This is a CPU-only simulation (no model, no audio loading). It reads train.csv +
train_soundscapes_labels.csv, builds multi-hot label vectors, then applies v5j's
exact MixUp math to 100,000 random pairs and records the distribution of the
resulting target values.

## Source

Run in venv `~/clef/birdclef-2026/.venv`:

```bash
cd user/murilo/analysis
python mixup_target_simulation.py
```

Result JSON: `data/mixup_sim/mixup_target_simulation.json`.

## Headline numbers

```
Base (pre-mixup) label distribution:
  Total samples: 37027 (35,549 focal + 1,478 SS segments)
  Mean positives per sample: 1.328
  Median positives per sample: 1
  Max positives per sample: 16

=== Post-mixup target value distribution (100K simulations) ===
  Total positive target values: 264,561
  Full positives (= 1.0) after theta cap:  54,262 (20.5% of positives)
  Sub-1 positives (in (0, 1.0)) after cap: 210,299 (79.5%)
  % samples with ≥ 1 full-positive after cap:    40.9%
  Mean positives per mixed sample post-cap:       2.646
  Mean full-positives per mixed sample post-cap:  0.543
```

## Histogram

```
Bin               Pre-cap     Post-cap
[0.00, 0.10)       26,226       26,226
[0.10, 0.20)       26,602       26,602
[0.20, 0.30)       26,006       26,006
[0.30, 0.40)       26,567       26,567
[0.40, 0.50)       26,369       26,369
[0.50, 0.60)       26,282       26,282
[0.60, 0.70)       26,549       26,549
[0.70, 0.80)       25,698       25,698
[0.80, 0.90)       26,545            0   <- snapped to 1.0
[0.90, 1.00]       27,717       54,262   <- 27,717 (1.0 + 0.9-0.999) + 26,545 snapped
```

The `[0.80, 0.90) → 0` post-cap is the theta=0.8 mechanism: anything ≥ 0.8 is
snapped to exactly 1.0, which lands in the last bin. Pre-cap the values are
distributed roughly uniformly across [0, 1.0) — consistent with `lam ~ U[0.5, 1.0]`
giving uniform spread on [0.5, 1.0] for y1-positives and on [0, 0.5] for
y2-positives.

## Sanity check on the math

For binary multi-hot y1, y2 ∈ {0, 1}^234, with `lam ~ U[0.5, 1.0]`:

- Class positive in BOTH y1 and y2: `target = lam·1 + (1-lam)·1 = 1.0` (always)
- Class positive in y1 ONLY: `target = lam·1 + (1-lam)·0 = lam ∈ [0.5, 1.0]`
- Class positive in y2 ONLY: `target = lam·0 + (1-lam)·1 = 1-lam ∈ [0, 0.5]`
- Class negative in both: `target = 0`

Theta cap (`target >= 0.8 → 1.0`) catches:
- All "both positive" cases (target = 1.0 already, no-op)
- y1-only positives where lam ∈ [0.8, 1.0] (~40% of y1-only cases)
- y2-only positives never (1-lam ≤ 0.5 < 0.8)

So the cap promotes y1-only positives from "interpolated" to "full" only when lam
is large; everything else stays sub-1.

## Compare to Sydorskyi's recipe

For the same pair (y1, y2), Sydorskyi's `target = max(y1, y2)` produces:

- Both positive: 1.0
- y1 only: 1.0
- y2 only: 1.0
- Neither: 0.0

100% of active classes at exactly 1.0. **0% sub-1 active classes.** Mean
full-positives per mixed sample = mean total positives per mixed sample.

## What this means

For each training step where MixUp fires (post warmup; v5j has p=1.0):

- v5j produces a target vector with ~2.6 non-zero values, of which only ~0.5 are 1.0
- 59% of training samples have NO active class at full presence (1.0)
- The model is asked to predict "this class is 0.7 present" or "this class is 0.3 present" most of the time

The soundscape test distribution looks nothing like this. A test 5s chunk with two
birds present has target `[..., 1, ..., 1, ...]` — both at 1.0. v5j has barely
ever seen that target shape during training.

## Connection to the H1 vs H2 split

The 04-25 finding was: v5j produces diffuse attention + collapsed logit_std on
val mixed_uncertainty crops. The interpretation was "the model is honestly
diffuse because the call is weak/absent in those crops." This simulation adds a
**second** structural reason for diffuse-on-hard-crops:

1. **Acoustic honesty** (04-25): low SNR audio → no clear call to localize.
2. **Trained honesty** (this simulation): the model has been explicitly trained
   that "active class with confidence ~0.5–0.7" is the normal state for 59% of
   training examples.

These aren't competing — they reinforce each other. But they have different
strategic implications:

- If diffuse attention is *purely* (1) → the lever is more soundscape
  information (H1, pseudo-distillation).
- If diffuse attention is partly (2) → fixing the MixUp recipe to Sydorskyi-faithful
  (`max(y1, y2)` targets) is a *separate* lever from H1/H2 — it changes the
  training distribution to match the test distribution without adding new data.

D1 (train-side attention on v5j + v6k, in flight) should help separate these.
v6k uses the same MixUp recipe as v5j (same theta=0.8, same interpolated
targets) — so if the diffuse pattern is *purely* a learned response to training
targets, both should show it equally on high-loss training samples. If v6k
shows a different pattern (peaked, even if wrong), that suggests the head is
capable of sharper attention than v5j's training recipe is teaching it to
produce.

## What we did NOT measure

- **Whether changing the MixUp recipe would actually move LB.** This simulation
  characterizes the training distribution; it doesn't predict downstream LB.
- **The interaction with v5j's `mixup_warmup_epochs=5`**. The first 5 epochs
  have no MixUp at all — model sees pure binary multi-hot during early training.
  Whether 15 epochs of interpolated targets after that "overrides" what was
  learned in the warmup phase is unclear.
- **What v6k's targets actually look like during training**. v6k uses the same
  mixup config, so the simulation applies. We did not separately simulate.

## Files

- Script: `user/murilo/analysis/mixup_target_simulation.py`
- Result JSON: `user/murilo/analysis/data/mixup_sim/mixup_target_simulation.json`
- Audit doc: `user/murilo/analysis/notes/2026-04-26-mixup-implementation-audit.md`
