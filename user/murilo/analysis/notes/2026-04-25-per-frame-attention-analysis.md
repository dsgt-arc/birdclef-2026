# 2026-04-25 — per-frame analysis on v5j (LSEHead)

Context: paused [exp 028 (BG noise mixing)](../../experiments/028-background-noise-mixing/proposal.md)
mid-implementation after Murilo expressed low confidence in the augmentation
direction (21+ recipe changes have all failed; Salman himself says external
BG noise doesn't move his LB). Pivoted to a model-side investigation more
faithful to Salman's documented methodology — *"investigating attention
values on examples with highest loss during training, and minimum loss
during training, and now it's much better."*

This note captures the methodology and the questions we're trying to
answer. Findings are appended when the data is available.

## What this is, and isn't

**Is**: a structural look at v5j's per-frame logit distributions across
triage categories. The LSEHead pools per-frame logits via
`(1/α) · log(Σ exp(α · z_t))`. The implicit per-frame weighting derived
from this is `softmax(α · z_t)` — that's the analogue of "attention
values" for an LSE-pooled head.

**Isn't**: an inspection of learned attention weights. Salman uses AttHead
(`softmax(tanh(W·x))`); his "attention values" are *learned parameters*.
LSEHead has no such parameters. The diagnostic here is the LSEHead-equivalent,
not a replica of his methodology.

## Why per-frame logits, not gradient saliency

Earlier we attempted gradient-saliency analysis (`attribution.py`). Two
problems killed that path as a primary diagnostic:

1. **Sigmoid saturation**: ~65% of val samples have `gt_prob > 0.9` or
   `< 0.05`, where `∂(prob) / ∂(input) ≈ 0`. The saliency map is noise
   on those samples regardless of internal model behavior.
2. **Saliency lives in input (mel) space**, but the question we want to
   ask is about *the head's behavior given the backbone's features* —
   that's per-frame logit space, not mel space.

Per-frame logits sidestep both problems. They're the head's output
directly, no gradient required, no saturation issue.

## The five questions

1. **Attention diffuseness across triage categories.** Is the LSE-implicit
   attention on the GT class systematically more diffuse on
   `mixed_uncertainty` than on `correct_confident`? Equivalently: does the
   GT-class per-frame logit distribution flatten on hard cases?
   Quantified by Shannon entropy of `softmax(α · z_g_t)` per sample.

2. **Frame-level top-1 wandering.** For each sample, look at the per-frame
   top-1 species (across all 234 classes). On failures, does the top-1
   stay locked on a wrong species (committed misidentification) or wander
   across frames (model can't decide)? Quantified by `n_distinct_top1`
   and `fraction of frames where frame-top-1 == clipwise-top-1`.

3. **Peak-frame agreement with energy** (deferred). For samples where the
   call is acoustically present (saliency-informative subset of
   mixed_uncertainty), does the GT-class peak frame coincide with the
   highest-energy frame in the mel? Requires loading mel — deferred to
   a follow-up if questions 1–2 surface a pattern worth grounding in
   audio.

4. **Bimodal-species deep dive.** Take species we already inspected
   via audio (`roahaw`, `coffal1`, `socfly1`, etc.) and compare per-frame
   stats between their `correct_confident` crops and `mixed_uncertainty`
   crops. Same species, different SNR conditions (per the audio
   listening). If the head's per-frame behavior differs between the two
   sets, the failure is structural at the head level. If it's the same,
   the failure is upstream (input or backbone).

5. **Logit-magnitude variability.** `gt_logit_std` across frames per
   sample. Low std = uniform per-frame logits (head sees the call
   "everywhere or nowhere"); high std = varies sharply (head localizes,
   though possibly to wrong frames). Cross-reference with category and
   with #1.

## Implementation

**Code added:**
- `analysis/precompute_losses.py` — extended to dump per-frame logits
  alongside clipwise. Output: `losses_v5j_perframe_logits.npy` with shape
  `(N, n_classes, T)`, stored as float16 to save space.
- `analysis/per_frame_analysis.py` — operates on the per-frame logits
  + triage parquet. Computes per-sample metrics, aggregates by triage
  category, runs the bimodal-species deep dive.

**To run:**
```bash
# 1. Re-run the GPU pre-compute pass (now also saves per-frame logits)
cd user/murilo/analysis
bash sbatch/run.sh

# 2. After it completes (~3 min), run the analysis on CPU
python per_frame_analysis.py
# Output: data/perframe/triage_v5j_perframe.parquet
#         data/perframe/perframe_summary_by_category.csv
#         data/perframe/perframe_bimodal_species_deepdive.csv
```

## Findings

_(Pending — appended after the analysis runs.)_

## What this enables next

If the analysis surfaces a clear pattern — for example, "attention is
uniformly diffuse on `mixed_uncertainty` regardless of whether the call
is present in the crop" — that's evidence for a head-level intervention
(attention regularization, deeper supervision on framewise predictions,
alternative head architecture).

If it doesn't surface a pattern — same per-frame behavior across
categories — we know the head is not the bottleneck and can drop the
model-internals path. The next step would be to step further back and
reconsider the strategy entirely (likely 2025 pretraining or
backbone diversity for ensemble), not just trying another recipe knob.

## Why this is single-purpose, not multi-axis

We're explicitly *not* mixing this with any training change. No new
loss, no new mixup variant, no architectural swap. The point is to
*see* a thing in the model, not to *change* the model. After the
analysis, any change we propose should be one variable, with the
analysis as the evidence — same discipline as the pivot from bundled
exp 028 to single-variable.
