# 2026-04-24 — certainty scalars + species composition of v5j failures

Follow-up to [2026-04-24-triage-v5j-failure-modes.md](2026-04-24-triage-v5j-failure-modes.md).
Triage told us which samples are genuine failures (label-aware); this note asks:
**given the failures, how decisive is the model on them, and which species do they
come from?**

Goal is narrow: build a reproducible way to pick the right v5j samples to look
at next. Not "how do we fix the model" yet — one step earlier.

## Why we needed a new lens

Gradient saliency's main failure is sigmoid saturation. At `gt_prob > 0.9` or
`< 0.05` the derivative is near zero, so the saliency map is noise regardless
of what the model is doing internally. That's 65% of our val set.

Certainty scalars sidestep this by reading the output distribution **directly
from logits**, no derivatives. Same model, same samples, different summary.

## The four scalars (from `certainty.py`)

Computed per-sample from the clipwise logits already in
`losses_v5j_logits.npy`. Added as columns to `triage_v5j.parquet`.

| Column | Formula | What it measures |
|---|---|---|
| `shannon_norm` | `H(softmax-normalized sigmoid probs)` | How peaked is the prediction? (largely redundant with top1_prob) |
| `bernoulli_sum` | `Σᵢ [-pᵢlog pᵢ − (1−pᵢ)log(1−pᵢ)]` | How much total class-wise uncertainty mass is there? |
| `top1_top2_gap` | `logit₁ − logit₂` | Is the top-1 decisive over the runner-up? |
| `n_active` | `#{i : pᵢ > 0.1}` | How many species is the model treating as active? |

Multi-label note: the 234 sigmoid probs don't sum to 1, so Shannon entropy on
them is a proxy. `bernoulli_sum` is the principled version (sum of independent
Bernoulli entropies). Both variants are saved — they disagree on one bucket in
an informative way (see below).

## Category × scalar summary

| Category | n | mean loss | shannon_norm | bernoulli_sum | top1_top2_gap | n_active |
|---|---:|---:|---:|---:|---:|---:|
| correct_confident | 1,643 | 0.005 | **0.94** | 1.30 | 7.37 | 1.34 |
| correct_uncertain | 986 | 0.008 | 1.94 | 2.49 | 2.80 | 1.48 |
| mixed_uncertainty | 551 | 0.021 | **2.92** | 2.93 | 1.07 | 1.27 |
| confident_wrong | 38 | 0.033 | **1.80** | **2.85** | **3.48** | 1.92 |
| gt_invisible | 225 | 0.036 | 3.18 | 3.26 | 1.28 | 1.03 |
| cooccurrence | 211 | 0.039 | 1.97 | 3.09 | 2.08 | 3.76 |

Monotonic in the expected direction on three of four scalars. Passes face
validity.

## Validation — am I sure this is correct?

Two honest checks:

**(1) Face validity — passes.** Entropy grows as loss grows, except where the
category structure demands otherwise (`confident_wrong` and `cooccurrence` are
not uncertainty failures, they're different kinds of label mismatch). `n_active
= 3.76` on cooccurrence lines up with the triage definition (multi-species
recordings). Nothing in the table breaks the story.

**(2) Redundancy with existing columns — concerning.** `shannon_norm` has
correlation `−0.857` with `top1_prob`, which was already in the triage parquet.
As a "real-failure" predictor: `top1_prob` AUC 0.910, `shannon_norm` AUC 0.879,
`top1_top2_gap` AUC 0.882, `loss` AUC 0.883. The new scalars are slightly
*worse* than `top1_prob` at ranking failures. **So entropy does not add new
information beyond what the triage parquet already contained.** I oversold
this in an earlier turn.

**What entropy does add, narrowly:** it distinguishes `confident_wrong`
from `correct_uncertain` through `bernoulli_sum`:

| Sample state | shannon_norm | bernoulli_sum |
|---|---:|---:|
| correct_uncertain | 1.94 | 2.49 |
| **confident_wrong** | 1.80 (≈ correct_uncertain) | **2.85 (≈ mixed_uncertainty)** |

Shannon says "confident_wrong is peaked, like correct_uncertain." Bernoulli
says "confident_wrong has broader class-wise uncertainty than correct_uncertain,
almost as much as mixed_uncertainty." Both are right — they measure different
aspects. The model in `confident_wrong` is peaked on the WRONG species (Shannon
agrees with correct_uncertain) AND simultaneously has extra low-confidence
classes firing in the background that shouldn't (Bernoulli agrees with
mixed_uncertainty). This is the one pattern `top1_prob` alone doesn't show.

## So what is the methodology actually?

**Not** "entropy is a magic sample picker." **Yes** "triage_category +
saliency_informative + top1_prob ranking (dedupe on (filename, offset_sec)) is
the reproducible way to pick real failures." Certainty scalars are confirmation
that the triage buckets are internally consistent — samples within a category
share structural similarity on independent metrics. They are not a replacement
for triage.

Two distinct failure modes on the table, to be clear:

| Failure mode | n | signature | interpretation |
|---|---:|---|---|
| **Uncertainty failure** (`mixed_uncertainty`) | 551 | spread across classes, top1_top2_gap=1.07, bernoulli_sum=2.93 | model doesn't know what it's hearing |
| **Confusion failure** (`confident_wrong`) | 38 | peaked on wrong class, top1_top2_gap=3.48, but high bernoulli_sum=2.85 | model decisively misidentifies *and* has spurious background activations |

These need different fixes. Keep them separated in all downstream analysis.

## Species composition of mixed_uncertainty (the 551-sample bucket)

From `species_composition.py --compare mixed_uncertainty correct_uncertain`.

**Concentration test:** 155 distinct species appear in mixed_uncertainty. Share
of samples from the top-K species:

| K | share | uniform expected |
|---:|---:|---:|
| 10 | **21.2%** | 6.5% |
| 20 | 36.1% | 12.9% |
| 50 | 64.6% | 32.3% |
| 100 | 88.9% | 64.5% |

Not extreme concentration but clearly non-uniform. Top-50 species drive 65% of
mixed_uncertainty while being 32% of the species set. Balanced sampling should
help some of this, but not all.

**Species most over-represented in mixed_uncertainty vs correct_uncertain**
(log2 ratio of category-shares, filtered to `total ≥ 5`):

| species | n_mixed | n_correct | log2_ratio | failure rate (mixed / total) |
|---|---:|---:|---:|---:|
| `grepot1` | 5 | 0 | 4.30 | 100% |
| `magtan2` | 4 | 1 | 2.42 | 80% |
| `roahaw` | 15 | 6 | **2.09** | **71% (largest n)** |
| `rufgna3` | 5 | 2 | 1.98 | 71% |
| `sobcac1` | 8 | 4 | 1.76 | 67% |
| `epaori4` | 8 | 4 | 1.76 | 67% |
| `rutjac1` | 6 | 3 | 1.73 | 67% |
| `coffal1` | 8 | 5 | 1.47 | 62% |
| `socfly1` | 13 | 11 | 1.07 | 54% (common) |

Species the model handles well even when uncertain (for contrast):

| species | n_mixed | n_correct | log2_ratio |
|---|---:|---:|---:|
| `varant1` | 0 | 17 | −4.29 |
| `rufhor2` | 1 | 17 | −2.71 |
| `chobla1` | 1 | 16 | −2.62 |

**Implication for sample picking:** the most productive failure samples to
inspect are the ones from high log2_ratio species with meaningful sample count
(≥ 5), especially `roahaw` (n=15) where the pattern is replicated across many
segments. `grepot1` (100% failure) is surgical but n=5 limits what we can
generalize.

## Followups this raises (do not do yet, scope discipline)

- Is `roahaw` actually rare in train.csv? If yes → balanced sampling
  hypothesis. Need to join species counts.
- Do the over-represented species in mixed_uncertainty cluster taxonomically?
  (Raptors? Flycatchers? Same genus confusions?) Open the CSV in pandas and
  check family/genus from `taxonomy.csv`.
- Does the `confident_wrong` set (38 samples) show systematic species pairs
  (`primary → top-1`)? That's a different analysis than the species-composition
  one — pairwise, not marginal.

These are one query each. Not doing them here to keep the note tight.

## How to reproduce

All data is under `user/murilo/analysis/`.

```bash
# (1) triage — already run; produces triage_v5j.parquet
python triage.py --losses data/losses_v5j.parquet

# (2) certainty scalars — in-place enrichment of triage_v5j.parquet
python certainty.py --triage data/triage_v5j.parquet --summary

# (3) species comparison — writes a CSV of all 189 species with log2 ratios
python species_composition.py \
    --triage data/triage_v5j.parquet \
    --compare mixed_uncertainty correct_uncertain \
    --top-n 20 \
    --output data/species_composition_mixed_vs_correct.csv
```

Generated artifacts:
- `data/triage_v5j.parquet` — now has 4 extra columns: `shannon_norm`,
  `bernoulli_sum`, `top1_top2_gap`, `n_active`
- `data/species_composition_mixed_vs_correct.csv` — full per-species table
  (189 rows) sorted by log2_ratio

## Action items (narrow)

- Use `top1_prob` as the primary "how decisive" sort key, not `shannon_norm`.
  Keep `bernoulli_sum` around only for `confident_wrong` diagnostics.
- When picking sample figures to inspect: filter to
  `triage_category == 'mixed_uncertainty' AND saliency_informative`, then rank
  by `-top1_top2_gap` (most uncertain first) OR by `primary_label IN
  {roahaw, epaori4, sobcac1, coffal1}` (species most concentrated in
  mixed_uncertainty).
- Cross-check the `roahaw` hypothesis: open 5–10 roahaw mixed_uncertainty
  samples, see what the model predicts and whether saliency points anywhere
  coherent.
- Do not re-derive "entropy" claims elsewhere without citing this note —
  `shannon_norm` and `top1_prob` are ~equivalent.

## Files

- `analysis/certainty.py` — computes the four scalars
- `analysis/species_composition.py` — per-category and pairwise species stats
- `analysis/data/triage_v5j.parquet` — enriched triage table (mutated in place)
- `analysis/data/species_composition_mixed_vs_correct.csv` — full species
  comparison
