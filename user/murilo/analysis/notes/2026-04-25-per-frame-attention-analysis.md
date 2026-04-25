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

The analysis surfaced a strong structural pattern. v5j's per-frame
behavior on hard crops is qualitatively different from on easy crops,
and the difference is consistent with the audio finding from earlier
today: hard crops have weak/absent calls, and the model honestly
reports low confidence by producing diffuse, low-magnitude per-frame
logits.

**Headline**: v5j is structurally honest. The "failures" we have been
calling `mixed_uncertainty` are not model failures — they are the model
correctly reporting low confidence on low-content crops. The training
objective (BCE on recording-level labels with random 5s crops) penalizes
this honest behavior, which mechanistically explains why 21+ training-
recipe changes have all failed to move LB.

### Category-level results

T = 16 frames per 5s crop. Uniform-attention entropy = log(16) = 2.77
nats. Perfect peak = 0 nats. Run on 2026-04-25, SLURM job 7432078.

| Category | n | mean loss | gt_prob | attn entropy | gt_logit_std | top1_majority |
|---|---:|---:|---:|---:|---:|---:|
| correct_confident | 1,643 | 0.005 | 0.972 | **1.05** | **4.31** | 0.52 |
| correct_uncertain | 986 | 0.008 | 0.586 | 1.09 | 3.19 | 0.38 |
| mixed_uncertainty | 551 | 0.021 | 0.097 | **1.53** | **1.97** | 0.27 |
| confident_wrong | 38 | 0.033 | 0.011 | 1.89 | 1.68 | 0.39 |
| cooccurrence | 211 | 0.039 | 0.245 | 1.79 | 1.94 | 0.51 |
| gt_invisible | 225 | 0.036 | 0.003 | **2.12** | **1.30** | 0.29 |

Two strong gradients:

1. **Attention entropy increases monotonically with failure severity** on
   the GT class. correct (1.05) → mixed (1.53) → confident_wrong (1.89)
   → gt_invisible (2.12, near uniform 2.77). The head's implicit
   per-frame weighting flattens out as the model loses confidence.

2. **GT-class logit std collapses on failures**, even more dramatically
   than entropy. correct_confident has high cross-frame variability
   (4.31) — the per-frame logits vary sharply ("this frame yes, that
   frame no"). On `mixed_uncertainty` (1.97) and `gt_invisible` (1.30),
   the per-frame logits are nearly flat — the model isn't picking any
   specific frame as evidence of the GT species.

### Bimodal-species deep dive

Same species, different categories. If the head behaved identically
across categories within a species, the failure would be upstream
(input/backbone). If different, the head is doing something different
under low-SNR conditions.

| Species | Category | n | attn entropy | logit std | top1_maj |
|---|---|---:|---:|---:|---:|
| roahaw  | correct_confident | 26 | 0.78 | 4.93 | 0.37 |
| roahaw  | mixed_uncertainty | 15 | **2.29** | **1.13** | 0.34 |
| coffal1 | correct_confident | 29 | 0.72 | 4.66 | 0.41 |
| coffal1 | mixed_uncertainty |  8 | **1.74** | **1.47** | 0.26 |
| socfly1 | correct_confident | 18 | 0.66 | 4.44 | 0.33 |
| socfly1 | mixed_uncertainty | 13 | **1.59** | **1.59** | 0.23 |
| compot1 | correct_confident | 11 | 1.13 | 5.55 | 0.52 |
| compot1 | mixed_uncertainty | 12 | **2.46** | **0.85** | 0.42 |
| whtdov  | correct_confident |  8 | 0.87 | 3.87 | 0.48 |
| whtdov  | mixed_uncertainty | 14 | **1.98** | **1.21** | 0.26 |
| barant1 | correct_confident | 18 | 1.36 | 4.25 | 0.53 |
| barant1 | mixed_uncertainty | 11 | **1.69** | **1.71** | 0.20 |

All six bimodal species show the same pattern: entropy roughly doubles
or triples between `correct_confident` and `mixed_uncertainty`, and
logit std collapses 3–6×. The same species's calls produce qualitatively
different head outputs depending on whether the call is acoustically
dominant in the crop.

### Connection to the audio listening

From the [audio-validation note](2026-04-25-audio-validation-low-snr-crops.md)
(same date): `mixed_uncertainty` crops are dominated by ambient/
anthropogenic noise with weak, distant, or absent bird calls.
`correct_confident` crops have the call as the foreground.

Per-frame analysis adds the structural observation: when the call is
acoustically dominant, the model produces peaked attention and high
logit std — it localizes the call to specific frames. When the call is
weak or absent, the model produces flat attention and collapsed logit
std — it doesn't pick any frame as evidence.

These observations are internally consistent. The model's "failure" on
`mixed_uncertainty` is not a model failure; it is **honest signal-matching
under low-SNR conditions**.

### What this means structurally

The training objective penalizes this honest behavior. The recording's
primary species is labeled present whether or not a specific 5s crop
contains that species's call; BCE pushes the model to predict the primary
species confidently across all crops — including silent or noise-dominated
ones. v5j has resisted that push.

That resistance is exactly the property that makes v5j generalize to
soundscape inference (test labels are segment-level; false positives are
punished). It is also exactly the property that makes our val/auroc
improvements regress on LB: any recipe change that "improves val/auroc"
by getting the model to predict more confidently on hard crops is
teaching dishonesty, which hurts LB.

**The diffuse-on-hard-crops behavior is a feature, not a bug. The 21
recipe failures were attempts to fix a non-failure.**

### Caveats

- We measured only **GT-class** per-frame behavior. For `confident_wrong`
  (top1_prob=0.756 on a wrong species, n=38), we don't yet know whether
  the head is *peaked on the wrong species* or *diffuse there too*.
  Adding top1-class metrics would clarify whether `confident_wrong` is
  a real head-level failure or just BCE noise.
- We didn't subset `mixed_uncertainty` to `saliency_informative=True`
  (~196 samples, gt_prob ∈ [0.05, 0.9], the subset where the call IS
  acoustically present at moderate confidence). Those might still
  represent a real failure mode worth targeting separately.
- T=16 frames per 5s = ~313 ms per frame. This is coarse — short calls
  may straddle frame boundaries. Possibly a structural ceiling on
  temporal localization.
- v5j inference uses TTA + smoothing over multiple offsets; val uses a
  single random crop. The honesty pattern at val may not translate
  identically to inference behavior.
- Single-model finding (v5j only). Cross-model validation on v5g would
  confirm whether this is a property of LSEHead training or a v5j
  artifact.

## What this enables next

The analysis surfaced a clear pattern with a mechanistic explanation
for our 21+ training-recipe failures. This argues against further
training-recipe tweaks on v5j and points toward three plausible next
paths:

1. **Architecture / backbone change.** Salman's 0.937 isn't from
   "fixing diffuse behavior on hard crops" — diffuse on hard crops is
   what the head SHOULD do. Whatever drives his +0.031 over us is
   probably backbone or attention-architecture quality, not loss
   tweaking. EffNetV2-B0 with Salman's disclosed recipe is the most
   direct test (queued as exp 029 candidate).
2. **Test-time inference improvements.** PP / TTA / segment smoothing /
   prior fusion. Most of this is already in our pipeline; small tuning
   may find +0.005.
3. **Pretraining / external data.** Ali Ozan's 2025-pretrain path.
   Different bet — adds data variety rather than tweaking the training
   objective.

**Demoted by this finding:**

- **BG noise mixing (exp 028, paused).** Forcing the model to predict
  species through noise teaches false positives, which is exactly what
  hurts LB. The structural argument against this lever is now strong.
- **Energy-weighted BCE / loss reweighting.** Same problem in different
  clothes.
- **AttHead substitution as a single-variable swap.** Prior 7 AttHead
  tests regressed; the diffuse-on-hard-crops pattern would persist
  there too because it's data-driven, not LSEHead-specific.

**Future experiment candidate (exp 030): explicit silent / no-call class.**

Add a 235th class for "no signal in this 5s window," supervise it
during training, and use it at inference to gate predictions. Matches
what the model is already learning to do; it would just give it
permission to say so explicitly. Engages directly with the structural
finding (model is honest; training objective punishes honesty) rather
than going around it like augmentation tweaks do.

**Why this is queued and not the next experiment**: it has its own
data-acquisition challenge that overlaps with the BG noise problem we
just paused. The silent-class supervision data MUST contain no bird
calls, otherwise the model learns "label silent when this species is
also present" — defeating the purpose. Sources we considered:

- Labeled soundscapes: every 5s segment has at least one species
  annotated (verified 2026-04-25). Not viable.
- Synthetic silence (zeros + tiny gaussian): clean by construction
  but out-of-distribution from real soundscapes. Weak test of the
  concept.
- Self-bootstrap from unlabeled 2026 soundscapes via low-RMS energy
  filter + v5j confidence filter (find segments where all 234 species
  probs are < 0.1 — model is already confident no species is present).
  Most pragmatic option, but requires a precompute pass and accepts
  some implicit dependence on v5j's existing predictions.
- External clean datasets (freefield1010 no-bird subset, ESC-50): we
  reviewed these for the BG noise experiment. Available but require
  download + format normalization.

If exp 029 (2025 pretrain) helps, exp 030 is the natural follow-up
to test whether the structural finding can be acted on directly.
If exp 029 lands at 0.92+ already, exp 030 may be lower priority.

## Why this is single-purpose, not multi-axis

We're explicitly *not* mixing this with any training change. No new
loss, no new mixup variant, no architectural swap. The point is to
*see* a thing in the model, not to *change* the model. After the
analysis, any change we propose should be one variable, with the
analysis as the evidence — same discipline as the pivot from bundled
exp 028 to single-variable.
