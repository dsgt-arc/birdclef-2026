# 2026-04-24 — three regimes of mixed_uncertainty (hypothesis, not yet validated)

Extension of [the roahaw analysis](2026-04-24-roahaw-failure-mode-analysis.md).
Looked at one rank01 figure each from `coffal1`, `socfly1`, `epaori4`
(`mixed_uncertainty`) plus a `roahaw correct_confident` baseline. **Hypothesis
formed**: the silent-crop pattern does NOT generalize uniformly;
`mixed_uncertainty` may decompose into three structurally different failure
regimes, identifiable by **median gt_prob across all val samples for the
species**.

**Important — confidence is medium, not high.** The narrative below describes
what 4 figures + per-species statistics suggest. It is not yet validated to
the degree that should drive experiment design. See the confidence audit
section below.

## Confidence audit

Tier 1 (high confidence — mechanical / statistical facts):

- The triage category counts and the per-sample certainty scalars.
- Median gt_prob per species. The numbers are what they are.
- Per-species focal counts in `train.csv`.
- The aggregate split (87 / 50 / 23 species) **at the chosen thresholds 0.8 / 0.3**.

Tier 2 (medium confidence — supported by small visual evidence, extrapolated):

- "roahaw mixed_uncertainty failures look like silent-crop label noise."
  Based on 4 of 15 spectrograms inspected. **No audio has been listened to.**
  The "silent" framing is inferred from spectrogram-visual reasoning.
- "epaori4 is real acoustic confusion (model decisively wrong)" — based on
  one rank01 figure.
- "socfly1 is a 'uniformly uncertain' regime" — based on one rank01 figure
  and process-of-elimination thinking.
- "saliency_informative correctly flagged the one real failure in the roahaw
  set." N=1.

Tier 3 (low confidence — conjecture):

- "50% of mixed_uncertainty is label noise" — extrapolation from one bimodal
  species (roahaw) to all 87 bimodal species. No direct evidence on the
  other 86. coffal1 rank01 already had visible chirp content, not pure
  silence — a counter-data-point I should have weighted more.
- "Three regimes" as a structural property of v5j. I picked thresholds (0.8,
  0.3) without sensitivity analysis. **Median gt_prob does not actually
  measure bimodality** — a unimodal distribution centered at 0.85 also has
  median 0.85. Bimodality has not been verified directly.
- "8 abundant-but-model-blind species are the cleanest target." Multi-step
  inference. I have not checked whether these species appear meaningfully
  in the public test set; if they don't, fixing them won't help LB.
- The story generalizes to other models (v5g, etc.). v5j-only result.

### Specific weaknesses

1. **No audio listened.** Cheapest verification step — never done.
2. **6 figures total drive the entire interpretive layer** (rank01s for 4
   species + rank15 of roahaw + correct_confident baseline).
3. **Cherry-picked species.** The 4 inspected were the over-represented
   species from species composition. Non-random — biased toward "interesting
   failures."
4. **Threshold sensitivity untested.** Why 0.8 and 0.3? Made up. A sweep
   would tell us if the 50/33/17 split is robust or an artifact of cuts.
5. **Median != bimodality.** I conflated "median is high" with "distribution
   is bimodal." Should plot histograms or compute a bimodality coefficient.
6. **Test-set relevance not checked.** LB doesn't care about species absent
   from the test soundscapes.
7. **Inference window vs val window mismatch.** v5j val computes loss on a
   single 5s crop per recording; LB inference uses TTA + smoothing. The
   relationship between val-set bimodality and test-set behavior is unclear.

## The per-species diagnostic

For each species, compute median gt_prob over all its val samples. This
cheap scalar separates three regimes:

| Regime | median gt_prob | What's happening | What `mixed_uncertainty` means |
|---|---|---|---|
| **Bimodal** | ≥ 0.8 | Model is very confident when the call is present, totally dismissive otherwise. | **Silent-crop label noise** — crop landed between calls, BCE penalizes honest dismissal. |
| **Uniformly uncertain** | 0.3–0.8 | Model is moderately unsure across all crops for this species. | Real discrimination difficulty — genuine acoustic ambiguity. |
| **Model-blind** | < 0.3 | Model rarely confident about this species at all. | Real model weakness — hasn't learned the species. |

## Quantification (n_val ≥ 5 per species, 160 species)

| Regime | n_species | val_samples | mixed_uncertainty count | share of mixed_uncertainty |
|---|---:|---:|---:|---:|
| Bimodal | 87 | 2,248 | 266 | **49.8%** |
| Uniformly uncertain | 50 | 988 | 175 | **32.8%** |
| Model-blind | 23 | 335 | 93 | **17.4%** |

**~half of `mixed_uncertainty` is likely label noise** (from bimodal species
where crops didn't catch the vocalization). Training to reduce this portion of
the val loss would not help — it would teach the model to hallucinate the
primary species regardless of crop content.

**~half is real**: 32.8% real discrimination difficulty +
17.4% real model weakness. These are the useful failure modes to work on.

## Visual validation (one rank01 per regime)

Figures exported with the new `export_figures.py` filtered mode. One
representative rank01 image for each regime, all 4 species:

| Species | n_mixed | Regime | Visual signature at rank01 |
|---|---:|---|---|
| `roahaw` | 15 | Bimodal | Ambient noise / silence. Top-1 prob 0.18. No clear calls. |
| `roahaw` (correct_confident baseline) | — | Bimodal | Clean harmonic stack. Prob 0.986. Attention peaked on call frames. |
| `coffal1` | 8 | Bimodal | Visible chirp in mid-frequency. Top-10 all ≤ 0.08 (model is flat). |
| `socfly1` | 13 | Uniformly uncertain | Percussive/short calls visible. Top-1 an ambiguous numeric species at 0.35, socfly1 low. |
| `epaori4` | 8 | Model-blind | **Strong clear vocalization, gradient saliency focused, top-1 saffin 0.60 confidently wrong.** |

Each regime looks qualitatively distinct. Regime assigned from median gt_prob
alone, without viewing any figures — the visual inspection confirmed the
prediction.

## The 23 model-blind species by subgroup

Focal training counts tell a second-order story within model_blind:

| Subgroup | n_species | Median focal | Interpretation |
|---|---:|---:|---|
| No focal data | 2 (`1491113`, `47158son03`) | 0 | Soundscape-only species (numeric insect codes). Can't be fixed with focal-only training. |
| Under-represented (focal < 150) | 13 | ~70 | Potentially helpable with balanced sampling. |
| **Abundant but still model-blind (focal ≥ 150)** | **8** | **188** | **Acoustically hard. Enough training data; model genuinely can't learn the distinction.** |

The 8 **abundant-but-model-blind** species are the cleanest targets for
modeling work — no data excuses, just a real discrimination gap:

| Species | n_val | n_mixed | med_gt_prob | n_focal |
|---|---:|---:|---:|---:|
| compot1 | 36 | 12 | 0.27 | **368** |
| greela | 22 | 5 | 0.07 | 224 |
| redjun | 20 | 7 | 0.05 | 208 |
| epaori4 | 18 | 8 | 0.04 | 188 |
| y00678 | 18 | 6 | 0.20 | 188 |
| plcjay1 | 18 | 6 | 0.15 | 186 |
| strowl1 | 18 | 6 | 0.09 | 184 |
| sobcac1 | 18 | 8 | 0.25 | 183 |

The worst ones (`greela`, `redjun`, `epaori4`) have median gt_prob under 0.1
despite 200+ focal recordings. **The model encodes a decisive WRONG answer
for these species.** Visible in the `epaori4` rank01 figure: clean
vocalization, focused attention, decisively identifies it as `saffin`.

## Implications — revised from the roahaw-only story

**Previous (from roahaw note):** ~64% of `mixed_uncertainty` is silent-crop
label noise.

**Revised:** ~50% is silent-crop label noise (bimodal species), ~33% is real
discrimination difficulty (uniformly uncertain), ~17% is real model weakness
(model-blind). The original 64% estimate extrapolated from a single bimodal
species — it overstated the label-noise share.

**These three regimes need different fixes, not one:**

| Regime | Plausible lever (labeled data only) |
|---|---|
| Bimodal / silent-crop | VAD or energy-based crop filtering; energy-weighted BCE; secondary_labels as positive supervision. Do NOT try to "fix" high loss here — it's honest dismissal. |
| Uniformly uncertain | Better features (mel resolution, backbone); discrimination-focused losses. Less clear lever with labeled data only. |
| Model-blind (abundant) | Margin-based or contrastive loss to push acoustically-similar species apart; hard-negative mining on these 8 species. |
| Model-blind (rare / no-focal) | Balanced sampling; soundscape pseudo-labels for the 2 no-focal species. |

**A single training change will not help all three.** Any recipe change
should be targeted at one regime, and evaluated against metrics restricted to
that regime.

**One thing the aggregate confirmed across regimes**: `top-1 confusions in
mixed_uncertainty are scattered — not systematic pairs.** coffal1 had 8
distinct top-1s across 8 samples, socfly1 had 13 across 13, epaori4 had 7
across 8 (only `fusfly1` appeared twice). Hard-negative mining on specific
pairs is unlikely to help; the failure is broader than that.

## Hypotheses this raises

1. **Why are `compot1` / `epaori4` / `redjun` model-blind despite abundant focal
   data?** Plausible: acoustic similarity to dominant/common species. Check
   what their top-1 predictions are across val — if the confusions cluster
   taxonomically or spectrally, that's a signal. Possible follow-up: tSNE or
   per-class mean embedding distance plot.
2. **Is the 50% bimodal-silent-crop share stable across models?** Same
   analysis on v5g (0.892 LB) would show if this is a v5j artifact or a
   general SED-on-this-data property. Cheap to test.
3. **Does the share of model-blind species correlate with LB movements?**
   Retrospectively check: did exps 016→019 (big LB gains from backbone + mel
   changes) reduce the model-blind count? That would validate this lens for
   future experiments.

## How to reproduce

```bash
cd user/murilo/analysis

# One-liner to classify all species into regimes:
python -c "
import pandas as pd
tri = pd.read_parquet('data/triage_v5j.parquet')
train = pd.read_csv('/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train.csv')
stats = tri.groupby('primary_label').agg(n_val=('idx','size'), med_gt_prob=('gt_prob','median')).reset_index()
stats['regime'] = stats['med_gt_prob'].apply(lambda m: 'bimodal' if m>=0.8 else ('model_blind' if m<0.3 else 'uniformly_uncertain'))
fc = train['primary_label'].value_counts().reset_index()
fc.columns = ['primary_label','n_focal']
stats = stats.merge(fc, on='primary_label', how='left').fillna(0)
print(stats[stats['n_val']>=5].groupby('regime').size())
"

# Filtered figure export per regime (rank01 inspected for this note):
python export_figures.py --losses data/losses_v5j.parquet \
    --config ../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml \
    --checkpoint /storage/.../best-epochepoch=19.ckpt \
    --filter-category mixed_uncertainty --filter-species <species> \
    --top-k -1 --batch-k 0 --device cpu

# Baseline:
python export_figures.py ... \
    --filter-category correct_confident --filter-species roahaw --top-k 10
```

## Action items (validation first, then investigation)

The deep-dive audit (see top of note) identified that the three-regime story
sits at medium confidence. Validation steps are required before basing any
experiment on it.

### Validation (run before any experiment design)

1. **Listen to 15 roahaw `mixed_uncertainty` WAVs + 5 `correct_confident`
   WAVs.** Cheapest, highest-signal verification. Settles "silent crop"
   literally. Files at `figures/v5j/mixed_uncertainty_roahaw/*.wav` and
   `figures/v5j/correct_confident_roahaw/*.wav`.
2. **Plot full gt_prob distributions** (histograms, not just medians) for
   5 species in each regime. Tests bimodality directly. If "bimodal"
   species show unimodal distributions, the regime label is wrong even if
   median value is correct.
3. **Threshold sensitivity sweep**: recompute regime shares at thresholds
   `(0.7, 0.3)`, `(0.8, 0.3)`, `(0.9, 0.3)`, `(0.8, 0.2)`, `(0.8, 0.4)`.
   If 50/33/17 is stable, the regime story is robust; if it shifts a lot,
   classification is just a Procrustean cut.
4. **Random-sample figure verification**: pick 3 random species per regime
   (NOT from species composition top list) and view rank01. If patterns
   hold on a non-cherry-picked sample, much stronger evidence.
5. **Test-set relevance check** for the 8 "abundant-but-model-blind"
   species: do they appear meaningfully in the public test soundscapes?
   If not, modeling work on them won't move LB.
6. **Cross-model validation** on v5g: re-run the regime split on a
   `losses_v5g.parquet`. Same patterns → property of the data; different
   patterns → v5j-specific artifact.

### Investigation (only if validation holds)

- **Per-species top-1 confusion analysis** for the 8 abundant-but-model-blind
  species. Do confusions cluster on 1–2 common species (actionable for
  margin loss / hard-negative mining), or scatter (broader representation
  problem)?

### Experiment design (only if both validation AND investigation produce a
clear lever)

- One experiment per regime, in isolation, evaluated against
  regime-restricted metrics. See the regime/lever table earlier in this
  note. **Do NOT** propose any experiment yet — the evidence isn't there.

## Files

- `analysis/figures/v5j/mixed_uncertainty_{roahaw, coffal1, socfly1, epaori4}/`
- `analysis/figures/v5j/correct_confident_roahaw/`
