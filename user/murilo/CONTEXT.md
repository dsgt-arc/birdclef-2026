# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-24 (end of session)

## Read these files first

Read at session start to get full context. Priority matters — skip the "skim"
ones if you're short on time, but read the top three analysis notes fully.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/analysis/notes/2026-04-24-three-regimes-of-mixed-uncertainty.md` | **Hypothesis (medium confidence, not yet validated)** — `mixed_uncertainty` may split by median gt_prob into 3 regimes (bimodal silent-crop ~50%, uniformly uncertain ~33%, model-blind ~17%). Includes a confidence audit at the top. **Validation steps must run before any experiment is designed.** | Read fully (start here) |
| `user/murilo/analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md` | Earlier finding on roahaw only. Silent-crop hypothesis on 4/15 figures. Read for the tooling-validation story; the broader extrapolation is captured (with caveats) in the three-regimes note. | Read fully |
| `user/murilo/analysis/notes/2026-04-24-certainty-and-species-composition.md` | `shannon_norm` is redundant with `top1_prob`; `bernoulli_sum` uniquely catches peaked-but-wrong. Species composition identifies roahaw as the first investigation target. | Read fully |
| `user/murilo/analysis/notes/2026-04-24-triage-v5j-failure-modes.md` | Foundation — triage categorization of 3,654 val samples. Only 1% is `confident_wrong`; 50% of top-20-by-loss is co-occurrence, not failure. | Read fully |
| `user/murilo/analysis/README.md` | Tooling map — six scripts (triage, certainty, species_composition, export_figures filtered mode, etc.) with usage | Skim |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights (Salman 0.937, Boredom 0.950+) | Skim |
| `user/murilo/FINDINGS.md` | What worked / didn't across exps 006–027 | Skim |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |
| `user/murilo/experiments/028-domain-shift-augmentation/proposal.md` | Still paused — diagnostics first | Skim only if resuming exp 028 |
| `user/murilo/notes/sed-architecture-overview.md` | Pipeline reference (13 sections) | Reference material |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Best single SED: 0.906** (v5j, 5s LSEHead, exp 019)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/model-attention-analysis` — PR #38

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold** |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.906** | **Best single SED — our ceiling** |
| v5j-dualval (exp 026) | 0.897 | Soundscape checkpoint selection (worse) |
| v5j-pl-v1 (exp 025 Phase B) | 0.897 | Pseudo-labels as separate rows (regressed) |
| v3-waveform-mix (exp 025 Phase C) | 0.894 | FAILED — Mario-style PL waveform mixing −0.012 |
| v11b simple mel norm (exp 027) | 0.895 | FAILED — mel norm regresses −0.011 |
| v10b addmix+BCE (exp 027) | 0.875 | FAILED — additive MixUp regresses −0.031 |
| v10a addmix+CE (exp 027) | 0.494 | FAILED — CE logits too small for sigmoid |

No LB runs this session — all work was analysis on the existing v5j checkpoint.

## What happened recently

### 2026-04-24 — analysis methodology + working hypothesis on failure regimes

Built a reproducible analysis pipeline on top of the 04-23 attention explorer.
All scripts operate on `losses_v5j.parquet` + `losses_v5j_logits.npy` (already
precomputed) — no GPU or retraining needed for any of this work.

**New tooling** (under `user/murilo/analysis/`):
- `triage.py` — label-aware failure classification (6 categories, reads `train.csv` for secondary labels)
- `certainty.py` — four scalars from clipwise logits (`shannon_norm`, `bernoulli_sum`, `top1_top2_gap`, `n_active`), appended to triage parquet
- `species_composition.py` — per-category and pairwise species stats with log2 ratio
- `export_figures.py` filtered mode — triage-aware figure export, `--filter-category`/`--filter-species`/`--filter-saliency-informative`, dedupes on `(filename, offset_sec)`, output at `figures/<model>/<slug>/`
- Triage is now chained after `precompute_losses.py` in the sbatch job

**Four notes (read in order)**:

1. **[triage note](analysis/notes/2026-04-24-triage-v5j-failure-modes.md)** — classifies all 3,654 val samples. Only 1% (`confident_wrong`, n=38) is unambiguous model failure; 50% of top-20-highest-loss is `cooccurrence_label_ambiguity` (the "wrong" top-1 is actually a known secondary species). `confident_wrong` is 100% focal; soundscape is 76% co-occurrence.

2. **[certainty + species composition](analysis/notes/2026-04-24-certainty-and-species-composition.md)** — `shannon_norm` is largely redundant with `top1_prob` (corr −0.86); entropy is a sanity check on triage, not a new signal. `bernoulli_sum` uniquely distinguishes `confident_wrong` from `correct_uncertain`. Species composition identified `roahaw` (n=15, 71% failure rate) as the first investigation target.

3. **[roahaw failure-mode analysis](analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md)** — exported all 15 roahaw `mixed_uncertainty` figures, viewed 4. Visually, ~14/15 looked like silent/low-content crops (BCE penalizing honest dismissal); 1/15 was a real `saffin` vs `roahaw` confusion (and the only sample with `saliency_informative=True`). roahaw is NOT rare (489 focal recordings, 92nd percentile). Falsified the rare-species hypothesis for roahaw specifically.

4. **[three regimes of mixed_uncertainty (working hypothesis)](analysis/notes/2026-04-24-three-regimes-of-mixed-uncertainty.md)** — extended to `coffal1`, `socfly1`, `epaori4` + `roahaw correct_confident` baseline. **One rank01 figure per species (4 species mixed_uncertainty + 1 baseline).** Suggests `mixed_uncertainty` may split by **median gt_prob across all val samples per species** into:
   - **Bimodal (median gt_prob ≥ 0.8)**: 87 species, ~50% of mixed_uncertainty. Hypothesized "silent-crop label noise."
   - **Uniformly uncertain (0.3–0.8)**: 50 species, ~33%. Hypothesized "real discrimination difficulty."
   - **Model-blind (< 0.3)**: 23 species, ~17%. Hypothesized "model hasn't learned the species."

   Within model-blind, 8 species have 150+ focal recordings AND median gt_prob < 0.3 (`compot1`, `greela`, `redjun`, `epaori4`, `y00678`, `plcjay1`, `strowl1`, `sobcac1`). If the regime story holds, these would be the cleanest modeling targets.

### Confidence summary (session deep-dive audit)

The narrative above is medium confidence at best. Specifically:

- **Tier 1 (high)**: triage counts, median gt_prob values, focal counts, the 87/50/23 species split at the chosen thresholds. These are mechanical statistics.
- **Tier 2 (medium)**: roahaw failures look like silent crops (4 of 15 figures viewed, **0 audio listened**). epaori4 looks like real model-blindness (1 figure). socfly1 is "uniformly uncertain" by elimination.
- **Tier 3 (low)**: "50% of mixed_uncertainty is label noise" (extrapolation from one species to 87). "Three regimes" as a real structural property (thresholds arbitrary; median ≠ bimodality). "8 abundant-but-model-blind = cleanest target" (test-set relevance not checked).

**Key uncovered weaknesses**: no audio listened, only 6 figures total drove the interpretive layer, species selection was cherry-picked from species composition (not random), threshold sensitivity untested, median is not actually a bimodality measure, and we haven't checked if model-blind species even appear in the test soundscapes.

**Practical implication**: do not design training experiments on the basis of these findings yet. Validate first.

### 2026-04-23 — model attention explorer + competitor context

- Built `analysis/model_explorer.ipynb` + supporting scripts (`attribution.py`, `model_loader.py`, `precompute_losses.py`, sbatch).
- Updated Kaggle discussion notes:
  - **Salman: 0.922→0.937** via "investigating attention values on examples with highest loss and minimum loss during training." EfficientNetV2-B0, focal-only.
  - **Boredom: 0.950+** (first iteration on unlabeled). Without unlabeled: 0.938–0.939. "Never been able to break 0.94 with a single model."
  - **Tucker Arrants: 0.925 without deep learning** — Perch embeddings + clustering + metadata.
- Paused exp 028 (domain-shift augmentation) — diagnostics first.

## What needs to happen next

**Validation comes before everything.** The three-regime story is a working
hypothesis sitting on thin visual evidence. The next session should resolve
that before committing to any experiment.

### Priority 1: Validate the three-regime hypothesis (cheap, parallelizable)

1. **Listen to WAVs.** 15 from `figures/v5j/mixed_uncertainty_roahaw/*.wav` and 5 from `figures/v5j/correct_confident_roahaw/*.wav`. Cheapest, biggest signal — never done. Settles "silent crop" literally. ~5 min of human time.

2. **Plot full gt_prob distributions** (histograms, not just medians) for 5 species in each regime. Tests bimodality directly. If "bimodal" species show unimodal distributions, the regime label is wrong even if the median value is correct.

3. **Threshold sensitivity sweep**: recompute regime shares at `(0.7, 0.3)`, `(0.8, 0.3)`, `(0.9, 0.3)`, `(0.8, 0.2)`, `(0.8, 0.4)`. If 50/33/17 is stable, the story is robust; if it shifts a lot, classification is just an arbitrary cut.

4. **Random-sample figure verification**: pick 3 random species per regime (NOT from species composition top list) and view rank01. Tests whether the pattern is real or an artifact of cherry-picking.

5. **Test-set relevance check** for the 8 "abundant-but-model-blind" species: do they appear meaningfully in the public test soundscapes? If not, modeling work on them won't move LB.

6. **Cross-model validation**: regime split on v5g (LB 0.892). Same patterns → property of the data; different → v5j artifact. Requires `losses_v5g.parquet` (need GPU re-run on v5g checkpoint).

### Priority 2: If validation holds — investigate the 8 abundant-but-model-blind species

For each of `compot1`, `greela`, `redjun`, `epaori4`, `y00678`, `plcjay1`, `strowl1`, `sobcac1`:

1. List top-1 predictions across val samples.
2. If they cluster on 1–2 common species → actionable confusion pairs (margin loss / hard-negative mining).
3. If they scatter → broader representation problem, not pair-specific.

### Priority 3: Regime-specific experiment design (only after P1 + P2)

| Regime | Plausible lever (labeled data only) |
|---|---|
| Bimodal / silent-crop (if confirmed) | VAD or energy-based crop filtering; energy-weighted BCE; secondary_labels as positive supervision |
| Uniformly uncertain | Better mel/backbone features; discrimination-focused losses |
| Model-blind (abundant) | Margin-based or contrastive loss; hard-negative mining (if P2 finds clusters) |
| Model-blind (rare / no-focal) | Balanced sampling; soundscape pseudo-labels for the 2 no-focal species |

**Each in isolation. Evaluated against regime-restricted metrics, not aggregate val loss.**

### Priority 4: Exp 028 (still paused)

Combined domain-shift augmentation recipe. Fall back if diagnostics dead-end.

## Key patterns (updated 2026-04-24)

- **21+ changes, 0 LB successes** — 19 training recipe changes + 2 pseudo-label approaches. Neither moved LB.
- **NEW (hypothesis)**: `mixed_uncertainty` may split into three regimes by median gt_prob per species. ~50% bimodal, ~33% uniformly uncertain, ~17% model-blind. **If true**, any fix must target one regime — chasing aggregate val loss conflates three failure modes. **Confidence is medium and validation is required.**
- **NEW (hypothesis)**: 8 abundant-but-model-blind species (`compot1`, `greela`, `redjun`, `epaori4`, `y00678`, `plcjay1`, `strowl1`, `sobcac1`) may be the cleanest targets — 150+ focal recordings AND median gt_prob < 0.3. Test-set relevance unverified.
- **NEW: `saliency_informative` flag is essential.** Restricts to `gt_prob ∈ [0.05, 0.9]` — where sigmoid gradients aren't saturated AND the model is in genuine discrimination space (not silent-crop dismissal). For roahaw mixed_uncertainty: 1 of 15 samples was saliency-informative, and that 1 was the real confusion.
- **NEW: rare-species hypothesis falsified** for 4 of 10 top over-represented species in `mixed_uncertainty` (roahaw 92nd pct, coffal1 98th, socfly1 97th, epaori4 65th). Balanced sampling wouldn't help these — but it could help the other ~13 model-blind species that DO have under-represented focal data.
- **Salman's 0.937 is a modeling breakthrough, not a recipe one** — "investigating attention on high/low loss examples." If the silent-crop hypothesis is correct, label noise on focal crops is a modeling framing issue, not a recipe dial.
- **The gap is on labeled data**: Boredom (0.938) and Salman (0.937) beat us by 0.031 using the same focal + labeled soundscape data. No KD, no Perch.
- **Salman trains focal-only** — adding soundscapes drops him to 0.85. Uses PP/TTA. Our v5j trains on focal + soundscapes.
- **CE loss is dead** (7 tests). Additive MixUp is dead (3 tests). Pseudo-labels are dead (2 approaches). Mel norm, max trick, multi-context all regressed.
- **Offline metrics remain anti-correlated with LB — DO NOT trust them.**

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/analysis/` | **Active work — analysis tooling and notes** |
| `user/murilo/analysis/triage.py` | Classify val samples into 6 failure-mode categories (label-aware) |
| `user/murilo/analysis/certainty.py` | Add shannon_norm / bernoulli_sum / top1_top2_gap / n_active to triage parquet |
| `user/murilo/analysis/species_composition.py` | Per-category and pairwise species stats |
| `user/murilo/analysis/export_figures.py` | Bulk figure export; filtered mode takes triage categories + species + saliency flags |
| `user/murilo/analysis/model_explorer.ipynb` | Single-sample exploration (superseded by filtered export for bulk) |
| `user/murilo/analysis/notes/*.md` | All four 2026-04-24 findings notes — read them in order |
| `user/murilo/analysis/data/triage_v5j.parquet` | Enriched val table (triage + certainty columns) |
| `user/murilo/analysis/data/losses_v5j.parquet` | Precomputed val losses + logits (GPU step done) |
| `user/murilo/analysis/figures/v5j/mixed_uncertainty_*` | Filtered exports already generated for roahaw, coffal1, socfly1, epaori4 |
| `user/murilo/analysis/figures/v5j/correct_confident_roahaw/` | Top-10 baseline for the "good crop" comparison |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights |
| `user/murilo/FINDINGS.md` | Cross-experiment history |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints |
| `user/murilo/experiments/028-domain-shift-augmentation/` | Paused experiment — diagnostics-first decision stands |
