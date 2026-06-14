# Experiment 300 Results

**Status**: Complete (2026-06-11); eval hub later extended for exp 301 (2026-06-13).
Ran end-to-end on PACE: 6 encoders extracted to the unified cache, ridge LOSO probe over
all of them, plus a token-pooling sensitivity pass. Source artifacts:
`artifacts/probe_metrics_frozen.json` (+ per encoder `probe_{enc}_frozen.json`) and
`artifacts/probe_metrics_frozen-cls.json`. The frozen ridge ranking below is unchanged;
the prototypical readout + per-patch cache added for 301's masked-SSL DAPT (and the proto
frozen→adapted Δ tables) live in **exp 301's `results.md`**, not here.

## Headline: frozen-embedding ridge LOSO ranking

739 deduped labeled soundscape windows (5 s @ 16 kHz), 70 species, leave-one-site-out,
per-fold z-score standardization. **Primary metric = pooled-OOF macro ROC-AUC** —
one AUROC per species computed over all out-of-fold predictions pooled across sites,
then macro-averaged (skipping classes with no positives). This matches the BirdCLEF
leaderboard metric. λ selected on this metric over the grid `[1e-3 … 1e5]`.

| Rank | Encoder   | Backend     | embed_dim | **macro AUROC (LB-aligned)** | fold-avg AUROC | macro AP | best λ |
|------|-----------|-------------|-----------|:---------------------------:|:--------------:|:--------:|:------:|
| 1    | perch     | perch_cache | 1536      | **0.6200**                  | 0.6938         | 0.1437   | 1e4    |
| 2    | beats     | torch       | 768       | **0.6119**                  | 0.6742         | 0.1187   | 1e4    |
| 3    | birdnet   | birdnet_bmz | 1024      | **0.5875**                  | 0.6405         | 0.1259   | 1e4    |
| 4    | ast       | torch       | 768       | 0.5762                      | 0.6092         | 0.1185   | 10     |
| 5    | eat       | torch       | 768       | 0.5689                      | 0.6216         | 0.1099   | 1000   |
| 6    | ssamba_hf | torch       | 384       | 0.5625                      | 0.5777         | 0.1031   | 1e4    |

**Findings:**

1. **Perch (bird specialist) leads, but only narrowly over BEATs** (general AudioSet
   encoder): 0.620 vs 0.612. The gap between a bioacoustic-specialist foundation model
   and a general audio one is ~0.008 on frozen features — small.
2. **BirdNET is 3rd, not last.** This is the headline nuance for §6 and depends
   entirely on the metric (see below). Under the LB-aligned pooled metric the bird
   specialist BirdNET ranks above both transformers AST and EAT.
3. **The transformers (AST, EAT) and SSAMBA cluster at the bottom**, 0.563–0.576 —
   AudioSet-pretrained transformers transfer worse to the bird soundscape task than
   either bioacoustic specialist or the masked-prediction BEATs.

## The metric matters: pooled (LB) vs per-fold aggregation reorders the field

The probe's *original* headline averaged AUROC **within each site, then across sites,
then species** — a double-average the leaderboard never does. Switching to the
LB definition (pooled-OOF, above) changes both the absolute numbers (~0.06 lower) and
**the ranking**:

| Encoder   | pooled-OOF (LB) | rank | per-fold avg | rank |
|-----------|:---------------:|:----:|:------------:|:----:|
| perch     | 0.6200          | 1    | 0.6938       | 1    |
| beats     | 0.6119          | 2    | 0.6742       | 2    |
| birdnet   | 0.5875          | **3**| 0.6405       | 3*   |
| ast       | 0.5762          | 4    | 0.6092       | 5*   |
| eat       | 0.5689          | 5    | 0.6216       | 4*   |
| ssamba_hf | 0.5625          | 6    | 0.5777       | 6    |

\* The per-fold column above is at the λ selected on the *pooled* metric; under the
original run (λ selected on per-fold) BirdNET ranked **last** (6th). Either way the
two estimators disagree, and BirdNET is the encoder most sensitive to the choice: it
ranks species well *globally* (pooled across sites) but poorly *within a single site*,
so the within-site average punishes it. The pooled metric is the one that mirrors how
the Kaggle LB scores, so it is the primary; `fold_avg_auroc` is retained as a
within-site-consistency diagnostic only.

## Token-pooling sensitivity (CLS vs mean-patch)

The torch encoders emit a patch-token grid per window; we reduce it to one vector.
Primary convention: **mean-pool patch tokens** (HEAR-benchmark style), uniform across
the roster — BEATs/SSAMBA have no CLS token, so CLS-only is not a roster-wide option.
For the two encoders that *do* expose a CLS token we measured both (variant `frozen-cls`):

| Encoder | mean_patch (primary) | cls   | Δ (cls − mean_patch) |
|---------|:--------------------:|:-----:|:--------------------:|
| ast     | 0.5762               | 0.6006| **+0.024 (cls better)** |
| eat     | 0.5689               | 0.5606| −0.008 (mean_patch better) |

Mixed result under the LB-aligned metric: **CLS helps AST** (and AST-cls at 0.6006
would slot into 3rd, above BirdNET), but **mean_patch still wins for EAT** — even
though EAT's utterance objective trains its CLS token as the clip summary. Differences
are small (~0.02) and noisy on 70 species. We keep **uniform mean_patch** as the fair
primary convention and report CLS as a sensitivity data point; it does not change the
top of the ranking.

(Note: under the earlier per-fold metric, CLS lost for *both* encoders — another way
the aggregation choice changes the story. The numbers above are LB-aligned.)

## λ regularization

With per-fold standardization the optimum sits high: 4 of 6 encoders select λ = 1e4
(the old grid ceiling of 1000 *was* slightly constraining for the pooled metric, so
the grid was extended to 1e5). AST selects 10 and EAT 1000. Past the optimum AUROC
declines monotonically (e.g. eat-cls 0.561 → 0.540 → 0.490 at 1e3/1e4/1e5), confirming
the grid now brackets the optimum rather than railing.

## Metric bridge (AUROC ↔ Kaggle LB) — open

Perch is the shared anchor: LOSO pooled AUROC 0.620 here vs Perch V18 = 0.925 Kaggle LB.
The two are **not directly comparable** — different generalization axis (leave-one-site
vs hidden random/temporal split), different species support (70 species present in the
labeled windows vs the full 234-species roster, ~159 of which have zero labeled
soundscape data), and a tiny per-fold training set (~370 windows). The probe is a
**relative yardstick for ranking encoders on equal footing**, not an LB predictor;
this is consistent with the project finding that offline metrics anti-correlate with LB.
Bridging the two axes is deferred to the working-note §6 metric-bridge analysis.

### Reconciliation with exp 106 (Perch 0.749)

Exp 106 reported Perch ridge AUROC ≈ 0.749 (embedding + empirical-Bayes prior track).
This probe reads 0.620 pooled / 0.694 fold-avg for Perch. The fold-avg 0.694 is the
closer comparison; the residual gap is attributable to (a) exp 106's EB prior vs this
probe's identity prior, (b) exp 300's self-contained 70-species list, and (c) the
metric-aggregation difference documented above. The number to cite for LB-alignment is
the pooled 0.620; exp 106's 0.749 is a different estimator and should not be read as the
LB-aligned figure.

## Reproduction notes

- Encoders/variants live in the unified cache under
  `~/scratch/birdclef-2026/features/exp300/embeddings/encoder={enc}/variant={var}/source=soundscape-labeled/data.parquet`.
- First run shook out 6 venv dependency bugs (commit `7b45f0a`): matched
  torch/torchaudio/torchvision trio, mamba-ssm `--no-build-isolation`, transformers<5
  (EAT remote code), timm + pandas, ai-edge-litert (BirdNET tflite).
- Token pooling is configurable per encoder (`pooling: mean_patch|cls`, commit
  `e6a4b29`); `--pooling cls` + a distinct `--variant` tag produces the sensitivity set.
- Probe metric/grid (commit `6e0cf61`): pooled-OOF macro ROC-AUC primary, λ grid to 1e5.
- Window count: 739 deduped labeled windows (the "749" in the proposal is pre-dedup).

## Deploy: Perch Kaggle baseline (`deploy/` + `kaggle_kernel/`)

The probe only does LOSO CV; for a submittable baseline we fit one final head on ALL
training data and ship it. New code: `src/birdclef_300/deploy/{fit_head,species_map}.py`,
`birdclef-300 fit-head`, `sbatch/fit-head.sbatch`, `kaggle_kernel/submission.py`.

**Architecture (backbone-agnostic).** `emb → standardize(mean,std) → W → clip[0,1]` over
the full 234-species roster. For species the head can't cover, Perch's native logits are
an *optional Perch-only* fallback — but see below, for Perch+focal it never fires. Swapping
backbones = swap the encoder + retrain `head.npz`; the kernel CSV logic is unchanged.

**Head fit** (job 9932535, focal+soundscape, λ=1e4, 266,663 windows, ~3 min CPU):
- `head.npz` = W (1536×234) + mean/std + species_list + covered mask.
- **234/234 species covered** by training labels (median 728 positives; 203 with ≥25) —
  even the 25 insect sonotypes appear via soundscape/secondary labels, so the Perch
  fallback is dead code here. This is the answer to "does the head cover all species?":
  yes, focal+soundscape covers the whole roster without needing Perch logits.

**Honest validation.** The in-process replay evaluates on the 739 soundscapes:
- focal+soundscape head → ~0.95, but that is **IN-SAMPLE** (soundscape was in the fit) —
  a pipeline smoke test only, not a generalization estimate.
- focal-only head → **HELD-OUT** pooled macro AUROC over 70 species (honest focal→soundscape
  transfer). Beats the LOSO 0.620 yardstick because 266K focal windows train far better
  per-species models than the ~370-window LOSO folds. Consistent with exp 106's focal track.

**λ tuning** (`deploy/sweep_lambda.py`, job 9932981). λ=1e4 was carried from the
soundscape-only LOSO and under-regularizes the 266K-window fit. Held-out sweep (focal-only
fit, eval on the 739 soundscapes):

| λ | 1e2 | 1e3 | 1e4 | **1e5** | 1e6 | 1e7 | 1e8 |
|---|-----|-----|-----|---------|-----|-----|-----|
| held-out AUROC | .690 | .690 | .694 | **.706** | .704 | .695 | .688 |

**λ=1e5 is the selected default** (+0.012 over 1e4, cleanly bracketed). The deployed
`head.npz` (job 9933036) is refit on focal+soundscape at λ=1e5.

**Kernel dry-run** (job 9932812, 4 soundscapes via the SAME Perch ONNX `justinchuby/
Perch-onnx perch_v2_no_dft.onnx` used for the exp 106 cache → no SavedModel↔ONNX drift):
48 rows × 234 cols, no NaN, probs ∈ [0,1], max 0.528, ~126 nonzero species/row. PASS.

**Kaggle LB (submitted 2026-06-14, kernel `birdclef-2026-exp300-perch-ridge-baseline` v2):**
**public 0.8656 / private 0.8617.** Beats exp 106's ridge (0.804 LB) by +0.062 — the
focal+soundscape fit over the full roster at tuned λ=1e5 (embeddings, not logits) is a
real step up. Trails the logit-based/ensemble stacks (Murilo 018 = 0.893; exp 038
Perch+SED ≈ 0.929), as expected for a deliberately minimal single-encoder baseline.
Another instance of offline ≪ LB (held-out ~0.71 → LB 0.866).

Inputs: Kaggle datasets `acmiyaguchi/{birdclef-2026-exp300-head, perch-v2-onnx}` +
`birdclef-2025-deps` (onnxruntime/librosa cp312 wheels). The kernel runs the SAME ONNX
that built the exp 106 cache, so the head transfers with no SavedModel↔ONNX drift.

**Next (backbone transfer):** the whole point is that this pipeline is encoder-agnostic.
To extend to BEATs/EAT/SSAMBA/BirdNET: extract their focal+soundscape embeddings into the
unified cache, `fit-head` per encoder (sweep λ — it under-regularizes at 1e4), swap
`encode_windows` in the kernel. The Perch fallback is Perch-only and unneeded here
(234/234 covered).
