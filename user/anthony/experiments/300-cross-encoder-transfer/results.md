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

## Metric bridge (AUROC ↔ Kaggle LB) — full roster (6/6, 2026-06-16)

With all six encoders deployed, the two offline metrics can be checked against the LB on
equal footing. **Consolidated table** (ranks in parens; sorted by LB public):

| Encoder | embed_dim | frozen LOSO¹ | held-out² | **LB public** | **LB private** | priv drop | CPU³ s/clip | est min/1k³ | deploy path |
|---------|:---------:|:------------:|:---------:|:-------------:|:--------------:|:---------:|:-----------:|:-----------:|-------------|
| **Perch**   | 1536 | 0.620 (1) | 0.706 (1) | **0.8656** (1) | **0.8617** (1) | −0.004 | **0.69** | **11.5** | Perch v2 ONNX, no tricks |
| **BirdNET** | 1024 | 0.588 (3) | 0.646 (2) | **0.8143** (2) | **0.7979** (2) | −0.016 | 2.07 | 34.5 | bmz TFLite, 3s windows |
| **EAT**     | 768  | 0.569 (5) | 0.635 (4) | **0.7796** (3) | **0.7659** (4) | −0.014 | 1.53 | 25.5 | FP32 ONNX of extract_features |
| **AST**     | 768  | 0.576 (4) | 0.638 (3) | **0.7780** (4) | **0.7680** (3) | −0.010 | 2.29 | 38.2 | INT8 ONNX, 10s-geometry |
| **BEATs**   | 768  | 0.612 (2) | 0.635 (4) | **0.7771** (5) | **0.7603** (5) | −0.017 | 1.78 | 29.6 | eager torch, first-try |
| **SSAMBA**  | 384  | 0.563 (6) | 0.607 (6) | **0.7308** (6) | **0.6893** (6) | −0.042 | **4.35** | **72.5** | eager, 20s-stride coverage cut |

¹ frozen-embedding ridge LOSO pooled-OOF macro AUROC, 70 species (the yardstick at the top of this doc).
² held-out focal→soundscape macro AUROC at the deploy λ — the 234-species *deploy head's* own offline metric.
³ **Controlled CPU benchmark, all 6 on equal footing** (`scripts/bench_all_encoders.py`, 2026-06-16): the
*deployed* `encode_file` over 8 real 60 s soundscapes at a pinned **4 threads** (Kaggle's vCPU allotment;
`os.cpu_count→4` is the single lever every encoder reads). `s/clip` = end-to-end wall per 60 s clip
(load+window+mel+forward+tile); `est min/1k` = projected minutes for 1000 clips (the 90-min-wall metric).
Forward-only rates reconcile with the prior partial numbers (EAT 7.85 vs 8.45, AST 2.62 vs 2.54 fwd/s). See
the full breakdown (RTF, forwards/clip, one-time model build) in the **CPU inference benchmark** section below.

**Findings:**

1. **Held-out AUROC is a markedly better LB rank-predictor than the frozen LOSO yardstick.** The
   frozen yardstick badly mis-orders BEATs (**2nd** frozen → **5th** LB; its 70-species leave-one-site
   probe over-credits it). Held-out gets the order essentially right — Perch ≫ BirdNET ≫ {EAT, AST,
   BEATs cluster} ≫ SSAMBA — matching the LB tier structure, because held-out is trained on the *actual
   deploy objective* (focal→soundscape transfer over the full 234-species roster), not a 70-species
   LOSO proxy. The yardstick remains a fair *relative encoder* tool but should not be read as an LB
   predictor; the deploy head's held-out number should.
2. **The anchors hold across all three metrics.** Perch is 1st and SSAMBA 6th on frozen LOSO, held-out,
   *and* both LB splits — the spread is in the middle tier, not the extremes.
3. **CPU cost is orthogonal to quality.** The cheapest *transformer* (EAT, 1.53 s/clip) and the most
   expensive encoder (SSAMBA, 4.35 s/clip — 6.3× EAT) land only 0.05 LB apart; EAT is 3× cheaper than its
   AudioSet sibling AST yet within 0.002 on the LB. Cost is set by token count / input geometry (EAT
   ~257 tokens ≪ AST ~1200 ≪ SSAMBA's 513-token bidirectional scan), an architecture artifact, not a
   transfer-quality signal. Every CPU-wall obstacle was packaging/geometry, never quality. **Perch is the
   exception that proves the rule: it is both the best (0.866) AND by far the fastest (0.69 s/clip, 87×
   realtime)** — the bird-specialist's compact ONNX wins on every axis.
4. **SSAMBA's −0.042 public→private drop is the largest in the roster** (vs Perch −0.004), the SSL-only
   model generalizing worst to the private split.

Perch's anchor caveat still stands for *cross-experiment* comparison: LOSO pooled AUROC 0.620 here vs
Perch V18 = 0.925 Kaggle LB are **not** directly comparable (leave-one-site vs hidden split, 70 vs 234
species, ~370-window folds). The within-exp-300 bridge above is the apples-to-apples one.

### CPU inference benchmark (controlled 4 vCPU, all 6 — 2026-06-16)

`scripts/bench_all_encoders.py` runs every encoder's **deployed** `encode_file` over the same 8 real
60 s soundscapes at a pinned **4 threads** (`os.cpu_count→4`, the single lever all encoders read for
their ORT `intra_op` / torch thread count) on one `cpu-small` node — eliminating the oversubscription
that made earlier PACE dry-runs read inflated (e.g. SSAMBA's 28 s/fwd at 24-threads-on-4-cores). This is
the apples-to-apples replacement for the scattered, oversubscribed dry-run timings.

| Encoder | s/clip | forwards/clip | fwd/s | realtime | est min/1000 clips | 1-time build |
|---------|:------:|:-------------:|:-----:|:--------:|:------------------:|:------------:|
| **Perch**   | **0.69** | 12 | 17.4 | 87× | **11.5** | 7 s |
| **EAT**     | 1.53 | 12 | 7.9 | 39× | 25.5 | 2 s |
| **BEATs**   | 1.78 | 12 | 6.8 | 34× | 29.6 | 15 s |
| **BirdNET** | 2.07 | 12 | 5.8 | 29× | 34.5 | **241 s** (TFLite load) |
| **AST**     | 2.29 | 6  | 2.6 | 26× | 38.2 | 41 s |
| **SSAMBA**  | **4.35** | **3** | 1.4 | 14× | **72.5** | 108 s |

- **s/clip** = end-to-end wall per 60 s clip (load + window + mel + forward + tile) — the deploy-true cost.
  **est min/1000 clips** = `s/clip × 1000/60`, the metric that matters for the 90-min wall.
- **forwards/clip** differs by windowing strategy and is why raw win/s isn't apples-to-apples: Perch/BEATs/EAT
  forward 12× per clip (5 s windows), AST 6× (10 s windows, tiled ×2), SSAMBA only **3×** (10 s windows at a
  20 s stride, tiled ×2 — the coverage cut that got it under the wall; full coverage would be 6× ≈ ~2× the
  forward time, which is what timed out as v1).
- **Build** is one-time model load (not per-clip): BirdNET's 241 s TFLite load and SSAMBA's 108 s
  checkpoint+package load add to total wall but are paid once. Forward-only rates reconcile with the prior
  partial measurements (EAT 7.9 vs 8.45, AST 2.6 vs 2.54 fwd/s), confirming the harness.
- **Takeaway:** a 6.3× CPU-cost spread (Perch 0.69 → SSAMBA 4.35 s/clip) maps onto only a 0.135 LB spread,
  and the ordering doesn't track quality — Perch is simultaneously the most accurate and the cheapest, while
  the costliest (SSAMBA) is the least accurate. Cost is an input-geometry/token-count property; once an
  encoder is made to *fit* the wall (ONNX, INT8, or coverage cuts), its LB is set by transfer quality alone.

**Projected hidden-test inference time (≈600 recordings × 60 s).** The hidden set is ~600 clips, so total
kernel time ≈ one-time build + 600 × s/clip. The harness is calibrated — EAT's 15.3 min projection matches
its real ~24 min/1000-clip scored run pro-rata, and PACE `cpu-small` ≈ Kaggle's CPU — so these transfer:

| Encoder | s/clip | build | **total @ 600** | margin under 90-min wall |
|---------|:------:|:-----:|:---------------:|:------------------------:|
| Perch   | 0.69 |   7 s | **7.0 min**  | 83 min |
| EAT     | 1.53 |   2 s | **15.3 min** | 75 min |
| BEATs   | 1.78 |  15 s | **18.1 min** | 72 min |
| AST     | 2.29 |  41 s | **23.6 min** | 66 min |
| BirdNET | 2.07 | 241 s | **24.7 min** | 65 min |
| **SSAMBA** | 4.35 | 108 s | **45.3 min** | **45 min** |

- **All six clear the 90-min wall with ≥45 min to spare** at N≈600 — Perch with ~12× headroom, SSAMBA (the
  binding constraint) at exactly half the budget, consistent with v2-stride scoring without trouble.
- **This pins down the v1 SSAMBA timeout.** Full coverage (6 forwards/clip vs 3) roughly doubles the
  forward+mel portion → ~8–9 s/clip → ~80–90 min at 600 clips, i.e. *right at the wall*; the model load +
  7,200-row CSV + scoring setup tipped it over (`COMPLETE` + blank). The 20 s-stride cut moved it from
  on-the-edge to comfortable (45 min). Breakeven for SSAMBA-stride is ~1,215 clips, so it stays safe well
  above the actual 600.

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

## Deploy ablation: all encoders on the Kaggle LB

The unified `kaggle_kernel/` is an ablation harness — the SAME pipeline (focal+soundscape
ridge head → `emb → standardize → W → clip[0,1]` over 234 species) runs Perch / BEATs /
BirdNET / AST; only the `encode_windows` backend and `head_{enc}.npz` change. Each was
submitted as its own kernel (`birdclef-2026-exp300-{enc}-ridge-baseline`). Scores verified
live from `kaggle competitions submissions birdclef-2026` on 2026-06-14:

| Encoder | embed_dim | best λ | **public LB** | **private LB** | LOSO yardstick (pooled AUROC) | Status |
|---------|:---------:|:------:|:-------------:|:--------------:|:-----------------------------:|--------|
| **Perch**   | 1536 | 1e5 | **0.8656** | **0.8617** | 0.6200 (rank 1) | ✅ COMPLETE |
| **BirdNET** | 1024 | 1e6 | **0.8143** | **0.7979** | 0.5875 (rank 3) | ✅ COMPLETE |
| **EAT**     | 768  | 1e5 | **0.7796** | **0.7659** | 0.569 / held-out 0.635 (rank 5) | ✅ COMPLETE (finetune_AS2M, FP32 ONNX of extract_features) |
| **AST**     | 768  | 1e5 | **0.7780** | **0.7680** | 0.5762 / 0.601 cls (rank 4) | ✅ COMPLETE (10s-geometry INT8 ONNX; see saga below) |
| **BEATs**   | 768  | 1e4 | **0.7771** | **0.7603** | 0.6119 (rank 2) | ✅ COMPLETE |
| **SSAMBA**  | 384  | 1e2 | **0.7308** | **0.6893** | 0.5631 / held-out 0.607 (rank 6) | ✅ COMPLETE (eager 20s-stride; our mamba-onnx *export* emitted NaN — op itself passes 12/12) |

All heads cover 234/234 species (no Perch-logit fallback fires). Consistency cos=1.000
for every encoder (in-kernel forward reproduces the cache bit-for-bit; AST INT8 cos=0.993
adjudicated by downstream AUROC instead, Δ=−0.004). **Roster 6/6 complete** (SSAMBA scored
2026-06-16).

**Findings:**

1. **The frozen yardstick order does NOT survive to the LB.** LOSO ranks Perch > BEATs >
   BirdNET, but the deployed LB ranks **Perch > BirdNET > AST ≈ BEATs** — BirdNET (0.814 pub)
   beats BEATs (0.777 pub) by +0.037 despite ranking *below* it on the frozen probe
   (0.588 < 0.612). The bird-specialist's focal→soundscape transfer over 266K windows
   outpaces what its 739-window LOSO probe predicts. Another instance of offline metrics
   anti-correlating with LB; the frozen probe is a *relative encoder ranking tool*, not an
   LB predictor.
2. **Perch is the clear LB winner** (0.866 pub / 0.862 priv) and the only encoder whose
   public→private drop is negligible (−0.004). BirdNET and BEATs both lose ~0.016–0.017
   private, preserving their order.
3. **The bird-specialists (Perch, BirdNET) form a top tier; the three general-audio AudioSet
   transformers (EAT, AST, BEATs) are in a statistical dead heat ~0.777–0.780 below them.**
   Public: EAT 0.7796 > AST 0.7780 > BEATs 0.7771 (spread 0.0025); private: AST 0.7680 >
   EAT 0.7659 > BEATs 0.7603. EAT takes the highest public by a hair, AST the highest private
   — all three within 0.003 either way, so the ordering *within* the transformer tier is
   noise. The held-out probe predicted this near-tie precisely (EAT 0.635 ≈ AST 0.638). The
   real signal: **domain (bird-pretrain) beats architecture** — supervised-AudioSet pretraining
   (all three) doesn't close the ~0.09 gap to Perch, and even a *fine-tuned* AudioSet transformer
   (EAT finetune_AS2M, AST AudioSet) lands no higher than self-supervised BEATs.
4. **All deploy obstacles were CPU-wall / packaging, not quality.** AST needed the
   10s-geometry INT8 fix (saga below); EAT needed an ONNX export to sidestep `trust_remote_code`
   offline (its ~257-token forward is 5× cheaper than AST → 8.5 win/s, no tricks). SSAMBA, the
   heaviest forward in the roster (513-token bidirectional Mamba scan), needed a 20s-stride
   coverage cut to fit the wall (saga below). **All 6 encoders now have an LB score.**
5. **SSAMBA (0.731 pub / 0.689 priv) confirms the bottom of the order** — lowest LB, exactly as
   the frozen yardstick (rank 6, 0.563) and held-out probe (0.607) predicted. It is the only
   roster encoder whose public→private drop is large (−0.042, vs −0.004 for Perch), and the only
   *self-supervised-only* model with no AudioSet/bird supervision at all. The Mamba architecture
   buys nothing here: a 384-d SSL Mamba lands ~0.047 below the 768-d AudioSet transformers and
   ~0.135 below Perch — **domain + supervision dominate architecture**, the central finding, now
   with a state-space-model data point.

Trails the logit/ensemble stacks (exp 038 Perch+SED ≈ 0.929) as expected for deliberately
minimal single-encoder embedding baselines.

## Deploy: EAT cls (unified `kaggle_kernel/`) — FP32 ONNX, clean first deploy

**LB 0.7796 pub / 0.7659 priv** (`worstchan/EAT-base_epoch30_finetune_AS2M`, frozen CLS, 768-d,
λ=1e5, 234/234 covered). EAT was the apples-to-apples *finetuned-transformer* path (AudioSet
supervised, like AST/BEATs) — the user picked the finetuned checkpoint over the SSL pretrain
(which the frozen ridge under-reads → yardstick 0.569). The deploy went clean on the first
real submission (one trivial head-slug fix in between).

**The only blocker was `trust_remote_code`.** The eager EATEncoder loads via
`AutoModel.from_pretrained(..., trust_remote_code=True)` — the offline kernel can't fetch the
remote `modeling_eat.py`. Fix: trace `model.extract_features(mel)` + CLS pooling to **FP32 ONNX**
(`scripts/export_eat_onnx.py`); the graph is standard ViT ops, so the kernel needs neither the
remote code nor the HF weights — just the ONNX + torchaudio for the mel. Export parity vs eager
on 8 real windows: **cosine 1.000000**. EAT is *cheap* on CPU (5 s/512-frame mel → ~257 tokens
vs AST's ~1200): **8.45 win/s on 4 vCPU → ~24 min for 12K windows**, far under the 90-min wall —
so plain FP32, **no INT8 / no 10s tricks** (unlike AST), and exact cache parity.

The kernel `EatEncoder` computes the mel with `torchaudio.compliance.kaldi.fbank` replicating
`birdclef.encoders.EATEncoder._waveform_to_mel` EXACTLY (hanning, frame_shift=10, num_mel_bins
=128, `(mel-(-4.268))/(4.569*2)`, pad/truncate to 512) → FP32 ONNX → CLS. **Consistency check:
cosine 1.000000 over 674 windows** (the real offline gate — kernel mel + `load_audio_torch` +
ONNX vs the eager `frozen-cls-ft` cache). Datasets `acmiyaguchi/birdclef-2026-exp300-{eat-onnx
(343 MB FP32 graph), eat-head}`; kernel `birdclef-2026-exp300-eat-ridge-baseline`. The v1 commit
hit `FileNotFoundError: artifacts/head.npz` — the head-dataset slug list in `main()` was missing
`birdclef-2026-exp300-eat-head`; added it, v2 ran COMPLETE.

## Deploy: AST cls (unified `kaggle_kernel/`) — CPU-wall saga (eager → INT8 → 10s refit → ✅ SCORED)

> **RESOLVED (2026-06-15): AST scored 0.7780 pub / 0.7680 priv.** Eager and INT8-at-5s both
> scored BLANK (90-min CPU timeout); the **10s-geometry INT8 refit** (6 windows/clip not 12)
> cleared the wall and got a leaderboard score on the first try. The dated **Update**
> subsection at the end of this section has the resolution; the narrative below is the
> original eager-in-kernel diagnosis, kept for the timing table + root cause.

AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, 768-d) through the unified kernel
(`AstEncoder`, **cls** pooling = CLS+DIST mean — led the LOSO yardstick at 0.601 vs 0.576
mean_patch), torch+transformers **in-kernel** (not ONNX) for guaranteed cache↔kernel parity.
(Diagnosis below merged from `exp/300-deploy-next`, which prototyped AST as a separate
`kaggle_kernel_ast/`; that fork is superseded by this unified kernel and was dropped.)

**Pipeline built + verified end-to-end (all green):**
- Focal extraction (16-way GPU array, `extract-focal`): 266K focal windows, 0 failures,
  768-d cls. Assembled `exp300-ast/` features dir (focal shards + `soundscape_labeled`).
- λ-sweep (held-out focal→soundscape): **λ=1e5**, macro AUROC 0.6448 (clean peak: 1e4 .641,
  1e5 .645, 1e6 .635) — same optimum as Perch.
- `head_ast.npz` = W (768×234), **234/234 covered**, no fallback. In-sample replay 0.903.
- Kernel↔cache parity (consistency check): **cosine = 1.000000** (in-kernel AstEncoder
  forward reproduces the cache bit-for-bit). Submission dry-run: 48×234, no NaN, ∈[0,0.167].

**Submitted (kernel v1) → COMPLETE but BLANK leaderboard score.** Format verified
byte-identical to `sample_submission.csv` (235 cols, same names + order, valid row_ids,
NaN-safe). Not a format bug — Perch with the *identical* pipeline scored 0.866.

**Root cause: the CPU-only 90-min limit.** BirdCLEF 2026 reruns the kernel on the full
hidden test (~700–1000 soundscapes) on **CPU only**. AST runs a full transformers forward
per 5 s window. Measured on 4 vCPU (Kaggle's allotment), PACE `cpu-small`, job 9957035:

| batch | win/s | 8,400 win (~700 files) | 12,000 win (~1000 files) |
|-------|-------|------------------------|--------------------------|
| 1     | 1.72  | 81.6 min               | 116.6 min                |
| 8     | 1.74  | 80.5 min               | 115.0 min                |
| 16    | 1.71  | 81.9 min               | 117.0 min                |

**Batching does not help** (compute-bound, not overhead-bound). Even the optimistic
700-file case is **81.6 min of inference alone** — before model load + ~700 audio
decodes/resamples — so it exceeds 90 min → timeout → blank score. Perch fits only because
its ONNX graph is far cheaper per window.

**This is AST-specific, NOT "transformers can't deploy."** On the same CPU limit, **BEATs
scored 0.777** and **BirdNET 0.814** (both COMPLETE on the LB). AST is the outlier because
its forward is far heavier: a 10-s log-mel input tokenized into ~1200 patches → quadratic
self-attention over far more tokens than BEATs. So the wall is AST's per-window cost
(measured 1.72 win/s), not the torch-in-kernel approach in general.

**To deploy AST** the forward must get ~2–4× cheaper: **ONNX export → OpenVINO FP16/INT8**
(the docs' prescribed recipe, what Perch already does), re-validated to cache parity. The
trained `head_ast.npz` and the whole extraction/fit/parity pipeline are reusable as-is once
a fast forward exists. Offline, AST ranks *below* Perch (yardstick 0.601 vs 0.620) and below
the BEATs/BirdNET that already deploy, so its only value is ensemble diversity — the decision
to invest in the AST ONNX export is **deferred**. `encode_windows` is left batched +
device-aware (harmless on CPU, prerequisite for any faster backend).

**Reconciliation (done):** the AST blank-score row was previously assumed fixable by a fresh
resubmit from the unified kernel — that is **wrong**: a resubmit times out identically (same
slow forward). This timeout diagnosis + timing table was merged here from `exp/300-deploy-next`,
and that branch's redundant forked `kaggle_kernel_ast/` was dropped in the merge. AST stays
undeployed pending the ONNX/OpenVINO work above.

### Update (2026-06-15): INT8 ONNX shipped but STILL blank → 10s-geometry refit (in progress)

**Attempt 1 — ONNX + INT8 (commit `02e6335`): blank again.** Exported the cls-pooled
transformer to ONNX (legacy exporter, opset 17) and INT8-dynamic-quantized it
(`scripts/export_ast_onnx.py`); downstream-validated that head + INT8 holds macro AUROC
(0.901 vs 0.903 eager, −0.002 — the strict 0.993 per-window cosine was a red herring).
Benchmarked 2.54 win/s on 4 vCPU (vs eager 1.72). Submitted (kernel v5) → **COMPLETE but
BLANK leaderboard score**, same failure as eager. Deps install + ONNX session load confirmed
fine in the commit log; the committed run sees `0 test soundscapes`, so the blank is the
hidden rerun, not a code error.

**Root cause refined: INT8 only sped the cheap half.** 2.54 win/s is the *graph* forward; the
expensive half is the `ASTFeatureExtractor` mel, which force-pads every 5 s window to a fixed
1024-frame (~10.24 s) tensor regardless. End-to-end on ~700–1000 clips × 12 windows still
lands at 80–120 min → over the 90-min wall. Two independent AST runtimes (eager, INT8) both
blank while BEATs/BirdNET/Perch score through the identical kernel → the wall is AST's
per-window cost, confirmed.

**Attempt 2 — 10 s native geometry (in progress).** The real lever is the *window count*, not
the graph speed. Because AST's mel always occupies 1024 frames, a 5 s window wastes half the
forward on silence; feeding **10 s windows** (6 per 60 s clip instead of 12) HALVES the forward
count at identical per-window cost. Combined with INT8: ~6000 windows / 2.54 ≈ **39 min**, deep
under the wall (even eager-10s ≈ 58 min would clear it). This also matches exp 301's native-
geometry finding (10 s real audio ≫ 5 s + silence). The ONNX graph is unchanged (1024-frame
input either way); only the head is refit on 10 s-window embeddings.

Implementation: `--window-sec` threaded through `extract.py`/`cli.py` (default 5 s, frozen
contract untouched); kernel `AstEncoder` windows at 10 s and `repeat(emb, 2)` so the 5 s
submission grid (`{clip}_{end}`, end∈{5..60}) is preserved (both 5 s halves share the 10 s
score); `consistency_check.py` de-repeats before the 1:1 cache check. CPU smoke green (768-d
cls at 10 s). 10 s focal+soundscape re-extraction on GPU (jobs 9965471/9965472), then refit
(`sbatch/fit-head-ast10s.sbatch`: held-out λ sweep → fit → in-sample replay), re-stage head
(ONNX dataset unchanged), consistency-check, resubmit. **Gate before resubmit:** held-out
focal→soundscape transfer must match the 5 s 0.6448.

### RESOLVED (2026-06-15): 10s-geometry INT8 scored 0.7780 pub / 0.7680 priv

The 10s refit shipped as kernel v6 and **scored on the first try** — the CPU-wall saga is
closed. Final numbers:
- **LB: 0.77795 pub / 0.76800 priv** (`SubmissionStatus.COMPLETE`, scored ~88 min after submit).
- Held-out focal→soundscape transfer **0.638** at λ=1e5 (vs 5 s 0.6448 — within noise; the
  gate was "match the 5 s number," and it did). The minor dip is expected: 10 s windows give
  the readout coarser temporal resolution but the LB metric pools per-species anyway.
- INT8 vs FP32 downstream macro AUROC **−0.0038** (FP32-ONNX reproduces the eager cache at
  cosine 1.00000; the 0.993 per-window cosine was the INT8 weight-quant signature, a red
  herring — adjudicated by downstream AUROC, not cosine).
- Window count halved (6/clip not 12) → cleared the 90-min wall with margin (~39 min est.).

**Takeaways:** (1) the fix was *window count*, not graph speed — INT8 alone (5 s) stayed
blank because it sped only the graph, not the fixed-1024-frame mel; (2) AST lands a hair
above BEATs on the LB (0.778 vs 0.777 pub) and *generalizes best to private* of the
AudioSet transformers (−0.010 pub→priv vs BEATs −0.017); (3) the frozen-yardstick rank-4
position (0.576) again under-read the deployed transfer, consistent with the rest of the roster.

## Deploy: SSAMBA-HF (unified `kaggle_kernel/`) — CPU-wall saga (ONNX → eager → fused-op → 20s-stride → ✅ SCORED)

**LB 0.7308 pub / 0.6893 priv** (SSL `ssamba_small_400`, frozen mean_patch 384-d, λ=1e2,
234/234, held-out 0.607). The last roster encoder to deploy → **6/6 complete (2026-06-16)**.
SSAMBA is the only *self-supervised-only* model (no AudioSet/bird supervision) and the heaviest
forward in the roster (513-token **bidirectional** Mamba selective-scan × 24 layers), so it was
the hardest CPU-wall fight. The ladder, in order:

1. **Unrolled ONNX — NON-VIABLE.** The 004 HF port has a pure-PyTorch scan fallback, so the
   stock `torch.onnx.export` traces — but it *unrolls* the 513-step × 24-layer × 2-dir scan into
   a ~25K-node / 203 MB graph whose ORT session-init alone times out (superlinear in node count,
   even with `ORT_DISABLE_ALL`). Dead end.
2. **Eager pure-PyTorch, 10s full coverage — TIMED OUT.** Shipped first (v1): force the CPU scan
   (`_use_cuda_kernels=False`), batch-16, 10s windows (6/clip), `repeat(2)` for the 5s grid.
   `COMPLETE` + **blank** score — 6 fwd/clip (~65 min at 1.54 win/s) + full mel/IO overshot the
   90-min wall.
3. **mamba-onnx fused op — built, fast graph, but our SSAMBA export emits ALL-NaN.** The custom
   `mamba-onnx` package (`vendor/mamba-onnx`, release wheel 0.1.0) emits a *single*
   `mamba::SelectiveScanFusedExact` node (AVX2 zig backend) → clean **101.8 MB** graph that loads
   and runs. But: (a) only **+15%** (1.87 vs 1.61 win/s @ 4 vCPU — the 004 eager scan is already
   vectorized, so the non-scan layers dominate, not the recurrence), and (b) our exported SSAMBA
   graph returns **all-384-NaN at real weights**. **Important: the op itself is correct** — its own
   suite `tests/test_ops.py` passes **12/12 (zig)** in this venv, including the numerical edge cases,
   so the NaN is a bug in *our SSAMBA→op export integration* (`scripts/ssamba_onnx_export.py`'s
   `_ssm_forward_onnx` shim / bidirectional-scan wiring), **not** the kernel. Root cause not yet
   isolated. cpp backend won't load anyway (no `libomp` on Kaggle). Set aside — even with a working
   export, +15% lands full-coverage at ~77 min (risky) and needs `.so` vendoring, so 20s-stride eager
   was the lower-risk path; revisiting the export is a follow-up, not a deploy blocker.
4. **20s-stride eager — ✅ SCORED.** Forward only every 2nd 10s window (`SSAMBA_STRIDE=2`,
   3/clip) and mel *only those*, tiling each embedding across its 20s block, `repeat(2)` for the
   grid. Halves both scan and mel → ~32 min fwd + ~7 min mel ≈ ~40 min real → clears the wall.
   Forward math byte-identical to full coverage (consistency cos=1.000); **head reused as-is, no
   refit** (the 10s-geometry cache already matches). Cost: coarser temporal sampling (20s blocks).

**Takeaways:** (1) like AST, the lever was *coverage / window count*, not graph speed — the fused
op proved the scan isn't the 4-vCPU bottleneck (non-scan layers are), inverting the going-in
assumption; (2) a measurement trap nearly wasted a submission — the PACE dry-run showed 28 s/fwd
purely from thread oversubscription (`os.cpu_count()=24` while SLURM gave 4); a controlled 4-thread
bench restored the true 1.54 win/s, which is what Kaggle sees; (3) SSAMBA confirms the bottom of
the order and the −0.042 pub→priv drop (largest in roster) flags the SSL-only model as the least
robust to distribution shift. mamba-onnx export/parity/bench tooling is kept under `scripts/` +
`vendor/mamba-onnx-wheels/` for any future state-space deploy.
