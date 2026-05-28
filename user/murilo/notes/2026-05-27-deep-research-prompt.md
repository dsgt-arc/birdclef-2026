# Deep Research Prompt — Pivot Decision after Exp 033 REJECT (2026-05-27)

**Purpose**: Independent assessment of the BirdCLEF 2026 pivot decision after exp 033's REJECT verdict. Designed to **challenge** the project's current direction (Sydorskyi backbone yoink), not validate it.

**How to use**: Paste the body below into Claude.ai's deep research interface. Attach the listed companion docs (or paste their contents). Fire it overnight; review the output tomorrow morning along with the a2_kd_v3 LB result.

**Why this prompt is shaped this way**: Our prior research docs (2026-04-13 Claude + Gemini, 2026-04-11 Claude) all answered "what should I try?" — they were strategy-from-zero. We now have specific failures (exp 033 REJECT + 24 data points of offline-LB anti-correlation) and a specific candidate pivot (Sydorskyi-yoink). This prompt asks for independent ranking against that landscape, not new ideas from scratch.

---

## The prompt body (paste this into Claude.ai's deep research)

I'm in the BirdCLEF 2026 Kaggle competition. **7 days remaining** before the 2026-06-03 deadline. **5 LB submissions/day** quota, with 2 already used today on diagnostic runs.

My goal: break **LB 0.94** with the highest probability of success in the remaining time. Current best LB is 0.929 (`exp 023 perch-sed-ensemble-mlp-only`). Best single model is 0.906 (`exp 032 distilled-sed-kd` Phase D, fold 0).

I need an **independent assessment** — not validation of my current plan. Challenge the assumptions. Identify whether I'm wrong about anything before I commit to a 3-4 day pivot.

## What I've done (verbatim, including failures)

1. **Exp 032: Tucker distilled-SED replication.**
   - EffNet-B0 NS-JFT + AttHeadLogitSpace + dual-supervision BCE + Perch v2 embedding KD + Tucker mel + 5-fold.
   - Single fold LB 0.906 (Phase D), 5-fold ensemble LB 0.915, HGNet backbone ablation LB 0.901 (backbone refuted as the bottleneck).
   - Verdict: PARTIAL REPLICATION (within 0.002 of Tucker's published 0.917 single-fold).

2. **Exp 033: 1:1 own-teacher pseudo-labeling on distilled-sed-kd.**
   - 10 ablations: 2 teachers (v5j-4fold LB 0.913, distilled-sed-kd-5fold LB 0.915) × 5 protocol cells (init × sampler × focal/PL ratio).
   - All 10 trained to full max_epochs; D.5 e2e validation all 10 PASS.
   - 3 LB submissions, all REJECT (Δ ≤ 0 vs 0.906 baseline):
     - a4-kd (warm + Uniform + T2): LB 0.899 (Δ −0.007)
     - a2-kd ⭐ (random + Uniform + T2, canonical Antoine): LB 0.886 (Δ −0.020)
     - a4-v5j (warm + Uniform + T1): LB 0.897 (Δ −0.009)
   - Paired teacher-axis LB Δ (a4-kd vs a4-v5j): +0.002 (T2 over T1). Refutes the proposal's "teacher diversity" hypothesis at LB scale (T2 won offline by +0.011 auroc).

3. **Audit of exp 033 vs Sydorskyi 2025 2nd place protocol** (the only published 0.93+ PL recipe with line-by-line code).
   - **16 of 20 protocol axes diverge** between our implementation and Sydorskyi's published mechanism.
   - Top 5 high-impact divergences:
     - PL × focal MixUp fusion missing (Sydorskyi: focal hard + PL soft → 50/50 audio mix + sum-clip targets → implicit distillation. Us: PL flows as independent batch slots with no fusion. Sydorskyi explicitly calls this "the cleverest piece" of the pipeline.)
     - Trim probability missing (Sydorskyi: per-class probs < 0.1 zeroed before training. Us: temperature + power sharpening but no zero-out.)
     - Chunk discard missing (Sydorskyi: drop chunks where max(prob) < 0.5; ~83% discard rate. Us: kept all 127K chunks.)
     - 40% primary-class-matched swap sampling missing.
     - Focal loss missing (Sydorskyi: BCE + sigmoid_focal(α=0.25, γ=2.0); us: dual-sup BCE only).

4. **Phase 5 Test 1 launched**: re-issued PL parquet with chunk-discard + trim filters (v3 = 85K chunks, 33% discard rate, 98% per-cell trim). Training a2_kd_v3 right now. My honest prior on outcome: 50% PARTIAL recovery to LB 0.910-0.913 / 20% true PL benefit > 0.915 / 30% genuine REJECT (LB < 0.906).
   - **Important note**: even my best-case v3 outcome (CONFIRM) lands at ~0.92, not 0.94.

## What top teams disclosed (BirdCLEF 2026 discussion thread, 2026-05-27)

- **Sydorskyi 2025 2nd place (Public 0.925, Private 0.928)**: in-domain XC pretrain on 819K recordings + iterative semi-supervised distillation + TopN N=1 postprocessing. Ablations show pretrain = +0.018 Private, pseudo-labels = +0.020 Private, PP = +0.015 Private.
- **Ali Ozan Memetoglu (7th, LB 0.941)**: yoinked Sydorskyi's published 2025 backbone (`vladimirsydor/bird-clef-2025-all-pretrained-models`), attached own head (attention dim 1028, melspec match required to within ±0.012 LB), fine-tuned on 2026 data. No XC retraining.
- **Antoine Masq (33rd, LB 0.943 single, 0.948 ensemble)**: own XC pretrain + own-teacher PL + custom ASL + sliding-window TTA.
- **Boredom (2nd, LB 0.950+ single)**: timm + SED, non-Perch, 1-iter PL on unlabeled. Without PL: 0.938-0.939.
- **Salman Ahmed (43rd, LB 0.939 solo / 0.944 w/ PL / 0.952 ensemble)**: attention investigation breakthrough. EffNetV2-B0. PL adds +0.005 solo; primary value is enabling diversified ensembling.
- **Multiple confirmations on PL**: "doesn't work for more than one pass" (Anil Ozturk, Boredom, Zejun_, Salman). "Trim probability" (Ali Ozan, Jiacheng Ma) — confirmed our exp 033 was missing this.

## My implementation track record (the honest part)

Our codebase has had **three significant silent bugs** found post-hoc, all of which passed offline validation:
1. Exp 032 Phase 1: training/inference mel pipeline mismatch caused LB 0.522 collapse (offline showed nothing).
2. Exp 033: num_workers=0 inherited from Tucker's Jupyter notebook → 4× slower training (offline showed nothing).
3. D.5 validation script: fold_k hardcoded to 0; folds 1-4 evaluated against their own training data (gave false-PASS at 0.995).

**Pattern**: 24+ data points of offline-LB anti-correlation. Most recent: exp 032 fold 4 best offline (auroc 0.931, e2e 0.937) → LB 0.903 (near-worst of the folds submitted).

## The candidate pivot I'm leaning toward

**Sydorskyi-yoink (Path A)**: download `vladimirsydor/bird-clef-2025-all-pretrained-models`, attach my own head matching attention dim 1028 + melspec exactly (n_mels=128, n_fft=2048, hop=512, fmin=20, top_db=80, AmplitudeToDB → standardize → min-max), fine-tune on 2026 data. Validated to LB 0.941 by Ali Ozan and Jack. Estimated 3-4 days implementation.

## Questions I need answered

Be specific and provide reasoning, not just verdicts.

1. **Is the Sydorskyi-yoink pivot the right move, or am I confirmation-biased?** Consider: my audit was self-conducted, my probability estimates are self-estimated. What would a different framing recommend?

2. **What's the realistic LB landing band for Sydorskyi-yoink** on a competitor codebase like mine (3 prior pipeline bugs found post-hoc, anti-correlation pattern)? Should I expect 0.91-0.92 (Murphy's law on implementation), 0.93-0.94 (clean execution matches Ali Ozan), or 0.94-0.95 (additional ensemble lift)?

3. **What's the realistic LB landing for the "fix all 5 PL divergences" alternative?** If I instead spent the same 3-4 days correctly implementing Sydorskyi's PL protocol (focal × PL fusion, chunk discard, trim, 40% swap, focal loss) on top of our existing 0.906 distilled-sed-kd base, where does it land? Is the base too weak to extract PL signal regardless?

4. **Implementation risk enumeration for Sydorskyi-yoink.** Where are bugs most likely to land in this pivot? Mel mismatch, head dim mismatch, ONNX export, checkpoint key remapping, training-time class balancing? Rank by likelihood × LB impact.

5. **Pre-flight verification steps** for each path. How would I know I'd implemented Sydorskyi-yoink correctly before burning a LB submission?

6. **What unknown levers might I be missing?** Specifically:
   - Anything from BirdCLEF 2024 winners I haven't considered (Kefir 1st place)?
   - Any 2025/2026 papers on bird audio classification that propose orthogonal levers?
   - Any disclosed but underweighted lever in the BirdCLEF 2026 discussion threads?
   - Site/hour priors + Gaussian smoothing (Tucker's +0.005-0.010 lever) — is this big enough to matter as a closing move on whatever I land on?

7. **Concrete day-by-day plan** for the remaining 7 days (starting tomorrow, 2026-05-28). Should include:
   - Which path (Sydorskyi-yoink / PL fix / ensemble / other) to commit to.
   - Day-by-day milestones with realistic compute + LB-slot budgets.
   - Pre-flight gates between days (what triggers continuation vs pivot).
   - A specific "if X happens by Day N, pivot to Y" decision matrix.

## Output format

A markdown document I can save to my repo at `docs/references/deep-research/2026-05-27-pivot-assessment.md`. Sections:

1. Executive summary (3-5 sentences with the verdict).
2. Question-by-question answers (1-7 above) with sources cited.
3. Concrete day-by-day plan with milestones + decision gates.
4. Risk enumeration table (axis, what could go wrong, mitigation).
5. References (URLs, papers, Kaggle threads, code repos).

Be willing to tell me I'm wrong. Don't validate; assess.

---

## Companion documents to attach (paste content if Claude.ai can't read files directly)

| File | What it provides |
|---|---|
| `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` | Full 20-axis divergence table + audit (the document Question 1 should challenge) |
| `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` | Sydorskyi 2025 writeup (479 lines, line-by-line protocol detail) |
| `user/murilo/experiments/032-tucker-distilled-sed/results.md` | Tucker recipe results — confirms our pipeline can match 0.906 / 0.915 |
| `user/murilo/experiments/033-own-teacher-pl/proposal.md` + `results.md` | Exp 033 design + REJECT verdict |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | All competitor disclosures captured to date |
| `user/murilo/notes/kaggle-discussion-non-perch-limit.md` | Companion thread (non-Perch ceiling discussions) |
| `user/murilo/MODEL-RUNS.md` | Full experiment ledger (28+ runs) |

If Claude.ai's interface doesn't accept attachments, paste the content of each into the chat or summarize what's needed.

## Expected output

A markdown file at `docs/references/deep-research/2026-05-27-pivot-assessment.md`. Read it tomorrow morning alongside the a2_kd_v3 LB result. Together they drive the exp 034 commit/pivot decision.
