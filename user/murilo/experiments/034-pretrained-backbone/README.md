# Exp 034 — Pretrained-backbone xcpretrain

**Status**: draft (2026-05-28) · scaffold only · no code yet · branch `mg/exp034-pretrained-backbone` (to be created)

Reuse the publicly-released BirdCLEF+ 2025 2nd-place pretrained backbone weights (Sydorskyi & Gonçalves; EffNetV2-S + NFNet-L0, pretrained on 819K Xeno-Canto recordings / 7,489 species) as a starting point for fine-tuning on BirdCLEF 2026. Target: lift single-fold LB from 0.906 (current best, exp 032 Phase D `distilled-sed-kd`) to 0.91-0.93. Ensemble + PP target 0.94+.

## Why now

- Exp 033 REJECT verdict (3/3 Day-1 cells regressed; Phase 5 v3 filter diagnostic LB 0.879 — filters HURT) closes the PL-recipe path on our current base.
- Deep research re-read of Sydorskyi's CEUR Table 5: PL is a multiplier on a strong base, not a substitute. Our 0.906 ImageNet-only base is structurally below the PL-payoff band.
- Top-team disclosures: Ali Ozan (7th) 0.941, Jack (18th) ~0.92x, Antoine (33rd) 0.943, Nikita (1st) 0.935 no-PL — all rely on in-domain pretraining as the base-quality lever.

## Quickstart (phases, not days)

| Phase | Goal | LB? |
|---|---|---|
| **A** — Source reading + verified config | Read `code_base/{models,dataset,losses,samplers}.py` line by line; write `docs/sydorskyi-config-verified.md` | No |
| **B** — Implementation (nnAudio mel + SED head + ckpt loader + loss + sampler) | Standalone module + unit tests | No |
| **C** — Gate 1: feature parity | Cosine ≥0.999 vs Sydorskyi's published inference notebook on 5 fixed audios | No |
| **D** — Gate 2: frozen-backbone head sanity | Fold-0 OOF AUC ≥ 0.95 (Sydorskyi: 0.984) | No |
| **E** — Frozen-backbone fine-tune + **first LB** | Fold-0 frozen-backbone fine-tune → ONNX → Kaggle | 1 slot |
| **F** — 5-fold frozen ensemble | Train remaining 4 folds; logit-space average; LB | 1 slot |
| **G** — Full fine-tune (unfreeze) | Paired test vs Phase E frozen-backbone | 1 slot |
| **H** — PP + TTA layers | TopN N=1, site/hour priors, Gaussian smooth, sliding-window TTA | 1-2 slots |
| **I** — Ensemble | Rank-ensemble xcpretrain ⊕ exp 032 5-fold (0.915) ⊕ exp 023 Perch-MLP (0.929) ⊕ optional NFNet-L0 | 1-2 slots |
| **J** — Final submission selection | 2 picks; <90 min CPU inference; submit by 2026-06-03 18:00 UTC | Final |

**Phase progression is sequential by dependency, NOT by calendar.** Parallelizable work (e.g., reading source while downloading pretrained dataset) overlaps freely. LB submissions only fire after the in-phase gate passes.

## File map

| File | Purpose |
|---|---|
| `proposal.md` | Hypothesis, falsifiable matrix, full setup + phase plan, pivot triggers, limitations |
| `tasks.md` | Itemized checklist for every phase + closeout |
| `README.md` | This file — navigation |
| (planned) `configs/` | YAML configs (deferred to Phase B) |
| (planned) `scripts/` | Parity test, checkpoint inspector, etc. (deferred to Phase A-B) |
| (planned) `pyproject.toml` | Pinned deps incl. `nnAudio` (deferred to Phase B) |
| (planned) `sbatch/` | SLURM wrappers (deferred to Phase E) |
| `results.md` (planned) | LB results per phase (deferred) |

## Source materials

- **Repo**: `github.com/VSydorskyy/BirdCLEF_2025_2nd_place`
- **Pretrained weights**: Kaggle dataset `vladimirsydor/bird-clef-2025-all-pretrained-models`
- **Parity reference**: Kaggle notebook `vladimirsydor/bird-clef-2025-minimul-inference` (scriptVersionId 245080754)
- **Paper**: Sydorskyi & Gonçalves 2025, CEUR Vol-4038 paper_256 (configs in Table 11, ablation in Table 5)

## Reference docs

- `user/murilo/notes/2026-05-28-research-report.md` — independent deep research review (realistic LB bands + 10 implementation risks + pre-flight gates)
- `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` — top-team disclosures incl. Ali Ozan / Jack / Nikita / Boredom
- `user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md` — divergence audit
- `experiments/033-own-teacher-pl/results.md` — exp 033 REJECT + v3 0.879 LB
- `experiments/032-tucker-distilled-sed/results.md` — base recipe (LB 0.906 single, 0.915 5-fold)

## Key risk callouts (from research)

- **nnAudio (NOT torchaudio)** for mel — 60% silent-bug probability if mistranslated.
- **Head hidden dim is 512** (NOT 1028 — that was a typo in earlier planning; verify in Phase A).
- **Backbone num_features**: EffNetV2-S = 1280, NFNet-L0 = **2304** (NOT 1024).
- **Standardize → min-max[0,1]** post-mel, axes unspecified in paper — read source.
- **strict=True** ckpt load on first attempt; assert non-zero conv weights.
- Aggregate "≥ 1 silent bug" probability on first attempt: **80-85%**. Phase A-D gates exist to catch these before LB.
