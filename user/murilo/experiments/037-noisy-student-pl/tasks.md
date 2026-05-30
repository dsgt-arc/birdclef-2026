# Tasks — Exp 037 Multi-Iterative Noisy-Student Soft PL

Tests live in `user/murilo/tests/037-noisy-student-pl/`; run with the interactive venv
(`~/clef/birdclef-2026/.venv`). Murilo runs SLURM/Kaggle; Claude provides commands.

## Phase 0 — validation harness (Day 1)
- [x] `scripts/eval_labeled_ss.py`: leakage-free labeled-SS macro-AUC reusing `analysis/data/cv_trust/`
      scaffold (gt_windows, gt_Y, species_cols, oof_fold_maps) + `macro_auc_skip_empty`. **Built + verified.**
- [x] Reference bars locked (reproduce cv_scores): **distilled (baseline student) OOF = 0.7632 = Gate-A bar**;
      v5j 0.7909; teacher blend on labeled = 0.9981 (LEAKED via Perch probe → NOT a fair bar).
      ⚠ labeled-SS OOF anti-correlates with LB (distilled 0.763 OOF → 0.915 LB) → rough screen only.

## Phase 1 — generate soft PLs (Day 1–2)
- [x] `scripts/generate_ensemble_pl.py`: blend `0.5·perch_234_probs + 0.5·sigmoid(mean_folds(sed))`,
      transform (T, γ), write `(filename, chunk_idx, +234)` parquet. No filtering. **Built + verified.**
- [x] Generated `raw` (T=1,γ=1) + `t2` (T=2) parquets under `shared/pseudo-labels/birdclef-2026/v037-ens-{raw,t2}/`.
      **g05 (γ=0.5) DROPPED**: it over-inflates the already-sparse blend (frac>0.5: 0.95% → 56%) — power-soften
      assumes a peaked teacher; ours isn't. Temperature (t2) is the principled soften.
- [x] Sanity: 127,896 rows, 234 cols ∈ [0,1]; raw mean=0.239, frac>0.5=0.95% (~2 species/window — sparse, realistic).

## Phase 2 — round-1 student (Day 2–3)
- [x] `train.py` (clone exp 033, EXPERIMENT_NAME=037) + `configs/_common.yaml` + 4 round-1 grid configs. **Built.**
- [ ] **(Murilo, GPU)** fast-dev-run gate, then single-fold grid: parquet {raw, t2} × sampler {mixsamp 50/50, uniform natural}.
- [ ] Gate A (offline, rough): best cell's labeled-SS OOF vs baseline 0.7632 → 5-fold the winner.
- [ ] Export ONNX (mirror exp 032) → LB: single-fold vs 0.906, 5-fold vs 0.915, ensemble-blend vs 0.929.

## Phase 3 — round 2 (Day 3–4, only if round 1 LB-green)
- [ ] Promote round-1 student into teacher → regenerate softened PL → retrain → LB.
- [ ] Stop if ensemble gain < +0.005.

## Phase 4 — final picks (Day 4)
- [ ] #1 safe = reproduced 0.929; #2 upside = best PL-student blend.

## Verification gates
- [ ] Full `user/murilo/tests/` suite green (032/033/034/036 regression + new 037 tests).
- [ ] fast-dev-run of a round-1 config (gradient flow + 6-tuple PL collate).
- [ ] PL parquet schema + softening monotonicity tests.
- [ ] val fold shares no soundscape file with train (leakage check).
