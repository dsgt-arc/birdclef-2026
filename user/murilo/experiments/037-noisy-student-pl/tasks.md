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

## Phase 2 — round-1 ablation factorial (Day 2–3)
- [x] `train.py` (clone exp 033, EXPERIMENT_NAME=037) + `_common.yaml` + **15 factorial configs** (gen_round1_configs.py). **Built.**
- [x] New flags wired + tested: `kd_detach_head` (cell C, gradient-routing test), `loss_type`/`SoftAUCLoss`
      (+ component log), `pseudo_label_aug`, `train/logit_std`, GT-only `steps_per_epoch=500`, 25 ep. 43 unit + 290 regression PASS.
- [x] fast-dev-run gates (A/B/SoftAUC) all green; **all 15 single-fold cells trained** (jobs 9346722–37, fold 0, 25 ep).
- [x] Offline pulled + screened (see `results.md`). **Direction:** B/C (PL→backbone) > A (head-only) on both
      metrics; raw > t2; aug/SoftAUC inconclusive. Best val/auroc = `r1-B-raw-augF` 0.9555. **Caveat:** all beat
      no-PL baseline offline but exp 033 did too & REJECTED → offline = direction only, LB decides.
- [x] **Metric fix:** reverted `_common.yaml` monitor `val/cmap` → **`val/auroc`** (competition metric, matches
      exp 021). Round 1 lost nothing (save_top_k=-1 → all 25 epochs saved; export best-by-val/auroc epoch).
- [x] **Export WIRED** for the 5 picks (**last.ckpt = final epoch**, fold 0) — exp 010 pipeline:
      `export_exp037_pl` in `sbatch/export.sbatch` + 5 MODELS entries + 5 kernel-metadata JSONs +
      5 MODEL_SLUGS. Uses `find_last_ckpt` (lineage convention: 0.906/0.915 + exp 033 all submitted
      last.ckpt). **Do NOT cherry-pick best-val/auroc epoch** — the 168-window val is noisy (stable
      cells plateau ±0.002, noisy cells oscillate ±0.02) & anti-correlates with LB → argmax-epoch
      = noise-steering. Model slugs `exp037-r1-{a,b,c}-raw-augf` + `*-raw-augt-softauc`.
- [x] Export → upload (fixed species_mapping 41→234 via exp037 token in upload_kaggle heuristic) →
      pushed → **A/B/C-raw-augF trio submitted to LB**.
- [x] **LB VERDICT = REJECT.** C 0.832 / B 0.790 / A 0.754 — all far below 0.906 base (best −0.074).
      KD axis transferred (C>B>A offline→LB) but PL **poisons** regardless (diffuse soft targets:
      raw blend mean 0.24, ~233 cls at 0.1–0.3/window × 50/50 + loss_weight=1.0 → flat calibration).
      **5th PL failure.** 2 SoftAUC picks NOT submitted (decisive; LB conserved).

## Phase 3 — REJECT → PIVOT (PL is 0/5; deadline 06-03)
- [ ] **Secure the 0.929 ensemble (exp 023)** as final pick #1; A/B PP/TTA + ensemble-diversity for #2.
- [ ] Post-deadline paper track (compute free, doesn't count for LB): 1 SoftAUC PL submission to close
      the ranking-loss question; Perch-only PLs; PL-target trim (<0.05 tail); broad XC-pretraining.

## Phase 4 — final picks (Day 4)
- [ ] #1 safe = reproduced 0.929; #2 upside = best PP/TTA/diversity blend on it. (exp 037 PL model NOT a pick — REJECT.)

## Verification gates
- [ ] Full `user/murilo/tests/` suite green (032/033/034/036 regression + new 037 tests).
- [ ] fast-dev-run of a round-1 config (gradient flow + 6-tuple PL collate).
- [ ] PL parquet schema + softening monotonicity tests.
- [ ] val fold shares no soundscape file with train (leakage check).
