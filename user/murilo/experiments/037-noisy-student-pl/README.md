# Exp 037 — Multi-Iterative Noisy-Student Soft Pseudo-Labeling

**Goal:** break the 0.929 LB plateau before 2026-06-03 by distilling the 0.929 ensemble's
soundscape-domain knowledge into a `distilled-sed-kd` student (0.906/0.915) via **soft, un-sharpened,
multi-round** pseudo-labeling — the one lever that moved every 2025 top solution (0.872 → 0.930).

Inverts every knob our 4 prior PL failures got wrong: **strong ensemble teacher** (not own-SED),
**soft un-sharpened** labels (not T=0.6/P^1.5), **no aggressive filtering**, **~50/50 GT:PL**,
**loss_weight=1.0**, **multiple rounds**.

## Layout
- `proposal.md` — hypothesis, falsifiable thresholds, method, provenance.
- `tasks.md` — phased checklist + gates.
- `scripts/generate_ensemble_pl.py` — blend cached teacher parquets → softened PL parquet.
- `scripts/eval_labeled_ss.py` — leakage-free labeled-SS macro-AUC (reuses exp 035 cv_trust scaffold).
- `train.py` + `configs/` — clone of exp 032/033 wiring; round-1 grid (parquet × sampler).
- `sbatch/` — `$TMPDIR`-venv SLURM jobs (gen_pl, train).

## Teacher is cached (no new inference)
- Perch (TF, 0.925-quality): `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet`
- SED v5j-4fold logits: `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/fold{0..3}/logits.parquet`

## Run (Murilo, on PACE)
```bash
cd user/murilo/experiments/037-noisy-student-pl
# Phase 1: generate PL parquets (CPU, minutes)
python scripts/generate_ensemble_pl.py --variant raw
python scripts/generate_ensemble_pl.py --variant t2
python scripts/generate_ensemble_pl.py --variant g05
# Phase 2: round-1 grid (single-fold), then 5-fold the winner
CONFIG=configs/round1_raw_5050.yaml bash sbatch/train.sh   # + the other 3 cells
```
