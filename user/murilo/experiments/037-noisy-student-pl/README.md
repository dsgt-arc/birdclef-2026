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

## Round-1 ablation factorial (15 single-fold cells)

`r1_{A,B,C}_{raw,t2}_{augF,augT}[_softauc].yaml` — KD routing × PL transform × PL aug (+ SoftAUC).
A = KD-on/detach (PL→head), B = KD-off/e2e (PL→whole student), C = KD-on/no-detach (PL→both).
Configs are emitted by `scripts/gen_round1_configs.py` (re-run to regenerate). PL parquets (raw, t2)
are already generated.

## Run (Murilo, on PACE)
```bash
cd user/murilo/experiments/037-noisy-student-pl
# Phase 1 (already done; ~15s each on interactive venv if regenerating):
#   python scripts/generate_ensemble_pl.py --variant raw   (and --variant t2)
# 0) gradient-flow GATE first:
bash sbatch/run.sh configs/r1_A_raw_augT.yaml --fast-dev-run
# 1) launch the 15-cell single-fold grid (parallel):
for c in configs/r1_*.yaml; do bash sbatch/run.sh "$c" --fold 0; done
# 2) offline screen (rough — eliminate/direction only; LB is the arbiter):
python scripts/eval_labeled_ss.py --pred <student_oof.parquet>   # Gate-A bar = 0.7632
# 3) LB ~5 mechanism-diverse cells; 5-fold the winner:
bash sbatch/run.sh configs/<winner>.yaml --all-folds
```
