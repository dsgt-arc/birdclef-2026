# Experiment 014: Teacher Pseudo-Labeling — Tasks

## 1. Goal

Pseudo-label 10,592 unlabeled soundscapes using exp 012 teacher ensemble (ProtoCLR +
ConvNeXt, no EAT). Average sigmoid predictions → soft pseudo-labels for downstream
student training (exp 015).

## 2. Setup

- [x] 2.1 Create experiment directory structure
- [x] 2.2 Create `pyproject.toml` with GPU dependencies
- [x] 2.3 Create `config/direct.yaml` (ProtoCLR + ConvNeXt, no EAT)
- [x] 2.4 Create SLURM scripts (`sbatch/run.sh`, `sbatch/_job.sh`) with GPU monitoring
- [x] 2.5 Write README.md, proposal.md, tasks.md

## 3. Pseudo-Labeling Script (v1: direct ensemble)

- [x] 3.1 Create `pseudo_label.py` — ProtoCLR + ConvNeXt only (no EAT)
  - Per-segment: compute mel → ONNX GPU inference → raw logits
  - Average sigmoid predictions across 2 teachers → soft pseudo-labels
  - Save per-model logits as npz (for offline re-aggregation)
  - Save ensemble pseudo-labels as parquet
  - Save full input config in output config.json for reproducibility
- [x] 3.2 Create `config/direct.yaml` with teacher model slugs

## 4. Validation on Labeled Soundscapes

- [x] 4.1 Run on labeled split (66 files): `bash sbatch/run.sh config/direct.yaml --split labeled`
- [x] 4.2 Output: `shared/pseudo-labels/birdclef-2026/pcr-cnx-soundscape-v1/labeled/`
  - `pseudo_labels.parquet` (792 segments × 236 cols)
  - `logits/protoclr-2026-soundscape.npz` + `logits/convnext-2026-soundscape.npz`
  - `config.json` with full provenance
- [ ] 4.3 Compare pseudo-label quality vs ground truth (per-species/taxon AUROC)
- [ ] 4.4 Compare ensemble vs single-model quality
- [ ] 4.5 Check GPU utilization from `logs/gpu_labeled_*.csv`

## 5. Full Unlabeled Run

- [ ] 5.1 Submit on unlabeled split: `bash sbatch/run.sh config/direct.yaml --split unlabeled`
- [ ] 5.2 Verify output: ~127K segments in `pcr-cnx-soundscape-v1/unlabeled/pseudo_labels.parquet`

## 6. Results

- [ ] 6.1 Write results.md with pseudo-label quality metrics
- [ ] 6.2 Update MODEL-RUNS.md with pseudo-label run details

## 7. Future Extensions (deferred)

- [ ] 7.1 Temperature scaling: sweep T on labeled validation (2025 1st used T=0.55-0.6)
- [ ] 7.2 MixIT-enhanced: max-pool across 4 separated channels + original per model (Bird MixIT paper Section 7.1)
- [ ] 7.3 Per-taxon model weighting (e.g., higher ProtoCLR weight for insects)
- [ ] 7.4 Include EAT for bird-only species with per-taxon weights
- [ ] 7.5 Re-export ONNX models with dual outputs (logits + embeddings)
- [ ] 7.6 Regenerate pseudo-labels with stronger teachers (e.g., SED models)
