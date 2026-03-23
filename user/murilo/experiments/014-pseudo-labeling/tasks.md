# Experiment 014: Teacher Pseudo-Labeling — Tasks

## 1. Goal

Pseudo-label 10,592 unlabeled soundscapes using exp 012 teacher ensemble, following
the Bird MixIT paper's max-pool-across-channels approach. Produce soft labels for
downstream student training (exp 015).

## 2. Setup

- [ ] 2.1 Create experiment directory structure
- [ ] 2.2 Create `pyproject.toml` with GPU dependencies
- [ ] 2.3 Create YAML configs for Mode A (direct) and Mode B (MixIT-enhanced)
- [ ] 2.4 Create SLURM scripts (`sbatch/run.sh`, `sbatch/_job.sh`)
- [ ] 2.5 Write README.md and tasks.md

## 3. Pseudo-Labeling Script (v1: direct ensemble)

- [x] 3.1 Create `pseudo_label.py` — ProtoCLR + ConvNeXt only (no EAT)
  - Per-segment: compute mel → ONNX GPU inference → raw logits
  - Average sigmoid predictions across 2 teachers → soft pseudo-labels
  - Save per-model logits as npz (for offline re-aggregation)
  - Save ensemble pseudo-labels as parquet
- [x] 3.2 Create `config/direct.yaml` with teacher model slugs

## 4. Validation

- [ ] 4.1 Create `validate.py` — compare pseudo-labels vs GT on 66 labeled files
- [ ] 4.2 Run Mode A validation on labeled soundscapes
- [ ] 4.3 Run Mode B validation on labeled soundscapes
- [ ] 4.4 Compare: Mode A vs Mode B, ensemble vs single model
- [ ] 4.5 Sweep temperature T (0.5, 0.6, 0.7, 1.0) and model weights

## 5. Full Unlabeled Run

- [ ] 5.1 Submit Mode A on all unlabeled soundscapes
- [ ] 5.2 Submit Mode B on all unlabeled soundscapes (if Mode B wins on validation)
- [ ] 5.3 Verify output: ~127K segments in pseudo_labels.parquet

## 6. Results

- [ ] 6.1 Write results.md with pseudo-label quality metrics
- [ ] 6.2 Compare Mode A vs Mode B

## 7. Future Extensions (deferred)

- [ ] 7.1 Temperature scaling: sweep T on labeled validation (2025 1st used T=0.55-0.6)
- [ ] 7.2 MixIT-enhanced: max-pool across 4 separated channels + original per model (Bird MixIT paper Section 7.1)
- [ ] 7.3 Per-taxon model weighting (e.g., higher ProtoCLR weight for insects)
- [ ] 7.4 Include EAT for bird-only species with per-taxon weights
- [ ] 7.5 Re-export ONNX models with dual outputs (logits + embeddings)
- [ ] 7.6 Regenerate pseudo-labels with stronger teachers (e.g., SED models)
