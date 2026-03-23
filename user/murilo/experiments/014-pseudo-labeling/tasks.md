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

## 3. Pseudo-Labeling Script

- [ ] 3.1 Create `pseudo_label.py` with Mode A (direct ensemble)
  - Load 3 ONNX teacher models (ProtoCLR, ConvNeXt, EAT)
  - Per-segment: compute mel → ONNX inference → store raw logits
  - Save per-model logits as npz for offline re-aggregation
- [ ] 3.2 Add ensemble aggregation: temperature scaling + per-taxon weighted average
- [ ] 3.3 Add Mode B support (load MixIT NPZ channels, max-pool per model)
- [ ] 3.4 Save final pseudo-labels as parquet

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

## 7. Future Work (deferred)

- [ ] 7.1 Re-export ONNX models with dual outputs (logits + embeddings)
- [ ] 7.2 Save 2026 embeddings alongside predictions (384/768/1024-dim per model)
- [ ] 7.3 Regenerate pseudo-labels with stronger teachers (e.g., SED models from future exps)
