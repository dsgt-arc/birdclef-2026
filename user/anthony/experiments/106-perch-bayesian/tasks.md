# Tasks

## 1. Setup
- [ ] 1.1 Create package skeleton (pyproject.toml, src/)
- [ ] 1.2 Copy perch_onnx.py and audio utilities from exp 105
- [ ] 1.3 Write extraction script for all soundscapes (labeled + unlabeled)
- [ ] 1.4 Write extraction script for focal recordings (sharded)
- [ ] 1.5 Write sbatch scripts for extraction jobs

## 2. Extraction
- [ ] 2.1 Run soundscape extraction (all 10,658 files)
- [ ] 2.2 Run focal extraction (35,549 files, sharded)
- [ ] 2.3 Verify output parquets (row counts, schema, no NaNs)

## 3. Analysis
- [ ] 3.1 Correlation analysis: Perch logits vs BirdCLEF ground truth on labeled windows
- [ ] 3.2 Visualize activation profiles per BirdCLEF species
- [ ] 3.3 Check non-bird taxa coverage (amphibians, mammals, insects)
- [ ] 3.4 Bayesian mapping: P(target | perch_logits) with regularization
- [ ] 3.5 Temporal aggregation sweep (per-species optimal window width)

## 4. Evaluation
- [ ] 4.1 Evaluate on held-out labeled windows (sc_cmap, AUROC)
- [ ] 4.2 Compare against exp 105 baselines
- [ ] 4.3 If promising: generate pseudo-labels for unlabeled soundscapes
- [ ] 4.4 Document findings in results.md
