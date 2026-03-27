# Experiment 018: Perch Ensemble — Tasks

## Goal

Ensemble EfficientNet-B0 SED (v4b, 0.859 LB) with Google Perch v2 foundation model
targeting 0.92+ LB. Start with chiranjithdharma's simple approach (LogReg probes on
PCA embeddings + metadata priors), then blend with SED predictions.

**SED Baseline**: v4b = 0.859 LB
**Perch Baseline**: chiranjithdharma = 0.910 LB
**Target**: 0.92+ LB (ensemble)

### Status: IN PROGRESS

- **Branch**: `mg/018-perch-ensemble`

---

## Step 1: Perch Species Mapping

- [ ] 1.1 Build species mapping script (Perch labels.csv -> BirdCLEF 2026)
- [ ] 1.2 Verify: 203 direct, 6 genus proxy, 25 unmapped insect sonotypes
- [ ] 1.3 Save `artifacts/perch_species_mapping.json`

## Step 2: Extract Perch Embeddings from Labeled Soundscapes

- [ ] 2.1 Create extraction script (TF SavedModel inference on 66 soundscapes)
- [ ] 2.2 Create SLURM job for Perch inference on PACE (CPU, TensorFlow venv)
- [ ] 2.3 Run extraction: 66 files x 12 windows = 792 samples
- [ ] 2.4 Verify outputs: embeddings (792, 1536), mapped scores (792, 234)

## Step 3: Train LogReg Probes + Metadata Priors

- [ ] 3.1 Create probe training script
- [ ] 3.2 PCA: 1536 -> 32 dimensions
- [ ] 3.3 Per-species features: PCA embed + Perch score + temporal context
- [ ] 3.4 Train LogReg probes (C=0.25, 5-fold GroupKFold by site)
- [ ] 3.5 Build metadata priors (site/hour tables, Bayesian smoothing)
- [ ] 3.6 Compute dynamic alpha per species
- [ ] 3.7 Serialize artifacts (PCA, probes, alphas, priors, mapping)

## Step 4: Kaggle Kernel

- [ ] 4.1 Create unified submission.py (SED ONNX + Perch TF + blend)
- [ ] 4.2 Create kernel metadata JSON (4 sources: deps, SED weights, Perch artifacts, Perch model)
- [ ] 4.3 Local validation on labeled soundscapes
- [ ] 4.4 Sweep blend weights (w_perch in {0.5, 0.6, 0.7, 0.8, 0.9})

## Step 5: Upload + Submit

- [ ] 5.1 Upload probe artifacts as Kaggle dataset
- [ ] 5.2 Push kernel to Kaggle
- [ ] 5.3 Record LB result vs baselines (SED=0.859, Perch standalone~0.910)

---

## Key Config: v1

| Parameter | Value |
|-----------|-------|
| Perch model | `google/bird-vocalization-classifier/TensorFlow2/perch_v2/1` |
| PCA components | 32 |
| LogReg C | 0.25 |
| Min positive samples | 8 |
| Probe alpha base | 0.25 |
| Lambda event (Aves) | 0.4 |
| Lambda texture (Amphibia/Insecta) | 1.0 |
| Smoothing alpha event | 0.15 |
| Smoothing alpha texture | 0.35 |
| SED weight | 0.3 |
| Perch weight | 0.7 |
| Temperature | 1.10 |
