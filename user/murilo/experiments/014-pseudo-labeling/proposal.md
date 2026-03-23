---
experiment: "014-pseudo-labeling"
status: proposed
date: 2026-03-22
author: Murilo Gustineli
---

# Experiment 014: Teacher Pseudo-Labeling Pipeline

## Hypothesis

Running exp 012 teacher models (ProtoCLR + ConvNeXt) on 10,592 unlabeled soundscapes
and averaging their predictions produces soft pseudo-labels that improve downstream
student training (exp 015) beyond the exp 012 baseline (ProtoCLR 0.767 LB).

## Background: Two Different Approaches

### The MixIT paper's approach (inference-time enhancement)

The Bird MixIT paper (Section 7.1) improves a **single classifier's predictions** by
feeding it separated audio channels at inference time:

```
Soundscape → MixIT → 4 channels + original → ONE classifier → max-pool per species
```

This produces better predictions from one model. It's an inference-time trick, not a
training data generation method. Our exp 013 MixIT separation enables this approach.

### The winning solutions' approach (pseudo-labeling for training)

BirdCLEF 2024/2025 winners use **iterative self-training** — a fundamentally different
technique:

```
Round 0: Train model on labeled data
Round 1: Model predicts on unlabeled data → pseudo-labels → retrain on labeled + pseudo-labeled
Round 2: Improved model → better pseudo-labels → retrain
...repeat 3-4 times
```

No source separation involved. Pseudo-labels come from the model's own predictions
(or an ensemble), optionally sharpened with temperature scaling.

### Our approach: start simple, extend later

We start with the proven winning approach — direct ensemble pseudo-labeling — and
can later extend with MixIT enhancement if validated on the 66 labeled files.

## Approach

### v1: Direct Ensemble (this experiment)

For each unlabeled soundscape, run ProtoCLR and ConvNeXt on each 5s segment, average
their sigmoid predictions → soft pseudo-label.

```
For each 5s segment:
  Original audio → ProtoCLR (exp 012, LB 0.767) → sigmoid(logits) → probs_pcr (234,)
  Original audio → ConvNeXt (exp 012, LB 0.755) → sigmoid(logits) → probs_cnx (234,)
  Pseudo-label = mean(probs_pcr, probs_cnx)  → (234,)
```

**Why only ProtoCLR + ConvNeXt, no EAT**: The error analysis
(`013-mixit-separation/error-analysis.md`) showed:
- EAT AUROC on insects: **0.673** (vs 0.856 ProtoCLR, 0.848 ConvNeXt)
- Simple 3-model mean ensemble **hurts** (−0.039 vs best single model)
- Root cause: EAT's kaldi fbank caps at 8kHz, missing insect stridulation

Including EAT would inject noise into pseudo-labels for 25/234 insect species.
ProtoCLR + ConvNeXt are close in quality and architecturally diverse.

### Future extensions (deferred)

These can be added in later iterations once v1 is validated:

1. **Temperature scaling**: Apply `sigmoid(logits / T)` with T < 1.0 to sharpen
   predictions. The 2025 1st place used T=0.55-0.6. Requires saving raw logits
   (which we do) and sweeping T on the 66 labeled files.

2. **MixIT-enhanced predictions**: Run each teacher on original + 4 MixIT channels,
   max-pool per model (Bird MixIT paper Section 7.1), then average across models.
   Requires exp 013 separation to be complete.

3. **Per-taxon weighting**: Assign different model weights per taxonomic class
   based on error analysis (e.g., higher ProtoCLR weight for insects).

4. **Iterative self-training**: Student from exp 015 becomes a new teacher, re-generate
   pseudo-labels, retrain. Repeat 2-3 times.

5. **EAT inclusion for bird-only species**: If per-taxon weighting is implemented,
   EAT could be included with weight=0 for insects and positive weight for birds.

### Key Design Decisions

1. **Hard labels for labeled data**: Per Anthony's exp A-014 finding, never replace
   ground truth with pseudo-labels. Pseudo-labels are only for unlabeled soundscapes.

2. **Save raw logits alongside final probs**: Enables offline re-aggregation (temperature
   sweep, weight tuning) without re-running inference.

3. **GPU inference**: Use `onnxruntime-gpu` on `gpu-rtx6000` for speed. Two models ×
   10,592 files × 12 segments = ~254K forward passes.

4. **Validate on labeled soundscapes first**: Run pseudo-labeling on 66 labeled files,
   compare against ground truth to measure quality before the full unlabeled run.

### Output

```
shared/pseudo-labels/birdclef-2026/direct/
├── logits/
│   ├── protoclr-2026-soundscape.npz   # (n_segments, 234) raw logits
│   └── convnext-2026-soundscape.npz   # (n_segments, 234) raw logits
├── pseudo_labels.parquet               # (n_segments, 236 cols: filename, end_sec, 234 avg probs)
└── config.json                         # models used, split, n_segments
```

## Evaluation

### Metrics
- **Pseudo-label AUROC on labeled soundscapes**: Per-species AUROC vs ground truth
- **Per-taxon quality**: Birds vs amphibians vs insects breakdown
- **Coverage**: % of unlabeled segments with ≥1 species above 0.5
- **Ensemble vs single model**: Does averaging ProtoCLR + ConvNeXt beat either alone?

### Baseline
- Individual model predictions (ProtoCLR alone, ConvNeXt alone)

### Success Criteria
- **Confirm if**: 2-model ensemble AUROC > best single model AUROC on labeled soundscapes
- **Reject if**: Ensemble AUROC < best single model (averaging hurts)

## Dependencies

- Exp 012 ONNX models: ProtoCLR + ConvNeXt (complete)
- For future Mode B: Exp 013 MixIT separated audio (complete)

## Leads to

Exp 015 (student training) consumes the pseudo-labels produced here.
