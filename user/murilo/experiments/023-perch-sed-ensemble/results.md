# Experiment 023: Perch + SED Ensemble — Results

## All Submissions

| Variant | W_PERCH | LB | Runtime (dry) | Notes |
|---------|---------|-----|---------------|-------|
| Phase A (w=0.5) | 0.50 | 0.927 | 145s (20 files) | +0.002 vs Perch TF (0.925) |
| Phase B (w=0.3) | 0.30 | 0.927 | 181s (20 files) | Tied with w=0.5 |
| Phase B (w=0.7) | 0.70 | 0.926 | 189s (20 files) | −0.001 vs w=0.5 |
| **Phase C MLP-only** | **0.50** | **0.929** | — | **New best! proto_weight=0.0 (skip ProtoSSM)** |
| Phase C No PP | 0.50 | 0.922 | — | Both PPs disabled, −0.005 vs w=0.5 |
| Phase C TF+single+MLP | 0.50 | TIMEOUT | 271s (20 files) | TF Perch ~10s/file, exceeded 90 min limit |

## Phase A Analysis (w=0.5, 0.927 LB)

### Score Breakdown

| Component | LB (standalone) | Notes |
|-----------|----------------|-------|
| Perch ProtoSSM v18 (TF) | 0.925 | Best individual model |
| Perch ProtoSSM v18 (ONNX) | 0.920 | −0.005 ONNX penalty |
| SED v5j-4fold-pp2 | 0.917 | Best SED |
| **Ensemble (w=0.5)** | **0.927** | **+0.002 vs best individual, +0.007 vs ONNX Perch** |

### Interpretation

The ensemble at 0.927 beats the best individual model (Perch TF 0.925) despite
using the weaker ONNX Perch backend (0.920). This confirms the models are
complementary — the +0.007 gain over ONNX Perch more than compensates for the
−0.005 ONNX penalty.

The gain is modest (+0.002 over Perch TF), suggesting significant overlap in
what both models get right. The weight grid (Phase B) will reveal whether
the optimal blend favors Perch or is truly balanced.

### Timing (Dry-Run, 20 Train Files)

| Component | Time | Est. 600 files |
|-----------|------|----------------|
| Perch ONNX load | 5.2s | 5s |
| Perch ONNX inference | 33s | ~10 min |
| ProtoSSM + MLP + PP | 9s | ~8 min |
| SED 4-fold + TTA | 103s | ~46 min |
| **Total** | **145s** | **~64 min** |

Well within 90-min Kaggle CPU limit.

## Phase B Analysis (Weight Grid)

| W_PERCH | LB | Delta vs w=0.5 |
|---------|-----|----------------|
| 0.30 | **0.927** | 0.000 |
| **0.50** | **0.927** | baseline |
| 0.70 | 0.926 | −0.001 |

The weight curve is remarkably flat — all three weights land at 0.926-0.927.
This suggests:

1. **Both models contribute roughly equally.** There's no clear winner; the
   ensemble benefits from both at any reasonable blend.
2. **The slight edge to SED-heavy (w=0.3) or balanced (w=0.5)** suggests SED
   adds marginally more unique signal than ONNX Perch in this ensemble.
3. **The ONNX penalty is the binding constraint.** If Perch were running at its
   TF quality (0.925 vs 0.920), w=0.7 would likely match or beat w=0.5.
4. **Diminishing returns from weight tuning.** The +0.001 difference between
   best (0.927) and worst (0.926) is within noise. Further weight fine-tuning
   (e.g., w=0.4, 0.6) is unlikely to yield meaningful gains.

## Phase C Analysis (Ablations)

### MLP-only (0.929 LB — new best!)

| Variant | Proto Weight | LB | Delta vs baseline |
|---------|-------------|-----|-------------------|
| ProtoSSM + MLP (default) | 0.50 | 0.927 | baseline |
| **MLP-only** | **0.00** | **0.929** | **+0.002** |

Dropping ProtoSSM and using MLP probes only **improves LB by +0.002**. This
matches the pantanal-distill reference (0.930 LB) which also used MLP-only.

Why ProtoSSM hurts in the ensemble context: the ProtoSSM was trained on TF
embeddings but runs on ONNX embeddings (−0.005 penalty). The MLP probes are
simpler and more robust to this distribution shift. Additionally, ProtoSSM's
gated fusion with Perch logits may introduce noise when those logits come from
ONNX rather than TF.

**This is the single biggest finding of exp 023**: the internal Perch pipeline
blend matters more than the Perch-vs-SED ensemble weight.

### No PP (0.922 LB)

| Variant | PP | LB | Delta vs baseline |
|---------|-----|-----|-------------------|
| Full PP (default) | Both enabled | 0.927 | baseline |
| **No PP** | **Both disabled** | **0.922** | **−0.005** |

Disabling post-processing from both pipelines costs −0.005 LB. This confirms
that the individual PPs are net positive even when their outputs are averaged.
The PP gain (+0.005) is comparable to the ONNX penalty (−0.005), reinforcing
that both calibration and backend quality matter.

## Updated Scoreboard

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **New best overall** |
| Perch+SED w=0.5 (exp 023) | 0.927 | |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual |
| Perch+SED no PP (exp 023) | 0.922 | PP disabled |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED |
| Competition #1 | 0.939 | Target |
