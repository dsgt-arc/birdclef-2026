# Experiment 015: Student Training — Results

## Summary

**Hypothesis**: Training on focal + labeled soundscapes + pseudo-labeled unlabeled
soundscapes (soft labels from exp 014) will improve LB by ≥0.02 over exp 012 baseline.

**Result**: **Rejected.** Pseudo-labels at w=0.5, T=1.0 **hurt** LB performance.
ProtoCLR dropped −0.005 (0.762 vs 0.767) and EAT dropped −0.024 (0.696 vs 0.720)
despite flat/improved offline metrics. The experiment confirms that the clip-level
architecture (global average pooling) is the bottleneck — more soundscape data
through the same architecture cannot improve and may even degrade LB performance.

---

## Training Configuration (Round 0)

| Parameter | Value |
|-----------|-------|
| Pseudo-label source | Exp 014 `pcr-cnx-soundscape-v1` (ProtoCLR + ConvNeXt ensemble) |
| Pseudo-label weight | w=0.5 |
| Temperature | T=1.0 (no scaling) |
| Starting checkpoint | Exp 012 best (234 classes, soundscape-adapted) |
| Training data | 213K focal + 570 soundscape + 127K pseudo = 340K total |
| Loss | Per-sample weighted BCE (w=1.0 for ground truth, w=0.5 for pseudo) |
| Architecture | Same as exp 012 (LoRA rank=8, alpha=16) |

## Training Results

| Model | val/auroc | test/auroc | test/cmap | Best Epoch | Early Stop |
|-------|-----------|-----------|----------|------------|------------|
| ProtoCLR | 0.827 | 0.846 | 0.485 | 5 | 9 |
| EAT | 0.829 | 0.842 | 0.465 | 2 | 8 |
| ConvNeXt | — | — | — | — | timed out |

All models early-stopped quickly, barely improving from the exp 012 starting point.
ConvNeXt student training timed out (24hr limit) and was not completed.

## Leaderboard Results

| Model | Exp 012 LB | Exp 015 LB | Delta |
|-------|-----------|-----------|-------|
| ProtoCLR | **0.767** | 0.762 | **−0.005** |
| EAT | 0.720 | 0.696 | **−0.024** |
| ConvNeXt | 0.755 | — | — |

**Both models regressed on LB.** EAT's larger drop (−0.024) may reflect the
pseudo-label quality mismatch: EAT was excluded from the teacher ensemble
(exp 014) due to poor insect AUROC (0.673), yet the pseudo-labels encode
ProtoCLR+ConvNeXt teachers' behavior which may conflict with EAT's learned
representations, effectively teaching EAT to mimic models with different
spectral characteristics.

## Comparison vs Exp 012 Baseline (Offline Test Set)

| Model | Exp 012 AUROC | Exp 015 AUROC | Delta | Exp 012 cmAP | Exp 015 cmAP | Delta |
|-------|--------------|--------------|-------|-------------|-------------|-------|
| ProtoCLR | 0.845 | 0.846 | +0.001 | 0.481 | 0.485 | +0.004 |
| EAT | 0.838 | 0.842 | +0.004 | 0.452 | 0.465 | +0.013 |
| ConvNeXt | 0.844 | — | — | 0.470 | — | — |

**Key discrepancy**: Offline metrics improved slightly but LB regressed. The
offline test set is dominated by focal data (~26K segments vs ~86 soundscape)
which masks soundscape-specific degradation. The pseudo-labels may have caused
the model to overfit to the teachers' soundscape behavior at the expense of
generalization to unseen test soundscapes.

## GPU Utilization (ProtoCLR)

| Metric | Avg |
|--------|-----|
| GPU Utilization | 67.9% |
| Data Loading % | 13.6% (spikes to 84%) |
| Memory Used | 0.62 GB / 25.2 GB (2.4%) |
| Throughput | 577 samples/s |

Training was I/O bound due to loading 10,592 unlabeled soundscape files from shared
filesystem. Updated num_workers from 8 → 12 for future runs.

## Temperature Scaling Sweep

AUROC is rank-based and invariant to temperature. Log loss and Brier score favored
T=1.0 (teachers slightly over-confident). Used T=1.0 for round 0.

| T | Log Loss | Brier | AUROC |
|---|----------|-------|-------|
| 0.50 | 0.3163 | 0.0414 | 0.8771 |
| 0.60 | 0.2678 | 0.0407 | 0.8771 |
| 0.80 | 0.2058 | 0.0393 | 0.8771 |
| 1.00 | 0.1699 | 0.0380 | 0.8771 |

## Analysis

### Why pseudo-labels didn't help

1. **Architectural bottleneck**: Our clip-level head applies global average pooling over
   the entire 5s segment, diluting brief vocalizations (<0.5s) into noise. The error
   analysis (exp 013) showed median positive confidence of only 0.05-0.19 on true
   positives. More soundscape data trained through the same bottleneck can't fix this.

2. **Teacher-student similarity**: The student starts from the teacher's checkpoint
   (exp 012) and uses the teacher's predictions as targets. With the same architecture
   and soft labels that preserve teacher uncertainty, the student converges to
   approximately the same model — evidenced by 0.976 logit correlation between models.

3. **Gradient dilution**: 127K pseudo-labeled segments (37% of training data) at w=0.5
   contribute significant gradient signal, but it's signal that reinforces the teacher's
   existing behavior rather than providing novel information. The model early-stopped
   within 5-9 epochs, suggesting rapid convergence to the teacher's knowledge.

### Comparison to winning solutions

The 2024 1st place gained +0.025 from pseudo-labeling, but they also used:
- CE loss (not BCE)
- 10s overlapping context (not 5s)
- Data quality filtering (signal statistics + Google classifier)
- Ensemble of 6-9 models for inference

The 2025 1st place used iterative pseudo-labeling (3-4 rounds) combined with SED
attention heads — the architecture change was inseparable from the pseudo-labeling gain.

## Recommendations

1. **Priority pivot to Phase A** (SED architecture): Replace global avg pool + linear
   head with AttHead (GeM freq + attention pooling). This is the single biggest gap vs
   winners (+0.03-0.05 estimated).

2. **CE loss**: Switch from BCE to CE for training, sigmoid at inference. Two winners
   found it "significantly better."

3. **Overlapping context**: Process 10s windows, predict center 5s. The 2024 winner
   gained +0.015 from this alone.

4. **Return to pseudo-labeling with SED teachers**: Stronger teachers that detect brief
   calls will produce better pseudo-labels. Iterate with temperature scaling (T=0.6).

## Artifacts

- Checkpoints: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/015-student-training/checkpoints/`
- ONNX models: `010-submission-pipeline/artifacts/weights/{protoclr,eat}-2026-student-v1/`
- GPU check: `outputs/gpu_check/2026-protoclr-student-v1_5415802.{txt,png}`
- Training logs: `logs/slurm_*_student_*.log`
