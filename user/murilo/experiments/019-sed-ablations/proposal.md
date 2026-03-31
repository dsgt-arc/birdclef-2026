---
experiment: "019-sed-ablations"
status: proposed
date: 2026-03-30
author: Murilo Gustineli
---

# Experiment 019: Perch-Guided SED Training

## Hypothesis

Using Perch ProtoSSM (~0.920 LB) as a teacher to pseudo-label ALL 10K+
unlabeled soundscapes, then training a strong SED model on labeled +
pseudo-labeled data with HGNet-inspired improvements, will push the SED
model from 0.860 to 0.900-0.915 LB — closing the gap to a competitive
single-model submission and enabling future Perch+SED ensemble.

## Background

### The unlabeled data opportunity

BirdCLEF 2026 has ~10,600 unlabeled soundscape files (127K 5-second windows)
vs only 59 fully-labeled files (708 windows). Our SED model has never seen
99.4% of the available soundscape data. Meanwhile, Perch ProtoSSM scores
~0.920 on the test set and can label those files offline.

### Why exp 015 pseudo-labeling failed

Our previous attempt (exp 015) used a weak teacher (ProtoCLR/ConvNeXt clip-level,
0.767 LB) with aggressive pseudo-label weight (0.5). Both ProtoCLR (−0.005) and
EAT (−0.024) regressed on LB despite flat/improved offline metrics. Root causes:
- Teacher too weak on soundscapes (clip-level architecture can't handle 60s files)
- Weight too aggressive (0.5 mixed noisy pseudo-labels equally with ground truth)
- Single round (no iterative refinement)

### What's different now

| Factor | Exp 015 | Exp 019 |
|--------|---------|---------|
| Teacher | Clip-level 0.767 LB | Perch ProtoSSM ~0.920 LB |
| Teacher architecture | Global average pool | Temporal SSM + metadata priors |
| Pseudo-label weight | 0.5 | 0.1-0.2 (conservative, ramping) |
| Rounds | 1 | 3-4 (iterative refinement) |
| SED model | EfficientNet-B0 (exp 012) | Improved EfficientNet/HGNet (this exp) |
| Confidence filtering | None | Power transform γ=1.5-2.5 |

### HGNet findings

ttahara's HGNetV2-B0 (0.884 LB) outperforms our v4b (0.860 LB) by +0.024.
Gap analysis (`notes/gap-analysis-hgnet-vs-effnet.md`) identified quick wins:
- LSEHead (Log-Sum-Exp pooling) instead of attention head
- BCEWithLogitsLoss instead of soft CE
- Stronger MixUp (α=1.0, θ=0.8 label clipping, 5-epoch warmup)
- Lower weight_decay (1e-4 vs 0.01)
- 4-fold MultiLabel Stratified GroupKFold ensemble

Applying these before pseudo-labeling gives us a stronger student model.

### Competition context

- Current #1: 0.939 LB
- Our Perch ProtoSSM v4.1: ~0.920 (pending)
- Our SED v4b: 0.860 LB
- 2025 winners used multi-round pseudo-labeling for +0.032-0.06 LB gain
- 2 months until deadline, unlimited PACE GPU compute

## Method

### Approach

Three-phase experiment building on each other:

**Phase A: Improved SED baseline** — Apply HGNet quick wins to establish a stronger
student model before pseudo-labeling.

**Phase B: Perch pseudo-label extraction** — Run Perch ProtoSSM on all unlabeled
soundscapes (offline on PACE GPU) to generate soft labels.

**Phase C: Pseudo-label training** — Train the improved SED model on labeled +
pseudo-labeled data with multi-round refinement.

### Setup

| Component | Details |
|-----------|---------|
| Data | 35K focal + 708 labeled soundscape windows + 127K pseudo-labeled windows |
| Compute | PACE RTX 6000 GPU, ~4h per training run, ~2-4h for Perch extraction |
| Dependencies | timm (HGNetV2-B0), torch, torchaudio, Perch v2 (TF, GPU) |
| Code | `user/murilo/experiments/019-sed-ablations/` |
| Branch | `mg/019-sed-ablations` |

### Procedure

#### Phase A: Improved SED Baseline (2-3 days)

**Step 1: Quick wins ablation** (parallel, 4h each on PACE)

| Run | Change from v4b | Expected |
|-----|----------------|----------|
| v5a | BCEWithLogitsLoss (replaces soft CE) | Better multi-label optimization |
| v5b | MixUp α=1.0, θ=0.8, warmup=5 epochs | Stronger regularization |
| v5c | weight_decay=1e-4 (from 0.01) | Less aggressive regularization |
| v5d | LSEHead (replaces AttHead) | Simpler, better for sparse events |
| v5e | HGNetV2-B0 backbone (replaces EfficientNet-B0) | Different architecture |

**Step 2: Combine winners** — Stack the ablations that improve over v4b.

**Step 3: 4-fold training** — Train the best variant with MultiLabel Stratified
GroupKFold (4 folds). Export all 4 models as ONNX/OpenVINO.

**Step 4: Validate on LB** — Target: single-fold ~0.870, 4-fold ensemble ~0.885.

#### Phase B: Perch Pseudo-Label Extraction (1 day)

**Step 5: Extract Perch embeddings + logits on ALL soundscapes** (GPU on PACE)

Run Perch v2 on the full ~10,600 unlabeled soundscape files. Use GPU variant
(`perch_v2` not `perch_v2_cpu`) for ~10x speedup. We did similar extraction
in exp 014 on the labeled soundscapes.

Output per file: embeddings (12, 1536) + mapped logits (12, 234).

**Step 6: Run ProtoSSM pipeline on extracted embeddings**

Apply the full ProtoSSM v4 pipeline (priors + probes + ProtoSSM + ResidualSSM +
post-processing) to generate soft labels for all 127K windows.

Output: soft probabilities (127K, 234) — one probability per species per window.

**Step 7: Confidence filtering with power transform**

Apply power transform to sharpen confident predictions:
`p_sharp = p^γ / (p^γ + (1-p)^γ)` with γ = 1.5 (conservative first round).

Filter windows where max species probability < threshold (e.g., 0.1) to remove
very uncertain predictions.

#### Phase C: Pseudo-Label Training (3-5 days)

**Step 8: Round 1 — Conservative pseudo-labels**

Train improved SED (from Phase A) on:
- Focal recordings (35K files, weight=1.0)
- Labeled soundscapes (708 windows, weight=1.0)
- Pseudo-labeled soundscapes (127K windows, weight=0.1, γ=1.5)

Use BCE loss with per-sample weighting. Validate on LB.

**Step 9: Round 2 — Increase confidence**

Use Round 1 SED model + Perch ensemble as teacher for Round 2:
- Regenerate pseudo-labels with stronger teacher
- Increase weight to 0.15, increase γ to 2.0
- Train new SED model

**Step 10: Round 3 — Final refinement**

- Weight 0.2, γ=2.5
- 4-fold ensemble of Round 3 models
- Validate on LB — target: 0.900-0.915

### Variables

- **Independent**: SED architecture improvements, pseudo-label weight/confidence,
  number of training rounds
- **Dependent**: LB score, offline val AUROC, offline cmAP
- **Controlled**: Perch teacher (frozen at v4.1), labeled data split, training
  hyperparameters (lr, batch_size, epochs)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| LB score | Competition metric (row-wise AUROC) |
| val/auroc | Offline validation AUROC (90/10 split) |
| val/cmap | Class-mean average precision |
| Pseudo-label quality | Teacher AUROC on labeled soundscapes |

### Baselines

| Model | LB | Notes |
|-------|-----|-------|
| EfficientNet-B0 SED v4b | 0.860 | Current best SED |
| ttahara HGNetV2-B0 | 0.884 | Reference (4-fold ensemble) |
| Perch ProtoSSM v4.1 | ~0.920 | Teacher model |
| Competition #1 | 0.939 | Target to approach |
| Exp 015 (failed pseudo-labels) | 0.762 | Weak teacher, aggressive weight |

### Success Criteria

- **Phase A confirm if**: Any SED improvement ≥ 0.870 LB (single fold)
- **Phase A stretch**: 4-fold ensemble ≥ 0.885 LB
- **Phase C confirm if**: Pseudo-label SED ≥ 0.900 LB (clear gain from unlabeled data)
- **Phase C stretch**: Pseudo-label SED ≥ 0.915 LB (competitive with Perch)
- **Reject if**: Round 1 pseudo-labels regress from Phase A baseline (same failure as exp 015)

## Limitations

- Perch pseudo-labels have ceiling at ~0.920 — student can't exceed teacher
  (unless ensemble with Perch later adds diversity)
- Power transform sharpening loses calibration — pseudo-labels become more
  confident but may be overconfident for rare species
- Multi-round training is slow (3-4 rounds × 4-16h each) — but we have
  2 months and unlimited PACE compute
- LB submission limit (5/day) constrains how many variants we can validate
- The 0.939 gap likely requires ensemble (Perch + SED) not just better SED

## Leads To

- **Exp 020**: Perch + SED ensemble (two-kernel or single-kernel with distilled Perch)
- **Exp 021**: Advanced post-processing and inference optimization
- If SED reaches 0.915+: SED alone may be competitive, freeing the kernel
  budget from Perch inference entirely

## References

- [HGNet gap analysis](../../notes/gap-analysis-hgnet-vs-effnet.md)
- [ttahara HGNetV2-B0 annotated](../../notes/ttahara-hgnetv2-b0-annotated.md)
- [Exp 015 pseudo-label failure](../015-student-training/README.md)
- [Exp 018 ProtoSSM pipeline](../018-perch-probes/tasks.md)
- [Competition strategy](../../notes/competition-strategy.md)
- [2025 winner analysis](../../docs/references/kaggle/birdclef-2025/)
