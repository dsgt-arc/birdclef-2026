# Pantanal-Distill V18 (0.930 LB) — Annotated Reference

Source: `kaggle kernels pull dingjiarun/pantanal-distill-birdclef2026-improvement-a4dc68`
Score: **0.930 LB** (currently #2 on public LB)
Extracted code: `user/murilo/notes/pantanal-distill-0930.py` (2985 lines)
Pulled: 2026-03-30

---

## Architecture

### ProtoSSM v2 (V18 config)

Same Perch + ProtoSSM architecture as our exp 018, but **significantly scaled up**.

| Parameter | Our v4 (0.918) | V17 (0.925) | V18 (0.930) |
|-----------|---------------|-------------|-------------|
| d_model | 128 | 192 | **320** |
| d_state | 16 | 16 | **32** |
| n_ssm_layers | 2 | 3 | **4** |
| n_prototypes | 1 | 1 | **2** |
| meta_dim | 16 | 16 | **24** |
| cross_attn_heads | 4 | 4 | **8** |
| dropout | 0.15 | 0.15 | **0.12** |

Pipeline: `Perch → OOF probes → ProtoSSM + MLP ensemble → ResidualSSM → TTA → post-processing`

### ResidualSSM (V18)

| Parameter | Our v4 | V18 |
|-----------|--------|-----|
| d_model | 64 | **128** (2x) |
| d_state | 8 | **16** (2x) |
| n_ssm_layers | 1 | **2** (2x) |
| correction_weight | 0.3 | **0.35** |

### MLP Probes (V18)

| Parameter | Our v4 | V18 |
|-----------|--------|-----|
| hidden_layers | (128,) | **(256, 128)** (2-layer) |
| PCA dim | 64 | **128** (2x) |
| alpha (blend) | 0.40 | **0.45** |
| C (LogReg) | 0.50 | **0.75** |
| min_pos | 8 | **5** |
| max_iter | 300 | **500** |

## Training

| Parameter | Our v4 | V18 |
|-----------|--------|-----|
| n_epochs | 60 | **80** |
| lr | 1e-3 | **8e-4** |
| weight_decay | 2e-3 | **1e-3** |
| patience | 15 | **20** |
| mixup_alpha | 0.3 | **0.4** |
| focal_gamma | 2.0 | **2.5** |
| label_smoothing | 0.02 | **0.03** |
| OOF splits | 3 | **5** |
| SWA start | 70% | **65%** |
| Scheduler | OneCycleLR | **+ CosineRestart(T=20)** |

### Loss Function

```
loss = focal_bce(gamma=2.5, pos_weight capped at 25.0, label_smoothing=0.03)
     + 0.15 * distill_mse(ProtoSSM_logits, Perch_logits)
     + 0.1 * family_bce(taxonomy aux head)
```

Taxonomy aux head uses `class_name` (5 groups: Aves/Amphibia/Insecta/Mammalia/Reptilia),
NOT genus (161 groups). This matches what our v4.2 tried — but v4.2 regressed. The
difference may be in the implementation details or interaction with other changes.

### Key Finding: MLP-Only Ensemble

The final ensemble weight is **ProtoSSM=0.0, MLP=1.0** — the MLP probes alone outperform
the ProtoSSM+MLP blend on this dataset. This is determined via OOF grid search.
Our v4 used a fixed 0.5/0.5 blend.

## Post-Processing (7 stages)

1. **Prior fusion**: site-hour priors as logit shifts (λ_event=0.45, λ_texture=1.1)
2. **ProtoSSM + MLP ensemble**: weight=0.0 (MLP-only in V18)
3. **ResidualSSM correction**: +0.35 * correction (if wall_time < 4 min)
4. **Per-taxon temperature**: Aves=1.10 (sharpen), texture=0.95 (soften)
5. **File-level confidence**: `pred *= file_max^0.4` (rank-aware scaling)
6. **Adaptive delta shift**: `alpha = 0.20 * (1 - confidence)` — smooth uncertain windows
7. **Per-class threshold sharpening**: 10-value grid, OOF-optimized per species

## Fusion Parameters

```python
"best_fusion": {
    "lambda_event": 0.45,          # (our v4: 0.4)
    "lambda_texture": 1.1,          # (our v4: 1.0)
    "lambda_proxy_texture": 0.9,    # (our v4: 0.8)
    "smooth_texture": 0.35,
    "smooth_event": 0.15,
}
```

## What Drives the +0.012 over Yuriy (0.925)

1. **Bigger model** (+0.006 est): 320 d_model, 4 SSM layers, 8 attention heads
2. **Deeper probes + ResidualSSM** (+0.003 est): 2-layer MLP, 128-dim ResSSM
3. **Post-processing refinements** (+0.002 est): finer threshold grid (10 vs 5),
   5 TTA shifts (vs 3), adaptive smoothing
4. **Cosine restart + longer training** (+0.001 est): 80 epochs with T=20 restarts

## What Drives the +0.012 over Our v4 (0.918)

All of the above plus:
- 5-fold OOF (vs our 3-fold) — better model selection
- MLP-only ensemble (vs our 50/50 blend) — found via OOF grid
- PCA-128 (vs our 64) — more embedding variance captured
- Per-class thresholds (we use fixed 0.5)
- The v4→v4.2 changes that regressed us may work correctly in this larger architecture

## Implications for Exp 018

**Path to 0.925-0.930:**
1. Scale d_model from 128→256 (matching Yuriy), then 256→320 (matching V18)
2. Add 5-fold OOF (currently 3-fold)
3. Wider TTA shifts [0, ±1, ±2]
4. Per-class threshold optimization from OOF
5. 2-layer MLP probes with PCA-128
6. Cosine restart scheduler

**But first: fix the v4.2 regression.** We need to recover 0.918 before scaling up.
The V18 notebook uses 5-way taxonomy aux head (same as our v4.2 "fix"), so the
regression must be from one of the other 3 changes or an implementation bug.
