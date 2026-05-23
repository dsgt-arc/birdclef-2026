# 2026-05-21 — nina2025 EoS.6 (LB 0.949) + hideyukizushi reproduce notebook: methodology deep-dive vs. our exp 031/032 direction

**Sources vendored at:**
- `vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/` (nina2025 ensemble, LB 0.949)
- `vendor/kaggle/notebooks/birdclef-2026/bird26-reproduce-perch-protossm-resssm-inf-train/` (hideyukizushi, the training notebook the ensemble inherits from)
- `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/` (Tucker's SED, used as component)

**Pre-existing internal reference:**
- `user/murilo/notes/yuriy-protossm-v5-annotated.md` covers Yuriy's V5 (LB 0.925, 439 lines). This new note documents the **delta** between V5 and EoS.6's V18 stack + the Karnakbayev "Power Optimization" derivative that pushes the single-pipeline LB from 0.928 → 0.949.
- `user/murilo/experiments/018-perch-probes/` is our own Perch+probes ladder that hit ~0.911 OOF.

**Time budget**: today is 2026-05-21; competition ends 2026-06-03 (12 days). This note is written with the explicit framing that the LB-targeting decision is "what's the highest-EV use of the remaining 12 days."

---

## TL;DR — what nina2025's stack actually is

The **0.949 LB stack is NOT a better CNN-SED model**. It's a layered post-Perch pipeline where the heavy lifter is **per-file temporal aggregation of pre-computed Perch v2 embeddings**, with Tucker's SED CNN included only as a ~40% mixing weight inside the per-component blend.

**Stack composition (final EoS.6 (bz) blend, v.15):**
| Weight | Component | Single-LB | What it is |
|---:|---|---:|---|
| 0.965 | `Model_74` = Karnakbayev_PowerOptimization_LB0948 | 0.949 | ProtoSSM+ResSSM + Tucker SED (60/40) + power-optimized post-processing |
| 0.021 | `Model_52` = Karnakbayev_PowerOptimization_LB0948 (variant) | 0.949 | Same recipe with different per-component xSED `[0.605, 0.395]` |
| 0.014 | `Model_21` = yukiZ ProtoSSM+ResSSM | 0.928 | Pure ProtoSSM stack, no SED, no power-opt |

96.5% of the signal comes from one Karnakbayev variant. The other two contribute diversity (1.4% + 2.1%). This isn't a clever blend of independent models — it's "submit your strongest single pipeline + a sliver of variance reduction."

**Karnakbayev = ProtoSSM stack (yukiZ recipe) + Tucker SED + post-processing fusion + rank-aware power=0.6.** All three of those building blocks are public.

**Single-model trail** (each row is *one* component, before any blending):
- Tucker SED (CNN): **0.917**
- yukiZ ProtoSSM+ResSSM (no SED): **0.928**
- Karnakbayev PowerOpt (ProtoSSM+SED+post-proc): **0.948–0.949**
- Derek `exp019-eos4-rank-power-06` (also 0.949)
- Yaroslav `v6_0949_replay` (also 0.949)

The +0.021 LB jump from 0.928 (ProtoSSM alone) to 0.949 (Karnakbayev) is the **biggest single-recipe lift in the public ecosystem right now** and it comes from three things stacked, not from any single change:
1. Adding Tucker's SED into the per-row blend at ~40%
2. `rank_aware_scaling(power=0.6)` instead of 0.5
3. Isotonic calibration + `lambda_prior=0.5` (up from 0.4)

---

## Section 1 — the ProtoSSM+ResSSM training notebook (hideyukizushi/bird26-reproduce-...)

This is the training counterpart that EoS.6 ingests via the `kernel_sources` pointer to `hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177`. **The trained weights are public**: `hideyukizushi/sgkfk-202604041716`.

### Inputs and shape

- Train on **labeled soundscapes only** (`train_soundscapes_labels.csv`, ~708 trusted full windows = 59 files × 12 windows).
- Per file: 60s → 12 × 5s windows → Perch v2 embeddings `(B=N_files, T=12, D=1536)` + Perch BC-classifier logits `(B, T, 14795)` → mapped to 234 species via `bc_index` lookup → `(B, T, 234)`.
- Metadata: `site` (20 sites), `hour_utc` (24 hours).
- All training happens **on CPU**, takes minutes (file count is tiny).

### Architecture: `ProtoSSMv2` (called v4 in code comments)

```
Inputs:
  emb         (B, T=12, 1536)    Perch v2 embeddings
  perch_logits (B, T, 234)        Perch logits mapped to comp classes
  site_ids    (B,)                site index
  hours       (B,)                hour-of-day

Pipeline:
  emb → Linear(1536→d_model) → LayerNorm → GELU → Dropout  (d_model=320 in V18)
        + learnable positional encoding (1, 12, d_model)
        + meta_proj([site_emb || hour_emb])                (meta_dim=24)

  For each of n_ssm_layers (=4 in V18):
    h_f = SelectiveSSM(h)                    # Mamba-style fwd scan, d_state=32
    h_b = SelectiveSSM(h.flip(1)).flip(1)    # bwd scan
    h = LayerNorm(Linear(2d_model→d_model)([h_f||h_b]) + residual)

  h = TemporalCrossAttention(h)              # MHA(n_heads=8) + FFN with LayerNorm

  # Prototype head:
  h_norm = F.normalize(h, dim=-1)
  p_norm = F.normalize(self.prototypes, dim=-1)    # (n_classes, d_model)
  sim = (h_norm @ p_norm.T) * proto_temp + class_bias   # learnable proto_temp via softplus

  # Gated fusion with Perch logits (per-class alpha):
  alpha = sigmoid(self.fusion_alpha)                # learnable, per-class
  species_logits = alpha * sim + (1 - alpha) * perch_logits

  family_logits (aux) = Linear(d_model→n_families)(h.mean(dim=1))
```

**Key design moves:**
- **Selective SSM (Mamba-style)** with per-step input-dependent discretization. For T=12 the sequential scan is trivially cheap on CPU.
- **Prototypes initialized from data**: `prototypes[c] = mean of normalized backbone features for samples where label[:,c]==1`.
- **Per-class learnable alpha** for "trust ProtoSSM vs trust Perch logits." After training, alpha distribution typically shows ProtoSSM dominant for well-represented classes, Perch dominant for rare ones.
- **Class bias** = per-class learnable scalar baked into the logit before BCE.

### V18 CFG (the upgrade over Yuriy V5)

```python
proto_ssm = {
    "d_model": 320,           # V5 was 128/256
    "d_state": 32,            # V5 was 16
    "n_ssm_layers": 4,        # V5 was 2
    "dropout": 0.12,          # V5 was 0.15-0.20
    "n_prototypes": 2,
    "meta_dim": 24,
    "use_cross_attn": True,
    "cross_attn_heads": 8,    # V5 was 4
}
proto_ssm_train = {
    "n_epochs": 80,           # V5 was 60
    "lr": 8e-4,               # V5 was 1e-3
    "weight_decay": 1e-3,
    "patience": 20,
    "pos_weight_cap": 25.0,
    "distill_weight": 0.15,   # KD term coefficient
    "proto_margin": 0.15,
    "label_smoothing": 0.03,
    "oof_n_splits": 5,
    "mixup_alpha": 0.4,
    "focal_gamma": 2.5,
    "swa_start_frac": 0.65,
    "swa_lr": 4e-4,
    "use_cosine_restart": True,
    "restart_period": 20,
}
```

### Loss

```python
# Per-class pos_weight from class frequency, capped at 25
pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=25)

# Primary: focal BCE on species logits
loss_main = focal_bce_with_logits(species_logits, labels,
                                  gamma=2.5, pos_weight=pos_weight)

# KD: MSE against Perch logits (NOTE: this is logit-space KD, category #2 in our taxonomy)
loss_distill = F.mse_loss(species_logits, perch_logits)

# Family auxiliary
loss_family = F.binary_cross_entropy_with_logits(family_logits, family_targets)

# Total
loss = loss_main + 0.15 * loss_distill + 0.1 * loss_family
```

**Important**: the `loss_distill = MSE(species_logits, perch_logits)` is **distillation from Perch's 234-d mapped logits**, NOT from Perch's 1536-d embeddings. This is structurally distinct from Tucker's "Perch KD" (which distills embeddings into a CNN backbone). And the SSM is trained ONLY on labeled soundscapes — the distillation just keeps it from drifting too far from Perch's prior on the same 708 windows.

### Training mechanics

- **File-level mixup**: pairs of files mixed with `lam ~ Beta(0.4, 0.4)`, `lam = max(lam, 1-lam)` (dominant sample stays dominant). Skip first 5 epochs (warmup).
- **OneCycleLR** with `pct_start=0.1, anneal_strategy='cos'`.
- **SWA** from epoch ≥ `n_epochs * 0.65`. After training, if ≥3 SWA snapshots collected, average them and load as final weights; else use best-val checkpoint.
- **CutMix on temporal dim** (with `cutmix_prob=0.3`): random contiguous T-slice swapped with another file's slice.
- **Early stopping** with `patience=20`.

### ResidualSSM (second pass)

Lightweight (`d_model=128, d_state=16, n_ssm_layers=2` in V18) SSM trained on the **residual** between `labels_float - sigmoid(first_pass_score)`. Output head initialized to zero so corrections start small. Applied at inference as:

```python
final_score = first_pass_score + correction_weight * residual_ssm(emb, first_pass_score)
# correction_weight = 0.35 in V18
```

This is the "boost the systematic errors" trick — directly analogous to gradient-boosting on residuals.

---

## Section 2 — post-processing chain (the hidden +0.02 LB)

Every score below is logit-space until the final sigmoid. Pipeline executes in this order:

1. **Per-taxon temperature scaling**: `Aves=1.10, Texture=0.95` (texture = Amphibia + Insecta). Sigmoid after dividing by class temperature.
2. **Genus proxies for unmapped species** (Cell 23): for species not in Perch's vocabulary, take the max (or mean) of all genus-matched Perch logits. Enabled for Amphibia, Insecta, AND Aves.
3. **Metadata priors** (`fit_prior_tables` + `fuse_scores_with_tables`): site-level, hour-level, and site×hour-level positive-class frequency tables computed from labeled soundscapes. Fused with `lambda_event=0.45, lambda_texture=1.10, lambda_proxy_texture=0.90` (V18). Texture gets stronger prior weight because texture species are spatially deterministic. **Karnakbayev variant**: bumps `lambda_prior` from 0.4 to **0.5** in `apply_prior`.
4. **Temporal smoothing**: `smooth_cols_fixed12(alpha=0.35)` for texture species (mean-neighbor average), `smooth_events_fixed12(alpha=0.15)` for event species (local-MAX neighbor — preserves transient calls).
5. **File-level confidence scaling** (`file_level_confidence_scale`, top_k=2): `score *= top2_mean_over_windows`. Boosts confident files, suppresses uncertain ones. **2025 Rank 1/2 technique.**
6. **Rank-aware power scaling** (`rank_aware_scaling`): `score *= max_over_windows ** power`. V18 default = 0.4; **Karnakbayev variant** uses **0.6**. 2025 Rank 3 technique.
7. **Delta shift smoothing** (`adaptive_delta_smooth`): `new[t] = (1-a) * old[t] + a * mean(old[t-1], old[t+1])` where `a = base_alpha * (1 - confidence)`. Smooths low-confidence windows toward neighbors. base_alpha=0.20 in V18. **2025 Rank 1 technique.**
8. **Per-class threshold sharpening** (`apply_per_class_thresholds`): per-class thresholds learned from OOF F1 grid search `[0.25, ..., 0.70]`, applied as a piecewise sharpening map (push above-threshold up, below-threshold down).

In addition, the V18 stack runs:
- **TTA via circular shifts** of the 12-window sequence (`tta_shifts = [0, 1, -1, 2, -2]` in V18; 5x compute, averaged).
- **OOF-optimized ensemble weight** between ProtoSSM and MLP-probe scores (grid search 0.0–1.0 step 0.05 on macro AUC).
- **Isotonic calibration** per class (with F1 threshold optimization) before threshold sharpening.

---

## Section 3 — the MLP probe branch (parallel to ProtoSSM)

For each class with ≥`min_pos=5` (V18) positive samples:
- Take Perch embedding → standardize → PCA(128) → concat with 14 hand-crafted features:
  - `raw, prior, base, prev_base, next_base, mean_base, max_base, std_base`
  - `diff_mean, diff_prev, diff_next` (deviation from file context)
  - `raw*prior, raw*base, prior*base` (interactions)
- Train per-class MLP `(256, 128) → 1` with class-balanced oversampling.
- Blend with prior-fused base: `score = (1-alpha)*base + alpha*probe`, `alpha=0.45` (V18).

This adds ~5-10 classes of orthogonal lift to ProtoSSM (i.e., classes where temporal modeling didn't help but raw embedding similarity does).

---

## Section 4 — the SED branch (Tucker's distilled CNN, Model_1)

Tucker's distilled SED is identical to the audit we already did in `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`. Quick recap:
- `tf_efficientnet_b0.ns_jft_in1k` backbone (NS-JFT, not EffNetV2)
- AttHead with **logit-space aggregation** (exp 032's planned port)
- Perch v2 KD on 1536-d embedding via `feat.detach()` on cls branch
- 25 epochs cosine, batch 64, `MIXUP_HARD=True` (Sydorskyi-style), focal × labeled-SS mixup
- Dual-supervision BCE (clip + frame_max), blend at inference `0.5*sigmoid(clip) + 0.5*sigmoid(frame_max)`
- **Trained at 5-fold; LB 0.917 with 5-tap Gaussian smoothing.**

In Karnakbayev's stack the SED branch enters as:

```python
PROTO_W, SED_W = 0.60, 0.40                # row in solutions['Models']['xSED']
final = PROTO_W * proto_score + SED_W * sed_score
```

After fusion at the per-class row level, rank-aware power=0.6 + other post-proc is applied. The 60/40 weighting is the load-bearing number — pure ProtoSSM (Model_21) is 0.928, adding the SED at 40% with post-proc tuning gets to 0.948-0.949. **The SED is necessary but not sufficient** — it adds ~+0.02 LB only when paired with the right post-processing.

---

## Section 5 — how this differs from exp 031 / proposed exp 032

### Our exp 031 KD (REJECTED, paired-Δ = −0.006):
We added **Perch-embedding KD via MSE+detach** as a single-variable knob on top of HGNet+LSEHead+20ep+α=1.0. The KD branch trains the backbone toward Perch alignment; the SED head learns to read off detached features. Verdict: didn't transfer, paired Δ = −0.006.

### Our proposed exp 032 (KD + bug-fixed AttHead):
Same KD branch + replace LSEHead with logit-space AttHead. Tests whether the *consumer* of the Perch-aligned features matters. Still operates on per-window 5s mel crops, no temporal modeling, no metadata, no post-processing chain.

### What nina2025's 0.949 stack does that we don't:

| Axis | Our exp 031/032 path | EoS.6 / Karnakbayev path |
|---|---|---|
| Training data | Full focal recordings + labeled SS | **Labeled SS only** (708 windows) |
| Backbone training | Long GPU CNN training (~40-120 min/run) | None for ProtoSSM (CPU minutes); SED is one component |
| Where Perch enters | Teacher for KD on backbone features | **Frozen feature extractor + frozen classifier at inference** |
| Temporal modeling | Single 5s window (or 15s in multi-context) | **Bidirectional Mamba-style SSM over 12×5s sequence** |
| Metadata | Ignored | **Site/hour/site-hour priors fused via Bayes-ish logit add** |
| Cross-window context | None (per-window inference) | **Cross-attention + SSM + smoothing across windows** |
| Genus proxies | Not used | **Used for Amphibia/Insecta/Aves unmapped species** |
| Post-processing | TopN + Gaussian smoothing (pp2) | **8-step chain: temp → file-conf scale → rank-aware power → delta smooth → per-class threshold + isotonic + priors** |
| Ensemble | 4-fold same recipe (v5j 4-fold + pp2 = 0.917) | **ProtoSSM + ResidualSSM + MLP probes + SED at 60/40 + 5x TTA** |
| Single-fold LB | 0.890–0.906 | 0.928 (ProtoSSM alone), 0.949 (full Karnakbayev) |

### The fundamental methodological inversion

We've been **trying to build a stronger SED CNN to compete with Perch's prior**. Their pipeline **uses Perch's prior as the foundation and adds the smallest possible amount of learning on top** (ProtoSSM has only ~2-3M params, trains on 708 windows in minutes).

This is the same insight from `2026-04-09-data-centric-pivot.md` and the construction of exp 018, but the field has now developed it into a tight playbook with a known ceiling around 0.949. **Our exp 018 reached 0.911-0.925** on the same family of ideas — we're already on this ladder, just one rung below where the leaderboard is.

### What exp 031's KD lever should have been

The KD term that actually contributes to the 0.949 stack is INSIDE the ProtoSSM training loop:

```python
loss_distill = F.mse_loss(species_logits, perch_logits)   # both (B, T, 234)
```

That's **logit-space distillation on the same labeled soundscapes** — KD category #2 in our taxonomy (the EliKal/yukiZ style we noted as "NEVER TRIED" in `project_kd_categories_2026_05_09.md`). It's a much smaller-stakes regularizer (`distill_weight=0.15` vs our `alpha_distill=1.0`) and it lives inside a temporal model, not inside a per-window CNN backbone. **The KD that works here is structurally not the KD we tested in exp 031.**

---

## Section 6 — implications for the final 12 days

### What's no longer worth pursuing (low EV)

1. **exp 032 as currently scoped** (KD + bug-fixed AttHead on HGNet 5s mel) — the structural ceiling is wrong. Even if it lifts to 0.917 (Tucker single-CNN), we'd still be 0.032 LB below the public 0.949 single-pipeline.
2. **Any "from-scratch CNN-SED" track** that doesn't compose into the ProtoSSM family.
3. **More v5j retrains** — we already understand the hardware/code variance; this stack is past the v5j-alone ceiling regardless.

### What's high EV

1. **Reproduce/inherit the ProtoSSM+ResSSM stack as our baseline starting point.**
   - Public weights at `hideyukizushi/sgkfk-202604041716` (the `kernel_sources` of EoS.6).
   - Inference is CPU-only (no GPU constraint). Single-pipeline LB = 0.928.
   - Our `user/murilo/experiments/018-perch-probes/` already has 80% of the surrounding infrastructure (perch ONNX, BC-index mapping, OOF probes, prior tables). We need to add the ProtoSSM module + ResidualSSM + V18 CFG + the post-processing chain.

2. **Reproduce Karnakbayev "Power Optimization" wrapper** (the +0.021 LB delta) on top of #1. Three components: 60/40 SED fusion, rank-aware power=0.6, lambda_prior=0.5 + isotonic. All are in the eos-6-bz notebook Cells 13 & 19 — code is available.

3. **Stack our existing assets as small-weight ensemble components for variance reduction.**
   - Our v5j single-fold (LB 0.906 reference, 0.897-0.899 retrains) can slot in as a SED variant alongside Tucker's.
   - Our Perch+SED ensemble (LB 0.929) can be averaged in at low weight.
   - 4-fold ensembles of either reduce single-fold variance.

4. **Specific levers to explore IF time permits after replication**:
   - Train a fresh ProtoSSM with our v5j SED as the SED branch (vs Tucker's). Net swap, one knob.
   - Pseudo-label more "trusted" soundscape windows (the current pipeline uses only 708 — there are unlabeled soundscapes we could PL-bootstrap). This is the data side of the same idea exp 014/015/025 tried, but applied to ProtoSSM input rather than CNN training.
   - Try **logit-space KD inside our current SED training** (category #2, never tried) — but only if a fresh SED gives orthogonal signal to Tucker's; otherwise this is wasted compute.

### Risk note

The 0.949 ceiling is **the public-notebook saturation point**. ~12 teams are at 0.949 right now per the discussion. Reproducing it puts us at the median of the top cluster, not ahead of it. The actual top-of-leaderboard delta likely comes from un-published-in-notebook tricks that the discussion posts (which you mentioned you'll share next) might illuminate.

---

## Files and pointers

| Path | Purpose |
|---|---|
| `vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/birdclef-2026-eos-6-bz.ipynb` | The 0.949 ensemble inference notebook |
| `vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/birdclef-2026-eos-6-bz.py` | Extracted code for quick scrolling (724KB; most of it is duplicate Karnakbayev variants) |
| `vendor/kaggle/notebooks/birdclef-2026/bird26-reproduce-perch-protossm-resssm-inf-train/` | The training notebook (ProtoSSM+ResSSM training loop) |
| `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/` | Tucker's distilled SED (single LB 0.917, the SED branch in Karnakbayev) |
| `user/murilo/notes/yuriy-protossm-v5-annotated.md` | Existing annotated reference for Yuriy V5 ProtoSSM (LB 0.925, V5 CFG) — V18 in this note is the upgrade path |
| `user/murilo/experiments/018-perch-probes/` | Our internal Perch+probe ladder (~0.911 OOF) — most of the surrounding infra already exists |
| `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Tucker SED audit (used unchanged here) |
| `analysis/notes/2026-04-28-next-experiment-direction.md` | The "PL vs KD distinction" note — KD category #2 (logit-space) is the ProtoSSM's `loss_distill`; we never explored it |

## Open questions to resolve as more discussion findings arrive

(Place to append once Murilo shares additional discussion posts.)

- What's the public LB top-cluster spread? Is 0.949 the saturation or are there 0.95+ tracks?
- Are there discussion posts about the **post-Perch tricks** that aren't in the published notebook (e.g., better priors, additional regularizers, alternative aggregators)?
- Is anyone publishing a *better SED branch* than Tucker's 0.917? Our 0.929 Perch+SED ensemble could be that, if the path is clear.
- Are there hidden-test domain-shift findings that argue for or against site/hour priors at scale?
