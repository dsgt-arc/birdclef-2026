# 2026-04-26 — Sydorskyi 2nd place 2025 vs our v5j

Side-by-side mapping of [Sydorskyi 2nd-place BirdCLEF 2025 (Public 0.925, Private **0.928**)](../../../docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md) against our v5j (single 0.906 LB, 4-fold+pp2 0.917, ensemble-with-Perch 0.929). Goal: identify the differences most likely to explain the gap and produce an ordered list of bets for closing it.

Reference doc: `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` (read that first if you haven't).

## TL;DR

The gap is **not** any single piece of architecture or augmentation. It is a stacked combination of three structural decisions, each worth roughly +0.01 to +0.03 LB on its own, that we have not yet engaged:

1. **In-domain pretraining at scale** (Sydorskyi: 819K recordings, 7,489 species). Our exp 029 tested this at *3% of his scale* (28K recordings, 206 species) and lost −0.021 LB. The honest read: 2025 pretraining at our scale isn't enough; the right replication is XC-at-scale, not 2025-only.
2. **Iterative semi-supervised distillation on unlabeled soundscapes** with **soft pseudo-labels** + 40% sampling probability + sum-MixUp fusion. Our prior PL attempts (exp 015 row-addition w=0.5 → −0.005; exp 025 Mario waveform-mix 30% → −0.012) used either binarized labels or hard-mixing with weak teachers. Sydorskyi's recipe is structurally different.
3. **AttHead instead of LSEHead, and BCE+Focal instead of BCE.** We have ablated both individually (AttHead × 7 attempts: −0.007 to −0.018; CE × 7 attempts: regressed). But Sydorskyi's stack is **AttHead + BCE+Focal + label smoothing + balanced sampler + in-domain pretrained backbone** — no piece tested in isolation, but the *combination* is what reaches 0.917 without pseudo-labels.

If we have to pick *one* bet to invest in, **#2 (soft pseudo-labels with sum-MixUp fusion)** is highest leverage: it doesn't depend on the 819K-recording pretrain corpus we don't have, it directly engages our 04-25 per-frame finding (low-SNR soundscape chunks need *soundscape-domain* supervision, not training-recipe tweaks on focal-only data), and the published recipe is reproducible from public artifacts.

## Top-down comparison

| Dimension | Sydorskyi 2025 (selected_ebs) | Our v5j | Verdict |
|---|---|---|---|
| **Backbone** | EfficientNetV2-S `tf_efficientnetv2_s_in21k` (~21M params) | HGNetV2-B0 `hgnetv2_b0.ssld_stage2_ft_in1k` (~6M params) | Different family, ~3× param difference |
| **Init** | **In-domain pretrain on 819K recordings, 7,489 species** | ImageNet (no audio pretrain) | Big gap |
| **Head** | AttHead (GeMFreq + Conv1D attention) | LSEHead (mean freq + LSE temporal pool) | Different architecture |
| **Loss** | BCEWithLogits + sigmoid_focal_loss(α=0.25, γ=2) (1:1 sum) | BCEWithLogits only | Sydorskyi adds focal term |
| **Label smoothing** | 0.05 (mass-preserving) | None | Different |
| **Mel n_mels / n_fft / hop** | 128 / 2048 / 512 | 256 / 2048 / 313 | We have higher mel resolution |
| **Mel norm** | AmplitudeToDB(top_db=80) → standardize → min-max | AmplitudeToDB(top_db=80) → standardize → min-max | **Identical** |
| **Window** | 5s random crop | 5s random crop | **Identical** |
| **Sample rate** | 32 kHz | 32 kHz | **Identical** |
| **Batch size** | 64 | 64 | **Identical** |
| **Epochs** | 50 | 20 | We train for 40% as long |
| **Optimizer** | AdamW lr=1e-4, β=(0.9, 0.999), ε=1e-8 | AdamW lr=5e-4 | Different LR (5× higher) |
| **Scheduler** | CosineAnnealingWarmRestarts, T_0=50ep×steps, eta_min=1e-6 | OneCycleLR(div=25, final_div=4), warmup=5ep | Different shape |
| **Precision** | 32-true (no AMP) | 16-mixed (AMP) | We use FP16 |
| **MixUp variant** | Audio-domain, equal weight (lam=0.5), label SUM, p=0.5, no warmup | Spectrogram-domain, Beta(α=1.0), label INTERPOLATE, p=1.0, 5-epoch warmup, theta=0.8, per-sample | Many subtle differences |
| **BG noise mixing** | OneOf(2023-nocall-soundscapes, ESC-50 20 cats), p=0.5, convex blend amp ~ U[0.25, 0.75] | None | Big gap (we paused exp 028) |
| **SpecAugment** | freq mask(L=10, ×3, p=0.3) + time mask(L=20, ×3, p=0.3) | None | Big gap |
| **RandomFiltering** | 4 bands × U[-20, 0] dB, applied via STFT/iSTFT | None | Big gap |
| **Class sampling** | balanced (γ=−1) WeightedRandomSampler | None (uniform) | Big gap |
| **CV split** | 5-fold StratifiedGroupKFold by primary_label, grouped by recordist | 90/10 random | Different (no recordist grouping) |
| **Secondary labels** | Used (multi-hot at value 1.0, summed + clipped) | Used (same) | **Identical** |
| **Pseudo-labels** | 3 iterations × {Full + OOF}, 8-model teacher → soft → 40% sampling + sum-MixUp fusion | None in v5j; exp 015 (row-addition w=0.5) and exp 025 (Mario wave-mix 30%) both regressed | Big gap |
| **Inference** | OpenVINO FP16, 5s non-overlap, mean across folds, then 3-exp mean, then TopN N=1 | Custom ONNX, 5s with TTA(±2.5s, 50/25/25 weights), Gaussian temporal smoothing, TopN, site/hour priors | Different stack |

The "**identical**" rows are useful: mel pipeline (post-AmplitudeToDB+std+minmax), 5s window, 32 kHz, BS=64, secondary-label handling. Those are the things we *don't* need to change.

The "**Big gap**" rows are the candidate hypothesis space.

## Augmentation cascade — exact comparison

Sydorskyi's training step:

```
random 5s crop
→ (p=0.5) audio MixUp (lam=0.5, label SUM clipped to [0,1])
→ (p=0.5) BG noise: OneOf(nocall_soundscapes, ESC-50_20_cats), convex blend amp~U[0.25, 0.75]
→ peak-normalize wave
→ batch collate
→ RandomFiltering(4 bands, U[-20, 0] dB, via STFT/iSTFT)
→ nnAudio mel + AmplitudeToDB + std + minmax
→ SpecAugment(freq L=10×3 p=0.3, time L=20×3 p=0.3)
→ backbone forward
```

Our v5j training step:

```
random 5s crop
→ (always, p=1.0) Beta(α=1, α=1) per-sample SpecMixUp on the mel (interpolated wave + interpolated targets, with theta=0.8 weight cap)
   - skipped during the first 5 warmup epochs
→ (no BG noise mixing)
→ batch collate
→ NNAudio mel + AmplitudeToDB + std + minmax
→ (no SpecAugment, no RandomFiltering)
→ backbone forward
```

So ours is **monolithic on MixUp** (always on, but Beta-distributed lam) and **no other augmentation**. Sydorskyi runs **5 different augmentations stacked**, each at p=0.3 to 0.5, none mandatory.

## What we have already ruled out (and why this changes the read)

The CONTEXT-tracked pattern is: **22+ training-recipe changes, 0 LB wins**, including:

- AttHead × 7 (Phase C exp 021, Phase F v6k): all regressed −0.007 to −0.018
- Waveform MixUp × 3 (exp 017 v4j, exp 021 v6b, exp 025 v3): all regressed −0.012 to −0.031
- CE loss × 7: all regressed (most catastrophic: v10a = 0.494)
- BG noise mixing: paused (exp 028) before LB submission per the per-frame finding
- SpecAugment / RandomFiltering: never directly tested as single-variable on v5j
- Label smoothing: never tested
- Focal loss: never tested directly (only ASL once, regressed)
- Class balancing sampler: never tested
- 50 epochs: never tested (we always use 20–25)
- CosineAnnealingWarmRestarts: never tested (we use OneCycleLR)

The 04-25 per-frame finding adds: **the 22 failures share a pattern — they all try to make v5j more confident on hard crops, which teaches false positives.** This is consistent with each augmentation tested *in isolation* failing, while a *stack* of five augmentations (each at low probability) might not fail because none of them individually pushes the model toward overconfidence on a specific crop.

That's the structural reason why "Sydorskyi-style augmentation cascade" might work where any single augmentation hasn't:

- Each augmentation runs at p=0.3–0.5, so any individual sample sees only 0–2 of them.
- The five together produce *broad acoustic invariance* (channel filtering + spectral masking + temporal masking + BG noise + MixUp) without a single dominant pressure direction.
- BCE+Focal+label-smoothing on top adds *gradient-magnitude regularization* on the loss side.

This is testable. **It is also a multi-variable change** — a faithful Sydorskyi replication is a 4-axis ablation matrix, not a single experiment.

## The three big levers, ranked

### Lever 1 (highest EV) — Iterative pseudo-distillation on unlabeled soundscapes

**Why this is the top bet:**

1. **Engages our 04-25 mechanistic finding directly.** The per-frame analysis says the model is honest on low-SNR crops because BCE+random-crop gives no soundscape-domain supervision. Pseudo-labels *are* soundscape-domain supervision, with soft targets that match the model's actual confidence on those chunks.
2. **Doesn't depend on the 819K-recording pretrain we don't have.** Sydorskyi's pseudo-labels were the +0.029 Public lift on top of the pretrained baseline; the absolute lift on a non-pretrained baseline is testable.
3. **Reproducible from public artifacts.** Sydorskyi published [`bird-clef-2025-pseudo`](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pseudo), but more importantly the *recipe* is fully documented and we can run it on 2026 unlabeled soundscapes (we have ~127K segments of unlabeled data per CONTEXT.md; he had ~117K). Our exp 014 already produced 127K pseudo-labels but with a 0.885 teacher.
4. **Our prior PL failures are explainable** as recipe issues, not as proof PL doesn't work:
   - exp 015 used 0.767-LB clip-level teacher (way too weak) with hard pseudo-labels added as separate rows, w=0.5. Sydorskyi: 0.917-LB SED ensemble teacher, soft labels with `trim_min_prob=0.1`, mixed in via 40%-sampling + sum-MixUp.
   - exp 025 used Mario-style wave-mix at 30%. Sydorskyi: per-batch swap (40%) + audio MixUp at 50% — different fusion mechanism, also different threshold (`primary_label_min_prob=0.5`).

**What an exp 030 looks like**:

- Generate teacher predictions on 2026 unlabeled soundscapes using our **best** model (Perch+SED MLP-only ensemble, 0.929 LB) — much stronger teacher than exp 015's 0.767 or exp 025's 0.929 (we can use what's already there).
- Apply Sydorskyi's filter: `max(prob_c) >= 0.5` to keep chunk; zero out `prob_c < 0.1`.
- Add to training as a separate dataset with `sampling_prob=0.4`.
- Use audio-domain MixUp (not spectrogram MixUp) at p=0.5, lam=0.5, label-sum.
- Train v5j (HGNetV2-B0 + LSEHead + BCE) for 25-50 epochs.
- Decision gate: LB > 0.910 on stage 1 → run iteration 2.

**Risk**: our backbone init is ImageNet, not in-domain pretrained. The pseudo-distillation lift Sydorskyi measured was on top of in-domain pretrain, so we may see less. **Counter**: the gain is largely about *adding soundscape-domain training data*, which is a separate axis from backbone init.

**Estimated cost**: 1 day teacher inference + 1 day v5j retrain + LB submission. Reuses exp 014/025 infrastructure.

### Lever 2 — In-domain pretraining at meaningful scale

**Why this matters:**

1. Sydorskyi's pretrain → fine-tune transition was **+0.046 Public, +0.018 Private** (Table 5 single largest single-stage Public lift).
2. Our exp 029 tested 2025-only pretraining (28K recordings, 206 species) and lost −0.021 LB. The natural conclusion was "pretrain at our scale doesn't help." Sydorskyi's results suggest the conclusion should be "**pretrain at this small a scale doesn't help; pretrain at 30× larger scale helps a lot.**"
3. Sydorskyi's published checkpoint is **available directly** on Kaggle: [`bird-clef-2025-all-pretrained-models`](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models). We don't have to redo the 819K-recording pretrain.

**The catch**: the published checkpoints are EfficientNetV2-S and ECA-NFNet-L0. We use HGNetV2-B0. **Adopting his pretrained checkpoint requires switching backbone**, which itself is a multi-variable change vs v5j.

**What an exp 031 looks like**:

- Download Sydorskyi's pretrained EfficientNetV2-S backbone.
- Train **EffNetV2-S + LSEHead + BCE** on 2026 (single-variable change vs v5j: HGNetV2-B0 → EffNetV2-S with in-domain pretrain).
- Risk: this is *two* variables (backbone family AND init), but Sydorskyi himself reports "EfficientNetV2-S beat NFNet-L0 only after the pretrain" — the backbone choice may be load-bearing here.

**Estimated cost**: download + train + LB = 1.5 days. Bigger backbone (~21M vs 6M) so longer training but feasible.

**Lever 1 + Lever 2 stacked** is the closest replication of Sydorskyi to ours: get +0.018 from in-domain pretrain *plus* +0.020+ from soft-pseudo-distillation. If our prior is that the gains stack (paper Table 5 confirms they do), this is a credible path to 0.92+ on a single model.

### Lever 3 — The augmentation + loss + sampler stack

This is the *risky* bet. We have 22 failed isolated training-recipe changes. The hypothesis here is that **the stack works because no single augmentation dominates**. Tested as a single experiment:

- BCE + sigmoid_focal_loss (1:1) replaces pure BCE
- Label smoothing 0.05 (mass-preserving variant)
- WeightedRandomSampler (γ=−1, balanced)
- Audio MixUp(p=0.5, lam=0.5, label-sum) replaces our spec MixUp(p=1.0, Beta(1,1), label-interpolate)
- BG noise(p=0.5, OneOf nocall+ESC-50, amp~U[0.25, 0.75]) added
- SpecAugment(freq L=10×3 p=0.3, time L=20×3 p=0.3) added
- RandomFiltering(4 bands, U[-20, 0] dB) added
- 50 epochs (vs 20)
- CosineAnnealingWarmRestarts(eta_min=1e-6) (vs OneCycleLR)
- AdamW lr=1e-4 (vs 5e-4)

That's **10 simultaneous changes**. If it wins, we won't know which axes mattered. If it loses, we won't know which axes hurt.

**Recommended split**: don't do this as one experiment. Instead do an ablation matrix where each axis is added on top of the previous winner:

1. v5j → v5j + audio-MixUp variant (from spec to audio, label-sum) — single variable, testable
2. + sampler (balanced)
3. + label smoothing 0.05
4. + sigmoid_focal_loss in addition to BCE
5. + SpecAugment
6. + RandomFiltering
7. + BG noise (this is exp 028, paused)
8. + 50 epochs
9. + CosineAnnealingWarmRestarts in place of OneCycleLR

That's a 9-step staircase of single-variable changes, each gating on LB > previous step. Following the lit-review pattern, the best EV is to start the staircase at the augmentations Sydorskyi *ablates against* in his Table 8 (since he measured them directly and they all hurt Private when removed) and skip the ones he didn't measure.

But this is **not the recommendation right now.** The 22-failure pattern says the staircase is unlikely to climb.

### The recommendation in two sentences

Run **Lever 1 first** (pseudo-distillation with Sydorskyi's recipe, our existing teacher and unlabeled data). If that lands at 0.91+ on a single model, run **Lever 2 next** (Sydorskyi's published EffNetV2-S backbone + our v5j-style head/loss). Defer Lever 3 (full augmentation cascade) until either of the first two succeeds and gives a baseline that's no longer a "diffuse-on-hard-crops" structure.

## Specific implementation differences worth flagging

Things that look the same but aren't, or that we might quietly mis-implement:

### MixUp differences

| | Sydorskyi | v5j |
|---|---|---|
| Domain | Audio (waveform) | Spectrogram (mel) |
| Weight distribution | `lam = 0.5` (deterministic) | `lam ~ Beta(α=1.0, α=1.0)` (uniform on [0,1] effectively) |
| Label aggregation | `target = sum(t1, t2)` clipped to [0, 1] | `target = lam * t1 + (1 - lam) * t2` (interpolate) |
| Application probability | p=0.5 | p=1.0 |
| Warmup | None | 5 epochs disabled |
| Theta cap | None | 0.8 (max effective weight) |
| Per-sample vs batch | Per-sample | Per-sample |

The label aggregation difference is structural: Sydorskyi treats both samples as **equally present** (positives stay positive after mix), v5j treats them as **interpolated** (a 50/50 mix gives 0.5 to both classes). For multi-label, sum-and-clip preserves "both species are here," while interpolation says "one of these two species is here, with this confidence."

This matches Sydorskyi's data analysis (paper Section 4.2 Table 10): the soundscape distribution has 7%+ of files with 2+ species per 5s chunk, while training data has 90% with single species. Sum-mixup pushes the training distribution toward the test distribution; interpolation-mixup does not.

### Pseudo-label semantics

Our exp 015 used hard binarized pseudo-labels added as separate rows with weight 0.5 in the loss. Sydorskyi uses **soft probability vectors**, with `trim_min_prob=0.1` zero-out, fused into training via 40%-sampling-replacement and audio-domain MixUp with sum-aggregation. The labels themselves carry uncertainty information, and the MixUp combining a hard label with a soft label produces a mixed-uncertainty target vector for the loss.

### Inference postproc

| | Sydorskyi | v5j |
|---|---|---|
| TTA | None | ±2.5s shifts, 50/25/25 weights |
| Temporal smoothing | None | Gaussian |
| TopN | N=1, applied at *ensemble* level after averaging | TopN, applied per-segment |
| Priors | None (tested, didn't work) | Site/hour priors (+0.003 in pp2) |
| Per-model PP | Avoided (regressed) | Yes |

Sydorskyi's inference is *simpler* than ours but achieves +0.015 from PP alone. We achieve +0.003 from priors and have TTA + smoothing baked in. Worth checking whether **applying TopN N=1 at the ensemble level** post-averaging is a free win for our existing 4-fold or Perch+SED ensemble.

### Inference TopN — quick test

This is genuinely cheap. The math is `prob_a,s,c *= max_s'(prob_a,s',c)` per file × class. We can apply it post-hoc to our existing submission probabilities and resubmit. Cost: ~1 hour. Estimated lift: small (+0.003 to +0.010 if our pipeline doesn't already do something equivalent — we should check `submission.py:ENABLE_TOPN`).

## Things Sydorskyi did NOT do that we already do

Worth noting because some of our choices look correct against this baseline:

1. **AmplitudeToDB + standardize + min-max mel norm** — same as us (and Sydorskyi *picked* this over alternatives)
2. **5s window** — same; he tested longer / "first 7s" / "last 7s" variants and stuck with 5s
3. **Use of secondary labels at value 1.0** — same
4. **timm-only backbones, no Perch/BirdNET KD** — same
5. **GeMFreq pooling** — Sydorskyi's AttHead uses GeMFreq before its attention; our LSEHead uses *mean* freq pool then LSE. We tested GeMFreq variants in exp 019 (v5h/v5l with GeM at backbone level — flat to v5j); the Sydorskyi-style use is *inside* the head, before attention.
6. **Site/hour priors** — Sydorskyi tested these in exp PP and rejected them; we kept them (+0.003 LB). Could be an inference-stage win we're capturing that he isn't.

## Open questions for the next session

1. **Does Sydorskyi's published `bird-clef-2025-pseudo` dataset transfer to BirdCLEF+ 2026?** The 2025 species list is 206; 2026 has 234. Not directly reusable, but the *recipe* is.
2. **Can we use our existing 0.929 ensemble (Perch+SED MLP-only) as the teacher for soft pseudo-labels on 2026 unlabeled soundscapes?** It's stronger than the 8-model 0.917-LB teacher Sydorskyi used at iteration 1.
3. **Does our 4-fold v5j ensemble + TopN-N=1 (Sydorskyi style, applied at ensemble level after averaging) move LB?** Cheapest test, no retraining.
4. **What exactly does our `submission.py` ENABLE_TOPN do?** It's currently set; need to confirm it's the same definition Sydorskyi uses (per-segment vs per-file, per-model vs at ensemble level, N=1 vs other).
5. **Is the 819K-recording pretrain corpus actually accessible via the metadata CSVs Sydorskyi published?** If yes, we could pretrain HGNetV2-B0 (our backbone) ourselves on it. Estimate: 100-300 GPU hours.
6. **Can we test soft-vs-hard pseudo-labels as a single-variable change against our exp 015 ROW recipe?** Soft labels with `trim_min_prob=0.1` could be a clean ablation.

## Files

- Reference doc: `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md`
- Sydorskyi repo (cloned at /tmp/sydorskyi_2025): https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- Paper: https://ceur-ws.org/Vol-4038/paper_256.pdf
- Our v5j config: `user/murilo/experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml`
- Cross-cutting comparison: `docs/references/kaggle/winning-solutions-analysis.md` (older, may want refresh)
