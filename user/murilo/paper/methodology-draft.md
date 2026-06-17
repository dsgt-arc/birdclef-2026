# Methodology draft — "the simplest model that gets the most gains"

> **Working draft for the BirdCLEF 2026 working-note (Murilo's 2–3 page contribution).**
> The paper lives on Overleaf; this markdown is for iteration. LaTeX-ready versions are in
> `methodology.tex` and `floats/`. Prose is intentionally concise — push detail to the appendix.

---

## Source map (every LB number traces to a tracked result)

| Claim | Value | Source |
|---|---|---|
| Final ensemble (Perch probe + SED, MLP-only blend) | **0.929** | exp 023 `results.md`; EXPERIMENTS.md §3.23 |
| SED member standalone (HGNetV2-B0 v5j-4fold-pp2) | 0.917 | exp 023 score breakdown; EXPERIMENTS.md L712 |
| Perch probe member standalone (ONNX, MLP) | 0.920 | exp 023 score breakdown L21 |
| Best single Perch (TF, ProtoSSM v18) | 0.925 | exp 020; EXPERIMENTS.md L709 |
| Base EfficientNet-B0 SED (v1, raw / +PP) | 0.795 / 0.827 | exp 016; EXPERIMENTS.md L723 |
| + dB-mel frontend (v1+PP 0.827 → v4b 0.859) | +0.032 cumulative | exp 017; EXPERIMENTS.md L484–485, L720 |
| ↳ high-res mel (v1 raw 0.795 → v4a 0.825) | +0.030 paired | exp 017; EXPERIMENTS.md L484 |
| ↳ AmplitudeToDB (v4a 0.825 → v4b 0.859) | **+0.034 paired** | exp 017; EXPERIMENTS.md L485, L720 |
| + HGNetV2-B0 + NNAudio dB-mel (v4b 0.859 → v5j 0.906) | +0.047 cumulative | exp 019; L720, L715 |
| ↳ backbone swap (v5f EffNet 0.855 → v5g HGNet 0.892) | **+0.037 paired** | exp 019; EXPERIMENTS.md L538, L718, L721 |
| v5j (HGNet + AmplitudeToDB/NNAudio, n_fft=2048, n_mels=256) | 0.906 | exp 019 `config/hgnet_b0_sed_v5j.yaml`; L715 |
| v5j-4fold | 0.913 (+0.007 vs v5j) | exp 019; EXPERIMENTS.md L713 |
| + site/hour priors (v5j-4fold-pp2) | 0.917 (+0.004) | exp 021 Phase A; EXPERIMENTS.md L587, L712 |
| KD parity (EffNet-B0: 0.856 control → 0.906 +KD) | +0.050 paired | exp 032 `results.md` |
| XC-pretrain parity (EffNetV2-S, single / 4-fold) | 0.907 / 0.916 | exp 038 `results.md` |
| PCEN | 0.880 (−0.026) | exp 022; EXPERIMENTS.md §3.22 |
| CE loss (catastrophic) | 0.494 | exp 027 |
| 2025→2026 warm-start | −0.021 | exp 029 |
| Training-recipe changes (exp 021 phases B–G) | 0-for-14 on LB | exp 021; EXPERIMENTS.md L596 |
| Pseudo-labeling | 0-for-6 (−0.005 to −0.074) | exps 014/015/025/033/037 |

**Deployed v5j config (`hgnet_b0_sed_v5j.yaml`):** HGNetV2-B0 (`hgnetv2_b0.ssld_stage2_ft_in1k`,
ImageNet, embed_dim 1024) · **LSEHead** (gem_p 3, dropout 0.5) · NNAudio mel n_fft=2048,
hop=313, n_mels=256, top_db=80, **AmplitudeToDB** · 5 s windows, 32 kHz · **BCE loss** ·
**spectrogram mixup** (p=1.0, α=1.0, θ=0.8, 5-epoch warmup) · clipwise supervision
(framewise_weight=0) · 20 epochs, one-cycle, lr 5e-4.

> ✅ **RESOLVED (2026-06-17).** Confirmed the deployed 0.917 SED member is HGNetV2-B0
> `v5j-4fold-pp2` (named in `023-.../configs/ensemble_w05.yaml`; MODEL-RUNS.md L502–531), **not**
> the EfficientNetV2 model (exp 038 `a4fold` was explored but never beat the banked 0.929).
> `v5j` uses `SEDFinetuneModule` with **BCE loss + spectrogram mixup** and **no SpecAugment**.
> The paper's "focal loss" + "time/frequency masking" wording was therefore wrong and has been
> corrected in `sections/40_baseline.tex` and `40_fig_ensemble.tex`; the orphaned `lin2017focal`
> bib entry was removed.
>
> ⚠️ The exp-017 high-res-mel gain (+0.030) was measured on **EfficientNet-B0** (n_fft=4096,
> n_mels=224). The deployed **HGNet** v5j reverted to n_fft=2048 / n_mels=256 — v5l (v5j +
> n_fft=4096) scored 0.904, −0.002. So do **not** claim the shipped model uses 4096/224.

---

## §X Supervised Baseline System (expanded)

### Overview

The 2026 task provides roughly one hour of labeled soundscape audio, and the leaderboard rewards
supervised pipelines fit directly to it. Rather than chase architectural novelty, we take an
Occam's-razor view: we ask how much of the leaderboard a *single* spectrogram CNN can recover,
and which few design choices actually move the public score. Our final submission is a
score-level blend of a frozen Perch v2 probe and a trained HGNetV2-B0 sound-event-detection (SED)
network, reaching **0.929** public LB (Figure~\ref{fig:baseline-ensemble},
Table~\ref{tab:baseline-ablation}). The central finding of this section is that a single SED CNN
— ImageNet-pretrained, with a high-resolution dB-scaled mel frontend and fold ensembling —
already reaches **0.917**, i.e. the blend with Perch adds only the final **+0.012**. Almost all
of the gain comes from a handful of frontend and architecture choices; data augmentation tricks,
extra unlabeled data, and alternate training objectives did not transfer (Section~\ref{sec:negatives}).

### The SED model

The SED member is an HGNetV2-B0 backbone (ImageNet-pretrained, 6\,M parameters) over log-mel
spectrograms computed on-GPU: 256 mel bins, 2048-point FFT, 313-sample hop at 32\,kHz over 5\,s
windows, normalized with AmplitudeToDB (per-sample standardization, 80\,dB floor). A
log-sum-exp (LSE) pooling head aggregates frame logits to a clip-level prediction; we train with
binary cross-entropy and spectrogram-space mixup ($\alpha{=}1.0$, 5-epoch warmup) for 20 epochs
under a one-cycle schedule. We fit four folds and average their logits, then apply site- and
hour-conditioned priors as post-processing. The frozen Perch v2 member is described above; the
two are blended with equal weight in probability space.

### Ablation ladder

Table~\ref{tab:ablation-ladder} traces the single-model SED system from the simplest baseline to
the deployed configuration, and then the ensemble step. Two choices dominate. **dB-scale mel
normalization** (AmplitudeToDB) is worth **+0.034** in a paired single-variable test (exp 017,
v4a→v4b), the single largest frontend lever: it makes quiet and loud recordings comparable, which
matters in a domain where focal training clips and soundscape test clips differ sharply in level.
**Backbone choice** is the largest architecture lever: swapping EfficientNet-B0 for HGNetV2-B0
under an otherwise fixed recipe adds **+0.037** (exp 019, v5f→v5g). On top of these, NNAudio mel
plus AmplitudeToDB on HGNet reaches **0.906** single-fold (v5j); four-fold logit averaging adds
**+0.007** (0.913); and site/hour priors — pure post-processing, no retraining — add **+0.004**
(0.917). The final score-level blend with the Perch probe adds **+0.012** to reach **0.929**.
Every step was validated by a leaderboard submission, because offline metrics did not predict LB
rank (Section~\ref{sec:negatives}).

### Why HGNetV2-B0 — a parity result

We adopt the ImageNet HGNetV2-B0 recipe because it is the *simplest* route to its score, not the
only one. Two more elaborate pipelines reach the same place: (i) an ImageNet EfficientNet-B0 SED
distilled from Perch v2 embeddings recovers the identical **0.906** single-fold (a **+0.050**
paired gain over its no-distillation control, exp 032), and (ii) an EfficientNetV2-S pretrained
from scratch on broad Neotropical Xeno-Canto audio reaches **0.907** single-fold and **0.916** at
four folds (exp 038). All three converge to ~0.906/~0.916. Since knowledge distillation requires
a teacher pipeline and custom pretraining requires an extra large-scale training stage, neither
buys accuracy over the plain HGNet recipe; we report them as evidence that the ceiling is set by
the data, not the backbone (Appendix~\ref{app:parity}).

### What does not transfer
\label{sec:negatives}

A recurring and, we think, generalizable finding is that **offline validation AUROC/cmAP
anti-correlate with the leaderboard** across 20+ submissions — the best offline model was
repeatedly the worst on LB (e.g. exp 021, 026) — so we treated every change as a single-variable
experiment arbitrated by a Kaggle submission. Under that discipline, most "improvements" failed:
pseudo-labeling on unlabeled soundscapes regressed in all six attempts (−0.005 to −0.074; exps
014/015/025/033/037), and 14 training-recipe changes in a single sweep (cross-entropy loss,
waveform mixup, attention pooling heads, multi-context windows, manifold mixup) were 0-for-14 on
LB (exp 021); learnable PCEN normalization lost −0.026 (exp 022) and cross-entropy loss was
catastrophic (0.494; exp 027). The levers that worked were few and frontend/architecture-level;
the curated record is in Appendix~\ref{app:curated}.

---

## Appendix material

### Appendix A — parity of three SED recipes (`app:parity`)

| Recipe | Pretraining | Distillation | Single-fold LB | 4-fold LB | Exp |
|---|---|---|---|---|---|
| HGNetV2-B0 SED (deployed) | ImageNet | — | 0.906 | 0.913–0.917 | 019/021 |
| EfficientNet-B0 SED + Perch KD | ImageNet | Perch v2 (MSE) | 0.906 (+0.050 vs control) | 0.915 | 032 |
| EfficientNetV2-S SED | broad XC (2.7k sp.) | — | 0.907 | 0.916 | 038 |

All three independent recipes converge to ~0.906 single-fold / ~0.916 four-fold. Distillation and
custom pretraining match, but do not beat, the plain ImageNet HGNet recipe.

### Appendix B — curated negative results (`app:curated`)

| Change | Best LB | Δ vs baseline | Root cause | Exp |
|---|---|---|---|---|
| Pseudo-labeling (6 variants) | 0.832–0.901 | −0.005 to −0.074 | diffuse soft labels poison training | 014/015/025/033/037 |
| Cross-entropy loss | 0.494 | −0.41 | logits collapse without per-class margin | 027 |
| Waveform mixup | 0.875–0.894 | −0.019 to −0.036 | amplitude shift = train/infer mismatch | 021/025/027/030 |
| Learnable PCEN | 0.880 | −0.026 | empirically inferior to AmplitudeToDB | 022 |
| Attention pooling head | 0.868–0.899 | −0.013 to −0.035 | train/infer window mismatch | 021/026 |
| Multi-context (10–20 s) | 0.877–0.896 | −0.010 to −0.025 | window-length mismatch | 021/026 |
| 2025→2026 warm-start | 0.885 | −0.021 | under-trained random-init head rows | 029 |
| Sydorskyi mixup variant | 0.901 | +0.003 (within noise) | offline gain did not transfer | 030 |
| Training-recipe sweep (phases B–G) | — | 0-for-14 on LB | v5j sits at a local optimum | 021 |

Across 20+ LB-submitted single-variable tests, offline val/AUROC and cmAP were anti-correlated
with the public leaderboard; LB submission was the only reliable arbiter of model quality.

### Figure (already committed)

Figure~\ref{fig:taxa-spectrograms} (`user/murilo/figures/taxa_spectrograms.{pdf,png}`):
representative spectrograms of the three vocalizing taxa on a shared 0–16 kHz axis, showing they
overlap in frequency band and separate by spectro-temporal structure — the motivation for a
learned spectrogram CNN over hand-crafted frequency-band features.
