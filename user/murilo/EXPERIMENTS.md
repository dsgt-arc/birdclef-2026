# Experiments Summary

Consolidated findings from experiments 001–023, covering BirdSet benchmarks, architecture comparisons, tokenization studies, LoRA fine-tuning across three encoder families, Kaggle submission pipeline, soundscape domain adaptation, source separation, pseudo-labeling, SED architecture pivot, Perch probes, and ensemble fusion.

## 1. Quick Reference

| Exp | Name | Dataset | Question | Best Score‡ | Key Finding | Verdict |
|-----|------|---------|----------|-------------|-------------|---------|
| 001 | BirdSet Benchmark | BirdSet (8 datasets)† | Do BirdSet checkpoints reproduce published baselines? | 0.36† | cmAP validated within ±0.02 of paper; audio pretraining baseline set | Confirmed |
| 002 | Supervised ViT Baseline | BirdSet HSN (21 cls) | Can ImageNet-pretrained ViT compete on spectrograms? | 0.186 | 39% of audio-pretrained baseline; training infrastructure validated | Confirmed |
| 003 | Architecture Comparison | BirdSet HSN (21 cls) | How do architecture and pretraining interact? | 0.191 | Pretraining gives 2× lift; audio pretraining 2.5× better than ImageNet | Confirmed |
| 004 | Frame Tokenization | BirdSet HSN (21 cls) | Do vertical frame patches improve ViT from scratch? | 0.118 | +37% over 16×16 baseline; overfitting is the primary bottleneck | Confirmed |
| 005 | EAT Pretrained Fine-Tune | BirdCLEF 2025 (206 cls) | Does AudioSet pretraining transfer to bird classification? | 0.890 | Frozen probe establishes strong first BirdCLEF 2025 baseline | Inconclusive |
| 006 | EAT LoRA Fine-Tune | BirdCLEF 2025 (206 cls) | Can LoRA beat a frozen probe with <1% of parameters? | 0.945 | +6% AUROC over frozen probe; MLP head matters more than LoRA scope | Confirmed |
| 007 | ConvNeXt LoRA Fine-Tune | BirdCLEF 2025 (206 cls) | Does an ImageNet CNN compete with an AudioSet transformer? | 0.941 | Within 0.4% AUROC of EAT; ImageNet features transfer surprisingly well | Confirmed |
| 008 | Frame Tokenization (EAT) | BirdCLEF 2025 (206 cls) | Do frame tokens help a pretrained transformer? | 0.933 | Frame tokens hurt pretrained models; pretrained patches are critical | Rejected |
| 009 | ProtoCLR LoRA Fine-Tune | BirdCLEF 2025 (206 cls) | Does bird-specific pretraining add ensemble value? | 0.942 | 5× smaller model matches EAT/ConvNeXt at AUROC parity | Confirmed |
| 010 | Submission Pipeline | BirdCLEF 2026 | Can ONNX export + Kaggle CPU kernel work end-to-end? | 0.531‡ | Pipeline validated; 3 architectures export and score on Kaggle | Confirmed |
| 011 | LoRA Baseline 2026 | BirdCLEF 2026 (206 cls) | Does the 2025 LoRA pipeline transfer to 2026 data? | 0.736‡ | Transfers without changes; ProtoCLR fulldata best single model | Confirmed |
| 012 | Soundscape Fine-Tuning | BirdCLEF 2026 (234 cls) | Does labeled soundscape data improve LB scores? | 0.767‡ | All 3 models improve (+0.022 to +0.060); ProtoCLR 0.767 best overall | Confirmed |
| 013 | MixIT Source Separation | 10,592 unlabeled soundscapes | Can Bird MixIT separate soundscapes into useful channels? | ~62% meaningful | TF1→ONNX pipeline works; ~62% of files show meaningful separation | Confirmed |
| 014 | Teacher Pseudo-Labeling | 127K unlabeled segments | Can teacher ensemble pseudo-label unlabeled soundscapes? | 0.885‡‡ | ProtoCLR + ConvNeXt ensemble AUROC 0.885 on labeled val; EAT excluded (insect AUROC 0.673) | Confirmed |
| 015 | Student Training | Focal + soundscape + 127K pseudo | Does pseudo-labeled data improve student models? | 0.762‡ | Regressed on LB (ProtoCLR −0.005, EAT −0.024); clip-level architecture is the bottleneck | Rejected |
| 016 | SED Architecture | BirdCLEF 2026 (234 cls) | Does EfficientNet-B0 SED outperform clip-level models? | 0.827‡ | SED validates architecture pivot; PP (TTA+smoothing+TopN) adds +0.028 | Confirmed |
| 017 | Mel Overhaul | BirdCLEF 2026 (234 cls) | Do high-res mel + AmplitudeToDB improve SED? | 0.859‡ | v4b AmplitudeToDB +0.034 over v4a; best single SED at the time | Confirmed |
| 018 | Perch Probes | BirdCLEF 2026 (234 cls) | Can frozen Perch v2 + probes beat fine-tuned models? | 0.918‡ | Raw 0.825 → probes 0.893 → OOF 0.911 → ProtoSSM v4 0.918 | Confirmed |
| 019 | SED Ablations | BirdCLEF 2026 (234 cls) | Which single variables improve SED most? | 0.913‡ | HGNetV2-B0 +0.037; v5j (AmplitudeToDB+NNAudio) 0.906; 4-fold 0.913 | Confirmed |
| 020 | Perch ONNX + ProtoSSM | BirdCLEF 2026 (234 cls) | Can ONNX Perch match TF Perch on LB? | 0.925‡ | ONNX 9× faster but −0.005-0.010 penalty; V18+TF 0.925 best Perch | Confirmed |
| 021 | SED Training Phases | BirdCLEF 2026 (234 cls) | Can training recipe changes improve SED beyond v5j? | 0.917‡ | Site/hour priors +0.003/+0.004; all training recipe changes 0-for-14 on LB | Confirmed |
| 022 | PCEN Normalization | BirdCLEF 2026 (234 cls) | Does learnable PCEN improve domain invariance? | 0.880‡ | PCEN 0.880 (−0.026 vs v5j); v5j-repro 0.903 confirms pipeline correct | Rejected |
| 023 | Perch + SED Ensemble | BirdCLEF 2026 (234 cls) | Does score-level fusion of Perch + SED improve LB? | 0.929‡ | MLP-only (proto_weight=0.0) 0.929 new best; config-driven YAML push system | Confirmed |

† Exp 001 evaluated all 8 BirdSet datasets (HSN, PER, NES, UHH, POW, SSW, SNE, NBP); best cmAP shown is the mean across all 8 (HSN alone: 0.480 ConvNeXT, 0.380 EfficientNetB1).
‡ BirdSet experiments (001–004) report cmAP; BirdCLEF experiments (005–023) report ROC-AUC. Experiments 005–009 report offline test-split AUROC; experiments 010–023 report Kaggle public LB macro-AUROC.
‡‡ Exp 014 reports ensemble AUROC on labeled validation, not LB.

### Data Splits

**BirdSet experiments (001–004)** use the BirdSet-provided train/test splits directly — no custom splitting is performed.

**BirdCLEF 2025 experiments (005–009)** use only the `train_audio/` focal recordings (28,564 files from `train.csv`). Since BirdCLEF 2025 provides no labeled validation or test set (the Kaggle `test_soundscapes/` is hidden), we create our own **80/10/10 stratified train/val/test split** of the focal recordings:

- **Stratification**: by `primary_label`, so every species appears in every split.
- **Seed**: `seed=42` via `np.random.RandomState(42)` — the split is fully deterministic and reproducible.
- **Edge cases**: species with <3 samples are duplicated across splits to ensure presence in all three.
- **Result**: ~22,973 train / ~2,803 val / ~2,803 test recordings.

The `train_soundscapes/` (9,726 unlabeled soundscape files) and `test_soundscapes/` (hidden, contains only a readme) are **not used** in any experiment. The "test" set in experiment results refers to the held-out 10% of focal recordings, not the Kaggle test soundscapes.

Implementation: `src/birdclef/data/dataset.py` → `BirdCLEFDataModule._split_dataframe()`.

**BirdCLEF 2026 experiments (010–012)** use the BirdCLEF 2026 dataset (35,549 focal recordings from 206 species, plus 66 labeled soundscape files covering 234 species including 28 soundscape-only insect/amphibian sonotypes). The same **80/10/10 stratified split** is applied to focal recordings; soundscape files are split **by file** (not by segment) to prevent data leakage (~50 train / ~8 val / ~8 test files).

BirdCLEF 2026 also provides 10,592 **unlabeled** soundscape files (used in exp 013, not in 010–012). The Kaggle evaluation is on a hidden test set of ~600 soundscape files; the public LB uses ~34% of this data.

Implementation: Same `BirdCLEFDataModule` as 2025 (renamed from `BirdCLEF2025DataModule`); soundscape data via `MixedDataModule` in `src/birdclef/data/mixed.py` (exp 012).

## 2. Research Progression

The first three experiments established foundations. Experiment 001 validated the BirdSet benchmark pipeline by reproducing published ConvNeXT and EfficientNetB1 baselines within ±0.02 cmAP across all 8 evaluation datasets, setting the audio-pretrained reference at 0.480 cmAP on HSN. Experiments 002 and 003 then tested whether ImageNet-pretrained vision models could close this gap: ConvNeXT-Tiny and ViT-Small with ImageNet weights reached only 0.17–0.19 cmAP, while training from scratch yielded 0.087–0.098. The clear takeaway was that pretraining is essential and audio-domain pretraining is far more valuable than ImageNet pretraining for bird sound classification.

Two tokenization experiments explored whether better patch design could compensate for weaker representations. Experiment 004 showed that vertical frame patches (128×4, spanning the full frequency axis) improve ViT from scratch by +37% over standard 16×16 patches on HSN, confirming that spectrogram-aware tokenization provides a better inductive bias. Experiment 008 then tested the same idea on EAT with AudioSet-pretrained transformer blocks — and found the opposite: frame tokens degraded performance by 43% relative to pretrained 16×16 patches. The conclusion is that tokenization shape matters when training from scratch, but pretrained representations are so strongly coupled to their original patch structure that changing it destroys more value than the better inductive bias recovers.

The LoRA fine-tuning arc (experiments 005–009) produced the most actionable results. Starting from a frozen EAT linear probe baseline of AUROC=0.890, LoRA fine-tuning of less than 1% of parameters improved all three encoder families: EAT reached AUROC=0.945, ConvNeXt-Base 0.941, and ProtoCLR 0.942. The convergence of three architecturally diverse models — a self-supervised audio transformer (90M params), an ImageNet-pretrained CNN (89M params), and a compact bird-specific contrastive model (19M params) — to within 0.004 AUROC of each other suggests a performance ceiling at the current data scale. Their architectural and pretraining diversity makes them natural candidates for ensemble-based pseudo-labeling.

Experiments 010–012 moved from offline evaluation to real Kaggle leaderboard scores. Experiment 010 built the end-to-end submission pipeline: ONNX export with merged LoRA weights, species mapping, and a unified Kaggle CPU inference kernel supporting all three architectures. The first submission — ProtoCLR trained on BirdCLEF 2025 — scored 0.531 LB with only 41/234 species overlapping, validating the pipeline. Experiment 011 retrained the same LoRA configurations on BirdCLEF 2026 focal data (35,549 recordings, 206 species) with no hyperparameter changes, confirming that the pipeline generalizes across years. ProtoCLR fulldata reached 0.736 LB, the best focal-only score. EAT and ConvNeXt fulldata regressed (0.698 and 0.695 respectively), revealing that fixed-epoch training without early stopping causes overfitting — the faster a model converges, the more it overshoots.

Experiment 012 addressed the focal-to-soundscape domain gap by fine-tuning from exp 011 checkpoints on mixed focal + labeled soundscape data, expanding the classification head from 206 to 234 species. All three models improved on the leaderboard: ProtoCLR reached **0.767** (+0.031), ConvNeXt **0.755** (+0.060), and EAT **0.720** (+0.022). ConvNeXt showed the largest gain, recovering from its fulldata overfitting and surpassing its original 80/10/10 score. The consistent improvement across all architectures confirms that soundscape domain adaptation and species expansion are both valuable, and positions these models as stronger teachers for pseudo-labeling in exp 013.

Experiments 013–015 explored the pseudo-labeling pipeline: MixIT source separation (exp 013) converted Bird MixIT from TF1 to ONNX and separated 10,592 unlabeled soundscapes into 4 channels with ~62% showing meaningful separation. Experiment 014 used a ProtoCLR + ConvNeXt teacher ensemble (AUROC 0.885 on labeled validation) to pseudo-label 127K unlabeled segments, excluding EAT due to poor insect AUROC (0.673). However, experiment 015 trained students on focal + soundscape + pseudo-labels and regressed on LB (ProtoCLR −0.005, EAT −0.024), revealing that clip-level architecture — not data quantity — was the bottleneck. This dead end motivated the architecture pivot to SED.

Experiments 016–017 and 019 built the SED pipeline from scratch. Experiment 016 validated the EfficientNet-B0 SED with GeM + AttHead, reaching 0.827 LB with post-processing (TTA + smoothing + TopN). Experiment 017 demonstrated that mel frontend improvements (n_fft=4096, n_mels=224, AmplitudeToDB) had outsized impact: v4b scored 0.859, a +0.034 improvement from AmplitudeToDB alone. Experiment 019 ran v5a–v5n single-variable ablations, finding HGNetV2-B0 backbone (+0.037 over EfficientNet), AmplitudeToDB + NNAudio (v5j, 0.906 LB), and 4-fold ensembling (v5j-4fold, 0.913). Critically, offline val/AUROC proved anti-correlated with LB performance — every model ranking had to be validated by submission. Experiment 021's 7-phase training recipe search (CE loss, waveform MixUp, AttHead, manifold mixup, multi-context) was 0-for-14 on LB, confirming that the SED backbone and mel frontend drive performance while training tricks do not transfer. Experiment 022 tested learnable PCEN for domain invariance and regressed to 0.880 (−0.026 vs v5j), closing that avenue.

In parallel, experiments 018 and 020 developed the Perch probe pipeline. Frozen Perch v2 embeddings with LogReg/MLP probes progressed from 0.825 (raw) to 0.893 (probes) to 0.918 (ProtoSSM v4), establishing a separate strong submission track. Experiment 020 showed ONNX Perch runs 9× faster but carries a −0.005–0.010 LB penalty versus TF; the two-kernel V18+TF submission reached 0.925 LB, the best Perch result. Finally, experiment 023 fused Perch + SED via score-level ensembling in a single Kaggle kernel, discovering that MLP-only weights (proto_weight=0.0) achieved **0.929 LB** — the new best overall — with a config-driven YAML push system for rapid iteration.

## 3. Detailed Findings

### 3.1 Exp 001: BirdSet Bioacoustics Benchmark

**Hypothesis**: BirdSet-native pretrained models will reproduce published baselines via direct inference, establishing reference performance across diverse acoustic environments.

**Dataset**: BirdSet — all 8 evaluation datasets (HSN, PER, NES, UHH, POW, SSW, SNE, NBP), 321,819 test samples per model.

**Models**: BirdSet ConvNeXT, BirdSet EfficientNetB1 (both pretrained on BirdSet XCL).

| Model | HSN | PER | NES | UHH | POW | SSW | SNE | NBP | Mean |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|------|
| ConvNeXT | 0.48 | 0.17 | 0.34 | 0.25 | 0.34 | 0.34 | 0.30 | 0.63 | 0.36 |
| EfficientNetB1 | 0.38 | 0.18 | 0.32 | 0.22 | 0.35 | 0.37 | 0.34 | 0.62 | 0.35 |

- cmAP and AUROC show excellent alignment with published BirdSet baselines (mean difference: 0.00 for both models). Top-1 accuracy showed systematic discrepancies (−0.18 to −0.20) likely due to multi-label handling differences.
- Performance varies 3.4× between best (NBP, 0.63) and worst (PER, 0.17), driven by species diversity and acoustic environment complexity.
- Phase 2B (BirdNET/Perch evaluation via linear probing) was deferred; the experiment achieved its core goal of validating checkpoints and establishing baseline performance.

**Limitation**: Only BirdSet-native models evaluated; external model comparison (BirdNET, Perch) remains future work.

Full results: [RESULTS_SUMMARY.md](experiments/001-birdset-bioacoustics-benchmark/RESULTS_SUMMARY.md)

---

### 3.2 Exp 002: Supervised ViT Baseline

**Hypothesis**: ViT-Base pretrained on ImageNet-21k can achieve competitive cmAP (>0.30) on HSN when trained end-to-end with on-the-fly mel spectrograms.

**Dataset**: BirdSet HSN — 66,504 training chunks (streaming pipeline), 12,000 test samples, 21 classes.

**Models**: ViT-Base-Patch16 (85.5M params, ImageNet-21k pretrained).

| Model | cmAP | AUROC | F1 Macro | Top-1 | % of Baseline |
|-------|------|-------|----------|-------|---------------|
| ViT-Base (ImageNet-21k) | 0.186 | 0.662 | 0.092 | 0.100 | 39% |
| ConvNeXT (exp 001 baseline) | 0.480 | 0.883 | 0.139 | 0.251 | 100% |

- ViT-Base reaches only 39% of the audio-pretrained ConvNeXT baseline despite having 85.5M parameters, confirming that ImageNet pretraining is insufficient for competitive bird sound classification.
- The streaming pipeline (non-overlapping 5s chunks from full recordings) improved cmAP by +13% over center-crop (0.186 vs 0.164), validating the data pipeline design used in subsequent experiments.
- Early stopping triggered at epoch 9 of 50, indicating rapid overfitting on the small HSN dataset.

**Limitation**: Single dataset (HSN, 21 classes); ViT-Base's 85.5M parameters make fair comparison with smaller models difficult.

Full results: [results.md](experiments/002-supervised-vit-baseline/results.md)

---

### 3.3 Exp 003: Architecture Comparison (ConvNeXT-Tiny & ViT-Small)

**Hypothesis**: ImageNet-pretrained variants will substantially outperform scratch-trained variants for both architectures, quantifying the pretraining advantage at matched model sizes.

**Dataset**: BirdSet HSN — 66,504 training chunks (streaming pipeline), 12,000 test samples, 21 classes.

**Models**: ConvNeXT-Tiny (27.8M) and ViT-Small (21.5M), each trained from scratch and with ImageNet pretraining.

| Model | Params | Pretraining | cmAP | AUROC | Pretrain Lift |
|-------|--------|-------------|------|-------|---------------|
| ConvNeXT-Tiny | 27.8M | ImageNet | 0.191 | 0.638 | 2.2× |
| ViT-Small | 21.5M | ImageNet | 0.174 | 0.707 | 1.8× |
| ViT-Small | 21.5M | None | 0.098 | 0.594 | — |
| ConvNeXT-Tiny | 27.8M | None | 0.087 | 0.607 | — |

- Pretraining provides a consistent 1.8–2.2× cmAP advantage for both architectures. ConvNeXT-Tiny pretrained (0.191) slightly leads ViT-Small pretrained (0.174) with 30% more parameters.
- All models remain far below exp 001's audio-pretrained baseline (0.480), confirming that audio-domain pretraining is 2.5× more valuable than ImageNet pretraining on this task.
- The streaming pipeline improved all 4 configs, with ViTs benefiting more (+34–44%) than ConvNeXTs (+6–45%), suggesting ViTs are more data-hungry.

**Limitation**: Only HSN dataset tested; single seed per configuration; hyperparameters not tuned per-architecture.

Full results: [results.md](experiments/003-model-training/results.md)

---

### 3.4 Exp 004: Frame Tokenization

**Hypothesis**: Replacing ViT's 16×16 square patches with vertical frame patches (128×W, spanning the full frequency axis) will improve cmAP on HSN when training from scratch.

**Dataset**: BirdSet HSN — 66,504 training chunks (streaming pipeline), 12,000 test samples, 21 classes.

**Models**: ViT-Small (21.5M) trained from scratch with four patch configurations.

| Config | Patch Shape | Tokens | cmAP | cmAP Lift |
|--------|-------------|--------|------|-----------|
| 128×4 (full-band wide) | 128×4 | 125 | 0.118 | +37% |
| 128×2 (full-band narrow) | 128×2 | 250 | 0.115 | +34% |
| 128×1 (full-band single) | 128×1 | 500 | 0.099 | +15% |
| 16×16 baseline | 16×16 | 248 | 0.086 | — |

- All three vertical frame configurations outperform the 16×16 baseline, with 128×4 providing the best tradeoff between temporal resolution (40ms) and sequence length (125 tokens).
- Severe overfitting dominates all configs (train cmAP ~0.94 vs test ~0.12), indicating that regularization and data augmentation are more impactful than tokenization at this data scale.
- Shorter sequences perform better: 128×4 (125 tokens) slightly beats 128×2 (250 tokens), while 128×1 (500 tokens) underperforms due to O(n²) attention cost exceeding the model's effective capacity.

**Limitation**: Scratch training only; hyperparameters tuned for 16×16 patches; single seed per configuration.

Full results: [results.md](experiments/004-frame-tokenization/results.md)

---

### 3.5 Exp 005: EAT Pretrained Fine-Tune

**Hypothesis**: Fine-tuning EAT (pretrained on AudioSet-2M) on BirdCLEF 2025 will produce a model that generalizes to held-out data, and full fine-tuning will outperform a frozen linear probe.

**Dataset**: BirdCLEF 2025 — 22,973 train / 2,803 val / 2,803 test recordings (80/10/10 stratified split), 206 species.

**Models**: EAT-base (90M params, AudioSet self-supervised pretraining).

| Variant | Trainable Params | Test AUROC | Test cmAP | Val cmAP | Training Time |
|---------|-----------------|------------|-----------|----------|---------------|
| Frozen linear probe | 158K | **0.890** | 0.301 | 0.238 | ~9h |
| Full fine-tune (staged) | 90.5M | pending | pending | 0.291 | >12h (timed out) |

- The frozen linear probe establishes a strong first baseline on BirdCLEF 2025 (test AUROC=0.890, cmAP=0.301), validating that AudioSet self-supervised features contain species-discriminative information without any encoder adaptation.
- Test cmAP (0.301) exceeds val cmAP (0.238) by 26%, indicating no overfitting in the frozen regime.
- Full fine-tuning showed promising val cmAP (0.291, surpassing frozen val) but timed out before test evaluation, leaving the frozen-vs-finetune comparison incomplete.

**Limitation**: Fine-tune run exceeded SLURM 12h limit; only focal recordings evaluated (no soundscape domain gap assessment).

Full results: [results.md](experiments/005-pretrained-finetune/results.md)

---

### 3.6 Exp 006: EAT LoRA Fine-Tune

**Hypothesis**: LoRA fine-tuning of EAT-base (~0.67% of parameters trainable) will outperform the frozen linear probe baseline (AUROC=0.890) while fitting within single-GPU time and memory constraints.

**Dataset**: BirdCLEF 2025 — same 80/10/10 split as exp 005, 206 species.

**Models**: EAT-base with PEFT and adapters LoRA backends, varying LoRA targets and head architecture.

| Config | Backend | LoRA Targets | Head | AUROC | cmAP |
|--------|---------|-------------|------|-------|------|
| attn + MLP head | adapters | qkv, proj | MLP 512 | **0.945** | 0.536 |
| attn + MLP head | PEFT | qkv, proj | MLP 512 | 0.942 | 0.511 |
| attn | adapters | qkv, proj | Linear | 0.938 | 0.485 |
| attn | PEFT | qkv, proj | Linear | 0.927 | 0.469 |
| all layers | PEFT | qkv, proj, fc1, fc2 | Linear | 0.923 | 0.482 |

- LoRA fine-tuning dramatically outperforms frozen probing: the best config (adapters, attn LoRA, MLP head) reaches AUROC=0.945, a +6.2% improvement over exp 005's 0.890 baseline (cmAP improves +78%, from 0.301 to 0.536).
- The MLP classification head (768→512→206) matters more than LoRA scope: adding an MLP head gives +0.004–0.015 AUROC, while expanding LoRA from attention to all layers can hurt AUROC.
- The adapters library slightly outperforms PEFT on equivalent configs, likely due to finer-grained handling of fused QKV layers.

**Limitation**: Single seed (seed=42); one adapters config timed out (adapters-eat-lora-all, >12h).

Full results: [results.md](experiments/006-lora-finetune/results.md)

---

### 3.7 Exp 007: ConvNeXt-Base LoRA Fine-Tune

**Hypothesis**: ConvNeXt-Base with ImageNet pretraining and LoRA will produce comparable or lower AUROC than EAT (0.945), since ImageNet captures visual features while AudioSet captures audio-specific features.

**Dataset**: BirdCLEF 2025 — same split as exps 005–006, 206 species.

**Models**: ConvNeXt-Base (~89M params, ImageNet-22K+1K pretrained) with PEFT LoRA on MLP layers.

| Config | Head | LoRA Params | AUROC | cmAP | F1 Macro |
|--------|------|-------------|-------|------|----------|
| MLP LoRA + MLP head | MLP 512 | ~1.4M | **0.941** | 0.525 | 0.455 |
| MLP LoRA + Linear head | Linear | ~1.4M | 0.939 | 0.527 | 0.467 |

- ConvNeXt-Base (AUROC=0.941) falls only 0.4% below EAT's best (0.945), far closer than expected. ImageNet-pretrained CNNs learn generic hierarchical features (edges, textures, frequency patterns) that transfer well to mel spectrograms.
- The MLP head gives a slight AUROC edge (0.941 vs 0.939) but a slight cmAP deficit (0.525 vs 0.527). Since LoRA already targets the MLP layers (fc1, fc2), the head architecture has minimal impact.
- ConvNeXt converges faster (best epoch 12–15, stopped 23–26) than EAT (best epoch 27–36), suggesting quicker LoRA adaptation but less room for continued improvement.

**Limitation**: Only PEFT backend tested (no adapters comparison); LoRA can only target MLP layers in ConvNeXt (no attention to adapt).

Full results: [results.md](experiments/007-convnext-finetune/results.md)

---

### 3.8 Exp 008: Frame Tokenization for EAT

**Hypothesis**: Replacing EAT's 16×16 patch embedding with full-frequency frame tokens (2×128 or 4×128) will produce comparable or higher performance despite requiring a randomly initialized patch embedding, because full-band tokens better capture harmonic structure.

**Dataset**: BirdCLEF 2025 — same split, 206 species.

**Models**: EAT-base with PEFT LoRA (attn, r=8), four patch configurations.

| Config | Patch Embed | Patch Size | Tokens | AUROC | cmAP | Δ AUROC vs Baseline |
|--------|-------------|------------|--------|-------|------|---------------------|
| A: baseline | Pretrained (frozen) | 16×16 | 256 | **0.933** | 0.494 | — |
| B: control | Random (trained) | 16×16 | 256 | 0.899 | 0.360 | −3.6% |
| C1b: frame | Random (trained) | 4×128 | 128 | 0.873 | 0.284 | −6.4% |
| C1a: frame | Random (trained) | 2×128 | 256 | 0.864 | 0.260 | −7.4% |

- Pretrained 16×16 patches dominate: the frozen pretrained patch embedding (config A) outperforms a fully trained random one (config B) by +3.8% AUROC (0.933 vs 0.899), confirming the patch embedding is a learned feature extractor, not just a projection.
- Frame tokenization actively hurts with pretrained transformers: both frame configs score below even the random square control in AUROC, the opposite of exp 004's improvement from scratch.
- The pretrained transformer's attention patterns, position embeddings, and layer norms are calibrated for a 32×8 grid of 16×16 patches. Frame tokens produce a fundamentally incompatible 1×N grid that LoRA alone cannot bridge.

**Limitation**: Phase 2 (higher LoRA rank, partial unfreeze) was cancelled given the magnitude of the deficit; PEFT only (no adapters).

Full results: [results.md](experiments/008-frame-tokenization-eat/results.md)

---

### 3.9 Exp 009: ProtoCLR (CvT-13) LoRA Fine-Tune

**Hypothesis**: ProtoCLR's domain-invariant representations (prototypical contrastive learning on Xeno-Canto) will provide complementary features to EAT and ConvNeXt for BirdCLEF teacher model ensembles.

**Dataset**: BirdCLEF 2025 — same split, 206 species.

**Models**: ProtoCLR CvT-13 (19M params, Xeno-Canto pretrained) with PEFT LoRA.

| Config | FT Mode | Trainable Params | AUROC | cmAP | F1 Macro | Walltime |
|--------|---------|-----------------|-------|------|----------|----------|
| Attention only | LoRA (attn) | ~148K | **0.942** | 0.531 | 0.463 | 2h 41m |
| All layers | LoRA (attn + MLP) | ~247K | 0.939 | 0.537 | 0.504 | 3h 24m |
| Full fine-tune | Full (all params) | 19M | 0.925 | 0.525 | 0.502 | ~4h 30m |

- ProtoCLR matches EAT's best AUROC (0.942 vs 0.945) and ConvNeXt (0.941) despite being 5× smaller (19M vs ~90M params). Domain-specific bird audio pretraining on Xeno-Canto is remarkably effective.
- Adding MLP LoRA targets slightly lowers AUROC (0.939 vs 0.942) but gives a substantial F1 macro improvement (+8.8%, from 0.463 to 0.504), suggesting MLP adaptation helps with underrepresented classes at some cost to ranking quality.
- Full fine-tuning (all 19M params, lr=1e-4) underperformed LoRA by -1.7% AUROC (0.925 vs 0.942). The model ran all 50 epochs without early stopping, indicating catastrophic forgetting of the pretrained Xeno-Canto representations. LoRA's low-rank constraint preserves these task-aligned features while allowing targeted adaptation.
- Training completes in 2.7–3.4 hours — 3–4× faster than EAT and 2× faster than ConvNeXt — thanks to the compact architecture, though the CPU mel transform is the throughput bottleneck.

**Limitation**: Only PEFT backend tested; no MLP classification head tested (exp 006 showed heads help EAT); single seed.

Full results: [results.md](experiments/009-protoclr-finetune/results.md)

---

### 3.10 Exp 010: Submission Pipeline

**Hypothesis**: LoRA fine-tuned models (EAT, ConvNeXt, ProtoCLR) can be exported to ONNX and scored on the Kaggle leaderboard via a unified CPU inference kernel within the 90-minute time limit.

**Pipeline**: Checkpoint → merge LoRA → ONNX export (opset 17) → species mapping → Kaggle kernel with model-specific mel preprocessing → 5s window inference → probability mapping to 234 submission columns.

**Models**: 3 architectures × 3 training variants (2025-trained, 2026 80/10/10, 2026 fulldata) + 3 soundscape models (exp 012).

| Model | Training Data | Mapped Species | Public LB | Date |
|-------|--------------|---------------:|----------:|------|
| ProtoCLR LoRA attn | BirdCLEF 2025 (exp 009) | 41/234 | **0.531** | 2026-03-15 |
| ProtoCLR LoRA attn | BirdCLEF 2026 80/10/10 (exp 011) | 206/234 | **0.721** | 2026-03-15 |
| ConvNeXt LoRA MLP | BirdCLEF 2026 80/10/10 (exp 011) | 206/234 | **0.720** | 2026-03-15 |
| EAT LoRA attn | BirdCLEF 2026 80/10/10 (exp 011) | 206/234 | **0.701** | 2026-03-15 |
| ProtoCLR LoRA attn | BirdCLEF 2026 fulldata (exp 011) | 206/234 | **0.736** | 2026-03-17 |
| EAT LoRA attn | BirdCLEF 2026 fulldata (exp 011) | 206/234 | **0.698** | 2026-03-17 |
| ConvNeXt LoRA MLP | BirdCLEF 2026 fulldata (exp 011) | 206/234 | **0.695** | 2026-03-21 |
| ProtoCLR LoRA attn | Focal + soundscape (exp 012) | 234/234 | **0.767** | 2026-03-21 |
| ConvNeXt LoRA MLP | Focal + soundscape (exp 012) | 234/234 | **0.755** | 2026-03-22 |
| EAT LoRA attn | Focal + soundscape (exp 012) | 234/234 | **0.720** | 2026-03-22 |

- ONNX export works reliably for all 3 architectures. ProtoCLR required einops-compatible tracing but succeeded with opset 17. EAT and ConvNeXt export cleanly.
- CPU timing is well within the 90-minute Kaggle budget for all single-model submissions.
- Each model's mel preprocessing is encoded in `mel_config.json` (kaldi_fbank for EAT, log-mel for ConvNeXt, protoclr_mel for ProtoCLR), auto-detected by the unified kernel.
- Species mapping handles both 2025→2026 (41/234 overlap) and 2026→2026 (206/234 or 234/234) mappings. Unmapped species default to probability 0.0.

**Limitation**: Ensemble submission is infeasible — ProtoCLR + ConvNeXt alone would need ~120 minutes for ~600 soundscapes. Single-model submissions only.

Full results: [results.md](experiments/010-submission-pipeline/results.md)

---

### 3.11 Exp 011: LoRA Baseline on BirdCLEF 2026

**Hypothesis**: The LoRA fine-tuning pipeline from BirdCLEF 2025 (exps 006/007/009) will produce competitive baselines on BirdCLEF 2026 focal data without hyperparameter changes.

**Dataset**: BirdCLEF 2026 — 35,549 focal recordings, 206 species, 80/10/10 stratified split.

**Models**: EAT-base, ConvNeXt-Base, ProtoCLR/CvT-13 — all with LoRA rank=8, alpha=16, lr=3e-4, effective batch=256.

| Run | Encoder | Test AUROC | Test cmAP | F1 macro | Best Epoch | Stopped | LB (80/10/10) | LB (fulldata) |
|-----|---------|-----------|----------|----------|------------|---------|---------------|---------------|
| `convnext-lora-mlp-mlphead` | ConvNeXt-Base | **0.938** | 0.522 | 0.456 | 9 | 20 | **0.720** | 0.695 |
| `protoclr-lora-attn` | ProtoCLR/CvT-13 | 0.937 | **0.531** | **0.492** | 21 | 32 | 0.721 | **0.736** |
| `eat-lora-attn-mlphead` | EAT-base | 0.929 | 0.509 | 0.472 | 17 | 28 | 0.701 | 0.698 |

- The 2025 LoRA pipeline transfers to 2026 with zero hyperparameter changes. A unified `train.py` handles all 3 encoders.
- Offline AUROC (0.929–0.938) is slightly below 2025 (0.941–0.945), likely due to different species composition rather than pipeline issues.
- ProtoCLR is the best value model: highest cmAP (0.531) and F1 (0.492) despite being 5× smaller (~20M vs ~90M params). Its Xeno-Canto pretraining generalizes well to the new species set.
- **Fulldata overfitting is model-dependent**: ProtoCLR benefited from fulldata (+0.015 LB) while EAT (−0.003) and ConvNeXt (−0.025) regressed. Fixed-epoch training overshoots the optimal checkpoint — the faster a model converges, the worse it overfits. Future fulldata runs should cap epochs at ~1.5× the best epoch from the 80/10/10 run.

**Limitation**: Single seed (42); no hyperparameter tuning on 2026 data; fulldata runs used fixed 50 epochs without early stopping.

Full results: [results.md](experiments/011-lora-baseline-2026/results.md)

---

### 3.12 Exp 012: Labeled Soundscape Fine-Tuning

**Hypothesis**: Fine-tuning exp 011 models on labeled soundscape data will expand species coverage from 206 → 234 and improve LB scores by adapting to the soundscape domain.

**Dataset**: BirdCLEF 2026 — 35,549 focal recordings + 739 labeled soundscape segments (66 files), 234 species (206 focal + 28 soundscape-only insect/amphibian sonotypes). Soundscape files split by file (~50 train / ~8 val / ~8 test).

**Models**: EAT, ConvNeXt, ProtoCLR — fine-tuned from exp 011 best checkpoint (80/10/10) with head expanded 206 → 234, LoRA rank=8, lr=1e-4 (lower than exp 011's 3e-4), early stopping patience=8.

| Run | Encoder | Test AUROC | Test cmAP | Best Epoch | Stopped | Public LB | vs Exp 011 |
|-----|---------|-----------|----------|------------|---------|----------|-----------|
| `protoclr-soundscape` | ProtoCLR/CvT-13 | **0.845** | **0.481** | 11 | 13 | **0.767** | +0.031 |
| `convnext-soundscape` | ConvNeXt-Base | 0.844 | 0.470 | 2 | 11 | **0.755** | +0.060 |
| `eat-soundscape` | EAT-base | 0.838 | 0.452 | 3 | 12 | **0.720** | +0.022 |

- All 3 models improved on the Kaggle LB. Soundscape fine-tuning is universally beneficial — species expansion (206→234) and domain adaptation both contribute.
- ConvNeXt had the largest LB gain (+0.060), recovering from its fulldata overfitting in exp 011 and surpassing its original 80/10/10 score (0.720 → 0.755). This suggests ConvNeXt had the biggest focal→soundscape domain gap.
- ProtoCLR remains the best single model (**0.767 LB**), consistent with its strong bird-specific pretraining.
- Models converge much faster than exp 011 (best epoch 2–11 vs 9–21) due to warm-starting at lower LR.
- Offline test AUROC (234 classes) is not comparable to exp 011 (206 classes) — the 28 barely-trained soundscape-only species lower the aggregate metric.
- Offline test AUROC does not predict LB ranking: ConvNeXt (0.844) and ProtoCLR (0.845) are nearly tied offline, but ProtoCLR leads by 0.012 on LB.

**Limitation**: Only 739 soundscape segments (vs ~200K focal chunks); 28 soundscape-only species have minimal training signal; single seed; no hyperparameter tuning.

Full results: [results.md](experiments/012-soundscape-finetune/results.md)

---

### 3.13 Exp 013: MixIT Source Separation

**Hypothesis**: Bird MixIT (converted TF1 → ONNX) can separate unlabeled soundscapes into individual source channels, improving downstream pseudo-label quality.

**Dataset**: 10,592 unlabeled BirdCLEF 2026 soundscape files → 4-channel separation per file.

**Models**: Bird MixIT (TF1 → ONNX conversion).

- Converted TF1 SavedModel to ONNX via tf2onnx; pipeline processes all 10,592 files.
- ~62% of files showed meaningful separation (distinct source content across channels).
- Remaining ~38% produced redundant or near-silent channels — separation quality depends on source diversity in the mixture.
- Pseudo-labeling of separated channels deferred to exp 014.

**Limitation**: Non-bird taxa (frogs, insects) may not separate well since Bird MixIT was trained on bird audio. Quality validation was visual/manual, not quantitative.

Full results: [results.md](experiments/013-mixit-source-separation/results.md)

---

### 3.14 Exp 014: Teacher Pseudo-Labeling

**Hypothesis**: A ProtoCLR + ConvNeXt teacher ensemble can produce high-quality pseudo-labels for 127K unlabeled soundscape segments.

**Dataset**: 127K 5s segments from unlabeled soundscapes (including MixIT-separated channels from exp 013).

**Models**: ProtoCLR (exp 012) + ConvNeXt (exp 012) teacher ensemble.

| Teacher | Labeled Val AUROC | Role |
|---------|------------------|------|
| ProtoCLR + ConvNeXt ensemble | **0.885** | Pseudo-labeler |
| EAT | 0.673 (insects) | Excluded |

- Ensemble of ProtoCLR + ConvNeXt achieves 0.885 AUROC on labeled validation, sufficient for pseudo-labeling.
- EAT excluded due to poor insect/amphibian AUROC (0.673), which would inject noisy labels for the 28 soundscape-only sonotypes.
- Produced pseudo-labels for all 127K segments with per-species confidence scores.

**Limitation**: Pseudo-label quality bounded by teacher accuracy; no iterative refinement or self-training loop.

Full results: [results.md](experiments/014-teacher-pseudo-labels/results.md)

---

### 3.15 Exp 015: Student Training

**Hypothesis**: Training on focal + soundscape + 127K pseudo-labeled segments (w=0.5) will improve LB scores over exp 012 teachers.

**Dataset**: BirdCLEF 2026 focal + labeled soundscape + 127K pseudo-labeled segments, 234 species.

**Models**: ProtoCLR and EAT, fine-tuned with pseudo-label loss weight=0.5.

| Model | Public LB | vs Exp 012 | Best Epoch |
|-------|----------|-----------|------------|
| ProtoCLR student | 0.762 | −0.005 | 1–5 |
| EAT student | 0.696 | −0.024 | 1–5 |

- **REJECTED**: Both models regressed on LB. Pseudo-labels did not help.
- Student converges to teacher quality within 1–5 epochs, then overfits to noise in pseudo-labels.
- The bottleneck is clip-level architecture (5s window, single label), not data quantity. This motivated the SED architecture pivot.

**Limitation**: Only w=0.5 tested; no curriculum learning or confidence thresholding.

Full results: [results.md](experiments/015-student-training/results.md)

---

### 3.16 Exp 016: SED Architecture

**Hypothesis**: An EfficientNet-B0 SED (sound event detection) model with frame-level predictions and post-processing will outperform clip-level models on the soundscape evaluation.

**Dataset**: BirdCLEF 2026, 234 species.

**Models**: EfficientNet-B0 SED with GeM pooling + AttHead.

| Variant | Raw LB | PP LB | PP Gain |
|---------|--------|-------|---------|
| v1 (base SED) | 0.795 | **0.827** | +0.028 |
| v2 (NS-JFT + CE loss) | 0.755 | — | −0.040 |
| v3 (waveform mixup) | 0.802 | — | — |
| v3 (TF-ported) | 0.813 | — | — |

- SED architecture validates the pivot from clip-level models. PP (TTA + temporal smoothing + TopN filtering) adds +0.028.
- NS-JFT pretrained weights + CE loss (v2) regressed to 0.755 — pretrained weights don't always help.
- Waveform-level mixup (v3, 0.802) slightly better than baseline; TF-ported weights (0.813) strongest v3 variant.

**Limitation**: EfficientNet-B0 is small; larger backbones and mel frontend not yet explored.

Full results: [results.md](experiments/016-sed-architecture/results.md)

---

### 3.17 Exp 017: Mel Overhaul

**Hypothesis**: High-resolution mel spectrograms (n_fft=4096, n_mels=224) and AmplitudeToDB normalization will significantly improve SED performance.

**Dataset**: BirdCLEF 2026, 234 species.

**Models**: EfficientNet-B0 SED with upgraded mel frontend.

| Variant | Key Change | Public LB | Δ vs v1 |
|---------|-----------|----------|---------|
| v4a | n_fft=4096, n_mels=224 | 0.825 | +0.030 |
| v4b | + AmplitudeToDB | **0.859** | +0.064 |
| Steps 3–5 | Combinations, fulldata, loss | < 0.859 | Regressed |

- High-res mel (v4a) adds +0.030 over exp 016 v1, confirming frequency resolution matters for species discrimination.
- AmplitudeToDB (v4b) adds another +0.034, making it the single largest gain in the SED pipeline. dB-scale normalization is critical.
- Steps 3–5 (combining changes, fulldata training, loss ablations) all regressed — diminishing returns from stacking changes.

**Limitation**: Best single SED at the time but still below Perch probes (exp 018).

Full results: [results.md](experiments/017-mel-overhaul/results.md)

---

### 3.18 Exp 018: Perch Probes

**Hypothesis**: Frozen Perch v2 embeddings with simple classifiers (LogReg, MLP, ProtoSSM) can outperform fine-tuned SED models by leveraging Google's large-scale bird audio pretraining.

**Dataset**: BirdCLEF 2026, 234 species. Perch v2 embeddings extracted from all focal + soundscape data.

**Models**: Perch v2 (frozen) → LogReg / MLP / ProtoSSM classifiers.

| Variant | Approach | Public LB |
|---------|----------|----------|
| Raw Perch | Direct embedding similarity | 0.825 |
| Genus proxy | Genus-level aggregation | 0.836 |
| Probes | LogReg/MLP on embeddings | 0.893 |
| OOF v2-fix | Out-of-fold with bug fix | 0.911 |
| ProtoSSM v4 | Prototypical + state-space model | **0.918** |
| ProtoSSM v4.2 | Batched 4 changes | 0.899 |

- Perch probes dramatically outperform fine-tuned models: 0.893 LB from a simple LogReg/MLP, vs 0.859 best SED (exp 017).
- ProtoSSM v4 reaches 0.918, establishing a new best by a large margin.
- v4.2 regressed −0.019 by batching 4 changes simultaneously — single-variable testing is essential.
- Perch's large-scale bird audio pretraining provides representations that simple classifiers can exploit effectively.

**Limitation**: Frozen embeddings cannot adapt to novel species or domain shifts; TF dependency for Perch inference.

Full results: [results.md](experiments/018-perch-probes/results.md)

---

### 3.19 Exp 019: SED Ablations

**Hypothesis**: Systematic single-variable ablations (backbone, mel, normalization, loss, augmentation) will identify the optimal SED configuration.

**Dataset**: BirdCLEF 2026, 234 species.

**Models**: EfficientNet-B0 and HGNetV2-B0 SED variants, v5a–v5n.

| Variant | Key Change | Public LB | Δ vs v5g |
|---------|-----------|----------|----------|
| v5f | EfficientNet-B0 + ttahara recipe | 0.855 | −0.037 |
| v5g | HGNetV2-B0 + ttahara recipe | 0.892 | — |
| v5j | AmplitudeToDB + NNAudio | **0.906** | +0.014 |
| v5j-4fold | 4-fold ensemble | **0.913** | +0.021 |
| v5m/v5n | embed_dim ablations | < 0.906 | Regressed |

- HGNetV2-B0 backbone (v5g) outperforms EfficientNet-B0 (v5f) by +0.037 — backbone choice is the largest lever.
- AmplitudeToDB + NNAudio mel (v5j) adds +0.014 on top, reaching 0.906 LB (best single SED).
- 4-fold ensembling (v5j-4fold) pushes to 0.913, confirming fold diversity helps.
- Offline val/AUROC is anti-correlated with LB — cannot rank models without submitting.

**Limitation**: Single seed per variant; offline metrics unreliable for model selection.

Full results: [results.md](experiments/019-sed-ablations/results.md)

---

### 3.20 Exp 020: Perch ONNX + ProtoSSM

**Hypothesis**: ONNX-exported Perch can match TF Perch accuracy while enabling faster single-kernel inference on Kaggle.

**Dataset**: BirdCLEF 2026, 234 species.

**Models**: Perch v2 (TF vs ONNX) + ProtoSSM (V17 vs V18 configs).

| Variant | Config | Backend | Public LB | Δ vs V18+TF |
|---------|--------|---------|----------|-------------|
| V17+TF | d=256, 3L, 40ep, TTA×3 | TF | 0.922 | −0.003 |
| V17+ONNX MLP-only | d=256, 3L, proto_w=0.0 | ONNX | 0.911 | −0.014 |
| V18+ONNX | d=320, 4L, 80ep, TTA×5 | ONNX | 0.920 | −0.005 |
| **V18+TF** | **d=320, 4L, 80ep, TTA×5** | **TF (two-kernel)** | **0.925** | **—** |

V18 scaling over V17: d_model 256→320, SSM layers 3→4, cross-attn heads 4→8,
MLP probes PCA-64→128, training epochs 40→80, TTA shifts 3→5.

- ONNX Perch runs 9× faster (~8 min vs ~65 min) but incurs −0.005 LB penalty (max logit diff 2.7e-4 compounds through MLP probes → ProtoSSM fusion → per-class thresholds → TTA).
- Two-kernel split: training kernel saves artifacts → inference kernel loads AFTER TF inference (memory optimization: loading 75MB before TF caused 2× slowdown).
- Gap to pantanal reference (0.930): −0.005.

**Limitation**: ONNX penalty is real but small; two-kernel workflow adds complexity.

Full results: [results.md](experiments/020-perch-onnx/results.md)

---

### 3.21 Exp 021: SED Training — 7 Phases

**Hypothesis**: Training recipe modifications (loss functions, augmentation, head architecture, multi-context) can improve SED beyond the v5j baseline (0.906 LB).

**Dataset**: BirdCLEF 2026, 234 species.

**Models**: HGNetV2-B0 SED, phases A–G (14 training recipe changes).

| Phase | Changes | Best Variant | val/auroc | LB | Δ vs v5j |
|-------|---------|-------------|-----------|-----|----------|
| A | Site/hour priors (PP only) | v5j-4fold-pp2 | — | **0.917** | **+0.004** |
| B | CE loss, waveform MixUp, 25ep | v6b (wave MixUp) | 0.871 | 0.887 | −0.019 |
| C | AttHead + backbone scaling | v6j (EffNet-B0 NS) | 0.874 | 0.893 | −0.013 |
| D | Multi-context AttHead (15-20s) | v7c (EffNet 3×5s) | 0.851 | — | −0.023+ |
| E | Multi-context LSEHead | v8b (HGNet 4×5s) | **0.881** | 0.896 | −0.010 |
| F | Competition AttHead fix, fmax | v6k (fixed AttHead) | 0.871 | 0.888 | −0.018 |
| G | Manifold Mixup | v6l | 0.874 | 0.899 | −0.007 |

- **Phase A (PP only) is the sole positive result** — site/hour priors add +0.003 single, +0.004 4-fold. No retraining needed.
- **Phases B–G: 0-for-14 on LB.** Best offline (v8b, 0.881 val) is worst on LB (0.896). Offline metrics are anti-correlated with LB.
- v5j-4fold-pp2 (0.917) remains best SED. The v5j recipe is a local optimum — every perturbation falls off a generalization cliff.

**Limitation**: The gap to 0.920+ competitors is not in the training recipe — it's in something else (data augmentation details, inference tricks, or architecture differences). See `FINDINGS.md` insight #3.

Full results: [results.md](experiments/021-sed-training/results.md)

---

### 3.22 Exp 022: PCEN Normalization

**Hypothesis**: Learnable PCEN (Per-Channel Energy Normalization) will improve domain invariance between focal recordings and soundscapes, boosting LB scores.

**Dataset**: BirdCLEF 2026, 234 species.

**Models**: HGNetV2-B0 SED with learnable PCEN replacing AmplitudeToDB.

| Variant | Normalization | Public LB | Δ vs v5j |
|---------|--------------|----------|----------|
| v10a | Learnable PCEN | 0.880 | −0.026 |
| v5j-repro | AmplitudeToDB (reproduction) | 0.903 | −0.003 |

- **REJECTED**: PCEN regresses to 0.880, a −0.026 deficit versus v5j (0.906).
- v5j-repro (0.903) confirms the training pipeline is correct — PCEN itself is the cause of regression, not a bug.
- Despite theoretical motivation for domain-invariant normalization, AmplitudeToDB empirically dominates on this task.

**Limitation**: Only default PCEN hyperparameters tested; no grid search over PCEN's learnable parameters.

Full results: [results.md](experiments/022-pcen-normalization/results.md)

---

### 3.23 Exp 023: Perch + SED Ensemble

**Hypothesis**: Score-level fusion of Perch (ONNX, MLP probes) and SED (v5j-4fold-pp2) predictions in a single Kaggle kernel will outperform either model alone.

**Dataset**: BirdCLEF 2026, 234 species. Single Kaggle CPU kernel (~64 min for 600 files).

**Models**: Perch ONNX + ProtoSSM/MLP probes (0.920 standalone) + HGNetV2-B0 SED v5j-4fold-pp2 (0.917 standalone).

| Config | w_perch | proto_weight | Perch Backend | SED Model | LB | Δ vs best single |
|--------|---------|-------------|--------------|-----------|-----|-----------------|
| w=0.5 | 0.5 | 0.5 | ONNX | 4-fold | 0.927 | +0.002 |
| w=0.3 | 0.3 | 0.5 | ONNX | 4-fold | 0.927 | +0.002 |
| w=0.7 | 0.7 | 0.5 | ONNX | 4-fold | 0.926 | +0.001 |
| **MLP-only** | **0.5** | **0.0** | **ONNX** | **4-fold** | **0.929** | **+0.004** |
| No PP | 0.5 | 0.5 | ONNX | 4-fold | 0.922 | −0.003 |
| TF+single+MLP | 0.5 | 0.0 | TF | single | TIMEOUT | — |

- **MLP-only (proto_weight=0.0): 0.929 LB — new best overall.** Dropping ProtoSSM (which was trained on TF embeddings but runs on ONNX) and using only MLP probes gains +0.002. Simpler probes are more robust to ONNX embedding distribution shift.
- Weight curve is flat (0.926-0.927 across w=0.3/0.5/0.7) — both models contribute equally.
- PP is worth +0.005: each pipeline's individual PP helps even when outputs are averaged.
- **TF Perch timed out** (~10s/file × 600 files > 90 min) — TF backend not viable for ensemble.
- Config-driven YAML push system: `push.sh <config_name>` injects override, creates separate Kaggle notebook per config.

**Limitation**: Only score-level fusion tested; no learned fusion or stacking. TF Perch (0.925 quality) cannot fit within 90 min alongside SED.

Full results: [results.md](experiments/023-perch-sed-ensemble/results.md)

---

## 4. Cross-Experiment Analysis (Experiments 001–023)

### 4.1 Master Comparison Tables

#### BirdSet HSN (21 classes)

All configurations evaluated on HSN `test_5s` (12,000 samples), sorted by cmAP descending.

| Model | Exp | Params | Pretraining | cmAP | AUROC |
|-------|-----|--------|-------------|------|-------|
| BirdSet ConvNeXT | 001 | — | BirdSet XCL (audio) | 0.480 | 0.883 |
| BirdSet EfficientNetB1 | 001 | — | BirdSet XCL (audio) | 0.380 | 0.840 |
| ConvNeXT-Tiny (ImageNet) | 003 | 27.8M | ImageNet | 0.191 | 0.638 |
| ViT-Base (ImageNet-21k) | 002 | 85.5M | ImageNet-21k | 0.186 | 0.662 |
| ViT-Small (ImageNet) | 003 | 21.5M | ImageNet | 0.174 | 0.707 |
| ViT-Small 128×4 frame | 004 | 21.5M | None | 0.118 | 0.579 |
| ViT-Small 128×2 frame | 004 | 21.5M | None | 0.115 | 0.587 |
| ViT-Small 128×1 frame | 004 | 21.5M | None | 0.099 | 0.633 |
| ViT-Small scratch | 003 | 21.5M | None | 0.098 | 0.594 |
| ConvNeXT-Tiny scratch | 003 | 27.8M | None | 0.087 | 0.607 |
| ViT-Small 16×16 baseline | 004 | 21.5M | None | 0.086 | 0.544 |

#### BirdCLEF 2025 (206 classes)

All configurations evaluated on held-out test set (2,803 samples), sorted by ROC-AUC descending (the Kaggle competition metric).

| Model | Exp | Encoder | Pretraining | LoRA Targets | Head | Backend | AUROC | cmAP |
|-------|-----|---------|-------------|-------------|------|---------|-------|------|
| EAT LoRA attn + MLP head | 006 | EAT-base (90M) | AudioSet | attn | MLP 512 | adapters | **0.945** | 0.536 |
| ProtoCLR LoRA attn | 009 | CvT-13 (19M) | Xeno-Canto | attn | Linear | PEFT | 0.942 | 0.531 |
| EAT LoRA attn + MLP head | 006 | EAT-base (90M) | AudioSet | attn | MLP 512 | PEFT | 0.942 | 0.511 |
| ConvNeXt LoRA MLP + MLP head | 007 | ConvNeXt-Base (89M) | ImageNet-22K | MLP | MLP 512 | PEFT | 0.941 | 0.525 |
| ConvNeXt LoRA MLP | 007 | ConvNeXt-Base (89M) | ImageNet-22K | MLP | Linear | PEFT | 0.939 | 0.527 |
| ProtoCLR LoRA all | 009 | CvT-13 (19M) | Xeno-Canto | attn + MLP | Linear | PEFT | 0.939 | 0.537 |
| EAT LoRA attn | 006 | EAT-base (90M) | AudioSet | attn | Linear | adapters | 0.938 | 0.485 |
| ProtoCLR full fine-tune | 009 | CvT-13 (19M) | Xeno-Canto | — (all 19M) | Linear | — | 0.925 | 0.525 |
| EAT pretrained 16×16 baseline | 008 | EAT-base (90M) | AudioSet | attn | Linear | PEFT | 0.933 | 0.494 |
| EAT LoRA attn | 006 | EAT-base (90M) | AudioSet | attn | Linear | PEFT | 0.927 | 0.469 |
| EAT LoRA all | 006 | EAT-base (90M) | AudioSet | attn + MLP | Linear | PEFT | 0.923 | 0.482 |
| EAT random 16×16 control | 008 | EAT-base (90M) | AudioSet | attn | Linear | PEFT | 0.899 | 0.360 |
| EAT frozen linear probe | 005 | EAT-base (90M) | AudioSet | — | Linear | — | 0.890 | 0.301 |
| EAT frame 4×128 | 008 | EAT-base (90M) | AudioSet | attn | Linear | PEFT | 0.873 | 0.284 |
| EAT frame 2×128 | 008 | EAT-base (90M) | AudioSet | attn | Linear | PEFT | 0.864 | 0.260 |

#### BirdCLEF 2026 Kaggle Leaderboard (234 classes)

All models submitted via exp 010+ pipelines. Sorted by public LB macro-AUROC descending.

| Model | Exp | Architecture | Public LB |
|-------|-----|-------------|----------|
| Perch + SED ensemble (MLP-only) | 023 | Perch ProtoSSM + SED v5j-4fold | **0.929** |
| Perch + SED ensemble (w=0.5) | 023 | Perch ProtoSSM + SED v5j-4fold | 0.927 |
| Perch V18+TF | 020 | Perch v2 + ProtoSSM | 0.925 |
| Perch V18+ONNX | 020 | Perch v2 + ProtoSSM | 0.920 |
| Perch ProtoSSM v4 | 018 | Perch v2 + ProtoSSM | 0.918 |
| SED v5j-4fold-pp2 | 021 | HGNetV2-B0 SED | 0.917 |
| SED v5j-4fold | 019 | HGNetV2-B0 SED | 0.913 |
| Perch OOF v2-fix | 018 | Perch v2 + probes | 0.911 |
| SED v5j | 019 | HGNetV2-B0 SED | 0.906 |
| SED v6l (manifold mixup) | 021 | HGNetV2-B0 SED | 0.899 |
| Perch probes | 018 | Perch v2 + LogReg/MLP | 0.893 |
| SED v5g (ttahara) | 019 | HGNetV2-B0 SED | 0.892 |
| PCEN v10a | 022 | HGNetV2-B0 SED | 0.880 |
| SED v4b (AmplitudeToDB) | 017 | EfficientNet-B0 SED | 0.859 |
| SED v5f (EfficientNet) | 019 | EfficientNet-B0 SED | 0.855 |
| SED v4a (hi-res mel) | 017 | EfficientNet-B0 SED | 0.825 |
| SED v1 + PP | 016 | EfficientNet-B0 SED | 0.827 |
| ProtoCLR soundscape | 012 | ProtoCLR/CvT-13 LoRA | 0.767 |
| ProtoCLR student | 015 | ProtoCLR/CvT-13 LoRA | 0.762 |
| ConvNeXt soundscape | 012 | ConvNeXt-Base LoRA | 0.755 |
| ProtoCLR fulldata | 011 | ProtoCLR/CvT-13 LoRA | 0.736 |
| ProtoCLR 80/10/10 | 011 | ProtoCLR/CvT-13 LoRA | 0.721 |
| ConvNeXt 80/10/10 | 011 | ConvNeXt-Base LoRA | 0.720 |
| EAT soundscape | 012 | EAT-base LoRA | 0.720 |
| EAT 80/10/10 | 011 | EAT-base LoRA | 0.701 |
| EAT fulldata | 011 | EAT-base LoRA | 0.698 |
| EAT student | 015 | EAT-base LoRA | 0.696 |
| ConvNeXt fulldata | 011 | ConvNeXt-Base LoRA | 0.695 |
| ProtoCLR 2025 | 010 | ProtoCLR/CvT-13 LoRA | 0.531 |

### 4.2 Key Insights

1. **Pretraining domain matters, but less than expected for LoRA.** On HSN, audio-pretrained models (0.480 cmAP) outperform ImageNet-pretrained models (0.174–0.191) by 2.5×. On BirdCLEF 2025 with LoRA, however, the gap narrows dramatically: ImageNet-pretrained ConvNeXt (AUROC=0.941) nearly matches AudioSet-pretrained EAT (0.945), and Xeno-Canto-pretrained ProtoCLR (0.942) matches both despite being 5× smaller. LoRA appears to equalize pretraining domains when sufficient task-specific data is available.

2. **LoRA sufficiency.** Adapting <1% of EAT's parameters via LoRA produces AUROC=0.945, up from 0.890 with frozen probing (+6.2%). Full fine-tuning is not only impractical (exp 005 timed out at 12h) but actively harmful: ProtoCLR full FT (exp 009, all 19M params) scored AUROC=0.925, a -1.7% drop from LoRA (0.942), due to catastrophic forgetting of the pretrained representations. LoRA's low-rank constraint preserves task-aligned features while allowing targeted adaptation.

3. **Head–LoRA interaction.** For EAT, where LoRA targets only attention layers (qkv, proj), adding an MLP classification head compensates for the unadapted MLP layers and adds +0.004–0.015 AUROC. For ConvNeXt, where LoRA targets the MLP layers (fc1, fc2) directly, the MLP head provides a marginal AUROC gain (0.941 vs 0.939). The head compensates for whatever LoRA does not adapt.

4. **Pretrained patch embeddings are critical.** Freezing EAT's pretrained 16×16 patch embedding outperforms randomly initializing and fully training a new one by +3.8% AUROC (0.933 vs 0.899 in exp 008). The pretrained Conv2d is a learned spectro-temporal feature extractor, not merely a projection layer. Preserving the native tokenization scheme of pretrained models is essential.

5. **Frame tokenization paradox.** Vertical frame patches improve ViT from scratch by +37% cmAP (exp 004) but degrade pretrained EAT by −6.4% to −7.4% AUROC (exp 008). The pretrained transformer's position embeddings, attention patterns, and layer norms are calibrated for a 2D patch grid. Frame tokens produce an incompatible 1D sequence that LoRA alone cannot remap. Tokenization improvements are secondary to representation quality.

6. **Overfitting is pervasive.** On HSN, all models show severe train-test gaps (train cmAP ~0.94 vs test ~0.12 in exp 004). On BirdCLEF 2025, gaps are moderate (train ~0.85–0.90 vs test ~0.53 cmAP). No experiment applied data augmentation (SpecAugment, mixup, time masking). This remains the largest unexplored axis of improvement.

7. **Ensemble readiness.** Three architecturally diverse models reach AUROC parity (0.939–0.945) on BirdCLEF 2025: EAT (transformer, AudioSet pretraining), ConvNeXt (CNN, ImageNet pretraining), and ProtoCLR (CvT with convolutional tokenization, Xeno-Canto contrastive pretraining). Their different architectures, patch strategies, and pretraining domains should produce complementary error patterns, making them strong candidates for teacher ensemble pseudo-labeling.

8. **Pipeline portability (exp 010–011).** The LoRA fine-tuning pipeline transfers from BirdCLEF 2025 to 2026 with zero hyperparameter changes. A unified `train.py` handles all 3 encoders, and the ONNX export + Kaggle kernel works reliably across architectures. This infrastructure investment pays off with each new dataset or model variant.

9. **Fulldata training is a double-edged sword (exp 011).** Training on 100% of focal data (no val/test split) improved ProtoCLR by +0.015 LB but hurt EAT (−0.003) and ConvNeXt (−0.025). The root cause is fixed-epoch training without early stopping: ConvNeXt converges at epoch 9 but trained 50, massively overfitting. The faster a model converges, the more it overshoots. Future fulldata runs should cap epochs at ~1.5× the best epoch from the split run.

10. **Soundscape domain adaptation is universally beneficial (exp 012).** Fine-tuning on just 739 labeled soundscape segments improved all 3 models on the Kaggle LB (+0.022 to +0.060). The improvement comes from two sources: species expansion (206→234 covers 28 previously-zero species) and domain adaptation (models learn multi-species overlapping audio patterns). ConvNeXt gained the most (+0.060), suggesting it had the largest focal→soundscape domain gap.

11. **Offline metrics don't predict LB ranking (exp 012).** ConvNeXt (0.844 test AUROC) and ProtoCLR (0.845) are nearly tied on the offline test split, but ProtoCLR leads by 0.012 on the actual Kaggle LB (0.767 vs 0.755). The real evaluation on diverse soundscapes amplifies differences that the held-out focal test set underestimates. LB submission is essential for reliable model comparison.

12. **ProtoCLR consistently leads despite smallest size.** Across both exp 011 and 012, ProtoCLR (~20M params) achieves the highest LB scores (0.736 fulldata, 0.767 soundscape) while being 5× smaller than EAT and ConvNeXt (~90M). Its Xeno-Canto bird-specific pretraining provides a persistent advantage that neither AudioSet (EAT) nor ImageNet (ConvNeXt) pretraining can match on this task.

13. **Pseudo-labeling failed due to architecture, not data (exp 013–015).** The full pipeline worked technically — MixIT separated sources, teacher ensemble achieved 0.885 AUROC, and 127K segments were pseudo-labeled. But student training regressed on LB (ProtoCLR −0.005, EAT −0.024). Students converge to teacher quality in 1–5 epochs, then overfit to pseudo-label noise. The bottleneck was the clip-level 5s-window architecture, not data quantity, motivating the SED pivot.

14. **Architecture pivot to SED was the breakthrough (exp 016–019).** Switching from clip-level LoRA models (~0.767 ceiling) to EfficientNet-B0 SED immediately reached 0.827 with PP. The SED pipeline benefits from frame-level predictions that naturally handle multi-species soundscapes. Subsequent improvements came from backbone (HGNetV2-B0 +0.037), mel frontend (AmplitudeToDB +0.034), and fold ensembling (+0.007).

15. **Frozen Perch embeddings dominate fine-tuned models (exp 018, 020).** Simple probes on frozen Perch v2 embeddings (0.893) outperformed the best fine-tuned SED (0.859 at the time) with orders of magnitude less training. ProtoSSM v4 reached 0.918. Google's large-scale bird audio pretraining produces representations that simple classifiers can exploit more effectively than end-to-end fine-tuning on limited data.

16. **Training recipe changes are 0-for-14 on LB (exp 021).** CE loss, waveform MixUp, AttHead, multi-context, manifold mixup — every training recipe modification regressed on LB. Only post-processing priors (site/hour) helped (+0.003/+0.004). The SED backbone and mel frontend are the actual levers; training tricks do not transfer.

17. **Offline val/AUROC is anti-correlated with LB (exp 019, 021).** Models that score higher on the offline validation set consistently score lower on the public leaderboard. This makes offline metrics useless for model selection — every ranking decision requires a Kaggle submission.

18. **Score-level ensemble reaches 0.929, near the 0.930 reference (exp 023).** Fusing Perch and SED predictions in a single kernel, with MLP-only weighting (proto_weight=0.0), achieves the best score. The prototypical component adds noise; the MLP probe alone captures the useful signal from Perch embeddings.

### 4.3 Open Questions

**Closing the Gap to 0.930+**
- Can learned fusion (stacking, neural combiner) outperform score-level averaging?
- Would a larger SED backbone (HGNetV2-B2/B4) improve the SED component of the ensemble?
- Can multi-fold Perch probes (analogous to SED 4-fold) push Perch scores higher?

**SED Improvements**
- Are there backbone or mel frontend changes beyond HGNetV2-B0 + AmplitudeToDB + NNAudio that help?
- Would self-training or noisy student approaches work better now that SED provides frame-level labels?

**Perch Pipeline**
- Can the ONNX accuracy gap (−0.005–0.010) be closed with better quantization or calibration?
- Would fine-tuning Perch (rather than freezing) help, or does it suffer the same overfitting as LoRA models?

**Ensemble and Deployment**
- What is the optimal number of SED folds for the Kaggle time budget?
- Can the two-kernel Perch workflow be simplified to a single kernel without accuracy loss?

**Deferred Work**
- Multi-seed runs to quantify variance in SED and Perch pipelines
- Explore additional post-processing: per-species thresholds, calibration
- Test whether MixIT-separated pseudo-labels help SED (vs clip-level where they failed)
