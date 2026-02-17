# Results: Frame-Based Tokenization for ViT

**Date Completed**: 2026-02-17
**Author**: Murilo

## Summary

Vertical frame tokenization significantly outperforms standard 16×16 square patches for ViT on audio spectrograms. After controlling for the mel backend confound (torchaudio rerun), the best vertical config — **128×2 (cmAP=0.098)** — improves +49% over the torchaudio 16×16 baseline (0.066) and +34% over the exp 003 scratch baseline (0.073). All three vertical configs beat both baselines.

The initial nnAudio runs were confounded by a mel backend change that depressed the 16×16 baseline from 0.073 (exp 003) to 0.029. The torchaudio rerun resolved this: the torchaudio 16×16 baseline (0.066) is within 10% of exp 003 (0.073), confirming the mel backend was the primary confound — not hardware or other factors.

**Verdict**: confirmed — vertical frame tokenization improves cmAP by 30–49% over square 16×16 patches on HSN when training ViT-Small from scratch.

## Observations

### Quantitative

#### Torchaudio Rerun — Test Set Results (HSN `test_5s`, 12,000 samples)

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.066 | 0.533 | 0.005 | 0.028 | 0.161 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.086 | 0.648 | 0.014 | 0.137 | 0.153 | 21.54M |
| B: 128×4 | 128×4 | 125 | 0.087 | 0.579 | 0.051 | 0.438 | 0.134 | 21.55M |
| **A: 128×2** | **128×2** | **250** | **0.098** | **0.621** | **0.012** | **0.304** | **0.145** | **21.50M** |

#### Relative Improvement over 16×16 Baseline (torchaudio)

| Config | cmAP | cmAP Lift | AUROC Lift |
|--------|------|-----------|------------|
| Baseline (16×16) | 0.066 | — | — |
| C: 128×1 | 0.086 | **+30%** (1.30×) | +22% |
| B: 128×4 | 0.087 | **+33%** (1.33×) | +9% |
| A: 128×2 | 0.098 | **+49%** (1.49×) | +17% |

#### Torchaudio Rerun — Training Dynamics

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 8 | 24 | 0.066 | 5.13 it/s | 29m 16s |
| A: 128×2 | 6 | 22 | 0.098 | 5.12 it/s | 28m 43s |
| B: 128×4 | 8 | 24 | 0.087 | 8.01 it/s | 27m 19s |
| C: 128×1 | 6 | 22 | 0.086 | 2.59 it/s | 34m 38s |

#### Cross-Experiment Baseline Comparison (confound resolved)

| Run | Mel Backend | GPU | cmAP |
|-----|------------|-----|------|
| Exp 003 — ViT-Small scratch | CPU (torchaudio) | RTX 6000 | 0.073 |
| Exp 004 — Baseline 16×16 (nnAudio) | GPU (nnAudio) | V100 | 0.029 |
| Exp 004 — Baseline 16×16 (torchaudio rerun) | CPU (torchaudio) | V100 | 0.066 |

The torchaudio rerun baseline (0.066) is within 10% of exp 003 (0.073), confirming the mel backend was the primary confound. The remaining gap (0.007) may be due to hardware (V100 vs RTX 6000), seed variance, or minor environment differences.

#### nnAudio vs Torchaudio — Side-by-Side

| Config | nnAudio cmAP | Torchaudio cmAP | Lift from mel fix |
|--------|-------------|-----------------|-------------------|
| Baseline (16×16) | 0.029 | 0.066 | +128% (2.28×) |
| A: 128×2 | 0.054 | 0.098 | +81% (1.81×) |
| B: 128×4 | 0.063 | 0.087 | +39% (1.39×) |
| C: 128×1 | 0.051 | 0.086 | +69% (1.69×) |

All configs improved substantially with torchaudio mel. The baseline benefited the most (2.28×), narrowing the gap between baseline and vertical configs from 2.2× (nnAudio) to 1.49× (torchaudio). This suggests nnAudio's mel spectrogram disproportionately harms 16×16 square patches compared to vertical frames.

#### Original nnAudio Run — Test Set Results (for reference)

| Config | Patch | Tokens | cmAP | AUROC | Top-1 Acc | Top-5 Acc | Loss | Params |
|--------|-------|--------|------|-------|-----------|-----------|------|--------|
| Baseline | 16×16 | 248 | 0.029 | 0.458 | 0.001 | 0.544 | 0.179 | 21.50M |
| C: 128×1 | 128×1 | 500 | 0.051 | 0.536 | 0.483 | 0.566 | 0.122 | 21.54M |
| A: 128×2 | 128×2 | 250 | 0.054 | 0.536 | 0.093 | 0.573 | 0.123 | 21.50M |
| B: 128×4 | 128×4 | 125 | 0.063 | 0.601 | 0.440 | 0.547 | 0.118 | 21.55M |

#### Original nnAudio Run — Training Dynamics (for reference)

| Config | Best Epoch | Stopped | Best Val cmAP | Train Speed | Job Walltime |
|--------|-----------|---------|---------------|-------------|--------------|
| Baseline (16×16) | 7 | 17 | 0.029 | 4.16 it/s | 38m 27s |
| A: 128×2 | 2 | 18 | 0.054 | 4.15 it/s | 38m 28s |
| B: 128×4 | 2 | 17 | 0.063 | 5.95 it/s | 37m 23s |
| C: 128×1 | 2 | 17 | 0.051 | 2.34 it/s | 43m 01s |

### Qualitative

- **All vertical frame configs outperform 16×16 across both mel backends** — the relative ranking (vertical > square) holds regardless of mel backend, confirming this is a genuine tokenization effect, not an artifact
- **128×2 is the best config with torchaudio mel** (cmAP=0.098), while 128×4 was best with nnAudio (0.063). The ranking shift (128×2 > 128×4 with torchaudio vs 128×4 > 128×2 with nnAudio) suggests the mel backend interacts with temporal resolution — torchaudio's better spectrograms may contain finer temporal detail that 20ms tokens (128×2) can capture but 40ms tokens (128×4) cannot
- **Convergence shifted later with torchaudio** — best epochs moved from 2 (nnAudio) to 6–8 (torchaudio), suggesting torchaudio's richer spectrograms allow productive learning over more epochs. The nnAudio runs' epoch-2 peak may indicate impoverished inputs that the model quickly overfits
- **128×1 and 128×4 are nearly tied** (0.086 vs 0.087) — with torchaudio, the gap between configs is smaller, suggesting the mel quality matters more than fine differences in temporal resolution
- **Training speed still scales inversely with token count** — 128×4 (125 tokens) at 8.01 it/s, 128×2 (250) at 5.12 it/s, 128×1 (500) at 2.59 it/s, consistent with O(n²) attention cost
- **Job walltimes are shorter** than nnAudio runs (~27–35 min vs ~37–43 min) despite similar epoch counts, because cached mel loading is faster than waveform pre-loading + GPU mel computation
- **No NaN issues** in the torchaudio runs (unlike the nnAudio runs which showed NaN logging artifacts)

## Analysis

The torchaudio rerun confirms the core hypothesis: **vertical frame tokenization provides a better inductive bias than 16×16 square patches for ViT on spectrograms**. With the mel backend confound removed, the improvement is +30–49% in cmAP — smaller than the initial nnAudio estimate (76–117%) but still large and consistent across all three vertical configs.

**Why 128×2 wins with torchaudio**: The ranking change from 128×4 (nnAudio) to 128×2 (torchaudio) is informative. With a higher-fidelity mel spectrogram, finer temporal resolution (20ms vs 40ms) becomes beneficial — the model can leverage temporal detail that was absent or degraded in nnAudio's spectrograms. The 250-token sequence (128×2) is a good balance: enough tokens for fine temporal discrimination, few enough for effective attention learning with 5,460 training samples.

**Mel backend is a critical variable**: The nnAudio mel spectrogram consistently underperforms torchaudio across all 4 configs. The baseline was hit hardest (2.28× worse with nnAudio), but even the best vertical config improved 39% from mel fix alone. This implies nnAudio's FFT implementation, window function, or mel filter bank construction differs materially from torchaudio's. **Future experiments should use torchaudio mel unless nnAudio is specifically validated** to match.

**Convergence behavior**: The shift from epoch-2 peaks (nnAudio) to epoch 6–8 peaks (torchaudio) suggests the torchaudio spectrograms are more information-rich. With nnAudio, the model quickly extracted what it could from impoverished inputs and then overfit. With torchaudio, productive learning continues longer, resulting in both higher peaks and later convergence.

**Absolute performance context**: The best torchaudio vertical config (128×2, cmAP=0.098) is still below exp 003's pretrained ViT-Small (0.121) and ConvNeXT-Tiny (0.180). Tokenization strategy alone does not close the pretraining gap, but it narrows it — 0.098 scratch vs 0.121 pretrained is only a 19% gap, compared to 40% with 16×16 scratch (0.073).

### Confounders

**Mel backend (resolved)**: The torchaudio rerun confirmed the mel backend was the primary confound in the original results. The torchaudio baseline (0.066) closely matches exp 003 (0.073), validating cross-experiment comparison.

**Hardware**: Torchaudio reruns used V100 16GB (same as original nnAudio runs). Exp 003 used RTX 6000 24GB. This should not affect model accuracy (same 16-mixed AMP precision).

**Single seed**: Each config run once — no variance estimate. The consistent improvement across all 3 vertical configs (all beat baseline by 30%+) makes the finding robust, but multi-seed runs would strengthen it.

**Hyperparameters tuned for 16×16**: lr=5e-4, wd=0.05, warmup=10 carried over from exp 003. Vertical frame configs may benefit from different hyperparameters — best epochs (6–8) are near the end of the 10-epoch warmup, suggesting peak LR is nearly reached when the model peaks. A shorter warmup or higher initial LR could further improve vertical frame performance.

## Artifacts

### Torchaudio Rerun

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/baseline_16x16_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x2_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x4_torchaudio/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x1_torchaudio/HSN/metrics.json` |
| SLURM logs | `logs/slurm_baseline_16x16_torchaudio_4017565.log` |
| SLURM logs | `logs/slurm_vertical_128x2_torchaudio_4017567.log` |
| SLURM logs | `logs/slurm_vertical_128x4_torchaudio_4017569.log` |
| SLURM logs | `logs/slurm_vertical_128x1_torchaudio_4017568.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/004-frame-tokenization/checkpoints/*-torchaudio/` |

### Original nnAudio Run

| Type | Location |
|------|----------|
| Metrics | `artifacts/training/baseline_16x16/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x2/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x4/HSN/metrics.json` |
| Metrics | `artifacts/training/vertical_128x1/HSN/metrics.json` |
| Visualization | `artifacts/tokenization_comparison.png` |
| GPU Utilization | `artifacts/gpu-utilization.png` |
| SLURM logs | `logs/slurm_baseline_16x16_4011400.log` |
| SLURM logs | `logs/slurm_vertical_128x2_4011401.log` |
| SLURM logs | `logs/slurm_vertical_128x4_4011402.log` |
| SLURM logs | `logs/slurm_vertical_128x1_4011403.log` |
| Checkpoints | `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/004-frame-tokenization/checkpoints/` |
| W&B | [baseline-16x16](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/pwal) |
| W&B | [vertical-128x2](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/kqek) |
| W&B | [vertical-128x4](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/1ybfne3c) |
| W&B | [vertical-128x1](https://wandb.ai/dsgt-arc/birdset-frame-tokenization/runs/jan07qgv) |
| Code | Branch `mg/tokenization` |

## Next Steps

- [x] ~~Torchaudio mel rerun~~ — completed, mel backend confound resolved
- [ ] Tune hyperparameters for vertical frames — especially try shorter warmup (2–5 epochs) and higher LR, since the model peaks at epoch 6–8 (near end of 10-epoch warmup)
- [ ] Test 128×2 vertical frames with pretrained ViT weights (reshape/interpolate the 16×16 patch embedding to 128×2)
- [ ] Run multi-seed experiments on the top config (128×2) to estimate variance
- [ ] Investigate nnAudio vs torchaudio spectrograms numerically — why does nnAudio disproportionately hurt 16×16 patches?
- [ ] Try horizontal frame tokenization (W×500 patches spanning full time axis) as proposed in the proposal's future scope

**Leads to**: Combining 128×2 vertical frame tokenization with pretrained backbones (strongest expected combination); hyperparameter tuning for non-square patch ViTs; understanding the nnAudio mel quality issue.
