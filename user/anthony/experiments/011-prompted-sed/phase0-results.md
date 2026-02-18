# Results: Experiment 011 Phase 0 — Representation Feasibility Gate

**Date Completed**: 2026-02-17
**Author**: Anthony

## Summary

The SSAMBA Tiny backbone does not encode species-discriminative information well enough in its hidden states for prompted SED to work out of the box. Even the supervised finetuned checkpoint (cmAP=0.425) only achieves 24% linear probe accuracy on 20 species — below the 30% pass threshold. SSL pretraining (pretrain-full, pretrain-warm) shows marginal improvement over AudioSet baseline.

**Verdict**: inconclusive — finetuned backbone shows emerging signal (5x above chance) but below the pass gate. SSL pretraining alone does not produce species-discriminative representations.

## Observations

### Quantitative

Two runs were performed: v1 (50 species, random 5s clips) and v2 (20 species, peak-energy 5s clips). v2 is the fairer test — peak-energy windows maximize vocalization content, and fewer classes reduce noise in the linear probe.

#### v2 Results (20 species, 10 peak-energy clips each = 200 samples)

| Checkpoint | Pooling | Silhouette | Probe Acc | Cos-Within | Cos-Between |
|---|---|---|---|---|---|
| audioset | mean_pool | -0.526 | 11.0% | 0.902 | 0.898 |
| audioset | cls | -0.372 | 10.5% | 0.858 | 0.853 |
| audioset | last_patch | -0.494 | 11.0% | 0.835 | 0.832 |
| pretrain-full | mean_pool | -0.555 | 15.0% | 0.779 | 0.766 |
| pretrain-full | cls | -0.548 | 14.5% | 0.758 | 0.746 |
| pretrain-full | last_patch | -0.626 | 10.0% | 0.700 | 0.693 |
| pretrain-warm | mean_pool | -0.551 | 16.5% | 0.709 | 0.691 |
| pretrain-warm | cls | -0.509 | 14.0% | 0.871 | 0.864 |
| pretrain-warm | last_patch | -0.605 | 10.5% | 0.677 | 0.670 |
| **finetuned** | **mean_pool** | **-0.163** | **24.0%** | **0.445** | **0.405** |
| finetuned | cls | -0.414 | 13.0% | 0.810 | 0.798 |
| finetuned | first_patch | -0.166 | 17.5% | 0.446 | 0.410 |
| **finetuned** | **last_patch** | **-0.135** | **23.0%** | **0.416** | **0.376** |

Random chance = 5% (20 classes). Pass threshold: silhouette > 0.1 AND probe > 30%.

#### v1 Results (50 species, 10 random clips each = 500 samples)

| Checkpoint | mean_pool probe | last_patch probe | mean_pool silhouette |
|---|---|---|---|
| audioset | 6.0% | 3.4% | -0.651 |
| pretrain-full | 6.2% | 3.2% | -0.625 |
| pretrain-warm | 6.2% | 2.8% | -0.623 |
| finetuned | 12.0% | 10.8% | -0.300 |

Random chance = 2% (50 classes).

### Qualitative

- **All checkpoints FAIL the formal gate** (silhouette > 0.1 AND probe > 30%). All silhouette scores are negative, meaning within-class similarity is lower than between-class similarity on average.
- **Finetuned backbone has clear signal**: 24% mean_pool accuracy is 5x above chance, and cosine similarity drops from 0.90 (audioset) to 0.45 (finetuned), showing representations are more spread out and not collapsed into a narrow cone.
- **SSL pretraining provides modest lift**: pretrain-warm (16.5%) > pretrain-full (15.0%) > audioset (11.0%). The warm-start (AudioSet + SSL) is slightly better than from-scratch SSL, but neither approaches finetuned performance.
- **last_patch tracks mean_pool on finetuned** (23% vs 24%): The forward scan's final state does carry species-discriminative information in the finetuned model, suggesting state compression is partially viable for prompting.
- **CLS token is consistently the worst on finetuned** (13%): The CLS at position 0 does not accumulate useful global information. This makes sense — the model was trained with mean pooling, not CLS-based classification.
- **t-SNE plots** show the finetuned model has emerging species clusters in mean_pool/last_patch subplots, while audioset/SSL models show uniform mixing of species.

## Analysis

The core finding is that **species identity is encoded in the finetuned backbone but not strongly enough for unsupervised clustering**. The 24% linear probe is well above chance but well below what a dedicated classification head achieves (cmAP=0.425 ≈ ~60% top-1 accuracy). This means most species-discriminative power lives in the classification head weights, not in the backbone representations.

This is consistent with the SSAMBA Tiny architecture: 7M parameters with 128x1 patches create a very long token sequence (500 tokens for 5s), and the Mamba state must compress a lot of temporal information. The backbone may encode acoustic features (spectral shape, temporal patterns) but not have enough representational capacity to form distinct species manifolds without task-specific supervision.

**Implications for prompted SED:**

1. **Concatenation-based prompting (original plan) is risky**: If the backbone can't form clean species clusters from single clips, it's unlikely to form useful species representations from a reference prefix. The reference signal would be too diffuse to condition downstream token predictions.

2. **Prototypical similarity head (fallback) is more promising**: Process reference and target independently through the backbone, then compute per-token dot-product similarity. This doesn't require the backbone to compress species identity into state — it just needs the backbone to produce representations where same-species tokens are more similar than different-species tokens. The cosine similarity gap (within=0.445 vs between=0.405 on finetuned mean_pool) shows this gap exists, though it's narrow.

3. **The finetuned backbone is significantly better than SSL-only**: If pursuing prompted SED, use the finetuned backbone, not SSL-only. The SSL pretraining objective (masked patch reconstruction) does not push representations toward species discrimination.

### Confounders

- **Patch re-initialization**: The AudioSet checkpoint was pretrained with 16x16 patches. Loading into 128x1 architecture transfers only Mamba weights (98.2%), with patch_embed and pos_embed randomly initialized. This disadvantages the audioset condition specifically.
- **pos_embed center-cropping**: SSL/finetune checkpoints were trained with 1000 frames (10s clips). Loading into 500-frame (5s) models center-crops pos_embed from 1001→502 tokens. The cropped positions may not perfectly align.
- **Small sample size**: 200 samples (10 per class) with 5-fold CV means each fold trains on 160 samples. Linear probe results have high variance (std up to 0.08).
- **Peak-energy selection bias**: Selecting highest-energy windows may favor species with loud, continuous vocalizations over species with quiet or sparse calls.

## Artifacts

| Type | Location |
|------|----------|
| v1 results (50 sp, random) | `~/scratch/birdclef-2026/outputs/exp011/phase0/results.json` |
| v2 results (20 sp, peak) | `~/scratch/birdclef-2026/outputs/exp011/phase0-v2/results.json` |
| t-SNE plots (v2) | `~/scratch/birdclef-2026/outputs/exp011/phase0-v2/tsne_*.png` |
| Script | `scripts/phase0_feasibility.py` |
| SLURM logs | `logs/phase0-*.out` |

## Next Steps

- [ ] Consider prototypical similarity head instead of concatenation-based prompting
- [ ] Test with more samples per class (20-30) to reduce probe variance
- [ ] Evaluate intermediate layer representations (not just final layer) — species signal may be stronger in middle layers before the final normalization
- [ ] Try the conditioning delta test from the proposal: feed same target with different references, check if target embeddings change

**Leads to**: Redesign Phase 1 to use prototypical similarity head as primary approach, with concatenation-based prompting as secondary. The backbone doesn't naturally form species clusters, so the similarity head needs to be trained to project representations into a discriminative space.
