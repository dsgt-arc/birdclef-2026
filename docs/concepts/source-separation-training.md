# Concept: Training a PyTorch MixIT Separator on BirdCLEF Data

**Status**: developing
**Created**: 2026-03-14
**Updated**: 2026-03-14

## Idea

Train a domain-specific audio source separator using MixIT (Mixture Invariant Training) loss in PyTorch, initialized from biodenoising's pretrained Demucs dns48 weights. This replaces Google's legacy TF1 Bird MixIT (TDCN++, 2021) with a modern, multi-taxa model trained directly on BirdCLEF data. The combination is natural: Demucs was originally a 4-source music separator before being adapted for single-output denoising, and MixIT loss is architecture-agnostic. Asteroid's `MixITLossWrapper` provides the loss function out of the box.

The key insight is that **nobody has trained a PyTorch MixIT model for bioacoustics**. Google's Bird MixIT is TF1-only, bird-only, and frozen at 2021. Biodenoising has good animal vocalization priors but only produces one output. Combining the Demucs architecture + biodenoising weights + MixIT loss fills the gap: a PyTorch, multi-taxa, BirdCLEF-specific separator.

## Motivation

The [source separation augmentation](source-separation-augmentation.md) concept identifies foreground/background decomposition and multi-source remixing as promising augmentation strategies for bridging the focal-to-soundscape domain gap. But the only available multi-source separator (Google's Bird MixIT) has serious practical limitations:

1. **TensorFlow 1.x checkpoint** — requires isolated venv, TF compat mode, CUDA version headaches
2. **Bird-only** — trained on 704 Xeno-Canto bird species. BirdCLEF+ 2026 includes amphibians, insects, and mammals (206 species total)
3. **22.05 kHz fixed** — unusual sample rate, requires resampling from BirdCLEF's 32 kHz
4. **2019-era architecture** — TDCN++ predates modern separators by several generations
5. **No domain adaptation** — frozen checkpoint, can't fine-tune for BirdCLEF-specific environments

A custom PyTorch model solves all of these and integrates cleanly with the existing training pipeline.

## Background

### MixIT Training (Architecture-Agnostic)

MixIT is a loss function, not a model. It trains any separator via mixture-of-mixtures:

1. Sum two real recordings: `mom = recording_a + recording_b`
2. Model separates `mom` into M sources
3. Loss finds optimal binary assignment of sources back to the two originals
4. No clean references needed — fully unsupervised

Asteroid provides this in PyTorch: `asteroid.losses.MixITLossWrapper`. It wraps any pairwise loss (SI-SDR, MSE) and handles the combinatorial assignment over `2^M` possible mappings (feasible for M=4 or 8).

```python
from asteroid.losses import MixITLossWrapper, pairwise_neg_sisdr
from asteroid.models import ConvTasNet

model = ConvTasNet(n_src=4, sample_rate=32000)
loss = MixITLossWrapper(pairwise_neg_sisdr)

mom = recording_a + recording_b
est_sources = model(mom)  # (batch, 4, samples)
loss_val = loss(est_sources, torch.stack([recording_a, recording_b], dim=1))
```

References:
- Original MixIT: [arXiv:2006.12701](https://arxiv.org/abs/2006.12701) (NeurIPS 2020)
- Bird MixIT: [arXiv:2110.03209](https://arxiv.org/abs/2110.03209) (ICASSP 2022)
- Asteroid MixIT wrapper: [source](https://github.com/asteroid-team/asteroid/blob/master/asteroid/losses/mixit_wrapper.py)

### Demucs as Separator Backbone

Demucs was **originally designed for 4-source music separation** (vocals, drums, bass, other) before being adapted into the dns48 single-output speech denoiser. The biodenoising project then fine-tuned dns48 on animal vocalizations. The lineage:

```
Demucs (4-source music sep) → dns48 (1-source speech denoising) → biodenoising (1-source animal denoising)
```

Going back to multi-source output is returning Demucs to its native task, but with animal vocalization priors from biodenoising.

References:
- [Biodenoising vendor doc](../vendor/biodenoising.md)
- [Bird MixIT vendor doc](../vendor/bird-mixit.md)

### Modern Separator Architectures Available in PyTorch

| Architecture | Library | Params | SI-SNRi (WSJ0-2mix) | Notes |
|---|---|---|---|---|
| ConvTasNet | Asteroid | ~5M | 15.3 dB | Closest to TDCN++, simplest |
| DPRNNTasNet | Asteroid | ~3M | 18.8 dB | Dual-path RNN |
| DPTNet | Asteroid | ~2.7M | 20.2 dB | Dual-path transformer |
| SuDORMRF | Asteroid | ~2.6M | 17.4 dB | Lightweight, good for CPU |
| SPMamba | [Standalone](https://github.com/JusperLee/SPMamba) | — | 22.5 dB | Mamba-based, linear complexity |
| SepMamba | [Standalone](https://github.com/andrasschin/SepMamba) | — | — | U-Net + bidirectional Mamba, ICASSP 2025 |
| SepFormer | SpeechBrain | ~26M | 20.2 dB | Transformer, heavier |
| TF-GridNet | ESPnet | — | 23.5 dB | SOTA, but complex |

Speech benchmarks (WSJ0-2mix) are not directly transferable to bioacoustics, but they indicate relative architecture quality.

### Warm-Starting from Biodenoising

The biodenoising pretrained weights already encode:
- What animal vocalizations sound like (120K training samples across birds, primates, marine mammals, insects)
- How to separate foreground from background noise
- Temporal structure of animal calls (via time-scaling augmentation across body sizes)

The 2025 paper "Is MixIT Really Unsuitable for Correlated Sources?" showed that MixIT pretraining → supervised fine-tuning gives consistent >0.5 dB gains. The reverse direction (pretrained denoiser → MixIT fine-tuning) should transfer similarly, since the encoder/decoder representations are reusable.

Reference: [arXiv:2505.07631](https://arxiv.org/abs/2505.07631)

### The 16 kHz Problem: Why Biodenoising Operates at Speech Sample Rate

Biodenoising's 16 kHz operating rate is **inherited from speech enhancement, not chosen for bioacoustics**. The lineage:

1. **Demucs dns48** was trained for the DNS (Deep Noise Suppression) Challenge — a Microsoft speech enhancement benchmark
2. Human speech has virtually all useful content below 8 kHz (Nyquist at 16 kHz), so 16 kHz is standard for speech
3. Biodenoising's key insight was that speech enhancement priors transfer to animal vocalizations
4. But that transfer comes with the speech model's assumptions baked in, including the 16 kHz rate and the learned filterbank tuned for that rate

The `--keep_original_sr` flag processes at the native sample rate, but the model is operating **outside its training distribution** — the learned filterbank coefficients were optimized for 16 kHz temporal patterns.

**Impact on BirdCLEF+ taxa at 16 kHz (Nyquist: 8 kHz):**

| Taxon | Typical frequency range | Lost at 16 kHz? |
|---|---|---|
| Most songbirds | F0: 1-6 kHz, harmonics to ~10 kHz | Harmonics lost, but ID features mostly preserved |
| High-frequency birds (warblers, kinglets) | Up to 10-12 kHz | Partial loss of discriminative features |
| Amphibians | Mostly 0.5-6 kHz | Fine |
| Insects (orthopterans) | 8-20+ kHz | **Completely lost** — crickets, katydids inaudible |
| Mammals | Variable, mostly <8 kHz | Mostly fine |

**For BirdCLEF+ 2026 (146 birds + 34 amphibians + 17 insects + 9 mammals)**: the 16 kHz limitation is a serious concern for the 17 insect species specifically. A model trained at 32 kHz (BirdCLEF's native rate) would cover the full taxonomic scope.

This is the central tradeoff in the architecture decision: **pretrained weights at the wrong sample rate (Path A) vs no pretrained weights at the right sample rate (Path B)**.

### Existing MixIT Results for Calibration

| Model | Domain | SI-SNR improvement | Training data |
|---|---|---|---|
| Generic MixIT (AudioSet) | General audio | 4.4 dB | AudioSet |
| Bird MixIT (Xeno-Canto) | Birds | **10.5 dB** | 704 XC species |
| Domain delta | | **+6.1 dB** | from domain matching |

Domain matching gave a 6.1 dB boost. A BirdCLEF-specific model trained on the actual competition data should match or exceed this.

## Architecture Decision: Demucs vs ConvTasNet vs SPMamba

### Path A: Demucs dns48 + MixIT (warm start, 16 kHz)

- **Pros**: Strongest pretrained initialization (biodenoising weights), proven on animal audio, PyTorch native, already a project dependency. Fastest convergence.
- **Cons**: **16 kHz operating rate** — learned filterbank is tuned for speech-rate temporal patterns. Loses all content above 8 kHz, which means insect species (17 of 206 in BirdCLEF+) are completely inaudible and high-frequency bird features are lost. Requires modifying output layer from 1→4 sources.
- **Engineering**: Modify final decoder to produce 4 output channels. Initialize 1 channel from pretrained weights, 3 randomly. Or find the original multi-source Demucs config and initialize matching layers.
- **Training estimate**: 20K-50K steps, **3-8 hours** on 1x RTX 6000 (warm start)
- **Best for**: Quick proof-of-concept on birds + amphibians. Not suitable as the final model if insect separation matters.

### Path B: ConvTasNet + MixIT (simplest, 32 kHz, no pretrained weights)

- **Pros**: Closest architecture to the original Bird MixIT TDCN++, fully supported in Asteroid with MixIT loss, minimal engineering. **Operates at 32 kHz** — covers all BirdCLEF+ taxa including insects. Sample rate matches BirdCLEF data natively, no resampling needed.
- **Cons**: No pretrained weights for animal audio — trains from scratch. Architecturally simpler than Demucs.
- **Engineering**: Near-zero — `ConvTasNet(n_src=4, sample_rate=32000)` + `MixITLossWrapper` works out of the box
- **Training estimate**: 50K-200K steps, **6-24 hours** on 1x RTX 6000
- **Best for**: Full-spectrum multi-taxa separation with minimal engineering risk.

### Path C: SPMamba + MixIT (most ambitious)

- **Pros**: Mamba-based with linear complexity, aligns with SSAMBA thesis, SOTA-tier separation quality, potentially CPU-viable at inference
- **Cons**: Not in Asteroid — needs adapter for MixIT loss, less tested, no pretrained weights
- **Engineering**: Integrate SPMamba model with Asteroid's `MixITLossWrapper` (thin wrapper), handle input/output tensor formats
- **Training estimate**: Unknown, likely **2-5 days** on 1x RTX 6000

**Recommendation**: Run Paths A and B in parallel as a first experiment — they're both cheap (single sbatch jobs). Path A tests whether biodenoising priors accelerate convergence despite the 16 kHz limitation. Path B tests whether training from scratch at 32 kHz produces better separation across all taxa. The comparison directly answers H5 and determines the best foundation for the augmentation pipeline. Path C (SPMamba) is a stretch goal if A/B succeed.

## Open Questions

- **Does Demucs dns48 retain multi-source capability after denoising fine-tuning?** The weights may have collapsed to a single-output manifold. If so, the warm start may not help much and ConvTasNet from scratch (Path B) would be comparable.
- **Can biodenoising weights transfer across sample rates?** The 16 kHz learned filterbank is tuned for speech-rate temporal patterns. Naively loading these weights into a 32 kHz model would misalign all filterbank coefficients. One option: interpolate filterbank weights to 32 kHz (doubling the kernel length). Whether this preserves useful structure is unknown.
- **How many output sources?** Start with M=4 (same as Bird MixIT primary model). M=8 gives finer decomposition but more unused channels and 2^8=256 assignment search.
- **MixIT assumes source independence across the two input mixtures.** For species that always co-occur in BirdCLEF data (e.g., dawn chorus regulars in the same region), this assumption is violated. How much does this matter in practice?
- **How to evaluate separation quality without clean references?** MoM reconstruction (SI-SNR on held-out MoMs) is the standard MixIT evaluation, but it doesn't directly measure how well individual sources are isolated. Qualitative spectrogram inspection and downstream augmentation impact are needed.
- **Could Demucs at 16 kHz still be useful for augmentation?** For the 146 bird + 34 amphibian species, most discriminative features are below 8 kHz. A 16 kHz separator could be "good enough" for foreground/background remixing even if it can't handle insects. The downstream classifier (trained at 32 kHz) would still see the full spectrum in the remixed audio — only the separation step operates at reduced bandwidth.

## Testable Hypotheses

1. **H1 (Warm start)**: Demucs initialized from biodenoising weights achieves >6 dB SI-SNR improvement on BirdCLEF MoMs within 20K training steps (3-8 hours). ConvTasNet from scratch requires >50K steps to reach the same quality. This tests whether the biodenoising prior provides meaningful acceleration.

2. **H2 (Domain matching)**: A separator trained on BirdCLEF data achieves higher SI-SNR improvement on BirdCLEF soundscapes than Google's generic Bird MixIT (10.5 dB baseline), consistent with the domain-matching effect observed in the original paper.

3. **H3 (Multi-taxa)**: A model trained on all BirdCLEF+ taxa (birds + amphibians + insects + mammals) separates non-bird vocalizations at >4 dB SI-SNR improvement. Google's Bird MixIT, trained only on birds, fails on non-bird taxa (<2 dB).

4. **H4 (Augmentation impact)**: Using the trained separator for foreground/background augmentation (per [source-separation-augmentation](source-separation-augmentation.md) Strategy A) improves EfficientNet (exp 012) cmAP on soundscape evaluation by >3% absolute over no-augmentation baseline.

5. **H5 (Architecture comparison)**: ConvTasNet + MixIT from scratch at 32 kHz outperforms Demucs + MixIT from biodenoising at 16 kHz on BirdCLEF data, because the sample rate advantage outweighs the pretrained weight advantage. (This determines whether Path A or B is better.)

## Connections

- **Based on**: [Bird MixIT](../vendor/bird-mixit.md), [Biodenoising](../vendor/biodenoising.md), [Source Separation Augmentation](source-separation-augmentation.md), [Deep research: source separation](../references/deep-research/2026-03-14-source-separation-augmentation.md)
- **Related to**: [Soundscape Domain Shift](soundscape-domain-shift.md) (the problem this helps solve), [SSAMBA Audio Mamba](ssamba-audio-mamba.md) (SPMamba aligns with Mamba thesis)
- **Could lead to**: Experiment proposal for training a BirdCLEF-specific MixIT separator, feeding into the augmentation pipeline described in source-separation-augmentation.md
- **Depends on**: Asteroid library (`MixITLossWrapper`), biodenoising pretrained weights, BirdCLEF 2026 data

## Notes

- **Asteroid is already a biodenoising dependency**, so adding it to the pipeline introduces no new top-level dependency.
- The original Demucs codebase (Meta/facebookresearch) has both the music separation config (4 sources) and the dns48 denoising config. Finding the right config to restore multi-source output should be straightforward by examining the Demucs source.
- **MoM formation is trivial**: randomly sample two recordings from the training set, sum their waveforms. No labels, no alignment, no preprocessing beyond resampling to a common sample rate.
- For the **Mamba end-to-end vision**: SPMamba separator → SSAMBA classifier. Both use Mamba blocks with linear complexity. The separator decomposes the acoustic scene, the classifier identifies species in each channel. All CPU-viable at inference due to O(L) complexity.
- **Evaluation without clean references**: Form MoMs from held-out focal recordings (where each recording is treated as a single-source "ground truth"). Measure whether the separator can reconstruct the two original recordings from their sum. This is the standard MixIT evaluation protocol (MoMi score).
- A 2025 paper showed that even **correlated sources** (which violate MixIT's independence assumption) can be handled with large enough models and sufficient data. The "correlated sources" concern may be less severe than initially thought.
- **Practical first milestone**: Get ConvTasNet + MixIT running on a small subset (1000 recordings) as a proof-of-concept. If it produces reasonable spectrograms (visible source isolation), scale up.
