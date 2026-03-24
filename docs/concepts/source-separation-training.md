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

**Recommendation (revised)**: Before training any custom separator, validate whether source separation improves pseudo-labels at all using biodenoising (zero training cost). See "Validation Ladder" below.

## Validation Ladder: Biodenoising-First Strategy

**Key insight**: If the goal is foreground/background decomposition for augmentation, biodenoising already does this with zero training. MixIT's unique contribution is multi-channel max-pooling — but this must be validated before investing 6-24h of training.

### Why not jump to MixIT?

MixIT trains on mixture-of-mixtures (MoM): sum two recordings, separate back into originals. This works well when each recording has 1-3 species (focal recordings). But BirdCLEF 2026 soundscapes have a **median of 8 species and up to 20** per 5s window. Summing two such soundscapes creates 16-40 overlapping sources — far beyond a 4-channel model's capacity. The model degenerates to copying input to output.

**Practical implication**: MoMs should be formed from **focal recordings only** (344.5h, typically 1-3 species). The trained model is then *applied* to soundscapes at inference, where it won't cleanly isolate 8 species into 4 channels but can still usefully group foreground vs background. This is the same thing biodenoising already does — raising the question of whether MixIT adds value over biodenoising for this use case.

### The falsifiable ladder

**Step 1 (free)**: Run biodenoising on soundscapes → 2 channels (foreground + background). Classify each channel + original with BirdNET/Perch, max-pool per species. Compare pseudo-label quality against raw-mixture predictions (already available from exp 010).

- **Pass gate**: Biodenoising max-pool improves pseudo-label quality (higher teacher agreement, or downstream classifier cmAP gain >1%).
- **If fail**: Source separation doesn't help for pseudo-labeling on dense soundscapes. Skip MixIT entirely. Focus augmentation efforts on simpler approaches (Strategy A: foreground remix at varied SNR).

**Step 2 (6-24h)**: Only if Step 1 passes. Train ConvTasNet + MixIT at 32 kHz on focal MoMs. Apply to soundscapes → 4 channels + original. Compare 5-channel max-pool pseudo-labels against biodenoising's 3-channel max-pool.

- **Pass gate**: 4-channel separation improves pseudo-label quality beyond biodenoising's 2-channel by >2% relative.
- **If fail**: Biodenoising is sufficient. Use it for augmentation and pseudo-labeling; don't maintain a custom separator.

**Step 3 (augmentation)**: Whichever separator passes, test foreground/background remix augmentation on the downstream classifier (exp 012 EfficientNet or equivalent). This tests H4 from the augmentation concept doc.

### MixIT's unique value proposition (if any)

The only scenario where MixIT clearly beats biodenoising:
- A soundscape has species A (loud) and species B (quiet) co-occurring
- Biodenoising puts both in "foreground" (it doesn't separate species)
- MixIT might separate them into different channels, letting species B get a clean classification
- Max-pool across channels catches species B

But with 8+ species per soundscape, even MixIT's 4 channels will contain species mixtures. The marginal gain of 4 channels over 2 may be small. Step 2 tests exactly this.

## Open Questions

- **Does Demucs dns48 retain multi-source capability after denoising fine-tuning?** The weights may have collapsed to a single-output manifold. If so, the warm start may not help much and ConvTasNet from scratch (Path B) would be comparable.
- **Can biodenoising weights transfer across sample rates?** The 16 kHz learned filterbank is tuned for speech-rate temporal patterns. Naively loading these weights into a 32 kHz model would misalign all filterbank coefficients. One option: interpolate filterbank weights to 32 kHz (doubling the kernel length). Whether this preserves useful structure is unknown.
- **How many output sources?** Start with M=4 (same as Bird MixIT primary model). M=8 gives finer decomposition but more unused channels and 2^8=256 assignment search.
- **MixIT assumes source independence across the two input mixtures.** For species that always co-occur in BirdCLEF data (e.g., dawn chorus regulars in the same region), this assumption is violated. How much does this matter in practice?
- **How to evaluate separation quality without clean references?** MoM reconstruction (SI-SNR on held-out MoMs) is the standard MixIT evaluation, but it doesn't directly measure how well individual sources are isolated. Qualitative spectrogram inspection and downstream augmentation impact are needed.
- **Could Demucs at 16 kHz still be useful for augmentation?** For the 146 bird + 34 amphibian species, most discriminative features are below 8 kHz. A 16 kHz separator could be "good enough" for foreground/background remixing even if it can't handle insects. The downstream classifier (trained at 32 kHz) would still see the full spectrum in the remixed audio — only the separation step operates at reduced bandwidth.

## Testable Hypotheses

1. **H0 (Biodenoising baseline — free, test first)**: Biodenoising foreground/background separation + max-channel classification improves pseudo-label quality on BirdCLEF soundscapes compared to raw-mixture classification. This gates all further separation work. If biodenoising doesn't help, MixIT won't either.

2. **H1 (Multi-channel value)**: MixIT 4-channel separation provides >2% relative improvement in pseudo-label quality over biodenoising 2-channel on dense soundscapes (median 8 species). This determines whether custom separator training is justified.

3. **H2 (Domain matching)**: A separator trained on BirdCLEF data achieves higher SI-SNR improvement on BirdCLEF soundscapes than Google's generic Bird MixIT (10.5 dB baseline), consistent with the domain-matching effect observed in the original paper.

4. **H3 (Multi-taxa)**: A model trained on all BirdCLEF+ taxa (birds + amphibians + insects + mammals) separates non-bird vocalizations at >4 dB SI-SNR improvement. Google's Bird MixIT, trained only on birds, fails on non-bird taxa (<2 dB).

5. **H4 (Augmentation impact)**: Using the best available separator for foreground/background augmentation (per [source-separation-augmentation](source-separation-augmentation.md) Strategy A) improves EfficientNet (exp 012) cmAP on soundscape evaluation by >3% absolute over no-augmentation baseline.

6. **H5 (Architecture comparison)**: ConvTasNet + MixIT from scratch at 32 kHz outperforms Demucs + MixIT from biodenoising at 16 kHz on BirdCLEF data, because the sample rate advantage outweighs the pretrained weight advantage. Only test if H1 passes.

## Connections

- **Based on**: [Bird MixIT](../vendor/bird-mixit.md), [Biodenoising](../vendor/biodenoising.md), [Source Separation Augmentation](source-separation-augmentation.md), [Deep research: source separation](../references/deep-research/2026-03-14-source-separation-augmentation.md)
- **Related to**: [Soundscape Domain Shift](soundscape-domain-shift.md) (the problem this helps solve), [SSAMBA Audio Mamba](ssamba-audio-mamba.md) (SPMamba aligns with Mamba thesis)
- **Could lead to**: Experiment proposal for training a BirdCLEF-specific MixIT separator, feeding into the augmentation pipeline described in source-separation-augmentation.md
- **Depends on**: Asteroid library (`MixITLossWrapper`), biodenoising pretrained weights, BirdCLEF 2026 data

## Per-Taxa Separation and High-Frequency Support

### The insect problem at 16 kHz

The current pretrained models (biodenoising, Bird MixIT) operate at 16-22 kHz,
which loses all content above 8-11 kHz. BirdCLEF 2026 has 28 insect species
(25 sonotypes + 3 focal) with vocalizations at 8-20+ kHz. These are completely
invisible to low-sample-rate separators.

### Training at 32 kHz for full-spectrum separation

A separator trained at 32 kHz (BirdCLEF's native rate) would cover all taxa:

| Taxa | Frequency range | Covered at 16 kHz | Covered at 32 kHz |
|------|----------------|-------------------|-------------------|
| Birds | 1-12 kHz | Mostly (lose harmonics) | Fully |
| Amphibians | 0.5-6 kHz | Fully | Fully |
| Insects | 8-20+ kHz | **Not at all** | Fully |
| Mammals | 0.1-8 kHz | Fully | Fully |

ConvTasNet (Path B) at 32 kHz is the simplest path to full-spectrum separation.

### Per-taxa separation channels

Instead of generic 4-source separation, train with taxa-aware channel
assignment:

| Channel | Target content | Why |
|---------|---------------|-----|
| 1 | Birds | Tonal, harmonic, 1-12 kHz |
| 2 | Amphibians | Repetitive, narrow-band, 0.5-6 kHz |
| 3 | Insects | Broadband, high-frequency, 8-20+ kHz |
| 4 | Background/noise | Abiotic sounds, anthropophony |

This is NOT supervised separation (we don't have per-taxa ground truth in
soundscapes). But we can bias the MixIT training by forming MoMs from
same-taxa focal recordings:

```
MoM_birds = bird_recording_1 + bird_recording_2
MoM_frogs = frog_recording_1 + frog_recording_2
MoM_cross = bird_recording + frog_recording  # cross-taxa
```

Training on a mix of same-taxa and cross-taxa MoMs lets the model learn
both within-taxa separation and cross-taxa separation.

### Connection to codec motif search

Per-taxa separation + codec tokenization creates a clean pipeline:

```
soundscape → separator → 4 channels (bird/frog/insect/noise)
    → codec tokenizer → token sequences per channel
    → motif search per taxa (search bird tokens for bird calls,
      insect tokens for insect calls)
```

The separator reduces the search space: instead of searching for a bird
call in a dense multi-taxa soundscape, search in the bird-isolated channel.
The codec tokenization makes the search fast (DNA-style k-mer matching).

See `docs/concepts/codec-motif-search.md` for the search component.

## Prompted Separation: Reference-Conditioned Source Extraction

### Idea

Instead of blind separation into N fixed channels, condition the separator
on a reference call: "here's what species X sounds like — extract it from
this soundscape." This is **target sound extraction**, adapted from the
speech domain's "speaker extraction" literature.

```
Input:  [reference call of species X] + [60s soundscape]
Output: [just the species X parts of the soundscape]
```

Run it N times with N different reference calls to extract N species.
No fixed channel count, no channel assignment ambiguity.

### Why this is better than blind MixIT

| Blind MixIT (4 channels) | Prompted extraction |
|--------------------------|---------------------|
| Fixed 4 channels for 8+ species | One extraction per query species |
| No control over what goes where | Reference defines what to extract |
| Channels contain species mixtures | Each output is one species (ideally) |
| Can't handle new species at inference | Zero-shot: any species with a focal recording |

### Training

Use focal recordings as both references and targets:

1. Form MoM: `focal_A + focal_B`
2. Reference: a different recording of species A (or a chunk of focal_A)
3. Target: reconstruct `focal_A` from the MoM
4. The model learns: "given this reference, pull out matching sounds"

The reference and target should be different recordings of the same species
(not the same file) to force the model to learn species-level features
rather than memorizing specific recordings.

### Soundscape MoMs

Training can also use soundscape-only MoMs:

- `soundscape_A + soundscape_B` as input
- No reference needed for MixIT training (unsupervised)
- At inference, add the reference conditioning

Or a **hybrid**: MixIT pretraining on soundscape MoMs (learn to separate
in-domain audio), then supervised fine-tuning with focal references
(learn to condition on reference calls).

### Architecture

The conditioning can be injected via:

| Method | How | Precedent |
|--------|-----|-----------|
| FiLM | Reference embedding scales/shifts intermediate features | SpeakerBeam |
| Cross-attention | Separator attends to reference features | VoiceFilter-Lite |
| Concatenation | Reference features prepended to encoder input | Simple, works for Mamba |
| Codec tokens | Reference as token prefix, separator as sequence model | Novel — connects to codec motif search |

For Mamba: concatenation is natural. Prepend reference tokens to the
soundscape tokens, let the recurrence build up "what to listen for"
in the hidden state before processing the soundscape. This is the same
prompted processing pattern from exp 011 (prompted SED) but applied to
separation.

### Connection to codec motif search

The prompted separator and codec search solve the same problem differently:

| | Prompted separator | Codec motif search |
|---|---|---|
| Input | Reference waveform + soundscape | Reference tokens + soundscape tokens |
| Method | Neural network extraction | Sequence alignment |
| Output | Separated waveform | Match positions + scores |
| Training | Requires training | No training (search algorithm) |
| Temporal output | Full extracted waveform | Time positions of matches |

They could work together:
1. **Codec search** (fast, no training) identifies which species are present
   and approximately when
2. **Prompted separator** (slower, trained) extracts clean audio for those
   species at the identified times
3. **Classifier** confirms species identity on the cleaned audio

### Existing work

- **SpeakerBeam** (Delcroix et al., 2018): target speaker extraction
  conditioned on enrollment utterance
- **VoiceFilter** (Wang et al., 2019): Google's speaker-conditioned separation
- **Universal Sound Separation** (Kavalerov et al., 2019): query-based
  separation for arbitrary sounds
- **No published work on reference-conditioned bioacoustic separation**

### Testable hypothesis

**H6 (Prompted separation)**: A reference-conditioned separator, given a
focal recording of species X and a soundscape containing species X, can
extract species X's vocalization at >6 dB SI-SNR improvement. This would
enable per-species clean audio extraction from dense soundscapes, directly
feeding into both classification and codec-based indexing.

## Notes

- **Asteroid is already a biodenoising dependency**, so adding it to the pipeline introduces no new top-level dependency.
- The original Demucs codebase (Meta/facebookresearch) has both the music separation config (4 sources) and the dns48 denoising config. Finding the right config to restore multi-source output should be straightforward by examining the Demucs source.
- **MoM formation is trivial**: randomly sample two recordings from the training set, sum their waveforms. No labels, no alignment, no preprocessing beyond resampling to a common sample rate.
- For the **Mamba end-to-end vision**: SPMamba separator → SSAMBA classifier. Both use Mamba blocks with linear complexity. The separator decomposes the acoustic scene, the classifier identifies species in each channel. All CPU-viable at inference due to O(L) complexity.
- **Evaluation without clean references**: Form MoMs from held-out focal recordings (where each recording is treated as a single-source "ground truth"). Measure whether the separator can reconstruct the two original recordings from their sum. This is the standard MixIT evaluation protocol (MoMi score).
- A 2025 paper showed that even **correlated sources** (which violate MixIT's independence assumption) can be handled with large enough models and sufficient data. The "correlated sources" concern may be less severe than initially thought.
- **Practical first milestone (revised)**: Run biodenoising on a sample of soundscapes, classify separated channels with BirdNET/Perch, compare pseudo-label quality to raw predictions. This is free (no training) and directly gates whether any custom separator training is warranted. Only proceed to ConvTasNet + MixIT if biodenoising max-pool shows measurable improvement.
