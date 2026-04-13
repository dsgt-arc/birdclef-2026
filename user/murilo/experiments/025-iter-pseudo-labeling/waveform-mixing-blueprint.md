# Mario-Style Waveform-Mixing Pseudo-Label Training

Blueprint for Phase C of exp 025. Based on Mario Lasseck's BirdCLEF 2024
2nd place paper and Salman Ahmed's BirdCLEF 2025 implementation.

**Reference**: `notes/2026-04-12-mario-lasseck-birdclef2024.md`

## Why this is different from Phase B (which failed)

Phase B added pseudo-labeled samples as **separate rows** in the dataset.
The model saw either a focal sample OR a PL sample each step, with per-sample
loss weighting (w=1.0 for GT, w=0.5 for PL). Result: 0.897 LB (-0.006).

Mario's approach **mixes PL audio INTO focal samples** at the waveform level.
Every training step, the model sees a focal recording overlaid with soundscape
background. Three simultaneous benefits:

1. **Noise augmentation**: focal recordings acquire Pantanal background noise
2. **Data extension**: model sees soundscape species it never saw in focal data
3. **Knowledge distillation**: ensemble predictions transfer to student model

## The Recipe

### Step 1: Pick a focal training sample

Standard v5j behavior: random 5s crop from a focal recording.
```
focal_waveform: (160000,)   # 5s at 32kHz
focal_label: (234,)          # one-hot primary + secondary
```

### Step 2: With probability p, mix in a pseudo-labeled soundscape segment

```python
if random.random() < pseudo_label_chance:  # p = 0.25 to 0.45
    # Pick a random pseudo-labeled 5s segment
    pl_waveform, pl_soft_labels = random_pseudo_label_segment()

    # Amplitude control — random factor for each waveform
    amp_focal = 10 ** random.uniform(-0.5, 0.1)   # 0.32x to 1.26x
    amp_pl    = 10 ** random.uniform(-0.5, 0.1)

    # Mix at waveform level (additive, physically correct)
    mixed_waveform = amp_focal * focal_waveform + amp_pl * pl_waveform

    # Labels: element-wise maximum (union of species present)
    mixed_label = torch.maximum(focal_label, pl_soft_labels)
else:
    mixed_waveform = focal_waveform
    mixed_label = focal_label
```

### Step 3: Continue standard pipeline

```
mixed_waveform → mel spectrogram → MixUp → backbone → head → BCE loss
```

Everything after the waveform mixing is identical to v5j.

### Key parameters

| Parameter | Mario (2024) | Salman (2025) | Our starting point |
|-----------|-------------|---------------|-------------------|
| PL mixing probability | 25-45% | similar | **30%** |
| ampExpMin | -0.5 to -1.0 | — | **-0.5** |
| ampExpMax | 0.1 to 0.2 | — | **0.1** |
| Label combination | max(GT, PL) | max(GT, PL) | **max(GT, PL)** |
| PL source | ensemble on unlabeled | ensemble on unlabeled | **existing 347K PLs** |
| Loss | avg(BCE + Focal) | FocalBCE | **BCE** (keep v5j) |
| Epochs | 50 + soup | 15 + soup | **20** (keep v5j) |
| Iterations | 3 rounds | 2 stages | **Start with 1** |

### How this differs from additive MixUp (v10b, which failed)

v10b mixed TWO FOCAL recordings additively (x1 + x2). This doubles the
amplitude and creates unnatural energy distributions — the model saw
amplitude-doubled spectrograms at training but normal spectrograms at inference.

Mario's approach mixes a focal recording with a SOUNDSCAPE segment. The
soundscape acts as background noise — the amplitude factors (0.32x-1.26x)
keep the energy in a natural range. The model learns "what does species X
sound like in the Pantanal?" rather than "what do two overlapping species
sound like at 2x volume?"

## Implementation Plan

### What needs to change

**1. New dataset class: `PseudoLabelMixingDataset`**

Wraps the existing `SEDFocalDataset` (or GT+soundscape ConcatDataset).
On each `__getitem__`:
- Get the normal GT sample (waveform, label, primary_idx)
- With probability `pl_mix_chance`, load a random PL segment and mix
- Return the mixed (or unmixed) sample

This is a wrapper, not a replacement — the underlying dataset stays the same.

**2. PL segment pool**

Pre-load all pseudo-labeled soundscape segments as a pool:
- Read the parquet with soft labels (already exists from Phase A)
- Store file paths + segment start times + soft label vectors
- On mixing: load the 5s audio chunk, apply amplitude factor, add to focal

**3. Config additions** (in `v3_waveform_mix_train.yaml`)

```yaml
pseudo_labels:
  path: "/storage/.../v1/pseudo_labels.parquet"
  audio_dir: "/storage/.../birdclef-2026/train_soundscapes"
  mix_chance: 0.30           # probability of mixing PL into a focal sample
  amp_exp_min: -0.5          # amplitude factor range (log10 scale)
  amp_exp_max: 0.1
  label_combine: "max"       # max(GT, PL) for label fusion
```

**4. No changes to**: loss function, head, backbone, mel transform, scheduler,
or any other v5j parameter.

### What does NOT change

- The v5j training recipe (BCE, LSEHead, spectrogram MixUp, OneCycleLR)
- The mel spectrogram pipeline
- The inference/export pipeline
- The validation setup

### Where to put the code

The `PseudoLabelMixingDataset` wrapper goes in `src/birdclef/data/sed.py`
alongside the existing PL support classes. The training script (`train.py`)
constructs it when the config specifies PL mixing.

## Iteration Strategy

### Round 1: Use existing v1 pseudo-labels (347K segments)

- Teacher: 0.929 ensemble (Perch + SED 4-fold)
- PLs: already generated in Phase A, stored at `shared/pseudo-labels/birdclef-2026/v1/`
- Train v5j with waveform mixing at 30% probability
- Submit to LB → compare vs 0.906 baseline

### Round 2 (if Round 1 improves): Regenerate PLs with the improved student

- The round 1 student becomes the new teacher
- Generate v2 pseudo-labels with the improved model
- Train a new student with v2 PLs
- Expected: each iteration improves PLs → improves student → better PLs

### Round 3+ (if iterating works): Ensemble teacher for each round

- Use the round N student ensemble to generate round N+1 PLs
- Mario showed 3 iterations before diminishing returns

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Round 1 single model (waveform mix) | > 0.906 LB (improve on v5j) |
| Round 1 proves approach works | > 0.900 LB (not a regression) |
| After 2-3 iterations | > 0.920 LB (match Salman) |
| Stretch: in ensemble | > 0.935 LB |

## Risk Assessment

**Why this might work** (unlike all previous training changes):
- Every previous change modified HOW the model learns from the SAME data
- This changes WHAT data the model sees — 127K soundscape segments it never had
- Mario showed +8% from this exact technique (the largest gain of any method)
- Our 0.929 ensemble is a much stronger teacher than Mario's starting ensemble

**Why this might fail**:
- Our v1 pseudo-labels regressed in Phase B (but with the wrong method)
- PL quality may be too low for the 159 unseen species
- The v5j recipe may not absorb mixed audio well (though it absorbs spectrogram
  MixUp fine — the difference is the mixing happens at the waveform level)
