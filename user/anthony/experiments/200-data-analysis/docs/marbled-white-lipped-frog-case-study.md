# Case Study: Marbled White-lipped Frog

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| BirdCLEF label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Class | Amphibia |
| Train-audio files | 8 |
| Train-audio 5s windows | 33 |
| Main artifacts | `artifacts/species_frequency_perch_22967/`, `artifacts/marbled_frog_window_relevance/` |
| Main code | `src/data_analysis/species_frequency_perch_analysis.py`, `src/data_analysis/marbled_frog_window_relevance.py` |

## One-line takeaway

For Marbled White-lipped Frog, Perch is useful mostly as a **genus-level detector**, and the best current weak relevance signal combines **max *Leptodactylus* logit**, **1-3 kHz mel-band energy**, and **embedding/PaCMAP cluster structure**.

## Why this species

Marbled White-lipped Frog is useful as a small case study because it has:

- enough labeled soundscape support to matter: 310 labeled soundscape windows across 14 soundscape files,
- limited but nonzero train-audio support: 8 files / 33 Perch windows,
- multi-site labeled-soundscape presence,
- obvious examples where weak file-level train labels are not enough to know which 5s windows are actually relevant.

## Step 1: Start from train-audio windows

The regular training metadata says these 8 files have primary label `22967`. Perch embeddings/logits split them into 33 five-second windows.

But `train.csv` only gives file-level labels. It does not say which 5s windows contain the frog call. So every per-window relevance label here is a weak/pseudo label unless manually verified.

## Step 2: Check Perch logits

The direct Perch species logit for *Leptodactylus elenae* is weak. Across the 33 windows, the top mean Perch taxa are instead nearby congeners:

```text
rank|perch_label|mean_logit
1|Leptodactylus fragilis|7.19
2|Leptodactylus fuscus|6.78
3|Donacobius atricapilla|6.03
```

Interpretation:

- Do **not** use the direct *L. elenae* Perch logit as the main relevance signal.
- Use the maximum logit over Perch labels beginning with `Leptodactylus` as a genus-level frog evidence score.

This makes biological/modeling sense: Perch hears something strongly *Leptodactylus*-like, but does not resolve the exact BirdCLEF species reliably.

## Step 3: Estimate a useful frequency band

Using librosa mel spectra, I compared windows with high vs low max-*Leptodactylus* logits.

The enriched bands in high-genus-logit windows are:

```text
band_hz|high_minus_low|high_div_low
2000-3000|+0.204|3.69x
1500-2000|+0.147|3.19x
1000-1500|+0.090|1.38x
```

Working frequency prior:

```text
broad:   1-3 kHz
strong:  1.5-3 kHz
```

This is not a bounding box. It is a prior for inspection/filtering.

## Step 4: Initial weak relevance rule

The first window-level pseudo-label rule was:

```text
likely_relevant =
    max Leptodactylus Perch logit >= 8.5
    AND
    1-3 kHz mel energy fraction >= 0.5
```

This produced:

```text
likely_relevant:     16 windows
likely_not_relevant: 17 windows
```

The relevant spectrogram samples showed repeated rising frog-like calls concentrated around the 1-3 kHz band, especially near 1-2 kHz with upper structure.

## Step 5: PaCMAP revealed an obvious false-positive cluster

Plotting the 33 Perch window embeddings with PaCMAP showed a clear cluster around approximately:

```text
PaCMAP center ≈ (-4, 0)
manual radius = 3
```

This cluster contained several scalar-rule `likely_relevant` windows that visually look suspicious or contaminated. In other words, the scalar rule caught some false positives because high genus logit + 1-3 kHz energy alone is not enough.

The cleanest high-confidence relevant cluster is instead the isolated `iNat1698107` group. All seven 5s windows from that file cluster tightly and have strong repeated call structure:

```text
22967/iNat1698107.ogg 0-35s
```

## Step 6: Use the visible cluster as a seed, then compute a robust boundary

A fixed PaCMAP coordinate is not stable across recomputations, so the circle is used only as a seed-selection tool.

For the current plot, the circle:

```text
center = (-4, 0)
radius = 3
```

selects 21 windows. Its current 2D PaCMAP centroid is:

```text
centroid ≈ (-3.83, -0.01)
```

Using the seed points, I computed a robust radius from distances:

```text
radius = median(seed distances) + 2 * 1.4826 * MAD(seed distances)
```

For the current 2D PaCMAP view:

```text
median distance = 1.75
scaled MAD      = 0.81
z=2 radius      = 3.38
members         = 21 windows
```

I also tried doing this in PCA-compressed embedding space. With only 33 windows and a broad seed cluster, the embedding-space MAD boundary was too loose: it included 32/33 windows. So for this tiny case, the 2D boundary is useful for visual triage, but a better future version should seed from a tighter high-confidence cluster.

## Step 7: Estimate scalar-rule false positives/false negatives

Treating the `(-4,0)` PaCMAP/MAD cluster as the visually suspicious/not-relevant cluster, the original scalar rule has:

```text
TP = 10  predicted relevant, outside suspicious cluster
FP =  6  predicted relevant, inside suspicious cluster
FN =  2  predicted not relevant, outside suspicious cluster
TN = 15  predicted not relevant, inside suspicious cluster
```

Derived ratios:

```text
false positive rate      = 6 / 21 = 28.6%
false discovery rate     = 6 / 16 = 37.5%
false negative rate      = 2 / 12 = 16.7%
precision                = 10 / 16 = 62.5%
recall                   = 10 / 12 = 83.3%
```

This is not ground truth; it is relative to a visual cluster proxy. But it clearly says the scalar rule is too permissive.

False positives under this proxy:

```text
22967/iNat56387.ogg     0-5s
22967/iNat56387.ogg     5-10s
22967/iNat911059.ogg    15-20s
22967/iNat1448422.ogg   0-5s
22967/iNat1697350.ogg   0-5s
22967/iNat1697350.ogg   5-10s
```

False negatives under this proxy:

```text
22967/iNat1698105.ogg   0-5s
22967/iNat1698105.ogg   20-25s
```

## Current best interpretation

There are three levels of evidence:

### High confidence relevant

The clean, tight `iNat1698107` cluster. These windows have strong repeated rising calls in the 1-3 kHz band and high max-*Leptodactylus* logits.

### Candidate relevant / needs review

Windows outside the suspicious cluster that pass either logit or band-energy checks but do not belong to the clean `iNat1698107` prototype.

### Likely false positive / not relevant

Windows in the `(-4,0)` cluster, especially those marked relevant by the scalar rule. These are where logit/band evidence alone likely overcalls relevance.

## Practical next rule

A better pseudo-labeling rule should be tiered:

```text
high_confidence_relevant =
    close to high-confidence frog-call prototype/cluster
    AND max Leptodactylus logit high
    AND 1-3 kHz mel energy high

candidate_relevant =
    high genus logit or good 1-3 kHz call-like structure
    BUT not in clean prototype cluster

likely_not_relevant =
    in suspicious cluster or low genus/band evidence
```

For this species, I would not train directly on the initial binary labels. I would first manually inspect:

1. all 7 `iNat1698107` windows as positive anchors,
2. the 6 scalar-rule false positives,
3. the 2 scalar-rule false negatives,
4. a few low-score negatives.

That small review would calibrate a much better window-level training set.
