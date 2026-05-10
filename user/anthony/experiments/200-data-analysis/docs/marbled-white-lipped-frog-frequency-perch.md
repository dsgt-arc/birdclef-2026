# Notebook: Marbled White-lipped Frog Frequency Band + Perch Logits

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| BirdCLEF label | `22967` |
| Scientific name | *Leptodactylus elenae* |
| Common name | Marbled White-lipped Frog |
| Class | Amphibia |
| Train-audio files | 8 |
| Train-audio 5s windows | 33 |
| Analysis code | `src/data_analysis/species_frequency_perch_analysis.py` |
| Audio features | librosa mel spectrogram, 32 kHz, 128 mel bands, 50-8000 Hz |
| Perch logits | `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_audio/logits.parquet` |
| Output directory | `artifacts/species_frequency_perch_22967/` |

## At-a-glance takeaway

For Marbled White-lipped Frog, the data supports using a **rough 1-3 kHz working band**, with strongest enrichment around **1.5-3 kHz** in windows where Perch has high genus-level frog evidence.

Perch's direct *Leptodactylus elenae* logit is weak, but Perch strongly activates for nearby **Leptodactylus** taxa, especially *L. fragilis* and *L. fuscus*. So for this species, the useful Perch signal is **genus-level**, not exact species-level.

## Generated artifacts

| Artifact | Description |
|---|---|
| `top_mean_perch_logits.png` | top Perch taxa by mean logit over the 33 train-audio windows |
| `window_perch_scores.png` | direct *L. elenae* logit vs max *Leptodactylus* logit per window |
| `spectrum_high_vs_low_genus_logit.png` | mean mel spectra for high vs low max-*Leptodactylus* windows |
| `window_scores_and_spectral_features.csv` | per-window Perch scores and mel-band energy features |
| `band_enrichment.csv` | frequency-band enrichment table |
| `top_mean_perch_logits.csv` | top mean Perch taxa table |

## Perch logit finding

Top mean Perch logits across the 33 windows:

```text
rank|perch_index|perch_label|mean_logit|mean_prob
1|7004|Leptodactylus fragilis|7.187|0.9992
2|7006|Leptodactylus fuscus|6.783|0.9989
3|4286|Donacobius atricapilla|6.033|0.9976
4|6813|Laniarius ferrugineus|5.510|0.9960
5|5537|Glaucidium brasilianum|5.260|0.9948
```

The exact target, *Leptodactylus elenae*, is not among the top mean logits. This means we should not rank relevance by direct species logit alone. The max logit over Perch labels beginning with `Leptodactylus` is a better relevance proxy.

## Frequency-band finding

Band enrichment compares mean normalized mel energy in the top quartile vs bottom quartile of max-*Leptodactylus* logit windows.

Most enriched bands:

```text
band_hz|high_minus_low|high_div_low
2000-3000|+0.204|3.69x
1500-2000|+0.147|3.19x
1000-1500|+0.090|1.38x
```

Bands that were higher in low-genus-logit windows:

```text
4000-6000|-0.219
750-1000|-0.091
0-250|-0.061
```

Interpretation:

- Use **1-3 kHz** as a broad inspection/filtering band.
- The strongest data-supported portion is **1.5-3 kHz**.
- High energy above 4 kHz appears less associated with Perch's frog-genus evidence in this small set.

## Highest genus-logit windows

Top windows by max *Leptodactylus* logit are mostly from `iNat1698107` and `iNat1698105`:

```text
filename|time|max_genus_logit|max_genus_label|mel_peak_hz|mel_centroid_hz
22967/iNat1698107.ogg|30-35s|9.76|Leptodactylus fuscus|1518|1407
22967/iNat1698107.ogg|15-20s|9.73|Leptodactylus fuscus|1555|1411
22967/iNat1698105.ogg|0-5s|9.65|Leptodactylus fuscus|50|1083
22967/iNat1698107.ogg|25-30s|9.62|Leptodactylus fuscus|1518|1396
22967/iNat1698105.ogg|5-10s|9.38|Leptodactylus fuscus|50|1821
22967/iNat1698105.ogg|15-20s|9.32|Leptodactylus fuscus|2840|2235
22967/iNat1697350.ogg|0-5s|9.31|Leptodactylus fuscus|1979|1768
```

A few high-genus windows still show `mel_peak_hz=50`, probably because low-frequency background/noise dominates the single strongest mel bin. The centroid and band enrichment are more useful than raw peak frequency.

## Caveats

- `train_audio` labels are file-level, not window-level.
- Perch logits are not ground truth; they provide a useful prior.
- The 1-3 kHz band is a working prior for inspection/pseudo-labeling, not a confirmed box.
- Manual spectrogram review is still needed before using these as hard labels.
