# Proposal: Soundscape event boxing (weak->strong labels, insects first)

## Hypothesis

BirdCLEF 2026 soundscape labels give only **per-5s-window multi-label presence**, not
time-frequency boxes. For **spectrally stationary insects** (cicada/orthoptera
sonotypes `47158sonNN`), a sonotype's frequency signature can be recovered **without
manual boxing** by contrasting the mean spectrum of windows where it is present against
windows where it is absent (matched on the same files/sites). That fixed-band signature
should then **propagate** to other audio — auto-labeling instances in the 10,658-recording
unlabeled pool — with measurable precision/recall against held-out labeled windows.

## Background

- Labels: `train_soundscapes_labels.csv`, 1,478 windows x 66 files x 75 labels,
  `;`-separated multi-label, HH:MM:SS, no frequency info.
- 10,658 soundscapes exist (only 66 labeled), already Perch-embedded.
- Stationary insects make the easiest first target: a "box" is mostly a frequency band
  that is on continuously, so band discovery + a spectral-shape matcher suffices (no
  FM-sweep template machinery).

## Method

1. **Box proposal** (all taxa): per labeled window, PCEN spectrogram -> hysteresis +
   maad connected-component ROIs -> candidate time-freq boxes.
2. **Discriminative band** (per insect sonotype): present-vs-absent consensus-spectrum
   contrast -> `discover_bands` -> the sonotype's fixed band(s). Gate at n_present >= 8.
3. **Box->label attribution** (tentative): assign each candidate box to a window label by
   frequency-band overlap (insects win ties); flag ambiguous boxes for human priority.
4. **Curation**: review/correct boxes in Label Studio (`<Image>`+`<RectangleLabels>` over
   a spectrogram PNG, `<Audio>` to listen) -> gold boxes.
5. **Signature + propagation** (insects v1): build a compact per-sonotype signature (band
   + in-band spectral shape + optional pulse rate); scan held-out labeled windows then the
   unlabeled pool; auto-box matches.

## Evaluation

- Propagation precision/recall on **held-out labeled windows** using window-level presence
  as weak ground truth (fire X in a window truly containing X). Reported **in-site (S22) vs
  cross-site separately** — the honest number is cross-site.
- Box-level `interval_iou` (time & freq) where auto and gold boxes coexist (sanity).
- Yield on the unlabeled pool: windows/files/sites fired per sonotype; flag any sonotype
  firing on >~20% of all windows (background detector, not signal).

## Limitations

- Multi-label windows have **no per-box ground truth**; attribution is a prior, not truth
  (the reason the curation loop exists).
- Heavy S22 concentration risks signatures encoding site background rather than the
  sonotype; same-site absent-matching mitigates but cross-site recall may still collapse
  (itself a valid v1 finding).
- son25 (the dominant background cicada) is present in 84/168 insect windows, often
  co-occurring with other sonotypes -> its present/absent contrast is the weakest to isolate.
- v1 is DSP/signature-based; Perch embedding retrieval is the deferred v2 fallback.
