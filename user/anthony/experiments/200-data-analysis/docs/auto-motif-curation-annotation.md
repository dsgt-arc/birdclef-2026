# Auto-motif manual-curation anchor (human ground truth)

The only instrument that can say whether a mined motif is *real* on its own
terms. Every automated proxy was shown unreliable: the pipeline's label-free
`member_bg` d′ correlates only weakly with quality (and on the strand-5
attractor species it actively misleads — healthy d′ on a generic-energy
detector); independent Perch embeddings corroborate qualitatively but agree
with the per-species mode pick only ≈ chance; the labeled soundscape set is
underpowered for most species. So a small **exhaustive human annotation** is
the ground-truth anchor that calibrates d′→true-recovery and adjudicates the
suspected failures. Companion to `auto-motif-soundscape-efficacy.md` (its D1 /
strand-5 section is what motivates this).

## What is annotated and why

Five species, in two Label Studio projects split by role (the split reuses the
`require_recall` flag — the two roles need different effort):

| project | species | leg | task | what it answers |
|---|---|---|---|---|
| `broadband_anchor` | 74113, hyamac1, fusfly1 | production free+global | precision **and** recall | calibrate d′→recovery; adjudicate the 74113 false positive |
| `attractors` | 326272, 25092 | validated **consensus** | precision (recall secondary) | ground the strand-5 "d′ misleads" diagnosis on the real failures |

The attractors ran the *validated* consensus route with healthy d′ (≈2.2) yet
have `auc_vs_class ≤ 0.47` (their template prefers other species' clean focal
audio). `74113` is the prime suspected false positive (pipeline d′ 2.42 but
Perch-embedding AUC 0.43). Recordist metadata corroborates it: its top
recording `XC1039325.ogg` is `type=sniffing` (rating 5.0 — pristine *audio*,
zero vocal content), so its detections are firing on breathing/sniffing.

### Interpretation boundary (important)

Focal precision is **asymmetric** for the attractor question. On focal audio
the target dominates, so a generic-energy detector still lands on real calls →
*high* precision cannot refute strand-5; only *low* precision confirms it (e.g.
25092 tiles a continuous ~4.5 kHz insect band end-to-end, visibly not the
frog). The strand-5 failure is a *cross-species* discrimination failure that
the target's own focal audio cannot test by construction. So this anchor:
confirms generic-energy firing where it's visible on focal, calibrates the
broadband-anchor d′ scale, and adjudicates 74113 — but the decisive attractor
test remains the cross-species / soundscape one.

## Tooling: Label Studio, windowed clips

Code: `src/data_analysis/auto_motif/label_studio.py` (built on the recording
selection in `curation_scaffold.py`). Local setup, credentials, and the
serving gotchas live in agent memory (`label_studio_local_setup`).

- **Review is windowed.** Each selected recording is cut into ~5s clips (one
  task per clip) — Label Studio's `<Audio>` tag has no per-task viewport, so
  enforcing "5s at a time" requires physical clips. Clips carry ±0.5s context;
  a detection is *owned* by the single window its peak falls in, so it is
  judged exactly once (no double-count across adjacent clips). `broadband_anchor`
  keeps every window (empty ones drive recall); `attractors` keeps only
  detection-bearing windows.
- **Audio is normalized per recording, not per clip** — dividing each clip by
  its own peak boosted quiet call-less windows (a 3.2× boost on the 74113
  sniffing clip made breathing sound loud). Per-recording gain preserves true
  relative loudness.
- Each clip carries a per-clip PCEN spectrogram (shaded margins = the ±0.5s
  context) and a header showing `rating · type` (recordist context: `type`
  flags non-vocal recordings like `sniffing`; `rating` is audio quality, not
  vocal content).
- Detection regions are pre-loaded as Label Studio **predictions** with stable
  ids `det_<global_idx>`, so the scorer tells pipeline regions from
  annotator-drawn misses and aggregates each detection once.
- **Serving** of clips (under `/mnt/data/scratch/label-studio-clips/`) and
  spectrograms requires a `LocalFilesImportStorage` per project rooted at
  `/mnt/data/scratch` and `/mnt/data/research`; `push` registers them.
- Recording selection is **deterministic** (filename tiebreaker in
  `_stratified_recordings`) so re-exports reproduce the same clips and don't
  orphan annotations.

## Annotation protocol

Per detection region (cyan), set a verdict:

- **True call** — a real vocalization of *this* species.
- **False** — anything else: sniffing, breathing, noise, wind, rain, handling
  noise, silence, another species. ("Irrelevant" is False.)
- **Ambiguous** — only when genuinely undecidable (use sparingly).

Recall (`broadband_anchor`): drag a new region over any real call of the
species that has no box (verdict `True call`); approximate bounds are fine.
Empty windows with no real call → submit with nothing added (a valid "no
miss"). A whole sniffing recording → every box `False`, no recall regions; that
is the correct, expected result, not a failure.

## Scoring

```bash
python -m data_analysis.auto_motif.label_studio parse \
    --project <broadband_anchor|attractors> --export <exported.json>
```

Writes `results_summary.{csv,md}` under the project dir: per-species precision
(tp/fp/ambiguous) and recall (vs annotator-drawn misses), joined against the
production manifest d′ / curated-recovery / Perch-embedding AUC — i.e. the test
of whether pipeline d′ tracks true recovery.

## Reproduce

```bash
source /mnt/data/scratch/venvs/exp200-data-analysis/.venv/bin/activate
cd user/anthony/experiments/200-data-analysis
PYTHONPATH=src python -m data_analysis.auto_motif.label_studio export        # clips + tasks
PYTHONPATH=src python -m data_analysis.auto_motif.label_studio push --replace # provision + import
# annotate in the UI, export JSON, then `... parse` as above
```

## Next

If the anchor confirms the strand-5 diagnosis (attractor templates fire on
generic energy / 74113 is a false positive) and calibrates the d′ scale, the
constructive lever is **v2 contrastive matched-filter exemplar mining**
(`docs/concepts/contrastive-matched-filter-exemplar-mining.md`) — add a
discriminative term at template selection so self-recurrence alone can no
longer select generic broadband energy.
